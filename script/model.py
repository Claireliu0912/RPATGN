import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, kaiming_uniform, zeros
from script.RPATGN.layers.layer import RoleGCNConv,GraphGRU
from torch.autograd import Variable

class RPATGN(nn.Module):
    def __init__(self, args):
        super(RPATGN, self).__init__()
        self.num_nodes = args.num_nodes
        self.device = args.device
        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)), requires_grad=True)
        self.linear = nn.Linear(args.nfeat, args.nhid)

        self.linear_z = nn.Linear(args.nout, args.nhid)
        self.hidden_initial = torch.ones(args.num_nodes, args.nout).to(args.device)
        self.dropout1 = 0
        self.dropout2 = args.dropout
        self.act = F.relu

        self.enc = RoleGCNConv(args.nhid + args.nhid, args.nhid, agg=args.agg)            
        self.enc_mu = RoleGCNConv(args.nhid, args.nout, act=lambda x:x, agg=args.agg)
        self.enc_logstd = RoleGCNConv(args.nhid, args.nout, act=F.softplus, agg=args.agg)

        self.eps = args.EPS

        self.prior = nn.Sequential(nn.Linear(args.nhid,args.nhid), nn.ReLU())
        self.prior_mean = nn.Sequential(nn.Linear(args.nhid, args.nout))
        self.prior_std = nn.Sequential(nn.Linear(args.nhid, args.nout), nn.Softplus())

        self.graph_gru = GraphGRU(args.nout+args.nout, args.nout, agg=args.agg)

        self.nhid = args.nhid
        self.nout = args.nout
        
        self.Q = Parameter(torch.ones((args.nout, args.nhid)), requires_grad=True)
        self.r = Parameter(torch.ones((args.nhid, 1)), requires_grad=True)
        self.num_window = args.nb_window
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.Q)
        glorot(self.r)
        glorot(self.feat)
        glorot(self.linear.weight)
        glorot(self.hidden_initial)

    def reparameterize(self, mu, logstd):
        eps1 = torch.FloatTensor(logstd.size()).normal_().to(self.device)
        eps1 = Variable(eps1)
        return eps1.mul(logstd).add_(mu)

    def init_hiddens(self):
        self.hiddens = [self.hidden_initial] * self.num_window
        return self.hiddens

    def weighted_hiddens(self, hidden_window):
        # temporal self-attention
        e = torch.matmul(torch.tanh(torch.matmul(hidden_window, self.Q)), self.r)
        e_reshaped = torch.reshape(e, (self.num_window, -1))
        a = F.softmax(e_reshaped, dim=0).unsqueeze(2)
        hidden_window_new = torch.reshape(hidden_window, [self.num_window, -1, self.nhid])
        s = torch.mean(a * hidden_window_new, dim=0) # torch.sum is also applicable
        return s

    # replace all nodes
    def update_hiddens_all_with(self, z_t):
        self.hiddens.pop(0)
        self.hiddens.append(z_t.clone().detach().requires_grad_(False))
        return z_t

    # replace current nodes state
    def update_hiddens_with(self, z_t, nodes):
        last_z = self.hiddens[-1].detach_().clone().requires_grad_(False)
        last_z[nodes, :] = z_t[nodes, :].detach_().clone().requires_grad_(False)
        self.hiddens.pop(0)
        self.hiddens.append(last_z)
        return last_z


    def forward(self, edge_index, x = None, node_role = None):
        if x is None:  # using trainable feat matrix
            x = self.linear(self.feat)
        else:
            x = self.linear(x)

        hlist = torch.cat(self.hiddens, dim=0)
        h = self.weighted_hiddens(hlist)

        enc = self.enc(torch.cat([x, h], 1), edge_index, node_role)
        mu = self.enc_mu(enc, edge_index, node_role)
        logstd = self.enc_logstd(enc, edge_index, node_role)

        z = self.reparameterize(mu, logstd)

        z = self.linear_z(z)

        prior = self.prior(h)
        prior_mean = self.prior_mean(prior)
        prior_std = self.prior_std(prior)

        kld_loss = self._kld_gauss(mu, logstd, prior_mean, prior_std)
        # kld_loss = self._kld_gauss_zu(mu, logstd)

        h = self.graph_gru(torch.cat([x, z], 1), edge_index, h, node_role)

        return h,kld_loss
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)

    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
        std_log = torch.log(std_in + self.eps)
        kld_element =  torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
                                            torch.pow(torch.exp(std_log), 2), 1))
        return (-0.5 / num_nodes) * kld_element