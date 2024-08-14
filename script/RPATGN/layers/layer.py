import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter, scatter_add, scatter_max,scatter_mean
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

class RoleGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, act=F.relu, agg='mean', improved=True, bias=True):
        super(RoleGCNConv, self).__init__(aggr=None)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act 
        self.agg = agg

        self.improved = improved
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        self.num_node_roles = 3

        self.W_g = nn.Parameter(torch.Tensor(self.num_node_roles, out_channels))
        self.b_g = nn.Parameter(torch.Tensor(self.num_node_roles, out_channels))

        self.self_weights = nn.Parameter(torch.Tensor(self.num_node_roles, out_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_node_roles, out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_g, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.self_weights, a=math.sqrt(5))
        glorot(self.weight)
        zeros(self.bias)
        zeros(self.b_g)

    def forward(self, x, edge_index, node_role=None, edge_weight=None):
        self.node_role = node_role

        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1), ), dtype=x.dtype, device=x.device)
        edge_weight = edge_weight.view(-1)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        loop_weight = torch.full(
            (x.size(0), ),
            1 if not self.improved else 2,
            dtype=x.dtype,
            device=x.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0

        norm = deg_inv[row] * edge_weight * deg_inv[col]

        x = torch.matmul(x, self.weight)

        x = self.propagate(edge_index, x=x,norm=norm)

        x = self.act(x)

        return x

    def message(self, x_j, edge_index):
        if self.node_role==None:
            return x_j
        
        node_role = self.node_role[edge_index[0]]

        role_distributions = F.one_hot(node_role, num_classes=self.num_node_roles).float().to(x_j.device)

        g = torch.sigmoid(torch.matmul(role_distributions, self.W_g) + self.b_g[node_role])

        x_j *= g

        if self.bias is not None:
            bias = self.bias[node_role]
            x_j += bias

        return x_j
    
    def aggregate(self, inputs, index, dim_size=None):
        if self.agg=='mean':
            return scatter_mean(inputs, index, dim=0, dim_size=dim_size)
        elif self.agg=='add':
            return scatter_add(inputs, index, dim=0, dim_size=dim_size)
    
    def update(self, aggr_out,x):
        if self.node_role==None:
            return aggr_out
        for nr in torch.unique(self.node_role):
            nr_mask = self.node_role == nr
            nr_self = x[nr_mask]
            nr_aggr = aggr_out[nr_mask]
            weighted_self = torch.matmul(nr_self, self.self_weights[nr])
            aggr_out[nr_mask] = weighted_self + nr_aggr
            
        return aggr_out
    

class GraphGRU(nn.Module):
    def __init__(self, input_size, hidden_size, agg='mean'):
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size

        # Define graph convolution layers for update gate, reset gate, and candidate hidden state
        self.conv_xz = RoleGCNConv(input_size, hidden_size, agg=agg, bias=True)
        self.conv_hz = RoleGCNConv(hidden_size, hidden_size, agg=agg, bias=False)
        self.conv_xr = RoleGCNConv(input_size, hidden_size, agg=agg, bias=True)
        self.conv_hr = RoleGCNConv(hidden_size, hidden_size, agg=agg, bias=False)
        self.conv_xh = RoleGCNConv(input_size, hidden_size, agg=agg, bias=True)
        self.conv_hh = RoleGCNConv(hidden_size, hidden_size, agg=agg, bias=False)

    def forward(self, x, edge_index, h_prev, node_role = None):
        # the update gate
        z = torch.sigmoid(self.conv_xz(x, edge_index, node_role) + self.conv_hz(h_prev, edge_index, node_role))
        # the reset gate
        r = torch.sigmoid(self.conv_xr(x, edge_index, node_role) + self.conv_hr(h_prev, edge_index, node_role))
        # the candidate hidden state
        h_tilde = torch.tanh(self.conv_xh(x, edge_index, node_role) + self.conv_hh(r * h_prev, edge_index, node_role))
        # the new hidden state
        h_new = z * h_prev + (1 - z) * h_tilde

        return h_new
