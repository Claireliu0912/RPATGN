import os
import sys
import time
import torch
import numpy as np
from math import isnan
import torch.nn as nn
import warnings


warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

class Runner(object):
    def __init__(self):
        self.len = data['time_length']
        self.start_train = 0
        self.train_shots = list(range(0, int(self.len*args.train_ratio)))
        self.val_shots = list(range(int(self.len*args.train_ratio),int(self.len*(args.train_ratio+args.val_ratio))))
        self.test_shots = list(range(int(self.len*args.train_ratio), self.len))
        self.load_feature()
        self.load_role()
        self.model = RPATGN(args).to(args.device)
        self.loss = ReconLoss(args)
        logger.info('total length: {}, test length: {}'.format(self.len, len(self.test_shots)))

    def load_feature(self):
        if args.trainable_feat:
            self.x = None
            logger.info("using trainable feature, feature dim: {}".format(args.nfeat))
        else:
            if args.pre_defined_feature is not None:
                import scipy.sparse as sp
                feature = sp.load_npz(feature_path).toarray()
                self.x = torch.from_numpy(feature).float().to(args.device)
                logger.info('using pre-defined feature')
            else:
                self.x = torch.eye(args.num_nodes).to(args.device)
                logger.info('using one-hot feature')
            args.nfeat = self.x.size(1)
            print(args.nfeat)

    def load_role(self):
        self.roles = []
        for t in range(self.len):
            edge_index, pos_index, neg_index, _, _,_,_ = prepare(data, t)
            self.roles.append(get_node_roles(edge_index, args.num_nodes,args.threshold))

    def optimizer(self, using_riemannianAdam=True):
        if using_riemannianAdam:
            import geoopt
            optimizer = geoopt.optim.radam.RiemannianAdam(self.model.parameters(), lr=args.lr,
                                                          weight_decay=args.weight_decay)
        else:
            import torch.optim as optim
            optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def run(self):
        optimizer = self.optimizer()
        t_total0 = time.time()
        best_result = 0
        best_model = None
        clip = 10

        self.model.train()
        for epoch in range(1, args.max_epoch + 1):
            t0 = time.time()
            train_losses = []
            self.model.init_hiddens()
            # train
            self.model.train()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            for t in self.train_shots:
                edge_index, pos_index, neg_index, activate_nodes, edge_weigh, _, _ = prepare(data, t)

                optimizer.zero_grad()

                z, kld_loss = self.model(edge_index, self.x, self.roles[t])
                train_loss = self.loss(z, edge_index) + kld_loss

                train_loss.backward()
                optimizer.step()
                train_losses.append(train_loss.item())
                self.model.update_hiddens_all_with(z)

            self.model.eval()
            with torch.no_grad():
                val_auc_list, val_ap_list = [], []
                embeddings = z
                embeddings = embeddings.detach()
                for t in self.val_shots:
                    edge_index, pos_edge, neg_edge = prepare(data, t)[:3]
                    auc, ap = self.loss.predict(embeddings, pos_edge, neg_edge)
                    val_auc_list.append(auc)
                    val_ap_list.append(ap)

            average_val_auc = np.mean(val_auc_list)
            if average_val_auc > best_result:
                best_result = average_val_auc
                best_test_results = self.test(epoch, z)
                best_test_z = z
                best_model = self.model
                patience = 0
                logger.info(
                    "Epoch:{:}, Val AUC: {:.4f}, AP: {:.4f},".format(epoch,average_val_auc,np.mean(val_ap_list)))
            else:
                patience += 1
                if epoch > args.min_epoch and patience > args.patience:
                    print('early stopping')
                    break
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info('==' * 27)
                logger.info("Epoch:{}, Train Loss: {:.4f}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, np.mean(train_losses),
                                                                                          time.time() - t0,
                                                                                          gpu_mem_alloc))
            if isnan(train_loss):
                logger.info("Best Test AUC: {:.4f}, AP: {:.4f}".format(best_test_results[1], best_test_results[2]))
                print('nan loss')
                break
        
        logger.info("Best Test AUC: {:.4f}, AP: {:.4f}".format(best_test_results[1], best_test_results[2]))
        logger.info('>> Total time : %6.2f' % (time.time() - t_total0))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))

    def test(self, epoch, embeddings=None):
        auc_list, ap_list = [], []
        embeddings = embeddings.detach()
        for t in self.test_shots:
            edge_index, pos_edge, neg_edge = prepare(data, t)[:3]
            auc, ap = self.loss.predict(embeddings, pos_edge, neg_edge)
            auc_list.append(auc)
            ap_list.append(ap)
        logger.info(
            'Epoch:{}, average AUC: {:.4f}; average AP: {:.4f}'.format(epoch, np.mean(auc_list), np.mean(ap_list)))
        return epoch, np.mean(auc_list), np.mean(ap_list)


if __name__ == '__main__':
    from script.config import args
    from script.utils.util import set_random, logger, init_logger, feature_path, get_node_roles
    from script.model import RPATGN
    from script.loss import ReconLoss
    from script.utils.data_util import loader, prepare_dir
    from script.inits import prepare

    data = loader(dataset=args.dataset)
    args.num_nodes = data['num_nodes']
    set_random(args.seed)
    init_logger(prepare_dir(args.output_folder) + args.dataset + '.txt')
    runner = Runner()
    runner.run()
