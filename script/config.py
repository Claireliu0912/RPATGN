import argparse
import torch
import os
import yaml

parser = argparse.ArgumentParser(description='RPATGN')

parser.add_argument('--config', type=str, default='', help='hyperparameter configuration')
parser.add_argument('--dataset', type=str, default='bitcoin', help='datasets')
parser.add_argument('--data_pt_path', type=str, default='', help='need to be modified')
parser.add_argument('--num_nodes', type=int, default=-1, help='num of nodes')
parser.add_argument('--nfeat', type=int, default=128, help='dim of input feature')
parser.add_argument('--nhid', type=int, default=64, help='dim of hidden embedding')
parser.add_argument('--nout', type=int, default=64, help='dim of output embedding')

parser.add_argument('--max_epoch', type=int, default=200, help='number of epochs to train.')
parser.add_argument('--testlength', type=int, default=1, help='length for test, default:3')
parser.add_argument('--train_ratio', type=float, default=0.5, help='the ratio of train dataset, default:0.5')
parser.add_argument('--val_ratio', type=float, default=0.1, help='the ratio of validation dataset, default:0.1')
parser.add_argument('--device', type=str, default='cpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--repeat', type=int, default=1, help='running times')
parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')
parser.add_argument('--threshold', type=int, default=2, help='degree threshold for role discovery')
parser.add_argument('--output_folder', type=str, default='', help='need to be modified')
parser.add_argument('--debug_content', type=str, default='', help='debug_mode content')
parser.add_argument('--sampling_times', type=int, default=1, help='negative sampling times')
parser.add_argument('--log_interval', type=int, default=20, help='log interval, default: 20,[20,40,...]')
parser.add_argument('--pre_defined_feature', default=None, help='pre-defined node feature')
parser.add_argument('--save_embeddings', type=int, default=0, help='save or not, default:0')
parser.add_argument('--debug_mode', type=int, default=0, help='debug_mode, 0: normal running; 1: debugging mode')
parser.add_argument('--min_epoch', type=int, default=100, help='min epoch')

parser.add_argument('--model', type=str, default='RPATGN', help='model name')
parser.add_argument('--agg', type=str, default='mean', help='aggregation method: [mean, add]')
parser.add_argument('--EPS', type=float, default=1e-15, help='eps')
parser.add_argument('--nb_window', type=int, default=5, help='the length of window')
parser.add_argument('--bias', type=bool, default=True, help='use bias or not')
parser.add_argument('--trainable_feat', type=int, default=0,
                    help='using trainable feat or one-hot feat, default: none-trainable feat')
parser.add_argument('--dropout', type=float, default=0, help='dropout rate (1 - keep probability).')

args = parser.parse_args()

# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda".format(args.device_id))
    torch.cuda.set_device(int(args.device_id))
    print('using gpu:{} to train the model'.format(args.device_id))
else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')

# 设置随机数种子
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

args.output_folder = '../data/output/log/{}/{}/'.format(args.dataset, args.model)
args.result_txt = '../data/output/results/{}_{}_result.txt'.format(args.dataset, args.model)

# open debugging mode
if args.debug_mode == 1:
    print('start debugging mode!')
    folder = '../data/output/ablation_study/{}/'.format(args.debug_content)
    args.result_txt = folder + '{}_{}_result.txt'.format(args.dataset, args.model)
    if not os.path.isdir(folder):
        os.makedirs(folder)
