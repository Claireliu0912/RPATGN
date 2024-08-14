import os
import numpy as np
import torch
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
import pickle
from script.utils.make_edges_orign import mask_edges_det, mask_edges_prd


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def load_dataset(dataset):
    assert dataset in ['transaction','elliptic','bitcoin','alpha']
    with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')
    print('>> generating edges,negative edges and new edges, wait for a while ...')
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    print('>> processing finished!')
    assert len(edges) == len(biedges) == len(pedges) == len(nedges)
    edge_index_list, pedges_list, nedges_list = [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data

def loader(dataset='transaction'):
    # if cached, load directly
    data_root = '../data/input/cached/{}/'.format(dataset)
    filepath = mkdirs(data_root) + '{}.data'.format(dataset)
    if os.path.isfile(filepath):
        print('loading {} directly'.format(dataset))
        return torch.load(filepath)
    # if not cached, to process and cached
    print('>>data is not exits, processing ...')
    if dataset in ['transaction','elliptic','bitcoin','alpha']:
        data = load_dataset(dataset)
    torch.save(data, filepath)
    print('saved!')
    return data
