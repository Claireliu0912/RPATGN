import numpy as np
import scipy.sparse as sp


def mask_edges_prd(adjs_list):
    pos_edges_l, false_edges_l = [], []
    edges_list = []
    for i in range(0, len(adjs_list)):
        # Function to build test set with 10% positive links
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

        adj = adjs_list[i]
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0

        adj_triu = sp.triu(adj)
        adj_tuple = sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = sparse_to_tuple(adj)[0]
        num_false = int(edges.shape[0])

        pos_edges_l.append(edges)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        edges_false = []
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            edges_false.append([idx_i, idx_j])

        assert ~ismember(edges_false, edges_all)

        false_edges_l.append(np.asarray(edges_false))

    # NOTE: these edge lists only contain single direction of edge!
    return pos_edges_l, false_edges_l


def test_adj(adjs_list, adj_orig_dense_list):
    # this method is to test the adj_list and adj_orig_dense_list is same or not
    for i, a in enumerate(adj_orig_dense_list):
        a = sp.csr_matrix(a)
        a = a - sp.dia_matrix((a.diagonal()[np.newaxis, :], [0]), shape=a.shape)
        a.eliminate_zeros()
        assert np.diag(a.todense()).sum() == 0
        team1 = sp.csr_matrix(a).todok().tocoo()
        print(len(list(team1.col.reshape(-1))))

        adj = adjs_list[i]
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        assert np.diag(adj.todense()).sum() == 0
        team2 = sp.csr_matrix(adj).todok().tocoo()

        print(len(list(team2.col.reshape(-1))))
        print('==')

def tuple_to_array(lot):
    out = np.array(list(lot[0]))
    for i in range(1, len(lot)):
        out = np.vstack((out, np.array(list(lot[i]))))
    return out


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_edges_det(adjs_list):
    '''
    produce edge_index in np format
    '''
    edges_list = []
    biedges_list = []
    for i in range(0, len(adjs_list)):
        adj = adjs_list[i]
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0

        adj_triu = sp.triu(adj)
        edges = sparse_to_tuple(adj_triu)[0]  # single directional
        np.random.shuffle(edges)
        edges_list.append(edges)
        biedges = sparse_to_tuple(adj)[0]  # bidirectional edges
        np.random.shuffle(biedges)
        biedges_list.append(biedges)

    return edges_list, biedges_list
