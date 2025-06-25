import torch

import collections
import collections.abc

# re-expose for legacy imports
collections.Mapping  = collections.abc.Mapping
collections.Iterable = collections.abc.Iterable
collections.MutableMapping = collections.abc.MutableMapping

from utils import *
import dgl


import numpy as np
import scipy.sparse as sparse
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch.nn as nn

def diffusion_transform(data: Data):
    adj = data.edge_index
    degree = torch.diag(torch.sum(adj.to_dense(), dim = 0))
    diff_op = adj @ torch.inverse(degree)

    u0 = torch.eye(adj.shape[0])
    u2 = diff_op @ diff_op @ u0
    u16 = u2
    for i in range(14):
        u16 = diff_op @ u2
    return torch.stack((u0, u2, u16))

def diffusion_convolution(U, data: Data):
    h = []
    for k in range(U.shape[0]):
        h_k = U[k] @ data.edge_index
        h.append(padding(h_k))
    h = tuple(h)
    X = torch.cat(h, dim=1)

    return X

def padding(h_k, length=1000):
    pad = (0, length - h_k.shape[1])
    h_k = F.pad(h_k, pad)
    return h_k


def edge_index_to_sparse_adj(edge_index: torch.LongTensor, num_nodes: int) -> torch.Tensor:
    # edge_index: [2, E], num_nodes: N
    row, col = edge_index
    # if your graph is undirected, you may want to add the reverse edges here
    # e.g. row = torch.cat([row, col]); col = torch.cat([col, row])
    values = torch.ones(row.size(0), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0),
        values,
        (num_nodes, num_nodes),
    ).coalesce()

    return adj


def data_transforms(data: Data) -> Data:
    if data.x is None: # adding trivial features
        data.x = torch.ones(data.num_nodes, 1, dtype=torch.float32)
    data.edge_index = edge_index_to_sparse_adj(data.edge_index, data.num_nodes) # converting to an adjacency matrix

    U = diffusion_transform(data)
    X = diffusion_convolution(U, data)
    data.x = X

    return data


def enumerate_labels(labels):
    """ Converts the labels from the original
        string form to the integer [0:MaxLabels-1]
    """
    unique = list(set(labels))
    labels = np.array([unique.index(label) for label in labels])
    return labels


def normalize_adjacency(adj):
    """ Normalizes the adjacency matrix according to the
        paper by Kipf et al.
        https://arxiv.org/pdf/1609.02907.pdf
    """
    adj = adj + sparse.eye(adj.shape[0])

    node_degrees = np.array(adj.sum(1))
    node_degrees = np.power(node_degrees, -0.5).flatten()
    node_degrees[np.isinf(node_degrees)] = 0.0
    node_degrees[np.isnan(node_degrees)] = 0.0
    degree_matrix = sparse.diags(node_degrees, dtype=np.float32)

    adj = degree_matrix @ adj @ degree_matrix
    return adj

def normalize_by_batch(x, batch):
    num_groups = int(batch.max()) + 1
    norm_sq = x.new_zeros(num_groups, x.shape[-1]).index_add(0, batch, x.pow(2))
    norms  = norm_sq.sqrt()
    x_norm = x / norms[batch]

    return x_norm


def orthogonalize_by_batch(x, batch):
    """
    Orthonormalize the set of vectors in each batch group.
    
    x:     (m, k) tensor, where m = num_groups * 30
           each row is one k-dimensional vector
    batch: (m,) long tensor with values in {0,…,num_groups-1},
           exactly 30 rows per group
    returns: x_orth of same shape, where for each g,
             the 30 rows x[batch==g] are replaced by an
             orthonormal set in R^k.
    """
    num_groups = int(batch.max().item()) + 1
    x_orth = torch.empty_like(x)

    for g in range(num_groups):
        mask = (batch == g)
        Xg = x[mask]              # shape (30, k)
        # QR on the transpose → Q has shape (k, 30) with orthonormal columns
        Q, R = torch.linalg.qr(Xg)
        Xg_orth = Q            
        x_orth[mask] = Xg_orth
    
    return x_orth
 

def convert_scipy_to_torch_sparse(matrix):
    matrix_helper_coo = matrix.tocoo().astype('float32')
    data = torch.FloatTensor(matrix_helper_coo.data)
    rows = torch.LongTensor(matrix_helper_coo.row)
    cols = torch.LongTensor(matrix_helper_coo.col)
    indices = torch.vstack([rows, cols])

    shape = torch.Size(matrix_helper_coo.shape)
    matrix = torch.sparse.FloatTensor(indices, data, shape)
    return matrix


# def collate_fn(batch):
#     # batch is a list of tuple (graph, label)
#     graphs = [e[0] for e in batch]
#     g = dgl.batch(graphs)
#     labels = [e[1] for e in batch]
#     labels = torch.stack(labels, 0)
#     return g, labels


def load_data(config):

    # dataset and splits
    def supervised_transforms(data: Data) -> Data:
        data = data_transforms(data)
        adj = data.edge_index
        degree = torch.diag(torch.sum(adj.to_dense(), dim = 0))
        lap = degree - adj

        _, evecs = torch.linalg.eigh(lap)
        data.y = evecs[:, :config.num_eigenvectors]

        return data
    
    transform = None

    if config.use_supervised:
        transform = supervised_transforms
    else:
        transform = data_transforms


    dataset = PygGraphPropPredDataset(root='data', name='ogbg-ppa', transform=transform)

    split_idx = dataset.get_idx_split()

    
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True) # ISSUE: right now this is just concatenating everything in the batch, treating it lke a huge graph
    val_loader   = DataLoader(dataset[split_idx['valid']], batch_size=64, shuffle=False)
    test_loader  = DataLoader(dataset[split_idx['test']],  batch_size=1, shuffle=False)


    return dataset, train_loader, val_loader, test_loader