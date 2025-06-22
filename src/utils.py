import torch

import numpy as np
import scipy.sparse as sparse
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch.nn as nn


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


def convert_scipy_to_torch_sparse(matrix):
    matrix_helper_coo = matrix.tocoo().astype('float32')
    data = torch.FloatTensor(matrix_helper_coo.data)
    rows = torch.LongTensor(matrix_helper_coo.row)
    cols = torch.LongTensor(matrix_helper_coo.col)
    indices = torch.vstack([rows, cols])

    shape = torch.Size(matrix_helper_coo.shape)
    matrix = torch.sparse.FloatTensor(indices, data, shape)
    return matrix

def load_data(config):

    # dataset and splits 
    dataset = PygGraphPropPredDataset(root='data', name='ogbg-ppa', transform=data_transforms)

    split_idx = dataset.get_idx_split()

    
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True) # ISSUE: right now this is just concatenating everything in the batch, treating it lke a huge graph
    val_loader   = DataLoader(dataset[split_idx['valid']], batch_size=64, shuffle=False)
    test_loader  = DataLoader(dataset[split_idx['test']],  batch_size=64, shuffle=False)


    return dataset, train_loader, val_loader, test_loader