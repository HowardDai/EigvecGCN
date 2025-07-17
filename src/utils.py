import torch
import numpy as np
import scipy.sparse as sparse
from torch_geometric.utils import to_dense_adj


from typing import List


from easydict import EasyDict
import yaml

##Geometric Scattering

# for generating all possible wavelet combinations 
def all_index_combinations(k: int) -> List[List[int]]:
    """
    Return all subsets of {0,1,...,k-1}, sorted by
    the integer value of their binary inclusion mask.
    """
    result = []
    for mask in range(1, 1 << k):            # 0 .. 2^k - 1
        subset = [i for i in range(k) 
                  if (mask >> i) & 1]     # include i if bit i is 1
        result.append(subset)
    return result



def edge_index_to_sparse_adj(edge_index: torch.LongTensor, num_nodes: int) -> torch.Tensor:
    # edge_index: [2, E], num_nodes: N
    row, col = edge_index
    # if our graph is undirected, you may want to add the reverse edges here
    # e.g. row = torch.cat([row, col]); col = torch.cat([col, row])
    values = torch.ones(row.size(0), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0),
        values,
        (num_nodes, num_nodes),
    ).coalesce()

    return adj



import gc, psutil, os
from custom import *

def log_cpu(name=""):
    # gc.collect()
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
    print(f"[{name}] RSS: {rss:.1f} MB")

def get_lap(adj):
    degree = torch.diag(torch.sum(adj.to_dense(), dim = 0))
    lap = degree - adj
    return lap

def get_diffusion(adj):
    degree = torch.diag(torch.sum(adj.to_dense(), dim = 0))
    return adj @ torch.inverse(degree)


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


def merge_configs(default: EasyDict, override: EasyDict) -> EasyDict:
    """
    For each key in override, if both default[k] and override[k]
    are dict‑like, recurse; otherwise take override[k].
    """
    for k, v in override.items():
        if (
            k in default
            and isinstance(default[k], EasyDict)
            and isinstance(v, dict)
        ):
            # recurse into nested EasyDict
            default[k] = merge_configs(default[k], EasyDict(v))
        else:
            # override or add new key
            default[k] = v
    return default