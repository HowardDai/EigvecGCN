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

from tqdm import tqdm

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

# FOR THE POSITIONAL WAVELET EMBEDDINGS 

from collections import deque




def farthest_node_sparse(adj: torch.sparse_coo_tensor, start: int):
    """
    adj: N×N sparse adjacency in COO form
    start: starting node index
    returns: (farthest_node, distance)
    """
    # make sure we’re in COO and coalesced
    adj = adj.coalesce()
    N = adj.size(0)
    device = adj.device

    # bit‐vectors on nodes
    visited  = torch.zeros(N, dtype=torch.bool, device=device)
    dist     = torch.full((N,), -1, dtype=torch.int64, device=device)
    frontier = torch.zeros(N, dtype=torch.bool, device=device)

    # initialize
    visited[start]    = True
    dist[start]       = 0
    frontier[start]   = True
    current_distance = 0

    # BFS loop via sparse mm
    while frontier.any():
        # propagate to neighbors: adj @ frontier → float counts
        neigh_counts = torch.sparse.mm(adj, frontier.unsqueeze(1).float()).squeeze(1)
        # any nonzero → neighbor
        neigh = neigh_counts > 0

        # new frontier = those neighbors not yet visited
        new_frontier = neigh & (~visited)
        if not new_frontier.any():
            break

        current_distance += 1
        dist[new_frontier] = current_distance
        visited |= new_frontier
        frontier = new_frontier

    # pick the farthest
    far_node = int(dist.argmax().item())
    far_dist = int(dist.max().item())
    return far_node, far_dist


def find_diameter_endpoints(adj: torch.sparse_coo_tensor):
    """
    Returns (u1, u2) approximate diameter endpoints,
    by doing 2 runs of farthest_node_sparse.
    """
    # start from node 0 (or any arbitrary node)
    u1, _ = farthest_node_sparse(adj, start=0)
    u2, _ = farthest_node_sparse(adj, start=u1)
    return u1, u2

def degree_node_selection(adj: torch.sparse_coo_tensor, k, largest=True):
    degrees = torch.sum(adj.to_dense(), dim = 0)
    _, indices = degrees.topk(k, largest=largest)
    return indices

def generate_wavelet_bank(data: Data, num_scales=10, lazy_parameter=0.5, abs_val = False):
    adj = data.edge_index
    degree = torch.diag(torch.sum(adj.to_dense(), dim = 0))
    diff_op = adj @ torch.inverse(degree)

    N = adj.size(0)

    lazy_diff_op = lazy_parameter * torch.eye(N) + (1-lazy_parameter) * diff_op

    diff_op_1 = lazy_diff_op
    diff_op_2 = None

    filters = ()

    for i in range(num_scales):
        diff_op_2 = diff_op_1 @ diff_op_1 # iterative squaring
        wavelet_filter = diff_op_1  - diff_op_2 

        if abs_val:
            wavelet_filter = torch.abs(wavelet_filter)

        filters = filters + (wavelet_filter,)

        diff_op_1 = diff_op_2 

    return filters


# Given a graph, computes an pseudo-positional/harmonic embedding on the nodes by the following:
# 1. Find the two "outermost" nodes
# 2. For each of the two nodes, run wavelet transform with a starting signal as a dirac on that node, at variable scales
# 3. 
def wavelet_transform_positional(data: Data, num_scales=10, lazy_parameter=0.5):

    adj = data.edge_index
    n1, n2 = find_diameter_endpoints(adj)
    N = adj.size(0)

    filters = generate_wavelet_bank(data, num_scales=10, lazy_parameter=0.5)

    embs1 = torch.zeros(N, num_scales)
    embs2 = torch.zeros(N, num_scales)

    for i in range(num_scales):

        embs1[:, i] = filters[i][:, n1]
        embs2[:, i] = filters[i][:, n2]

    embs = torch.cat((embs1, embs2), dim=1)
    return embs 



def get_padded_eigvecs(adj: torch.Tensor, max_graph_size: int = 300):
    """
    Compute eigenvalues/eigenvectors of Laplacian(adj), then
    pad both to size `max_graph_size` with zeros (or trim if larger).
    
    Returns:
      evals:  Tensor of shape (max_graph_size,)
      evecs: Tensor of shape (max_graph_size, max_graph_size)
    """
    # Build Laplacian
    lap = get_lap(adj)

    # Full spectrum
    evals, evecs = torch.linalg.eigh(lap)      # shapes (n,), (n, n)
    n = evals.size(0)

    # If graph bigger, truncate
    if n > max_graph_size:
        evals = evals[:max_graph_size]
        evecs = evecs[:max_graph_size, :max_graph_size]
        return evals, evecs

    # Otherwise, pad up to max_graph_size
    pad_len = max_graph_size - n

    # 1) pad evals: concatenate zeros
    pad_evals = torch.zeros(pad_len, device=evals.device, dtype=evals.dtype)
    evals_padded = torch.cat([evals, pad_evals], dim=0)  # shape = (max_graph_size,)

    # 2) pad evecs: add zero‐rows and zero‐columns
    #    F.pad takes (pad_left, pad_right, pad_top, pad_bottom)
    evecs_padded = F.pad(evecs,
                        # columns: (left, right) = (0, pad_len)
                        # rows   : (top, bottom) = (0, pad_len)
                        pad=(0, pad_len, 0, pad_len),
                        mode='constant', value=0.0)
    # now shape = (n+pad_len, n+pad_len) = (max_graph_size, max_graph_size)

    return evals_padded, evecs_padded


##Geometric Scattering


def scattering_transform(data: Data, num_scales=10, lazy_parameter=0.5, wavelet_inds=[]):
    filters = generate_wavelet_bank(data, num_scales, lazy_parameter, abs_val = False)
    
    if len(wavelet_inds) != 0:
        filters = [filters[i] for i in wavelet_inds]
    U = torch.abs(filters[0])
    for i in range(1, len(filters)):
        U = torch.abs(U @ filters[i])

    nodes = list(find_diameter_endpoints(data.edge_index))

    embeddings = U[:, nodes]

    return embeddings

def global_scattering_transform(data: Data, num_scales=10, lazy_parameter=0.5, num_moments=5, wavelet_inds=[]):
    filters = generate_wavelet_bank(data, num_scales, lazy_parameter, abs_val = False)

    if len(wavelet_inds) != 0:
        
        filters = [filters[i] for i in wavelet_inds]

    U = torch.abs(filters[0])
    for i in range(1, len(filters)):
        U = torch.abs(U @ filters[i])
    
    


    m0 = torch.sum(torch.abs(U), dim=0)
    
    moments = torch.unsqueeze(m0, 1)

    for i in range(1, num_moments):
        m_i = torch.sum(U**(i+1), dim =0)
        m_i = torch.unsqueeze(m_i, 1)
        moments = torch.cat((moments, m_i), dim=1)
    
    return moments

    
        

        


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


def pre_transform(data):  # computes eigvecs eigvals and reformats edge_index
    data.edge_index = edge_index_to_sparse_adj(data.edge_index, data.num_nodes)
    evals, evecs = get_padded_eigvecs(data.edge_index)
    data.eigvecs = evecs 
    data.eigvals = evals
 
    return data

class DataEmbeddings:
    def __init__(self, config):
        self.config = config

    def __call__(self, data: Data) -> Data:
        # BUILDING EMBEDDINGS

        if self.config.diffusion_emb:
            U = diffusion_transform(data)
            data.diffusion_emb = diffusion_convolution(U, data)
            # data.x = torch.cat((data.x, diffusion_convolution(U, data)), dim=-1)

        if self.config.wavelet_emb:
            # data.x = torch.cat((data.x, wavelet_transform_positional(data)), dim=-1)
            data.wavelet_emb = wavelet_transform_positional(data)
        if self.config.scatter_emb:
            # data.x = torch.cat((data.x, scattering_transform(data)), dim=-1)
            wavelet_paths = custom_wavelet_choices()
            data.scatter_emb = torch.zeros(data.num_nodes, 0, dtype=torch.float32)
            for i in range(len(wavelet_paths)):
                data.scatter_emb = torch.cat((data.scatter_emb, scattering_transform(data,wavelet_inds = wavelet_paths[i])), dim=-1)

        if self.config.global_scatter_emb:
            wavelet_paths = custom_wavelet_choices()
            data.global_scatter_emb = torch.zeros(data.num_nodes, 0, dtype=torch.float32)
            for i in range(len(wavelet_paths)):
                data.global_scatter_emb = torch.cat((data.scatter_emb, global_scattering_transform(data,wavelet_inds = wavelet_paths[i])), dim=-1)

            # data.x = torch.cat((data.x, global_scattering_transform(data)), dim=-1)


        # CONCATENATING THE EMBEDDINGS 
        data.x = torch.ones(data.num_nodes, 0, dtype=torch.float32)


        if self.config.diffusion_emb:
            data.x = torch.cat((data.x, data.diffusion_emb), dim=-1)

        if self.config.wavelet_emb:
            data.x = torch.cat((data.x, data.wavelet_emb), dim=-1)

        if self.config.scatter_emb:
            data.x = torch.cat((data.x, data.scatter_emb), dim=-1)

        if self.config.global_scatter_emb:
            data.x = torch.cat((data.x, data.global_scatter_emb), dim=-1)


        if data.x.shape[-1] == 0: # trivial embeddings, if no other embeddings
            data.x = torch.ones(data.num_nodes, 1, dtype=torch.float32)
    
        return data


class DataTransform:
    def __init__(self, config):
        self.config = config

    def __call__(self, data: Data) -> Data:
        # data.edge_index = edge_index_to_sparse_adj(data.edge_index, data.num_nodes)

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




def load_data(config):

    # dataset and splits
    
    

    
        
    # pre_transform = DataPreTransform(config)
    transform = DataTransform(config)
    embeddings = DataEmbeddings(config)
    
    dataset = PygGraphPropPredDataset(root='data', name='ogbg-ppa', transform=transform, pre_transform=pre_transform)

    for data in tqdm(dataset): # apply embeddings to dataset first 
        data = embeddings(data)
    
    sample = dataset[0]
    print(sample.wavelet_emb.shape)
    print(sample.scatter_emb.shape)
    print(sample.global_scatter_emb.shape)
    # out = global_scattering_transform(sample)
    # # print(out.shape)

    split_idx = dataset.get_idx_split()

    
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True) # ISSUE: right now this is just concatenating everything in the batch, treating it lke a huge graph
    val_loader   = DataLoader(dataset[split_idx['valid']], batch_size=64, shuffle=False)
    test_loader  = DataLoader(dataset[split_idx['test']],  batch_size=1, shuffle=False)


    return dataset, train_loader, val_loader, test_loader