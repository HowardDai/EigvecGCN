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

import random
from torch.utils.data import Subset
from typing import List

from torch_geometric.datasets import ZINC


def diffusion_transform(data: Data):
    adj = data.edge_index
    degree = torch.diag(torch.sum(adj.to_dense(), dim = 0))
    diff_op = adj @ torch.inverse(degree)

    u0 = torch.eye(adj.shape[0])
    u2 = diff_op @ diff_op
    u16 = u2
    for i in range(4):
        u16 = u16 @ u16
    u2 = u2 @ u0
    u16 = u16 @ u0
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
    N = adj.size(0)

    filters = generate_wavelet_bank(data, num_scales=10, lazy_parameter=0.5)

    embs = torch.zeros(N, num_scales)
    signal = torch.ones(N)

    for i in range(num_scales):
        embs[:, i] = filters[i] @ signal
        
    return embs 



def get_padded_eigvecs(adj: torch.Tensor, max_graph_size: int):
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

# for generating all possible wavelet combinations 
def all_index_combinations(k: int) -> List[List[int]]:
    """
    Return all subsets of {0,1,...,k-1}, sorted by
    the integer value of their binary inclusion mask.
    """
    result = []
    for mask in range(1 << k):            # 0 .. 2^k - 1
        subset = [i for i in range(k) 
                  if (mask >> i) & 1]     # include i if bit i is 1
        result.append(subset)
    return result


def scattering_transform(data: Data, num_scales=5, lazy_parameter=0.5, wavelet_inds=[]):
    filters = generate_wavelet_bank(data, num_scales, lazy_parameter, abs_val = False)
    
    if len(wavelet_inds) != 0:
        filters = [filters[i] for i in wavelet_inds]
    
    signal = torch.ones(filters[0].shape[0]) / torch.norm(torch.ones(filters[0].shape[0]))

    U = torch.abs(filters[0] @ signal)
    for i in range(1, len(filters)):
        U = torch.abs(filters[i] @ U)



    return U.unsqueeze(dim=-1) # because there is only 1 signal in this case, edit this if more 

def global_scattering_transform(data: Data, num_scales=10, lazy_parameter=0.5, num_moments=5, wavelet_inds=[]):
    filters = generate_wavelet_bank(data, num_scales, lazy_parameter, abs_val = False)

    if len(wavelet_inds) != 0:
        
        filters = [filters[i] for i in wavelet_inds]

    U = torch.abs(filters[0])
    for i in range(1, len(filters)):
        U = torch.abs(filters[i] @ U)
    
    


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

# def get_diag(entries: torch.Tensor):
#     N = entries.size(0)
#     device = entries.device
#     # 2) build sparse diagonal: indices = [[0,1,2,…],[0,1,2,…]]
#     idx = torch.arange(N, device=device)
#     indices = torch.stack([idx, idx], dim=0)               # [2×N]
#     values  = entries                                    # [N]

#     D = torch.sparse_coo_tensor(indices, values, (N, N),
#                                 device=device)
#     return D
class DataPreTransform:

    def __init__(self, config):
        self.config = config
    
    def __call__(self, data: Data) -> Data:
        data.edge_index = edge_index_to_sparse_adj(data.edge_index, data.num_nodes)
        evals, evecs = get_padded_eigvecs(data.edge_index, self.config.evec_len)
        data.eigvecs = evecs 
        data.eigvals = evals
 
        return data

import time

class DataEmbeddings:
    def __init__(self, config):
        self.config = config

    def __call__(self, data: Data) -> Data:
        # BUILDING EMBEDDINGS
        t1 = time.time()
        if self.config.diffusion_emb:
            U = diffusion_transform(data)
            data.diffusion_emb = diffusion_convolution(U, data)
            # data.x = torch.cat((data.x, diffusion_convolution(U, data)), dim=-1)
        t2 = time.time()
        # print("Diffusion runtime:", t2-t1)

        t1 = time.time()
        if self.config.wavelet_emb:
            # data.x = torch.cat((data.x, wavelet_transform_positional(data)), dim=-1)
            data.wavelet_emb = wavelet_transform_positional(data)
        t2= time.time() 
        # print("Wavelet runtime:", t2-t1)


        t1 = time.time()
        if self.config.scatter_emb:
            # data.x = torch.cat((data.x, scattering_transform(data)), dim=-1)
            wavelet_paths = all_index_combinations(5)
            data.scatter_emb = torch.zeros(data.num_nodes, 0, dtype=torch.float32)
            for i in range(len(wavelet_paths)):
                data.scatter_emb = torch.cat((data.scatter_emb, scattering_transform(data,wavelet_inds = wavelet_paths[i])), dim=-1)
        t2 = time.time()
        # print("Scatter runtime:", t2-t1)

        t1=time.time()
        if self.config.global_scatter_emb:
            wavelet_paths = custom_wavelet_choices()
            data.global_scatter_emb = torch.zeros(data.num_nodes, 0, dtype=torch.float32)
            for i in range(len(wavelet_paths)):
                data.global_scatter_emb = torch.cat((data.scatter_emb, global_scattering_transform(data,wavelet_inds = wavelet_paths[i])), dim=-1)
        t2=time.time()
        # print("Global scatter runtime:", t2-t1)
        
            # data.x = torch.cat((data.x, global_scattering_transform(data)), dim=-1)


        # CONCATENATING THE EMBEDDINGS 
        t1=time.time()
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

        t2 = time.time()
        # print("Concatenation step:", t2-t1)

        #modifying the eigval and eigvec matrices to match the # of eigvecs we actually care about
        data.eigvecs = data.eigvecs[:, 0:self.config.num_eigenvectors]
        data.eigvals = data.eigvals[0:self.config.num_eigenvectors]
        return data


class DataTransform:
    def __init__(self, config):
        self.config = config

    def __call__(self, data: Data) -> Data:
        # data.edge_index = edge_index_to_sparse_adj(data.edge_index, data.num_nodes)
        # embeddings = DataEmbeddings(self.config)
        # data = embeddings(data)
        
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


from torch_geometric.data import InMemoryDataset

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None):
        super().__init__(root=None, transform=transform)
        self.data_list = data_list
        # if transform not None:
        #     self.transform = transform
        # else self.transform = lambda x: x

    def len(self):
        return len(self.data_list)

    def get(self, idx): # NOTE: Make sure these transforms actually work as intended 
        return self.data_list[idx]


def load_data(config):

    # dataset and splits
    
    data_root = 'data'
    data_name = config.dataset
    data_path = os.path.join(data_root, data_name)
    
        
    transform = DataTransform(config)
    embeddings = DataEmbeddings(config)
    pre_transform = DataPreTransform(config)
    
    subset_frac = config.use_mini_dataset


    if subset_frac < 1 and os.path.exists(os.path.join(data_path, f"mini_dataset_{subset_frac}")):
        print(f"Using {subset_frac} of dataset. Loading from previously saved subset")
        data_dict = torch.load(os.path.join(data_path, f"mini_dataset_{subset_frac}"))
        print("data_dict loaded!")
    elif config.dataset == 'ogbg_ppa':
        dataset = PygGraphPropPredDataset(root=data_root, name='ogbg-ppa', transform=transform, pre_transform=pre_transform)
        print('data object loaded!')
    elif config.dataset == 'zinc':
        dataset = dataset = ZINC(root=data_root, transform=transform, pre_transform=pre_transform)
        print('data object loaded!')

        # sample = dataset[0]
        # print(sample)

        split_idx = dataset.get_idx_split()


        # sample a fraction of each split
        seed = 42
        
        random.seed(seed)

        def sample_idx(idx_list):
            n = max(1, int(len(idx_list) * subset_frac))
            return random.sample(idx_list, n)

        if subset_frac < 1:
            print(f"sampling {subset_frac} of dataset")
            if os.path.exists(os.path.join(data_path, f"mini_dataset_indices_{subset_frac}")): # if the indices for this subset have already been generated
                print(f"loading previously generated subset indices...")
                idx_dict = torch.load(f"data/ogbg_ppa/mini_dataset_indices_{subset_frac}") 
                temp_train_idx = idx_dict['train']
                temp_val_idx = idx_dict['valid']
                temp_test_idx = idx_dict['test']
            else:
                temp_train_idx = torch.tensor(sample_idx(split_idx['train'].tolist()))
                temp_val_idx   = torch.tensor(sample_idx(split_idx['valid'].tolist()))
                temp_test_idx  = torch.tensor(sample_idx(split_idx['test'].tolist()))
                idx_dict = {'train': temp_train_idx, 'valid': temp_val_idx, 'test':temp_test_idx}
                torch.save(idx_dict, os.path.join(data_path, f"mini_dataset_indices_{subset_frac}"))
            

            all_indices = torch.cat((temp_train_idx, temp_val_idx, temp_test_idx))

            a = temp_train_idx.shape[0]
            b= temp_val_idx.shape[0] + a
            c= temp_test_idx.shape[0] + b 

            # produces train/val/test indices relative to NEW dataset (after subset)
            train_idx = torch.arange(start=0, end=a)
            val_idx = torch.arange(start=a, end=b)
            test_idx = torch.arange(start=b, end= c)
            print(a,b,c)
            print(all_indices.shape)
            print(split_idx['train'].shape)
            print(split_idx['valid'].shape)

        
        else:
            print("Using full dataset")
            train_idx = split_idx['train'] 
            val_idx = split_idx['valid']  
            test_idx = split_idx['test'] 
            N = len(dataset)
            all_indices = torch.arange(start=0, end=N) # basically just all the indices


        subdataset = dataset[all_indices]
        data_dict = {'train': subdataset[train_idx], 'valid': subdataset[val_idx], 'test': subdataset[test_idx]}


        if config.use_mini_dataset < 1:
            torch.save(data_dict, os.path.join(data_path, f"mini_dataset_{subset_frac}"))

    


    # preprocessing embeddings
    print("Processing embeddings...")

    need_emb = {'train': False, 'valid': False, 'test': False}
    data_dict_emb = {} 

    if config.train:
        need_emb['train'] = True
        need_emb['valid'] = True

    if config.test:
        need_emb['test'] = True



    for key in data_dict:
        if not need_emb[key]:
            print(f"No embeddings needed for {key}")
            continue

        print(f"Processing embeddings for {key}")
        modified_list = []
        embeddings = DataEmbeddings(config)
        
        for data in tqdm(data_dict[key]):
            data = embeddings(data)
            modified_list.append(data)

        # 2) wrap them into a tiny InMemoryDataset
        data_dict_emb[key] = CustomGraphDataset(modified_list)

    print("Embeddings processed!")

    if 'train' in data_dict_emb.keys():
        train_loader = DataLoader(data_dict_emb['train'], batch_size=32, shuffle=True) # ISSUE: right now this is just concatenating everything in the batch, treating it lke a huge graph
    else:
        train_loader = None
    if 'valid' in data_dict_emb.keys():
        val_loader   = DataLoader(data_dict_emb['valid'], batch_size=64, shuffle=False)
    else:
        val_loader = None
    if 'test' in data_dict_emb.keys():
        test_loader  = DataLoader(data_dict_emb['test'],  batch_size=1, shuffle=False)
    else:
        test_loader = None

    
    


    return data_dict_emb, train_loader, val_loader, test_loader