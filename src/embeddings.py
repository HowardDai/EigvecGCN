import torch 
import random
from typing import List
import numpy as np

from utils import *


# DIFFUSION TRANSFORM: 

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

def wavelet_moments(data: Data, num_moments=4, num_scales=10, lazy_parameter=0.5):
    adj = data.edge_index
    N = adj.size(0)

    filters = generate_wavelet_bank(data, num_scales=10, lazy_parameter=0.5)
    signal = torch.ones(N)

    embs = torch.zeros(N, num_scales * num_moments)

    for filter in filters:
        i = 0
        for moment in range(num_moments):
            embs[:, i] = (filter**moment) @ signal
             i += 1
    return embs


def neighbors_signal(data: Data, num_scales=10, lazy_parameter=0.5):
    adj = data.edge_index
    N = adj.size(0)

    filters = generate_wavelet_bank(data, num_scales=10, lazy_parameter=0.5)
    signal = torch.ones(N)

    embs = torch.zeros(N, num_scales)

    for i in range(N):
        for j in range(num_scales):
            signal = adj[i]
            embs[i, j] = filters[j, i] @ signal

    return embs

def local_diffused_signal(data: Data, num_scales=10, lazy_parameter=0.5):
    adj = data.edge_index
    N = adj.size(0)

    filters = generate_wavelet_bank(data, num_scales=10, lazy_parameter=0.5)
    signal = torch.zeros(N)

    embs = torch.zeros(N, num_scales)

    adj = data.edge_index
    degree = torch.diag(torch.sum(adj.to_dense(), dim = 0))
    diff_op = adj @ torch.inverse(degree)

    for i in range(N):
        for j in range(num_scales):
            signal = torch.zeros(N)
            signal[i] += 1

            signal = diff_op @ signal

            embs[i, j] = filters[j, i] @ signal

    return embs