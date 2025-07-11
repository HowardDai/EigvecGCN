import torch

import collections
import collections.abc

# re-expose for legacy imports
collections.Mapping  = collections.abc.Mapping
collections.Iterable = collections.abc.Iterable
collections.MutableMapping = collections.abc.MutableMapping

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader

from torch_geometric.datasets import ZINC

import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm


import random
from utils import *
from embeddings import *

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

        data.perms = []
        data.x = torch.ones(data.num_nodes, 0, dtype=torch.float32)
        if self.config.diffusion_emb: # PERM INVARIANCE REQUIRED 
            l = data.x.shape[-1]
            data.x = torch.cat((data.x, data.diffusion_emb), dim=-1)
            r = data.x.shape[-1]
            data.perms.append([l, r])

        if self.config.wavelet_emb:
            data.x = torch.cat((data.x, wavelet_emb(data)), dim=-1)

        if self.config.scatter_emb:
            data.x = torch.cat((data.x, scatter_emb(data)), dim=-1)

        if self.config.global_scatter_emb:
            data.x = torch.cat((data.x, global_scatter_emb(data)), dim=-1)
        
        if self.config.wavelet_moments_emb:
            data.x = torch.cat((data.x, wavelet_moments_emb(data)), dim=-1)

        if self.config.neighbor_bump_emb:
            data.x = torch.cat((data.x, neighbors_signal_emb(data)), dim=-1)

        if self.config.neighbor_bump_emb:
            data.x = torch.cat((data.x, local_diffused_signal_emb(data)), dim=-1)
        

        """# BUILDING EMBEDDINGS
        t1 = time.time()
        if self.config.diffusion_emb: # PERM INVARIANCE REQUIRED 
            data.diffusion_emb = diffusion_emb(data)
        t2 = time.time()
        # print("Diffusion runtime:", t2-t1)

        t1 = time.time()
        if self.config.wavelet_emb:
            data.wavelet_emb = wavelet_emb(data)
        t2= time.time() 
        # print("Wavelet runtime:", t2-t1)


        t1 = time.time()
        if self.config.scatter_emb:
            data.scatter_emb = scatter_emb(data)
        t2 = time.time()
        # print("Scatter runtime:", t2-t1)

        t1=time.time()
        if self.config.global_scatter_emb:
            data.global_scatter_emb = global_scatter_emb(data)
        t2=time.time()
        # print("Global scatter runtime:", t2-t1)

        t1=time.time()
        if self.config.wavelet_moments_emb:
            data.wavelet_moments = wavelet_moments_emb(data)
        t2=time.time()

        t1=time.time()
        if self.config.neighbor_bump_emb:
            data.neighbor_bump = neighbors_signal_emb(data)
        t2=time.time()

        t1=time.time()
        if self.config.diffused_dirac_emb:
            data.diffused_dirac = local_diffused_signal_emb(data)
        t2=time.time()

        
    

        # CONCATENATING THE EMBEDDINGS 
        t1=time.time()
        data.x = torch.ones(data.num_nodes, 0, dtype=torch.float32)

        data.perms = []

        if self.config.diffusion_emb: # scales in n, requires random permutations
            l = data.x.shape[-1]
            data.x = torch.cat((data.x, data.diffusion_emb), dim=-1)
            r = data.x.shape[-1]
            data.perms.append([l, r]) # append interval which keeps track of permutations needed

        if self.config.wavelet_emb: # constant-sized
            data.x = torch.cat((data.x, data.wavelet_emb), dim=-1)

        if self.config.scatter_emb: # constant-sized
            data.x = torch.cat((data.x, data.scatter_emb), dim=-1)

        if self.config.global_scatter_emb: # constant-sized
            data.x = torch.cat((data.x, data.global_scatter_emb), dim=-1)


        if data.x.shape[-1] == 0: # trivial embeddings, if no other embeddings
            data.x = torch.ones(data.num_nodes, 1, dtype=torch.float32)

        t2 = time.time()"""
        # print("Concatenation step:", t2-t1)

        #modifying the eigval and eigvec matrices to match the # of eigvecs we actually care about
        data.eigvecs = data.eigvecs[:, 0:self.config.num_eigenvectors]
        data.eigvals = data.eigvals[0:self.config.num_eigenvectors]
        return data



class RandomTransform:
    def __init__(self, config):
        self.config = config

    def __call__(self, data: Data) -> Data:
        # data.edge_index = edge_index_to_sparse_adj(data.edge_index, data.num_nodes)
        # embeddings = DataEmbeddings(self.config)
        # data = embeddings(data)
        perm_indices = torch.randperm(self.config.evec_len) 
        
        for interval in data.perms:

            n = interval[1]-interval[0]
            assert(n % self.config.evec_len == 0)

            for i in range(int(n / self.config.evec_len)):
                l = interval[0] + i * self.config.evec_len
                r = l + self.config.evec_len

                shifted_perm_indices = perm_indices + l
                # print(interval)
                # print(shifted_perm_indices)

                data.x[:, l:r] = data.x[:, shifted_perm_indices]
        
        return data


class ForcedOrderTransform:
    def __init__(self, config):
        self.config = config

    def __call__(self, data: Data) -> Data:
        # data.edge_index = edge_index_to_sparse_adj(data.edge_index, data.num_nodes)
        # embeddings = DataEmbeddings(self.config)
        # data = embeddings(data)
        adj = data.edge_index
        diff_op = get_diffusion(adj)
        for i in range(3):
            diff_op = diff_op @ diff_op

        perm_indices = torch.argsort(diff_op, dim=0)
        
        for interval in data.perms:

            n = interval[1]-interval[0]
            assert(n % self.config.evec_len == 0)

            for i in range(int(n / self.config.evec_len)):
                l = interval[0] + i * self.config.evec_len
                r = l + self.config.evec_len

                shifted_perm_indices = perm_indices + l
                print(interval)
                print(shifted_perm_indices)
                
                data.x[:, l:r] = data.x[shifted_perm_indices] # TODO: Check if this syntax is right, want to order each column by their respective ordering specified in the column of shifted_perm_indices
        
        return data



class CustomGraphDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None):
        super().__init__(root=None, transform=transform)
        self.data_list = data_list
        self.transform = transform

    def len(self):
        return len(self.data_list)

    def get(self, idx): # NOTE: Make sure these transforms actually work as intended 
        if self.transform == None:
            return self.data_list[idx]
        else:
            return self.transform(self.data_list[idx])


def load_data(config):

    # dataset and splits
    
    data_root = 'data'
    data_name = config.dataset
    data_path = os.path.join(data_root, data_name)
    
    if config.invariance_transform == "none":
        transform = None
    elif config.invariance_transform == "random":
        transform = RandomTransform(config)
    elif config.invariance_transform == "forcedorder":
        transform = ForcedOrderTransform(config)

    embeddings = DataEmbeddings(config)
    pre_transform = DataPreTransform(config)
    
    subset_frac = config.use_mini_dataset


    if subset_frac < 1 and os.path.exists(os.path.join(data_path, f"mini_dataset_{subset_frac}.pt")):
        print(f"Using {subset_frac} of dataset. Loading from previously saved subset")
        data_dict = torch.load(os.path.join(data_path, f"mini_dataset_{subset_frac}.pt"))
        print("data_dict loaded!")
    else:
        if config.dataset == 'zinc':
            if subset_frac >=  1:
                train_dataset = ZINC(root=data_root, transform=None, pre_transform=pre_transform) # no transforms here; add to transforms in the embedded CustomGraphDataset 
                val_dataset = ZINC(root=data_root, transform=None, pre_transform=pre_transform, split="val")
                test_dataset = ZINC(root=data_root, transform=None, pre_transform=pre_transform, split="test")
            else:
                train_dataset = ZINC(root=data_root, transform=None, pre_transform=pre_transform, subset=True)
                val_dataset = ZINC(root=data_root, transform=None, pre_transform=pre_transform, subset=True, split="val")
                test_dataset = ZINC(root=data_root, transform=None, pre_transform=pre_transform, subset=True, split="test")

            data_dict = {'train': train_dataset, 'valid': val_dataset, 'test': test_dataset}
            print('data object loaded!')

            # preprocessing embeddings

                

        elif config.dataset == 'ogbg_ppa':
            dataset = PygGraphPropPredDataset(root=data_root, name='ogbg-ppa', transform=None, pre_transform=pre_transform)
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
                if os.path.exists(os.path.join(data_path, f"mini_dataset_indices_{subset_frac}.pt")): # if the indices for this subset have already been generated
                    print(f"loading previously generated subset indices...")
                    idx_dict = torch.load(f"data/ogbg_ppa/mini_dataset_indices_{subset_frac}.pt") 
                    temp_train_idx = idx_dict['train']
                    temp_val_idx = idx_dict['valid']
                    temp_test_idx = idx_dict['test']
                else:
                    temp_train_idx = torch.tensor(sample_idx(split_idx['train'].tolist()))
                    temp_val_idx   = torch.tensor(sample_idx(split_idx['valid'].tolist()))
                    temp_test_idx  = torch.tensor(sample_idx(split_idx['test'].tolist()))
                    idx_dict = {'train': temp_train_idx, 'valid': temp_val_idx, 'test':temp_test_idx}
                    torch.save(idx_dict, os.path.join(data_path, f"mini_dataset_indices_{subset_frac}")) # Note: these indices are saved relative to the FULL dataset 
            

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

            print("ALL INDICES LENGTH:", all_indices.shape)
            subdataset = dataset[all_indices] # subdataset = dataset[all_indices] 
            data_dict = {'train': subdataset[train_idx], 'valid': subdataset[val_idx], 'test': subdataset[test_idx]}


      





    # preprocessing embeddings
    print("Processing embeddings...")

    need_emb = {'train': False, 'valid': False, 'test': False}
    data_dict_cache = {}
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
        cache_list = []
        modified_list = []
        embeddings = DataEmbeddings(config)

        for data in tqdm(data_dict[key]):
            if config.use_mini_dataset < 1 and config.dataset != "zinc":
                cache_list.append(data.clone())
            data = embeddings(data)
            modified_list.append(data)

        # 2) wrap them into a tiny InMemoryDataset
        # print("first element of cache_list:", cache_list[0])
        if config.use_mini_dataset < 1 and config.dataset != "zinc" :
            data_dict_cache[key] = CustomGraphDataset(cache_list)

        data_dict_emb[key] = CustomGraphDataset(modified_list, transform=transform)


    if config.use_mini_dataset < 1 and config.dataset != "zinc":
 
        torch.save(data_dict_cache, os.path.join(data_path, f"mini_dataset_{subset_frac}.pt"))
        print("Mini dataset saved")
    
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