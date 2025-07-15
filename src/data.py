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

import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import InMemoryDataset, Data
import os

from sklearn.model_selection import train_test_split
import numpy as np

import time
import scipy.sparse as sp

from torch_geometric.transforms import LargestConnectedComponents

def smiles_to_data(smiles):
    mol = Chem.MolFromSmiles(smiles)

    edge_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_list.append((i, j))
        edge_list.append((j, i))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data = Data(edge_index=edge_index)

    return data


class DrugBankDataset(InMemoryDataset):
    def __init__(self, root, csv_path, transform=None, pre_transform=None):
        self.csv_path = csv_path
        super(DrugBankDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        print(os.path.basename("drugbank.csv"))
        return [os.path.basename(self.csv_path)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.csv_path)
        data_list = []

        for _, row in df.iterrows():
            smiles = row['SMILES']
            data = smiles_to_data(smiles)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(dataset, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):

        num_graphs = len(dataset)
        indices = np.arange(num_graphs)

        train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=seed, shuffle=True)

        valid_size = valid_ratio / (valid_ratio + test_ratio)
        valid_idx, test_idx = train_test_split(temp_idx, train_size=valid_size, random_state=seed, shuffle=True)

        return {
            'train': train_idx,
            'valid': valid_idx,
            'test':  test_idx
        }

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

        total_nodes_before = data.num_nodes

        if self.config.use_largest_connected_components: # taken from "LargestConnectedComponents" in pytorch geometric
            data = LargestConnectedComponents(1)(data)
            
            if data.num_nodes != total_nodes_before:
                print(f"Taking largest connected component: {data.num_nodes} out of {total_nodes_before}")
            else:
                print(f"Graph already connected: {data.num_nodes}")


        data.edge_index = edge_index_to_sparse_adj(data.edge_index, data.num_nodes)

        if self.config.use_supervised:
            evals, evecs = get_padded_eigvecs(data.edge_index, self.config.evec_len)
            data.eigvecs = evecs 
            data.eigvals = evals

        return data



class DataEmbeddings:
    def __init__(self, config):
        self.config = config

    def __call__(self, data: Data) -> Data:




        data.perms = []
        data.x = torch.ones(data.num_nodes, 0, dtype=torch.float32)
        data.temp = torch.ones(data.num_nodes, 0, dtype=torch.float32)

        data.emb_runtimes = {}

        if self.config.diffusion_emb: # PERM INVARIANCE REQUIRED 
            t1 = time.time()
            l = data.x.shape[-1]
            data.x = torch.cat((data.x, diffusion_emb(data)), dim=-1)
            r = data.x.shape[-1]
            data.perms.append([l, r])
            data.emb_runtimes['diffusion_emb'] = time.time() - t1

        if self.config.wavelet_emb:
            t1 = time.time()
            data.x = torch.cat((data.x, wavelet_emb(data)), dim=-1)
            data.emb_runtimes['wavelet_emb'] = time.time() - t1
        
        if self.config.wavelet_positional_emb:
            t1 = time.time()
            data.x = torch.cat((data.x, wavelet_positional_emb(data)), dim=-1)
            data.emb_runtimes['wavelet_positional_emb'] = time.time() - t1

        if self.config.scatter_emb:
            t1 = time.time()
            data.x = torch.cat((data.x, scatter_emb(data)), dim=-1)
            data.emb_runtimes['scatter_emb'] = time.time() - t1

        if self.config.global_scatter_emb:
            t1 = time.time()
            data.x = torch.cat((data.x, global_scatter_emb(data)), dim=-1)
            data.emb_runtimes['global_scatter_emb'] = time.time() - t1
        
        if self.config.wavelet_moments_emb:
            t1 = time.time()
            data.x = torch.cat((data.x, wavelet_moments_emb(data)), dim=-1)
            data.emb_runtimes['wavelet_moments_emb'] = time.time() - t1

        if self.config.neighbor_bump_emb:
            t1 = time.time()
            data.x = torch.cat((data.x, neighbors_signal_emb(data)), dim=-1)
            data.emb_runtimes['neighbor_bump_emb'] = time.time() - t1

        if self.config.diffused_dirac_emb:
            t1 = time.time()
            data.x = torch.cat((data.x, diffused_dirac_emb(data)), dim=-1)
            data.emb_runtimes['diffused_dirac_emb'] = time.time() - t1
        
        
        t1 = time.time()
        # data.x = torch.cat((data.x, diffused_dirac_emb(data)), dim=-1)
        L, Q = torch.linalg.eigh(get_lap(data.edge_index)) # TEMPORARY, JUST FOR RUNTIME TESTING

        data.x = torch.cat((data.x, L.unsqueeze(dim=-1)), dim=-1) # include concatenate operation for fair runtime comparison 
        data.emb_runtimes['eigh'] = time.time() - t1

        # data.temp = torch.ones(data.num_nodes, 0, dtype=torch.float32) # clear out 


        

        # t1 = time.time()
        # # data.x = torch.cat((data.x, diffused_dirac_emb(data)), dim=-1)
        # L, Q = torch.linalg.eig(get_lap(data.edge_index)) # TEMPORARY, JUST FOR RUNTIME TESTING
        # data.emb_runtimes['eig'] = time.time() - t1

        # t1 = time.time()
        # if data.num_nodes >= 3 * self.config.num_eigenvectors:
        #     lobpcg = torch.lobpcg(get_lap(data.edge_index), k=self.config.num_eigenvectors, method="ortho")
        # else: 
        #     print("not enough nodes to compute lobpcg")
        # data.emb_runtimes['lobpcg'] = time.time() - t1

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
        diff_op = 0.5 * get_diffusion(adj) + 0.5 * torch.eye(adj.shape[0]) # lazy random walk
        for i in range(4):
            diff_op = diff_op @ diff_op
 
        perm_indices = torch.argsort(diff_op, dim=0).T # sort each node's embeddings by its column of diffusion operator
        perm_indices = torch.cat((perm_indices, torch.arange(data.num_nodes, self.config.evec_len).repeat(data.num_nodes, 1)), dim=-1) # for the padded indices
        for interval in data.perms:

            n = interval[1]-interval[0]
            assert(n % self.config.evec_len == 0)

            for i in range(int(n / self.config.evec_len)):
                l = interval[0] + i * self.config.evec_len
                r = l + self.config.evec_len
                # shifted_perm_indices = perm_indices + l
                # print(interval)
                # print(shifted_perm_indices)
                
                data.x[:, l:r] = torch.gather(data.x[:, l:r], 1, perm_indices) 
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

    print(data_path)
    
    if config.invariance_transform == "none":
        transform = None
    elif config.invariance_transform == "random":
        transform = RandomTransform(config)
    elif config.invariance_transform == "forced_order":
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

                

        elif config.dataset == 'ogbg_ppa' or config.dataset == 'drugbank':
            if config.dataset == 'ogbg_ppa':
                dataset = PygGraphPropPredDataset(root=data_root, name='ogbg-ppa', transform=None, pre_transform=pre_transform)
            else:
                dataset = DrugBankDataset(root=data_root, csv_path="/vast/palmer/pi/krishnaswamy_smita/nsn27/SUMRY/EigvecGCN/src/drugbank.csv", transform=None, pre_transform=pre_transform)
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

                print("ALL INDICES LENGTH:", all_indices.shape)
                subdataset = dataset[all_indices] # subdataset = dataset[all_indices] 
                data_dict = {'train': subdataset[train_idx], 'valid': subdataset[val_idx], 'test': subdataset[test_idx]}

                print("Saving mini dataset to cache...")
                data_dict_cache = {}
                for key in data_dict:
                    cache_list = []
                    for data in tqdm(data_dict[key]):
                        
                        cache_list.append(data.clone())
                
                    data_dict_cache[key] = CustomGraphDataset(cache_list)

                torch.save(data_dict_cache, os.path.join(data_path, f"mini_dataset_{subset_frac}.pt"))
                print(f"Mini dataset saved: {os.path.join(data_path, f"mini_dataset_{subset_frac}.pt")}")


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
    
    data_dict_emb = {} 


    if config.train:
        need_emb['train'] = True
        need_emb['valid'] = True

    if config.test:
        need_emb['valid'] = True
        need_emb['test'] = True # TODO: change to just test_loader when ready for final analysis


    for key in data_dict:
        if not need_emb[key]:
            print(f"No embeddings needed for {key}")
            continue
        
        print(f"Processing embeddings for {key}")
    
        modified_list = []
        embeddings = DataEmbeddings(config)
        runtimes_dict = {}
        for idx, data in enumerate(tqdm(data_dict[key])):
            

            
            
            data = embeddings(data)
            modified_list.append(data)

            if idx == 0:
                runtimes_dict = data.emb_runtimes
            else:
                for emb_key in runtimes_dict:
                    runtimes_dict[emb_key] += data.emb_runtimes[emb_key]

        print(runtimes_dict)

        # 2) wrap them into a tiny InMemoryDataset
        # print("first element of cache_list:", cache_list[0])

        data_dict_emb[key] = CustomGraphDataset(modified_list, transform=transform)

    

 
        
    
    print("Embeddings processed!")

    if 'train' in data_dict_emb.keys():
        train_loader = DataLoader(data_dict_emb['train'], batch_size=32, shuffle=True) # ISSUE: right now this is just concatenating everything in the batch, treating it lke a huge graph
    else:
        train_loader = None
    if 'valid' in data_dict_emb.keys():
        val_loader   = DataLoader(data_dict_emb['valid'], batch_size=64, shuffle=False)
    else:
        val_loader = None
    if 'valid' in data_dict_emb.keys():
        test_loader  = DataLoader(data_dict_emb['valid'],  batch_size=1, shuffle=False) # TODO: CHANGE BACK TO TEST WHEN READY 
    else:
        test_loader = None

    
    


    return data_dict_emb, train_loader, val_loader, test_loader