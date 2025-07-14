import torch
import numpy as np
from sklearn.decomposition import PCA
from utils import *
from torch_geometric.data import Data
import torch.nn.functional as F
from embeddings import *

from torch.utils.data import TensorDataset




class GlobalEmbeddings():
    def __init__(self, train_set, val_set, test_set, config):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.pca_adj = None
        self.pca_lap = None
        
        self.config = config

    def __call__(self):
        self.PCA_on_train_adj(81)
        self.PCA_on_train_lap(81)
        train_emb = self.compute_embeddings(self.train_set)
        val_emb = self.compute_embeddings(self.val_set)
        test_emb = self.compute_embeddings(self.test_set)

        train_adj = self.compute_adj_tensor(self.train_set)
        val_adj = self.compute_adj_tensor(self.val_set)      
        test_adj = self.compute_adj_tensor(self.test_set)

        train_eval = self.compute_eigval_tensor(self.train_set)
        val_eval = self.compute_eigval_tensor(self.val_set)      
        test_eval = self.compute_eigval_tensor(self.test_set)  

        train_evec = self.compute_eigvec_tensor(self.train_set)
        val_evec = self.compute_eigvec_tensor(self.val_set)      
        test_evec = self.compute_eigvec_tensor(self.test_set)                  

        train_dataset = TensorDataset(train_emb, train_adj, train_eval, train_evec)
        val_dataset =  TensorDataset(val_emb, val_adj, val_eval, val_evec)
        test_dataset =  TensorDataset(test_emb, test_adj, test_eval, test_evec)
        return train_dataset, val_dataset, test_dataset


    def vec_adj(self, data: Data):
        N = data.edge_index.size(0)
        vec = torch.reshape(data.edge_index.to_dense(), (1, N**2))
        pad = (0, 40**2-N**2)
        #print(f"Vec Shape: ", vec.shape)
        pad_vec = F.pad(vec, pad)
        #print(pad_vec.shape)
        return pad_vec


    def vec_lap(self, data: Data):
        N = data.edge_index.size(0)
        adj = data.edge_index.to_dense()
        lap = adj - torch.diag(torch.sum(adj,dim=1))

        vec = torch.reshape(lap, (1, N**2))
        pad = (0, 40**2-N**2)
        pad_vec = F.pad(vec, pad)
        return pad_vec


    ##Compute Wavelet and Scattering Moments
    def scattering_moments(signal: torch.Tensor, wavelet_path: tuple, moment: int):
        val = 0
        for i in range(signal.shape[0]):
            U = signal
            for wavelet in wavelet_path:
                U = torch.abs(wavelet @ U)
            val += U[i]**moment
        return val

    def wavelet_moments(wavelet: torch.Tensor, moment: int, signal: torch.Tensor):
        return torch.sum((wavelet**moment) @ signal)

    ##Compute Wavelets
    def compute_wavelets(self, data: Data, num_scales=3):
        adj = data.edge_index.to_dense()
        deg = torch.diag(torch.sum(adj, dim=1))

        P = adj @ torch.linalg.inv(deg)

        diff_scales = [P]

        for i in range(num_scales):
            diff_scales.append(diff_scales[-1] @ diff_scales[-1])

        wavelets = ()
    
        for i in range(num_scales):
            wavelets = wavelets + (diff_scales[i] - diff_scales[i + 1],)
    
        return wavelets

    ##Compute Signals

    def compute_signals(self, indices, data: Data):
        N = data.edge_index.size(0)
        ##Step 1: Compute three different scales of diffusion
        adj = data.edge_index.to_dense()
        deg = torch.diag(torch.sum(adj, dim=1))

        P = adj @ torch.linalg.inv(deg)

        diff_scales = [P]

        for i in range(2):
            diff_scales.append(diff_scales[-1] @ diff_scales[-1])

        ##Step 2: Start a dirac at each index
        diracs = []
        for i in indices:
            dirac = torch.zeros(N)
            dirac[i] += 1

        ##Step 3: Multiply each diffusion scale by each dirac
        signals = ()
        for dirac in diracs:
            for diff in diff_scales:
                signal = diff @ dirac
                signals = signals + signal

        ###Step 4: Return the signals
        return signals

    def PCA_on_train_adj(self, num_components=9):
        np_adj = self.get_adj_mat(self.train_set)
        pca = PCA(n_components=num_components)
        pca.fit(np_adj)
        self.pca_adj = pca


    def PCA_on_train_lap(self, num_components=9):
        np_lap = self.get_lap_mat(self.train_set)
        pca = PCA(n_components=num_components)
        pca.fit(np_lap)
        self.pca_lap  = pca

    def get_lap_mat(self, dataset):
        lap_vecs = ()
        for graph in dataset:
            lap_vecs += (self.vec_lap(graph),)
    
        lap_matrix = torch.cat(lap_vecs, dim=0)
        np_lap = np.array(lap_matrix)

        return np_lap

    def get_adj_mat(self, dataset):
        adj_vecs = ()
        for graph in dataset:
            adj_vecs += (self.vec_adj(graph),)
    
        adj_matrix = torch.cat(adj_vecs, dim=0)
        np_adj = np.array(adj_matrix)

        return np_adj

    def compute_embeddings(self, dataset):
        moments = [2, 3, 4]
        total_emb_size = len(moments) * 3 * 7 + len(moments) * 3 * 3
        embeddings = torch.zeros((dataset.len(), total_emb_size))
        node_count = torch.zeros(dataset.len(), 1)
        for i in range(dataset.len()):
            data = dataset[i]
            node_count[i] += data.num_nodes
            indices = degree_node_selection(data.edge_index, k=3)
            signals = self.compute_signals(indices, data)
            wavelets = self.compute_wavelets(data, num_scales=3)
            paths = all_index_combinations(3)
            j = 0
            for signal in signals:
                for moment in moments:
                    for wavelet in wavelets:
                        wave_emb = wavelet_moments(wavelet, moment, signal)
                        embeddings[i, j] += wave_emb
                        j += 1
                    for path in paths:
                        wavelet_path = wavelets[path]
                        scatt_emb = scattering_moments(signal, wavelet_path, moment)
                        emebeddings[i, j] += scatt_emb
                        j += 1
        adj_emb = torch.tensor(self.pca_adj.transform(self.get_adj_mat(dataset)))
        lap_emb = torch.tensor(self.pca_lap.transform(self.get_lap_mat(dataset)))
        embeddings = torch.cat((node_count, embeddings, adj_emb, lap_emb), dim=-1)
        print(embeddings.shape)
        return embeddings

    def compute_adj_tensor(self, dataset):
        target = torch.zeros((dataset.len(), self.config.evec_len, self.config.evec_len))
        for i in range(dataset.len()):
            data = dataset[i]
            adj = data.edge_index
            N = data.num_nodes
            padding_0 = 40-N

            padding = (0, padding_0, 0, padding_0)

            pad_adj = F.pad(adj.to_dense(), padding)

            target[i] = pad_adj
        return target

    def compute_eigval_tensor(self, dataset):
        target = torch.zeros((dataset.len(), self.config.num_eigenvectors, self.config.num_eigenvectors))
        for i in range(dataset.len()):
            data = dataset[i]
             L = torch.diag(data.eigvals)
            target[i] = L

        return target

    def compute_eigvec_tensor(self, dataset):
        target = torch.zeros((dataset.len(), self.config.eveclen, self.config.num_eigenvectors))
        for i in range(dataset.len()):
            data = dataset[i]
             U = data.eigvecs[:, :config.num_eigenvectors]
            target[i] = U

        return target





    

    

                
