import torch
import numpy as np
from sklearn.decomposition import PCA
from utils import *




class GlobalEmbeddings():
    def __init__(train_set, val_set, test_set):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.pca_adj = None
        self.pca_lap = None

    def __call__(self):
        PCA_on_train_adj(81)
        PCA_on_train_lap(81)
        train_emb = self.compute_embeddings(self.train_set)
        val_emb = self.compute_embeddings(self.val_set)
        test_emb = compute_embeddings(self.test_set)
        
        return train_emb, val_emb, test_emb


    def vec_adj(data: Data):
        N = data.edge_index.size(0)
        return torch.reshape(data.edge_index.to_dense(), (N**2, 1))

    def vec_lap(data: Data):
        N = data.edge_index.size(0)
        adj = data.edge_index.to_dense()
        lap = adj - torch.diag(torch.sum(adj,dim=1))

        return torch.reshape(lap, (N**2, 1))


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
    def compute_wavelets(data: Data, num_scales=3):
        adj = data.edge_index.to_dense()
        deg = torch.diag(torch.sum(adj, dim=1))

        P = adj @ torch.linalg.inv(deg)

        diff_scales = [P]

        for i in range(num_scales):
            diff_scales.append(diff_scales[-1] @ diff_scales[-1])

        wavelets = ()
    
        for i in range(num_scales):
            wavelets = wavelets + (diff_scales[i] - diff_scales[i + 1])
    
        return wavelets

    ##Compute Signals
    def degree_node_selection(adj: torch.sparse_coo_tensor, k, largest=True):
        """
        Returns the indicies of the k highest degree nodes
        """
        degrees = torch.sum(adj.to_dense(), dim = 0)
        _, indices = degrees.topk(k, largest=largest)
        return indices

    def compute_signals(indices, data: Data):
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
        for i in indicies:
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
        np_adg = get_adj_mat(self.train_set)
        pca = PCA(n_components=9)
        pca.fit(np_adj)
        self.pca_adj = pca


    def PCA_on_train_lap(self, num_components=9):
        np_lap = get_lap_mat(self.train_set)
        pca = PCA(n_components=9)
        pca.fit(np_lap)
        self.pca_lap  = pca

    def get_lap_mat(dataset):
        lap_vecs = ()
        for graph in dataset:
            lap_vecs += vec_lap(graph)
    
        lap_matrix = torch.cat(lap_vecs, dim=0)
        np_lap = np.array(lap_matrix)

        return np_lap

    def get_adj_mat(dataset):
        adj_vecs = ()
        for graph in dataset:
            adj_vecs += vec_adj(graph)
    
        adj_matrix = torch.cat(adj_vecs, dim=0)
        np_adg = np.array(adj_matrix)

        return np_adj

    def compute_embeddings(dataset):
        moments = [2, 3, 4]
        total_emb_size = len(moments) * 3 * 7 + len(moments) * 3 * 3
        embeddings = torch.zeros((dataset.shape[0], total_emb_size))
        for i in range(dataset.shape[0]):
            data = train_set[i]
            data.edge_index = edge_index_to_sparse_adj(data.edge_index, data.num_nodes)
            indices = degree_node_selection(data.edge_index, k=3)
            signals = compute_signals(data, indices)
            wavelets = compute_wavelets(data, num_scales=3)
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
        adj_emb = torch.tensor(self.pca_adj.transform(get_adj_mat(dataset)))
        lap_emb = torch.tensor(self.pca_lap.transform(get_lap_mat(dataset)))
        embeddings = torch.cat((embeddings, adj_emb, lap_emb), dim=1)
        return embeddings


    

    

                
