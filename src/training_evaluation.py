import torch
import torch.nn as nn
from torch_geometric.utils import scatter

import numpy as np
import time

from tqdm import tqdm

from utils import *

def SupervisedLoss(evecs_pred, evecs_gt):
    return torch.norm(evecs_pred - evecs_gt)


def EnergyLoss(eigvecs, adj, weights=None):
    # adj: SparseTensor in COO format on CUDA
    device = adj.device
    N = adj.size(0)
    # assert(False)
    # 1) sum to get a dense degree vector
    deg_vec = torch.sparse.sum(adj, dim=1).to_dense()      # [N]

    # 2) build sparse diagonal: indices = [[0,1,2,…],[0,1,2,…]]
    D = torch.diag(deg_vec)
    # 3) sparse-sparse subtraction (yields a sparse result)
    L = D - adj

    # 4) if you want energy in dense form, densify L first
    
    # L_dense = L.to_dense()
    

    num_eigenvectors = eigvecs.shape[-1]

    if weights == None:
        weights = torch.ones(num_eigenvectors).to(device)


    # build diagonal of weights
    diag_weights=torch.diag(weights)

    energy = torch.trace(
        eigvecs.transpose(-2, -1) @ L @ eigvecs @ diag_weights
    )


    # print(adj.size)
    # print("eigvecs", eigvecs.shape)
    # print("L", L_dense.shape)

    return energy

def OrthogonalityLoss(eigvecs):
    """
    eigvecs:  a [N × k] (or [..., N, k]) dense tensor on CUDA
    returns:  sum of all pairwise dot-products between distinct columns
    """
    # 1) Gram matrix G [k × k]
    G = eigvecs.transpose(-2, -1) @ eigvecs    # shape (..., k, k)
    # 2) Identity of size k on the same device & dtype
    k = G.size(-1)
    I = torch.eye(k, device=G.device, dtype=G.dtype)

    # 3) Zero out the diagonal by masking
    #    Off-diagonal = G * (1 - I)
    off_diag = G * (1.0 - I)

    # 4) Sum up the off-diagonals
    #    If you prefer a squared penalty, do off_diag.pow(2).sum()
    return torch.norm(off_diag)


def SupervisedEigenvalueLoss(eigvecs_pred, adj, eigvals_gt):
    lap = get_lap(adj)
    diag_eigvals = get_diag(eigvals_gt)
    return torch.norm(lap @ eigvecs_pred  - eigvecs_pred @ diag_eigvals)


def train(model, loader, optimizer, device, config):
    model.to(device)

    model.train()
    total_loss = 0
    total_energy_loss = 0
    total_ortho_loss = 0

    for data in tqdm(loader):
        data = data.to(device)
        # data.x = data.x.float()
        # data.edge_index = data.edge_index.float()
        # print(data.x.shape)
        # print(data.edge_index.shape)
        # print(data.num_nodes)
        out = model(data.x, data.edge_index)
        # print(out)
        if config.forced_ortho:
            out = orthogonalize_by_batch(out, data.batch)
        out = normalize_by_batch(out, data.batch)


        energy_loss = EnergyLoss(out, data.edge_index)
        ortho_loss = OrthogonalityLoss(out)

        if config.loss_function == 'energy':
            loss = config.lambda_energy * energy_loss + config.lambda_ortho * ortho_loss
        if config.loss_function == 'supervised_eigval':
            loss = SupervisedEigenvalueLoss(out, data.edge_index, data.eigvals)
        if config.loss_function == 'supervised_mse':
            loss = SupervisedLoss(out, data.eigvecs)
        elif config.loss_function == 'supervised_lap_reconstruction':
            return 


        # print("energy", energy_loss)
        # print("ortho", ortho_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        batch_size = int(data.batch.max()) + 1
        total_loss += loss.item()
        total_energy_loss += energy_loss.item()
        total_ortho_loss += ortho_loss.item()



    total_loss = total_loss / len(loader.dataset)
    total_energy_loss = total_energy_loss / len(loader.dataset)
    total_ortho_loss = total_ortho_loss / len(loader.dataset)


    return total_loss, total_energy_loss, total_ortho_loss

def validate(model, loader, optimizer, device, config):
    model.eval()

    total_loss = 0
    total_energy_loss = 0
    total_ortho_loss = 0
    for data in tqdm(loader):
        data = data.to(device)
        out = model(data.x, data.edge_index)
        if config.forced_ortho:
            out = orthogonalize_by_batch(out, data.batch)

        out = normalize_by_batch(out, data.batch)
        energy_loss = EnergyLoss(out, data.edge_index)
        ortho_loss = OrthogonalityLoss(out)
        loss = config.lambda_energy * energy_loss + config.lambda_ortho * ortho_loss

        if config.loss_function == 'energy':
            loss = config.lambda_energy * energy_loss + config.lambda_ortho * ortho_loss
        if config.loss_function == 'supervised_eigval':
            loss = SupervisedEigenvalueLoss(out, data.edge_index, data.eigvals)
        if config.loss_function == 'supervised_mse':
            loss = SupervisedLoss(out, data.eigvecs)
        elif config.loss_function == 'supervised_lap_reconstruction':
            return 

            
        batch_size = int(data.batch.max()) + 1
        total_loss += loss.item()
        total_energy_loss += energy_loss.item() 
        total_ortho_loss += ortho_loss.item() 

    total_loss = total_loss / len(loader.dataset)
    total_energy_loss = total_energy_loss / len(loader.dataset)
    total_ortho_loss = total_ortho_loss / len(loader.dataset)
    
    return total_loss, total_energy_loss, total_ortho_loss


def training_loop(model, train_loader, val_loader, optimizer, device, config):




    # validation_acc = []
    validation_loss = []


    best_val_loss = float('inf')

    if config.use_early_stopping:
        last_min_val_loss = float('inf')
        patience_counter = 0
        stopped_early = False

    t_start = time.time()
    for epoch in range(config.epochs):

        optimizer.zero_grad()
        
        train_loss, train_loss_energy, train_loss_ortho = train(model, train_loader, optimizer, device, config)

        with torch.no_grad():
            
            val_loss, val_loss_energy, val_loss_ortho = validate(model, val_loader, optimizer, device, config)

            validation_loss.append(val_loss)


            if val_loss < best_val_loss:
                torch.save(model, f"{config.checkpoint_folder}/{epoch}.pt")
                best_val_loss = val_loss

            if config.use_early_stopping:
                if val_loss < last_min_val_loss:
                    last_min_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter == config.patience:
                        stopped_early = True
                        t_end = time.time()

        if not config.multiple_runs:
            print(" | ".join([f"Epoch: {epoch:4d}", f"Train loss: {train_loss:.3f}", f"Train loss (energy): {train_loss_energy:.3f}", f"Train loss (ortho): {train_loss_ortho:.3f}",
                              f"Val loss: {val_loss:.3f}",  f"Val loss (energy): {val_loss_energy:.3f}",  f"Val loss (ortho): {val_loss_ortho:.3f}"
                             ]))
                             

        if config.use_early_stopping and stopped_early:
            break

    if (not config.multiple_runs) and config.use_early_stopping and stopped_early:
        print(f"EARLY STOPPING condition met. Stopped at epoch: {epoch}.")
    else:
        t_end = time.time()

    if not config.multiple_runs:
        print(f"Total training time: {t_end-t_start:.2f} seconds")



    return validation_loss


def evaluate_on_test(model, loader, device, config): # low frequency eigval eigvec decomposition 
    

    with torch.no_grad():
        model.eval()
        y_pred = model(features, adj)
        test_loss = criterion(y_pred[test_ind], labels[test_ind])
        test_acc = accuracy(y_pred[test_ind], labels[test_ind])

def mse_test_loss(model, loader, device):
    model.to(device)
    model.eval()
    loss = 0
    for data in loader:
        print(data.batch)
        evecs_pred = model(data.x, data.edge_index, data.batch)
        for i in range(data.num_graphs):
            graph = data.get_example(i)
            inds = torch.argwhere(data.batch == i)
            _, evecs_gt = torch.linalg.eigh(graph.edge_index)
            loss += torch.norm(evecs_pred[inds] - evecs_gt[:, :evecs_pred.shape[1]])
    return loss

def lap_reconstruction_loss(model, loader, adj, device): # REWRITE SO IT JUST TAKES ONE DATA OBJECT (the loading step should happen in a wrapper outside )
    model.to(device)
    model.eval()
    # adj: SparseTensor in COO format on CUDA
    device = adj.device
    N = adj.size(0)
    # assert(False)
    # 1) sum to get a dense degree vector
    deg_vec = torch.sparse.sum(adj, dim=1).to_dense()      # [N]

    # 2) build sparse diagonal: indices = [[0,1,2,…],[0,1,2,…]]
    idx = torch.arange(N, device=device)
    indices = torch.stack([idx, idx], dim=0)               # [2×N]
    values  = deg_vec                                    # [N]

    D = torch.sparse_coo_tensor(indices, values, (N, N),
                                device=device)

    # 3) sparse-sparse subtraction (yields a sparse result)
    L = D - adj

    loss = 0
    for data in loader:
        print(data.batch)
        evecs_pred = model(data.x, data.edge_index, data.batch)
        for i in range(data.num_graphs):
            graph = data.get_example(i)
            inds = torch.argwhere(data.batch == i)

            U_pred = evecs_pred[inds]
            lap = L[inds, inds]

            lambda_gt, evecs_gt = torch.linalg.eigh(graph.edge_index)
            lambda_pred = U_pred @ lap @ U_pred.T

            U = evecs_gt[:, evecs_pred.shape[1]]
            lambda_gt = lambda_gt[:evecs_pred.shape[1]]

            low_rank_pred = U_pred @ torch.diag(lambda_pred) @ U_pred.T
            low_rank_gt = U @ torch.diag(lambda_gt) @ U.T

            loss += torch.norm(low_rank_pred - low_rank_gt)
    return loss

def eigenvalue_loss(model, loader, adj, device):
    model.to(device)
    model.eval()
    # adj: SparseTensor in COO format on CUDA
    device = adj.device
    N = adj.size(0)
    # assert(False)
    # 1) sum to get a dense degree vector
    deg_vec = torch.sparse.sum(adj, dim=1).to_dense()      # [N]

    # 2) build sparse diagonal: indices = [[0,1,2,…],[0,1,2,…]]
    idx = torch.arange(N, device=device)
    indices = torch.stack([idx, idx], dim=0)               # [2×N]
    values  = deg_vec                                    # [N]

    D = torch.sparse_coo_tensor(indices, values, (N, N),
                                device=device)

    # 3) sparse-sparse subtraction (yields a sparse result)
    L = D - adj

    loss = 0
    for data in loader:
        print(data.batch)
        evecs_pred = model(data.x, data.edge_index, data.batch)
        for i in range(data.num_graphs):
            graph = data.get_example(i)
            inds = torch.argwhere(data.batch == i)

            U_pred = evecs_pred[inds]
            lap = L[inds, inds]

            lambda_gt, evecs_gt = torch.linalg.eigh(graph.edge_index)
            lambda_pred = U_pred @ lap @ U_pred.T

            U = evecs_gt[:, evecs_pred.shape[1]]
            lambda_gt = lambda_gt[:evecs_pred.shape[1]]

            loss += torch.norm(U_pred @ U_pred.T @ lap @ U_pred @ U_pred.T - low_rank_gt)
    return loss

    


def multiple_runs(model, features, labels, adj, indices, config, training_loop, evaluate_on_test):
    train_set_ind, val_set_ind, test_set_ind = indices
    acc = []
    loss = []

    t1 = time.time()
    for i in range(config.num_of_runs):
        print("Run:", i+1)
        model.initialize_weights()
        training_loop(model, features, labels, adj,
                      train_set_ind, val_set_ind, config)

        acc_curr, loss_curr = evaluate_on_test(model, features, labels,
                                               adj, test_set_ind, config)
        acc.append(acc_curr)
        loss.append(loss_curr)

    print(f"ACC:  mean: {np.mean(acc):.2f} | std: {np.std(acc):.2f}")
    print(f"LOSS: mean: {np.mean(loss):.2f} | std: {np.std(loss):.2f}")
    print(f"Total training time: {time.time()-t1:.2f} seconds")
