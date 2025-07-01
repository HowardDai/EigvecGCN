import torch
import torch.nn as nn
from torch_geometric.utils import scatter

import numpy as np
import time 

from tqdm import tqdm

from utils import *

import matplotlib.pyplot as plt
from datetime import datetime, date

import random

def SupervisedLoss(evecs_pred, evecs_gt, batch):


    eigval_inds = torch.arange(300, dtype=torch.long, device=adj.device)
    total_mse = 0

    for i in range(batch[-1] + 1):
        inds = list(torch.argwhere(batch == i).squeeze())

        evecs_pred = eigvecs_pred[inds, :]

        total_mse += torch.norm(evecs_pred - evecs_gt[eigval_inds, :][:evecs_pred.shape[0]]) ** 2

        eigval_inds = eigval_inds + 300
    
    return total_mse


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


# def SupervisedEigenvalueLoss(eigvecs_pred, adj, eigvals_gt):
#     lap = get_lap(adj)
#     diag_eigvals = torch.diag(eigvals_gt)
#     print(diag_eigvals.shape)
#     return torch.norm(lap @ eigvecs_pred  - eigvecs_pred @ diag_eigvals) # TODO: need to fix. Basically needthe column weighting to happen PER graph and PER set of eigvals


def SupervisedEigenvalueLoss(eigvecs_pred, adj, eigvals_gt, batch):
    L = get_lap(adj)
    loss = 0
    num_eigvals = eigvecs_pred.shape[-1]

    eigval_inds = torch.arange(num_eigvals, dtype=torch.long, device=adj.device)
    for i in range(batch[-1] + 1):
        inds = list(torch.argwhere(batch == i).squeeze())

        lap = L[inds][:, inds]
        
        evecs_pred = eigvecs_pred[inds, :]
        diag_eigvals = torch.diag(eigvals_gt[eigval_inds])
        loss += torch.norm(lap @ evecs_pred  - evecs_pred @ diag_eigvals)

        eigval_inds = eigval_inds + 30
    return loss

def SupervisedEigenvalueLossUnweighted(eigvecs_pred, adj, eigvals_gt, batch):
    L = get_lap(adj)
    loss = 0
    num_eigvals = eigvecs_pred.shape[-1]

    eigval_inds = torch.arange(num_eigvals, dtype=torch.long, device=adj.device)

    for i in range(batch[-1] + 1):
        inds = list(torch.argwhere(batch == i).squeeze())

        lap = L[inds][:, inds]
        eigvals_gt_inv = torch.reciprocal(eigvals_gt)
        evecs_pred = eigvecs_pred[inds, :]
        diag_eigvals_inv = torch.diag(eigvals_gt_inv[eigval_inds])

        loss += torch.norm(lap @ evecs_pred @ diag_eigvals_inv - evecs_pred)

        eigval_inds = eigval_inds + 30
    return loss

def compute_loss(config, out, data):
    if config.forced_ortho:
        out = orthogonalize_by_batch(out, data.batch)
    out = normalize_by_batch(out, data.batch)

    if config.loss_function == 'energy':
        loss = config.lambda_energy * EnergyLoss(out, data.edge_index) 
    if config.loss_function == 'supervised_eigval':
        loss = SupervisedEigenvalueLoss(out, data.edge_index, data.eigvals, data.batch)
    if config.loss_function == 'supervised_eigval_unweighted':
        loss = SupervisedEigenvalueLossUnweighted(out, data.edge_index, data.eigvals, data.batch)
    if config.loss_function == 'supervised_mse':
            loss = SupervisedLoss(out, data.eigvecs[:, :config.num_eigenvectors])
    if config.loss_function == 'supervised_lap_reconstruction':
        loss = lap_reconstruction_loss(out, data.eigvals, data.eigvecs[:, :config.num_eigenvectors], data.edge_index, data.batch)

        
    ortho_loss = config.lambda_ortho * OrthogonalityLoss(out)


    loss += ortho_loss 

    return loss, ortho_loss


def train(model, loader, optimizer, device, config):
    model.to(device)

    model.train()
    total_loss = 0
    total_ortho_loss = 0

    for data in tqdm(loader):
        data = data.to(device)
        out = model(data.x, data.edge_index)

        loss, ortho_loss = compute_loss(config, out, data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        batch_size = int(data.batch.max()) + 1
        total_loss += loss.item()
        total_ortho_loss += ortho_loss.item()

    total_loss = total_loss / len(loader.dataset)
    total_ortho_loss = total_ortho_loss / len(loader.dataset)


    return total_loss, total_ortho_loss

def validate(model, loader, optimizer, device, config):
    model.eval()

    total_loss = 0
    total_ortho_loss = 0
    for data in tqdm(loader):
        data = data.to(device)
        out = model(data.x, data.edge_index)
        if config.forced_ortho:
            out = orthogonalize_by_batch(out, data.batch)

        out = normalize_by_batch(out, data.batch)


        if config.loss_function == 'energy':
            loss = config.lambda_energy * EnergyLoss(out, data.edge_index)
        if config.loss_function == 'supervised_eigval':
            loss = SupervisedEigenvalueLoss(out, data.edge_index, data.eigvals, data.batch)
        if config.loss_function == 'supervised_eigval_unweighted':
            loss = SupervisedEigenvalueLossUnweighted(out, data.edge_index, data.eigvals, data.batch)
        if config.loss_function == 'supervised_mse':
            loss = SupervisedLoss(out, data.eigvecs[:, :config.num_eigenvectors])
        if config.loss_function == 'supervised_lap_reconstruction':
            loss = lap_reconstruction_loss(out, data.eigvals, data.eigvecs[:, :config.num_eigenvectors], data.edge_index, data.batch)

        ortho_loss = config.lambda_ortho * OrthogonalityLoss(out)


        loss += ortho_loss 
            
        batch_size = int(data.batch.max()) + 1
        total_loss += loss.item()
        total_ortho_loss += ortho_loss.item() 

    total_loss = total_loss / len(loader.dataset)
    total_ortho_loss = total_ortho_loss / len(loader.dataset)
    
    return total_loss, total_ortho_loss


def training_loop(model, train_loader, val_loader, optimizer, device, config):

    # validation_acc = []
    validation_loss_hist = []
    train_loss_hist = []


    best_val_loss = float('inf')

    if config.use_early_stopping:
        last_min_val_loss = float('inf')
        patience_counter = 0
        stopped_early = False

    t_start = time.time()
    for epoch in range(config.epochs):

        optimizer.zero_grad()
        
        train_loss, train_loss_ortho = train(model, train_loader, optimizer, device, config)

        train_loss_hist.append(train_loss)

        with torch.no_grad():
            
            val_loss, val_loss_ortho = validate(model, val_loader, optimizer, device, config)

            validation_loss_hist.append(val_loss)


            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f"{config.checkpoint_folder}/{epoch}.pt")
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
            print(" | ".join([f"Epoch: {epoch:4d}", f"Train loss: {train_loss:.3f}", f"Train loss (ortho): {train_loss_ortho:.3f}",
                              f"Val loss: {val_loss:.3f}",  f"Val loss (ortho): {val_loss_ortho:.3f}"
                             ]))
                             

        if config.use_early_stopping and stopped_early:
            break

    if (not config.multiple_runs) and config.use_early_stopping and stopped_early:
        print(f"EARLY STOPPING condition met. Stopped at epoch: {epoch}.")
    else:
        t_end = time.time()

    if not config.multiple_runs:
        print(f"Total training time: {t_end-t_start:.2f} seconds")

    plot_results(config, validation_loss_hist, train_loss_hist, model, device, val_loader)

    return validation_loss_hist

def plot_results(config, validation_loss_hist, train_loss_hist, model, device, val_loader):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, config.epochs + 1), validation_loss_hist, color='tab:blue', label='validation')
    ax.plot(range(1, config.epochs + 1), train_loss_hist, color='tab:orange', label='training')
    ax.set(xlabel='Epoch', ylabel='Loss', title=f"{config.loss_function} Loss History")
    ax.legend()
    fig.savefig(f"plots/{config.model}_{config.loss_function}_{date.today()}/loss_plots.png")

    sample_data = None
    for data in val_loader:  
        sample_data = data
        break

    inds = torch.argwhere(sample_data.batch == 0).tolist()
    model.to(device)
    sample_data.to(device)
    evecs_pred = model(sample_data.x, data.edge_index)
    evecs_pred = normalize_by_batch(evecs_pred, sample_data.batch)[:len(inds), :]

    evecs_gt = sample_data.eigvecs[:len(inds), :]
    sort_inds = torch.argsort(evecs_gt[:, 1]).tolist()
    evecs_gt = evecs_gt[sort_inds]

    evecs_pred = evecs_pred[sort_inds, :]


    fig_i, axs = plt.subplots(15, 2, figsize=(20, 45), sharex=True)
    k = 0
    for i in range(15):
        for j in [0, 1]:
            ax = axs[i, j]
            ax.plot(evecs_pred[:, k].cpu().detach().numpy(), color='tab:blue', label='prediction')
            ax.plot(evecs_gt[:, k].cpu().detach().numpy(), color='tab:orange', label='ground truth')
            if i == 14:
                ax.set(xlabel='Vertex')
            if j == 0:
                ax.set(ylabel='Eigenfunction')
            ax.set(title=f"Phi {k + 1}")
            k += 1
        handles, labels = ax.get_legend_handles_labels()
    fig_i.suptitle("First 30 Eigenvectors")
    fig_i.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95))
    fig_i.savefig(f"plots/{config.model}_{config.loss_function}_{date.today()}/eigvecs.png")



def evaluate(model, loader, optimizer, device, config):
    model.eval()

    total_loss = 0
    total_ortho_loss = 0
    loss_dict = {'energy': 0, 'supervised_eigval': 0, 'supervised_eigval_unweighted': 0, 'supervised_lap_reconstruction': 0, 'ortho': 0}
    
    total_runtime = 0


    total_eigval_sum = 0
    total_num_eigvecs = 0

    num_eigvecs = config.num_eigenvectors
    

    for data in tqdm(loader):
        data = data.to(device)
        t1 = time.time()
        out = model(data.x, data.edge_index)

        if config.forced_ortho:
            out = orthogonalize_by_batch(out, data.batch)

        out = normalize_by_batch(out, data.batch)
        total_runtime += time.time() - t1




        for loss_function in loss_dict:
            if loss_function == 'energy':
                loss = EnergyLoss(out, data.edge_index)
            if loss_function == 'supervised_eigval':
                loss = SupervisedEigenvalueLoss(out, data.edge_index, data.eigvals, data.batch)
            if loss_function == 'supervised_eigval_unweighted':
                loss = SupervisedEigenvalueLossUnweighted(out, data.edge_index, data.eigvals, data.batch)
            if loss_function == 'supervised_mse':
                loss = SupervisedLoss(out, data.eigvecs[:, :config.num_eigenvectors], data.batch)
            if loss_function == 'supervised_lap_reconstruction':
                loss = lap_reconstruction_loss(out, data.eigvals, data.eigvecs[:, :config.num_eigenvectors], data.edge_index, data.batch)
            if loss_function == 'ortho':
                loss = OrthogonalityLoss(out)
            loss_dict[loss_function] += loss.item()

        total_eigval_sum += torch.sum(data.eigvals)
        total_num_eigvecs += num_eigvecs * (data.batch[-1]+1)

    print("k: ", num_eigvecs)
    print("average eigval: ", (total_eigval_sum / total_num_eigvecs).item() )
    print("Size of test set: ", (len(loader.dataset)))
    for loss_function in loss_dict:
        loss_dict[loss_function] =  loss_dict[loss_function] / len(loader.dataset)
        print(loss_function, ": ", loss_dict[loss_function])
    
    
    avg_runtime = total_runtime / len(loader.dataset)
    print(avg_runtime)
    out_dict = loss_dict
    out_dict['runtime'] = avg_runtime

    return out_dict
    

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

def lap_reconstruction_loss(eigvecs_pred, lambda_gt, evecs_gt, adj, batch):
    
    L = get_lap(adj)
    loss = 0
    num_eigvals = eigvecs_pred.shape[-1]
    eigval_inds = torch.arange(num_eigvals, dtype=torch.long, device=adj.device)
    eigvec_start = 0
    evecs_gt_new = torch.clone(evecs_gt)

    inds_all = [torch.argwhere(batch == i).squeeze().tolist() for i in range(batch[-1] + 1)]


    for i in range(batch[-1] + 1):
        inds  = inds_all[i]

        lap = L[inds, :][:, inds]

        evecs_pred = eigvecs_pred[inds, :]

        diag_eigvals = torch.diag(lambda_gt[eigval_inds])
        
        eigvecs_gt = evecs_gt_new[eigvec_start:len(inds) + eigvec_start, :]
        

        lambda_pred = evecs_pred.T @ lap @ evecs_pred

        low_rank_pred = evecs_pred @ torch.diag(torch.diag(lambda_pred))

        low_rank_pred = low_rank_pred @ evecs_pred.T
        low_rank_gt = eigvecs_gt @ diag_eigvals @ eigvecs_gt.T

        loss += torch.norm(low_rank_pred - low_rank_gt)
        eigval_inds = eigval_inds + 30
        eigvec_start += 300

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


