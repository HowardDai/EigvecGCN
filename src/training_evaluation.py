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

import csv

from visualization import *

def SupervisedLoss(evecs_pred, evecs_gt, batch, config):

    eigval_inds = torch.arange(config.evec_len, dtype=torch.long, device=adj.device)
    total_mse = 0

    for i in range(batch[-1] + 1):
        inds = list(torch.argwhere(batch == i).squeeze())

        evecs_pred = eigvecs_pred[inds, :]

        total_mse += torch.norm(evecs_pred - evecs_gt[eigval_inds, :][:evecs_pred.shape[0]]) ** 2

        eigval_inds = eigval_inds + config.evec_len
    
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


def SupervisedEigenvalueLoss(eigvecs_pred, adj, eigvals_gt, batch, config):
    L = get_lap(adj).to_dense()
    loss = 0
    num_eigvals = eigvecs_pred.shape[-1]

    eigval_inds = torch.arange(num_eigvals, dtype=torch.long, device=adj.device)
    for i in range(batch[-1] + 1):
        inds = list(torch.argwhere(batch == i).squeeze())

        lap = L[inds, :][:, inds]
        
        evecs_pred = eigvecs_pred[inds, :]
        diag_eigvals = torch.diag(eigvals_gt[eigval_inds])
        loss += torch.norm(lap @ evecs_pred  - evecs_pred @ diag_eigvals)

        eigval_inds = eigval_inds + config.num_eigenvectors
    return loss


def SupervisedEigenvalueLossUnweighted(eigvecs_pred, adj, eigvals_gt, batch, config):
    L = get_lap(adj).to_dense()
    loss = 0
    num_eigvals = eigvecs_pred.shape[-1]

    eigval_inds = torch.arange(num_eigvals, dtype=torch.long, device=adj.device)

    for i in range(batch[-1] + 1):
        inds = list(torch.argwhere(batch == i).squeeze())

        lap = L[inds, :][:, inds]
        
        eigvals_gt_inv = torch.reciprocal(eigvals_gt)
        evecs_pred = eigvecs_pred[inds, :]

        evecs_pred[:, 0] = torch.zeros_like(evecs_pred[:, 0]) # not considering predictions for the lowest eigval
        eigvals_gt_inv[eigval_inds][0] = 0 
        

        diag_eigvals_inv = torch.diag(eigvals_gt_inv[eigval_inds])
        
        loss += torch.norm(lap @ evecs_pred @ diag_eigvals_inv - evecs_pred)

        eigval_inds = eigval_inds + config.num_eigenvectors
    return loss

def projection_loss(config, eigvecs_pred, eigvecs_gt, batch):
    loss = 0
    evec_inds = 0

    for i in range(batch[-1] + 1):
        inds = list(torch.argwhere(batch == i).squeeze())
        U_hat = eigvecs_pred[inds, :]
        U = eigvecs_gt[evec_inds: evec_inds + len(inds)]
        loss += torch.norm(U.T @ U_hat @ U.T - U_hat.T)
        evec_inds += config.evec_len
    
    return loss

        

def compute_loss(config, out, data):
    if config.forced_ortho:
        out = orthogonalize_by_batch(out, data.batch)
    out = normalize_by_batch(out, data.batch)

    loss = 0

    if config.energy:
        loss += config.lambda_energy * EnergyLoss(out, data.edge_index) 
    if config.supervised_eigval:
        loss += SupervisedEigenvalueLoss(out, data.edge_index, data.eigvals, data.batch, config)
    if config.supervised_eigval_unweighted:
        loss += SupervisedEigenvalueLossUnweighted(out, data.edge_index, data.eigvals, data.batch, config)
    if config.supervised_mse:
            loss += SupervisedLoss(out, data.eigvecs[:, :config.num_eigenvectors], config)
    if config.supervised_lap_reconstruction:
        loss += lap_reconstruction_loss(out, data.eigvals, data.eigvecs[:, :config.num_eigenvectors], data.edge_index, data.batch, config)

        
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
        out = model(data.x, data.edge_index, data.batch)

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
        out = model(data.x, data.edge_index, data.batch)
        if config.forced_ortho:
            out = orthogonalize_by_batch(out, data.batch)

        out = normalize_by_batch(out, data.batch)

        loss = 0
        if config.energy:
            loss += config.lambda_energy * EnergyLoss(out, data.edge_index)
        if config.supervised_eigval:
            loss += SupervisedEigenvalueLoss(out, data.edge_index, data.eigvals, data.batch, config)
        if config.supervised_eigval_unweighted:
            loss += SupervisedEigenvalueLossUnweighted(out, data.edge_index, data.eigvals, data.batch, config)
        if config.supervised_mse:
            loss += SupervisedLoss(out, data.eigvecs[:, :config.num_eigenvectors], config)
        if config.supervised_lap_reconstruction:
            loss += lap_reconstruction_loss(out, data.eigvals, data.eigvecs[:, :config.num_eigenvectors], data.edge_index, data.batch, config)

        ortho_loss = config.lambda_ortho * OrthogonalityLoss(out)


        loss += ortho_loss 
            
        batch_size = int(data.batch.max()) + 1
        total_loss += loss.item()
        total_ortho_loss += ortho_loss.item() 

    total_loss = total_loss / len(loader.dataset)
    total_ortho_loss = total_ortho_loss / len(loader.dataset)
    
    return total_loss, total_ortho_loss


def training_loop(model, train_loader, val_loader, test_loader, optimizer, device, config):

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
                torch.save(model.state_dict(), f"checkpoints/{config.checkpoint_folder}/{epoch}.pt")
                best_val_loss = val_loss

                evaluate(model, test_loader, device, config) 
                if not config.forced_ortho:
                    evaluate(model, test_loader, device, config, extra_ortho_results=True)
                

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

    plot_loss_history(validation_loss_hist, train_loss_hist, config)

    return validation_loss_hist


import signal

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, _timeout_handler)



def evaluate(model, loader, device, config, extra_ortho_results=False):
    # NOTE: extra_ortho_results is used to force ortho on models which don't normally use forced_ortho (i.e. learned ortho), just to see how results would be if ortho was forced afterwards
    
    if extra_ortho_results:
        print("Evaluating with additional forced orthogonality...")
    model.eval()

    total_loss = 0
    total_ortho_loss = 0
    loss_dict = {'energy': 0, 'supervised_eigval': 0, 'supervised_eigval_unweighted': 0, 'supervised_lap_reconstruction': 0, 'ortho': 0, 'projection': 0}
    
    total_runtime = 0


    total_eigval_sum = 0
    total_num_eigvecs = 0

    num_eigvecs = config.num_eigenvectors


    for idx, data in enumerate(tqdm(loader)):
        

        
        data = data.to(device)
        t1 = time.time()

        if config.model == "harmonic": # recording the runtime INSIDE julia, not including julia/python latency 
            out, runtime = model(data.x, data.edge_index, data.batch)
            out = out.to(device)
        else:
            out = model(data.x, data.edge_index, data.batch).to(device) 
        

        if config.forced_ortho or extra_ortho_results:
            out = orthogonalize_by_batch(out, data.batch)

        out = normalize_by_batch(out, data.batch)

        if config.model == "harmonic":
            total_runtime += runtime
        else:
            total_runtime += time.time() - t1




        for loss_function in loss_dict:
            if loss_function == 'energy':
                loss = EnergyLoss(out, data.edge_index)
            if loss_function == 'supervised_eigval':
                loss = SupervisedEigenvalueLoss(out, data.edge_index, data.eigvals, data.batch, config)
            if loss_function == 'supervised_eigval_unweighted':
                loss = SupervisedEigenvalueLossUnweighted(out, data.edge_index, data.eigvals, data.batch, config)
            if loss_function == 'supervised_mse':
                loss = SupervisedLoss(out, data.eigvecs[:, :config.num_eigenvectors], data.batch, config)
            if loss_function == 'supervised_lap_reconstruction':
                loss = lap_reconstruction_loss(out, data.eigvals, data.eigvecs[:, :config.num_eigenvectors], data.edge_index, data.batch, config)
            if loss_function == 'ortho':
                loss = OrthogonalityLoss(out)
            if loss_function =='projection':
                loss = projection_loss(config, out, data.eigvecs[:, :config.num_eigenvectors], data.batch)

            loss_dict[loss_function] += loss.item()


        

        total_eigval_sum += torch.sum(data.eigvals)
        total_num_eigvecs += num_eigvecs * (data.batch[-1]+1)

        if idx % 100 == 0: # print one set of example eigvecs every 100 cases 
            if extra_ortho_results:
                tag = str(idx) + "_with_ortho"
            else:
                tag = str(idx)
            plot_eigvecs(out, data.eigvecs[:, :config.num_eigenvectors], data.edge_index, tag, config)
    

    num_items = len(loader.dataset) 

    print("k: ", num_eigvecs)
    print("average eigval: ", (total_eigval_sum / total_num_eigvecs).item() )
    print("Size of test set: ", (num_items))
    for loss_function in loss_dict:
        loss_dict[loss_function] =  loss_dict[loss_function] / num_items
        print(loss_function, ": ", loss_dict[loss_function])
    
    
    avg_runtime = total_runtime / num_items
    print(avg_runtime)
    out_dict = loss_dict
    out_dict['runtime'] = avg_runtime
    out_dict['name'] = config.checkpoint_folder
    out_dict['dataset'] = config.dataset
    
    

    fieldnames = list(out_dict.keys())
    if extra_ortho_results: 
        csv_file_name = f"plots/{config.checkpoint_folder}/metrics_with_ortho.csv"
    else:
        csv_file_name = f"plots/{config.checkpoint_folder}/metrics.csv"
    with open(csv_file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([out_dict])
    


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

def lap_reconstruction_loss(eigvecs_pred, lambda_gt, evecs_gt, adj, batch, config):
    
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
        eigval_inds = eigval_inds + config.num_eigenvectors
        eigvec_start += config.evec_len

    return loss

def procrustes(evecs_gt, evecs_pred, batch, config):
    loss = 0
    eigvec_start = 0
    inds_all = [torch.argwhere(batch == i).squeeze().tolist() for i in range(batch[-1] + 1)]
    for i in range(batch[-1] + 1):
        inds  = inds_all[i]
        gt = evecs_gt[eigvec_start : eigvec_start + config.evec_len, :]
        gt = gt[:len(inds)]
        pred  = evecs_pred[inds]

        M = gt @ pred.T
        U, S, Vh = torch.linalg.svd(M)
        R = U @ Vh

        loss += torch.norm(R @ pred - gt)
        
        eigvec_start += config.evec_len

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