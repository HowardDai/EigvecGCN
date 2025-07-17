import torch
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math 
import phate
from utils import *


def plot_phate(evecs_pred, evecs_gt, adj, idx_label, config):

    fig_i, axs = plt.subplots(config.num_eigenvectors, 2, figsize=(20, 45), sharex=True)
    k = -1


    phate_op = phate.PHATE(n_components=2,knn_dist='precomputed_affinity')
    X_phate = phate_op(adj)

    for k in range(config.num_eigenvectors):
        
        
        ax_pred = axs[i, 0]
        ax_gt = axs[i, 1]

        if torch.dot(evecs_pred[:, k], evecs_gt[:, k]) < 0:

            evecs_pred[:, k] = evecs_pred[:, k] * -1


        ax_pred.scatter(X_phate[:, 0], X_phate[:, 1], c=evecs_pred[:, k], label='predicted')
        ax_gt.scatter(X_phate[:, 0], X_phate[:, 1], c=evecs_pred[:, k], label='ground truth')

        ax_pred.set(title=f"Phi {k + 1} (predicted)")
        ax_gt.set(title=f"Phi {k + 1} (ground truth)")


            
    fig_i.suptitle(f"First {config.num_eigenvectors} Eigenvectors")
    # fig_i.legend(h, l)
    fig_i.savefig(f"plots/{config.checkpoint_folder}/eigvecs_PHATE_{idx_label}.png")

    return


def plot_eigvecs(evecs_pred, evecs_gt, adj, idx_label, config):
    
    
    
    energies = torch.diag(evecs_pred.T @ get_lap(adj) @ evecs_pred)
    evecs_inds = torch.argsort(energies)

    evecs_gt = evecs_gt[:evecs_pred.shape[0]] # remove padding on gt eigvecs

    sort_inds = torch.argsort(evecs_gt[:, 1]).tolist()
    evecs_gt = evecs_gt[sort_inds].to_dense()

    evecs_pred = evecs_pred[sort_inds][:, evecs_inds].to_dense()

    # LINE PLOTS 
    num_rows = max(math.ceil(config.num_eigenvectors / 2), 2)
    fig_i, axs = plt.subplots(num_rows, 2, figsize=(20, 45), sharex=True)
    k = -1
    for i in range(num_rows):
        for j in [0, 1]:
            k += 1
            if k >= config.num_eigenvectors:
                break
            ax = axs[i, j]
    

            
            if torch.dot(evecs_pred[:, k], evecs_gt[:, k]) < 0:
    
                evecs_pred[:, k] = evecs_pred[:, k] * -1

            ax.plot(evecs_pred[:, k].cpu().detach().numpy(), color='tab:blue', label='prediction')
            ax.plot(evecs_gt[:, k].cpu().detach().numpy(), color='tab:orange', label='ground truth')
            ax.set_ylim(-1, 1) # normalized

            if i == num_rows - 1:
                ax.set(xlabel='Vertex')
            if j == 0:
                ax.set(ylabel='Eigenfunction')
            ax.set(title=f"Phi {k + 1}")
        h, l = ax.get_legend_handles_labels()
            
    fig_i.suptitle(f"First {config.num_eigenvectors} Eigenvectors")
    fig_i.legend(h, l)
    fig_i.savefig(f"plots/{config.checkpoint_folder}/eigvecs_{idx_label}.png")

    # plot_phate(evecs_pred, evecs_gt, sample_data.edge_index)
    # PHATE PLOTS

def plot_loss_history(validation_loss_hist, train_loss_hist, config):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, config.epochs + 1), validation_loss_hist, color='tab:blue', label='validation')
    ax.plot(range(1, config.epochs + 1), train_loss_hist, color='tab:orange', label='training')
    ax.set(xlabel='Epoch', ylabel='Loss', title=f"Loss History")
    ax.legend()
    fig.savefig(f"plots/{config.checkpoint_folder}/loss_plots.png")


# TO BE REPLACED BY PLOT_LOSS AND PLOT_EIGVECS
def plot_results(config, model, device, val_loader, validation_loss_hist=None, train_loss_hist=None):
    if config.train:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(range(1, config.epochs + 1), validation_loss_hist, color='tab:blue', label='validation')
        ax.plot(range(1, config.epochs + 1), train_loss_hist, color='tab:orange', label='training')
        ax.set(xlabel='Epoch', ylabel='Loss', title=f"Loss History")
        ax.legend()
        fig.savefig(f"plots/{config.checkpoint_folder}/loss_plots.png")

    sample_data = None
    for data in val_loader:  
        sample_data = data
        break

    inds = torch.argwhere(sample_data.batch == 0).tolist()
    model.to(device)
    sample_data.to(device)
    evecs_pred = model(sample_data.x, sample_data.edge_index, sample_data.batch)
    evecs_pred = normalize_by_batch(evecs_pred, sample_data.batch)[:len(inds), :]
    
    energies = torch.diag(evecs_pred.T @ get_lap(sample_data.edge_index)[:len(inds), :len(inds)] @ evecs_pred)
    evecs_inds = torch.argsort(energies)

    evecs_gt = sample_data.eigvecs[:len(inds), :]
    sort_inds = torch.argsort(evecs_gt[:, 1]).tolist()
    evecs_gt = evecs_gt[sort_inds]

    evecs_pred = evecs_pred[sort_inds][:, evecs_inds]

    # LINE PLOTS 
    fig_i, axs = plt.subplots(15, 2, figsize=(20, 45), sharex=True)
    k = -1
    for i in range(15):
        for j in [0, 1]:
            k += 1
            ax = axs[i, j]
    

            if torch.dot(evecs_pred[:, k], evecs_gt[:, k]) < 0:
    
                evecs_pred[:, k] = evecs_pred[:, k] * -1

            ax.plot(evecs_pred[:, k].cpu().detach().numpy(), color='tab:blue', label='prediction')
            ax.plot(evecs_gt[:, k].cpu().detach().numpy(), color='tab:orange', label='ground truth')
            if i == 14:
                ax.set(xlabel='Vertex')
            if j == 0:
                ax.set(ylabel='Eigenfunction')
            ax.set(title=f"Phi {k + 1}")
        h, l = ax.get_legend_handles_labels()
            
    fig_i.suptitle("First 30 Eigenvectors")
    fig_i.legend(h, l)
    fig_i.savefig(f"plots/{config.checkpoint_folder}/eigvecs.png")

    # plot_phate(evecs_pred, evecs_gt, sample_data.edge_index)
    # PHATE PLOTS
    