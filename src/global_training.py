import torch
import torch.nn as nn
import numpy as np
import time 
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt
from datetime import datetime, date
import random
import csv
from visualization import *
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


def forced_ortho(out):
    orthos = torch.zeros_like(out)
    for i in range(out.shape[0]):
        Q, R = torch.linalg.qr(out[i])
        orthos[i] = Q

    return orthos

def energy_loss(out, lap):
    energy_mat = torch.bmm(torch.bmm(torch.transpose(out, -2, -1), lap), out)
    energy = torch.sum(torch.diagonal(energy_mat, dim1=-2, dim2=-1))

    return energy

def eigval_loss(out, lap, eigvals_gt):
    pred = torch.bmm(lap, out)
    gt = torch.bmm(out, eigvals_gt)

    return torch.norm(pred - gt)

def projection_loss(out, eigvecs_gt):
    projected = torch.bmm(torch.bmm(torch.transpose(eigvecs_gt, -2, -1), out), torch.transpose(eigvecs_gt, -2, -1))
    true = torch.transpose(out, -2, -1)
    loss = torch.norm(projected -true)

    return loss


def run_model(model, loader, optimizer, device, config, train):
    model.to(device)
    total_loss = 0
    total_ortho_loss = 0

    for embeddings, adj, evals, evec, lap in tqdm(loader):
        embeddings = embeddings.to(device)
        adj = adj.to(device)
        evals = evals.to(device)
        lap = lap.to(device)

        out = model(embeddings)
        loss = 0

        if config.forced_ortho:
            out = forced_ortho(out)
        if config.energy:
            loss += config.energry_weight * energy_loss(out, lap)
        elif config.supervised_eigval:
            loss += config.eigval_weight * eigval_loss(out, lap, evals)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

        batch_size = embeddings.shape[0]
        total_loss += loss.item()

    total_loss = total_loss / len(loader.dataset)
    return total_loss


def global_training_loop(model, train_loader, val_loader, optimizer, device, config):
    validation_loss_hist = []
    train_loss_hist = []

    t_start = time.time()
    for epoch in range(config.epochs):

        optimizer.zero_grad()
        model.train()
        train_loss = run_model(model, train_loader, optimizer, device, config, True)
        train_loss_hist.append(train_loss)
        best_val_loss = float('inf')

        model.eval()
        with torch.no_grad():
            
            val_loss = run_model(model, val_loader, optimizer, device, config, False)

            validation_loss_hist.append(val_loss)


            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f"checkpoints/{config.checkpoint_folder}/{epoch}.pt")
                best_val_loss = val_loss

                out_dict = evaluate(model, val_loader, device, config)
                fieldnames = list(out_dict.keys())
                csv_file_name = f"plots/{config.checkpoint_folder}/metrics.csv"
                with open(csv_file_name, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows([out_dict])
        if not config.multiple_runs:
            print(" | ".join([f"Epoch: {epoch:4d}", f"Train loss: {train_loss:.3f}",
                              f"Val loss: {val_loss:.3f}"
                             ]))
    plt.plot(validation_loss_hist, label="val")
    plt.plot(train_loss_hist, label="train")
    plt.legend()
    plt.show()
    return validation_loss_hist


def evaluate(model, loader, device, config):
    model.eval()
    total_loss = 0
    total_ortho_loss = 0
    loss_dict = {'energy': 0, 'supervised_eigval': 0, 'projection': 0}

    total_runtime = 0

    for embeddings, adj, evals, evec, lap in tqdm(loader):
        embeddings = embeddings.to(device)
        adj = adj.to(device)
        evals = evals.to(device)
        lap = lap.to(device)
        evec = evec.to(device)

        t1 = time.time()
        out = model(embeddings)
        
        if config.forced_ortho:
            out = forced_ortho(out)
        
        total_runtime += time.time() - t1
        
        for loss_function in loss_dict:
            if loss_function == 'energy':
                loss = energy_loss(out, lap)
            if loss_function == 'supervised_eigval':
                loss = eigval_loss(out, lap, evals)
            if loss_function =='projection':
                loss = projection_loss(out, evec)

            loss_dict[loss_function] += loss.item()
        
    for loss_function in loss_dict:
        loss_dict[loss_function] =  loss_dict[loss_function] / len(loader.dataset)
        print(loss_function, ": ", loss_dict[loss_function])
    
    
    avg_runtime = total_runtime / len(loader.dataset)
    print(avg_runtime)
    out_dict = loss_dict
    out_dict['runtime'] = avg_runtime
    out_dict['dataset'] = config.dataset

    return out_dict