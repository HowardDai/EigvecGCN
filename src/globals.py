import torch
from argparse import ArgumentParser

use_cuda = torch.cuda.is_available()

parser = ArgumentParser()
# parser.add_argument("--cuda", type=bool, default=use_cuda)
parser.add_argument("--model", type=str, choices=["GCN", 'GIN', 'MLP'], default='GCN')

parser.add_argument("--hidden_dim", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--use_bias", type=bool, default=True)

parser.add_argument("--num_eigenvectors", type=float, default=30)

parser.add_argument("--lambda_energy", type=float, default=50)
parser.add_argument("--lambda_ortho", type=float, default=1)


parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=5e-3)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--use_early_stopping", type=bool, default=True)
parser.add_argument("--multiple_runs", type=bool, default=False)
parser.add_argument("--num_of_runs", type=int, default=100)
parser.add_argument("--follow_paper", type=bool, default=True)

parser.add_argument("--loss_function", type=str, choices=['energy', 'supervised_eigval', 'supervised_lap_reconstruction', 'supervised_mse'], default='energy')
parser.add_argument("--embedding_type", type=str, choices=['diffusion', 'wavelet', 'trivial', 'scatter'], default='trivial')


parser.add_argument("--use_supervised", type=bool, default=True) # To be replaced 

config = parser.parse_args()