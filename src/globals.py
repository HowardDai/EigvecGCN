import torch
from argparse import ArgumentParser

use_cuda = torch.cuda.is_available()

parser = ArgumentParser()
# parser.add_argument("--cuda", type=bool, default=use_cuda)

# MODEL ARCHITECTURE (some of these are not currently active)
parser.add_argument("--model", type=str, choices=["GCN", 'GIN', 'MLP', 'harmonic'], default='MLP')

parser.add_argument("--hidden_dim", type=int, default=30)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--use_bias", type=bool, default=True)

parser.add_argument("--num_eigenvectors", type=float, default=30)

parser.add_argument("--evec_len", type=float, default=300)


parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=5e-3)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--use_early_stopping", action="store_true")
parser.add_argument("--multiple_runs", type=bool, default=False)
parser.add_argument("--num_of_runs", type=int, default=100)
parser.add_argument("--follow_paper", type=bool, default=True)




# LOSS FUNCTIONS
parser.add_argument("--energy", action="store_true")
parser.add_argument("--supervised_eigval", action="store_true")
parser.add_argument("--supervised_eigval_unweighted", action="store_true")
parser.add_argument("--supervised_lap_reconstruction", action="store_true")
parser.add_argument("--supervied_mse", action="store_true")


# parser.add_argument("--embedding_type", type=str, choices=['diffusion', 'wavelet', 'trivial', 'scatter'], default='trivial')

parser.add_argument("--forced_ortho", action="store_true") 

parser.add_argument("--lambda_energy", type=float, default=50)

parser.add_argument("--lambda_ortho", type=float, default=1)

# EMBEDDINGS

parser.add_argument("--diffusion_emb",action="store_true")

parser.add_argument("--diffusion_row",action="store_true")

parser.add_argument("--wavelet_emb",action="store_true")
# Choose two nodes, dirac at each, wavelets at varied powers 

parser.add_argument("--scatter_emb",action="store_true")
# Choose two nodes, dirac at each, scatter wavelets

parser.add_argument("--global_scatter_emb",action="store_true")
# for each node, dirac, scatter wavelets, and then global moment aggregation 


# DATASET LOADING
parser.add_argument("--use_mini_dataset", type=float, default=1)
# parser.add_argument("--embed_each_epoch")

parser.add_argument("--use_supervised", action="store_true") # only needs to be included for preprocessing  



# MODEL LOAD PATHS 

parser.add_argument("--checkpoint_folder", type=str, default="checkpoints")

parser.add_argument("--load_model", type=str, default=None) 


# ACTIONS 

parser.add_argument("--train", action="store_true")

parser.add_argument("--test", action="store_true")


#DATA

parser.add_argument("--dataset", type=str, choices=["ogbg_ppa", 'zinc'])


config = parser.parse_args()
