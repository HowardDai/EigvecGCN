import torch
import random
import math

from julia import Julia
# disable compiled_modules on first run to avoid some linkage issues
jl = Julia(compiled_modules=False)                                 

# 2) Import the Julia packages we need
from julia import SparseArrays, Laplacians     

import numpy as np
from scipy import sparse

from typing import List

import torch.nn as nn
import torch.nn.functional as F



from utils import *

import scipy.sparse as sp


from julia import Main

Main.include("models/harmonic.jl")
# Main.eval("using .Harmonic")


class HarmonicAlgorithm(nn.Module):
    def __init__(self, output_dim, subset_method="random_uniform_subset"): 
        
        super().__init__()
        # self.subset_function = random_uniform_subset
        self.subset_method = subset_method
        self.output_dim = output_dim
    
    def forward(self, x, edge_index, batch):
        final_out = torch.zeros(0, self.output_dim)
        for i in range(batch[-1] + 1):
            
            numpy_adj = edge_index.to(torch.float64).to_dense().cpu().numpy()
            out = Main.get_schur_eigvec_approximations(numpy_adj, self.subset_method) # get_schur_eigvec_approximations(edge_index, self.output_dim, self.subset_function)
            arr = np.asarray(out)
            t = torch.from_numpy(arr)

            final_out = torch.cat((final_out, t[:, :self.output_dim]), dim=0)
            final_out = final_out.to(torch.float32)
            # print(final_out)
        return final_out

"""
TBD:
schur_subset
get_schur_eigvec_approximations


get_basic_eigvec_approximations
orthonormalize_first_k_columns 
rayleigh_quotient
get_predicted_eigenvals
sort_by_energy
make_plots
do_all_runs
[graph loading]
"""
