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

from utils import *

import scipy.sparse as sp



def random_uniform_subset(M: torch.Tensor, k0: int = 1, min_frac: float = 0.01) -> list[int]: 
    """
    Select a random “uniform” subset of node indices from a square torch tensor M.
    
    Parameters:
    -----------
    M : Tensor of shape (n, n)
        Square adjacency-like matrix.
    k0 : int, default=1
        Neighborhood depth: uses (max(M, M.T))**k0 to define connectivity.
    min_frac : float, default=0.01
        Minimum fraction of n to include in the result.
    
    Returns:
    --------
    subset : list of int
        A list of unique indices in [0, n).
    """

    n = M.size(0)
    selected = set()
    
    # Symmetrize and raise to the k0-th power
    Msym = M # TODO: make this symmetrized? torch.maximum does not work
    # torch.matrix_power works for integer powers

    device = M.device


    Mexp = torch.eye(n).to(device)

    for i in range(k0):
        Mexp = Msym @ Mexp 

    # Mexp = torch.matrix_power(Msym, k0) # using this causes error: unsupported memory format option Contiguous
    
    for i in range(n):
        # neighbors = i itself plus any j where Mexp[i, j] > 0
        condition = Mexp[i] > 0
        condition[i] = 1 # guarantee that i itself is included
        nbrs_tensor = (condition).nonzero(as_tuple=False).flatten()
        nbrs = nbrs_tensor.tolist()
        
        # if none of these neighbors have been picked yet
        if not any(nb in selected for nb in nbrs):
            selected.add(random.choice(nbrs))
        # otherwise, with probability 1/len(nbrs), add i anyway
        elif random.random() <= 1.0 / len(nbrs):
            selected.add(i)
    
    # ensure at least min_frac * n elements
    target_size = math.ceil(n * min_frac)
    while len(selected) < target_size:
        selected.add(random.randrange(n))
    
    return list(selected)





import torch


def solve_laplacians_fast(L: torch.Tensor,
                          boundary: list[dict[int, float]]
                         ) -> torch.Tensor:
    """
    Fast Schur‐extension solver using an approx‐Cholesky SDDM solver.
    
    Args:
    -----
    L : (n,n) Tensor
        Laplacian matrix.
    boundary : list of dicts
        length = num_vectors; each dict maps boundary‐node idx → boundary value.
        Indices must be 0‐based.
    
    Returns:
    --------
    ext_vectors : (n, num_vectors) Tensor
        Each column k is the extended solution from boundary[k]
    """
    n = L.size(0)
    device, dtype = L.device, L.dtype

    # 1) Identify boundary nodes B and interior nodes I
    B = list(boundary[0].keys())               # e.g. [0, 3, 7, ...]
    B_set = set(B)
    I = [i for i in range(n) if i not in B_set]     # the complement

    num_vectors = len(boundary)
    ext_vectors = torch.zeros((n, num_vectors), dtype=dtype, device=device)

    # 2) Partition L
    #    L_II is the submatrix on interior×interior
    #    L_IB is interior×boundary
    L_II = L[I][:, I]   # shape (|I|, |I|)
    L_IB = L[I][:, B]   # shape (|I|, len(B))

    # 3) Build the fast SDDM solver once
    #    (Julia: sol = approxchol_sddm(sparse(a)))

    L_II_np = L_II.cpu().numpy()

    # 1.2. Make it a SciPy CSR sparse
    L_II_csr = sp.csr_matrix(L_II_np)

    # 1.3. Convert that into a Julia SparseMatrixCSC
    L_II_jl  = SparseArrays.sparse(L_II_csr)

    # 1.4. Now build the solver
    sol = Laplacians.approxchol_sddm(L_II_jl)
    # sol = Laplacians.approxchol_sddm(L_II)

    # 4) For each right‐hand side (boundary condition)…
    for k, bdict in enumerate(boundary):
        # 4a) form x_B
        x_B = torch.tensor([bdict[i] for i in B],
                           dtype=dtype, device=device)           # shape (len(B),)

        # 4b) form the interior RHS: b = - L_IB @ x_B
        b = - L_IB @ x_B                                    # shape (|I|,)

        # 4c) solve approximately for x_I
        x_jl = sol(b, tol=1e-15)                             # shape (|I|,)
        print(type(x_jl))
        x_I = np.array(x_jl)

        # 4d) assemble full solution
        ext_vectors[I, k] = x_I
        ext_vectors[B, k] = x_B

        # 4e) normalize the new column
        ext_vectors[:, k] = ext_vectors[:, k] / ext_vectors[:, k].norm()

    return ext_vectors


def schur_subset(L: torch.Tensor, kept: List[int]) -> torch.Tensor:
    """
    Compute the Schur complement of the block indexed by `kept` in matrix L.
    
    Parameters:
    -----------
    L : (n, n) torch.Tensor
        A square matrix (e.g., Laplacian).
    kept : List[int]
        The indices to keep (0-based).
    
    Returns:
    --------
    L_new : (len(kept), len(kept)) torch.Tensor
        The Schur‐complemented submatrix on `kept`.
    """
    n = L.size(0)
    # interior indices = all indices not in `kept`
    kept_set = set(kept)
    v = [i for i in range(n) if i not in kept_set]
    
    # extract blocks
    L_vv = L[v][:, v]         # L[v, v]
    L_vK = L[v][:, kept]      # L[v, kept]
    L_Kv = L[kept][:, v]      # L[kept, v]
    L_KK = L[kept][:, kept]   # L[kept, kept]
    
    # Schur complement:  L_KK - L_Kv @ inv(L_vv) @ L_vK
    inv_Lvv = torch.inverse(L_vv)
    L_new = L_KK - (L_Kv @ inv_Lvv @ L_vK)
    
    return L_new

def get_schur_eigvec_approximations(M: torch.Tensor, k, subset_function):
    kept = subset_function(M, k0=1, min_frac=0.2) # Find a subset of the graph
    L = get_lap(M)

    # Compute schur subset (effective resistance weight updates)
    L_schur = schur_subset(L, kept) 

    # Find eigenvectors on the schur subset
    _, eigvecs_schur = torch.linalg.eigh(L_schur)
    eigvecs_schur = eigvecs_schur[:, :k]
    # harmonically extend


    d_schurs = [dict(zip(kept, eigvecs_schur[:,i])) for i in range(k)]
    eff_exts = solve_laplacians_fast(L, d_schurs)

    # eff_exts = eff_exts / norm(eff_exts) # all models get normalized in train loop by default

    return eff_exts 


from julia import Main

Main.include("models/harmonic.jl")
Main.eval("using .Harmonic")


class HarmonicAlgorithm(nn.Module):
    def __init__(self, output_dim, subset_method="random_uniform_subset"): 
        
        super().__init__()
        self.subset_function = random_uniform_subset
        self.output_dim = output_dim
    
    def forward(self, x, edge_index, batch):
        final_out = torch.zeros(0, self.output_dim)
        for i in range(batch[-1] + 1):
            out = Main.eval("Harmonic.get_schur_eigvec_approximations") # get_schur_eigvec_approximations(edge_index, self.output_dim, self.subset_function)
            final_out = torch.cat(final_out, out, dim=0)
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
