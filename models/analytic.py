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

from scipy.sparse.linalg import eigsh, ArpackNoConvergence
import warnings


from julia import Main

Main.include("models/harmonic.jl")
# Main.eval("using .Harmonic")


class HarmonicAlgorithm(nn.Module):
    def __init__(self, output_dim, subspace_size, subset_method="random_uniform_subset"): 
        
        super().__init__()
        # self.subset_function = random_uniform_subset
        self.subset_method = subset_method
        self.output_dim = output_dim
        self.subspace_size = subspace_size
    
    def forward(self, x, edge_index, batch):
        final_out = torch.zeros(0, self.output_dim)
        for i in range(batch[-1] + 1):
            
            numpy_adj = edge_index.to(torch.float64).to_dense().cpu().numpy()
            out, runtime = Main.get_schur_eigvec_approximations(numpy_adj, self.subspace_size, self.subset_method) # get_schur_eigvec_approximations(edge_index, self.output_dim, self.subset_function)
            arr = np.asarray(out)
            t = torch.from_numpy(arr)

            final_out = torch.cat((final_out, t[:, :self.output_dim]), dim=0)
            final_out = final_out.to(torch.float32)
            # print(final_out)
        return final_out, runtime


class GroundTruth(nn.Module):
    def __init__(self, output_dim): 
        
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x, edge_index, batch):
        L, Q = torch.linalg.eigh(get_lap(edge_index))
        Q = Q[:, :self.output_dim]
        return Q

    
class LanczosAlgorithm(nn.Module):
    def __init__(self, output_dim, subspace_size): 
        super().__init__()
        self.output_dim = output_dim
        self.subspace_size = subspace_size

    def forward(self, x, edge_index, batch):
        adj = get_lap(edge_index).cpu().numpy()
        sparse_adj = sparse.csr_matrix(adj)

        success = True # whether or not Lanczos successfully converged
        pad_vecs_count = 0 # number of vectors randomly filled (if not converged)


        try:
        # attempt to compute the k smallest eigenpairs
            vals, vecs = eigsh(sparse_adj, k=self.output_dim, ncv=self.subspace_size, which='SM')
            eigvecs = torch.from_numpy(vecs)
        except ArpackNoConvergence as e:
            # Warn, grab whatever converged, and keep going
            warnings.warn(
                f"ARPACK failed to converge for k={self.output_dim} ("
                f"only {len(e.eigenvalues)} converged)",
                RuntimeWarning
            )
            vals = e.eigenvalues
            vecs = e.eigenvectors

            eigvecs = torch.from_numpy(vecs)

            pad_vecs_count = self.output_dim - vecs.shape[-1] 
            random_pad = torch.rand(x.shape[0], pad_vecs_count).to(eigvecs.device)
            eigvecs = torch.cat((eigvecs,random_pad), dim=-1)

            success=False

        return eigvecs[:, :], success, pad_vecs_count



class RandomVectors(nn.Module):
    def __init__(self, output_dim): 
        
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x, edge_index, batch):
        return torch.rand(x.shape[0], self.output_dim)
        

def lanczos(A, k, v0=None, tol=1e-6):
    """
    Performs k-step Lanczos algorithm on symmetric matrix A.
    Args:
        A: torch.Tensor of shape (n, n) or a callable that performs A(x) for x of shape (n,).
        k: int, number of Lanczos iterations (dimension of Krylov subspace).
        v0: torch.Tensor of shape (n,), optional initial vector. If None, a random vector is used.
        tol: float, tolerance for convergence (if beta < tol, stops early).
    Returns:
        alphas: torch.Tensor of shape (m,), diagonal elements of the tridiagonal matrix, where m <= k.
        betas: torch.Tensor of shape (m-1,), off-diagonal elements.
        Q: torch.Tensor of shape (n, m), the orthonormal basis vectors.
    """
    # Determine matvec operation and dimension
    if isinstance(A, torch.Tensor):
        n = A.shape[0]
        matvec = lambda x: A.matmul(x)
        device = A.device
    elif callable(A):
        if v0 is None:
            raise ValueError("v0 must be provided when A is a callable")
        n = v0.shape[0]
        device = v0.device
        matvec = A
    else:
        raise TypeError("A must be a torch.Tensor or a callable")

    # Initialize
    if v0 is None:
        q = torch.randn(n, device=device)
    else:
        q = v0.to(device)
    q = q / torch.norm(q)
    Q = torch.zeros(n, k, device=device)
    alphas = torch.zeros(k, device=device)
    betas = torch.zeros(k-1, device=device)

    w = matvec(q)
    alpha = torch.dot(q, w)
    alphas[0] = alpha
    w = w - alpha * q

    Q[:, 0] = q

    m = 1
    for j in range(1, k):
        beta = torch.norm(w)
        if beta < tol:
            alphas = alphas[:m]
            betas = betas[:m-1]
            Q = Q[:, :m]
            break
        betas[j-1] = beta
        q = w / beta
        Q[:, j] = q
        w = matvec(q)
        alpha = torch.dot(q, w)
        alphas[j] = alpha
        # Orthogonalize against previous basis vector
        w = w - alpha * q - beta * Q[:, j-1]
        m += 1

    return alphas, betas, Q


def lanczos_eig(A, k, v0=None, tol=1e-6):
    """
    Computes approximate eigenvalues and eigenvectors of A using k-step Lanczos.
    Returns:
        eigvals: torch.Tensor of shape (m,), approximate eigenvalues.
        eigvecs: torch.Tensor of shape (n, m), approximate eigenvectors.
    """
    alphas, betas, Q = lanczos(A, k, v0, tol)
    # Build tridiagonal matrix
    m = alphas.shape[0]
    T = torch.diag(alphas)
    if m > 1:
        T += torch.diag(betas, 1) + torch.diag(betas, -1)
    # Compute eigendecomposition of T
    eigvals, eigvecs_T = torch.linalg.eigh(T)
    # Compute approximate eigenvectors of A
    eigvecs = Q @ eigvecs_T
    return eigvals, eigvecs
