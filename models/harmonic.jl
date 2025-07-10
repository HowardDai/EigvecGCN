# module Harmonic
using PyCall
using Laplacians, SparseArrays, LinearAlgebra, Random


function random_uniform_subset(M; k0=1, min_count=60) # TODO: add k0 parameter for neighborhood depth
    n = size(M)[1]
    random_values = Set{Int}()
    Mexp = max.(M,M')^k0
    for i in collect(1:n)
        # compute neighbors
        nbrs = 
            [j for j in 1:n if (Mexp[i, j] > 0 || j == i)]
        if !any(x -> in(x, random_values), nbrs)
      
            push!(random_values, rand(nbrs)) # randomly add i or a neighbor
        elseif rand() <= 1 / size(nbrs)[1] # chance of adding i anyway
            push!(random_values, i)
        end
    end
    while length(random_values) < min_count
        push!(random_values, rand(1:n))
    end
    return collect(random_values)
end



function solve_laplacians_fast(L::AbstractMatrix, boundary::Vector{Dict{Int, Float64}})

    n = size(L, 1)
    all_indices = 1:n
    B = collect(keys(boundary[1]))             # Boundary indices
    I = setdiff(all_indices, B)             # Interior indices
    num_vectors = size(boundary)[1]
    ext_vectors = zeros(n,num_vectors)                            # Full solution vector (to fill in)


    # Partition Laplacian
    L_II = L[I, I]
    L_IB = L[I, B]

    a = L_II # this is like a Laplacian but with extra degrees counted on diagonal
    sol = approxchol_sddm(sparse(a), maxits=20, maxtime=30, verbose=true)


    for k in collect(1:num_vectors)
      boundary_k = boundary[k]
      x_B = [boundary_k[i] for i in B]       # Boundary values
      # print(schur_subset(L, B) * x_B ./ x_B)
      b = -1 * L_IB * x_B
      x_I = sol(b, tol=1e-8)
      print("solved")
      flush(stdout)
      ext_vectors[I, k] = x_I
      ext_vectors[B,k] = x_B


      # x_I_exact = -L_II \ (L_IB * x_B) #
      # ext_vectors[I, k] = x_I_exact
      # error = norm(x_I_exact - x_I)
      # println("error at $k: $error") #

      ext_vectors[:, k] = ext_vectors[:, k] / norm(ext_vectors[:, k]) # normalization
    end

    return ext_vectors
end

# TODO: turn this into the fast version
function schur_subset(L::AbstractMatrix{Float64}, kept::Vector{Int})
    n = size(L, 1)
    v = setdiff(1:n, kept)

    # Extract blocks
    L_vv = L[v, v]
    L_vK = L[v, kept]
    L_Kv = L[kept, v]
    L_KK = L[kept, kept]

    # Schur complement update
    L_new = L_KK - L_Kv * sparse(inv(Matrix(L_vv))) * L_vK

    return L_new
end


function get_schur_eigvec_approximations(M::AbstractMatrix{Float64}, subset_method)
    runtimes = Dict()

    M = sparse(M)

    L = lap(M)
  
    t1 = time() #

    if subset_method == "random_uniform_subset"
        subset_indices = random_uniform_subset(M)
    end
    # L_schur = schur_subset_fast(L, subset_indices; approxchol_tol=1e-20, JLfac=20)
    L_schur = schur_subset(L, subset_indices)
    #L_schur = schur_subset_filtered(L, subset_indices)
    #L_schur = schur_subset_fast_subgraphs(L, subset_indices)
    runtimes["Schur"] = time() - t1 #
  
    t1 = time() #
    E_schur = eigen(Matrix(L_schur))
    runtimes["Eigen"] = time() - t1 #
  
    n = size(M)[1]
    eff_exts = zeros(n,n)
  
  
    t1 = time()
    d_schurs   = [Dict(zip(subset_indices, real(E_schur.vectors[:, i]))) for i in collect(1:length(subset_indices))]
    eff_exts = solve_laplacians_fast(L, d_schurs)
    runtimes["Laplacian"]  = time() - t1
  
  
  
  
  
    # print(runtimes)
    eff_exts = eff_exts / norm(eff_exts)

    flush(stdout)

    return Array(eff_exts)
  
  
end





# export get_schur_eigvec_approximations

# end