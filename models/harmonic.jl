# module Harmonic
using PyCall
using Laplacians, SparseArrays, LinearAlgebra, Random


macro timeout(seconds, expr_to_run, expr_when_fails)
    quote
        tsk = @task $(esc(expr_to_run))
        schedule(tsk)
        Timer($(esc(seconds))) do timer
            istaskdone(tsk) || Base.throwto(tsk, InterruptException())
        end
        try
            fetch(tsk)
        catch _
            $(esc(expr_when_fails))
        end
    end
end


function random_uniform_subset(M; k0=1, min_count=60) # TODO: add k0 parameter for neighborhood depth

    n = size(M)[1]
    if n <= min_count
        return collect(Set{Int}(1:n-1)) # choose the set which is all but one node
    end
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
    sol = approxchol_sddm(sparse(a), maxits=20, maxtime=30, verbose=false)


    for k in collect(1:num_vectors)
      boundary_k = boundary[k]
      x_B = [boundary_k[i] for i in B]       # Boundary values
      # print(schur_subset(L, B) * x_B ./ x_B)
      b = -1 * L_IB * x_B

      x_I = sol(b, tol=1e-8) 

 

    #   if x_I == zeros(length(I))
    #     print("timeout, set x_I to zeros")
    #   end
    
    #   flush(stdout)
      ext_vectors[I, k] = x_I
      ext_vectors[B,k] = x_B


    #   x_I_exact = -L_II \ (L_IB * x_B) #
    #   ext_vectors[I, k] = x_I_exact
    #   error = norm(x_I_exact - x_I)
    #   println("error at $k: $error") #

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


function solve_laplacians(L::AbstractMatrix, boundary::Vector{Dict{Int, Float64}})

    n = size(L, 1)
    all_indices = 1:n
    B = collect(keys(boundary[1]))             # Boundary indices
    I = setdiff(all_indices, B)             # Interior indices
    num_vectors = size(boundary)[1]
    ext_vectors = zeros(n,num_vectors)                            # Full solution vector (to fill in)


    # Partition Laplacian
    L_II = L[I, I]
    L_IB = L[I, B]

    for k in collect(1:num_vectors)
      boundary_k = boundary[k]
      x_B = [boundary_k[i] for i in B]       # Boundary values

      x_I_exact = -L_II \ (L_IB * x_B) #
      ext_vectors[I, k] = x_I_exact
      ext_vectors[B,k] = x_B


      ext_vectors[:, k] = ext_vectors[:, k] / norm(ext_vectors[:, k]) # normalization
    end

    return ext_vectors
end


function get_schur_eigvec_approximations(M::AbstractMatrix{Float64}, min_subset_size, subset_method)
    t1_overall = time()

    runtimes = Dict()

    M = sparse(M)

    L = lap(M)
  
    t1 = time() #

    if subset_method == "random_uniform_subset"
        subset_indices = random_uniform_subset(M, min_count=min_subset_size)
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
  
  
  
  
  
    
    eff_exts = eff_exts / norm(eff_exts)

    print(runtimes)
    flush(stdout)

    t2_overall = time()
    runtime = t2_overall - t1_overall 
    
    return Array(eff_exts), runtime
  
end


function get_schur_eigvec_approximations_timed(M::AbstractMatrix{Float64}, subset_method)
    runtimes = Dict()

    M = sparse(M)

    L = lap(M)
  
    t1 = time() #

    if subset_method == "random_uniform_subset"
        subset_indices = random_uniform_subset(M)
    end

    
    # L_schur = schur_subset_fast(L, subset_indices; approxchol_tol=1e-20, JLfac=20)
    L_schur = @timeout 3 begin 
        res = schur_subset(L, subset_indices)
        println("Schur subset computed successfully")
        flush(stdout)
        res
    end 0

    if L_schur == 0
      L_schur = L[subset_indices, subset_indices]
      println("Schur subset timed out")
      flush(stdout)
    end

    println("Computing eigenvectors on subgraph...")
    flush(stdout)

    #L_schur = schur_subset_filtered(L, subset_indices)
    #L_schur = schur_subset_fast_subgraphs(L, subset_indices)
    runtimes["Schur"] = time() - t1 #
  
    t1 = time() #


    E_schur = @timeout 3 begin
        res = eigen(Matrix(L_schur))
        println("Eigendecomposition successful")
        flush(stdout)
        res
    end 0

    if E_schur == 0
        println("Eigendecomposition timed out, returning early...")
        flush(stdout)
        return zeros(length(M[1,:]), length(subset_indices))
    end 

    runtimes["Eigen"] = time() - t1 #
  
    n = size(M)[1]
    eff_exts = zeros(n,n)
  
    println("Beginning laplacian solving... ")
    flush(stdout)
    
    t1 = time()
    d_schurs   = [Dict(zip(subset_indices, real(E_schur.vectors[:, i]))) for i in collect(1:length(subset_indices))]
    eff_exts = solve_laplacians(L, d_schurs)
    runtimes["Laplacian"]  = time() - t1

    println("Laplacians solved")
    flush(stdout)
  
  
  
    # print(runtimes)
    eff_exts = eff_exts / norm(eff_exts)

    flush(stdout)

    return Array(eff_exts)
  
  
end


function get_schur_eigvec_approximations_wrapper(M::AbstractMatrix{Float64}, subset_method)
    arg = M, subset_method
    println("Beginning computation...")
    result = get_schur_eigvec_approximations_timed(M, subset_method) # timeout(get_schur_eigvec_approximations, arg, 10, 0)
    
    if result == 0
        println("Timed out")
        return one(M)
    else
        println("Successful computation")
        return result
    end 

end




# export get_schur_eigvec_approximations

# end

function timeout(f, args, seconds, fail)
    tsk = @task f(args...)
    schedule(tsk)
    Timer(seconds) do timer
        istaskdone(tsk) || Base.throwto(tsk, InterruptException())
    end
    try
        fetch(tsk)
    catch e;
        showerror(stdout, e)
        fail
    end
end