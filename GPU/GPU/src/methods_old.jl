logspace(x1, x2, n; digits=1) = unique!(round.([10.0^y for y in range(log10(x1), log10(x2), length=n)], digits=digits))
each = eachindex
function remneg(x) 
    x[x .< 0] .= NaN
    return x
end

function pz(z) # to print the runtime in a readable format
    ss = "$(round(Int,z)) seconds"
    mm = "$(round(z/60,digits=2)) minutes"
    hh = "$(round(z/3600,digits=2)) hours"
    dd = "$(round(z/86400,digits=2)) days"
    if z < 60
        println("Runtime : $ss = $mm")
    elseif z < 3600
        println("Runtime : $mm = $hh")
    else
        println("Runtime : $hh = $dd")
    end
    return z
end

## ------------- Initialisation (main) ------------- ##
## ------------- Initialisation (main) ------------- ##
## ------------- Initialisation (main) ------------- ##
## ------------- Initialisation (main) ------------- ##
create_sigmas(N, R) = CuArray(rand(Float32.([-1, 1]), N, R))

function initialize_system(N, R, p, kappa)
    sigmas = create_sigmas(N, R)
    force = CUDA.zeros(Float32, N, R) # predeclaration for efficiency
    J = create_J(N, R, p, kappa)
    return sigmas, J, force
end 


function create_J(N, R, p, kappa) 
    # Here kappa is a scalar. It's the only place where kappa is provided. In the rest of the simulation, the only important thing is the coupling matrix J  
    @assert p in [2, 3]

    J_symmetric = create_J_symmetric(p, N, R)
    J_antisymmetric = create_J_antisymmetric(p, N, R)
    J = J_symmetric + kappa * J_antisymmetric
    
    normalisation_J = sqrt(factorial(p) / (2 * N^(p - 1) * (1 + kappa ^ 2)))
    weight_sum = 1 / factorial(p - 1)
    J = J * normalisation_J * weight_sum
       
    J_gpu = CuArray(J)
    return J_gpu
end


## ------------- Initialisation (auxiliary) ------------- ##
## ------------- Initialisation (auxiliary) ------------- ##
## ------------- Initialisation (auxiliary) ------------- ##
## ------------- Initialisation (auxiliary) ------------- ##

function create_J_symmetric(p, N, R)
    if p == 2

        J_symmetric = zeros(Float32, N, N, R) # empty matrix to start with, then fill it with the for loop
        for i in 1:N, j in i+1:N # if i = j, do nothing and keep the zeros
            tmp = randn(Float32, R)
            J_symmetric[i, j, :] = tmp
            J_symmetric[j, i, :] = tmp
        end

    elseif p == 3

        J_symmetric = zeros(Float32, N, N, N, R) # empty matrix to start with, then fill it with the for loop
        for i in 1:N, j in i+1:N, k in j+1:N # if any two indices are equal, do nothing and keep the zeros
            tmp = randn(Float32, R)
            J_symmetric[i, j, k, :] = tmp
            J_symmetric[i, k, j, :] = tmp
            J_symmetric[j, i, k, :] = tmp
            J_symmetric[j, k, i, :] = tmp
            J_symmetric[k, i, j, :] = tmp
            J_symmetric[k, j, i, :] = tmp
        end

    else
        error("p should be either 2 or 3 for now")
    end

    return J_symmetric
end

function create_J_antisymmetric(p, N, R)
    if p == 2

        J_antisymmetric = zeros(Float32, N, N, R)
        for i in 1:N, j in i+1:N # if i = j, do nothing and keep the zeros
            tmp = randn(Float32, R)
            J_antisymmetric[i, j, :] = +tmp
            J_antisymmetric[j, i, :] = -tmp
        end

    elseif p == 3

        J_antisymmetric = zeros(Float32, N, N, N, R)
        for i in 1:N, j in i+1:N, k in j+1:N # if any two indices are equal, do nothing and keep the zeros
            tmp = randn(Float32, R)
            J_antisymmetric[i, j, k, :] = +tmp
            J_antisymmetric[i, k, j, :] = -tmp
            J_antisymmetric[j, i, k, :] = -tmp
            J_antisymmetric[j, k, i, :] = +tmp
            J_antisymmetric[k, i, j, :] = +tmp
            J_antisymmetric[k, j, i, :] = -tmp
        end

    else
        error("p should be either 2 or 3 for now")
    end

    return J_antisymmetric
end


## ------------- Compute Forces ------------- ##
## ------------- Compute Forces ------------- ##
## ------------- Compute Forces ------------- ##
## ------------- Compute Forces ------------- ##


function compute_force_p2(force, sigmas, J, N::Int, R::Int, block2D, grid2D)
    @cuda threads = block2D blocks = grid2D kernel_compute_force_p2!(force, sigmas, J, N, R)
    return nothing  # it has modified force in place
end


function kernel_compute_force_p2!(force, sigmas, J, N::Int, R::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    r = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    check = i <= N && r <= R  # we need to check that we are not out of bounds
    if check
        tmp = Float32(0.0)
        for a in 1:N 
            tmp += J[i, a, r] * sigmas[a, r] # J_ia * σ_a
        end 
        force[i, r] = tmp
    end
    return nothing # it has modified force in place
end


function compute_force_p3(force, sigmas, J, N::Int, R::Int, block2D, grid2D)
    @cuda threads = block2D blocks = grid2D kernel_compute_force_p3!(force, sigmas, J, N, R)
    return nothing  # it has modified force in place
end


function kernel_compute_force_p3!(force, sigmas, J, N::Int, R::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    r = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    check = i <= N && r <= R # we need to check that we are not out of bounds
    if check
        tmp = Float32(0.0)

        for b in 1:N, a in 1:N
            tmp += J[i, a, b, r] * sigmas[a, r] * sigmas[b, r] # J_iab * σ_a * σ_b
        end

        force[i, r] = tmp
    end
    return nothing  # it has modified force in place
end


## ------------- Time Evolution ------------- ##
## ------------- Time Evolution ------------- ##
## ------------- Time Evolution ------------- ##
## ------------- Time Evolution ------------- ##

function update_sigmas_p2!(force, sigmas, J, mu, beta, dt, N::Int, R::Int, block2D, grid2D)
    compute_force_p2(force, sigmas, J, N, R, block2D, grid2D) # in-place, modifies force
    @. sigmas += -dt * mu * sigmas + dt * force + sqrt(2dt / beta) * randn()

    return nothing  # it has modified sigmas in place
end 

function update_sigmas_p3!(force, sigmas, J, mu, beta, dt, N::Int, R::Int, block2D, grid2D)

    compute_force_p3(force, sigmas, J, N, R, block2D, grid2D) # in-place, modifies force
    @. sigmas += -dt * mu * sigmas + dt * force + sqrt(2dt/beta)*randn()

    return nothing  # it has modified sigmas in place
end 


function evolve_sigmas!(force, sigmas, J, mu, beta, p, t, dt, tmax, N::Int, R::Int, block2D, grid2D)
    if p == 2 
        while t < tmax
            update_sigmas_p2!(force, sigmas, J, mu, beta, dt, N, R, block2D, grid2D)
            t += dt 
            # println("t = ", t)
        end
    elseif p == 3
        while t < tmax
            update_sigmas_p3!(force, sigmas, J, mu, beta, dt, N, R, block2D, grid2D)
            t += dt 
            println("t = ", t)
        end
    else 
        error("p has to be either 2 or 3 for now. ")
    end
   
    return t 
end




## ----------------- Measurements ----------------- ##
## ----------------- Measurements ----------------- ##
## ----------------- Measurements ----------------- ##
## ----------------- Measurements ----------------- ##

function get_magnetisation(sigmas)
    return mean(sigmas) # average over the spins (N) and the realisations (R)
end

function get_moments(sigmas)
    return mean(sigmas), std(sigmas), skewness(sigmas), kurtosis(sigmas) # over the spins (N) and the realisations (R)
end

function autocorrelation_avg_std(sigmas_new, sigmas_old)
    N, R = size(sigmas_new)
    return mean(sigmas_new .* sigmas_old), std(sigmas_new .* sigmas_old) / N # over the spins (N) and the realisations (R)
end

