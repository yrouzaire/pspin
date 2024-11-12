
## ----------------- Definitions and Initialisation ----------------- ##
## ----------------- Definitions and Initialisation ----------------- ##
## ----------------- Definitions and Initialisation ----------------- ##
## ----------------- Definitions and Initialisation ----------------- ##


abstract type pSpin{Float32} end

mutable struct p2Spin{Float32} <: pSpin{Float32} 
    beta::Float32
    p::Int
    N::Int
    kappa::Float32
    mu::Float32
    J::Array{Float32,2}
    sigmas::Vector{Float32}

    t::Float32
end

mutable struct p3Spin{Float32} <: pSpin{Float32} 
    beta::Float32
    p::Int
    N::Int
    kappa::Float32
    mu::Float32
    J::Array{Float32, 3}
    sigmas::Vector{Float32}
    
    t::Float32
end


function create_system(params)
    @unpack N, p, kappa, beta, mu, dt, tmax = params

    # J_symmetric = randn(Float32, ntuple(_ -> N, p)) # coupling matrix N x N x ... x N (p times)
    # enforce_symmetry!(J_symmetric, p)
    # J_antisymmetric = randn(Float32, ntuple(_ -> N, p)) # coupling matrix N x N x ... x N (p times)
    # enforce_antisymmetry!(J_antisymmetric, p)
    # J = 0*J_symmetric + kappa * J_antisymmetric

    J = randn(Float32, ntuple(_ -> N, p))

    sigmas = rand(Float32.([-1, +1]), N)

    time0 = 0.0
    if p == 2
        system = p2Spin{Float32}(beta, p, N, kappa, mu, J, sigmas, time0)
    elseif p == 3
        system = p3Spin{Float32}(beta, p, N, kappa, mu, J, sigmas, time0)
    else
        error("p must be 2 or 3 for now")
    end

    return system
end 

   
# function enforce_symmetry!(J_symmetric, p)
#     if p == 2
#         for i in 1:size(J_symmetric, 1)
#             for j in 1:size(J_symmetric, 2)
#                 J_symmetric[i, j] = J_symmetric[j, i]
#             end
#         end
#     end
# end

# function enforce_antisymmetry!(J_antisymmetric, p)
#     if p == 2
#         for i in 1:size(J_antisymmetric, 1)
#             for j in 1:size(J_antisymmetric, 2)
#                 J_antisymmetric[i, j] = -J_antisymmetric[j, i]
#             end
#         end
#     end
# end


## ----------------- Time Evolution ----------------- ##
## ----------------- Time Evolution ----------------- ##
## ----------------- Time Evolution ----------------- ##
## ----------------- Time Evolution ----------------- ##


function evolve_system!(system::pSpin, dt, tmax)
    while system.t < tmax
        update_system!(system, dt)
        system.t += dt
    end
end

function update_system!(system::p2Spin, dt) # one timestep
    sigmas, J, p, beta, mu, N = system.sigmas, system.J, system.p, system.beta, system.mu, system.N
    for i in 1:N
        neighbours_indices, indices_with_i = list_neighbours_of_i(i, N, p)
        force = 0 
        for i in 1:length(neighbours_indices)
            force += J[indices_with_i[i]] * sigmas[neighbours_indices[i]] 
        end
        
        sigmas[i] += -dt * mu * sigmas[i] + dt * force + sqrt(2 * dt / beta) * randn()
    end
end


function update_system!(system::p3Spin, dt) # one timestep
    sigmas, J, p, beta, mu, N = system.sigmas, system.J, system.p, system.beta, system.mu, system.N
    for i in 1:N
        neighbours_indices, indices_with_i = list_neighbours_of_i(i, N, p)
        force = 0 
        for i in 1:length(neighbours_indices)
            force += J[indices_with_i[i]] * sigmas[neighbours_indices[i][1]] * sigmas[neighbours_indices[i][2]]
        end
        
        sigmas[i] += -dt * mu * sigmas[i] + dt * force + sqrt(2 * dt / beta) * randn()
    end
end

function list_neighbours_of_i(i, N, p)
    if p == 2
        neighbours_indices = filter(j -> i != j, 1:N)
        indices_with_i = [CartesianIndex(i, j) for j in neighbours_indices]
    elseif p == 3
        neighbours_indices = [(j, k) for j in 1:N, k in 1:N if i != j != k]
        indices_with_i = [CartesianIndex(i, j, k) for (j, k) in neighbours_indices]
    else
        error("p must be 2 or 3 for now") 
    end
    return neighbours_indices, indices_with_i
end

## ----------------- Measurements ----------------- ##
## ----------------- Measurements ----------------- ##
## ----------------- Measurements ----------------- ##
## ----------------- Measurements ----------------- ##

function magnetization(system::pSpin)
    return mean(system.sigmas)
end

function moments(system::pSpin)
    return mean(system.sigmas), std(system.sigmas), skewness(system.sigmas), kurtosis(system.sigmas)
end 

