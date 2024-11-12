ft = Float32
using LinearAlgebra, CUDA, BenchmarkTools
include("../src/parameters_GPU.jl")
include("../src/methods.jl")
 
kappa = 0.5

# sigmas, J, force = initialize_system(N, R, p, kappas)

##
N = 1000
p = 2
sigmas = create_sigmas(N, R)
force = CUDA.zeros(Float32, N, R) # predeclaration for efficiency

J = create_J(N, R, p, kappa)
update_sigmas_p2!(force, sigmas, J, mu, beta, dt, N, R, block2D, grid2D)
CUDA.@time update_sigmas_p2!(force, sigmas, J, mu, beta, dt, N, R, block2D, grid2D)
# N = 100  143 μs 
# N = 500  374 μs 
# N = 700  491 μs 
# N = 1000 1.9 ms


##
N = 500
p = 3
sigmas = create_sigmas(N, R)
force = CUDA.zeros(Float32, N, R) # predeclaration for efficiency

J = create_J(N, R, p, kappa)
update_sigmas_p3!(force, sigmas, J, mu, beta, dt, N, R, block2D, grid2D)
CUDA.@time update_sigmas_p3!(force, sigmas, J, mu, beta, dt, N, R, block2D, grid2D)
# N = 100  6.9 ms 
# N = 500  0.15 s # maximum N because of my GPU memory allocation limit
# N = 700  ?  # maximum N because of my GPU memory allocation limit ?

