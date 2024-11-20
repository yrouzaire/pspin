ft = Float32
using LinearAlgebra, CUDA, BenchmarkTools
include("../src/parameters_GPU.jl")
include("../src/methods_new.jl")
 
kappa = 0.5

#=
Everything here is to be understood for R = 1 (worst case per R).
I compare the old method (kernel, for loops) with the new method (GPU style matrix operations).
=#

##
N = 1000
p = 2
sigmas, J, force = initialize_system(N, R, p, kappa)
update_sigmas_p2!(force, sigmas, J, mu, beta, dt, N, R)#, block2D, grid2D)
CUDA.@time update_sigmas_p2!(force, sigmas, J, mu, beta, dt, N, R)#, block2D, grid2D)
#  N   old (μs)    new  
# 100  143         242 
# 300  218         497
# 500  310         555
# 700  407         710
# 1000 600         1090


##
N = 600
p = 3
sigmas, J, force = initialize_system(N, R, p, kappa)
update_sigmas_p3!(force, sigmas, J, mu, beta, dt, N, R, block2D, grid2D)
CUDA.@time update_sigmas_p3!(force, sigmas, J, mu, beta, dt, N, R, block2D, grid2D)
# N = 50   0.0014 s 
# N = 100  0.0062 s 
# N = 200  0.024  s 
# N = 300  0.054  s 
# N = 500  0.14   s 
# N = 600  0.2    s # sometimes out of memory error

