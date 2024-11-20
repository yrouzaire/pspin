# include("parameters_GPU.jl")
ft = Float32
using LinearAlgebra, CUDA
 
kappas = ft.([0.5])
beta = ft(1.0)
mu = ft(1.0)
R = 1
N = 10
p = 3

dt = ft(0.1)


wrapsT = 16
block3D = (wrapsT, wrapsT, 1)
grid3D = (Int(ceil(N / wrapsT)), Int(ceil(N / wrapsT)), R)


## here p=2

sigmas = CUDA.rand(Float32, N, length(kappas), R)

J_symmetric = CUDA.rand(Float32, N, N)
J_antisymmetric = CUDA.rand(Float32, N, N)
J_symmetric = J_symmetric - Diagonal(J_symmetric)
J_antisymmetric = J_antisymmetric - Diagonal(J_antisymmetric)

J = J_symmetric + kappas[1]*J_antisymmetric

function compute_force_p2(force, sigmas, J, kappas, N::Int, R::Int, block3D, grid3D)
    @cuda threads = block3D blocks = grid3D kernel_compute_force_p2!(force, sigmas, J, kappas, N, R)
    return nothing
end


function kernel_compute_force_p2!(force, sigmas, J, kappas, N::Int, R::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    normalisation = 1.0
    check = i <= N && j <= length(kappas) && k <= R # if L is not a power of 2, we need to check that we are not out of bounds
    if check
        # force[i, j, k] = J * sigmas[:, j, k]
        tmp = 0
        for a in 1:N 
            tmp += J[i, a] * sigmas[a, j, k]
        end 
        force[i, j, k] = tmp
    end
    return nothing
end

force = CUDA.zeros(Float32, N, length(kappas), R)
compute_force_p2(force, sigmas, J, CuArray(kappas), N, R, block3D, grid3D)
show(err)

normalisation = ft(1.0)

@. sigmas += -dt * mu * sigmas + sqrt(2/beta)*randn() + force * normalisation

## here p=3
sigmas = CUDA.rand(Float32, N, length(kappas), R)

J_symmetric = CUDA.rand(Float32, N, N, N)
J_antisymmetric = CUDA.rand(Float32, N, N, N)

function set_diagonal_to_zero!(A)
    N = size(A, 1)  # A is N x N x N (same size along each dimension)
    @cuda threads = 10 kernel_set_diagonal_to_zero!(A, N)
    return 
end
function kernel_set_diagonal_to_zero!(A, N)
    i = threadIdx().x
    if i <= N
        # A[i, i, i] = 0.0f0
        for j in 1:N
            A[i, i, j] = 0.0f0
            A[i, j, i] = 0.0f0
            A[j, i, i] = 0.0f0
        end
    end
    return nothing
end
J = J_symmetric + kappas[1] * J_antisymmetric
set_diagonal_to_zero!(J)



function compute_force_p3(force, sigmas, J, kappas, N::Int, R::Int, block3D, grid3D)
    @cuda threads = block3D blocks = grid3D kernel_compute_force_p3!(force, sigmas, J, kappas, N, R)
    return nothing
end


function kernel_compute_force_p3!(force, sigmas, J, kappas, N::Int, R::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    check = i <= N && j <= length(kappas) && k <= R # if L is not a power of 2, we need to check that we are not out of bounds
    if check
        tmp = Float32(0.0)
        
        # First implementation
        for a in 1:N, b in 1:N
            tmp += J[i, a, b] * sigmas[a, j, k] * sigmas[b, j, k]
        end

        # # Second implementation
        # for a in 1:N
        #     tmp += sigmas[a, j, k] * (J[:, a, :] * sigmas[:, j, k])
        # end
        
        
        force[i, j, k] = tmp
    end
    return nothing
end


force = CUDA.zeros(Float32, N, length(kappas), R)
compute_force_p3(force, sigmas, J, CuArray(kappas), N, R, block3D, grid3D)
