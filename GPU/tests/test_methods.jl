ft = Float32
using LinearAlgebra, CUDA, CairoMakie
CairoMakie.activate!()
include("../src/parameters_GPU.jl")
include("../src/methods.jl")
 

##
beta = 1


fig = Figure()

ax = Axis(fig[1, 1])
kappas = CuArray(ft.([0.5]))

sigmas = create_sigmas(N, R, kappas)
CairoMakie.hist!(ax, Array(sigmas[:, 1, 1]))
force = CUDA.zeros(Float32, N, length(kappas), R) # predeclaration for efficiency
J = create_J(N, R, p, kappas)

update_sigmas_p2!(force, sigmas, J, kappas, mu, beta, dt, N, R, block3D, grid3D)

sigmas
CairoMakie.hist!(ax, Array(sigmas[:, 1, 1]))
fig


##
beta = 1
p=3

fig = Figure()

ax = Axis(fig[1, 1])
kappas = CuArray(ft.([0.5]))

sigmas = create_sigmas(N, R, kappas)
CairoMakie.hist!(ax, Array(sigmas[:, 1, 1]))
force = CUDA.zeros(Float32, N, length(kappas), R) # predeclaration for efficiency
J = create_J(N, R, p, kappas)

for i in 1:30
    update_sigmas_p3!(force, sigmas, J, kappas, mu, beta, dt, N, R, block3D, grid3D)
end
sigmas
CairoMakie.hist!(ax, Array(sigmas[:, 1, 1]))
fig