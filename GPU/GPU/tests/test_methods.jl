ft = Float32
using StatsBase, LinearAlgebra, CUDA, CairoMakie
CairoMakie.activate!()
custom_theme = Theme(
    Axis=(
        xgridvisible=false, ygridvisible=false,
        xtickalign=1, ytickalign=1,
        xticklabelpad=2, yticklabelpad=4
    ),
    fontsize=24,
    size=(450, 400),
)
set_theme!(custom_theme)

include("../src/parameters_GPU.jl")
 

## Common Parameters
N = 30
p = 3
R = 1
kappa = 0
beta = 1
mu = (1 / beta + 3beta / 2)

dt = 1E-5
tmax = 1E3 * dt
number_save = 10
# times = logspace(100dt, tmax, number_save, digits=5)
times = tmax / number_save : tmax / number_save : tmax


autocorrelations_avg = zeros(Float32, 2, length(times))
autocorrelations_std = zeros(Float32, 2, length(times))

magnetisation_avg = zeros(Float32, 2, length(times))

sigmas_to_save = zeros(Float32,2, N, R, number_save)


## ---------- New Methods ---------- ##
## ---------- New Methods ---------- ##
## ---------- New Methods ---------- ##
## ---------- New Methods ---------- ##

include("../src/methods_new.jl")

global t = 0.0
sigmas_new, J_new, force_new = initialize_system(N, R, p, kappa)
sigmas_new0 = copy(sigmas_new)
CUDA.seed!(1)
sigmas_new = update_sigmas_p3!(force_new, sigmas_new, J_new, mu, beta, dt, N, R)
my_tmax = 0.002
dt 
sigmas_new, t = evolve_sigmas!(force_new, sigmas_new, J_new, mu, beta, 3, t, dt, my_tmax, N, R)

## ---------- Old Methods ---------- ##
## ---------- Old Methods ---------- ##
## ---------- Old Methods ---------- ##
## ---------- Old Methods ---------- ##

# include("../src/methods_old.jl")
