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
tmax = 1E5 * dt
number_save = 100
times = logspace(100dt, tmax, number_save, digits=5)
times = tmax / number_save : tmax / number_save : tmax
autocorrelations_avg = zeros(Float32, 2, length(times))
autocorrelations_std = zeros(Float32, 2, length(times))

magnetisation_avg = zeros(Float32, 2, length(times))

sigmas_to_save = zeros(Float32,2, N, R, number_save)

## ---------- Old Methods ---------- ##
## ---------- Old Methods ---------- ##
## ---------- Old Methods ---------- ##
## ---------- Old Methods ---------- ##

include("../src/methods_old.jl")

global t = 0.0
sigmas_old, J_old, force_old = initialize_system(N, R, p, kappa)
update_sigmas_p3!(force_old, sigmas_old, J_old, mu, beta, dt, N, R, block2D, grid2D)
t = evolve_sigmas!(force_old, sigmas_old, J_old, mu, beta, p, t, dt, 0.001, N, R, block2D, grid2D)
t
dt
sigmas_old
z1 = @elapsed for tt in each(times)
    t = evolve_sigmas!(force_old, sigmas_old, J_old, mu, beta, p, t, dt, times[tt], N, R, block2D, grid2D)
    
    a, b = autocorrelation_avg_std(sigmas_old, sigmas_old)
    autocorrelations_avg[1, tt] = a
    autocorrelations_std[1, tt] = b

    magnetisation_avg[1, tt] = mean(sigmas_old)
    sigmas_to_save[1, :, :, tt] = Array(sigmas_old)


    println("t = ", times[tt])
end
pz(z1)



## ---------- New Methods ---------- ##
## ---------- New Methods ---------- ##
## ---------- New Methods ---------- ##
## ---------- New Methods ---------- ##

include("../src/methods_new.jl")

global t = 0.0
sigmas_new, J_new, force_new = initialize_system(N, R, p, kappa)
# sigmas_new = update_sigmas_p3!(force_new, sigmas_new, J_new, mu, beta, dt, N, R)
# t
# sigmas_new
# sigmas_new, t = evolve_sigmas!(force_new, sigmas_new, J_new, mu, beta, p, t, dt, tmax, N, R)

z2 = @elapsed for tt in each(times)
    sigmas_new, t = evolve_sigmas!(force_new, sigmas_new, J_new, mu, beta, p, t, dt, times[tt], N, R)

    a, b = autocorrelation_avg_std(sigmas_new, sigmas_new)
    autocorrelations_avg[2, tt] = a
    autocorrelations_std[2, tt] = b

    magnetisation_avg[2, tt] = mean(sigmas_new)

    sigmas_to_save[2, :, :, tt] = Array(sigmas_new)

    println("t = ", t)
end
pz(z2)

## ---------- Plotting ---------- ##
## ---------- Plotting ---------- ##
## ---------- Plotting ---------- ##
## ---------- Plotting ---------- ##

## explosion and NaN 
ind_r = rand(1:R)
ind_n = rand(1:N)
ind_first_nan_old = findfirst(isnan, sigmas_to_save[1, ind_n, ind_r, :])
ind_first_nan_new = findfirst(isnan, sigmas_to_save[2, ind_n, ind_r, :])
if ind_first_nan_old == nothing
    ind_first_nan_old = length(times)
end
if ind_first_nan_new == nothing
    ind_first_nan_new = length(times)
end
fig = Figure()
ax = Axis(fig[1, 1], title="$(ind_first_nan_old) and $(ind_first_nan_new)", xlabel="Time", ylabel="σ")#, xscale=log10)
scatterlines!(ax, times[1:ind_first_nan_old], sigmas_to_save[1, ind_n, ind_r, 1:ind_first_nan_old], label="Old")
scatterlines!(ax, times[1:ind_first_nan_new], sigmas_to_save[2, ind_n, ind_r, 1:ind_first_nan_new], label="New")
fig


## autocorrelation
fig = Figure()
ax = Axis(fig[1, 1])
scatterlines!(ax, times, autocorrelations_avg[1, :], label="Old")
scatterlines!(ax, times, autocorrelations_avg[2, :], label="New")
fig


##
