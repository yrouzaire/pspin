ft = Float32
using LinearAlgebra, CUDA, StatsBase, Random
using CairoMakie
CairoMakie.activate!()
custom_theme = Theme(
    Axis=(
        xgridvisible=false, ygridvisible=false,
        xtickalign=1, ytickalign=1,
        xticklabelpad=2, yticklabelpad=4
    ),
    fontsize=24,
    # size=(450, 400),
)
set_theme!(custom_theme)

include(pwd()*"/GPU/src/methods_new.jl")


##
#=

Some preliminary notes: 

This file is a template that :
    - creates the system
    - evolves the system in time
    - performs measurements on the system at different times (here times are logspaced and the measurements are magnetisation and autocorrelation)
    - does this for multiple kappas
    - does this for p=2 and p=3
    - plots the results

The system is defined by the parameters N, R, p, beta, kappa, mu, dt, tmax
All the functions are defined in the file `methods.jl`.
All the parameters are defined in the file `parameters_GPU.jl`.

Regarding kappa : 
    Each system is defined for a simgle scalar kappa. 
    Kappa is only needed to create the coupling matrix J. 
    Once J is created, it contains all the kappa related information needed 
        for the simulation (in the variance of the matrix), and kappa is not used anymore. 
    You thus need to loop over the different kappas independently, creating a new system for each kappa.

Regarding R : 
    In constrast, R (the number of independent realisations, defined in `parameters_GPU.jl`) is intrinsically parallized.
    Each system for will have 
        (for p = 2)
        sigmas : N x R
        force : N x R
        J : N x N x R

        (for p = 3)
        sigmas : N x R
        force : N x R
        J : N x N x N x R

    This is to take advantage of the inherent GPU parallelization. 
    I assumed that you would more often need to run multiple realisations in parallel than multiple kappas in parallel.
    Note that the limiting factor for p=3 is the GPU memory allocation of the coupling matrix J (size =  N x N x N x R), so increasing R will decrease the maximum N you can use.
=#

## Load the common parameters and define custom params

include(pwd()*"/GPU/src/parameters_GPU.jl")
kappas = [0]
ps = [3]
beta = 1.05
N = 50
R = 16
mu = (1/beta + 3beta/2)

dt = 1E-4
# tmax = 1E3 * dt
tmax = 2
number_save = 50
times = logspace(1dt, tmax, number_save, digits=5)

# Define the structures to store the results
# magnetisation_avg = zeros(Float32, length(ps), length(kappas), length(times))
# magnetisation_std = zeros(Float32, length(ps), length(kappas), length(times))
# sigmas_save = zeros(Float32, N, R, length(times))

autocorrelations_avg = zeros(Float32, length(ps), length(kappas), length(times))
autocorrelations_std = zeros(Float32, length(ps), length(kappas), length(times))

# # Loop over the different p and κ
CUDA.@time z = @elapsed for i in each(ps), j in each(kappas)
   
    Random.seed!(1) # seed for the initial configuration, created on GPU (generated using the library "Random")
    CUDA.seed!(2) # seed for the thermal noise, (generated using the library "CUDA")
    # checked that it does not make the shoulder appear

    p = ps[i]
    kappa = kappas[j]
    sigmas, J, force = initialize_system(N, R, p, kappa)
    sigmas_t0 = copy(sigmas)

    println("p = ", p, " ; κ = ", kappa, " ; N = ", N, " ; R = ", R, " ; μ = ", mu)
    
    ## Loop over the different times
    global t = 0.0 # global variable so that it can be updated in the function evolve_sigmas!
    for tt in each(times)
        sigmas, t = evolve_sigmas!(force, sigmas, J, mu, beta, p, t, dt, times[tt], N, R)#, block2D, grid2D) # evolve the system from t=times[tt-1] to the next time t=times[tt]
        
        # magnetisation_avg[i, j, tt] = mean(sigmas) # mean over the realisations (R) AND the spins (N)
        # magnetisation_std[i, j, tt] = std(sigmas)  # std  over the realisations (R) AND the spins (N)
        # sigmas_save[:, :, tt] = Array(sigmas)
        
        a, b = autocorrelation_avg_std(sigmas, sigmas_t0) # function defined at the end of the file methods_new.jl
        autocorrelations_avg[i, j, tt] = a
        autocorrelations_std[i, j, tt] = b
        
        println("t = ", t)
    end
    println() # empty line to separate the different kappas
end
pz(z) # print the runtime in a readable format



## Plot the results 
title_ = "p=3, N = $N, R = $R, β = $beta, 
μ = $(round(mu, digits=2)), dt = $dt, tmax = $tmax"

fig = Figure()
ax_p2 = Axis(fig[1, 1], xlabel="Time", ylabel="C(t)", 
    xscale=log10,  
    title=title_, 
    limits=(nothing, nothing, 0,1.05 )
    )
for j in each(kappas)
    scatterlines!(ax_p2, times[1:ind_nan], remneg(autocorrelations_avg[1, 1, 1:ind_nan]), label="κ = $(kappas[j])")
    # band!(ax_p2, times, autocorrelations_avg[1, j, :] .- autocorrelations_std[1, j, :], autocorrelations_avg[1, j, :] .+ autocorrelations_std[1, j, :], alpha=0.3)
end

resize_to_layout!(fig)
fig


