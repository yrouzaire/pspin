ft = Float32
using LinearAlgebra, CUDA, StatsBase 
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

include(pwd()*"/GPU/src/methods.jl")
 

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
kappas = [0.]
ps = [2]


dt = 0.01
tmax = 1E3 * dt
number_save = 30
times = logspace(10dt, tmax, number_save, digits=2)

number_waiting_save = 3
# waiting_times = logspace(100dt, tmax, number_waiting_save, digits=2) # it might be an option but times and waiting_times do not coincide.
waiting_times = times[end-number_waiting_save+1:end] # better for efficiency to have all waiting_times ∈ times, here the last 'number_waiting_save'

# Define the structures to store the results
magnetisation_avg = zeros(Float32, length(ps), length(kappas), length(times))
magnetisation_std = zeros(Float32, length(ps), length(kappas), length(times))

sigmas_waiting = zeros(Float32, N, R, length(waiting_times))
autocorrelations_avg = zeros(Float32, length(ps), length(kappas), length(waiting_times), length(times))
autocorrelations_std = zeros(Float32, length(ps), length(kappas), length(waiting_times), length(times))

# # Loop over the different kappas

CUDA.@time z = @elapsed for i in each(ps), j in each(kappas)
    p = ps[i]
    kappa = kappas[j]
    sigmas, J, force = initialize_system(N, R, p, kappa)
    tmp =  CUDA.zeros(Float32, N, R)

    println("p = ", p, " ; κ = ", kappa)
    
    ## Loop over the different times
    global t = 0.0
    for tt in each(times)
        println("t = ", times[max(1, tt)])
        evolve_sigmas!(force, sigmas, J, mu, beta, p, t, dt, times[tt], N, R, block2D, grid2D) # evolve the system from t=times[tt-1] to the next time t=times[tt]
        magnetisation_avg[i, j, tt] = mean(sigmas) # mean over the realisations (R) AND the spins (N)
        magnetisation_std[i, j, tt] = std(sigmas)  # std  over the realisations (R) AND the spins (N)
        
        for ttwait in each(waiting_times)
            if times[tt] ≤ waiting_times[ttwait] # if we are at a waiting time
                sigmas_old = sigmas_waiting[:,:,ttwait] 
                a, b = autocorrelation_avg_std(sigmas, sigmas_old) 
                autocorrelations_avg[i, j, ttwait, tt] = a 
                autocorrelations_std[i, j, ttwait, tt] = b 
            end
        end
        
    end
    println() # empty line to separate the different kappas
end
pz(z) # print the runtime in a readable format

## Plot the results 
fig = Figure()
ax_p2 = Axis(fig[1, 1], xlabel="Time", ylabel="Magnetisation", xscale=log10, title="p=2")
for j in each(kappas)
    scatterlines!(ax_p2, times, magnetisation_avg[1, j, :], label="κ = $(kappas[j])")
    band!(ax_p2, times, magnetisation_avg[1, j, :] .- magnetisation_std[1, j, :], magnetisation_avg[1, j, :] .+ magnetisation_std[1, j, :], alpha=0.3)
end
# ax_p3 = Axis(fig[1, 2], xlabel="Time", ylabel="Magnetisation", xscale=log10, title="p=3")
# for j in each(kappas)
#     scatterlines!(ax_p3, times, magnetisation_avg[2, j, :], label="κ = $(kappas[j])")
# end
resize_to_layout!(fig)
fig


##
