#= 
© Original Author : Ylann Rouzaire, 2024, rouzaire.ylann@gmail.com   

This is a template for a script that runs a simulation on the cluster.
To run the bash .sh script, you simply need to run the command 
sbatch  --time=23:59:00 bash_script_superfe.sh

Run squeue to see the status of all the running jobs on the server.
Note : A/I/O/T means Active / Idle=available / Offline / Total

=# 


ft = Float32
using LinearAlgebra, CUDA, StatsBase, Random, JLD2
include("methods_new.jl")

## ---------- Parameters ---------- ##
## ---------- Parameters ---------- ##
## ---------- Parameters ---------- ##
## ---------- Parameters ---------- ##

N = 50
R = 1
ps = [3]

beta = 1.
kappas = [0]
mu = (1/beta + 3beta/2)

dt = 1E-2
tmax = 1E1 * dt
n_save = 10
times = tmax / n_save : tmax / n_save : tmax # linear time 
times = logspace(10dt, tmax, n_save, digits=5) # logarithmic time

## ---------- Definitions ---------- ##
sigmas_save = zeros(Float32, N, R, length(times))

autocorrelations_avg = zeros(Float32, length(ps), length(kappas), length(times))
autocorrelations_std = zeros(Float32, length(ps), length(kappas), length(times))



## ---------- Simulation ---------- ##
## ---------- Simulation ---------- ##
## ---------- Simulation ---------- ##
## ---------- Simulation ---------- ##

CUDA.@time z = @elapsed for i in each(ps), j in each(kappas)

    # Random.seed!(1) # seed for the initial configuration, created on GPU (generated using the library "Random")
    # CUDA.seed!(2) # seed for the thermal noise, (generated using the library "CUDA")
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

        # sigmas_save[:, :, tt] = Array(sigmas)

        a, b = autocorrelation_avg_std(sigmas, sigmas_t0) # function defined at the end of the file methods_new.jl
        autocorrelations_avg[i, j, tt] = a
        autocorrelations_std[i, j, tt] = b

        println("t = ", round(t, digits=4))
    end
    println() # empty line to separate the different kappas
end
pz(z) # print the runtime in a readable format


## ---------- Save Data ---------- ##
## ---------- Save Data ---------- ##
## ---------- Save Data ---------- ##
## ---------- Save Data ---------- ##
comments = "" # add comments here
filename_base = "shoulder_p3_N$(N)_R$(R)_beta$(beta)_kappa$(kappa)" # do not add data/ at the beginning, it's being added automatically by the .sh script
@save filename_base * ".jld2" autocorrelations_avg autocorrelations_std sigmas_save times N R ps beta kappas mu dt tmax n_save comments runtime = z


