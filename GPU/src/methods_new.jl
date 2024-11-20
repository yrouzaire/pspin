logspace(x1, x2, n; digits=1) = unique!(round.([10.0^y for y in range(log10(x1), log10(x2), length=n)], digits=digits))
each = eachindex
function remneg(x) 
    x[x .< 0] .= NaN
    return x
end

function pz(z) # to print the runtime in a readable format
    ss = "$(round(Int,z)) seconds"
    mm = "$(round(z/60,digits=2)) minutes"
    hh = "$(round(z/3600,digits=2)) hours"
    dd = "$(round(z/86400,digits=2)) days"
    if z < 60
        println("Runtime : $ss = $mm")
    elseif z < 3600
        println("Runtime : $mm = $hh")
    else
        println("Runtime : $hh = $dd")
    end
    return z
end

## ------------- Initialisation (main) ------------- ##
## ------------- Initialisation (main) ------------- ##
## ------------- Initialisation (main) ------------- ##
## ------------- Initialisation (main) ------------- ##
create_sigmas(N, R) = CuArray(rand(Float32.([-1, 1]), N, R))

function initialize_system(N, R, p, kappa)
    sigmas = create_sigmas(N, R)
    force = CUDA.zeros(Float32, N, R) # predeclaration for efficiency
    J = create_J(N, R, p, kappa)
    return sigmas, J, force
end 


function create_J(N, R, p, kappa) 
    # Here kappa is a scalar. It's the only place where kappa is provided. In the rest of the simulation, the only important thing is the coupling matrix J  
    @assert p in [2, 3]

    J_symmetric = create_J_symmetric(p, N, R)
    J_antisymmetric = create_J_antisymmetric(p, N, R)
    J = J_symmetric + kappa * J_antisymmetric
    
    normalisation_J = sqrt(factorial(p) / (2 * N^(p - 1) * (1 + kappa ^ 2)))
    weight_sum = 1 / factorial(p - 1)
    J = J * normalisation_J * weight_sum
       
    J_gpu = CuArray(Float32.(J))
    return J_gpu
end


## ------------- Initialisation (auxiliary) ------------- ##
## ------------- Initialisation (auxiliary) ------------- ##
## ------------- Initialisation (auxiliary) ------------- ##
## ------------- Initialisation (auxiliary) ------------- ##

function create_J_symmetric(p, N, R)
    if p == 2

        J_symmetric = zeros(Float32, N, N, R) # empty matrix to start with, then fill it with the for loop
        for i in 1:N, j in i+1:N # if i = j, do nothing and keep the zeros
            tmp = randn(Float32, R)
            J_symmetric[i, j, :] = tmp
            J_symmetric[j, i, :] = tmp
        end

    elseif p == 3

        J_symmetric = zeros(Float32, N, N, N, R) # empty matrix to start with, then fill it with the for loop
        for i in 1:N, j in i+1:N, k in j+1:N # if any two indices are equal, do nothing and keep the zeros
            tmp = randn(Float32, R)
            J_symmetric[i, j, k, :] = tmp
            J_symmetric[i, k, j, :] = tmp
            J_symmetric[j, i, k, :] = tmp
            J_symmetric[j, k, i, :] = tmp
            J_symmetric[k, i, j, :] = tmp
            J_symmetric[k, j, i, :] = tmp
        end

    else
        error("p should be either 2 or 3 for now")
    end

    return J_symmetric
end

function create_J_antisymmetric(p, N, R)
    if p == 2

        J_antisymmetric = zeros(Float32, N, N, R)
        for i in 1:N, j in i+1:N # if i = j, do nothing and keep the zeros
            tmp = randn(Float32, R)
            J_antisymmetric[i, j, :] = +tmp
            J_antisymmetric[j, i, :] = -tmp
        end

    elseif p == 3

        J_antisymmetric = zeros(Float32, N, N, N, R)
        for i in 1:N, j in i+1:N, k in j+1:N # if any two indices are equal, do nothing and keep the zeros
            tmp = randn(Float32, R)
            J_antisymmetric[i, j, k, :] = +tmp
            J_antisymmetric[i, k, j, :] = -tmp
            J_antisymmetric[j, i, k, :] = -tmp
            J_antisymmetric[j, k, i, :] = +tmp
            J_antisymmetric[k, i, j, :] = +tmp
            J_antisymmetric[k, j, i, :] = -tmp
        end

    else
        error("p should be either 2 or 3 for now")
    end

    return J_antisymmetric
end


## ------------- Time Evolution ------------- ##
## ------------- Time Evolution ------------- ##
## ------------- Time Evolution ------------- ##
## ------------- Time Evolution ------------- ##

function update_sigmas_p2!(force, sigmas, J, mu, beta, dt, N::Int, R::Int)
    #= Comments:  
    size of J = N x N x R 
    size of sigmas = N x R 
    size of force = 1 x N x R 

    The `reshape(sigmas, 1, N, R)` is used to broadcast the spins over the first dimension of the coupling matrix J, 
    without using a for i loop, prohibited or more complex for GPUs.
    In the end, it implements, for all i = 1:N, for all r in 1:R, 
            force_ir = sum_j J_ijr * sigma_jr
    =#

    force = sum(J .* reshape(sigmas, 1, N, R), dims=(2))[:, 1, :]    
    thermal_noise = sqrt(2dt / beta) * CUDA.randn(Float32, N, R) # different noise for each spin, also different for each realisation
    sigmas += -dt * mu * sigmas + dt * force + thermal_noise

    return sigmas  
end 


function update_sigmas_p3!(force, sigmas, J, mu, beta, dt, N::Int, R::Int)
    #= Comments:  
    size of J = N x N x N x R 
    size of sigmas = N x N x R 
    size of force = 1 x N x R 

    The `reshape(sigmas, 1, N, R)` is used to broadcast the spins over the first dimension of the coupling matrix J, 
    without using for loops over i and r (prohibited or more complex for GPUs).
    In the end, it implements, for all i = 1:N, for all r in 1:R, 
        force_ir = sum_{j,k} J_ijkr * sigma_jr * sigma_kr
    
    =#
    sigmul = reshape(sigmas, 1, N, R) .* reshape(sigmas, N, 1, R) # 3d array of size N x N x R /// σ_mul_ijr = σ_ir * σ_jr
    sigmul = reshape(sigmul, 1, N, N, R) # size = 1 x N x N x R to fit the size of J, to be able to multiply element-wise
    
    force = sum(J .* sigmul, dims=(2, 3))[:, 1, 1, :]
    
    # thermal_noise = sqrt(2dt / beta) * CUDA.randn(Float32, N, R) # different noise for each spin, also different for each realisation
    thermal_noise = sqrt(2dt / beta) * repeat(CUDA.randn(Float32, N),1, R) # different noise for each spin, but SAME for each realisation
    sigmas += -dt * mu * sigmas + dt * force + thermal_noise
        

    return sigmas  
end 



function evolve_sigmas!(force, sigmas, J, mu, beta, p, t, dt, tmax, N::Int, R::Int)
    if p == 2 
        while t < tmax
           sigmas = update_sigmas_p2!(force, sigmas, J, mu, beta, dt, N, R)
            t += dt
        end
    elseif p == 3
        while t < tmax
            sigmas = update_sigmas_p3!(force, sigmas, J, mu, beta, dt, N, R)
            t += dt
            # println("t = ", t, " tmax = ", tmax)
        end
    else 
        error("p has to be either 2 or 3 for now. ")
    end

    return sigmas, t
end




## ----------------- Measurements ----------------- ##
## ----------------- Measurements ----------------- ##
## ----------------- Measurements ----------------- ##
## ----------------- Measurements ----------------- ##

function get_magnetisation(sigmas)
    return mean(sigmas) # average over the spins (N) and the realisations (R)
end

function get_moments(sigmas)
    return mean(sigmas), std(sigmas), skewness(sigmas), kurtosis(sigmas) # over the spins (N) and the realisations (R)
end

function autocorrelation_avg_std(sigmas_new, sigmas_old)
    N,R = size(sigmas_new)
    return sum(mean(mean(sigmas_new .* sigmas_old, dims=1), dims=2)), sum(std(mean(sigmas_new .* sigmas_old, dims=1), dims=2)) # the sum is here to tranform the 1x1 array into a scalar. Results are the same than the mean() function in the line below
    # return mean(sigmas_new .* sigmas_old), std(sigmas_new .* sigmas_old) / N # over the spins (N) and the realisations (R)
end
