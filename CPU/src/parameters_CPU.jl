
N = 100
p = 2

beta = 1.0
kappa = 0.5
mu = 1.0


dt = 0.01
tmax = 1E2 * dt

ft = Float32
param = Dict(
    "N" => N,
    "p" => p,
    "beta" => ft(beta),
    "kappa" => ft(kappa),
    "mu" => ft(mu),
    "dt" => ft(dt),
    "tmax" => ft(tmax)
)