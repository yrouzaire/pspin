
N = 100
R = 8
p = 2

beta = 1.0
kappa = 0.5
mu = 10.0


dt = 0.01
tmax = 1E1 * dt

wrapsT = 16
block2D = (wrapsT, 1)
grid2D = (Int(ceil(N / wrapsT)), R)

@assert p in [2, 3]
@assert N < 1001

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