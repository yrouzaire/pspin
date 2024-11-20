
N = 50
R = 1
p = 3

beta = 1.
kappa = 0
mu = (1/beta + 3beta/2)


dt = 0.01
tmax = 1E1 * dt

wrapsT = 16
block2D = (wrapsT, 1)
grid2D = (Int(ceil(N / wrapsT)), R)
block3D = (wrapsT, wrapsT, 1)
grid3D = (Int(ceil(N / wrapsT)), Int(ceil(N / wrapsT)), R)


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