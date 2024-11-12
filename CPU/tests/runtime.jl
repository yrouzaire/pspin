using BenchmarkTools

## ---------- A single runtime test ---------- ##
## ---------- A single runtime test ---------- ##
## ---------- A single runtime test ---------- ##
## ---------- A single runtime test ---------- ##

include("../src/load_libraries.jl");
include("../src/parameters_CPU.jl"); # exports param, dt, tmax


## ---------- 05/11/2024, First naive implementation ---------- ##
# the O(N^p) runtime complexity is verified. 

param["N"] = 100
param["p"] = 2
system = create_system(param)

@btime update_system!(system, dt)
# N = 100  45.208 μs (600 allocations: 335.94 KiB)
# N = 1000 5.516  ms (9000 allocations: 30.82 MiB)


param["N"] = 100
param["p"] = 3
system = create_system(param)

@btime update_system!(system, dt)
# N = 100  11.633 ms (1900 allocations: 86.16 MiB)
# N = 1000 20.597 s  (29000 allocations: 56.31 GiB)

