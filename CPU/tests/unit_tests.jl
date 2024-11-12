using Test
include("../src/methods.jl")

# Test for initialise_system function
@testset "initialise_system tests" begin
    N = 4
    p = 3
    kappa = 0.5

    J, sigmas = initialise_system(N, p, kappa)

    # Test dimensions of J
    @test size(J) == ntuple(_ -> N, p)

    # Test dimensions of sigmas
    @test size(sigmas) == ntuple(_ -> N, p)

    # Test antisymmetry of J_antisymmetric part
    J_symmetric = randn(Float32, ntuple(_ -> N, p))
    J_antisymmetric = randn(Float32, ntuple(_ -> N, p))
    J_antisymmetric = J_antisymmetric - J_antisymmetric'
    J_test = J_symmetric + kappa * J_antisymmetric

    @test J == J_test

    # Test values of sigmas
    @test all(sigmas .== -1 .| sigmas .== 1)
end

