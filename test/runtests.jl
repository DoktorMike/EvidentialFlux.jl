using EvidentialFlux
using Test

@testset "EvidentialFlux.jl" begin
    # Write your tests here.
    m = NIG(3 => 4)
    x = randn(Float32, 3, 10)
    y = m(x)
    @test size(y) == (16, 10)
    @test y[5:16, :] == abs.(y[5:16, :])
    #@test y[9:12, :] .> 1
end
