using EvidentialFlux
using Flux
using Test

@testset "EvidentialFlux.jl - Classification" begin
    # Creating a model and a forward pass

    ninp, nout = 3, 5
    m = DIR(ninp => nout)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (5, 10)
    # The α all have to be ≥ 1
    @test all(≥(1), ŷ)
end

@testset "EvidentialFlux.jl - NIG Regression" begin
    # Creating a model and a forward pass

    ninp, nout = 3, 5
    m = NIG(ninp => nout)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (20, 10)
    # The ν, α, and β all have to be positive
    @test ŷ[6:20, :] == abs.(ŷ[6:20, :])
    # The α all have to be ≥ 1
    @test all(≥(1), ŷ[11:15, :])

    # Testing convergence
    ninp, nout = 3, 1
    model = NIG(ninp => nout)
    x = Float32.(collect(1:0.1:10))
    x = cat(x', x' .- 10, x' .+ 5, dims = 1)
    y = 1 * x[1, :] .- 3 * x[2, :] .+ 2 * x[3, :] .+ randn(Float32, 91)
    opt_state = Flux.setup(Flux.Adam(0.005), model)  # will store optimiser momentum, etc.

    # Training loop, using the whole data set 1000 times:
    losses = []
    for epoch in 1:1_000
        # Unpack batch of data, and move to GPU:
        loss, grads = Flux.withgradient(model) do m
            ŷ = m(x)
            γ, ν, α, β = ŷ[1, :], ŷ[2, :], ŷ[3, :], ŷ[4, :]
            sum(nigloss(y, γ, ν, α, β, 0.1, 1.0e-4))
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
    @test losses[1] > losses[10] > losses[100] > losses[300]

    # Test the nigloss and uncertainty function
    ninp, nout = 3, 5
    m = NIG(ninp => nout)
    x = randn(Float32, 3, 10)
    y = randn(Float32, nout, 10) # Target (fake)
    ŷ = m(x)
    γ = ŷ[1:nout, :]
    ν = ŷ[(nout + 1):(nout * 2), :]
    α = ŷ[(nout * 2 + 1):(nout * 3), :]
    β = ŷ[(nout * 3 + 1):(nout * 4), :]
    myloss = nigloss(y, γ, ν, α, β, 0.1, 1.0e-4)
    @test size(myloss) == (nout, 10)
    myuncert = uncertainty(ν, α, β)
    @test size(myuncert) == size(myloss)
end

@testset "EvidentialFlux.jl - MVE Regression" begin
    # Creating a model and a forward pass

    ninp, nout = 3, 5
    m = MVE(ninp => nout)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (2 * nout, 10)
    # The σ all have to be positive
    @test ŷ[6:10, :] == abs.(ŷ[6:10, :])

end
