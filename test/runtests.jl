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

@testset "EvidentialFlux.jl - Regression" begin
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

    # Testing backward pass
    oldW = similar(m.W)
    oldW .= m.W
    loss(y, ŷ) = sum(abs, y - ŷ)
    pars = Flux.params(m)
    y = randn(Float32, nout, 10) # Target (fake)
    grads = Flux.gradient(pars) do
        ŷ = m(x)
        γ = ŷ[1:nout, :]
        loss(y, γ)
    end
    # Test that we can update the weights based on gradients
    opt = Descent(0.1)
    Flux.Optimise.update!(opt, pars, grads)
    @test m.W != oldW

    # Testing convergence
    ninp, nout = 3, 1
    m = NIG(ninp => nout)
    x = Float32.(collect(1:0.1:10))
    x = cat(x', x' .- 10, x' .+ 5, dims = 1)
    # y = 1 * sin.(x[1, :]) .- 3 * sin.(x[2, :]) .+ 2 * cos.(x[3, :]) .+ randn(Float32, 91)
    y = 1 * x[1, :] .- 3 * x[2, :] .+ 2 * x[3, :] .+ randn(Float32, 91)
    #scatterplot(x[1, :], y, width = 90, height = 30)
    pars = Flux.params(m)
    opt = AdamW(0.005)
    trnlosses = zeros(Float32, 1000)
    for i in 1:1000
        local trnloss
        grads = Flux.gradient(pars) do
            ŷ = m(x)
            γ, ν, α, β = ŷ[1, :], ŷ[2, :], ŷ[3, :], ŷ[4, :]
            trnloss = sum(nigloss(y, γ, ν, α, β, 0.1, 1e-4))
        end
        trnlosses[i] = trnloss
        # Test that we can update the weights based on gradients
        Flux.Optimise.update!(opt, pars, grads)
        #if i % 100 == 0 
        #	println("Epoch $i, Loss: $trnloss")
        #end
    end
    #scatterplot(1:1000, trnlosses, width = 80)
    @test trnlosses[10] > trnlosses[100] > trnlosses[300]

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
    myloss = nigloss(y, γ, ν, α, β, 0.1, 1e-4)
    @test size(myloss) == (nout, 10)
    myuncert = uncertainty(ν, α, β)
    @test size(myuncert) == size(myloss)
end
