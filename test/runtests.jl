using EvidentialFlux
using Flux
using Test

@testset "AbstractEvidentialLayer" begin
    @test NIG <: AbstractEvidentialLayer
    @test DIR <: AbstractEvidentialLayer
    @test MVE <: AbstractEvidentialLayer
    @test FDIR <: AbstractEvidentialLayer
    @test PG <: AbstractEvidentialLayer
    @test EG <: AbstractEvidentialLayer
    @test BB <: AbstractEvidentialLayer
    @test BNB <: AbstractEvidentialLayer
    @test ZIP <: AbstractEvidentialLayer
    @test VM <: AbstractEvidentialLayer
end

@testset "EvidentialFlux.jl - Classification" begin
    ninp, nout = 3, 5
    m = DIR(ninp => nout)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (5, 10)
    @test all(≥(1), ŷ)
end

@testset "EvidentialFlux.jl - NIG Regression" begin
    # Forward pass shape and constraints
    ninp, nout = 3, 5
    m = NIG(ninp => nout)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (20, 10)
    @test ŷ[6:20, :] == abs.(ŷ[6:20, :])
    @test all(≥(1), ŷ[11:15, :])

    # Convergence test
    ninp, nout = 3, 1
    model = NIG(ninp => nout)
    x = Float32.(collect(1:0.1:10))
    x = cat(x', x' .- 10, x' .+ 5, dims = 1)
    y = reshape(1 * x[1, :] .- 3 * x[2, :] .+ 2 * x[3, :] .+ randn(Float32, 91), 1, :)
    opt_state = Flux.setup(Flux.Adam(0.005), model)

    losses = []
    for epoch in 1:1_000
        loss, grads = Flux.withgradient(model) do m
            γ, ν, α, β = splitnig(m(x))
            sum(nigloss(y, γ, ν, α, β, 0.1, 1.0e-4))
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)
    end
    @test losses[1] > losses[10] > losses[100] > losses[300]

    # Loss and uncertainty shapes
    ninp, nout = 3, 5
    m = NIG(ninp => nout)
    x = randn(Float32, 3, 10)
    y = randn(Float32, nout, 10)
    γ, ν, α, β = splitnig(m(x))
    myloss = nigloss(y, γ, ν, α, β, 0.1, 1.0e-4)
    @test size(myloss) == (nout, 10)
    myuncert = uncertainty(ν, α, β)
    @test size(myuncert) == size(myloss)
end

@testset "EvidentialFlux.jl - MVE Regression" begin
    ninp, nout = 3, 5
    m = MVE(ninp => nout)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (2 * nout, 10)
    @test ŷ[6:10, :] == abs.(ŷ[6:10, :])
end

@testset "EvidentialFlux.jl - PG Count Regression" begin
    ninp, nout = 3, 5
    m = PG(ninp => nout)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (2 * nout, 10)
    α, β = splitpg(ŷ)
    @test all(>(0), α)
    @test all(>(0), β)
    @test size(α) == (nout, 10)
    @test size(β) == (nout, 10)
end

@testset "EvidentialFlux.jl - EG Positive Regression" begin
    ninp, nout = 3, 5
    m = EG(ninp => nout)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (2 * nout, 10)
    α, β = spliteg(ŷ)
    @test all(>(0), α)
    @test all(>(0), β)
    @test size(α) == (nout, 10)
    @test size(β) == (nout, 10)
end

@testset "EvidentialFlux.jl - BB Proportion Estimation" begin
    ninp, nout = 3, 5
    m = BB(ninp => nout)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (2 * nout, 10)
    α, β = splitbb(ŷ)
    @test all(>(0), α)
    @test all(>(0), β)
    @test size(α) == (nout, 10)
    # predicted probability is in (0, 1)
    p̂ = α ./ (α .+ β)
    @test all(>(0), p̂)
    @test all(<(1), p̂)
end

@testset "EvidentialFlux.jl - BNB Count Regression" begin
    ninp, nout = 3, 5
    m = BNB(ninp => nout)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (3 * nout, 10)
    r, α, β = splitbnb(ŷ)
    @test all(>(0), r)
    @test all(>(0), α)
    @test all(>(0), β)
    @test size(r) == (nout, 10)
end

@testset "EvidentialFlux.jl - ZIP Count Regression" begin
    ninp, nout = 3, 5
    m = ZIP(ninp => nout)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (4 * nout, 10)
    α_π, β_π, α_λ, β_λ = splitzip(ŷ)
    @test all(>(0), α_π)
    @test all(>(0), β_π)
    @test all(>(0), α_λ)
    @test all(>(0), β_λ)
    @test size(α_π) == (nout, 10)
end

@testset "EvidentialFlux.jl - VM Directional Regression" begin
    ninp, nout = 3, 5
    m = VM(ninp => nout)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (3 * nout, 10)
    μ₀, κ₀, κ = splitvm(ŷ)
    @test all(>(0), κ₀)
    @test all(>(0), κ)
    @test size(μ₀) == (nout, 10)
    @test size(κ₀) == (nout, 10)
end

@testset "EvidentialFlux.jl - FDIR Classification" begin
    ninp, nclasses = 3, 5
    m = FDIR(ninp => nclasses)
    x = randn(Float32, 3, 10)
    ŷ = m(x)
    @test size(ŷ) == (2 * nclasses + 1, 10)
    α, p, τ = splitfdir(ŷ)
    # α > 0 (exp activation)
    @test all(>(0), α)
    # p sums to 1 per sample (softmax)
    @test all(isapprox.(sum(p, dims = 1), 1, atol = 1.0f-5))
    # τ > 0 (softplus activation)
    @test all(>(0), τ)
    @test size(α) == (nclasses, 10)
    @test size(p) == (nclasses, 10)
    @test size(τ) == (1, 10)
end

@testset "splitnig" begin
    nout, batch = 3, 5
    γ = ones(Float32, nout, batch)
    ν = 2 * ones(Float32, nout, batch)
    α = 3 * ones(Float32, nout, batch)
    β = 4 * ones(Float32, nout, batch)
    y = vcat(γ, ν, α, β)
    γ2, ν2, α2, β2 = splitnig(y)
    @test γ2 == γ
    @test ν2 == ν
    @test α2 == α
    @test β2 == β
end

@testset "split_params" begin
    nout, batch = 3, 5

    # NIG split_params returns NamedTuple
    γ = ones(Float32, nout, batch)
    ν = 2 * ones(Float32, nout, batch)
    α = 3 * ones(Float32, nout, batch)
    β = 4 * ones(Float32, nout, batch)
    y_nig = vcat(γ, ν, α, β)
    p = split_params(NIG, y_nig)
    @test p isa NamedTuple{(:γ, :ν, :α, :β)}
    @test p.γ == γ
    @test p.ν == ν
    @test p.α == α
    @test p.β == β
    # round-trip: vcat of parts equals original
    @test vcat(p.γ, p.ν, p.α, p.β) == y_nig

    # PG split_params returns NamedTuple
    α_pg = 2 * ones(Float32, nout, batch)
    β_pg = 3 * ones(Float32, nout, batch)
    y_pg = vcat(α_pg, β_pg)
    q_pg = split_params(PG, y_pg)
    @test q_pg isa NamedTuple{(:α, :β)}
    @test q_pg.α == α_pg
    @test q_pg.β == β_pg
    @test vcat(q_pg.α, q_pg.β) == y_pg

    # EG split_params returns NamedTuple
    α_eg = 2 * ones(Float32, nout, batch)
    β_eg = 3 * ones(Float32, nout, batch)
    y_eg = vcat(α_eg, β_eg)
    q_eg = split_params(EG, y_eg)
    @test q_eg isa NamedTuple{(:α, :β)}
    @test q_eg.α == α_eg
    @test q_eg.β == β_eg
    @test vcat(q_eg.α, q_eg.β) == y_eg

    # BB split_params returns NamedTuple
    α_bb = 2 * ones(Float32, nout, batch)
    β_bb = 3 * ones(Float32, nout, batch)
    y_bb = vcat(α_bb, β_bb)
    q_bb = split_params(BB, y_bb)
    @test q_bb isa NamedTuple{(:α, :β)}
    @test q_bb.α == α_bb
    @test vcat(q_bb.α, q_bb.β) == y_bb

    # BNB split_params returns NamedTuple
    r_bnb = ones(Float32, nout, batch)
    α_bnb = 2 * ones(Float32, nout, batch)
    β_bnb = 3 * ones(Float32, nout, batch)
    y_bnb = vcat(r_bnb, α_bnb, β_bnb)
    q_bnb = split_params(BNB, y_bnb)
    @test q_bnb isa NamedTuple{(:r, :α, :β)}
    @test q_bnb.r == r_bnb
    @test q_bnb.α == α_bnb
    @test q_bnb.β == β_bnb
    @test vcat(q_bnb.r, q_bnb.α, q_bnb.β) == y_bnb

    # MVE split_params returns NamedTuple
    μ = ones(Float32, nout, batch)
    σ = 2 * ones(Float32, nout, batch)
    y_mve = vcat(μ, σ)
    q = split_params(MVE, y_mve)
    @test q isa NamedTuple{(:μ, :σ)}
    @test q.μ == μ
    @test q.σ == σ
    @test vcat(q.μ, q.σ) == y_mve

    # DIR split_params wraps in NamedTuple
    α_dir = ones(Float32, nout, batch)
    r = split_params(DIR, α_dir)
    @test r isa NamedTuple{(:α,)}
    @test r.α == α_dir

    # FDIR split_params returns NamedTuple with α, p, τ
    K = nout
    α_fd = ones(Float32, K, batch)
    p_fd = ones(Float32, K, batch) ./ K
    τ_fd = 2 * ones(Float32, 1, batch)
    y_fd = vcat(α_fd, p_fd, τ_fd)
    s = split_params(FDIR, y_fd)
    @test s isa NamedTuple{(:α, :p, :τ)}
    @test s.α == α_fd
    @test s.p == p_fd
    @test s.τ == τ_fd
    @test vcat(s.α, s.p, s.τ) == y_fd
end

@testset "splitfdir" begin
    K, batch = 3, 5
    α = ones(Float32, K, batch)
    p = ones(Float32, K, batch) ./ K
    τ = 2 * ones(Float32, 1, batch)
    y = vcat(α, p, τ)
    α2, p2, τ2 = splitfdir(y)
    @test α2 == α
    @test p2 == p
    @test τ2 == τ
end

@testset "splitpg" begin
    nout, batch = 3, 5
    α = 2 * ones(Float32, nout, batch)
    β = 3 * ones(Float32, nout, batch)
    y = vcat(α, β)
    α2, β2 = splitpg(y)
    @test α2 == α
    @test β2 == β
end

@testset "spliteg" begin
    nout, batch = 3, 5
    α = 2 * ones(Float32, nout, batch)
    β = 3 * ones(Float32, nout, batch)
    y = vcat(α, β)
    α2, β2 = spliteg(y)
    @test α2 == α
    @test β2 == β
end

@testset "splitbb" begin
    nout, batch = 3, 5
    α = 2 * ones(Float32, nout, batch)
    β = 3 * ones(Float32, nout, batch)
    y = vcat(α, β)
    α2, β2 = splitbb(y)
    @test α2 == α
    @test β2 == β
end

@testset "splitbnb" begin
    nout, batch = 3, 5
    r = ones(Float32, nout, batch)
    α = 2 * ones(Float32, nout, batch)
    β = 3 * ones(Float32, nout, batch)
    y = vcat(r, α, β)
    r2, α2, β2 = splitbnb(y)
    @test r2 == r
    @test α2 == α
    @test β2 == β
end

@testset "splitzip" begin
    nout, batch = 3, 5
    α_π = ones(Float32, nout, batch)
    β_π = 2 * ones(Float32, nout, batch)
    α_λ = 3 * ones(Float32, nout, batch)
    β_λ = 4 * ones(Float32, nout, batch)
    y = vcat(α_π, β_π, α_λ, β_λ)
    α_π2, β_π2, α_λ2, β_λ2 = splitzip(y)
    @test α_π2 == α_π
    @test β_π2 == β_π
    @test α_λ2 == α_λ
    @test β_λ2 == β_λ
end

@testset "splitvm" begin
    nout, batch = 3, 5
    μ₀ = randn(Float32, nout, batch)
    κ₀ = 2 * ones(Float32, nout, batch)
    κ = 3 * ones(Float32, nout, batch)
    y = vcat(μ₀, κ₀, κ)
    μ₀2, κ₀2, κ2 = splitvm(y)
    @test μ₀2 == μ₀
    @test κ₀2 == κ₀
    @test κ2 == κ
end

@testset "splitmve" begin
    nout, batch = 3, 5
    μ = ones(Float32, nout, batch)
    σ = 2 * ones(Float32, nout, batch)
    y = vcat(μ, σ)
    μ2, σ2 = splitmve(y)
    @test μ2 == μ
    @test σ2 == σ
end

@testset "Layer options" begin
    # bias=false
    @test size(NIG(3 => 2; bias = false)(randn(Float32, 3, 5))) == (8, 5)
    @test size(DIR(3 => 2; bias = false)(randn(Float32, 3, 5))) == (2, 5)
    @test size(MVE(3 => 2; bias = false)(randn(Float32, 3, 5))) == (4, 5)
    @test size(PG(3 => 2; bias = false)(randn(Float32, 3, 5))) == (4, 5)
    @test size(EG(3 => 2; bias = false)(randn(Float32, 3, 5))) == (4, 5)
    @test size(BB(3 => 2; bias = false)(randn(Float32, 3, 5))) == (4, 5)
    @test size(BNB(3 => 2; bias = false)(randn(Float32, 3, 5))) == (6, 5)
    @test size(ZIP(3 => 2; bias = false)(randn(Float32, 3, 5))) == (8, 5)
    @test size(VM(3 => 2; bias = false)(randn(Float32, 3, 5))) == (6, 5)
    @test size(FDIR(3 => 2; bias = false)(randn(Float32, 3, 5))) == (5, 5)

    # 3D input (higher-dimensional reshape)
    x3d = randn(Float32, 3, 4, 5)
    @test size(NIG(3 => 2)(x3d)) == (8, 4, 5)
    @test size(DIR(3 => 2)(x3d)) == (2, 4, 5)
    @test size(MVE(3 => 2)(x3d)) == (4, 4, 5)
    @test size(PG(3 => 2)(x3d)) == (4, 4, 5)
    @test size(EG(3 => 2)(x3d)) == (4, 4, 5)
    @test size(BB(3 => 2)(x3d)) == (4, 4, 5)
    @test size(BNB(3 => 2)(x3d)) == (6, 4, 5)
    @test size(ZIP(3 => 2)(x3d)) == (8, 4, 5)
    @test size(VM(3 => 2)(x3d)) == (6, 4, 5)
    @test size(FDIR(3 => 2)(x3d)) == (5, 4, 5)
end

@testset "Utility functions" begin
    # evidence(α) - DIR
    α = [2.0 3.0; 4.0 5.0]
    @test evidence(α) == [1.0 2.0; 3.0 4.0]

    # evidence(ν, α) - NIG
    ν = [1.0 2.0]
    α = [3.0 4.0]
    @test evidence(ν, α) ≈ [5.0 8.0]

    # uncertainty(α) - DIR epistemic
    α = [2.0 3.0; 3.0 7.0]
    @test uncertainty(α) ≈ 2 ./ sum(α, dims = 1)

    # uncertainty(α, β) - NIG aleatoric
    α = [3.0 4.0]
    β = [2.0 6.0]
    @test uncertainty(α, β) ≈ [1.0 2.0]

    # uncertainty(ν, α, β) - NIG epistemic
    ν = [2.0 4.0]
    α = [3.0 5.0]
    β = [4.0 8.0]
    @test uncertainty(ν, α, β) ≈ [1.0 0.5]

    # aleatoric(ν, α, β)
    ν = [2.0]
    α = [3.0]
    β = [6.0]
    @test aleatoric(ν, α, β) ≈ [(6.0 * (1 + 2.0)) / (2.0 * 3.0)]

    # epistemic(ν)
    ν = [4.0 9.0]
    @test epistemic(ν) ≈ [0.5 1 / 3]

    # --- Type-dispatched uncertainty ---

    # NIG type-dispatched: delegates to existing arity-dispatched methods
    ν_td = [2.0 4.0]
    α_td = [3.0 5.0]
    β_td = [4.0 8.0]
    @test epistemic(NIG, ν_td, α_td, β_td) ≈ epistemic(ν_td)
    @test aleatoric(NIG, ν_td, α_td, β_td) ≈ aleatoric(ν_td, α_td, β_td)
    @test uncertainty(NIG, ν_td, α_td, β_td) ≈ uncertainty(ν_td, α_td, β_td)

    # DIR type-dispatched
    α_dir_td = [2.0 3.0; 3.0 7.0]
    @test epistemic(DIR, α_dir_td) ≈ uncertainty(α_dir_td)

    # MVE type-dispatched: aleatoric is just σ
    σ_mve = Float32.([0.5 1.0; 0.3 0.8])
    @test aleatoric(MVE, σ_mve) == σ_mve

    # BB: epistemic = Var[p] = αβ/((α+β)²(α+β+1)), aleatoric = E[p(1-p)]
    α_bb = [3.0 4.0]
    β_bb = [2.0 6.0]
    epi_bb = epistemic(BB, α_bb, β_bb)
    ale_bb = aleatoric(BB, α_bb, β_bb)
    @test all(>(0), epi_bb)
    @test all(>(0), ale_bb)
    # manual: α=3, β=2, S=5 → epi = 6/(25*6) = 0.04, ale = 6/(5*6) = 0.2
    @test epi_bb[1] ≈ 6.0 / (25.0 * 6.0)
    @test ale_bb[1] ≈ 6.0 / (5.0 * 6.0)
    # epistemic = aleatoric / (α+β), since Var[p] = E[p(1-p)]/(α+β)
    @test epi_bb ≈ ale_bb ./ (α_bb .+ β_bb)

    # EG: epistemic = β²/((α-1)²(α-2)), aleatoric = β²/((α-1)(α-2))
    α_eg = [4.0 5.0]
    β_eg = [6.0 8.0]
    epi_eg = epistemic(EG, α_eg, β_eg)
    ale_eg = aleatoric(EG, α_eg, β_eg)
    @test all(isfinite, epi_eg)
    @test all(isfinite, ale_eg)
    @test all(>(0), epi_eg)
    @test all(>(0), ale_eg)
    # manual: α=4, β=6 → epi = 36/(9*2) = 2.0, ale = 36/(3*2) = 6.0
    @test epi_eg[1] ≈ 36.0 / (9.0 * 2.0)
    @test ale_eg[1] ≈ 36.0 / (3.0 * 2.0)
    # total = epi + ale = Var[Y] under Lomax = β²α/((α-1)²(α-2))
    @test epi_eg .+ ale_eg ≈ @. β_eg^2 * α_eg / ((α_eg - 1)^2 * (α_eg - 2))

    # PG: epistemic = α/β², aleatoric = α/β
    α_pg = [4.0 9.0]
    β_pg = [2.0 3.0]
    @test epistemic(PG, α_pg, β_pg) ≈ [1.0 1.0]
    @test aleatoric(PG, α_pg, β_pg) ≈ [2.0 3.0]
    # total = epistemic + aleatoric = α(β+1)/β²
    @test epistemic(PG, α_pg, β_pg) .+ aleatoric(PG, α_pg, β_pg) ≈ @. α_pg * (β_pg + 1) / β_pg^2

    # BNB: epistemic = r²α(α+β-1)/((β-1)²(β-2)), aleatoric = rα(α+β-1)/((β-1)(β-2))
    r_bnb = [2.0 3.0]
    α_bnb = [3.0 4.0]
    β_bnb = [5.0 6.0]  # > 2
    epi_bnb = epistemic(BNB, r_bnb, α_bnb, β_bnb)
    ale_bnb = aleatoric(BNB, r_bnb, α_bnb, β_bnb)
    @test size(epi_bnb) == (1, 2)
    @test size(ale_bnb) == (1, 2)
    @test all(isfinite, epi_bnb)
    @test all(isfinite, ale_bnb)
    @test all(>(0), epi_bnb)
    @test all(>(0), ale_bnb)
    # manual check for first element: r=2, α=3, β=5
    # epistemic = 4*3*7/(4²*3) = 84/48 = 1.75
    @test epi_bnb[1] ≈ 2.0^2 * 3.0 * 7.0 / (4.0^2 * 3.0)
    # aleatoric = 2*3*7/(4*3) = 42/12 = 3.5
    @test ale_bnb[1] ≈ 2.0 * 3.0 * 7.0 / (4.0 * 3.0)

    # ZIP: epistemic = Var[(1-π)λ], aleatoric = E[Var[Y|π,λ]]
    α_π_zip = [2.0 3.0]
    β_π_zip = [3.0 7.0]
    α_λ_zip = [4.0 5.0]
    β_λ_zip = [2.0 3.0]
    epi_zip = epistemic(ZIP, α_π_zip, β_π_zip, α_λ_zip, β_λ_zip)
    ale_zip = aleatoric(ZIP, α_π_zip, β_π_zip, α_λ_zip, β_λ_zip)
    @test all(isfinite, epi_zip)
    @test all(isfinite, ale_zip)
    @test all(>(0), epi_zip)
    @test all(>(0), ale_zip)
    # manual check for first element: α_π=2, β_π=3, α_λ=4, β_λ=2
    # S_π=5, E[1-π]=3/5, E[(1-π)²]=3·4/(5·6)=12/30=2/5, E[λ]=2, E[λ²]=4·5/4=5
    # epistemic = 2/5·5 - (3/5·2)² = 2 - 36/25 = 50/25 - 36/25 = 14/25
    @test epi_zip[1] ≈ 14.0 / 25.0
    # E[π(1-π)]=2·3/(5·6)=6/30=1/5, aleatoric = 3/5·2 + 1/5·5 = 6/5 + 1 = 11/5
    @test ale_zip[1] ≈ 11.0 / 5.0
    # total = epistemic + aleatoric = Var[Y] (law of total variance)
    @test epi_zip .+ ale_zip ≈ @. (
        β_π_zip * (β_π_zip + 1) / ((α_π_zip + β_π_zip) * (α_π_zip + β_π_zip + 1)) *
        α_λ_zip * (α_λ_zip + 1) / β_λ_zip^2 -
        (β_π_zip / (α_π_zip + β_π_zip) * α_λ_zip / β_λ_zip)^2 +
        β_π_zip / (α_π_zip + β_π_zip) * α_λ_zip / β_λ_zip +
        α_π_zip * β_π_zip / ((α_π_zip + β_π_zip) * (α_π_zip + β_π_zip + 1)) *
        α_λ_zip * (α_λ_zip + 1) / β_λ_zip^2
    )

    # VM: epistemic = 1 - I₁(κ₀)/I₀(κ₀), aleatoric = 1 - I₁(κ)/I₀(κ)
    using SpecialFunctions: besselix
    κ₀_vm = [2.0 5.0]
    κ_vm = [3.0 1.0]
    epi_vm = epistemic(VM, κ₀_vm)
    ale_vm = aleatoric(VM, κ_vm)
    @test all(isfinite, epi_vm)
    @test all(isfinite, ale_vm)
    @test all(>(0), epi_vm)
    @test all(>(0), ale_vm)
    @test all(<(1), epi_vm)
    @test all(<(1), ale_vm)
    # manual: A(κ) = besselix(1,κ)/besselix(0,κ) = I₁(κ)/I₀(κ)
    @test epi_vm ≈ 1 .- besselix.(1, κ₀_vm) ./ besselix.(0, κ₀_vm)
    @test ale_vm ≈ 1 .- besselix.(1, κ_vm) ./ besselix.(0, κ_vm)
    # κ=0 → circular variance = 1 (uniform)
    @test epistemic(VM, [0.0])[1] ≈ 1.0
    @test aleatoric(VM, [0.0])[1] ≈ 1.0

    # FDIR: epistemic and aleatoric, per sample
    K = 3
    α_fd = Float32.([2.0 3.0; 3.0 5.0; 5.0 2.0])
    p_fd = α_fd ./ sum(α_fd, dims = 1)  # proportional allocation
    τ_fd = ones(Float32, 1, 2)
    eu_fd = epistemic(FDIR, α_fd, p_fd, τ_fd)
    au_fd = aleatoric(FDIR, α_fd, p_fd, τ_fd)
    @test size(eu_fd) == (1, 2)
    @test size(au_fd) == (1, 2)
    @test all(isfinite, eu_fd)
    @test all(isfinite, au_fd)
    @test all(≥(0), eu_fd)
    # total = epistemic + aleatoric
    S_fd = sum(α_fd, dims = 1) .+ τ_fd
    μ_fd = (α_fd .+ τ_fd .* p_fd) ./ S_fd
    tu_fd = 1 .- sum(μ_fd .^ 2, dims = 1)
    @test eu_fd .+ au_fd ≈ tu_fd
end

@testset "Loss functions" begin
    nout, batch = 3, 5
    y = randn(Float32, nout, batch)
    γ = randn(Float32, nout, batch)
    ν = abs.(randn(Float32, nout, batch)) .+ 0.1f0
    α = abs.(randn(Float32, nout, batch)) .+ 1.1f0
    β = abs.(randn(Float32, nout, batch)) .+ 0.1f0

    # nllstudent
    nll = nllstudent(y, γ, ν, α, β)
    @test size(nll) == (nout, batch)
    @test all(isfinite, nll)

    # nigloss
    l1 = nigloss(y, γ, ν, α, β)
    @test size(l1) == (nout, batch)
    @test all(isfinite, l1)

    # nigloss_scaled
    l2 = nigloss_scaled(y, γ, ν, α, β)
    @test size(l2) == (nout, batch)
    @test all(isfinite, l2)

    # nigloss_ureg
    l3 = nigloss_ureg(y, γ, ν, α, β)
    @test size(l3) == (nout, batch)
    @test all(isfinite, l3)

    # dirloss returns (1, B) — one loss per batch element
    nclasses = 3
    y_oh = Float32.([1 0 0 1 0; 0 1 0 0 1; 0 0 1 0 0])
    α_dir = abs.(randn(Float32, nclasses, 5)) .+ 1.1f0
    dl = dirloss(y_oh, α_dir, 1)
    @test size(dl) == (1, 5)
    @test all(isfinite, dl)

    # dirloss annealing: λₜ = min(1.0, t/10) changes with epoch
    dl_early = dirloss(y_oh, α_dir, 1)
    dl_late = dirloss(y_oh, α_dir, 20)
    @test all(isfinite, dl_early)
    @test all(isfinite, dl_late)

    # dirloss_cor returns (1, B) and is finite
    dl2 = dirloss_cor(y_oh, α_dir, 1)
    @test size(dl2) == (1, 5)
    @test all(isfinite, dl2)

    # dirloss_cor correction is inactive when all evidence is high (o_gt > 0),
    # so dirloss_cor == dirloss for large α
    α_high = ones(Float32, nclasses, 5) .+ 10.0f0
    @test dirloss_cor(y_oh, α_high, 5) ≈ dirloss(y_oh, α_high, 5)

    # dirloss_cor ≥ dirloss (correction term is non-negative)
    @test all(dirloss_cor(y_oh, α_dir, 1) .≥ dirloss(y_oh, α_dir, 1) .- eps(Float32))

    # dirmultloss returns (1, B) and is finite
    y_counts_dm = Float32.([3 0 1; 2 5 0; 0 1 4])  # (3, 3) count vectors
    α_dm = abs.(randn(Float32, 3, 3)) .+ 1.1f0
    dml = dirmultloss(y_counts_dm, α_dm)
    @test size(dml) == (1, 3)
    @test all(isfinite, dml)

    # dirmultloss sanity: uniform Dir(1,1) with single observation [1,0]
    # should give NLL = log(2) since p(y|α=[1,1]) = 0.5
    y_single = Float32.([1; 0;;])
    α_uniform = Float32.([1; 1;;])
    @test dirmultloss(y_single, α_uniform) ≈ Float32.(log(2) * ones(1, 1)) atol = 1.0f-5

    # mveloss
    μ = randn(Float32, nout, batch)
    σ = abs.(randn(Float32, nout, batch)) .+ 0.1f0
    ml = mveloss(y, μ, σ)
    @test size(ml) == (nout, batch)
    @test all(isfinite, ml)

    # mveloss with β
    ml2 = mveloss(y, μ, σ, 0.5f0)
    @test size(ml2) == (nout, batch)
    @test all(isfinite, ml2)

    # fdirloss
    nclasses_fd = 3
    y_oh_fd = Float32.([1 0 0 1 0; 0 1 0 0 1; 0 0 1 0 0])
    α_fd = abs.(randn(Float32, nclasses_fd, 5)) .+ 0.5f0
    p_fd = abs.(randn(Float32, nclasses_fd, 5))
    p_fd = p_fd ./ sum(p_fd, dims = 1)  # normalize to simplex
    τ_fd = abs.(randn(Float32, 1, 5)) .+ 0.1f0
    fl = fdirloss(y_oh_fd, α_fd, p_fd, τ_fd)
    @test size(fl) == (1, 5)
    @test all(isfinite, fl)
    # loss is non-negative (sum of squared terms + Brier score)
    @test all(≥(0), fl)

    # Theorem 4.3: FD reduces to Dirichlet when τ=1 and p = α/Σα.
    # The evidence term (expected Brier score) of fdirloss should equal
    # the Dirichlet Bayes Risk MSE computed with the same concentrations.
    α_eq = Float32.([2.0 3.0; 3.0 5.0; 5.0 2.0])  # (3, 2)
    y_eq = Float32.([1 0; 0 1; 0 0])
    α₀ = sum(α_eq, dims = 1)
    τ_one = ones(Float32, 1, 2)
    p_eq = α_eq ./ α₀
    # Dirichlet Bayes Risk MSE: Σₖ [(yₖ - p̂ₖ)² + p̂ₖ(1-p̂ₖ)/(S+1)]
    p̂ = α_eq ./ α₀
    dir_evid = sum((y_eq - p̂) .^ 2 .+ p̂ .* (1 .- p̂) ./ (α₀ .+ 1), dims = 1)
    # FD evidence term = fdirloss - Brier regularizer on p
    fd_total = fdirloss(y_eq, α_eq, p_eq, τ_one)
    brier_reg = sum((y_eq .- p_eq) .^ 2, dims = 1)
    fd_evid = fd_total .- brier_reg
    @test fd_evid ≈ dir_evid

    # nllpg
    nout_pg, batch_pg = 3, 5
    y_counts = Float32.(rand(0:10, nout_pg, batch_pg))
    α_pg = abs.(randn(Float32, nout_pg, batch_pg)) .+ 0.5f0
    β_pg = abs.(randn(Float32, nout_pg, batch_pg)) .+ 0.5f0
    nll_pg = nllpg(y_counts, α_pg, β_pg)
    @test size(nll_pg) == (nout_pg, batch_pg)
    @test all(isfinite, nll_pg)

    # pgloss
    pl = pgloss(y_counts, α_pg, β_pg)
    @test size(pl) == (nout_pg, batch_pg)
    @test all(isfinite, pl)

    # pgloss with λ=0 equals nllpg
    @test pgloss(y_counts, α_pg, β_pg, 0) ≈ nllpg(y_counts, α_pg, β_pg)

    # nlleg
    y_pos = abs.(randn(Float32, nout_pg, batch_pg)) .+ 0.1f0
    α_eg = abs.(randn(Float32, nout_pg, batch_pg)) .+ 0.5f0
    β_eg = abs.(randn(Float32, nout_pg, batch_pg)) .+ 0.5f0
    nll_eg = nlleg(y_pos, α_eg, β_eg)
    @test size(nll_eg) == (nout_pg, batch_pg)
    @test all(isfinite, nll_eg)

    # egloss
    el = egloss(y_pos, α_eg, β_eg)
    @test size(el) == (nout_pg, batch_pg)
    @test all(isfinite, el)

    # egloss with λ=0 equals nlleg
    @test egloss(y_pos, α_eg, β_eg, 0) ≈ nlleg(y_pos, α_eg, β_eg)

    # nlleg sanity: α=1, β=1, y=1 → p(y|1,1) = 1·1/(1+1)^2 = 0.25 → NLL = log(4)
    @test nlleg(Float32[1;;], Float32[1;;], Float32[1;;]) ≈ Float32.(log(4) * ones(1, 1)) atol = 1.0f-5

    # nllbb
    nout_bb, batch_bb = 3, 5
    k_bb = Float32.(rand(0:5, nout_bb, batch_bb))
    n_bb = k_bb .+ Float32.(rand(0:5, nout_bb, batch_bb))  # n ≥ k
    α_bbl = abs.(randn(Float32, nout_bb, batch_bb)) .+ 0.5f0
    β_bbl = abs.(randn(Float32, nout_bb, batch_bb)) .+ 0.5f0
    nll_bb = nllbb(k_bb, n_bb, α_bbl, β_bbl)
    @test size(nll_bb) == (nout_bb, batch_bb)
    @test all(isfinite, nll_bb)

    # bbloss
    bbl = bbloss(k_bb, n_bb, α_bbl, β_bbl)
    @test size(bbl) == (nout_bb, batch_bb)
    @test all(isfinite, bbl)

    # bbloss with λ=0 equals nllbb
    @test bbloss(k_bb, n_bb, α_bbl, β_bbl, 0) ≈ nllbb(k_bb, n_bb, α_bbl, β_bbl)

    # nllbb sanity: Beta(1,1), n=1, k=0 → p = 0.5 → NLL = log(2)
    @test nllbb(Float32[0;;], Float32[1;;], Float32[1;;], Float32[1;;]) ≈ Float32.(log(2) * ones(1, 1)) atol = 1.0f-5

    # nllbnb
    r_bnb = abs.(randn(Float32, nout_pg, batch_pg)) .+ 0.5f0
    α_bnb = abs.(randn(Float32, nout_pg, batch_pg)) .+ 0.5f0
    β_bnb = abs.(randn(Float32, nout_pg, batch_pg)) .+ 0.5f0
    nll_bnb = nllbnb(y_counts, r_bnb, α_bnb, β_bnb)
    @test size(nll_bnb) == (nout_pg, batch_pg)
    @test all(isfinite, nll_bnb)

    # bnbloss
    bl = bnbloss(y_counts, r_bnb, α_bnb, β_bnb)
    @test size(bl) == (nout_pg, batch_pg)
    @test all(isfinite, bl)

    # bnbloss with λ=0 equals nllbnb
    @test bnbloss(y_counts, r_bnb, α_bnb, β_bnb, 0) ≈ nllbnb(y_counts, r_bnb, α_bnb, β_bnb)

    # nllbnb sanity: uniform Beta(1,1), r=1, y=0 should give NLL = log(2)
    @test nllbnb(Float32[0;;], Float32[1;;], Float32[1;;], Float32[1;;]) ≈ Float32.(log(2) * ones(1, 1)) atol = 1.0f-5

    # nllzip
    nout_zip, batch_zip = 3, 5
    y_counts_zip = Float32.(rand(0:10, nout_zip, batch_zip))
    α_π_zip = abs.(randn(Float32, nout_zip, batch_zip)) .+ 0.5f0
    β_π_zip = abs.(randn(Float32, nout_zip, batch_zip)) .+ 0.5f0
    α_λ_zip = abs.(randn(Float32, nout_zip, batch_zip)) .+ 0.5f0
    β_λ_zip = abs.(randn(Float32, nout_zip, batch_zip)) .+ 0.5f0
    nll_zip = nllzip(y_counts_zip, α_π_zip, β_π_zip, α_λ_zip, β_λ_zip)
    @test size(nll_zip) == (nout_zip, batch_zip)
    @test all(isfinite, nll_zip)

    # ziploss
    zl = ziploss(y_counts_zip, α_π_zip, β_π_zip, α_λ_zip, β_λ_zip)
    @test size(zl) == (nout_zip, batch_zip)
    @test all(isfinite, zl)

    # ziploss with λ=0 equals nllzip
    @test ziploss(y_counts_zip, α_π_zip, β_π_zip, α_λ_zip, β_λ_zip, 0) ≈
        nllzip(y_counts_zip, α_π_zip, β_π_zip, α_λ_zip, β_λ_zip)

    # nllzip sanity: y=0, uniform Beta(1,1), Gamma(1,1)
    # p(0) = 1/2 + 1/2·(1/2) = 3/4 → NLL = log(4/3)
    @test nllzip(Float32[0;;], Float32[1;;], Float32[1;;], Float32[1;;], Float32[1;;]) ≈
        Float32.(log(4 / 3) * ones(1, 1)) atol = 1.0f-5

    # nllzip sanity: y=1, uniform Beta(1,1), Gamma(1,1)
    # p(1) = 1/2 · NegBin(1|1,1) = 1/2 · 1/4 = 1/8 → NLL = log(8)
    @test nllzip(Float32[1;;], Float32[1;;], Float32[1;;], Float32[1;;], Float32[1;;]) ≈
        Float32.(log(8) * ones(1, 1)) atol = 1.0f-5

    # nllzip with π→0 (β_π >> α_π) should approach nllpg
    α_π_small = Float32[0.001;;]
    β_π_large = Float32[1000.0;;]
    α_λ_t = Float32[3.0;;]
    β_λ_t = Float32[2.0;;]
    for y_t in [Float32[0;;], Float32[3;;], Float32[7;;]]
        @test nllzip(y_t, α_π_small, β_π_large, α_λ_t, β_λ_t) ≈
            nllpg(y_t, α_λ_t, β_λ_t) atol = 0.01f0
    end

    # nllvm
    nout_vm, batch_vm = 3, 5
    θ_vm = Float32(π) .* (2 .* rand(Float32, nout_vm, batch_vm) .- 1)  # angles in [-π, π)
    μ₀_vm = randn(Float32, nout_vm, batch_vm)
    κ₀_vm = abs.(randn(Float32, nout_vm, batch_vm)) .+ 0.5f0
    κ_vm = abs.(randn(Float32, nout_vm, batch_vm)) .+ 0.5f0
    nll_vm = nllvm(θ_vm, μ₀_vm, κ₀_vm, κ_vm)
    @test size(nll_vm) == (nout_vm, batch_vm)
    @test all(isfinite, nll_vm)

    # vmloss
    vl = vmloss(θ_vm, μ₀_vm, κ₀_vm, κ_vm)
    @test size(vl) == (nout_vm, batch_vm)
    @test all(isfinite, vl)

    # vmloss with λ=0 equals nllvm
    @test vmloss(θ_vm, μ₀_vm, κ₀_vm, κ_vm, 0) ≈ nllvm(θ_vm, μ₀_vm, κ₀_vm, κ_vm)

    # nllvm sanity: uniform prior (κ₀=0), any θ → NLL = log(2π)
    @test nllvm(Float32[0;;], Float32[0;;], Float32[0;;], Float32[1;;]) ≈
        Float32.(log(2π) * ones(1, 1)) atol = 1.0f-5

    # nllvm sanity: θ near μ₀ gives lower NLL than θ far from μ₀
    nll_near = nllvm(Float32[0;;], Float32[0;;], Float32[2;;], Float32[2;;])
    nll_far = nllvm(Float32[Float32(π);;], Float32[0;;], Float32[2;;], Float32[2;;])
    @test nll_near[1] < nll_far[1]
end

@testset "predict" begin
    x = randn(Float32, 3, 5)

    # NIG predict returns NamedTuple, destructurable
    m_nig = Chain(Dense(3 => 10, relu), NIG(10 => 2))
    p_nig = predict(m_nig, x)
    @test p_nig isa NamedTuple{(:γ, :ν, :α, :β)}
    @test size(p_nig.γ) == (2, 5)
    @test size(p_nig.ν) == (2, 5)
    @test all(>(0), p_nig.ν)
    @test all(≥(1), p_nig.α)
    @test all(>(0), p_nig.β)
    # backward compat: tuple destructuring still works
    γ, ν, α, β = predict(m_nig, x)
    @test size(γ) == (2, 5)

    # MVE predict returns NamedTuple, destructurable
    m_mve = Chain(Dense(3 => 10, relu), MVE(10 => 2))
    p_mve = predict(m_mve, x)
    @test p_mve isa NamedTuple{(:μ, :σ)}
    @test size(p_mve.μ) == (2, 5)
    @test size(p_mve.σ) == (2, 5)
    @test all(>(0), p_mve.σ)
    # backward compat: tuple destructuring still works
    μ, σ = predict(m_mve, x)
    @test size(μ) == (2, 5)

    # DIR predict returns raw array (backward compatible)
    m_dir = Chain(Dense(3 => 10, relu), DIR(10 => 4))
    α = predict(m_dir, x)
    @test size(α) == (4, 5)
    @test all(≥(1), α)

    # FDIR predict returns NamedTuple with α, p, τ
    m_fdir = Chain(Dense(3 => 10, relu), FDIR(10 => 4))
    p_fdir = predict(m_fdir, x)
    @test p_fdir isa NamedTuple{(:α, :p, :τ)}
    @test size(p_fdir.α) == (4, 5)
    @test size(p_fdir.p) == (4, 5)
    @test size(p_fdir.τ) == (1, 5)
    @test all(>(0), p_fdir.α)
    @test all(>(0), p_fdir.τ)
    @test all(isapprox.(sum(p_fdir.p, dims = 1), 1, atol = 1.0f-5))

    # PG predict returns NamedTuple with α, β
    m_pg = Chain(Dense(3 => 10, relu), PG(10 => 2))
    p_pg = predict(m_pg, x)
    @test p_pg isa NamedTuple{(:α, :β)}
    @test size(p_pg.α) == (2, 5)
    @test size(p_pg.β) == (2, 5)
    @test all(>(0), p_pg.α)
    @test all(>(0), p_pg.β)

    # EG predict returns NamedTuple with α, β
    m_eg = Chain(Dense(3 => 10, relu), EG(10 => 2))
    p_eg = predict(m_eg, x)
    @test p_eg isa NamedTuple{(:α, :β)}
    @test size(p_eg.α) == (2, 5)
    @test all(>(0), p_eg.α)
    @test all(>(0), p_eg.β)

    # BB predict returns NamedTuple with α, β
    m_bb = Chain(Dense(3 => 10, relu), BB(10 => 2))
    p_bb = predict(m_bb, x)
    @test p_bb isa NamedTuple{(:α, :β)}
    @test size(p_bb.α) == (2, 5)
    @test all(>(0), p_bb.α)
    @test all(>(0), p_bb.β)

    # BNB predict returns NamedTuple with r, α, β
    m_bnb = Chain(Dense(3 => 10, relu), BNB(10 => 2))
    p_bnb = predict(m_bnb, x)
    @test p_bnb isa NamedTuple{(:r, :α, :β)}
    @test size(p_bnb.r) == (2, 5)
    @test all(>(0), p_bnb.r)
    @test all(>(0), p_bnb.α)
    @test all(>(0), p_bnb.β)

    # ZIP predict returns NamedTuple with α_π, β_π, α_λ, β_λ
    m_zip = Chain(Dense(3 => 10, relu), ZIP(10 => 2))
    p_zip = predict(m_zip, x)
    @test p_zip isa NamedTuple{(:α_π, :β_π, :α_λ, :β_λ)}
    @test size(p_zip.α_π) == (2, 5)
    @test all(>(0), p_zip.α_π)
    @test all(>(0), p_zip.β_π)
    @test all(>(0), p_zip.α_λ)
    @test all(>(0), p_zip.β_λ)

    # VM predict returns NamedTuple with μ₀, κ₀, κ
    m_vm = Chain(Dense(3 => 10, relu), VM(10 => 2))
    p_vm = predict(m_vm, x)
    @test p_vm isa NamedTuple{(:μ₀, :κ₀, :κ)}
    @test size(p_vm.μ₀) == (2, 5)
    @test size(p_vm.κ₀) == (2, 5)
    @test all(>(0), p_vm.κ₀)
    @test all(>(0), p_vm.κ)
end

@testset "predictive" begin
    x = randn(Float32, 3, 5)

    # NIG: ŷ = γ, both uncertainties present
    m_nig = Chain(Dense(3 => 10, relu), NIG(10 => 2))
    r_nig = predictive(m_nig, x)
    @test r_nig isa NamedTuple{(:ŷ, :epistemic, :aleatoric, :params)}
    @test r_nig.params isa NamedTuple{(:γ, :ν, :α, :β)}
    @test r_nig.ŷ == r_nig.params.γ
    @test size(r_nig.ŷ) == (2, 5)
    @test size(r_nig.epistemic) == (2, 5)
    @test size(r_nig.aleatoric) == (2, 5)

    # EG: ŷ = β/(α-1), both uncertainties present
    m_eg = Chain(Dense(3 => 10, relu), EG(10 => 2))
    r_eg = predictive(m_eg, x)
    @test r_eg isa NamedTuple{(:ŷ, :epistemic, :aleatoric, :params)}
    @test size(r_eg.ŷ) == (2, 5)
    @test all(>(0), r_eg.ŷ)
    @test size(r_eg.epistemic) == (2, 5)
    @test size(r_eg.aleatoric) == (2, 5)

    # BB: ŷ = α/(α+β), both uncertainties present
    m_bb = Chain(Dense(3 => 10, relu), BB(10 => 2))
    r_bb = predictive(m_bb, x)
    @test r_bb.ŷ ≈ r_bb.params.α ./ (r_bb.params.α .+ r_bb.params.β)
    @test all(>(0), r_bb.ŷ)
    @test all(<(1), r_bb.ŷ)
    @test size(r_bb.epistemic) == (2, 5)
    @test size(r_bb.aleatoric) == (2, 5)

    # PG: ŷ = α/β
    m_pg = Chain(Dense(3 => 10, relu), PG(10 => 2))
    r_pg = predictive(m_pg, x)
    @test r_pg.ŷ ≈ r_pg.params.α ./ r_pg.params.β
    @test size(r_pg.epistemic) == (2, 5)
    @test size(r_pg.aleatoric) == (2, 5)

    # BNB: ŷ = r·α/β
    m_bnb = Chain(Dense(3 => 10, relu), BNB(10 => 2))
    r_bnb = predictive(m_bnb, x)
    @test r_bnb.ŷ ≈ r_bnb.params.r .* r_bnb.params.α ./ r_bnb.params.β
    @test size(r_bnb.epistemic) == (2, 5)
    @test size(r_bnb.aleatoric) == (2, 5)

    # ZIP: ŷ = β_π/(α_π+β_π) · α_λ/β_λ
    m_zip = Chain(Dense(3 => 10, relu), ZIP(10 => 2))
    r_zip = predictive(m_zip, x)
    p_z = r_zip.params
    @test r_zip.ŷ ≈ p_z.β_π ./ (p_z.α_π .+ p_z.β_π) .* p_z.α_λ ./ p_z.β_λ
    @test all(>(0), r_zip.ŷ)
    @test size(r_zip.epistemic) == (2, 5)
    @test size(r_zip.aleatoric) == (2, 5)

    # VM: ŷ = μ₀, both uncertainties are circular variances in (0, 1)
    m_vm = Chain(Dense(3 => 10, relu), VM(10 => 2))
    r_vm = predictive(m_vm, x)
    @test r_vm.ŷ == r_vm.params.μ₀
    @test size(r_vm.epistemic) == (2, 5)
    @test size(r_vm.aleatoric) == (2, 5)
    @test all(>(0), r_vm.epistemic)
    @test all(<(1), r_vm.epistemic)
    @test all(>(0), r_vm.aleatoric)
    @test all(<(1), r_vm.aleatoric)

    # DIR: ŷ = α/Σα, aleatoric is nothing
    m_dir = Chain(Dense(3 => 10, relu), DIR(10 => 4))
    r_dir = predictive(m_dir, x)
    α_dir = r_dir.params
    @test r_dir.ŷ ≈ α_dir ./ sum(α_dir, dims = 1)
    @test all(isapprox.(sum(r_dir.ŷ, dims = 1), 1, atol = 1.0f-5))
    @test size(r_dir.epistemic) == (1, 5)
    @test r_dir.aleatoric === nothing

    # FDIR: ŷ = (α + τp)/(Σα + τ), both uncertainties present
    m_fdir = Chain(Dense(3 => 10, relu), FDIR(10 => 4))
    r_fdir = predictive(m_fdir, x)
    p_fd = r_fdir.params
    expected_mean = (p_fd.α .+ p_fd.τ .* p_fd.p) ./ (sum(p_fd.α, dims = 1) .+ p_fd.τ)
    @test r_fdir.ŷ ≈ expected_mean
    @test all(isapprox.(sum(r_fdir.ŷ, dims = 1), 1, atol = 1.0f-5))
    @test size(r_fdir.epistemic) == (1, 5)
    @test size(r_fdir.aleatoric) == (1, 5)

    # MVE: ŷ = μ, epistemic is nothing
    m_mve = Chain(Dense(3 => 10, relu), MVE(10 => 2))
    r_mve = predictive(m_mve, x)
    @test r_mve.ŷ == r_mve.params.μ
    @test r_mve.epistemic === nothing
    @test r_mve.aleatoric == r_mve.params.σ
end

@testset "Gradient flow" begin
    x = randn(Float32, 3, 5)
    y = randn(Float32, 2, 5)

    # nigloss
    m = Chain(Dense(3 => 10, relu), NIG(10 => 2))
    loss, grads = Flux.withgradient(m) do m
        γ, ν, α, β = splitnig(m(x))
        sum(nigloss(y, γ, ν, α, β))
    end
    @test isfinite(loss)
    @test !isnothing(grads[1])

    # nigloss_scaled
    loss2, grads2 = Flux.withgradient(m) do m
        γ, ν, α, β = splitnig(m(x))
        sum(nigloss_scaled(y, γ, ν, α, β))
    end
    @test isfinite(loss2)
    @test !isnothing(grads2[1])

    # nigloss_ureg
    loss3, grads3 = Flux.withgradient(m) do m
        γ, ν, α, β = splitnig(m(x))
        sum(nigloss_ureg(y, γ, ν, α, β))
    end
    @test isfinite(loss3)
    @test !isnothing(grads3[1])

    # dirloss
    y_oh = Float32.([1 0 1 0 0; 0 1 0 1 1])
    m_dir = Chain(Dense(3 => 10, relu), DIR(10 => 2))
    loss_d, grads_d = Flux.withgradient(m_dir) do m
        sum(dirloss(y_oh, m(x), 1))
    end
    @test isfinite(loss_d)
    @test !isnothing(grads_d[1])

    # dirloss_cor
    loss_d2, grads_d2 = Flux.withgradient(m_dir) do m
        sum(dirloss_cor(y_oh, m(x), 1))
    end
    @test isfinite(loss_d2)
    @test !isnothing(grads_d2[1])

    # dirmultloss (reuses DIR layer with count targets)
    y_counts_dm = Float32.([3 0 1 2 5; 2 5 0 4 1])
    loss_dm, grads_dm = Flux.withgradient(m_dir) do m
        sum(dirmultloss(y_counts_dm, m(x)))
    end
    @test isfinite(loss_dm)
    @test !isnothing(grads_dm[1])

    # fdirloss
    y_oh_fd = Float32.([1 0 1 0 0; 0 1 0 1 1])
    m_fdir = Chain(Dense(3 => 10, relu), FDIR(10 => 2))
    loss_fd, grads_fd = Flux.withgradient(m_fdir) do m
        α, p, τ = splitfdir(m(x))
        sum(fdirloss(y_oh_fd, α, p, τ))
    end
    @test isfinite(loss_fd)
    @test !isnothing(grads_fd[1])

    # bbloss
    k_bb = Float32.(rand(0:3, 2, 5))
    n_bb = k_bb .+ Float32.(rand(1:5, 2, 5))
    m_bb = Chain(Dense(3 => 10, relu), BB(10 => 2))
    loss_bb, grads_bb = Flux.withgradient(m_bb) do m
        α, β = splitbb(m(x))
        sum(bbloss(k_bb, n_bb, α, β))
    end
    @test isfinite(loss_bb)
    @test !isnothing(grads_bb[1])

    # egloss
    y_pos = abs.(randn(Float32, 2, 5)) .+ 0.1f0
    m_eg = Chain(Dense(3 => 10, relu), EG(10 => 2))
    loss_eg, grads_eg = Flux.withgradient(m_eg) do m
        α, β = spliteg(m(x))
        sum(egloss(y_pos, α, β))
    end
    @test isfinite(loss_eg)
    @test !isnothing(grads_eg[1])

    # pgloss
    y_counts = Float32.(rand(0:10, 2, 5))
    m_pg = Chain(Dense(3 => 10, relu), PG(10 => 2))
    loss_pg, grads_pg = Flux.withgradient(m_pg) do m
        α, β = splitpg(m(x))
        sum(pgloss(y_counts, α, β))
    end
    @test isfinite(loss_pg)
    @test !isnothing(grads_pg[1])

    # bnbloss
    m_bnb = Chain(Dense(3 => 10, relu), BNB(10 => 2))
    loss_bnb, grads_bnb = Flux.withgradient(m_bnb) do m
        r, α, β = splitbnb(m(x))
        sum(bnbloss(y_counts, r, α, β))
    end
    @test isfinite(loss_bnb)
    @test !isnothing(grads_bnb[1])

    # ziploss
    y_counts_zip = Float32.(rand(0:10, 2, 5))
    m_zip = Chain(Dense(3 => 10, relu), ZIP(10 => 2))
    loss_zip, grads_zip = Flux.withgradient(m_zip) do m
        α_π, β_π, α_λ, β_λ = splitzip(m(x))
        sum(ziploss(y_counts_zip, α_π, β_π, α_λ, β_λ))
    end
    @test isfinite(loss_zip)
    @test !isnothing(grads_zip[1])

    # vmloss
    θ_vm = Float32(π) .* (2 .* rand(Float32, 2, 5) .- 1)
    m_vm = Chain(Dense(3 => 10, relu), VM(10 => 2))
    loss_vm, grads_vm = Flux.withgradient(m_vm) do m
        μ₀, κ₀, κ = splitvm(m(x))
        sum(vmloss(θ_vm, μ₀, κ₀, κ))
    end
    @test isfinite(loss_vm)
    @test !isnothing(grads_vm[1])

    # mveloss
    m_mve = Chain(Dense(3 => 10, relu), MVE(10 => 2))
    loss_m, grads_m = Flux.withgradient(m_mve) do m
        ŷ = m(x)
        μ, σ = ŷ[1:2, :], ŷ[3:4, :]
        sum(mveloss(y, μ, σ))
    end
    @test isfinite(loss_m)
    @test !isnothing(grads_m[1])
end

using CUDA
if CUDA.functional()
    include("gpu.jl")
else
    @info "CUDA not functional, skipping GPU tests"
end
