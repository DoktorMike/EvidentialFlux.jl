using EvidentialFlux
using Flux
using Test

@testset "AbstractEvidentialLayer" begin
    @test NIG <: AbstractEvidentialLayer
    @test DIR <: AbstractEvidentialLayer
    @test MVE <: AbstractEvidentialLayer
    @test FDIR <: AbstractEvidentialLayer
    @test PG <: AbstractEvidentialLayer
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
    @test size(FDIR(3 => 2; bias = false)(randn(Float32, 3, 5))) == (5, 5)

    # 3D input (higher-dimensional reshape)
    x3d = randn(Float32, 3, 4, 5)
    @test size(NIG(3 => 2)(x3d)) == (8, 4, 5)
    @test size(DIR(3 => 2)(x3d)) == (2, 4, 5)
    @test size(MVE(3 => 2)(x3d)) == (4, 4, 5)
    @test size(PG(3 => 2)(x3d)) == (4, 4, 5)
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

    # nigloss2
    l2 = nigloss2(y, γ, ν, α, β)
    @test size(l2) == (nout, batch)
    @test all(isfinite, l2)

    # nigloss3
    l3 = nigloss3(y, γ, ν, α, β)
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

    # dirloss2 returns (1, B) and is finite
    dl2 = dirloss2(y_oh, α_dir, 1)
    @test size(dl2) == (1, 5)
    @test all(isfinite, dl2)

    # dirloss2 correction is inactive when all evidence is high (o_gt > 0),
    # so dirloss2 == dirloss for large α
    α_high = ones(Float32, nclasses, 5) .+ 10.0f0
    @test dirloss2(y_oh, α_high, 5) ≈ dirloss(y_oh, α_high, 5)

    # dirloss2 ≥ dirloss (correction term is non-negative)
    @test all(dirloss2(y_oh, α_dir, 1) .≥ dirloss(y_oh, α_dir, 1) .- eps(Float32))

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

    # nigloss2
    loss2, grads2 = Flux.withgradient(m) do m
        γ, ν, α, β = splitnig(m(x))
        sum(nigloss2(y, γ, ν, α, β))
    end
    @test isfinite(loss2)
    @test !isnothing(grads2[1])

    # nigloss3
    loss3, grads3 = Flux.withgradient(m) do m
        γ, ν, α, β = splitnig(m(x))
        sum(nigloss3(y, γ, ν, α, β))
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

    # dirloss2
    loss_d2, grads_d2 = Flux.withgradient(m_dir) do m
        sum(dirloss2(y_oh, m(x), 1))
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

    # pgloss
    y_counts = Float32.(rand(0:10, 2, 5))
    m_pg = Chain(Dense(3 => 10, relu), PG(10 => 2))
    loss_pg, grads_pg = Flux.withgradient(m_pg) do m
        α, β = splitpg(m(x))
        sum(pgloss(y_counts, α, β))
    end
    @test isfinite(loss_pg)
    @test !isnothing(grads_pg[1])

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
