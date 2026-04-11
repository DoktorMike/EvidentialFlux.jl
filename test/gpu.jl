using CUDA
using EvidentialFlux
using Flux
using Test

@testset "GPU - Forward pass" begin
    x = CUDA.randn(Float32, 3, 10)

    # NIG
    m_nig = NIG(3 => 2) |> gpu
    ŷ = m_nig(x)
    @test size(ŷ) == (8, 10)
    @test ŷ isa CuArray
    @test all(≥(1), Array(ŷ[5:6, :]))  # α ≥ 1
    @test all(>(0), Array(ŷ[3:4, :]))   # ν > 0
    @test all(>(0), Array(ŷ[7:8, :]))   # β > 0

    # DIR
    m_dir = DIR(3 => 4) |> gpu
    ŷ_dir = m_dir(x)
    @test size(ŷ_dir) == (4, 10)
    @test ŷ_dir isa CuArray
    @test all(≥(1), Array(ŷ_dir))

    # MVE
    m_mve = MVE(3 => 2) |> gpu
    ŷ_mve = m_mve(x)
    @test size(ŷ_mve) == (4, 10)
    @test ŷ_mve isa CuArray
    @test all(>(0), Array(ŷ_mve[3:4, :]))  # σ > 0

    # FDIR
    m_fdir = FDIR(3 => 4) |> gpu
    ŷ_fdir = m_fdir(x)
    @test size(ŷ_fdir) == (9, 10)
    @test ŷ_fdir isa CuArray
    α_fd, p_fd, τ_fd = splitfdir(Array(ŷ_fdir))
    @test all(>(0), α_fd)
    @test all(>(0), τ_fd)
    @test all(isapprox.(sum(p_fd, dims = 1), 1, atol = 1.0f-5))
end

@testset "GPU - split_params" begin
    x = CUDA.randn(Float32, 3, 10)

    # NIG
    m_nig = NIG(3 => 2) |> gpu
    p = split_params(NIG, m_nig(x))
    @test p.γ isa CuArray
    @test size(p.γ) == (2, 10)

    # MVE
    m_mve = MVE(3 => 2) |> gpu
    q = split_params(MVE, m_mve(x))
    @test q.μ isa CuArray
    @test size(q.μ) == (2, 10)

    # FDIR
    m_fdir = FDIR(3 => 4) |> gpu
    r = split_params(FDIR, m_fdir(x))
    @test r.α isa CuArray
    @test size(r.α) == (4, 10)
    @test size(r.τ) == (1, 10)
end

@testset "GPU - predict" begin
    x = CUDA.randn(Float32, 3, 5)

    # NIG predict
    m_nig = Chain(Dense(3 => 10, relu), NIG(10 => 2)) |> gpu
    p = predict(m_nig, x)
    @test p isa NamedTuple{(:γ, :ν, :α, :β)}
    @test p.γ isa CuArray
    @test size(p.γ) == (2, 5)

    # MVE predict
    m_mve = Chain(Dense(3 => 10, relu), MVE(10 => 2)) |> gpu
    q = predict(m_mve, x)
    @test q isa NamedTuple{(:μ, :σ)}
    @test q.μ isa CuArray

    # DIR predict
    m_dir = Chain(Dense(3 => 10, relu), DIR(10 => 4)) |> gpu
    α = predict(m_dir, x)
    @test α isa CuArray
    @test size(α) == (4, 5)

    # FDIR predict
    m_fdir = Chain(Dense(3 => 10, relu), FDIR(10 => 4)) |> gpu
    r = predict(m_fdir, x)
    @test r isa NamedTuple{(:α, :p, :τ)}
    @test r.α isa CuArray
    @test size(r.α) == (4, 5)
    @test size(r.τ) == (1, 5)
end

@testset "GPU - Loss functions" begin
    nout, batch = 3, 5
    y = CUDA.randn(Float32, nout, batch)
    γ = CUDA.randn(Float32, nout, batch)
    ν = CUDA.rand(Float32, nout, batch) .+ 0.1f0
    α = CUDA.rand(Float32, nout, batch) .+ 1.1f0
    β = CUDA.rand(Float32, nout, batch) .+ 0.1f0

    # nllstudent
    nll = nllstudent(y, γ, ν, α, β)
    @test size(nll) == (nout, batch)
    @test all(isfinite, Array(nll))

    # nigloss
    l1 = nigloss(y, γ, ν, α, β)
    @test size(l1) == (nout, batch)
    @test all(isfinite, Array(l1))

    # nigloss2
    l2 = nigloss2(y, γ, ν, α, β)
    @test size(l2) == (nout, batch)
    @test all(isfinite, Array(l2))

    # nigloss3
    l3 = nigloss3(y, γ, ν, α, β)
    @test size(l3) == (nout, batch)
    @test all(isfinite, Array(l3))

    # dirloss
    nclasses = 3
    y_oh = cu(Float32.([1 0 0 1 0; 0 1 0 0 1; 0 0 1 0 0]))
    α_dir = CUDA.rand(Float32, nclasses, 5) .+ 1.1f0
    dl = dirloss(y_oh, α_dir, 1)
    @test size(dl) == (1, 5)
    @test all(isfinite, Array(dl))

    # dirloss2
    dl2 = dirloss2(y_oh, α_dir, 1)
    @test size(dl2) == (1, 5)
    @test all(isfinite, Array(dl2))

    # fdirloss
    nclasses_fd = 3
    y_oh_fd = cu(Float32.([1 0 0 1 0; 0 1 0 0 1; 0 0 1 0 0]))
    α_fd = CUDA.rand(Float32, nclasses_fd, 5) .+ 0.5f0
    p_fd = CUDA.rand(Float32, nclasses_fd, 5)
    p_fd = p_fd ./ sum(p_fd, dims = 1)
    τ_fd = CUDA.rand(Float32, 1, 5) .+ 0.1f0
    fl = fdirloss(y_oh_fd, α_fd, p_fd, τ_fd)
    @test size(fl) == (1, 5)
    @test all(isfinite, Array(fl))

    # mveloss
    μ = CUDA.randn(Float32, nout, batch)
    σ = CUDA.rand(Float32, nout, batch) .+ 0.1f0
    ml = mveloss(y, μ, σ)
    @test size(ml) == (nout, batch)
    @test all(isfinite, Array(ml))
end

@testset "GPU - Gradient flow" begin
    x = CUDA.randn(Float32, 3, 5)
    y = CUDA.randn(Float32, 2, 5)

    # nigloss
    m = Chain(Dense(3 => 10, relu), NIG(10 => 2)) |> gpu
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
    y_oh = cu(Float32.([1 0 1 0 0; 0 1 0 1 1]))
    m_dir = Chain(Dense(3 => 10, relu), DIR(10 => 2)) |> gpu
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

    # fdirloss
    y_oh_fd = cu(Float32.([1 0 1 0 0; 0 1 0 1 1]))
    m_fdir = Chain(Dense(3 => 10, relu), FDIR(10 => 2)) |> gpu
    loss_fd, grads_fd = Flux.withgradient(m_fdir) do m
        α, p, τ = splitfdir(m(x))
        sum(fdirloss(y_oh_fd, α, p, τ))
    end
    @test isfinite(loss_fd)
    @test !isnothing(grads_fd[1])

    # mveloss
    m_mve = Chain(Dense(3 => 10, relu), MVE(10 => 2)) |> gpu
    loss_m, grads_m = Flux.withgradient(m_mve) do m
        μ, σ = splitmve(m(x))
        sum(mveloss(y, μ, σ))
    end
    @test isfinite(loss_m)
    @test !isnothing(grads_m[1])
end

@testset "GPU - CPU/GPU output agreement" begin
    x_cpu = randn(Float32, 3, 5)
    x_gpu = cu(x_cpu)

    # NIG
    m_nig = NIG(3 => 2)
    ŷ_cpu = m_nig(x_cpu)
    ŷ_gpu = (m_nig |> gpu)(x_gpu)
    @test Array(ŷ_gpu) ≈ ŷ_cpu atol = 1.0f-5

    # DIR
    m_dir = DIR(3 => 4)
    ŷ_cpu = m_dir(x_cpu)
    ŷ_gpu = (m_dir |> gpu)(x_gpu)
    @test Array(ŷ_gpu) ≈ ŷ_cpu atol = 1.0f-5

    # MVE
    m_mve = MVE(3 => 2)
    ŷ_cpu = m_mve(x_cpu)
    ŷ_gpu = (m_mve |> gpu)(x_gpu)
    @test Array(ŷ_gpu) ≈ ŷ_cpu atol = 1.0f-5

    # FDIR
    m_fdir = FDIR(3 => 4)
    ŷ_cpu = m_fdir(x_cpu)
    ŷ_gpu = (m_fdir |> gpu)(x_gpu)
    @test Array(ŷ_gpu) ≈ ŷ_cpu atol = 1.0f-5
end
