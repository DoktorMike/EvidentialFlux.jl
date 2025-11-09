using Flux
using EvidentialFlux
using Statistics
using GLMakie
using ProgressMeter

function plotprediction!(fig, model, x, y)
    γ, ν, α, β = predict(model, x')
    eu = epistemic(ν)
    au = aleatoric(ν, α, β)
    # Plot it
    ax = fig.content[1]
    empty!(ax)
    scatter!(ax, x, y)
    scatter!(ax, x, γ[1, :])
    band!(ax, x, γ[1, :] - eu[1, :], γ[1, :] + eu[1, :], alpha = 0.2)
    band!(ax, x, γ[1, :] - au[1, :], γ[1, :] + au[1, :], alpha = 0.2)
    ylims!(ax, -220, 220)
    vlines!(ax, [-4, 4])
    ax = fig.content[2]
    empty!(ax)
    ev = evidence(ν, α)
    scatter!(ax, x, ev[1, :])
    ax = fig.content[3]
    empty!(ax)
    scatter!(ax, x, α[1, :])
    return fig
end

xtrn = Float32.(-4:0.1:4)
ytrn = xtrn .^ 3 .+ randn(Float32, length(xtrn)) .* 5
xtst = vcat(Float32.(-6:0.1:-4), Float32.(4:0.1:6))
ytst = xtst .^ 3 .+ randn(Float32, length(xtst)) .* 5

# For total plots and predictions do not use for training
x = vcat(xtrn, xtst)
y = vcat(ytrn, ytst)
inds = sortperm(x)
x = x[inds]
y = y[inds]

fig = Figure()
ax = Axis(fig[1:2, 1], title = "Cubic Regression")
ax2 = Axis(fig[3, 1])
ax3 = Axis(fig[4, 1])

lr = 0.001
model = Chain(Dense(1 => 100, relu), LayerNorm(100), Dense(100 => 100, relu), LayerNorm(100), Dense(100 => 100, relu), LayerNorm(100), NIG(100 => 1))
plotprediction!(fig, model, x, y)
# opt = OptimiserChain(ClipGrad(1.0e-2), Flux.AdamW(lr))
opt = Flux.Adam(lr)
opt_state = Flux.setup(opt, model)  # will store optimiser momentum, etc.
losses = []
@showprogress for epoch in 1:50_000
    loss, grads = Flux.withgradient(model) do m
        ŷ = m(xtrn')
        γ, ν, α, β = ŷ[1, :], ŷ[2, :], ŷ[3, :], ŷ[4, :]
        sum(nigloss(ytrn, γ, ν, α, β, 0.001))
        # Statistics.mean(nigloss3(y, γ, ν, α, β, 1, 1))
        # Statistics.mean(nigloss2(ytrn, γ, ν, α, β, 0.001, 2))
        # Statistics.mean(nigloss(ytrn, γ, ν, α, β, 0.001))
        # sum(nigloss2(ytrn, γ, ν, α, β, 5, 2))
        # sum(nigloss3(ytrn, γ, ν, α, β, 0.1, 0.1)) # Works sort of sometimes
        # sum(nigloss3(ytrn, γ, ν, α, β, 0.0, 0.0)) # Works the best, i.e., only use the NLL of Student T predictive distribution
    end
    Flux.update!(opt_state, model, grads[1])
    push!(losses, loss)
    if epoch % 500 == 0
        plotprediction!(fig, model, x, y)
        # readline()
    end
end
plotprediction!(fig, model, x, y)
