using Flux
using EvidentialFlux
using Statistics
using GLMakie
using ProgressMeter

function plotprediction!(fig, model, x, y)
    μ, σ = predict(model, x')
    # Plot it
    ax = fig.content[1]
    empty!(ax)
    scatter!(ax, x, y)
    scatter!(ax, x, μ[1, :])
    band!(ax, x, μ[1, :] - σ[1, :], μ[1, :] + σ[1, :], alpha = 0.2)
    ylims!(ax, -220, 220)
    vlines!(ax, [-4, 4])
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

lr = 0.001
model = Chain(
    Dense(1 => 100, relu), LayerNorm(100),
    Dense(100 => 100, relu), LayerNorm(100),
    Dense(100 => 100, relu), LayerNorm(100),
    MVE(100 => 1, identity)
)
# NOTE: Kill the uncertainty part by forcing 0 on all weights
# model.layers[end].chain.layers[1].layers.σw.weight .= 0.0
# model.layers[end].chain.layers[1].layers.σw.bias .= 0.0
plotprediction!(fig, model, x, y)
# opt = OptimiserChain(ClipGrad(1.0e-2), Flux.AdamW(lr))
opt = Flux.Adam(lr)
opt_state = Flux.setup(opt, model)  # will store optimiser momentum, etc.
losses = []
# NOTE: Freeze uncertainty param
# Flux.freeze!(opt_state.layers[end][1][1][1][2].σw)
@showprogress for epoch in 1:50_000
    loss, grads = Flux.withgradient(model) do m
        ŷ = m(xtrn')
        μ, σ = ŷ[1, :], ŷ[2, :]
        sum(mveloss(ytrn, μ, σ, 10))
    end
    Flux.update!(opt_state, model, grads[1])
    push!(losses, loss)
    if epoch % 500 == 0
        plotprediction!(fig, model, x, y)
        # readline()
    end
    if epoch == 20_000
        # NOTE: Thaw uncertainty param
        Flux.thaw!(opt_state.layers[end][1][1][1][2].σw)
    end
end
plotprediction!(fig, model, x, y)
