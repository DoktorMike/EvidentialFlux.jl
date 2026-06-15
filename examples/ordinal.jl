# Evidential ordinal regression with FDIR + ofdirloss.
#
# Ordinal targets have ordered classes (Very Low < Low < Medium < High <
# Very High). Two things matter that plain classification ignores:
#
#   1. Order — predicting "Very High" when the truth is "Very Low" should cost
#      more than predicting "High". `ofdirloss` scores the cumulative CDF
#      (expected Ranked Probability Score), so rank-distant errors cost more.
#   2. Shape — some inputs have a *bimodal* conditional: the answer is either
#      Very Low or Very High, rarely the middle. FDIR is a mixture of
#      Dirichlets and represents this; a structurally unimodal model
#      (Beta-Binomial, CORAL) collapses to the rare middle and is wrong.
#
# This example builds a dataset where one input group is bimodal and the other
# is unimodal, trains an FDIR model, and shows that both conditional shapes are
# recovered.

using EvidentialFlux
using Flux
using Random
using GLMakie

# --- Synthetic ordinal data ----------------------------------------------
# Group A: bimodal / U-shaped conditional (mass at both extremes).
# Group B: unimodal conditional concentrated on the high end.
const K = 5
const pA = Float32[0.45, 0.03, 0.04, 0.03, 0.45]
const pB = Float32[0.0, 0.05, 0.05, 0.2, 0.7]

rng = MersenneTwister(1)
sampcat(pr) = findfirst(cumsum(pr) .>= rand(rng))

function gendata(n)
    g = rand(rng, Bool, n)                      # group flag
    X = Float32.(vcat(g', .!g'))                # 2×n one-hot group indicator
    labels = [g[i] ? sampcat(pB) : sampcat(pA) for i in 1:n]
    Y = zeros(Float32, K, n)
    for i in 1:n
        Y[labels[i], i] = 1
    end
    return X, Y
end

N = 1500
X, Y = gendata(N)

# Inverse-frequency class weights, available for imbalanced problems. Each
# sample's loss is scaled by the weight of its true class. NOTE: weighting
# re-balances per-class recall but deliberately distorts the learned
# *conditional* away from the data distribution — so we train UNWEIGHTED below
# to recover the true bimodal shape. Pass `weights` to `ofdirloss` when class
# balance matters more than calibrated conditional probabilities.
freq = vec(sum(Y, dims = 2)) ./ N
weights = Float32.(1 ./ (freq .+ 1.0f-3))
weights ./= sum(weights) / K                    # normalize so mean weight ≈ 1

# --- Model ----------------------------------------------------------------
model = Chain(Dense(2 => 32, relu), Dense(32 => 32, relu), FDIR(32 => K))
opt_state = Flux.setup(Flux.AdamW(0.01), model)

# --- Train ----------------------------------------------------------------
epochs = 2000
trnlosses = zeros(Float32, epochs)
for e in 1:epochs
    loss, grads = Flux.withgradient(model) do m
        α, p, τ = splitfdir(m(X))
        # Add `; weights` below to counter class imbalance (see note above).
        sum(ofdirloss(Y, α, p, τ)) / N
    end
    trnlosses[e] = loss
    Flux.update!(opt_state, model, grads[1])
end

# --- Inspect recovered conditionals --------------------------------------
# Predicted Flexible-Dirichlet mean = per-level probabilities.
fdmean(pr) = vec((pr.α .+ pr.τ .* pr.p) ./ (sum(pr.α, dims = 1) .+ pr.τ))
μA = fdmean(predict(model, reshape(Float32[0.0; 1.0], 2, 1)))   # bimodal group
μB = fdmean(predict(model, reshape(Float32[1.0; 0.0], 2, 1)))   # unimodal group

println("Group A (bimodal)  true: ", pA)
println("Group A (bimodal)  pred: ", round.(μA, digits = 3))
println("Group B (unimodal) true: ", pB)
println("Group B (unimodal) pred: ", round.(μB, digits = 3))

# Expected ordinal level = Σ level · P(level). For the bimodal group this lands
# near the middle (≈2) even though the middle is the *least* likely outcome —
# a reminder to report the full distribution / uncertainty for ordinal data,
# not just a point level.
levelvals = Float32.(0:(K - 1))
println("Group A expected level: ", round(sum(levelvals .* μA), digits = 3))
println("Group B expected level: ", round(sum(levelvals .* μB), digits = 3))

# Inference-time bundle. For FDIR, `ŷ` is the per-level probability vector
# (K × B); `epistemic` and `aleatoric` are per-sample scalars (1 × B). The high
# aleatoric for the bimodal group reflects genuine answer ambiguity.
rA = predictive(model, reshape(Float32[0.0; 1.0], 2, 1))
println(
    "Group A  epistemic=", round(only(rA.epistemic), digits = 3),
    "  aleatoric=", round(only(rA.aleatoric), digits = 3)
)

# --- Plots ----------------------------------------------------------------
levels = ["VLow", "Low", "Med", "High", "VHigh"]

fig = Figure(size = (900, 350))

ax1 = Axis(
    fig[1, 1], title = "Group A (bimodal): true vs predicted",
    xticks = (1:K, levels), ylabel = "P(level)"
)
barplot!(ax1, (1:K) .- 0.2, pA, width = 0.4, label = "true")
barplot!(ax1, (1:K) .+ 0.2, μA, width = 0.4, label = "pred")
axislegend(ax1)

ax2 = Axis(
    fig[1, 2], title = "Group B (unimodal): true vs predicted",
    xticks = (1:K, levels), ylabel = "P(level)"
)
barplot!(ax2, (1:K) .- 0.2, pB, width = 0.4, label = "true")
barplot!(ax2, (1:K) .+ 0.2, μB, width = 0.4, label = "pred")
axislegend(ax2)

ax3 = Axis(fig[1, 3], title = "Training loss", xlabel = "epoch", ylabel = "loss")
lines!(ax3, 1:epochs, trnlosses)

fig
