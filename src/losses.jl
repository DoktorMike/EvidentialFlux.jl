# --- Bessel function helpers (numerically stable, AD-compatible) ---

"""
    _logI₀(x::Real)

Numerically stable computation of `log(I₀(x))` where `I₀` is the modified
Bessel function of the first kind of order 0. Uses the scaled Bessel function
`besselix` to avoid overflow for large `x`. A custom `rrule` is provided
since `besseli` lacks AD support in current SpecialFunctions.jl.
"""
_logI₀(x::Real) = log(SpecialFunctions.besselix(0, x)) + abs(x)

function ChainRulesCore.rrule(::typeof(_logI₀), x::Real)
    Ix₀ = SpecialFunctions.besselix(0, x)
    y = log(Ix₀) + abs(x)
    _logI₀_pb(ȳ) = (NoTangent(), ȳ * SpecialFunctions.besselix(1, x) / Ix₀)
    return y, _logI₀_pb
end

"""
    nllstudent(y, γ, ν, α, β)

Returns the negative log likelihood of the StudentT distribution which in this
case is the model evidence for a gaussian likelihood with a normal inverse
gamma prior.

# Arguments:
- `y`: the targets whose shape should be (O, B)
- `γ`: the γ parameter of the NIG distribution which corresponds to it's mean and whose shape should be (O, B)
- `ν`: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `α`: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `β`: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)
"""
function nllstudent(y, γ, ν, α, β)
    Ω = 2 * β .* (1 .+ ν)
    logγ = SpecialFunctions.loggamma
    nll = 0.5 * log.(π ./ ν) -
        α .* log.(Ω) +
        (α .+ 0.5) .* log.(ν .* (y - γ) .^ 2 + Ω) +
        logγ.(α) -
        logγ.(α .+ 0.5)
    return nll
end

"""
    _nig_nll_reg(y, γ, ν, α, β)

Internal helper that computes the shared components of the NIG loss functions:
the negative log likelihood, absolute prediction error, and evidence regularizer.
"""
function _nig_nll_reg(y, γ, ν, α, β)
    nll = nllstudent(y, γ, ν, α, β)
    error = abs.(y - γ)
    reg = error .* evidence(ν, α)
    return nll, error, reg
end

"""
    nigloss(y, γ, ν, α, β, λ = 1, ϵ = 0.0001)

This is the standard loss function for Evidential Inference given a
NormalInverseGamma posterior for the parameters of the gaussian likelihood
function: μ and σ.

# Arguments:
- `y`: the targets whose shape should be (O, B)
- `γ`: the γ parameter of the NIG distribution which corresponds to it's mean and whose shape should be (O, B)
- `ν`: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `α`: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `β`: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)
- `λ`: the weight to put on the regularizer (default: 1)
- `ϵ`: the threshold for the regularizer (default: 0.0001)
"""
function nigloss(y, γ, ν, α, β, λ = 1, ϵ = 1.0e-4)
    nll, _, reg = _nig_nll_reg(y, γ, ν, α, β)
    return nll + λ .* (reg .- ϵ)
end

"""
    nigloss_scaled(y, γ, ν, α, β, λ = 1, p = 1)

Corrected DER loss from Meinert, Gawlikowski & Lavin, "The Unreasonable
Effectiveness of Deep Evidential Regression" (2022). Normalizes the prediction
error by the aleatoric uncertainty before scaling by evidence, preventing the
network from inflating variance to reduce the regularizer.

# Arguments:
- `y`: the targets whose shape should be (O, B)
- `γ`: the γ parameter of the NIG distribution which corresponds to it's mean and whose shape should be (O, B)
- `ν`: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `α`: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `β`: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)
- `λ`: the weight to put on the regularizer (default: 1)
- `p`: the power which to raise the scaled absolute prediction error (default: 1)
"""
function nigloss_scaled(y, γ, ν, α, β, λ = 1, p = 1)
    nll = nllstudent(y, γ, ν, α, β)
    uₐ = aleatoric(ν, α, β)
    error = (abs.(y - γ) ./ uₐ) .^ p
    Φ = evidence(ν, α)
    reg = error .* Φ
    return nll + λ * reg
end

"""
    nigloss_ureg(y, γ, ν, α, β, λ = 1, λ₁ = 1)

Uncertainty-regularized evidential regression loss from Ye, Chen, Wei & Zhan,
"Uncertainty Regularized Evidential Regression" (AAAI 2024). Adds a term that
ensures non-zero gradients in high-uncertainty regions where the standard
regularizer's gradient vanishes.

# Arguments:
- `y`: the targets whose shape should be (O, B)
- `γ`: the γ parameter of the NIG distribution which corresponds to it's mean and whose shape should be (O, B)
- `ν`: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `α`: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `β`: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)
- `λ`: the weight to put on the evidence regularizer (default: 1)
- `λ₁`: the weight to put on the uncertainty loss (default: 1)
"""
function nigloss_ureg(y, γ, ν, α, β, λ = 1, λ₁ = 1)
    nll, error, reg = _nig_nll_reg(y, γ, ν, α, β)
    unc = .- error .* log.(exp.(α .- 1) .- 1)
    return nll .+ λ .* reg .+ λ₁ .* unc
end

# Deprecated aliases — use nigloss_scaled and nigloss_ureg instead
"""
    nigloss2

Deprecated alias for [`nigloss_scaled`](@ref). Use `nigloss_scaled` in new code.
"""
const nigloss2 = nigloss_scaled

"""
    nigloss3

Deprecated alias for [`nigloss_ureg`](@ref). Use `nigloss_ureg` in new code.
"""
const nigloss3 = nigloss_ureg

# The α here is actually the α̃ which has scaled down evidence that is good.
# the α heres is a matrix of size (K, B) or (O, B)
function kl(α)
    ψ = SpecialFunctions.digamma
    lnΓ = SpecialFunctions.loggamma
    K = first(size(α))
    # Actual computation
    ∑α = sum(α, dims = 1)
    ∑lnΓα = sum(lnΓ.(α), dims = 1)
    A = lnΓ.(∑α) .- lnΓ(K) .- ∑lnΓα
    B = sum((α .- 1) .* (ψ.(α) .- ψ.(∑α)), dims = 1)
    kl = A + B
    return kl
end

"""
    dirloss(y, α, t)

Regularized version of a type II maximum likelihood for the Multinomial(p)
distribution where the parameter p, which follows a Dirichlet distribution has
been integrated out.

# Arguments:
- `y`: the targets whose shape should be (O, B)
- `α`: the parameters of a Dirichlet distribution representing the belief in each class which shape should be (O, B)
- `t`: counter for the current epoch being evaluated
"""
function dirloss(y, α, t)
    S = sum(α, dims = 1)
    p̂ = α ./ S
    # Main loss
    loss = (y - p̂) .^ 2 .+ p̂ .* (1 .- p̂) ./ (S .+ 1)
    loss = sum(loss, dims = 1)
    # Regularizer
    λₜ = min(1.0, t / 10)
    # Keep only misleading evidence, i.e., penalize stuff that fit badly
    α̂ = @. y + (1 - y) * α
    reg = kl(α̂)
    # Total loss = likelihood + regularizer, shape (1, B) — one loss per batch element
    return loss .+ λₜ .* reg
end

"""
    dirloss_cor(y, α, t)

Dirichlet classification loss with correct evidence regularization from Pandey,
Choi & Yu, "Generalized Regularized Evidential Deep Learning Models" (2025).

Extends `dirloss` with an additional term `ℒ_cor` that prevents gradient
vanishing when the ground-truth class has low evidence (the "learning freeze"
problem). The correction is weighted by the vacuity `ν = K/S` and only active
when the pre-activation logit for the ground-truth class is negative (i.e.,
evidence below the softplus inflection point).

The total loss is `ℒ_evid + λₜ·ℒ_inc + ℒ_cor` where `ℒ_cor = -𝟙(o_gt < 0)·ν·o_gt`.

# Arguments:
- `y`: one-hot encoded targets, shape `(K, B)`
- `α`: Dirichlet concentration parameters from a DIR layer, shape `(K, B)`
- `t`: current epoch (used for KL annealing on `ℒ_inc`)
"""
function dirloss_cor(y, α, t)
    base = dirloss(y, α, t)
    # ℒ_cor: Correct evidence regularization
    K = first(size(α))
    S = sum(α, dims = 1)
    e_gt = sum(y .* (α .- 1), dims = 1)                    # GT class evidence, (1, B)
    o_gt = log.(expm1.(max.(e_gt, eps(eltype(α)))))         # inverse softplus, clamped
    ν = ignore_derivatives(K ./ S)                           # vacuity (detached)
    cor = .- (o_gt .< 0) .* ν .* o_gt
    return base .+ cor
end

# Deprecated alias — use dirloss_cor instead
"""
    dirloss2

Deprecated alias for [`dirloss_cor`](@ref). Use `dirloss_cor` in new code.
"""
const dirloss2 = dirloss_cor

"""
    dirmultloss(y, α)

Negative log-likelihood of the Dirichlet-Multinomial distribution, obtained by
integrating out `p ~ Dir(α)` from `Multinomial(y | n, p)`. Use this with the
`DIR` layer when targets are count vectors (e.g., word counts, event tallies)
rather than one-hot categories.

    p(y|α) = [n!/Πₖyₖ!] · B(y+α)/B(α)

where `n = Σyₖ`, `S = Σαₖ`, and B is the multivariate Beta function.

Unlike `dirloss` (which uses a Bayes Risk MSE + KL regularizer for one-hot
targets), this is a proper type II maximum likelihood loss that needs no
additional regularization.

# Arguments:
- `y`: non-negative count targets, shape `(K, B)` where `K` is the number of categories
- `α`: Dirichlet concentration parameters from a DIR layer, shape `(K, B)`
"""
function dirmultloss(y, α)
    logΓ = SpecialFunctions.loggamma
    n = sum(y, dims = 1)                                                    # (1, B)
    S = sum(α, dims = 1)                                                    # (1, B)
    nll = .- logΓ.(n .+ 1) .+ sum(logΓ.(y .+ 1), dims = 1) .+
        logΓ.(n .+ S) .- logΓ.(S) .+
        sum(logΓ.(α) .- logΓ.(y .+ α), dims = 1)
    return nll
end

"""
    mveloss(y, μ, σ)

Calculates the Mean-Variance loss for a Normal distribution. This is merely the negative log likelihood.
This loss should be used with the MVE network type.

# Arguments:
- `y`: targets
- `μ`: the predicted mean
- `σ`: the predicted variance
"""
mveloss(y, μ, σ) = (1 / 2) * (((y - μ) .^ 2) ./ σ + log.(σ))

"""
    mveloss(y, μ, σ, β)

Calculates the Mean-Variance loss for a Normal distribution. This is merely the negative log likelihood.
This loss should be used with the MVE network type.

# Arguments:
- `y`: targets
- `μ`: the predicted mean
- `σ`: the predicted variance
- `β`: used to increase or decrease the effect of the predicted variance on the loss
"""
mveloss(y, μ, σ, β) = mveloss(y, μ, σ) .* ignore_derivatives(σ) .^ β

"""
    fdirloss(y, α, p, τ)

Loss for the Flexible Dirichlet EDL model from Yoon & Kim, "Uncertainty
Estimation by Flexible Evidential Deep Learning" (2025).

Computes the expected Brier score under the Flexible Dirichlet distribution
plus a Brier score regularizer on the allocation probabilities `p`. The FD
distribution is a mixture of Dirichlets `Σⱼ pⱼ Dir(α + τeⱼ)`, and the loss
decomposes analytically as:

    ℒ = Σₖ [E_FD[πₖ²] - 2yₖ E[πₖ] + yₖ] + ‖y - p‖²

No manual hyperparameter tuning is needed for the regularization (unlike the
KL-based regularizer in `dirloss`).

# Arguments:
- `y`: one-hot encoded targets, shape `(K, B)`
- `α`: Gamma concentration parameters (> 0) from an FDIR layer, shape `(K, B)`
- `p`: allocation probabilities (Σp = 1) from an FDIR layer, shape `(K, B)`
- `τ`: shared dispersion parameter (> 0) from an FDIR layer, shape `(1, B)`
"""
function fdirloss(y, α, p, τ)
    S = sum(α, dims = 1) .+ τ                                          # (1, B)
    # E[πₖ²] under the FD mixture of Dirichlets
    Eπ² = (α .* (α .+ 1) .+ p .* τ .* (2 .* α .+ τ .+ 1)) ./ (S .* (S .+ 1))
    # E[πₖ] under FD
    μ = (α .+ τ .* p) ./ S
    # Expected Brier score: Σₖ [E[πₖ²] - 2·yₖ·E[πₖ] + yₖ]
    evid = sum(Eπ² .- 2 .* y .* μ .+ y, dims = 1)                     # (1, B)
    # Brier score regularizer on allocation probabilities
    brier = sum((y .- p) .^ 2, dims = 1)                               # (1, B)
    return evid .+ brier
end

"""
    nllpg(y, α, β)

Negative log-likelihood of the Negative Binomial marginal obtained by
integrating out the Poisson rate λ ~ Gamma(α, β):

    p(y|α,β) = Γ(y+α) / [Γ(y+1)·Γ(α)] · βᵅ / (β+1)^(y+α)

Use this with the `PG` layer for evidential count regression.

# Arguments:
- `y`: non-negative count targets, shape `(O, B)`
- `α`: Gamma shape parameter (> 0), shape `(O, B)`
- `β`: Gamma rate parameter (> 0), shape `(O, B)`
"""
function nllpg(y, α, β)
    logΓ = SpecialFunctions.loggamma
    return logΓ.(y .+ 1) .+ logΓ.(α) .- logΓ.(y .+ α) .- α .* log.(β) .+ (y .+ α) .* log.(β .+ 1)
end

"""
    pgloss(y, α, β, λ = 1)

Loss for Poisson-Gamma evidential count regression. Combines the Negative
Binomial NLL (from `nllpg`) with a regularizer that penalizes high confidence
(large α) when the predicted rate `α/β` is far from the observed count.

# Arguments:
- `y`: non-negative count targets, shape `(O, B)`
- `α`: Gamma shape parameter (> 0) from a PG layer, shape `(O, B)`
- `β`: Gamma rate parameter (> 0) from a PG layer, shape `(O, B)`
- `λ`: regularization weight (default: 1)
"""
function pgloss(y, α, β, λ = 1)
    nll = nllpg(y, α, β)
    reg = abs.(y .- α ./ β) .* α
    return nll .+ λ .* reg
end

"""
    nlleg(y, α, β)

Negative log-likelihood of the Lomax (Pareto Type II) marginal obtained by
integrating out `λ ~ Gamma(α, β)` from `Exp(y | λ)`:

    p(y|α,β) = α·βᵅ / (β+y)^(α+1)

Use this with the `EG` layer for evidential positive continuous regression.

# Arguments:
- `y`: positive continuous targets, shape `(O, B)`
- `α`: Gamma shape parameter (> 0), shape `(O, B)`
- `β`: Gamma rate parameter (> 0), shape `(O, B)`
"""
nlleg(y, α, β) = .- log.(α) .- α .* log.(β) .+ (α .+ 1) .* log.(β .+ y)

"""
    egloss(y, α, β, λ = 1)

Loss for Exponential-Gamma evidential positive regression. Combines the Lomax
NLL (from `nlleg`) with a regularizer that penalizes high confidence (large α)
when the predicted duration `β/(α-1)` is far from the observed value.

# Arguments:
- `y`: positive continuous targets, shape `(O, B)`
- `α`: Gamma shape parameter (> 0) from an EG layer, shape `(O, B)`
- `β`: Gamma rate parameter (> 0) from an EG layer, shape `(O, B)`
- `λ`: regularization weight (default: 1)
"""
function egloss(y, α, β, λ = 1)
    nll = nlleg(y, α, β)
    ŷ = β ./ max.(α, 1 .+ eps(eltype(α)))
    reg = abs.(y .- ŷ) .* α
    return nll .+ λ .* reg
end

"""
    nllbb(k, n, α, β)

Negative log-likelihood of the Beta-Binomial marginal obtained by integrating
out `p ~ Beta(α, β)` from `Binomial(k | n, p)`:

    p(k|n,α,β) = C(n,k) · B(k+α, n-k+β) / B(α,β)

Use this with the `BB` layer for evidential proportion estimation.

# Arguments:
- `k`: observed successes (non-negative), shape `(O, B)`
- `n`: number of trials (positive), shape `(O, B)`
- `α`: Beta shape parameter (> 0), shape `(O, B)`
- `β`: Beta shape parameter (> 0), shape `(O, B)`
"""
function nllbb(k, n, α, β)
    logΓ = SpecialFunctions.loggamma
    return .- logΓ.(n .+ 1) .+ logΓ.(k .+ 1) .+ logΓ.(n .- k .+ 1) .+
        logΓ.(n .+ α .+ β) .+ logΓ.(α) .+ logΓ.(β) .-
        logΓ.(k .+ α) .- logΓ.(n .- k .+ β) .- logΓ.(α .+ β)
end

"""
    bbloss(k, n, α, β, λ = 1)

Loss for Binomial-Beta evidential proportion estimation. Combines the
Beta-Binomial NLL (from `nllbb`) with a regularizer that penalizes high
confidence (large α+β) when the predicted probability `α/(α+β)` is far from
the observed proportion `k/n`.

# Arguments:
- `k`: observed successes (non-negative), shape `(O, B)`
- `n`: number of trials (positive), shape `(O, B)`
- `α`: Beta shape parameter (> 0) from a BB layer, shape `(O, B)`
- `β`: Beta shape parameter (> 0) from a BB layer, shape `(O, B)`
- `λ`: regularization weight (default: 1)
"""
function bbloss(k, n, α, β, λ = 1)
    nll = nllbb(k, n, α, β)
    p̂ = α ./ (α .+ β)
    reg = abs.(k ./ max.(n, eps(eltype(n))) .- p̂) .* (α .+ β)
    return nll .+ λ .* reg
end

"""
    nllbnb(y, r, α, β)

Negative log-likelihood of the Beta-Negative Binomial marginal obtained by
integrating out `p ~ Beta(α, β)` from `NB(y | r, p)`:

    p(y|r,α,β) = [Γ(y+r)/(Γ(y+1)Γ(r))] · B(y+α, r+β) / B(α, β)

Use this with the `BNB` layer for evidential overdispersed count regression.

# Arguments:
- `y`: non-negative count targets, shape `(O, B)`
- `r`: NB dispersion parameter (> 0), shape `(O, B)`
- `α`: Beta shape parameter (> 0), shape `(O, B)`
- `β`: Beta shape parameter (> 0), shape `(O, B)`
"""
function nllbnb(y, r, α, β)
    logΓ = SpecialFunctions.loggamma
    return logΓ.(y .+ 1) .+ logΓ.(r) .- logΓ.(y .+ r) .+
        logΓ.(y .+ α .+ r .+ β) .+ logΓ.(α) .+ logΓ.(β) .-
        logΓ.(y .+ α) .- logΓ.(r .+ β) .- logΓ.(α .+ β)
end

"""
    bnbloss(y, r, α, β, λ = 1)

Loss for Beta-Negative Binomial evidential count regression. Combines the
Beta-NB NLL (from `nllbnb`) with a regularizer that penalizes high confidence
(large α+β) when the predicted count `r·α/β` is far from the observed count.

# Arguments:
- `y`: non-negative count targets, shape `(O, B)`
- `r`: NB dispersion parameter (> 0) from a BNB layer, shape `(O, B)`
- `α`: Beta shape parameter (> 0) from a BNB layer, shape `(O, B)`
- `β`: Beta shape parameter (> 0) from a BNB layer, shape `(O, B)`
- `λ`: regularization weight (default: 1)
"""
function bnbloss(y, r, α, β, λ = 1)
    nll = nllbnb(y, r, α, β)
    ŷ = r .* α ./ β
    reg = abs.(y .- ŷ) .* (α .+ β)
    return nll .+ λ .* reg
end

"""
    nllzip(y, α_π, β_π, α_λ, β_λ)

Negative log-likelihood of the Zero-Inflated Negative Binomial marginal
obtained by integrating out π ~ Beta(α_π, β_π) and λ ~ Gamma(α_λ, β_λ)
from the ZIP(π, λ) likelihood:

    p(0|α_π,β_π,α_λ,β_λ) = E[π] + E[1-π]·p_NB(0|α_λ,β_λ)
    p(y|α_π,β_π,α_λ,β_λ) = E[1-π]·p_NB(y|α_λ,β_λ)   for y > 0

where `p_NB` is the Negative Binomial PMF from the Poisson-Gamma conjugacy
and `E[π] = α_π/(α_π+β_π)`.

Use this with the `ZIP` layer for evidential zero-inflated count regression.

# Arguments:
- `y`: non-negative count targets, shape `(O, B)`
- `α_π`: Beta shape parameter for zero-inflation (> 0), shape `(O, B)`
- `β_π`: Beta shape parameter for zero-inflation (> 0), shape `(O, B)`
- `α_λ`: Gamma shape parameter for Poisson rate (> 0), shape `(O, B)`
- `β_λ`: Gamma rate parameter for Poisson rate (> 0), shape `(O, B)`
"""
function nllzip(y, α_π, β_π, α_λ, β_λ)
    logΓ = SpecialFunctions.loggamma
    S_π = α_π .+ β_π

    # NegBin log-PMF from Poisson-Gamma conjugacy (= -nllpg)
    log_pNB = logΓ.(y .+ α_λ) .- logΓ.(y .+ 1) .- logΓ.(α_λ) .+
        α_λ .* log.(β_λ) .- (y .+ α_λ) .* log.(β_λ .+ 1)

    # Log mixing weights from Beta prior on π
    log_1mπ = log.(β_π) .- log.(S_π)       # log E[1-π]
    log_count = log_1mπ .+ log_pNB          # log of count component

    # For y=0: logaddexp(log E[π], log_count) via softplus identity
    # For y>0: log_count (zero-inflation term does not contribute)
    log_π = log.(α_π) .- log.(S_π)          # log E[π]
    is_zero = ignore_derivatives(y .== 0)
    ll = log_count .+ is_zero .* NNlib.softplus.(log_π .- log_count)

    return .-ll
end

"""
    ziploss(y, α_π, β_π, α_λ, β_λ, λ = 1)

Loss for Zero-Inflated Poisson evidential count regression. Combines the
ZINB NLL (from `nllzip`) with a regularizer that penalizes high confidence
when the predicted count is far from the observed count.

# Arguments:
- `y`: non-negative count targets, shape `(O, B)`
- `α_π`: Beta shape parameter for zero-inflation (> 0), shape `(O, B)`
- `β_π`: Beta shape parameter for zero-inflation (> 0), shape `(O, B)`
- `α_λ`: Gamma shape parameter for Poisson rate (> 0), shape `(O, B)`
- `β_λ`: Gamma rate parameter for Poisson rate (> 0), shape `(O, B)`
- `λ`: regularization weight (default: 1)
"""
function ziploss(y, α_π, β_π, α_λ, β_λ, λ = 1)
    nll = nllzip(y, α_π, β_π, α_λ, β_λ)
    ŷ = β_π ./ (α_π .+ β_π) .* α_λ ./ β_λ
    reg = abs.(y .- ŷ) .* (α_π .+ β_π .+ α_λ)
    return nll .+ λ .* reg
end

"""
    nllvm(θ, μ₀, κ₀, κ)

Negative log-likelihood of the Von Mises marginal obtained by integrating out
the mean direction `μ ~ VonMises(μ₀, κ₀)` from `VonMises(θ | μ, κ)`:

    p(θ|μ₀,κ₀,κ) = I₀(R) / (2π · I₀(κ) · I₀(κ₀))

where `R = √(κ² + κ₀² + 2κκ₀cos(θ-μ₀))` and `I₀` is the modified Bessel
function of the first kind of order 0.

Use this with the `VM` layer for evidential directional regression.

# Arguments:
- `θ`: angular targets in radians, shape `(O, B)`
- `μ₀`: prior mean direction (unconstrained), shape `(O, B)`
- `κ₀`: prior concentration parameter (> 0), shape `(O, B)`
- `κ`: observation concentration parameter (> 0), shape `(O, B)`
"""
function nllvm(θ, μ₀, κ₀, κ)
    R = sqrt.(κ .^ 2 .+ κ₀ .^ 2 .+ 2 .* κ .* κ₀ .* cos.(θ .- μ₀))
    return log(2π) .+ _logI₀.(κ) .+ _logI₀.(κ₀) .- _logI₀.(R)
end

"""
    vmloss(θ, μ₀, κ₀, κ, λ = 1)

Loss for Von Mises evidential directional regression. Combines the marginal
NLL (from `nllvm`) with a regularizer that penalizes high prior concentration
(κ₀) when the predicted direction is far from the observed angle, using the
circular distance `1 - cos(θ - μ₀)`.

# Arguments:
- `θ`: angular targets in radians, shape `(O, B)`
- `μ₀`: prior mean direction (unconstrained), shape `(O, B)`
- `κ₀`: prior concentration parameter (> 0), shape `(O, B)`
- `κ`: observation concentration parameter (> 0), shape `(O, B)`
- `λ`: regularization weight (default: 1)
"""
function vmloss(θ, μ₀, κ₀, κ, λ = 1)
    nll = nllvm(θ, μ₀, κ₀, κ)
    reg = (1 .- cos.(θ .- μ₀)) .* κ₀
    return nll .+ λ .* reg
end
