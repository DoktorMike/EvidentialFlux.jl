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
Based on Ye, K., Chen, T., Wei, H. & Zhan, L. Uncertainty Regularized
Evidential Regression. AAAI 38, 16460–16468 (2024). this loss function handles
training in high uncertainty areas by making sure the gradients are not 0. The
parameters are the same as in the other nigloss functions except that here we
have a `λ₁` controlling the extent we want to weight the uncertainty loss.
"""
function nigloss3(y, γ, ν, α, β, λ = 1, λ₁ = 1)
    nll, error, reg = _nig_nll_reg(y, γ, ν, α, β)
    unc = .- error .* log.(exp.(α .- 1) .- 1)
    return nll .+ λ .* reg .+ λ₁ .* unc
end


"""
    nigloss2(y, γ, ν, α, β, λ = 1, p = 1)

This is the corrected loss function for DER as recommended by Meinert, Nis,
Jakob Gawlikowski, and Alexander Lavin. “The Unreasonable Effectiveness of Deep
Evidential Regression.” arXiv, May 20, 2022. http://arxiv.org/abs/2205.10060.
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
- `p`: the power which to raise the scaled absolute prediction error (default: 1)
"""
function nigloss2(y, γ, ν, α, β, λ = 1, p = 1)
    nll = nllstudent(y, γ, ν, α, β)
    # REG: Calculate regularizer based on absolute error of prediction
    uₐ = aleatoric(ν, α, β)
    error = (abs.(y - γ) ./ uₐ) .^ p
    Φ = evidence(ν, α) # Total evidence
    reg = error .* Φ
    # Combine negative log likelihood and regularizer
    loss = nll + λ * reg
    return loss
end

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
    dirloss2(y, α, t)

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
function dirloss2(y, α, t)
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
