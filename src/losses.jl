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
    nll
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
function nigloss(y, γ, ν, α, β, λ = 1, ϵ = 1e-4)
    nll = nllstudent(y, γ, ν, α, β)
    # REG: Calculate regularizer based on absolute error of prediction
    error = abs.(y - γ)
    Φ = evidence(ν, α) # Total evidence
    reg = error .* Φ
    # Combine negative log likelihood and regularizer
    loss = nll + λ .* (reg .- ϵ)
    loss
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
    loss
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
    kl
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
    # Total loss = likelihood + regularizer
    #sum(loss .+ λₜ .* reg, dims = 2)
    sum(loss .+ λₜ .* reg)
end
