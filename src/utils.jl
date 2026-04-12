"""
    split_params(::Type{<:NIG}, y)

Split NIG layer output into a NamedTuple `(γ, ν, α, β)`.

    split_params(::Type{<:MVE}, y)

Split MVE layer output into a NamedTuple `(μ, σ)`.

    split_params(::Type{<:DIR}, y)

Wrap DIR layer output into a NamedTuple `(α,)`.

    split_params(::Type{<:PG}, y)

Split PG layer output into a NamedTuple `(α, β)`.

    split_params(::Type{<:EG}, y)

Split EG layer output into a NamedTuple `(α, β)`.

    split_params(::Type{<:BB}, y)

Split BB layer output into a NamedTuple `(α, β)`.

    split_params(::Type{<:BNB}, y)

Split BNB layer output into a NamedTuple `(r, α, β)`.

    split_params(::Type{<:FDIR}, y)

Split FDIR layer output into a NamedTuple `(α, p, τ)`.
The first K rows are α, the next K rows are p, and the last row is τ,
where `K = (size(y,1) - 1) ÷ 2`.
"""
split_params(::Type{<:NIG}, y) = let (γ, ν, α, β) = _split_equal(y, 4)
    (γ = γ, ν = ν, α = α, β = β)
end
split_params(::Type{<:PG}, y) = let (α, β) = _split_equal(y, 2)
    (α = α, β = β)
end
split_params(::Type{<:EG}, y) = let (α, β) = _split_equal(y, 2)
    (α = α, β = β)
end
split_params(::Type{<:BB}, y) = let (α, β) = _split_equal(y, 2)
    (α = α, β = β)
end
split_params(::Type{<:BNB}, y) = let (r, α, β) = _split_equal(y, 3)
    (r = r, α = α, β = β)
end
split_params(::Type{<:ZIP}, y) = let (α_π, β_π, α_λ, β_λ) = _split_equal(y, 4)
    (α_π = α_π, β_π = β_π, α_λ = α_λ, β_λ = β_λ)
end
split_params(::Type{<:VM}, y) = let (μ₀, κ₀, κ) = _split_equal(y, 3)
    (μ₀ = μ₀, κ₀ = κ₀, κ = κ)
end
split_params(::Type{<:MVE}, y) = let (μ, σ) = _split_equal(y, 2)
    (μ = μ, σ = σ)
end
split_params(::Type{<:DIR}, y) = (α = y,)
function split_params(::Type{<:FDIR}, y)
    K = (size(y, 1) - 1) ÷ 2
    α = y[1:K, :]
    p = y[(K + 1):(2 * K), :]
    τ = y[(2 * K + 1):(2 * K + 1), :]
    return (α = α, p = p, τ = τ)
end

"""
    splitnig(y)

Splits the concatenated output of a NIG layer into its four components: γ, ν, α, β.
The input `y` should have shape `(nout*4, batch...)` where `nout` is the number of
output neurons.

# Arguments:
- `y`: the concatenated NIG output with shape `(nout*4, batch...)`

# Returns:
- `(γ, ν, α, β)`: tuple of arrays each with shape `(nout, batch...)`
"""
splitnig(y) = let p = split_params(NIG, y); (p.γ, p.ν, p.α, p.β) end

"""
    splitmve(y)

Splits the concatenated output of an MVE layer into its two components: μ, σ.
The input `y` should have shape `(nout*2, batch...)` where `nout` is the number of
output neurons.

# Arguments:
- `y`: the concatenated MVE output with shape `(nout*2, batch...)`

# Returns:
- `(μ, σ)`: tuple of arrays each with shape `(nout, batch...)`
"""
splitmve(y) = let p = split_params(MVE, y); (p.μ, p.σ) end

"""
    splitfdir(y)

Splits the concatenated output of an FDIR layer into its three components: α, p, τ.
The input `y` should have shape `(K*2 + 1, batch...)` where `K` is the number of classes.

# Arguments:
- `y`: the concatenated FDIR output with shape `(K*2 + 1, batch...)`

# Returns:
- `(α, p, τ)`: tuple where α and p have shape `(K, batch...)` and τ has shape `(1, batch...)`
"""
splitfdir(y) = let r = split_params(FDIR, y); (r.α, r.p, r.τ) end

"""
    splitpg(y)

Splits the concatenated output of a PG layer into its two components: α, β.
The input `y` should have shape `(nout*2, batch...)`.

# Arguments:
- `y`: the concatenated PG output with shape `(nout*2, batch...)`

# Returns:
- `(α, β)`: tuple of arrays each with shape `(nout, batch...)`
"""
splitpg(y) = let p = split_params(PG, y); (p.α, p.β) end

"""
    spliteg(y)

Splits the concatenated output of an EG layer into its two components: α, β.
The input `y` should have shape `(nout*2, batch...)`.

# Arguments:
- `y`: the concatenated EG output with shape `(nout*2, batch...)`

# Returns:
- `(α, β)`: tuple of arrays each with shape `(nout, batch...)`
"""
spliteg(y) = let p = split_params(EG, y); (p.α, p.β) end

"""
    splitbb(y)

Splits the concatenated output of a BB layer into its two components: α, β.
The input `y` should have shape `(nout*2, batch...)`.

# Arguments:
- `y`: the concatenated BB output with shape `(nout*2, batch...)`

# Returns:
- `(α, β)`: tuple of arrays each with shape `(nout, batch...)`
"""
splitbb(y) = let p = split_params(BB, y); (p.α, p.β) end

"""
    splitbnb(y)

Splits the concatenated output of a BNB layer into its three components: r, α, β.
The input `y` should have shape `(nout*3, batch...)`.

# Arguments:
- `y`: the concatenated BNB output with shape `(nout*3, batch...)`

# Returns:
- `(r, α, β)`: tuple of arrays each with shape `(nout, batch...)`
"""
splitbnb(y) = let p = split_params(BNB, y); (p.r, p.α, p.β) end

"""
    splitzip(y)

Splits the concatenated output of a ZIP layer into its four components: α_π, β_π, α_λ, β_λ.
The input `y` should have shape `(nout*4, batch...)`.

# Arguments:
- `y`: the concatenated ZIP output with shape `(nout*4, batch...)`

# Returns:
- `(α_π, β_π, α_λ, β_λ)`: tuple of arrays each with shape `(nout, batch...)`
"""
splitzip(y) = let p = split_params(ZIP, y); (p.α_π, p.β_π, p.α_λ, p.β_λ) end

"""
    splitvm(y)

Splits the concatenated output of a VM layer into its three components: μ₀, κ₀, κ.
The input `y` should have shape `(nout*3, batch...)`.

# Arguments:
- `y`: the concatenated VM output with shape `(nout*3, batch...)`

# Returns:
- `(μ₀, κ₀, κ)`: tuple of arrays each with shape `(nout, batch...)`
"""
splitvm(y) = let p = split_params(VM, y); (p.μ₀, p.κ₀, p.κ) end

"""
    uncertainty(ν, α, β)

Calculates the epistemic uncertainty of the predictions from the Normal Inverse Gamma (NIG) model.
Given a ``\\text{N-}\\Gamma^{-1}(γ, υ, α, β)`` distribution we can calculate the epistemic uncertainty as

``Var[μ] = \\frac{β}{ν(α-1)}``

# Arguments:
- `ν`: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `α`: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `β`: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)
"""
uncertainty(ν, α, β) = @. β / (ν * (α - 1))

"""
    uncertainty(α, β)

Calculates the aleatoric uncertainty of the predictions from the Normal Inverse Gamma (NIG) model.
Given a ``\\text{N-}\\Gamma^{-1}(γ, υ, α, β)`` distribution we can calculate the aleatoric uncertainty as

``\\mathbb{E}[σ^2] = \\frac{β}{(α-1)}``

# Arguments:
- `α`: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `β`: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)
"""
uncertainty(α, β) = @. β / (α - 1)

"""
    uncertainty(α)

Calculates the epistemic uncertainty associated with a MultinomialDirichlet model (DIR) layer.

- `α`: the α parameter of the Dirichlet distribution which relates to it's concentrations and whose shape should be (O, B)
"""
uncertainty(α) = first(size(α)) ./ sum(α, dims = 1)

"""
    evidence(α)

Calculates the total evidence of assigning each observation in α to the respective class for a DIR layer.

- `α`: the α parameter of the Dirichlet distribution which relates to it's concentrations and whose shape should be (O, B)
"""
evidence(α) = α .- 1

"""
    evidence(ν, α)

Returns the evidence for the data pushed through the NIG layer. In this setting one way of looking at the NIG
distribution is as ν virtual observations governing the mean μ of the likelihood and α virtual observations governing the variance ``\\sigma^2``. The evidence is then a sum of the virtual observations. Amini et. al. goes through this interpretation in their 2020 paper.

# Arguments:
- `ν`: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `α`: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
"""
evidence(ν, α) = @. 2ν + α

"""
    aleatoric(ν, α, β)

This is the aleatoric uncertainty as recommended by Meinert, Nis, Jakob
Gawlikowski, and Alexander Lavin. 'The Unreasonable Effectiveness of Deep
Evidential Regression.' arXiv, May 20, 2022. http://arxiv.org/abs/2205.10060.
This is precisely the ``σ_{St}`` from the Student T distribution.

# Arguments:
- `ν`: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `α`: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `β`: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)
"""
aleatoric(ν, α, β) = @. (β * (1 + ν)) / (ν * α)

"""
    epistemic(ν)

This is the epistemic uncertainty as recommended by Meinert, Nis, Jakob
Gawlikowski, and Alexander Lavin. 'The Unreasonable Effectiveness of Deep
Evidential Regression.' arXiv, May 20, 2022. http://arxiv.org/abs/2205.10060.

# Arguments:
- `ν`: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
"""
epistemic(ν) = 1 ./ sqrt.(ν)

# --- Type-dispatched uncertainty ---

"""
    epistemic(::Type{<:NIG}, ν, α, β)

Epistemic uncertainty for the NIG model: `1/√ν` (Meinert et al. 2022).
"""
epistemic(::Type{<:NIG}, ν, α, β) = epistemic(ν)

"""
    aleatoric(::Type{<:NIG}, ν, α, β)

Aleatoric uncertainty for the NIG model: the Student-T standard deviation
`σ_St = β(1+ν)/(να)` (Meinert et al. 2022).
"""
aleatoric(::Type{<:NIG}, ν, α, β) = aleatoric(ν, α, β)

"""
    uncertainty(::Type{<:NIG}, ν, α, β)

Epistemic uncertainty for the NIG model: `Var[μ] = β/(ν(α-1))`.
"""
uncertainty(::Type{<:NIG}, ν, α, β) = uncertainty(ν, α, β)

"""
    epistemic(::Type{<:DIR}, α)

Epistemic uncertainty for the Dirichlet model: `K/Σα`.
"""
epistemic(::Type{<:DIR}, α) = uncertainty(α)

"""
    epistemic(::Type{<:MVE}, σ)

Aleatoric uncertainty for the MVE model: the predicted variance `σ` itself.
MVE has no epistemic uncertainty — it only models aleatoric.
"""
aleatoric(::Type{<:MVE}, σ) = σ

"""
    epistemic(::Type{<:EG}, α, β)

Epistemic uncertainty for the Exponential-Gamma model: the variance of the
expected duration `E[Y|λ] = 1/λ` under the Gamma prior.

    Var[1/λ] = β² / ((α-1)²(α-2))

Requires α > 2 for the moments to exist; α is clamped internally.
"""
function epistemic(::Type{<:EG}, α, β)
    α_c = max.(α, 2 .+ eps(eltype(α)))
    return @. β^2 / ((α_c - 1)^2 * (α_c - 2))
end

"""
    aleatoric(::Type{<:EG}, α, β)

Aleatoric uncertainty for the Exponential-Gamma model: the expected Exponential
variance under the Gamma prior.

    E[Var[Y|λ]] = E[1/λ²] = β² / ((α-1)(α-2))

Requires α > 2 for the moments to exist; α is clamped internally.
"""
function aleatoric(::Type{<:EG}, α, β)
    α_c = max.(α, 2 .+ eps(eltype(α)))
    return @. β^2 / ((α_c - 1) * (α_c - 2))
end

"""
    epistemic(::Type{<:BB}, α, β)

Epistemic uncertainty for the Binomial-Beta model: the variance of the
success probability under the Beta prior.

    Var[p] = αβ / ((α+β)²(α+β+1))
"""
epistemic(::Type{<:BB}, α, β) = @. α * β / ((α + β)^2 * (α + β + 1))

"""
    aleatoric(::Type{<:BB}, α, β)

Aleatoric uncertainty for the Binomial-Beta model: the expected Bernoulli
variance under the Beta prior.

    E[p(1-p)] = αβ / ((α+β)(α+β+1))
"""
aleatoric(::Type{<:BB}, α, β) = @. α * β / ((α + β) * (α + β + 1))

"""
    epistemic(::Type{<:PG}, α, β)

Epistemic uncertainty for the Poisson-Gamma model: the variance of the
Poisson rate under the Gamma prior, `Var[λ] = α/β²`.
"""
epistemic(::Type{<:PG}, α, β) = @. α / β^2

"""
    aleatoric(::Type{<:PG}, α, β)

Aleatoric uncertainty for the Poisson-Gamma model: the expected Poisson
variance, `E[Var[Y|λ]] = E[λ] = α/β`.
"""
aleatoric(::Type{<:PG}, α, β) = @. α / β

"""
    epistemic(::Type{<:BNB}, r, α, β)

Epistemic uncertainty for the Beta-Negative Binomial model: the variance of
the conditional mean `E[Y|p] = rp/(1-p)` under the Beta prior.

    Var[E[Y|p]] = r²·α(α+β-1) / ((β-1)²(β-2))

Requires β > 2 for the moments to exist; β is clamped internally.
"""
function epistemic(::Type{<:BNB}, r, α, β)
    β_c = max.(β, 2 .+ eps(eltype(β)))
    return @. r^2 * α * (α + β_c - 1) / ((β_c - 1)^2 * (β_c - 2))
end

"""
    aleatoric(::Type{<:BNB}, r, α, β)

Aleatoric uncertainty for the Beta-Negative Binomial model: the expected
NB variance under the Beta prior.

    E[Var[Y|p]] = r·α(α+β-1) / ((β-1)(β-2))

Requires β > 2 for the moments to exist; β is clamped internally.
"""
function aleatoric(::Type{<:BNB}, r, α, β)
    β_c = max.(β, 2 .+ eps(eltype(β)))
    return @. r * α * (α + β_c - 1) / ((β_c - 1) * (β_c - 2))
end

"""
    epistemic(::Type{<:ZIP}, α_π, β_π, α_λ, β_λ)

Epistemic uncertainty for the Zero-Inflated Poisson model: the variance of the
conditional mean `E[Y|π,λ] = (1-π)λ` under the independent Beta and Gamma priors.

    Var[(1-π)λ] = E[(1-π)²]E[λ²] - (E[1-π])²(E[λ])²

where `E[(1-π)²] = β_π(β_π+1) / (S_π(S_π+1))` and `E[λ²] = α_λ(α_λ+1)/β_λ²`.
"""
function epistemic(::Type{<:ZIP}, α_π, β_π, α_λ, β_λ)
    S_π = α_π .+ β_π
    E_1mπ = β_π ./ S_π
    E_1mπ² = β_π .* (β_π .+ 1) ./ (S_π .* (S_π .+ 1))
    E_λ = α_λ ./ β_λ
    E_λ² = α_λ .* (α_λ .+ 1) ./ β_λ .^ 2
    return @. E_1mπ² * E_λ² - (E_1mπ * E_λ)^2
end

"""
    aleatoric(::Type{<:ZIP}, α_π, β_π, α_λ, β_λ)

Aleatoric uncertainty for the Zero-Inflated Poisson model: the expected
variance of the ZIP observation given the parameters.

    E[Var[Y|π,λ]] = E[1-π]·E[λ] + E[π(1-π)]·E[λ²]

where `E[π(1-π)] = α_π β_π / (S_π(S_π+1))`.
"""
function aleatoric(::Type{<:ZIP}, α_π, β_π, α_λ, β_λ)
    S_π = α_π .+ β_π
    E_1mπ = β_π ./ S_π
    E_π1mπ = α_π .* β_π ./ (S_π .* (S_π .+ 1))
    E_λ = α_λ ./ β_λ
    E_λ² = α_λ .* (α_λ .+ 1) ./ β_λ .^ 2
    return @. E_1mπ * E_λ + E_π1mπ * E_λ²
end

"""
    epistemic(::Type{<:VM}, κ₀)

Epistemic uncertainty for the Von Mises model: the circular variance of the
prior on the mean direction μ.

    CV[μ] = 1 - A(κ₀)

where `A(κ) = I₁(κ)/I₀(κ)` is the mean resultant length. Ranges from 0
(certain, κ₀ → ∞) to 1 (uniform on the circle, κ₀ → 0).
"""
function epistemic(::Type{<:VM}, κ₀)
    Ix₀ = SpecialFunctions.besselix.(0, κ₀)
    Ix₁ = SpecialFunctions.besselix.(1, κ₀)
    return 1 .- Ix₁ ./ Ix₀
end

"""
    aleatoric(::Type{<:VM}, κ)

Aleatoric uncertainty for the Von Mises model: the circular variance of the
observation noise.

    CV[θ|μ] = 1 - A(κ)

where `A(κ) = I₁(κ)/I₀(κ)` is the mean resultant length.
"""
function aleatoric(::Type{<:VM}, κ)
    Ix₀ = SpecialFunctions.besselix.(0, κ)
    Ix₁ = SpecialFunctions.besselix.(1, κ)
    return 1 .- Ix₁ ./ Ix₀
end

"""
    epistemic(::Type{<:FDIR}, α, p, τ)

Epistemic uncertainty for the Flexible Dirichlet model (Yoon & Kim 2025).
Returns a `(1, B)` scalar per sample:

    EU = Σₖ [μₖ(1-μₖ)/(S+1) + τ²pₖ(1-pₖ)/(S(S+1))]

where `μₖ = (αₖ+τpₖ)/S` and `S = Σαₖ + τ`.
"""
function epistemic(::Type{<:FDIR}, α, p, τ)
    S = sum(α, dims = 1) .+ τ
    μ = (α .+ τ .* p) ./ S
    return sum(μ .* (1 .- μ) ./ (S .+ 1) .+ τ .^ 2 .* p .* (1 .- p) ./ (S .* (S .+ 1)), dims = 1)
end

"""
    aleatoric(::Type{<:FDIR}, α, p, τ)

Aleatoric uncertainty for the Flexible Dirichlet model: `AU = TU - EU` where
`TU = 1 - Σₖ μₖ²` is the total uncertainty. Returns `(1, B)`.
"""
function aleatoric(::Type{<:FDIR}, α, p, τ)
    S = sum(α, dims = 1) .+ τ
    μ = (α .+ τ .* p) ./ S
    tu = 1 .- sum(μ .^ 2, dims = 1)
    eu = epistemic(FDIR, α, p, τ)
    return tu .- eu
end

"""
    predict(m, x)

Returns the predictions along with the available epistemic and aleatoric uncertainty.
Dispatches on the last layer type of the model:
- **NIG**: returns `(γ, ν, α, β)` NamedTuple
- **MVE**: returns `(μ, σ)` NamedTuple
- **DIR**: returns α directly (raw array, for backward compatibility)

# Arguments:
- `m`: the model whose last layer is an `AbstractEvidentialLayer`
- `x`: the input data which has to be given as an array or vector
"""
predict(m, x) = predict(last_type(m), m, x)
last_type(m::Chain) = last_type(m[end])
last_type(m) = typeof(m)

predict(::Type{T}, m, x) where {T <: AbstractEvidentialLayer} = split_params(T, m(x))
predict(::Type{<:DIR}, m, x) = m(x)

# --- Predictive mean (data-space point predictions) ---

"""
    predictive_mean(::Type{<:NIG}, params)
    predictive_mean(::Type{<:PG}, params)
    predictive_mean(::Type{<:BNB}, params)
    predictive_mean(::Type{<:DIR}, params)
    predictive_mean(::Type{<:FDIR}, params)
    predictive_mean(::Type{<:MVE}, params)

Returns the point prediction in data space given the raw distributional
parameters. This is the mean of the posterior predictive distribution.
"""
predictive_mean(::Type{<:NIG}, p) = p.γ
predictive_mean(::Type{<:BB}, p, n = 1) = n .* p.α ./ (p.α .+ p.β)
predictive_mean(::Type{<:PG}, p) = p.α ./ p.β
function predictive_mean(::Type{<:EG}, p)
    α_c = max.(p.α, 1 .+ eps(eltype(p.α)))
    return p.β ./ (α_c .- 1)
end
predictive_mean(::Type{<:BNB}, p) = p.r .* p.α ./ p.β
predictive_mean(::Type{<:ZIP}, p) = p.β_π ./ (p.α_π .+ p.β_π) .* p.α_λ ./ p.β_λ
predictive_mean(::Type{<:VM}, p) = p.μ₀
predictive_mean(::Type{<:DIR}, α, n = 1) = n .* α ./ sum(α, dims = 1)
predictive_mean(::Type{<:FDIR}, p, n = 1) = n .* (p.α .+ p.τ .* p.p) ./ (sum(p.α, dims = 1) .+ p.τ)
predictive_mean(::Type{<:MVE}, p) = p.μ

# --- Predictive output (inference-time bundle) ---

"""
    predictive(m, x)

Inference-time prediction returning a NamedTuple with:
- `ŷ`: point prediction in data space (posterior predictive mean)
- `epistemic`: epistemic uncertainty (`nothing` if not available for this layer)
- `aleatoric`: aleatoric uncertainty (`nothing` if not available for this layer)
- `params`: raw distributional parameters from `predict`

Use `predict` during training (returns raw parameters for loss computation).
Use `predictive` at inference time for a complete uncertainty-aware output.

# Examples
```julia
r = predictive(model, x)
r.ŷ          # point prediction
r.epistemic  # model uncertainty
r.aleatoric  # data noise
r.params     # raw (γ, ν, α, β) etc. for advanced use
```
"""
predictive(m, x) = predictive(last_type(m), m, x)

function predictive(::Type{T}, m, x) where {T <: NIG}
    p = predict(T, m, x)
    return (ŷ = predictive_mean(T, p),
        epistemic = epistemic(T, p.ν, p.α, p.β),
        aleatoric = aleatoric(T, p.ν, p.α, p.β),
        params = p)
end

function predictive(::Type{T}, m, x) where {T <: BB}
    p = predict(T, m, x)
    return (ŷ = predictive_mean(T, p),
        epistemic = epistemic(T, p.α, p.β),
        aleatoric = aleatoric(T, p.α, p.β),
        params = p)
end

function predictive(::Type{T}, m, x) where {T <: EG}
    p = predict(T, m, x)
    return (ŷ = predictive_mean(T, p),
        epistemic = epistemic(T, p.α, p.β),
        aleatoric = aleatoric(T, p.α, p.β),
        params = p)
end

function predictive(::Type{T}, m, x) where {T <: PG}
    p = predict(T, m, x)
    return (ŷ = predictive_mean(T, p),
        epistemic = epistemic(T, p.α, p.β),
        aleatoric = aleatoric(T, p.α, p.β),
        params = p)
end

function predictive(::Type{T}, m, x) where {T <: BNB}
    p = predict(T, m, x)
    return (ŷ = predictive_mean(T, p),
        epistemic = epistemic(T, p.r, p.α, p.β),
        aleatoric = aleatoric(T, p.r, p.α, p.β),
        params = p)
end

function predictive(::Type{T}, m, x) where {T <: ZIP}
    p = predict(T, m, x)
    return (ŷ = predictive_mean(T, p),
        epistemic = epistemic(T, p.α_π, p.β_π, p.α_λ, p.β_λ),
        aleatoric = aleatoric(T, p.α_π, p.β_π, p.α_λ, p.β_λ),
        params = p)
end

function predictive(::Type{T}, m, x) where {T <: VM}
    p = predict(T, m, x)
    return (ŷ = predictive_mean(T, p),
        epistemic = epistemic(T, p.κ₀),
        aleatoric = aleatoric(T, p.κ),
        params = p)
end

function predictive(::Type{T}, m, x) where {T <: DIR}
    α = predict(T, m, x)
    return (ŷ = predictive_mean(T, α),
        epistemic = epistemic(T, α),
        aleatoric = nothing,
        params = α)
end

function predictive(::Type{T}, m, x) where {T <: FDIR}
    p = predict(T, m, x)
    return (ŷ = predictive_mean(T, p),
        epistemic = epistemic(T, p.α, p.p, p.τ),
        aleatoric = aleatoric(T, p.α, p.p, p.τ),
        params = p)
end

function predictive(::Type{T}, m, x) where {T <: MVE}
    p = predict(T, m, x)
    return (ŷ = predictive_mean(T, p),
        epistemic = nothing,
        aleatoric = aleatoric(T, p.σ),
        params = p)
end
