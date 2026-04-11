"""
    split_params(::Type{<:NIG}, y)

Split NIG layer output into a NamedTuple `(γ, ν, α, β)`.

    split_params(::Type{<:MVE}, y)

Split MVE layer output into a NamedTuple `(μ, σ)`.

    split_params(::Type{<:DIR}, y)

Wrap DIR layer output into a NamedTuple `(α,)`.
"""
split_params(::Type{<:NIG}, y) = let (γ, ν, α, β) = _split_equal(y, 4)
    (γ = γ, ν = ν, α = α, β = β)
end
split_params(::Type{<:MVE}, y) = let (μ, σ) = _split_equal(y, 2)
    (μ = μ, σ = σ)
end
split_params(::Type{<:DIR}, y) = (α = y,)

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
