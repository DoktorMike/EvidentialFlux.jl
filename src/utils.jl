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

Returns the predictions along with the epistemic and aleatoric uncertainty.

# Arguments:
- `m`: the model which has to have the last layer be Normal Inverse Gamma(NIG) layer
- `x`: the input data which has to be given as an array or vector
"""
predict(m, x) = predict(last_type(m), m, x)
last_type(m::Chain) = last_type(m[end])
last_type(m) = typeof(m)

function predict(::Type{<:NIG}, m, x)
    #(pred = γ, eu = uncertainty(ν, α, β), au = uncertainty(α, β))
    nout = Int(size(m[end].W)[1] / 4)
    ŷ = m(x)
    γ, ν, α, β = ŷ[1:nout, :], ŷ[(nout + 1):(2 * nout), :],
    ŷ[(2 * nout + 1):(3 * nout), :], ŷ[(3 * nout + 1):(4 * nout), :]
    #return γ, uncertainty(ν, α, β), uncertainty(α, β)
    γ, ν, α, β
end

function predict(::Type{<:MVE}, m, x)
    nout = Int(size(m[end].W)[1] / 2)
    ŷ = m(x)
    μ, σ = ŷ[1:nout, :], ŷ[(nout + 1):(2 * nout), :]
    μ, σ
end

function predict(::Type{<:DIR}, m, x)
    ŷ = m(x)
    ŷ
end
