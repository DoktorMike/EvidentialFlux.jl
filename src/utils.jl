
"""
    uncertainty(ν, α, β)

Calculates the epistemic uncertainty of the predictions from the Normal Inverse Gamma (NIG) model.
Given a ``NIG(γ, υ, α, β)`` distribution we can calculate the epistemic uncertainty as

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
Given a ``NIG(γ, υ, α, β)`` distribution we can calculate the aleatoric uncertainty as

``\\mathbb{E}[σ^2] = \\frac{β}{(α-1)}``

# Arguments:
- `α`: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `β`: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)
"""
uncertainty(α, β) = @. β / (α - 1)

"""
    predict(m, x)

Returns the predictions along with the epistemic and aleatoric uncertainty.

# Arguments:
- `m`: the model which has to have the last layer be Normal Inverse Gamma(NIG) layer
- `x`: the input data which has to be given as an array or vector
"""
predict(m, x) = predict(typeof(m.layers[end]), m, x)

function predict(::Type{<:NIG}, m, x)
    #(pred = γ, eu = uncertainty(ν, α, β), au = uncertainty(α, β))
    nout = Int(size(m[end].W)[1] / 4)
    ŷ = m(x)
    γ, ν, α, β = ŷ[1:nout, :], ŷ[(nout+1):(2*nout), :], ŷ[(2*nout+1):(3*nout), :], ŷ[(3*nout+1):(4*nout), :]
    #return γ, uncertainty(ν, α, β), uncertainty(α, β)
    γ, ν, α, β
end
