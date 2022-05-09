
"""
    uncertainty(ν, α, β)

Calculates the epistemic uncertainty of the predictions from the Normal Inverse Gamma (NIG) model.

# Arguments:
- `ν`: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `α`: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `β`: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)
"""
uncertainty(ν, α, β) = @. β / (ν * (α - 1))

"""
    uncertainty(α, β)

Calculates the aleatoric uncertainty of the predictions from the Normal Inverse Gamma (NIG) model.

# Arguments:
- `α`: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)
- `β`: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)
"""
uncertainty(α, β) = @. β / (α - 1)

