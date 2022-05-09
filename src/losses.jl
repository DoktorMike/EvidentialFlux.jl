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
        # NLL: Calculate the negative log likelihood of the Normal-Inverse-Gamma distribution
        twoβλ = 2 * β .* (1 .+ ν)
        logγ = SpecialFunctions.loggamma
        nll = 0.5 * log.(π ./ ν) -
              α .* log.(twoβλ) +
              (α .+ 0.5) .* log.(ν .* (y - γ) .^ 2 + twoβλ) +
              logγ.(α) -
              logγ.(α .+ 0.5)
        nll

        # REG: Calculate regularizer based on absolute error of prediction
        error = abs.(y - γ)
        reg = error .* (2 * ν + α)

        # Combine negative log likelihood and regularizer
        loss = nll + λ .* (reg .- ϵ)
        loss
end
