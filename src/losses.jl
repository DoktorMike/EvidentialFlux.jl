
function nignll(y, γ, ν, α, β, λ = 1, ϵ = 1e-4)
        # Calculate the negative log likelihood of the Normal-Inverse-Gamma distribution
        twobl = 2 * β .* (1 .+ ν)
        logγ = SpecialFunctions.loggamma
        nll = 0.5 * log.(π ./ ν) -
              α .* log.(twobl) +
              (α .+ 0.5) .* log.(ν .* (y - γ) .^ 2 + twobl) +
              logγ.(α) -
              logγ.(α .+ 0.5)
        nll

        # Calculate regularizer based on absolute error of prediction
        error = abs.(y - γ)
        reg = error .* (2 * ν + α)

        # Combine negative log likelihood and regularizer
        loss = nll + λ .* (reg .- ϵ)
        loss
end
