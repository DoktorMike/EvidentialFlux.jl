using EvidentialFlux
using Flux
using Flux.Optimise: ADAM
using UnicodePlots
using Statistics


const epochs = 10000
const lr = 0.001

function gendata()
        x = Float32.(collect(-2π:0.1:2π))
        y = Float32.(sin.(x) .+ 0.3 * randn(size(x)))
        #scatterplot(x, y)
        x, y
end

"""
    predict(m, x)

Predicts the output of the model m on the input x.
"""
function predict(m, x)
        ŷ = m(x)
        γ, ν, α, β = ŷ[1, :], ŷ[2, :], ŷ[3, :], ŷ[4, :]
        (pred=γ, eu=uncertainty(ν, α, β), au=uncertainty(α, β))
end

mae(y, ŷ) = Statistics.mean(abs.(y - ŷ))

x, y = gendata()
p = scatterplot(x, y, width = 80, height = 30)
lines!(p, x, sin.(x))

m = Chain(Dense(1 => 100, tanh), NIG(100 => 1))
#m(x')
opt = ADAM(lr)
pars = Flux.params(m)
trnlosses = zeros(epochs)
for epoch in 1:epochs
        local trnloss = 0
        grads = Flux.gradient(pars) do
                ŷ = m(x')
                γ, ν, α, β = ŷ[1, :], ŷ[2, :], ŷ[3, :], ŷ[4, :]
                trnloss = Statistics.mean(nigloss(y, γ, ν, α, β, 0.01, 1e-4))
                trnloss
        end
        trnlosses[epoch] = trnloss
        # Test that we can update the weights based on gradients
        Flux.Optimise.update!(opt, pars, grads)
end
# The convergance plot shows the loss function converges to a local minimum
scatterplot(1:epochs, trnlosses, width = 80)
# And the MAE corresponds to the noise we added in the target
ŷ, u, au = predict(m, x')
println("MAE: $(mae(y, ŷ))")

# Correlation plot confirms the fit
p = scatterplot(y, ŷ, width = 80, height = 30, marker = "o");
lines!(p, -2:0.01:2, -2:0.01:2)

p = scatterplot(x, y, width = 80, height = 30, marker = "o");
scatterplot!(p, x, ŷ, color = :red, marker = "x");
scatterplot!(p, x, u)

p = scatterplot(x, ŷ, marker = :x, width = 80, height = 30, color = :red);
scatterplot!(p, x, y, marker = :x, color = :blue)

## Out of sample predictions

x = Float32.(collect(0:0.1:3π));
ŷ, u, au = predict(m, x');

p = scatterplot(x, sin.(x), width = 80, height = 30, marker = "o");
scatterplot!(p, x, ŷ, color = :red, marker = "x");
scatterplot!(p, x, u)
scatterplot!(p, x, au)
