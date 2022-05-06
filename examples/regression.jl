using EvidentialFlux
using Flux
using Flux.Optimise: ADAM
using UnicodePlots
using Statistics


const epochs = 10000
const lr = 0.001

function gendata()
        x = Float32.(collect(0:0.1:2π))
        y = Float32.(sin.(x) .+ 0.3 * randn(size(x)))
        #scatterplot(x, y)
        x, y
end

function predict(m, x)
        ŷ = m(x)
        γ, ν, α, β = ŷ[1, :], ŷ[2, :], ŷ[3, :], ŷ[4, :]
        γ, uncertainty(ν, α, β)
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
                trnloss = Statistics.mean(nigloss(y, γ, ν, α, β, 0.5, 1e-4))
                trnloss
        end
        trnlosses[epoch] = trnloss
        # Test that we can update the weights based on gradients
        Flux.Optimise.update!(opt, pars, grads)
end
scatterplot(1:epochs, trnlosses, width = 80)
ŷ, u = predict(m, x')
mae(y, ŷ)

scatterplot(y, u)
scatterplot(y, ŷ)

p = scatterplot(x, ŷ, marker = :x, width = 80, height = 30, color = u);
scatterplot!(p, x, y, marker = :x, color = :blue)

