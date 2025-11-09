# EvidentialFlux

[![Documentation](https://github.com/DoktorMike/EvidentialFlux.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/DoktorMike/EvidentialFlux.jl/actions/workflows/documentation.yml)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://doktormike.github.io/EvidentialFlux.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://doktormike.github.io/EvidentialFlux.jl/dev)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![DOI](https://zenodo.org/badge/487830887.svg)](https://zenodo.org/badge/latestdoi/487830887)

This is a Julia implementation in Flux of the Evidential Deep Learning
framework. It strives to estimate heteroskedastic aleatoric uncertainty as well
as epistemic uncertainty along with every prediction made. All of it calculated
in one glorious forward pass. Boom!

## Installing

If you want bleeding edge you can install it directly from my repo like this:

```julia
using Pkg; Pkg.add(url="https://github.com/DoktorMike/EvidentialFlux.jl")
```

The package is not registered so unfortunately

```julia
using Pkg; Pkg.add("EvidentialFlux.jl")
```

will not work.

## For the impatient

Below is an example of how to train Deep Evidential Regression model, extract
the predictions as well as the epistemic and aleatoric uncertainty. For a more
elaborate example have a look in the examples folder.

```julia
using Flux
using EvidentialFlux
using Statistics

xtrn = Float32.(-4:0.1:4)
ytrn = xtrn .^3 .+ randn(Float32, length(xtrn)) .* 3
xtst = vcat(Float32.(-6:0.1:-4), Float32.(4:0.1:6))
ytst = xtst .^3 .+ randn(Float32, length(xtst)) .* 3

fig = Figure()
ax = Axis(fig[1,1])

lr = 0.001
model = Chain(Dense(1 => 100, relu), Dense(100 => 100, relu), Dense(100 => 100, relu), NIG(100 => 1))
opt_state = Flux.setup(Flux.AdamW(lr), model)  # will store optimiser momentum, etc.
losses = []
for epoch in 1:3000
    loss, grads = Flux.withgradient(model) do m
        ŷ = m(xtrn')
        γ, ν, α, β = ŷ[1, :], ŷ[2, :], ŷ[3, :], ŷ[4, :]
        # Statistics.mean(nigloss3(y, γ, ν, α, β, 1, 1))
        # Statistics.mean(nigloss2(ytrn, γ, ν, α, β, 0.001, 2))
        Statistics.mean(nigloss(ytrn, γ, ν, α, β, 0.001))
    end
    Flux.update!(opt_state, model, grads[1])
    push!(losses, loss)
end

function plotprediction()
    x = vcat(xtrn, xtst)
    y = vcat(ytrn, ytst)
    inds = sortperm(x)
    x = x[inds]
    y = y[inds]
    γ, ν, α, β = predict(model, x')
    eu = epistemic(ν)
    au = aleatoric(ν, α, β)
    # Plot it
    empty!(ax)
    scatter!(ax, x, y)
    scatter!(ax, x, γ[1,:])
    band!(ax, x, γ[1,:] - eu[1,:], γ[1,:] + eu[1,:], alpha=0.2)
    band!(ax, x, γ[1,:] - au[1,:], γ[1,:] + au[1,:], alpha=0.2)
    ylims!(ax, -220,220)
    vlines!(ax, [-4,4])
end


```

## Classification

Deep evidential modeling works for classification as well as for regression. In
the plot below you can see the epistemic uncertainty as a consequence of
position in the plot. The task is to separate three Gaussians in 2D. The code
for this example can be found in
[classification.jl](examples/classification.jl).

![uncertainty](images/threegaussians.png)

## Regression

In the case of a regression problem, we utilize the NormalInverseGamma
distribution to model a type II likelihood function that then explicitly
models the aleatoric and epistemic uncertainty. The code for the example
producing the plot below can be found in
[regression.jl](examples/regression.jl).

![uncertainty](images/cubefun.png)

## Summary

Uncertainty is crucial for the deployment and utilization of robust machine
learning in production. No model is perfect and each one of them has
strengths and weaknesses, but as a minimum requirement, we should all
at least demand that our models report uncertainty in every prediction.
