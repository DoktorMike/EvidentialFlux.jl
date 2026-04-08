# EvidentialFlux

[![Documentation](https://github.com/DoktorMike/EvidentialFlux.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/DoktorMike/EvidentialFlux.jl/actions/workflows/documentation.yml)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://doktormike.github.io/EvidentialFlux.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://doktormike.github.io/EvidentialFlux.jl/dev)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![DOI](https://zenodo.org/badge/487830887.svg)](https://zenodo.org/badge/latestdoi/487830887)

A Julia/Flux implementation of the Evidential Deep Learning framework. Estimate
heteroskedastic aleatoric and epistemic uncertainty in a single forward pass.

## Installing

```julia
using Pkg; Pkg.add(url="https://github.com/DoktorMike/EvidentialFlux.jl")
```

## Features

EvidentialFlux provides three output layer types, each suited to a different
modeling scenario:

| Layer | Use case | Output | Uncertainty |
|-------|----------|--------|-------------|
| `NIG(in => out)` | Regression | γ, ν, α, β (4 × out) | Aleatoric + epistemic |
| `DIR(in => out)` | Classification | Dirichlet concentrations (out) | Epistemic |
| `MVE(in => out)` | Regression | μ, σ (2 × out) | Aleatoric |

### Loss functions

| Function | Description |
|----------|-------------|
| `nigloss(y, γ, ν, α, β, λ, ϵ)` | Standard evidential regression loss (Amini et al. 2020) |
| `nigloss2(y, γ, ν, α, β, λ, p)` | Corrected DER loss (Meinert et al. 2022) |
| `nigloss3(y, γ, ν, α, β, λ, λ₁)` | Uncertainty regularized loss (Ye et al. 2024) |
| `dirloss(y, α, t)` | Dirichlet classification loss with KL regularization |
| `mveloss(y, μ, σ)` | Gaussian negative log-likelihood |
| `nllstudent(y, γ, ν, α, β)` | Student-T negative log-likelihood |

### Utilities

| Function | Description |
|----------|-------------|
| `predict(model, x)` | Unified prediction dispatching on the last layer type |
| `splitnig(y)` | Split concatenated NIG output into (γ, ν, α, β) |
| `uncertainty(ν, α, β)` | Epistemic uncertainty: β / (ν(α-1)) |
| `uncertainty(α, β)` | Aleatoric uncertainty: β / (α-1) |
| `uncertainty(α)` | DIR epistemic uncertainty: K / Σα |
| `epistemic(ν)` | Simplified epistemic: 1/√ν |
| `aleatoric(ν, α, β)` | Student-T std: β(1+ν) / (να) |
| `evidence(ν, α)` | NIG total evidence: 2ν + α |
| `evidence(α)` | DIR evidence: α - 1 |

## Quick start

### Evidential regression (NIG)

```julia
using Flux, EvidentialFlux, Statistics

x = Float32.(-4:0.1:4)
y = x .^ 3 .+ randn(Float32, length(x)) .* 3

model = Chain(Dense(1 => 100, relu), Dense(100 => 100, relu), NIG(100 => 1))
opt_state = Flux.setup(AdamW(1e-3), model)

for epoch in 1:3000
    loss, grads = Flux.withgradient(model) do m
        γ, ν, α, β = splitnig(m(x'))
        mean(nigloss(y, γ, ν, α, β, 0.001))
    end
    Flux.update!(opt_state, model, grads[1])
end

# Extract predictions and uncertainty
γ, ν, α, β = predict(model, x')
eu = epistemic(ν)
au = aleatoric(ν, α, β)
```

### Mean-variance estimation (MVE)

```julia
model = Chain(Dense(1 => 100, relu), Dense(100 => 100, relu), MVE(100 => 1))
opt_state = Flux.setup(AdamW(1e-3), model)

for epoch in 1:3000
    loss, grads = Flux.withgradient(model) do m
        μ, σ = predict(m, x')
        mean(mveloss(y, μ, σ))
    end
    Flux.update!(opt_state, model, grads[1])
end
```

## Classification

Deep evidential modeling works for classification as well. The plot below shows
epistemic uncertainty when separating three Gaussians in 2D. See
[classification.jl](examples/classification.jl).

![uncertainty](images/threegaussians.png)

## Regression

For regression, the NormalInverseGamma distribution models a type II likelihood
that explicitly captures aleatoric and epistemic uncertainty. See
[regression.jl](examples/regression.jl).

![uncertainty](images/cubefun.png)

## Examples

The [examples/](examples/) folder contains complete working examples:

- [regression.jl](examples/regression.jl) -- NIG with `nigloss`
- [regression2.jl](examples/regression2.jl) -- NIG with `nigloss2` (corrected DER)
- [regression3.jl](examples/regression3.jl) -- NIG with LayerNorm and evidence tracking
- [regression4.jl](examples/regression4.jl) -- MVE with parameter freezing/thawing
- [classification.jl](examples/classification.jl) -- DIR for multi-class classification

## References

- Amini, A., Schwarting, W., Soleimany, A. & Rus, D. Deep Evidential Regression. NeurIPS (2020).
- Meinert, N., Gawlikowski, J. & Lavin, A. The Unreasonable Effectiveness of Deep Evidential Regression. arXiv (2022).
- Ye, K., Chen, T., Wei, H. & Zhan, L. Uncertainty Regularized Evidential Regression. AAAI 38 (2024).
- Sensoy, M., Kaplan, L. & Kandemir, M. Evidential Deep Learning to Quantify Classification Uncertainty. NeurIPS (2018).

## Summary

Uncertainty is crucial for the deployment and utilization of robust machine
learning in production. No model is perfect and each one of them has
strengths and weaknesses, but as a minimum requirement, we should all
at least demand that our models report uncertainty in every prediction.
