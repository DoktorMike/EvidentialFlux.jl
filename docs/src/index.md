# EvidentialFlux.jl

```@contents
```

Evidential Deep Learning is a way to generate predictions and the uncertainty
associated with them in one single forward pass. This is in stark contrast to
traditional Bayesian neural networks which are typically based on Variational
Inference, Markov Chain Monte Carlo, Monte Carlo Dropout or Ensembles.

!!! tip "New to EvidentialFlux?"
    See the [Choosing the Right Layer](@ref) guide for practical advice on
    which layer and loss to use for your problem, with real-world examples.

## How it works

The framework places a **conjugate prior** over the parameters of a
**likelihood** function. The neural network predicts the prior's
hyperparameters, and the **marginal likelihood** (prior integrated out) serves
as the training loss. This yields calibrated uncertainty in a single forward
pass.

| Layer | Likelihood | Prior | Marginal | Use case |
|-------|-----------|-------|----------|----------|
| [`NIG`](@ref) | Normal | Normal-Inverse-Gamma | Student-T | Real-valued regression |
| [`PG`](@ref) | Poisson | Gamma | Negative Binomial | Count regression |
| [`EG`](@ref) | Exponential | Gamma | Lomax | Positive continuous regression |
| [`BB`](@ref) | Binomial | Beta | Beta-Binomial | Proportion estimation |
| [`BNB`](@ref) | Negative Binomial | Beta | Beta-NB | Overdispersed counts |
| [`ZIP`](@ref) | Zero-Inflated Poisson | Beta × Gamma | Zero-Inflated NB | Zero-inflated counts |
| [`VM`](@ref) | Von Mises | Von Mises | Von Mises marginal | Directional/circular data |
| [`DIR`](@ref) | Categorical | Dirichlet | Dir-Multinomial | Classification |
| [`FDIR`](@ref) | Categorical | Flexible Dirichlet | Mixture of Dir-Mult | Calibrated classification |
| [`MVE`](@ref) | Normal | _(point estimate)_ | Normal | Simple variance estimation |

## Quick example

```julia
using Flux, EvidentialFlux

# Build a model with an evidential output layer
model = Chain(Dense(10 => 64, relu), Dense(64 => 64, relu), NIG(64 => 1))
opt_state = Flux.setup(AdamW(1e-3), model)

# Training: use predict + loss
for epoch in 1:1000
    loss, grads = Flux.withgradient(model) do m
        γ, ν, α, β = splitnig(m(x))
        mean(nigloss_scaled(y, γ, ν, α, β, 0.01))
    end
    Flux.update!(opt_state, model, grads[1])
end

# Inference: use predictive for the full picture
r = predictive(model, x_test)
r.ŷ          # predicted value
r.epistemic  # model uncertainty (high when extrapolating)
r.aleatoric  # data noise (high when data is inherently noisy)
```

## Deep Evidential Regression

Deep Evidential Regression[^amini2020] applies the principles of Evidential
Deep Learning to regression problems.

It works by putting a prior distribution over the likelihood parameters
``\mathbf{\theta} = \{\mu, \sigma^2\}`` governing a likelihood model where we
observe a dataset ``\mathcal{D}=\{x_i, y_i\}_{i=1}^N`` where ``y_i`` is
assumed to be drawn i.i.d. from a Gaussian distribution.

```math
y_i \sim \mathcal{N}(\mu_i, \sigma^2_i)
```

We can express the posterior parameters ``\mathbf{\theta}=\{\mu, \sigma^2\}``
as ``p(\mathbf{\theta}|\mathcal{D})``. We seek to create an approximation
``q(\mu, \sigma^2) = q(\mu)(\sigma^2)`` meaning that we assume that the
posterior factorizes. This means we can write
``\mu\sim\mathcal{N}(\gamma,\sigma^2\nu^{-1})`` and
``\sigma^2\sim\Gamma^{-1}(\alpha,\beta)``. Thus, we can now form

```math
p(\mathbf{\theta}|\mathbf{m})=\mathcal{N}(\gamma,\sigma^2\nu^{-1})\Gamma^{-1}(\alpha,\beta)=\mathcal{N-}\Gamma^{-1}(γ,υ,α,β)
```

which can be plugged in to the posterior below.

```math
p(\mathbf{\theta}|\mathbf{m}, y_i) = \frac{p(y_i|\mathbf{\theta}, \mathbf{m})p(\mathbf{\theta}|\mathbf{m})}{p(y_i|\mathbf{m})}
```

Now since the likelihood is Gaussian we would like to put a conjugate prior on
the parameters of that likelihood and the Normal Inverse Gamma
``\mathcal{N-}\Gamma^{-1}(γ, υ, α, β)`` fits the bill. This allows us to
express the prediction and the associated uncertainty as below.

```math
\underset{Prediction}{\underbrace{\mathbb{E}[\mu]=\gamma}}~~~~
\underset{Aleatoric}{\underbrace{\mathbb{E}[\sigma^2]=\frac{\beta}{\alpha-1}}}~~~~
\underset{Epistemic}{\underbrace{\text{Var}[\mu]=\frac{\beta}{\nu(\alpha-1)}}}
```

The `NIG` layer outputs 4 tensors for each target variable, namely
``\gamma,\nu,\alpha,\beta``. This means that in one forward pass we can
estimate the prediction, the heteroskedastic aleatoric uncertainty as well as
the epistemic uncertainty.

### NIG loss variants

Three loss functions are available for NIG regression, each improving on the
previous:

- [`nigloss`](@ref) — standard DER loss[^amini2020]
- [`nigloss_scaled`](@ref) — corrected DER that normalizes error by aleatoric uncertainty[^meinert2022]. **Recommended default.**
- [`nigloss_ureg`](@ref) — adds a term to prevent gradient vanishing in high-uncertainty regions[^ye2024]

## Deep Evidential Classification

We follow [^sensoy2018] in our implementation of Deep Evidential
Classification. The neural network layer is implemented to output the
``\alpha_k`` representing the parameters of a Dirichlet distribution. These
parameters has the additional interpretation ``\alpha_k = e_k + 1`` where
``e_k`` is the evidence for class ``k``. Further, it holds that ``e_k > 0``
which is the reason for us modeling them with a softplus activation function.

Since we are now constructing a network layer that outputs evidence for each
class we can apply Dempster-Shafer Theory (DST) to those outputs. DST is a
generalization of the Bayesian framework of thought and works by assigning
`belief mass` to states of interest. We can further concretize this notion by
Subjective Logic (SL) which places a Dirichlet distribution over these belief
masses. Belief masses are defined as ``b_k=e_k/S`` where ``e_k`` is the
evidence of state ``k`` and ``S=\sum_i^K(e_i+1)``. Further, SL requires that
``K+1`` states all sum up to 1. This practically means that
``u+\sum_k^K~b_k=1`` where ``u`` represents the uncertainty of the possible K
states, or the "I don't know." class.

Now, since ``S=\sum_i^K(e_i+1)=S=\sum_i^K(\alpha_i)`` SL refers to ``S`` as the
Dirichlet strength which is basically a sum of all the collected evidence in
favor of the ``K`` outcomes. Consequently the uncertainty ``u=K/S`` becomes 1
in case there is no evidence available. Therefor, ``u`` is a normalized
quantity ranging between 0 and 1.

### DIR loss variants

- [`dirloss`](@ref) — standard Bayes Risk MSE + KL regularizer[^sensoy2018]
- [`dirloss_cor`](@ref) — adds correct evidence regularization to fix gradient vanishing[^pandey2025]
- [`dirmultloss`](@ref) — Dirichlet-Multinomial NLL for count vector targets (reuses `DIR` layer)
- [`fdirloss`](@ref) — Flexible Dirichlet loss with Brier regularizer, no hyperparameter tuning[^yoon2025]

## API Reference

### Layers

```@docs
NIG
PG
EG
BB
BNB
ZIP
VM
DIR
FDIR
MVE
AbstractEvidentialLayer
```

### Loss functions — Regression

```@docs
nigloss
nigloss_scaled
nigloss_ureg
nllstudent
mveloss
```

### Loss functions — Count data

```@docs
pgloss
nllpg
egloss
nlleg
bbloss
nllbb
bnbloss
nllbnb
ziploss
nllzip
vmloss
nllvm
```

### Loss functions — Classification

```@docs
dirloss
dirloss_cor
dirmultloss
fdirloss
```

### Prediction

```@docs
predictive
predictive_mean
predict
split_params
splitnig
splitmve
splitpg
spliteg
splitbb
splitbnb
splitzip
splitvm
splitfdir
```

### Uncertainty and evidence

```@docs
epistemic
aleatoric
uncertainty
evidence
```

## Index

```@index
```

## References

[^amini2020]: Amini, Alexander, Wilko Schwarting, Ava Soleimany, and Daniela Rus. "Deep Evidential Regression." ArXiv:1910.02600 [Cs, Stat], November 24, 2020. http://arxiv.org/abs/1910.02600.

[^sensoy2018]: Sensoy, Murat, Lance Kaplan, and Melih Kandemir. "Evidential Deep Learning to Quantify Classification Uncertainty." Advances in Neural Information Processing Systems 31 (June 2018): 3179-89.

[^meinert2022]: Meinert, Nis, Jakob Gawlikowski, and Alexander Lavin. "The Unreasonable Effectiveness of Deep Evidential Regression." arXiv, May 20, 2022. http://arxiv.org/abs/2205.10060.

[^ye2024]: Ye, K., Chen, T., Wei, H. & Zhan, L. "Uncertainty Regularized Evidential Regression." AAAI 38, 16460-16468 (2024).

[^pandey2025]: Pandey, D. S., Choi, H. & Yu, Q. "Generalized Regularized Evidential Deep Learning Models." arXiv (2025).

[^yoon2025]: Yoon, T. & Kim, H. "Uncertainty Estimation by Flexible Evidential Deep Learning." arXiv (2025).
