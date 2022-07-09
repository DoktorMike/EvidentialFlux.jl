# EvidentialFlux.jl

Evidential Deep Learning is a way to generate predictions and the uncertainty
associated with them in one single forward pass. This is in stark contrast to
traditional Bayesian neural networks which are typically based on Variational
Inference, Markov Chain Monte Carlo, Monte Carlo Dropout or Ensembles. 

## Deep Evidential Regression

Deep Evidential Regression[^amini2020] is an attempt to apply the principles of
Evidential Deep Learning to regression type problems.

It works by putting a prior distribution over the likelihood parameters
``\mathbf{\theta} = \{\mu, \sigma^2\}`` governing a likelihood model where we observe a
dataset ``\mathcal{D}=\{x_i, y_i\}_{i=1}^N`` where ``y_i`` is assumed to be
drawn i.i.d. from a Gaussian distribution.

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
``\mathcal{N-}\Gamma^{-1}(γ, υ, α, β)`` fits the bill. I'm being a bit handwavy
here but this allows us to express the prediction and the associated
uncertainty as below.

```math
\underset{Prediction}{\underbrace{\mathbb{E}[\mu]=\gamma}}~~~~
\underset{Aleatoric}{\underbrace{\mathbb{E}[\sigma^2]=\frac{\beta}{\alpha-1}}}~~~~
\underset{Epistemic}{\underbrace{\text{Var}[\mu]=\frac{\beta}{\nu(\alpha-1)}}}
```

The `NIG` layer in `EvidentialFlux.jl` outputs 4 tensors for each target
variable, namely ``\gamma,\nu,\alpha,\beta``. This means that in one forward
pass we can estimate the prediction, the heteroskedastic aleatoric uncertainty
as well as the epistemic uncertainty. Boom!

## Deep Evidential Classification

Not yet implemented.

## Functions

```@docs
NIG
predict
uncertainty
nigloss
```

## Index

```@index
```

## References

[^amini2020]: Amini, Alexander, Wilko Schwarting, Ava Soleimany, and Daniela Rus. “Deep Evidential Regression.” ArXiv:1910.02600 [Cs, Stat], November 24, 2020. http://arxiv.org/abs/1910.02600.

