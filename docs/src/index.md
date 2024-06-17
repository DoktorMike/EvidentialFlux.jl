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

### Theoretical justifications

Although for the problems illustrated by Amini et. al., this approach seems to
work well it has been shown in [^nis2022] that there are theoretical
shortcomings regarding the expression of the aleatoric and epistemic
uncertainty. They propose a correction of the loss, and the uncertainty
calculations. In this package I have implemented both.

## Deep Evidential Classification

We follow [^sensoy2018] in our implementation of Deep Evidential
Classification. The neural network layer is implemented to output the
``\alpha_k`` representing the parameters of a Dirichlet distribution. These
parameters has the additional interpretation ``\alpha_k = e_k + 1`` where
``e_k`` is the evidence for class ``k``. Further, it holds that ``e_k > 0``
which is the reason for us modeling them with a softplus activation function. 

Ok, so that's all well and good, but what's the point? Well, the point is that
since we are now constructing a network layer that outputs evidence for each
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

## Functions

```@docs
DIR
NIG
MVE
predict
uncertainty
aleatoric
epistemic
evidence
nigloss
nigloss2
dirloss
mveloss
```

## Index

```@index
```

## References

[^amini2020]: Amini, Alexander, Wilko Schwarting, Ava Soleimany, and Daniela Rus. “Deep Evidential Regression.” ArXiv:1910.02600 [Cs, Stat], November 24, 2020. http://arxiv.org/abs/1910.02600.

[^sensoy2018]: Sensoy, Murat, Lance Kaplan, and Melih Kandemir. “Evidential Deep Learning to Quantify Classification Uncertainty.” Advances in Neural Information Processing Systems 31 (June 2018): 3179–89.

[^nis2022]: Meinert, Nis, Jakob Gawlikowski, and Alexander Lavin. “The Unreasonable Effectiveness of Deep Evidential Regression.” arXiv, May 20, 2022. http://arxiv.org/abs/2205.10060.

