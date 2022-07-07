# EvidentialFlux.jl

Evidential Deep Learning is a way to generate predictions and the uncertainty
associated with them in one single forward pass. This is in stark contrast to
traditional Bayesian neural networks which are typically based on Variational
Inference, Markov Chain Monte Carlo, Monte Carlo Dropout or Ensembles. 

```math
\underset{Prediction}{\underbrace{\mathbb{E}[\mu]=\gamma}}~~~~
\underset{Aleatoric}{\underbrace{\mathbb{E}[\sigma^2]=\frac{\beta}{\alpha-1}}}~~~~
\underset{Epistemic}{\underbrace{\text{Var}[\mu]=\frac{\beta}{\nu(\alpha-1)}}}
```


## Functions

```@docs
uncertainty
predict
nigloss
NIG
```

## Index

```@index
```
