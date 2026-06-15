# AGENTS.md

## Project overview

EvidentialFlux.jl is a Julia/Flux package implementing the Evidential Deep
Learning framework. It provides output layers that predict the parameters of
conjugate prior distributions, enabling uncertainty quantification (aleatoric
and epistemic) in a single forward pass. The package supports regression,
classification, and count data.

## Commands

```bash
# Run full CPU test suite
julia --project -e 'using Pkg; Pkg.test()'

# Run a single testset: edit test/runtests.jl, or include it in a loaded REPL
julia --project -e 'using EvidentialFlux, Test; include("test/runtests.jl")'

# Format code in place (Runic, via `julia -m Runic`)
make format
# Check formatting without writing (non-zero exit + diff if unformatted)
make formatcheck

# REPL with package loaded
julia --project -e 'using EvidentialFlux'
```

There is no separate lint step — formatting is the check. Tests run via the
`[targets] test` config in `Project.toml` (Test + CUDA). `test/gpu.jl` is
included automatically only when `CUDA.functional()` is true, so the GPU suite
is skipped on machines without a working GPU.

## Releases

Versioning is driven by Conventional Commits via `standard-version`. Use the
Makefile targets — they bump `version` in `Project.toml`, commit, then tag:

```bash
make releasepatch   # or releaseminor / releasemajor
```

`bump.sh <major|minor|patch>` is the underlying script. `.versionrc` controls
which commit types appear in `CHANGELOG.md`.

## Architecture

### Source layout

```
src/
  EvidentialFlux.jl   # Module definition and exports
  dense.jl            # Layer structs, constructors, forward passes
  losses.jl           # Loss functions and NLL computations
  utils.jl            # split_params, predict, uncertainty, evidence
test/
  runtests.jl         # CPU test suite
  gpu.jl              # GPU test suite (conditionally included)
examples/             # Runnable training scripts
```

### Abstract type hierarchy

All evidential output layers subtype `AbstractEvidentialLayer`. This enables
generic dispatch for `predict` and `split_params`.

### Layer patterns

There are two internal patterns for layers:

1. **W/b/sigma layers** (`NIG`, `PG`, `BNB`): Store a weight matrix `W`,
   bias `b`, and activation `σ`. The forward pass does
   `W * x .+ b`, splits via `_split_equal(o, n)`, applies activations per
   chunk, and `vcat`s. These support a custom activation via the `σ`
   constructor argument.

2. **Chain-wrapped layers** (`MVE`, `FDIR`): Store a `Chain(Parallel(vcat, ...))`
   with named branches (e.g. `μw`, `σw`). This supports selective
   `Flux.freeze!/thaw!` on individual branches. The forward pass delegates
   to the inner chain.

`DIR` is a special case: single W/b with hardcoded `softplus .+ 1`.

### Key internal helpers

- `_split_equal(y, n)` — splits the first dimension of a matrix into `n`
  equal chunks. Used by layer forward passes and `split_params`.
- `_reshape_call(a, x)` — handles 3D+ array inputs by reshaping to 2D,
  applying the layer, and reshaping back.

### Adding a new distribution

1. Define struct `<: AbstractEvidentialLayer` in `dense.jl`
2. Define `split_params(::Type{<:YourLayer}, y)` in `utils.jl`
3. Define loss function(s) in `losses.jl`
4. Add `epistemic(::Type{<:YourLayer}, ...)` and/or
   `aleatoric(::Type{<:YourLayer}, ...)` in `utils.jl`
5. Add `predictive_mean(::Type{<:YourLayer}, params)` in `utils.jl`
6. Add a `predictive(::Type{<:YourLayer}, m, x)` method in `utils.jl`
7. `predict` works automatically via generic dispatch

### Evidential framework pattern

Every distribution follows the same Bayesian structure:

| Component | Role | Example (NIG) |
|-----------|------|---------------|
| Likelihood | Data model | Normal(y \| mu, sigma^2) |
| Prior | Conjugate prior over likelihood params | Normal-Inverse-Gamma(gamma, nu, alpha, beta) |
| Marginal | Prior integrated out (type II ML) | Student-T |
| Loss | NLL of marginal + optional regularizer | `nigloss` = `nllstudent` + evidence penalty |

The same pattern applies to all layers:

- **NIG**: Normal likelihood, NIG prior, Student-T marginal
- **PG**: Poisson likelihood, Gamma prior, Negative Binomial marginal
- **EG**: Exponential likelihood, Gamma prior, Lomax marginal
- **BB**: Binomial likelihood, Beta prior, Beta-Binomial marginal
- **BNB**: Negative Binomial likelihood, Beta prior, Beta-NB marginal
- **DIR**: Categorical likelihood, Dirichlet prior, Dirichlet-Multinomial marginal
- **FDIR**: Categorical likelihood, Flexible Dirichlet prior, mixture of Dir-Multinomial
- **MVE**: Normal likelihood, point estimate (no prior)

#### Ordinal targets

For *ordered* categories (e.g. Very Low < … < Very High), reuse the `DIR` or
`FDIR` layer — the layer is unchanged; order-awareness lives in the loss.
`ofdirloss(y, α, p, τ; weights)` is the ordinal loss for `FDIR`: it is the
expected **Ranked Probability Score** (squared earth-mover distance) under the
Flexible Dirichlet, scoring the cumulative CDF instead of the one-hot vector so
rank-distant errors cost more. It reduces to the Dirichlet cumulative
Bayes-risk MSE when τ=1 and `pₖ = αₖ/Σα` (ordinal analogue of `fdirloss`'s
Theorem 4.3 reduction). The `weights` kwarg (length `K`, per-class) scales each
sample by its true-class weight — use inverse class frequency to counter
imbalance. Because `FDIR` is a mixture of Dirichlets it represents **bimodal**
ordinal conditionals (mass at both extremes), which structurally unimodal models
(Beta-Binomial, CORAL) cannot.

### Uncertainty API

Two patterns coexist:

1. **Legacy arity-dispatched** (NIG/DIR only): `uncertainty(nu, alpha, beta)`,
   `epistemic(nu)`, `aleatoric(nu, alpha, beta)`, `uncertainty(alpha)`.
   Kept for backward compatibility.

2. **Type-dispatched** (all layers): `epistemic(LayerType, params...)`,
   `aleatoric(LayerType, params...)`. The canonical API for new code.
   NIG/DIR/MVE type-dispatched methods delegate to the legacy implementations.

### predict vs predictive

Two prediction functions serve different use cases:

- `predict(model, x)` — returns raw distributional parameters (NamedTuple
  or array). Used in **training loops** where parameters feed directly into
  loss functions.
- `predictive(model, x)` — returns `(ŷ, epistemic, aleatoric, params)`.
  Used at **inference time** for uncertainty-aware predictions. `ŷ` is the
  posterior predictive mean in data space. Fields are `nothing` when a layer
  doesn't support that uncertainty type (e.g. MVE has no epistemic).

Both dispatch on `last_type(model)` to determine the layer type.
`predictive_mean(LayerType, params)` is the internal helper that computes
the data-space point prediction from raw parameters.

### Loss return shapes

- Regression losses (`nigloss*`, `mveloss`, `pgloss`, `bnbloss`): return
  `(O, B)` — one value per output per batch element
- Classification losses (`dirloss*`, `dirmultloss`, `fdirloss`, `ofdirloss`):
  return `(1, B)` — one value per batch element (summed over classes)

### GPU compatibility

All layers and losses work on GPU via standard Flux `gpu()` transfer.
No CUDA-specific code exists; compatibility comes from using `AbstractMatrix`,
`AbstractVecOrMat`, and broadcast operations throughout. GPU tests in
`test/gpu.jl` run automatically when `CUDA.functional()` is true.

## Conventions

- Preserve mathematical Unicode variable names (gamma, nu, alpha, beta, mu, sigma, etc.)
- Loss functions take raw array parameters, not layer objects (decoupled from layers)
- `split_params` returns NamedTuples; convenience `split*` functions return plain tuples
- `predict` returns NamedTuples for all layers except DIR (raw array for backward compat)
- Tests verify shapes, finiteness, value constraints, gradient flow, and (where applicable) known analytical results
- GPU compatibility comes from `AbstractMatrix`/`AbstractVecOrMat` and broadcasts — no CUDA-specific code; keep it that way
- Several exports are deprecated aliases (`nigloss2`→`nigloss_scaled`, `nigloss3`→`nigloss_ureg`, `dirloss2`→`dirloss_cor`); don't add new uses
