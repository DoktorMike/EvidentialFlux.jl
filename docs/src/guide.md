# Choosing the Right Layer

EvidentialFlux provides several evidential output layers, each designed for a
specific type of data. This guide helps you pick the right one for your problem.

## Decision flowchart

Ask yourself: **what does my target variable look like?**

- **Real numbers** (can be negative, zero, or positive) → [NIG](#Real-valued-targets-NIG) or [MVE](#Simple-variance-estimation-MVE)
- **Strictly positive numbers** (always > 0) → [EG](#Positive-continuous-targets-EG)
- **Counts** (0, 1, 2, ...) → [PG](#Count-targets-PG), [BNB](#Overdispersed-count-targets-BNB), or [ZIP](#Zero-inflated-count-targets-ZIP)
- **One of K classes** → [DIR](#Classification-targets-DIR) or [FDIR](#Flexible-classification-FDIR)
- **Counts per category** (multiple categories, totals vary) → [DIR + dirmultloss](#Count-vectors-across-categories)
- **Proportions / success rates** (k successes out of n trials) → [BB](#Proportions-and-success-rates-BB)
- **Binary yes/no outcomes** (probability estimation) → [BB with n=1](#Binary-outcomes-(Beta-Bernoulli)-BB-with-n1)

## Real-valued targets — NIG

**Use when** your target is a continuous value that can be any real number.

**Real-world examples:**
- Temperature forecasting (tomorrow's high in °C)
- Stock price changes (daily returns, positive or negative)
- Sensor calibration residuals
- Energy demand prediction (MW, centered around a mean)
- Patient blood pressure readings

**Layer:** `NIG(in => out)` — predicts 4 parameters (γ, ν, α, β) per target.

**Why NIG over MVE?** NIG gives you both aleatoric *and* epistemic uncertainty.
Epistemic uncertainty tells you when the model is extrapolating beyond its
training data — critical for safety-sensitive applications. MVE only gives
aleatoric uncertainty.

```julia
model = Chain(Dense(10 => 64, relu), Dense(64 => 64, relu), NIG(64 => 1))

# Training
loss, grads = Flux.withgradient(model) do m
    γ, ν, α, β = splitnig(m(x))
    mean(nigloss(y, γ, ν, α, β, 0.01))
end

# Inference
r = predictive(model, x_test)
r.ŷ          # predicted temperature
r.epistemic  # high when far from training data
r.aleatoric  # high when inherent measurement noise is large
```

**Loss options:**
- `nigloss` — standard evidential regression (Amini et al. 2020)
- `nigloss_scaled` — better calibrated, normalizes error by aleatoric (Meinert et al. 2022)
- `nigloss_ureg` — fixes gradient issues in high-uncertainty regions (Ye et al. 2024)

Start with `nigloss_scaled` — it's the most robust default.

## Simple variance estimation — MVE

**Use when** you only need aleatoric uncertainty and want a simpler model.

**Real-world examples:**
- Heteroskedastic noise modeling (variance changes across the input space)
- Quick baseline before trying NIG
- Situations where you trust the model is always interpolating (no OOD concern)

**Layer:** `MVE(in => out)` — predicts mean μ and variance σ per target.

```julia
model = Chain(Dense(10 => 64, relu), Dense(64 => 64, relu), MVE(64 => 1))

loss, grads = Flux.withgradient(model) do m
    μ, σ = splitmve(m(x))
    mean(mveloss(y, μ, σ))
end
```

**When to upgrade to NIG:** If you need to detect out-of-distribution inputs or
quantify model uncertainty (not just data noise), switch to NIG.

## Positive continuous targets — EG

**Use when** your target is strictly positive and often right-skewed.

**Real-world examples:**
- Customer lifetime value (dollars spent, always > 0)
- Time-to-event / survival (days until churn, hours until failure)
- Insurance claim amounts
- Waiting times (minutes between bus arrivals)
- Drug concentration in blood (mg/L)
- Distance measurements (meters to nearest object)

**Layer:** `EG(in => out)` — predicts Gamma parameters (α, β) per target.

```julia
model = Chain(Dense(10 => 64, relu), Dense(64 => 64, relu), EG(64 => 1))

loss, grads = Flux.withgradient(model) do m
    α, β = spliteg(m(x))
    mean(egloss(durations, α, β, 0.1))
end

r = predictive(model, x_test)
r.ŷ          # expected duration (β/(α-1))
r.epistemic  # uncertain about the rate — need more data like this
r.aleatoric  # inherent variability in durations
```

**Why not just use NIG?** NIG assumes the target can be any real number. If your
data is strictly positive, NIG can predict negative values. EG's Lomax
posterior predictive is naturally supported on (0, ∞), matching your data.

## Count targets — PG

**Use when** your target is a non-negative integer count without significant
overdispersion (variance ≈ mean).

**Real-world examples:**
- Number of emails received per hour
- Website visits per day (when traffic is relatively stable)
- Number of defects found in a code review
- Photon counts in a sensor
- Number of arrivals at a queue in a fixed time window

**Layer:** `PG(in => out)` — predicts Gamma parameters (α, β) over the Poisson rate.

```julia
model = Chain(Dense(10 => 64, relu), Dense(64 => 64, relu), PG(64 => 1))

loss, grads = Flux.withgradient(model) do m
    α, β = splitpg(m(x))
    mean(pgloss(counts, α, β, 0.1))
end

r = predictive(model, x_test)
r.ŷ          # expected count (α/β)
r.epistemic  # uncertain about the rate
r.aleatoric  # inherent Poisson randomness
```

## Overdispersed count targets — BNB

**Use when** your count data has variance significantly larger than the mean
(overdispersion), or when you need to model the success probability rather
than a rate.

**Real-world examples:**
- Number of insurance claims per customer (highly variable between customers)
- Hospital readmissions (some patients are much more likely to return)
- Species counts in ecological surveys (clumped spatial distribution)
- Number of purchases per customer per month (heavy-tailed)
- Gene expression read counts in RNA-seq (biological + technical variance)

**Layer:** `BNB(in => out)` — predicts dispersion r and Beta parameters (α, β).

```julia
model = Chain(Dense(10 => 64, relu), Dense(64 => 64, relu), BNB(64 => 1))

loss, grads = Flux.withgradient(model) do m
    r, α, β = splitbnb(m(x))
    mean(bnbloss(claims, r, α, β, 0.1))
end
```

**When to use BNB vs PG:** If your data looks "clumpy" — lots of zeros and
occasional large values — that's overdispersion, and BNB handles it better.
If counts are relatively uniform around the mean, PG is simpler and sufficient.

## Zero-inflated count targets — ZIP

**Use when** your count data has more zeros than a standard Poisson or Negative
Binomial can explain. Zero-inflation arises when there are two distinct
data-generating processes: a "structural zero" process (the event can't happen)
and a count process (the event can happen, and follows Poisson statistics).

**Real-world examples:**
- Number of insurance claims per customer (many customers never file)
- Number of cigarettes smoked per day (many people are non-smokers)
- Number of fish caught per trip (many trips yield nothing at all)
- Number of doctor visits per year (many healthy people never go)
- Number of defects per product (many products are defect-free)
- Number of workplace accidents per month (most months are zero)

**Layer:** `ZIP(in => out)` — predicts Beta parameters (α_π, β_π) for the
zero-inflation probability π and Gamma parameters (α_λ, β_λ) for the Poisson
rate λ.

```julia
model = Chain(Dense(10 => 64, relu), Dense(64 => 64, relu), ZIP(64 => 1))

loss, grads = Flux.withgradient(model) do m
    α_π, β_π, α_λ, β_λ = splitzip(m(x))
    mean(ziploss(counts, α_π, β_π, α_λ, β_λ, 0.1))
end

r = predictive(model, x_test)
r.ŷ          # expected count: E[1-π]·E[λ] = β_π/(α_π+β_π) · α_λ/β_λ
r.epistemic  # Var[(1-π)λ]: uncertain about zero-inflation AND/OR rate
r.aleatoric  # E[Var[Y|π,λ]]: inherent ZIP randomness
```

**When to use ZIP vs PG:** If your data has a suspicious spike at zero — more
zeros than a Poisson model predicts — that's zero-inflation. A good diagnostic:
if the observed fraction of zeros is much larger than `exp(-mean(y))` (the
Poisson prediction), use ZIP. If zeros are roughly in line with the overall
count rate, PG is sufficient.

**When to use ZIP vs BNB:** Both handle excess zeros, but for different reasons.
BNB explains excess zeros via overdispersion (high variance inflates the zero
probability). ZIP explicitly models a separate zero-generating process. If your
zeros come from a distinct subpopulation that simply *can't* produce events
(non-smokers, non-customers), ZIP is the better structural match. If your zeros
are just part of a highly variable count distribution, BNB may suffice.

## Classification targets — DIR

**Use when** each observation belongs to exactly one of K classes.

**Real-world examples:**
- Image classification (cat vs dog vs bird)
- Sentiment analysis (positive / negative / neutral)
- Medical diagnosis (healthy / disease A / disease B)
- Spam detection (spam / not spam)
- Fault type classification in manufacturing

**Layer:** `DIR(in => out)` — predicts Dirichlet concentration parameters per class.

```julia
model = Chain(Dense(10 => 64, relu), Dense(64 => 64, relu), DIR(64 => 3))

loss, grads = Flux.withgradient(model) do m
    α = m(x)
    sum(dirloss(y_onehot, α, epoch))
end

r = predictive(model, x_test)
r.ŷ          # class probabilities
r.epistemic  # high = "I don't know which class" (OOD detection)
```

**Loss options:**
- `dirloss` — standard Dirichlet EDL (Sensoy et al. 2018)
- `dirloss_cor` — fixes gradient vanishing for low-evidence samples (Pandey et al. 2025)

For OOD detection, the epistemic uncertainty `K/Σα` is particularly useful — it
approaches 1 when the model has seen no evidence for any class.

## Flexible classification — FDIR

**Use when** you need more expressive uncertainty modeling than standard DIR, or
when DIR's uncertainty estimates are not well-calibrated on your data.

**Real-world examples:**
- Safety-critical classification where uncertainty calibration matters (autonomous driving, medical AI)
- Noisy label settings where standard DIR produces overconfident wrong predictions
- Any classification problem where DIR's OOD detection (AUROC/AUPR) is insufficient

**Layer:** `FDIR(in => out)` — predicts Flexible Dirichlet parameters (α, p, τ).

```julia
model = Chain(Dense(10 => 64, relu), Dense(64 => 64, relu), FDIR(64 => 3))

loss, grads = Flux.withgradient(model) do m
    α, p, τ = splitfdir(m(x))
    sum(fdirloss(y_onehot, α, p, τ))
end
```

**Why FDIR over DIR?** FDIR uses a mixture of Dirichlets, enabling multimodal
uncertainty representations. It also replaces the KL regularizer (which needs
manual λ tuning) with a Brier score regularizer that's hyperparameter-free.
Standard DIR is a special case of FDIR (Theorem 4.3 in Yoon & Kim 2025).

**Trade-off:** FDIR has ~1.8% more parameters (three output heads vs one) and
doesn't benefit from `dirloss_cor`. Use DIR first; switch to FDIR if you need
better uncertainty calibration.

## Count vectors across categories

**Use when** you observe counts per category (not just which category), and the
total count varies per observation.

**Real-world examples:**
- Word counts in documents (bag-of-words text classification)
- Survey response tallies (how many people chose each option)
- Allele counts in population genetics
- Shopping basket composition (counts per product category)
- Event type counts in a time window (types of support tickets per week)

**Layer:** Reuse `DIR(in => out)` with `dirmultloss` instead of `dirloss`.

```julia
model = Chain(Dense(10 => 64, relu), Dense(64 => 64, relu), DIR(64 => K))

loss, grads = Flux.withgradient(model) do m
    α = m(x)
    sum(dirmultloss(word_counts, α))  # word_counts is (K, B), not one-hot
end
```

No new layer needed — the Dirichlet prior is the same, only the loss changes
from Bayes Risk MSE (categorical) to Dirichlet-Multinomial NLL (count vectors).

## Proportions and success rates — BB

**Use when** you observe `k` successes out of `n` trials and want to estimate the
underlying success probability with uncertainty.

**Real-world examples:**
- A/B test conversion rates (k purchases out of n visitors)
- Clinical trial response rates (k responders out of n patients)
- Manufacturing defect rates (k defects out of n items inspected)
- Free throw shooting percentage (k makes out of n attempts)
- Click-through rates (k clicks out of n impressions)

**Layer:** `BB(in => out)` — predicts Beta parameters (α, β) over the success probability.

```julia
model = Chain(Dense(10 => 64, relu), Dense(64 => 64, relu), BB(64 => 1))

loss, grads = Flux.withgradient(model) do m
    α, β = splitbb(m(x))
    mean(bbloss(successes, trials, α, β, 0.1))
end

r = predictive(model, x_test)
r.ŷ          # estimated probability (with n=1 default)
r.epistemic  # Var[p]: shrinks as α+β grows (more evidence)
r.aleatoric  # E[p(1-p)]: inherent Bernoulli variance
```

**Note:** `predictive_mean` defaults to `n=1` (probability scale). For expected
counts, pass `n` explicitly: `predictive_mean(BB, params, n)`.

## Binary outcomes (Beta-Bernoulli) — BB with n=1

**Use when** each observation is a binary yes/no outcome (not one-hot
classification, but a probability you want to estimate with uncertainty).

**Real-world examples:**
- Will this patient respond to treatment? (per-patient probability)
- Will this loan default? (per-loan probability)
- Will this user churn? (per-user probability)
- Will this component fail within warranty? (per-component probability)
- Is this transaction fraudulent? (per-transaction probability)

The Beta-Bernoulli model is a special case of the Binomial-Beta with `n=1`.
There is no separate layer — use `BB` and pass `n = 1` (a scalar) to the loss:

```julia
model = Chain(Dense(10 => 64, relu), Dense(64 => 64, relu), BB(64 => 1))

# y is 0 or 1 per observation
loss, grads = Flux.withgradient(model) do m
    α, β = splitbb(m(x))
    mean(bbloss(y, 1, α, β, 0.1))   # n=1 for Bernoulli
end

r = predictive(model, x_test)
r.ŷ          # predicted probability of success
r.epistemic  # high when the model lacks evidence (few similar training samples)
r.aleatoric  # high when the true probability is near 0.5 (inherently uncertain)
```

**Why BB instead of DIR with K=2?** Both model binary outcomes, but they differ
in framing. DIR treats it as "which class?" and gives epistemic uncertainty
only. BB treats it as "what's the probability?" and gives both epistemic
(uncertainty about p) and aleatoric (inherent coin-flip variance). Use BB when
you care about the calibrated probability, DIR when you care about the
classification decision.

## Summary table

| Problem | Layer | Loss | Prediction | Example |
|---------|-------|------|------------|---------|
| Real-valued regression | `NIG` | `nigloss_scaled` | γ | Temperature forecasting |
| Simple regression | `MVE` | `mveloss` | μ | Quick baseline |
| Positive regression | `EG` | `egloss` | β/(α-1) | Customer lifetime value |
| Count regression | `PG` | `pgloss` | α/β | Emails per hour |
| Overdispersed counts | `BNB` | `bnbloss` | r·α/β | Insurance claims |
| Zero-inflated counts | `ZIP` | `ziploss` | β_π/(α_π+β_π)·α_λ/β_λ | Doctor visits per year |
| Classification | `DIR` | `dirloss` | α/Σα | Image classification |
| Calibrated classification | `FDIR` | `fdirloss` | (α+τp)/(Σα+τ) | Safety-critical AI |
| Count vectors | `DIR` | `dirmultloss` | α/Σα | Bag-of-words NLP |
| Proportions | `BB` | `bbloss` | α/(α+β) | A/B test conversion |
| Binary outcomes | `BB` | `bbloss(y, 1, ...)` | α/(α+β) | Loan default prediction |

## Special cases and modeling tips

Several common modeling scenarios don't need a dedicated layer — they fall out
as special cases of the layers above. This section helps you recognize them.

### Binary classification — DIR with K=2

If you only care about the class *decision* (not the calibrated probability),
use `DIR(in => 2)` instead of `BB`. This is the standard evidential approach
to binary classification:

```julia
model = Chain(Dense(10 => 64, relu), DIR(64 => 2))

loss, grads = Flux.withgradient(model) do m
    sum(dirloss(y_onehot, m(x), epoch))  # y_onehot is (2, B)
end

r = predictive(model, x_test)
predicted_class = r.ŷ[1, :] .> 0.5   # class decision
r.epistemic                           # OOD detection signal
```

**DIR with K=2 vs BB:** DIR gives you epistemic uncertainty ("I don't know which
class") but no aleatoric. BB gives you both ("I think the probability is around
0.7, but I'm not sure" vs "the probability really is 0.5 — it's a coin flip").
Use DIR for OOD detection, BB for calibrated probability estimation.

### Categorical via type II maximum likelihood — DIR + dirmultloss with n=1

If you pass one-hot encoded targets (counts summing to 1) to `dirmultloss`
instead of `dirloss`, you get the Categorical-Dirichlet marginal likelihood —
the proper type II ML loss for single-observation classification:

```julia
model = Chain(Dense(10 => 64, relu), DIR(64 => 5))

loss, grads = Flux.withgradient(model) do m
    sum(dirmultloss(y_onehot, m(x)))  # y_onehot is one-hot, sums to 1
end
```

This is an alternative to `dirloss` (which uses Bayes Risk MSE + KL
regularizer). The marginal likelihood naturally balances fit and complexity
without a separate regularization hyperparameter. Try it if you find `dirloss`
sensitive to the KL annealing schedule.

### FDIR as a drop-in upgrade for DIR

Standard DIR is a special case of FDIR (when τ=1 and p=α/Σα). This means you
can always swap `DIR` → `FDIR` without changing your problem setup — FDIR
is strictly more expressive:

```julia
# Before
model = Chain(Dense(10 => 64, relu), DIR(64 => 5))
# After — same architecture, better uncertainty calibration
model = Chain(Dense(10 => 64, relu), FDIR(64 => 5))
```

The trade-offs: FDIR adds ~1.8% more parameters (three output heads vs one),
uses `fdirloss` instead of `dirloss`/`dirloss_cor`, and needs no KL annealing
hyperparameter. Switch to FDIR when DIR's OOD detection metrics plateau.

### Geometric / waiting-time counts — BNB with r=1

The Geometric distribution (number of trials until first success) is a special
case of the Negative Binomial with r=1. If your BNB model learns r≈1, it's
effectively modeling geometric data:

**Real-world examples:**
- Number of sales calls before closing a deal
- Number of job applications before getting an offer
- Number of coin flips before heads
- Number of server requests before a timeout

You don't need to constrain r=1 manually — BNB can learn it. But if you know
your data is geometric, you could use `PG` instead (since the Geometric is also
a special case of Poisson processes in the limit).

## Choosing a loss function

### NIG regression: nigloss vs nigloss\_scaled vs nigloss\_ureg

All three share the same Student-T NLL base; they differ in their regularizer:

| Loss | Regularizer | Start with λ | When to use |
|------|------------|-------------|-------------|
| `nigloss` | `\|y-γ\| · evidence` | 0.01-0.1 | Baseline, simple problems |
| `nigloss_scaled` | `(\|y-γ\|/σ_St)^p · evidence` | 0.01-0.1 | **Recommended default.** Prevents the model from inflating variance to cheat the regularizer |
| `nigloss_ureg` | `\|y-γ\| · evidence + uncertainty_loss` | λ=0.1, λ₁=0.1 | When you observe loss plateaus with high uncertainty predictions (gradient vanishing) |

**Start with `nigloss_scaled`** at `λ=0.01`. If the model produces reasonable
predictions but uncertainty is poorly calibrated, try increasing λ. If
uncertainty is always high and the model seems stuck, try `nigloss_ureg`.

### DIR classification: dirloss vs dirloss\_cor vs fdirloss

| Loss | Regularizer | When to use |
|------|------------|-------------|
| `dirloss(y, α, t)` | Annealed KL divergence | Standard choice. The epoch counter `t` anneals the KL weight as `min(1, t/10)` — no tuning needed |
| `dirloss_cor(y, α, t)` | Annealed KL + correct evidence term | When training is slow or accuracy plateaus early — the correction helps samples stuck in low-evidence regions |
| `fdirloss(y, α, p, τ)` | Brier score on allocation `p` | When switching to the FDIR layer. No λ or epoch counter needed — the Brier regularizer is hyperparameter-free |
| `dirmultloss(y, α)` | None (type II ML) | When targets are count vectors instead of one-hot. Also usable with one-hot targets as a hyperparameter-free alternative to `dirloss` |

**Start with `dirloss`** — it works well out of the box. The KL annealing
(`min(1, t/10)`) ramps up over the first 10 epochs automatically.

### Count/positive regression: pgloss, egloss, bnbloss, bbloss

These all follow the same pattern: NLL + `λ · |error| · evidence`.

| Loss | Evidence term | Start with λ |
|------|-------------|-------------|
| `pgloss` | α (Gamma shape) | 0.01-0.1 |
| `egloss` | α (Gamma shape) | 0.01-0.1 |
| `bnbloss` | α+β (Beta concentration) | 0.01-0.1 |
| `bbloss` | α+β (Beta concentration) | 0.01-0.1 |
| `ziploss` | α_π+β_π+α_λ (Beta + Gamma evidence) | 0.01-0.1 |

Set `λ=0` to train with pure marginal NLL (no regularizer). This is a valid
starting point — the marginal likelihood already balances fit and complexity.
Add regularization (`λ=0.01`) if the model produces overconfident predictions.

## Hyperparameter tips

### Regularization weight λ

The regularization weight λ controls how strongly the model is penalized for
being confident about wrong predictions.

- **Too low** (λ → 0): model fits the data well but may be overconfident,
  especially on out-of-distribution inputs
- **Too high** (λ → ∞): model becomes underconfident everywhere, predicting
  high uncertainty even where data is plentiful
- **Sweet spot**: typically 0.001-0.1. Start with **0.01** and adjust based on
  calibration plots

**Practical workflow:**
1. Start with `λ=0` (pure NLL) to verify the model can fit your data
2. Add `λ=0.01` and check if uncertainty grows in data-sparse regions
3. If uncertainty is too uniform, increase to `λ=0.1`
4. If predictions degrade, reduce λ

### Learning rate

Evidential layers are sensitive to learning rate. If uncertainty collapses to
near-zero early in training, try reducing the learning rate. A good starting
point is `1e-3` with `AdamW`.

### KL annealing (DIR / dirloss)

The `dirloss` epoch counter `t` anneals the KL weight as `min(1.0, t/10)`.
This means the KL regularizer is off at epoch 1, reaches half strength at
epoch 5, and full strength at epoch 10+. This prevents the regularizer from
dominating early training before the model has learned useful representations.

If your model converges quickly (< 50 epochs), the annealing happens fast
enough. For very long training runs (thousands of epochs), the annealing
is effectively instant and has no impact.
