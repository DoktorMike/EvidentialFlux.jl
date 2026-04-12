# Choosing the Right Layer

EvidentialFlux provides several evidential output layers, each designed for a
specific type of data. This guide helps you pick the right one for your problem.

## Decision flowchart

Ask yourself: **what does my target variable look like?**

- **Real numbers** (can be negative, zero, or positive) → [NIG](#Real-valued-targets-NIG) or [MVE](#Simple-variance-estimation-MVE)
- **Strictly positive numbers** (always > 0) → [EG](#Positive-continuous-targets-EG)
- **Counts** (0, 1, 2, ...) → [PG](#Count-targets-PG) or [BNB](#Overdispersed-count-targets-BNB)
- **One of K classes** → [DIR](#Classification-targets-DIR) or [FDIR](#Flexible-classification-FDIR)
- **Counts per category** (multiple categories, totals vary) → [DIR + dirmultloss](#Count-vectors-across-categories)
- **Proportions / success rates** (k successes out of n trials) → [BB](#Proportions-and-success-rates-BB)

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

## Summary table

| Problem | Layer | Loss | Prediction | Example |
|---------|-------|------|------------|---------|
| Real-valued regression | `NIG` | `nigloss_scaled` | γ | Temperature forecasting |
| Simple regression | `MVE` | `mveloss` | μ | Quick baseline |
| Positive regression | `EG` | `egloss` | β/(α-1) | Customer lifetime value |
| Count regression | `PG` | `pgloss` | α/β | Emails per hour |
| Overdispersed counts | `BNB` | `bnbloss` | r·α/β | Insurance claims |
| Classification | `DIR` | `dirloss` | α/Σα | Image classification |
| Calibrated classification | `FDIR` | `fdirloss` | (α+τp)/(Σα+τ) | Safety-critical AI |
| Count vectors | `DIR` | `dirmultloss` | α/Σα | Bag-of-words NLP |
| Proportions | `BB` | `bbloss` | α/(α+β) | A/B test conversion |
