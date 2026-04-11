"""
    AbstractEvidentialLayer

Abstract supertype for all evidential output layers (NIG, DIR, MVE, etc.).
Subtypes participate in generic `predict` and `split_params` dispatch.
"""
abstract type AbstractEvidentialLayer end

"""
    _reshape_call(a, x::AbstractArray)

Internal helper to handle higher-dimensional (3D+) array inputs by reshaping
to a matrix, applying the layer, and reshaping back.
"""
_reshape_call(a, x::AbstractArray) = reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

"""
    _split_equal(y::AbstractVecOrMat, n::Int)

Split the first dimension of `y` into `n` equal-sized chunks.
Returns an `n`-tuple of arrays each with shape `(nout, batch...)` where `nout = size(y,1) ÷ n`.
"""
function _split_equal(y::AbstractVecOrMat, n::Int)
    nout = size(y, 1) ÷ n
    return ntuple(i -> y[(1 + (i - 1) * nout):(i * nout), :], n)
end

"""
    NIG(in => out, σ=NNlib.softplus; bias=true, init=Flux.glorot_uniform)
    NIG(W::AbstractMatrix, [bias, σ])

Create a fully connected layer which implements the NormalInverseGamma Evidential distribution
whose forward pass is simply given by:

    y = W * x .+ bias

The input `x` should be a vector of length `in`, or batch of vectors represented
as an `in × N` matrix, or any array with `size(x,1) == in`.
The out `y` will be a vector  of length `out*4`, or a batch with
`size(y) == (out*4, size(x)[2:end]...)`
The output will have applied the function `σ(y)` to each row/element of `y` except the first `out` ones.
Keyword `bias=false` will switch off trainable bias for the layer.
The initialisation of the weight matrix is `W = init(out*4, in)`, calling the function
given to keyword `init`, with default [`glorot_uniform`].
The weight matrix and/or the bias vector (of length `out`) may also be provided explicitly.
Remember that in this case the number of rows in the weight matrix `W` MUST be a multiple of 4.
The same holds true for the `bias` vector.

# Arguments:
- `(in, out)`: number of input and output neurons
- `σ`: The function to use to secure positive only outputs which defaults to the softplus function.
- `init`: The function to use to initialise the weight matrix.
- `bias`: Whether to include a trainable bias vector.
"""
struct NIG{F, M <: AbstractMatrix, B} <: AbstractEvidentialLayer
    W::M
    b::B
    σ::F
    function NIG(W::M, b = true, σ::F = NNlib.softplus) where {M <: AbstractMatrix, F}
        b = Flux.create_bias(W, b, size(W, 1))
        return new{F, M, typeof(b)}(W, b, σ)
    end
end

function NIG(
        (in, out)::Pair{<:Integer, <:Integer}, σ = NNlib.softplus;
        init = Flux.glorot_uniform, bias = true
    )
    return NIG(init(out * 4, in), bias, σ)
end

Flux.@layer NIG

function (a::NIG)(x::AbstractVecOrMat)
    o = a.W * x .+ a.b
    γ, ν_raw, α_raw, β_raw = _split_equal(o, 4)
    return vcat(γ, a.σ.(ν_raw), a.σ.(α_raw) .+ 1, a.σ.(β_raw))
end

(a::NIG)(x::AbstractArray) = _reshape_call(a, x)

"""
    PG(in => out, σ=NNlib.softplus; bias=true, init=Flux.glorot_uniform)
    PG(W::AbstractMatrix, [bias, σ])

Create a fully connected layer which implements a Poisson-Gamma evidential model
for count regression. Places a Gamma(α, β) prior over the Poisson rate parameter λ,
yielding a Negative Binomial marginal likelihood.

The output has shape `(out*2, batch...)` containing `[α, β]` stacked vertically,
where both α and β are passed through `σ` to ensure positivity.

Use with `pgloss` for training and `splitpg` / `split_params(PG, y)` to
decompose the output. The expected count is `E[λ] = α/β`.

# Arguments:
- `(in, out)`: number of input features and output count targets
- `σ`: activation ensuring positivity (default: softplus)
- `init`: weight initialisation function (default: `glorot_uniform`)
- `bias`: whether to include a trainable bias vector
"""
struct PG{F, M <: AbstractMatrix, B} <: AbstractEvidentialLayer
    W::M
    b::B
    σ::F
    function PG(W::M, b = true, σ::F = NNlib.softplus) where {M <: AbstractMatrix, F}
        b = Flux.create_bias(W, b, size(W, 1))
        return new{F, M, typeof(b)}(W, b, σ)
    end
end

function PG(
        (in, out)::Pair{<:Integer, <:Integer}, σ = NNlib.softplus;
        init = Flux.glorot_uniform, bias = true
    )
    return PG(init(out * 2, in), bias, σ)
end

Flux.@layer PG

function (a::PG)(x::AbstractVecOrMat)
    o = a.W * x .+ a.b
    α_raw, β_raw = _split_equal(o, 2)
    return vcat(a.σ.(α_raw), a.σ.(β_raw))
end

(a::PG)(x::AbstractArray) = _reshape_call(a, x)

"""
    BNB(in => out, σ=NNlib.softplus; bias=true, init=Flux.glorot_uniform)
    BNB(W::AbstractMatrix, [bias, σ])

Create a fully connected layer which implements a Beta-Negative Binomial
evidential model for overdispersed count regression. Places a Beta(α, β) prior
over the Negative Binomial success probability `p`, with a learned dispersion
parameter `r`.

The output has shape `(out*3, batch...)` containing `[r, α, β]` stacked
vertically, where all three are passed through `σ` to ensure positivity.

Use with `bnbloss` for training and `splitbnb` / `split_params(BNB, y)` to
decompose the output. The predicted count at the Beta mean is `r·α/β`.

# Arguments:
- `(in, out)`: number of input features and output count targets
- `σ`: activation ensuring positivity (default: softplus)
- `init`: weight initialisation function (default: `glorot_uniform`)
- `bias`: whether to include a trainable bias vector
"""
struct BNB{F, M <: AbstractMatrix, B} <: AbstractEvidentialLayer
    W::M
    b::B
    σ::F
    function BNB(W::M, b = true, σ::F = NNlib.softplus) where {M <: AbstractMatrix, F}
        b = Flux.create_bias(W, b, size(W, 1))
        return new{F, M, typeof(b)}(W, b, σ)
    end
end

function BNB(
        (in, out)::Pair{<:Integer, <:Integer}, σ = NNlib.softplus;
        init = Flux.glorot_uniform, bias = true
    )
    return BNB(init(out * 3, in), bias, σ)
end

Flux.@layer BNB

function (a::BNB)(x::AbstractVecOrMat)
    o = a.W * x .+ a.b
    r_raw, α_raw, β_raw = _split_equal(o, 3)
    return vcat(a.σ.(r_raw), a.σ.(α_raw), a.σ.(β_raw))
end

(a::BNB)(x::AbstractArray) = _reshape_call(a, x)

"""
    DIR(in => out; bias=true, init=Flux.glorot_uniform)
    DIR(W::AbstractMatrix, [bias])

A Linear layer with a softplus activation function in the end to implement the
Dirichlet evidential distribution. In this layer the number of output nodes
should correspond to the number of classes you wish to model. This layer should
be used to model a Multinomial likelihood with a Dirichlet prior. Thus the
posterior is also a Dirichlet distribution. Moreover the type II maximum
likelihood, i.e., the marginal likelihood is a Dirichlet-Multinomial
distribution. Create a fully connected layer which implements the Dirichlet
Evidential distribution whose forward pass is simply given by:

    y = softplus.(W * x .+ bias)

The input `x` should be a vector of length `in`, or batch of vectors represented
as an `in × N` matrix, or any array with `size(x,1) == in`.
The out `y` will be a vector  of length `out`, or a batch with
`size(y) == (out, size(x)[2:end]...)`
The output will have applied the function `softplus(y)` to each row/element of `y`.
Keyword `bias=false` will switch off trainable bias for the layer.
The initialisation of the weight matrix is `W = init(out, in)`, calling the function
given to keyword `init`, with default [`glorot_uniform`].
The weight matrix and/or the bias vector (of length `out`) may also be provided explicitly.

# Arguments:
- `(in, out)`: number of input and output neurons
- `init`: The function to use to initialise the weight matrix.
- `bias`: Whether to include a trainable bias vector.
"""
struct DIR{M <: AbstractMatrix, B} <: AbstractEvidentialLayer
    W::M
    b::B
    function DIR(W::M, b = true) where {M <: AbstractMatrix}
        b = Flux.create_bias(W, b, size(W, 1))
        return new{M, typeof(b)}(W, b)
    end
end

function DIR((in, out)::Pair{<:Integer, <:Integer}; init = Flux.glorot_uniform, bias = true)
    return DIR(init(out, in), bias)
end

Flux.@layer DIR

function (a::DIR)(x::AbstractVecOrMat)
    return NNlib.softplus.(a.W * x .+ a.b) .+ 1
end

(a::DIR)(x::AbstractArray) = _reshape_call(a, x)

"""
    MVE(in => out, σ=NNlib.softplus; bias=true, init=Flux.glorot_uniform)

Create a fully connected layer which implements the Mean-Variance Network which is just a Normal 
distribution whose forward pass is simply given by:

    y = W * x .+ bias

The input `x` should be a vector of length `in`, or batch of vectors represented
as an `in × N` matrix, or any array with `size(x,1) == in`.
The out `y` will be a vector  of length `out*2`, or a batch with
`size(y) == (out*2, size(x)[2:end]...)`
The output will have applied the function `σ(y)` to each row/element of `y` except the first `out` ones.
Keyword `bias=false` will switch off trainable bias for the layer.
The initialisation of the weight matrix is `W = init(out*2, in)`, calling the function
given to keyword `init`, with default `glorot_uniform`.
The weight matrix and/or the bias vector (of length `out`) may also be provided explicitly.
Remember that in this case the number of rows in the weight matrix `W` MUST be a multiple of 2.
The same holds true for the `bias` vector.

# Arguments:
- `(in, out)`: number of input and output neurons
- `σ`: The function to apply to the μ layers which defaults to the softplus.
- `init`: The function to use to initialise the weight matrix.
- `bias`: Whether to include a trainable bias vector.
"""
struct MVE{T <: Chain} <: AbstractEvidentialLayer
    chain::T
end

function MVE(
        (in, out)::Pair{<:Integer, <:Integer}, σ = NNlib.softplus;
        init = Flux.glorot_uniform, bias = true
    )
    return MVE(
        Chain(
            Parallel(
                vcat, μw = Dense(in => out, σ, bias = bias, init = init),
                σw = Dense(in => out, NNlib.softplus, bias = bias, init = init)
            )
        )
    )
end

Flux.@layer MVE

function (a::MVE)(x::AbstractVecOrMat)
    return a.chain(x)
end

(a::MVE)(x::AbstractArray) = _reshape_call(a, x)

"""
    FDIR(in => out; bias=true, init=Flux.glorot_uniform)

Create a Flexible Dirichlet evidential layer from Yoon & Kim, "Uncertainty
Estimation by Flexible Evidential Deep Learning" (2025). Predicts the parameters
of a Flexible Dirichlet (FD) distribution, a mixture of Dirichlets that
generalizes the standard Dirichlet used by `DIR`.

The layer has three output heads from a shared input:
- `α` (out): Gamma concentration parameters via `exp` (α > 0)
- `p` (out): allocation probabilities via `softmax` (Σp = 1)
- `τ` (1): shared dispersion via `softplus` (τ > 0)

The output shape is `(out*2 + 1, batch...)`. Use `split_params(FDIR, y)` or
`splitfdir(y)` to decompose the output into `(α, p, τ)`.

Standard Dirichlet EDL is a special case when τ=1 and p_k = α_k/Σα.

# Arguments:
- `(in, out)`: number of input features and output classes
- `init`: weight initialisation function (default: `glorot_uniform`)
- `bias`: whether to include trainable bias vectors
"""
struct FDIR{T <: Chain} <: AbstractEvidentialLayer
    chain::T
end

function FDIR((in, out)::Pair{<:Integer, <:Integer}; init = Flux.glorot_uniform, bias = true)
    return FDIR(
        Chain(
            Parallel(
                vcat,
                αw = Dense(in => out, exp, bias = bias, init = init),
                pw = Chain(Dense(in => out, bias = bias, init = init), NNlib.softmax),
                τw = Dense(in => 1, NNlib.softplus, bias = bias, init = init)
            )
        )
    )
end

Flux.@layer FDIR

function (a::FDIR)(x::AbstractVecOrMat)
    return a.chain(x)
end

(a::FDIR)(x::AbstractArray) = _reshape_call(a, x)
