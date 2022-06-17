"""
    NIG(in => out, σ=NNlib.softplut; bias=true, init=Flux.glorot_uniform)
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
given to keyword `init`, with default [`glorot_uniform`](@doc Flux.glorot_uniform).
The weight matrix and/or the bias vector (of length `out`) may also be provided explicitly.
Remember that in this case the number of rows in the weight matrix `W` MUST be a multiple of 4.
The same holds true for the `bias` vector.

# Arguments:
- `(in, out)`: number of input and output neurons
- `σ`: The function to use to secure positive only outputs which defaults to the softplus function.
- `init`: The function to use to initialise the weight matrix.
- `bias`: Whether to include a trainable bias vector.
"""
struct NIG{F,M<:AbstractMatrix,B}
        W::M
        b::B
        σ::F
        function NIG(W::M, b = true, σ::F = NNlib.softplus) where {M<:AbstractMatrix,F}
                b = Flux.create_bias(W, b, size(W, 1))
                return new{F,M,typeof(b)}(W, b, σ)
        end
end

function NIG((in, out)::Pair{<:Integer,<:Integer}, σ = NNlib.softplus;
        init = Flux.glorot_uniform, bias = true)
        NIG(init(out * 4, in), bias, σ)
end

Flux.@functor NIG

function (a::NIG)(x::AbstractVecOrMat)
	nout = Int(size(a.W, 1) / 4)
        o = a.W * x .+ a.b
        γ = o[1:nout, :]
	ν = o[(nout+1):(nout*2), :]
        ν = a.σ.(ν)
	α = o[(nout*2+1):(nout*3), :]
        α = a.σ.(α) .+ 1
	β = o[(nout*3+1):(nout*4), :]
        β = a.σ.(β)
        return vcat(γ, ν, α, β)
end

(a::NIG)(x::AbstractArray) = reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

#function predict(m::NIG, x::AbstractVecOrMat)
#	nout = Int(size(m.W, 1) / 4)
#        o = m.W * x .+ m.b
#        γ = o[1:nout, :]
#	ν = o[(nout+1):(nout*2), :]
#        ν = m.σ.(ν)
#	α = o[(nout*2+1):(nout*3), :]
#        α = m.σ.(α) .+ 1
#	β = o[(nout*3+1):(nout*4), :]
#        β = m.σ.(β)
#        return γ, uncertainty(ν, α, β), uncertainty(α, β)
#end

