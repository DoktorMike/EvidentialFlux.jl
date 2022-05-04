
struct NIG{F,M<:AbstractMatrix,B}
        W::M
        b::B
        σ::F
        function NIG(W::M, b = true, σ::F = identity) where {M<:AbstractMatrix,F}
                b = Flux.create_bias(W, b, size(W, 1))
                return new{F,M,typeof(b)}(W, b, σ)
        end
end

function NIG((in, out)::Pair{<:Integer,<:Integer}, σ = identity;
        init = Flux.glorot_uniform, bias = true)
        NIG(init(out * 4, in), bias, σ)
end

Flux.@functor NIG

function (a::NIG)(x::AbstractVecOrMat)
        o = a.σ.(a.W * x .+ a.b)
        μ = @view o[1:4, :]
        ν = @view o[5:8, :]
        ν .= NNlib.softplus.(ν)
        α = @view o[9:12, :]
        α .= NNlib.softplus.(α) .+ 1
        β = @view o[13:16, :]
        β .= NNlib.softplus.(β)
        return o
end
(a::NIG)(x::AbstractArray) = reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)
