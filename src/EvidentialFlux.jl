module EvidentialFlux

using Flux
using NNlib
using SpecialFunctions

# Write your package code here.

hello() = println("Hello from EvidentialFlux!")

include("dense.jl")
export NIG

include("losses.jl")
export nignll

end
