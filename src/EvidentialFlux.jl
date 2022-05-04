module EvidentialFlux

using Flux
using NNlib

# Write your package code here.

hello() = println("Hello from EvidentialFlux!")


include("dense.jl")
export NIG

end
