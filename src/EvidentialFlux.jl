module EvidentialFlux

using Flux
using NNlib
using SpecialFunctions

# Write your package code here.

include("dense.jl")
export NIG

include("losses.jl")
export nigloss

include("utils.jl")
export uncertainty
export evidence
export predict

end
