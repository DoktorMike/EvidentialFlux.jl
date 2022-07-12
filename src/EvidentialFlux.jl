module EvidentialFlux

using Flux
using NNlib
using SpecialFunctions

# Write your package code here.

include("dense.jl")
export NIG
export DIR

include("losses.jl")
export nigloss
export dirloss

include("utils.jl")
export uncertainty
export evidence
export predict

end
