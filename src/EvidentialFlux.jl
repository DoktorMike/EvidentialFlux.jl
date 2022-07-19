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
export nigloss2
export dirloss

include("utils.jl")
export uncertainty
export aleatoric
export epistemic
export evidence
export predict

end
