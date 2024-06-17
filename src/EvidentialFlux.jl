module EvidentialFlux

using Flux
using NNlib
using SpecialFunctions

include("dense.jl")
export NIG
export DIR
export MVE

include("losses.jl")
export nigloss
export nigloss2
export dirloss
export mveloss

include("utils.jl")
export uncertainty
export aleatoric
export epistemic
export evidence
export predict

end
