module EvidentialFlux

using Flux
using Flux: ignore_derivatives
using NNlib
using SpecialFunctions

include("dense.jl")
export AbstractEvidentialLayer
export NIG
export PG
export DIR
export MVE
export FDIR

include("losses.jl")
export nigloss
export nigloss2
export nigloss3
export dirloss
export dirloss2
export dirmultloss
export fdirloss
export nllpg
export pgloss
export mveloss
export nllstudent

include("utils.jl")
export split_params
export splitnig
export splitmve
export splitfdir
export splitpg
export uncertainty
export aleatoric
export epistemic
export evidence
export predict

end
