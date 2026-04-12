module EvidentialFlux

using Flux
using Flux: ignore_derivatives
using NNlib
using SpecialFunctions

include("dense.jl")
export AbstractEvidentialLayer
export NIG
export PG
export EG
export BB
export BNB
export DIR
export MVE
export FDIR

include("losses.jl")
export nigloss
export nigloss_scaled
export nigloss_ureg
export nigloss2  # deprecated alias for nigloss_scaled
export nigloss3  # deprecated alias for nigloss_ureg
export dirloss
export dirloss_cor
export dirloss2  # deprecated alias for dirloss_cor
export dirmultloss
export fdirloss
export nllpg
export pgloss
export nlleg
export egloss
export nllbb
export bbloss
export nllbnb
export bnbloss
export mveloss
export nllstudent

include("utils.jl")
export split_params
export splitnig
export splitmve
export splitfdir
export splitpg
export spliteg
export splitbb
export splitbnb
export uncertainty
export aleatoric
export epistemic
export evidence
export predict
export predictive
export predictive_mean

end
