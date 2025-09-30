push!(LOAD_PATH, "../src/")

using Documenter
using EvidentialFlux
using Flux

# makedocs(sitename = "EvidentialFlux", checkdocs = :exports,
#     format = Documenter.HTML(),
#     modules = [EvidentialFlux])
makedocs(sitename = "EvidentialFlux")

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
deploydocs(repo = "github.com/DoktorMike/EvidentialFlux.jl.git")
