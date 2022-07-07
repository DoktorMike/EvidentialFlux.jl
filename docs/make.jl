using Documenter
using EvidentialFlux

makedocs(
    sitename = "EvidentialFlux",
    format = Documenter.HTML(),
    modules = [EvidentialFlux]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
