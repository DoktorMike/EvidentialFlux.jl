using EvidentialFlux
using Flux
using UnicodePlots


function gendata(n)
    x1 = Float32.(randn(2, n))
    x2 = Float32.(randn(2, n) .+ 2)
    y1, y2 = Float32.(ones(n)), Float32.(zeros(n))
    hcat(x1, x2), hcat(vcat(y1, y2), 1 .- vcat(y1, y2))'
end
n = 200
X, y = gendata(n)

# See the data
p = scatterplot(X[1, 1:n], X[2, 1:n], color = :green, width = 80, height = 30)
scatterplot!(p, X[1, (n+1):(n+n)], X[2, (n+1):(n+n)], color = :red)

m = Chain(Dense(2 => 30), DIR(30 => 2))
opt = Flux.Optimise.AdamW(0.01)
p = Flux.params(m)

epochs = 500
trnlosses = zeros(epochs)
for e in 1:epochs
    local trnloss = 0
    grads = Flux.gradient(p) do
        α = m(X)
        trnloss = dirloss(y, α, e)
        trnloss
    end
    trnlosses[e] = trnloss
    Flux.Optimise.update!(opt, p, grads)
end
scatterplot(1:epochs, trnlosses, width = 80, height = 30)

α̂ = m(X)
ŷ = α̂ ./ sum(α̂, dims = 1)
u = uncertainty(α̂)

contourplot(-5:.01:5, -5:.01:5, (x, y) -> uncertainty(m(vcat(y,x)))[1])
