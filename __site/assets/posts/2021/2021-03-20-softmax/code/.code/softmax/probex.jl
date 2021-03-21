# This file was generated, do not modify it. # hide
using Plots

x = [1, 1, 2, 3.4, -0.1, 3, 1.6]  # unnormalized prob vector
p = softmax(x)

bar(p, label="", color=:green)
savefig(joinpath(@OUTPUT, "probs.svg")) # hide

plot((bar(gumbelsoftmax(x, κ=κ), label="", color=c, title="kappa=$κ")
			for rep in 1:3  for (κ, c) in zip([0.1, 0.5, 1, 10], [:darkred, :red, :orange, :yellow]))...,
			layout=(3, 4), yticks=[])

savefig(joinpath(@OUTPUT, "gsm.svg")) # hide