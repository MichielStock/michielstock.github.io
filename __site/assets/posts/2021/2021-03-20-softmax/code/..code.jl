# This file was generated, do not modify it. # hide
/softmax/probex
using Plots

x = [1, 1, 2, 3.4, -0.1, 3, 1.6]  # unnormalized prob vector
p = softmax(x)

bar(p, label="", color=:green)

savefig(joinpath(@OUTPUT, "probs.svg")) # hide