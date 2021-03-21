# This file was generated, do not modify it. # hide
@show gumbelsoftmax(x)

plot((bar(gumbelsoftmax(x, κ=κ), label="", color=c, title="kappa=$κ")
			for rep in 1:3  for (κ, c) in zip([0.1, 0.5, 1, 10], [:darkred, :red, :orange, :yellow]))...,
			layout=(3, 4), yticks=[])

savefig(joinpath(@OUTPUT, "gsm.svg")) # hide