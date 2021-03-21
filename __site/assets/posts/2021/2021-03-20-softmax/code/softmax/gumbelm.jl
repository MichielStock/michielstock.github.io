# This file was generated, do not modify it. # hide
randg() = - log(-log(rand()))  # standard Gumbel sample
randg(n::Int...) = - log.(-log.(rand(n...)))  # array filled with standard Gumbel values

gumbelmaxtrick(x; γ=1) = x ./ γ .+ randg(length(x)) |> argmax