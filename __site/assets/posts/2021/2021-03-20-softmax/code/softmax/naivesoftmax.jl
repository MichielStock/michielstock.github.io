# This file was generated, do not modify it. # hide
naivesoftmax(x; γ=1) = exp.(x ./ γ) / sum(exp.(x/γ))