# This file was generated, do not modify it. # hide
softmax(x; γ=1) = exp.((x .- logsumexp(x; γ)) ./ γ)