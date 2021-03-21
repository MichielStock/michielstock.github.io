# This file was generated, do not modify it. # hide
gumbelsoftmax(x; κ=1) = softmax(x .+ randg(length(x)), γ=κ)