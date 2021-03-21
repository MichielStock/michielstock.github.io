# This file was generated, do not modify it. # hide
@show x = [1, 1, 2, 3.4, -0.1, 3, 1.6]  # unnormalized prob vector
@show p = softmax(x)
@show gumbelsoftmax(x)