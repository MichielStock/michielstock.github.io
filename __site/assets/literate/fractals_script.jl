# This file was generated, do not modify it.

f(x) = x^3 - 1

using Symbolics

@variables z

p = f(z)

p′ = Differential(z)(p) |> expand_derivatives

zmap = z - p / p′

update = build_function(zmap, z)  |> eval

using Plots

plot(f, -2, 2, xlabel="x", label="f(x)")

