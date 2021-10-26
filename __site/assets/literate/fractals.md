<!--This file was generated, do not modify it.-->
# Newton fractal in Julia using Symbolics.jl and Colors.jl

The most recent video of 3blue1brown covered Newton fractals, a familie of fractal curves that are obtained by applying  Newton's method to complex numbers. Since I am covering Newton's method in my optimization course this week, it might be a nice exercise play around with this topic.

## Newton's method

Newton's method is a *root-finding* algorithm. We can use it to find roots of a function $f$, i.e., inputs $x$ for which it holds that $f(x)=0$.

When executing Newton's method, we start with an initial $x$ and iteratively apply the following rule till convergence:

$$
x \mapsto x - \frac{f(x)}{f'(x)}\,.
$$

This rule can be derived by taking the first-order Taylor approximation of $f(x+h)$ in $x$ and solving for the step $h$ that sets the approximation to zero.

Let us apply this on a simple function!

```julia:ex1
f(x) = x^3 - 1
```

Taking the derivative is easy but let us use this opportunity to explore Symbolics.jl, Julia's new CAS system.

```julia:ex2
using Symbolics
```

define the variable

```julia:ex3
@variables z
```

function

```julia:ex4
p = f(z)
```

derivative

```julia:ex5
p′ = Differential(z)(p) |> expand_derivatives
```

newton step

```julia:ex6
zmap = z - p / p′
```

compile to Julia function

```julia:ex7
update = build_function(zmap, z)  |> eval

using Plots

plot(f, -2, 2, xlabel="x", label="f(x)")
```

