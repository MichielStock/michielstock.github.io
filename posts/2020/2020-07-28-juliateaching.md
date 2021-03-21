+++
title = "Julia for teaching optimization"
hascode = false
date = Date(2020, 02, 18)
rss = "SWOT of Julia for an optimization course."
+++
@def tags = ["computing", "code"]

# Julia for teaching optimization

Last year, I taught my optimization course for the fourth time. The first year I was the TA and involved in designing the course, before becoming the lecturer giving both theory lectures and the exercises. Having fought tooth and nail to be able to use Python (as opposed to Matlab), I switched to Julia in the last year. Time for some reflection in using this language for my course.

## Background

Some background of the course. It is called 'Selected Topics in Mathematical Optimization' (STMO) and is taught in our Master in Bioinformatics program to introduce them to various methods in optimization. Most of the students have no computer science background. Students entering from the BioScience engineering track had mathematics courses involving numerically solving differential equations and implementing sampling algorithms in Matlab. Most students arriving from Biochemistry had less mathematics and programming in their bachelor. All students in the master have several programming and algorithms courses, in addition to general courses in machine learning, statistics, etc., which use Python and R.

The goal of STMO is twofold. Firstly, students should get an overview of the different optimization problems and see how these can be applied to computational biology. We see convex optimization, optimal transportation, graph problems (minimal spanning tree, shortest paths, etc.), NP-complete problems (TSP, knapsack, etc.) and metaheuristics (simulated annealing, evolutionary algorithms, CMA-ES, Bayesian optimization, etc.). The secondary goal is to learn how to implement such optimization algorithms and tailor them towards particular problems. Many computational biology problems require solutions that are not off-the-shelf, so students should learn to make their implementation, if only for educational reasons. STMO tries to be a stealth introduction to scientific computing.

Since a large part of the course deals with implementations and driven by my interest in the Julia language, I wanted to switch to the language. I was especially impressed by the recent book [*Algorithms for Optimization*](https://mitpress.mit.edu/books/algorithms-optimization) in which each algorithm is supported by Julia code. After attending Juliacon last year (and saw the talk of one of the authors), I decided to take the leap.

A typical lecture involves an hour of theory followed by an hour of implementing the covered algorithms in a Jupyter notebook. For your information, the repo for the exercises can be found [here](https://github.com/MichielStock/SelectedTopicsOptimization) for the old Python version and [here](https://github.com/MichielStock/STMO) for the new edition in Julia.

## Strengths of Julia

One of the most important benefits of teaching using Julia is the syntax, which is supremely suited for mathematics. Using Unicode symbols may not be entirely good programming practice; it makes it easy to translate equations to code. Simple one-line functions such as `f(x) = 2x^2 + 3x + 4` are also a joy to work with. The `LinearAlgebra` library is also, IMO, much cleaner than Numpy. I spent less time explaining the quirks of the language compare and more explaining the math. Compare a typical piece of code, in Python and Julia.

**Python**:
```python
import numpy as np

def gradient_descent(P, q, x0, alpha=0.1,
                maxiter=1000, epsilon=1e-5):
    x = np.copy(x0)
    grad_x = P.dot(x) + q
    iter = 0
    while np.linalg.norm(grad_x) > epsilon or\
               iter <= maxiter:
        iter += 1
        x -= alpha * grad_x
        grad_x[:] = P.dot(x) + q
    return x

P = np.array([[10, -1],
              [-1, 1]])
q = np.array([[0], [-10]])
x0 = np.zeros((2, 1))

gradient_descent(P, q, x0)
```
**Julia**:
```julia
using LinearAlgebra

function gradient_descent(P, q, x₀; α=0.1,
                maxiter=1000, ϵ=1e-5)
    x = copy(x₀)
    ∇f = x -> P * x + q
    Δx = -∇f(x)
    iter = 0
    while norm(Δx) > ϵ || iter ≤ maxiter
        iter += 1
        x .+= α * Δx
        Δx .= - ∇f(x)
    end
    return x
end

P = [10.0 -1.0;
    -1.0 1.0];
q = [0; -10.0];
x₀ = zeros(2);
```
I find the latter to be much more readable.

Of course, a great benefit of Julia is the speed. Since it is a course for bioinformaticians, students are expected to work with large datasets. Since most of them (including myself) mainly work with high-level programming languages, Julia is an excellent alternative to C to get good performances. Most students don't write truly performant code during the course, but I am confident that they can generate it when the situation arises.

Finally, a strength I enjoyed as a teacher was the ease of developing and distributing the code. It was quite easy to create a small package containing example functions, plotting functions and other utilities. I could add unit tests to make sure everything works. The package manager made it a breeze for the students to install this. This made it also easy to make use of the ecosystem of Julia packages, from automatic differentiation, to Colors, to off-the-shelf optimization packages such as JuMP and Optim.

## Weaknesses of Julia

The most common issue students reported in the feedback form is the lack of sufficient documentation. The Julia community seems to be more active on its [Slack](https://julialang.slack.com/) and [Discourse](https://discourse.julialang.org/) channels, rather than, say, Stack Overflow. I noted that students don't always know where to look. Of course, there are magnitudes more examples and solutions available for Python problems than for Julia, so this will remain a bottleneck. Due to the pandemic, I also was not possible to help with the more obvious issues, so this should also be taken into account.

Another irritation was the long waiting times. I could tout Julia's speed all day long, but from the students' point of view, there were long waiting times to get started. The Time To The First Plot is a well-known annoyance. Because I was updating and tweaking the course repo before every class, students had to update the STMO package and needed to wait before it was recompiled. Since most of the exercises were didactic toy examples, the difference in running time between Python (e.g., milliseconds) and Julia (e.g., microseconds) are often unnoticeable. The projects were more computationally demanding, so there were luckily moments when the language could shine.

## Opportunities for Julia

I tried to avoid going too deep into the type system as this is already an advanced concept. However, I can see great potential in using the type system and dispatch for creating an ontology of the algorithms and problems. We could define `DescentMethod` as an abstract type with, for example, `GradientDescent` as a concrete type. One thing I tried is giving 'computational definitions' for certain concepts, in contrast to mathematical ones. For example, graphs are defined as:

```julia
EdgeList{T} = Array{Tuple{T,T},1}
WeightedEdgeList{R,T} = Array{Tuple{R,T,T},1}
Vertices{T} = Array{T,1}
AdjList{R,T} = Dict{T,Array{Tuple{R,T},1}}
```
which the different representations precisely!

Another thing I want to explore further is macros. Last year, I did not feel confident enough to rely on them. They seem to be promising to add to the functionality to, for example, track convergence of optimization algorithms. I should also make more use of the `PlotRecipes` library to create custom plotting functions. Intuitively, students seem to follow the idea of metaprogramming, so I would be nice to explore this more.

Finally, Julia can be used to explore the low-level details of code execution, up to the [hardware level](https://biojulia.net/post/hardware/). We could elaborate somewhat on this, including how to run specific optimization algorithms on GPU.

## Threats for Julia

In the context of teaching, a justified remark would be if it is advisable to learn a new language for a single course. One can undoubtedly motivate the language for numerical computing, but its future position in other domains remains uncertain. Will it grow in machine learning? Julia has potential for [bioinformatics applications](https://biojulia.net/post/seq-lang/) but will it gain prominence there? Will Julia grow beyond academia? And can we not do all this stuff in Python using Numba/Pytorch/JAX (for the record, I don't think so)? Known issues such as the language not being perceived as stable or many packages with a low [bus factor](https://en.wikipedia.org/wiki/Bus_factor) might also demotivate.

## Conclusions

All in all, I am happy with the transition to Julia for my course. I found it smoother than expected, but this might also because I am a bit more experienced than when I made the course material in Python. The language is well-suited to teach mathematics and lends itself well for diving deeper in real-world applications. Students found the language an added value, though it was not always evident to become proficient at it.
