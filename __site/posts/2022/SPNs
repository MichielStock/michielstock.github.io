### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 5198b54e-e5a7-11ec-3e8c-e5dbc729326d
using Distributions, Plots, PlutoUI

# ╔═╡ 65af1e37-81e0-444d-a184-dbd19843012c
md"""
# Sum product networks: $\mathcal{O}(1)$ inference

In this post, I will discuss the basic ideas behind sum-product networks, powerful probabilistic models that are fully tractable. 
"""

# ╔═╡ ef9a151c-c67f-4ae6-b13f-4e0d33a4633e
md"""


## Motivation

In statistical modelling, we often want to model the distribution between a bunch of variables $X_1,\ldots, X_n$. For example, we might want to model the microbiome composition of patients together with a bunch of meta-variables such as age, gender, gene expression, health etc. Not only do these variables follow different distributions (e.g., the abundance of a microbial species is negative binomial or Poisson, presence-absence is Bernoulli distributed, expression Gaussian etc.), but the variable likely depends on each other in complex ways.

Concepts such as Bayesian networks and graphical models (see [Bishop's book](https://docs.google.com/viewer?a=v&pid=sites&srcid=aWFtYW5kaS5ldXxpc2N8Z3g6MjViZDk1NGI1NjQzOWZiYQ) for an excellent overview) allow combining the distributions of the individual variables in a complex joint distribution. 

Given the joint distribution (and provided we can estimate its parameters from data), we can ask all kinds of interesting questions:
- How often does a kind of bacterial species occur? How often do two species occur together? How often do they appear together in patients that exhibit a particular phenotype?
- If we know some bacteria are present, how likely is it that other bacteria are present?
- What is the health status (e.g. IBS) given a microbial profile? What if we know some additional meta-data?

All the above questions can be quantitatively answered by marginalizing, conditioning, computing modes, sampling etc. Unfortunately, for graphical models in general, this can be rather complex.
"""

# ╔═╡ acf24236-ad7f-4318-9ae7-00599abaeacb
md"""

## Models with efficient inference

Some important and popular probabilistic models can be manipulated easily. For example, for the good old multivariate normal (MVN) distribution, we can trivially compute the [marginal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Marginal_distributions) and the [conditional distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions). The reason is that if a random vector is MVN, any linear combination of the variables is normally distributed, greatly simplifying computations.

Take a look for yourself.
"""

# ╔═╡ dcddc488-d5c0-4464-943a-1d8e619c848a
mvn = MultivariateNormal([2, 8], [5 3; 3 2.5])

# ╔═╡ 129600f1-3beb-403d-9f13-05138330037d
md"We can pick a value of $x_1$ to condition on:"

# ╔═╡ f48ac486-34be-4f73-8b43-0e48ac41c238
@bind cx1 Slider(-2:0.2:7, default=3, show_value=true)

# ╔═╡ b703c105-9db5-4da3-b73c-87f55134c652
begin
	pmvnpdf = contourf(-5:0.1:7, 3:0.1:15, (x,y)->pdf(mvn, [x,y]), color=:speed, title="PDF of a MVN", xlab="x₁", ylab="x₂")
	vline!([cx1], color=:orange, label="condition X₁=$cx1", lw=2)
	
end

# ╔═╡ 37691acf-5a11-4564-b70b-db2d3fdfedfe
begin
	pmvnmargcond  = plot(x2->pdf(Normal(mvn.μ[2], √(mvn.Σ[2,2])), x2), 3, 15, xlab="x₂", color=:blue, lw=2, label="marginal PDF of X₂")
	plot!(x2->pdf(mvn, [cx1, x2]) / pdf(Normal(mvn.μ[1], √(mvn.Σ[1,1])), cx1), 3, 15, color=:orange, lw=2, label="conditional PDF of P(X₂ | x₁ = $cx1)")
end

# ╔═╡ af5f1e48-62c9-4d86-b553-d4674674e186
md"""
Important that the MVN may be, it is a too simplistic (dare we say, boring) model for numerous applications. The MVN has only a single peak, can only capture linear correlations in the data and has famously narrow tails. 

If we restrict ourselves to specific topologies of graphical models such as chains (think hidden Markov models) or trees, we can still use efficient inference. However, these types of graphical models greatly restrict their applicability. 

Let us try to develop some general operations we can use to construct graphical models that are both expressive as well as tractable!
"""

# ╔═╡ a60b6236-7079-4e37-aa7a-8b795f8046d1
md"""
## Operations that retain tractability

Let us look at two types of operations to combine probability distributions: *products* and *mixtures*.
"""

# ╔═╡ f9f54997-9fc5-4a08-b76b-226757285c41
md"""
### Multiplying PDFs of disjoint variables

If we have distributions of stochastic variables $X_1$ and $X_2$, we can easily combine them into a joint distribution by assuming that they are independent. This can simply be done by *multiplying* the respective PDFs:

$$f_{X_1X_2}(x_1,x_2) = f_{X_1}(x_1)f_{X_2}(x_2)\,.$$

In `Distributions`, one can obtain such a binding using `product_distribution`. For example.
"""

# ╔═╡ 496640ea-3831-4452-8653-08ae7f059e97
distX₁ = Laplace(8, 6)

# ╔═╡ 9b1d317b-2615-4f87-a015-7d606e8dc5c5
plot(x->pdf(distX₁, x), -10, 25, label="PDF of X₁", xlab="x₁")

# ╔═╡ dbd43b55-6cb5-45e8-b7c5-2bdd5e8bb738
distX₂ = TriangularDist(-2, 2, 1)

# ╔═╡ 03074a30-1589-41ae-83e8-23a149bc7a49
plot(x->pdf(distX₂, x), -3, 3, label="PDF of X₁₂", xlab="x₂")

# ╔═╡ f8eb7e0a-8b28-4c49-a98c-0f055c40ec4b
distX₁X₂ = product_distribution([distX₁, distX₂])

# ╔═╡ 1fa7deda-0bb5-40d6-aee8-a45ea8e4195b
contourf(-10:0.1:25, -2:0.1:2, (x1,x2)->pdf(distX₁X₂, [x1,x2]), color=:speed, xlabel="x₁",ylabel="x₂", title="product distribution")

# ╔═╡ c27ffce9-09d7-49b3-addb-6272556305b3
md"We see a reasonably interesting PDF for the joint distribution. As we model $X_1$ and $X_2$ as indpendent, the conditonal distribtion is the same as the marginal distribution, e.g., $P(X_1\mid X_2) = P(X_1)$."

# ╔═╡ 09238687-027f-4e19-8c93-f7fa52e04421
md"""
### Mixtures of variables

A mixture distribution is an elegant way of combining several simple distributions for a variable (or set of variables) into a more complex distribution. Given $k$ PDFs with as as input $x$, $f^{(1)}(x), \ldots f^{(k)}(x)$, we have

$$f(x) = \sum_iw_if^{(i)}(x)\,,$$

where the $w_i$'s with $w_i\ge 0$ and $\sum_iw_i$ are the mixing coefficients. We can think of these coefficients as a prior probability of sampling one of the $k$ distributions in order to sample a value of $X$ from the resulting distribution.

In `Distributions`, one can easily compose a mixture using `MixtureModel`:
"""

# ╔═╡ d116615b-9428-41ea-8604-ef4fcdaf8005
f¹ₓ, f²ₓ, f³ₓ = Normal(3, 0.5), Normal(6, 1), Normal(5, 4)

# ╔═╡ 805bf2b0-c34c-4251-95ca-a9b0bb6c9efc
w = [0.6, 0.3, 0.1]  # mixing weights

# ╔═╡ aac677db-8170-4c45-aade-c6c588f3acea
fmix = MixtureModel([f¹ₓ, f²ₓ, f³ₓ], w)

# ╔═╡ 22ecb9d8-ef57-4625-91b1-f0b6cea6b3a1
md"""
Given a sample from the mixture, we can easily infer from which component it was likely drawn. To this end, we 

$$P(\text{sample is from component }i)\propto w_i\times f^{(i)}\,.$$

Pick a value of $x$:
"""

# ╔═╡ 6b2aa326-4c98-4df5-84b0-c52b8c2277af
@bind xs Slider(0:0.2:15, default=6, show_value=true)

# ╔═╡ 6739045b-b3e7-468b-8104-bcc4f96905cb
begin
	plot(x->pdf(fmix, x), 0, 15, lw=3, label="mixture", xlab="x", ylab="density")
	plot!(x->w[1]pdf(f¹ₓ, x), 0, 15, label="f¹")
	plot!(x->w[2]pdf(f²ₓ, x), 0, 15, label="f²")
	plot!(x->w[3]pdf(f³ₓ, x), 0, 15, label="f³")
	vline!([xs], label="", lw=2, color=:gold)
end

# ╔═╡ dc3cf268-f580-4f0f-a06b-9df873302ffa
begin 
	post = w .* [pdf(f, xs) for f in [f¹ₓ, f²ₓ, f³ₓ]]
	post ./= sum(post)
	bar(post, xticks=[1, 2, 3], xlabel="component", ylabel="posterior probability", label="X=$xs", color=:gold)
end

# ╔═╡ 368b9212-efad-4689-8852-d2a0d5e307ca
md"""
## Sum-product networks

Given these two basic concepts, product distributions and mixtures, we can introduce the sum-product networks (SPN). The main idea is that we want to construct probabilistic models from which we can still perform exact and efficient inference. The property we desire is being efficiently summable:

> A function is **efficiently summable** iff its sum over any subset of its scope can be computed in polynomial time of de cardinality of the subset.

If a function is:
- a sum of efficiently summable functions with the *same scope*, or;
- a product of efficiently summable functions with *disjoint scopes*,
then it is efficiently summable.

As we have seen, we can, of course, use many more variables and mix and match the rules. Let us look at a simple example.
"""

# ╔═╡ c8fa646c-e77a-4fc8-9039-78cbe6d5dc23
md"""
Below, two variables are combined by creating two mixed product distributions. 
```
       +
    /     \
   *       *
 /  \    /   \
X₁   X₂  X₁   X₂
```
We can easily compute everything we want:
- **computing probabilities**: given and $(x_1, x_2)$ pair, we use the PDFs (or PMFs if they are discrete), plug in the values and just follow the tree to the root;
- **marginalizing**: if we want the marginal distribution of $X_1$, we merely ignore $X_2$, as the two are independent *within* a cluster;
- **conditioning**: if we want something like $P(X_2\mid X_1)$, we can condition on $X_1$ in the same way. For a given $X_1=x_1$, we can compute the *posterior* probability that the observation arose from each cluster (again, within the cluster, these variables are independent) and use the posterior to update the weights of the sum operation accordingly;
- **mod**: the mode of the distribution can also be considered per cluster. 

"""

# ╔═╡ e020ffc7-89c5-4335-bbd4-cdf89315bed0
md"Let us implement a version of this SPN with two binary variables. Julia's type system allows use to be a bit clever, we can assume that when `nothing` is given as an input, we want to marginalize over this variable."

# ╔═╡ 21e03a45-5a1e-46c0-99ab-11b31b72a2ad
pbin(x::Bool, p) = x ? p : 1 - p

# ╔═╡ c1e78778-989e-4c1b-9838-5429099be143
pbin(x::Nothing, p) = 1

# ╔═╡ 7b29bc39-8666-48fa-84b0-96c5caea283a
simple_spn(x₁=nothing, x₂=nothing) = 0.7pbin(x₁, 0.5) * pbin(x₂, 0.2) + 0.3pbin(x₁, 0.1) * pbin(x₂, 0.1)

# ╔═╡ 5048ac99-e44b-4d24-a2a1-bdfc5e957d95
for (x₁, x₂) in Iterators.product([false, true], [false, true])
	println("P(X₁ = $x₁, X₂=$x₂) = $(simple_spn(x₁, x₂))")
end

# ╔═╡ b62ab732-6804-4216-b833-85c0f864109d
for x₁ in [false, true]
	println("P(X₁ = $x₁) = $(simple_spn(x₁))")
end

# ╔═╡ 27519334-a088-44e5-a6ff-69a7391c355a
simple_spn_cond_x1(x₁::Bool, x₂::Bool) = simple_spn(x₁, x₂) / simple_spn(x₁, nothing)

# ╔═╡ 97389744-a10b-498c-935a-36e4a1a8d34d
for (x₂, x₁) in Iterators.product([false, true], [false, true])
	println("P(X₂=$x₂ | X₁ = $x₁) = $(simple_spn_cond_x1(x₁, x₂))")
end

# ╔═╡ 95beb5fc-4ce2-4f3d-bae9-d341a8279790
md"""
A different configuration is given below.
```
       *
    /     \
   +       +
 /  \    /   \
X₁   X₁  X₂   X₂
```
Here, one first creates more complex distributions for the individual variables before combining them with a product distribution.
"""

# ╔═╡ 58ffbdff-bf65-4346-989f-0734eb03a82b
simple_spn2(x₁=nothing, x₂=nothing) = (0.5pbin(x₁, 0.7) + 0.5pbin(x₁, 0.2)) * (0.7pbin(x₂, 0.9) * 0.3pbin(x₂, 0.7))

# ╔═╡ 7da5d3f5-21db-48ae-8ac3-ceea81b10b60
simple_spn2(true, false)

# ╔═╡ 1199cda4-aba2-423e-bc7b-5258f4b9ee9b
md"Of course, we could always combine the two versions of these distributions in another mixture if we want!"

# ╔═╡ 1bb17e59-f5b9-4a16-8b93-7960ab3b88ab
md"Let use construct a slightly more interesting example. We will use the functiontionality of `Distributions`. We limit ourselves to two real-valued variables."

# ╔═╡ 81375947-0ff8-4ae9-9f9c-e340b6516424
md"> **Warning** The following implementations are just didactic implementations compatible with `Distributions`. It is not meant to be generable, though a `SNP` type could likely be implemented fairly easily."

# ╔═╡ 6a8c2c06-b6c7-4c7b-9119-11d3558e9af9
spn = MixtureModel(
	[MixtureModel(
		[product_distribution([Normal(10, 1), Normal(20, 1.5)]),
				product_distribution([Normal(20, 1.5), Normal(21, 0.8)]),
				product_distribution([Uniform(5, 25), Uniform(7, 9)]),
				product_distribution([TriangularDist(12, 17), TriangularDist(14, 16, 15.5)]),
				product_distribution([Uniform(2, 28), LogNormal(log(27), 0.1)]),
			],[0.3, 0.25, 0.25, 0.15, 0.05]),
		product_distribution([Uniform(0, 30), Uniform(0, 30)])], [0.99, 0.01])

# ╔═╡ c7b4f50d-0c83-43dc-8980-0a8af08dd050
md"This works just like any distribution:"

# ╔═╡ db6fef20-7a71-4fd7-86b7-26146aec089f
logpdf(spn, [8.3, 12.4])

# ╔═╡ 6d984c89-acb1-49ff-b27c-c91b510c8714
mean(spn)

# ╔═╡ d7209426-dd4c-4cde-adbe-59636f9490bd
rand(spn)

# ╔═╡ 417d1646-4fd1-4985-b367-a231a5bd2e1c
cov(spn)

# ╔═╡ f0a8db15-4f76-4a0e-8d41-1027f379b29f
pcontour = contourf(-0:0.1:30, -0:0.1:30, (x,y)->logpdf(spn, [x,y]), color=:speed, xlab="x₁", ylab="x₂")

# ╔═╡ b41eb5db-be21-4f3c-b808-9885cebea42e
md"Samples cover the whole distribution."

# ╔═╡ e25a29bf-9a1a-489e-9892-07204c44ab3b
X = rand(spn, 10_000)

# ╔═╡ 10fa7add-a009-4d66-b4bc-9e7761f53c44
scatter(X[1,:], X[2,:], alpha=0.1, xlab="x₁", ylab="x₂")

# ╔═╡ edd4804c-5e9d-4677-a41e-3a5533567978
md"Computing the marginal distribution of $X_1$ can be done by providing the appropriate methods. As a `product_distribution` of two normals is a `DiagNormal`, we also cover this type."

# ╔═╡ b532e032-8fb2-4def-b6e1-0750c264c7c2
marginal_X1(prod::Product) = prod.v[1]

# ╔═╡ f37df05b-6bc9-42b5-87d6-450f16236d09
marginal_X1(mixture::MixtureModel) = MixtureModel(marginal_X1.(components(mixture)),
										mixture.prior.p)	

# ╔═╡ 1f5d54f8-bceb-41a3-b620-d9098f9a8dbb
marginal_X1(mvn::MultivariateNormal) = Normal(mvn.μ[1], √(mvn.Σ[1,1]))

# ╔═╡ 8a9d80ba-713a-4102-85c6-7a3807738c40
md"We see that this works! The marginal is just another SPN."

# ╔═╡ e9b4362b-cd35-4385-b9b9-4e155d2a1729
marginal_X1(spn)

# ╔═╡ a632cc79-ec51-41e2-846a-ee6a078ce093
plot(x->logpdf(marginal_X1(spn), x), -0:0.1:30, xlabel="x₁", label="marginal PDF of X₁", lw=2)

# ╔═╡ d4a41154-7154-4e78-8ae2-b0d6056ff5ce
md"Conditioning on $X_1=x_1$ is also not too difficult."

# ╔═╡ 3254e3bd-5d9a-4448-97bd-45655c40c714
condition_on_X1(prod::Product, x1=nothing) = prod.v[2]

# ╔═╡ 751534bc-a2f4-4499-89c0-89e1046b9e6e
function condition_on_X1(mixture::MixtureModel, x1)
	priors = mixture.prior.p
	# find likelihood cluster
	ll = [pdf(marginal_X1(comp), x1) for comp in components(mixture)]
	posteriors = priors .* ll  # update
	posteriors ./= sum(posteriors)  # renormalize
	return MixtureModel([condition_on_X1(comp, x1) for comp in components(mixture)], posteriors)
end

# ╔═╡ b829503b-ecb9-4b25-aa9b-3c6871f9a21d
condition_on_X1(mvn::DiagNormal, x1=nothing) = Normal(mvn.μ[2], √(mvn.Σ[2,2]))

# ╔═╡ 3b094dca-a532-4c7e-bc64-815d8c656b3b
md"Again, pick a value of $X_1$ to condition on:"

# ╔═╡ b67252c7-2fad-4553-a248-c54ca492f638
@bind x1c Slider(0:0.2:30, show_value=true, default=10)

# ╔═╡ ebcd776d-2311-45b5-9826-38a2c5e266eb
cond = condition_on_X1(spn, x1c)

# ╔═╡ 3354324a-86a0-4686-a430-5086617dd096
begin
	contourf(-0:0.25:30, -0:0.25:30, (x,y)->logpdf(spn, [x,y]), color=:speed, xlab="x₁", ylab="x₂")
	vline!([x1c], lw=2, label="X₁=$x1c", color="orange")
end

# ╔═╡ f31ce7df-5e05-4cd2-b811-e0a40ba53d2b
plot(x->pdf(cond, x), 0:0.1:30, label="PDF of P(X₂ | X₁=$x1c)", xlab="x₂", color="orange", lw=2)

# ╔═╡ 6e9afe84-1b3c-43f6-9227-f4200d4b366a
md"""
## Conclusion

Sum product networks allow one to create complex distributions with basic building blocks. I found it a really cool concept when I first heard from them from Pedro Domingos in the MLSS in 2012. It combines some fundamental concepts in stochastic modelling in a clever way. 

- Hoifung Poon, Pedro Domingos, *Sum-Product Networks: A New Deep Architecture* [Arxiv](https://arxiv.org/abs/1202.3732)
- [Learning Tractable Probabilistic Models](https://ix.cs.uoregon.edu/~lowd/ltpm-uai2014.pdf)
- [Juice](https://github.com/Juice-jl): Probabilistic circuits in Julia
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Distributions = "~0.25.62"
Plots = "~1.29.1"
PlutoUI = "~0.7.39"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "0f4e115f6f34bbe43c19751c90a38b2f380637b9"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.3"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "924cdca592bc16f14d2f7006754a621735280b74"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.1.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "0ec161f87bf4ab164ff96dfacf4be8ffff2375fd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.62"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c98aea696662d09e215ef7cda5296024a9646c75"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "3a233eeeb2ca45842fe100e0413936834215abf5"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.4+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "3e32c8dbbbe1159a5057c80b8a463369a78dd8d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.12"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "9e42de869561d6bdf8602c57ec557d43538a92f0"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.29.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "a9e798cae4867e3a41cae2dd9eb60c047f1212db"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "383a578bdf6e6721f480e749d503ebc8405a0b22"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.6"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2c11d7290036fe7aac9038ff312d3b3a2a5bf89e"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.4.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "9abba8f8fb8458e9adf07c8a2377a070674a24f1"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.8"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─65af1e37-81e0-444d-a184-dbd19843012c
# ╠═5198b54e-e5a7-11ec-3e8c-e5dbc729326d
# ╟─ef9a151c-c67f-4ae6-b13f-4e0d33a4633e
# ╟─acf24236-ad7f-4318-9ae7-00599abaeacb
# ╠═dcddc488-d5c0-4464-943a-1d8e619c848a
# ╟─b703c105-9db5-4da3-b73c-87f55134c652
# ╟─129600f1-3beb-403d-9f13-05138330037d
# ╟─f48ac486-34be-4f73-8b43-0e48ac41c238
# ╟─37691acf-5a11-4564-b70b-db2d3fdfedfe
# ╟─af5f1e48-62c9-4d86-b553-d4674674e186
# ╟─a60b6236-7079-4e37-aa7a-8b795f8046d1
# ╟─f9f54997-9fc5-4a08-b76b-226757285c41
# ╠═496640ea-3831-4452-8653-08ae7f059e97
# ╟─9b1d317b-2615-4f87-a015-7d606e8dc5c5
# ╠═dbd43b55-6cb5-45e8-b7c5-2bdd5e8bb738
# ╟─03074a30-1589-41ae-83e8-23a149bc7a49
# ╠═f8eb7e0a-8b28-4c49-a98c-0f055c40ec4b
# ╟─1fa7deda-0bb5-40d6-aee8-a45ea8e4195b
# ╟─c27ffce9-09d7-49b3-addb-6272556305b3
# ╟─09238687-027f-4e19-8c93-f7fa52e04421
# ╠═d116615b-9428-41ea-8604-ef4fcdaf8005
# ╠═805bf2b0-c34c-4251-95ca-a9b0bb6c9efc
# ╠═aac677db-8170-4c45-aade-c6c588f3acea
# ╟─6739045b-b3e7-468b-8104-bcc4f96905cb
# ╟─22ecb9d8-ef57-4625-91b1-f0b6cea6b3a1
# ╟─6b2aa326-4c98-4df5-84b0-c52b8c2277af
# ╟─dc3cf268-f580-4f0f-a06b-9df873302ffa
# ╟─368b9212-efad-4689-8852-d2a0d5e307ca
# ╟─c8fa646c-e77a-4fc8-9039-78cbe6d5dc23
# ╟─e020ffc7-89c5-4335-bbd4-cdf89315bed0
# ╠═21e03a45-5a1e-46c0-99ab-11b31b72a2ad
# ╠═c1e78778-989e-4c1b-9838-5429099be143
# ╠═7b29bc39-8666-48fa-84b0-96c5caea283a
# ╟─5048ac99-e44b-4d24-a2a1-bdfc5e957d95
# ╟─b62ab732-6804-4216-b833-85c0f864109d
# ╠═27519334-a088-44e5-a6ff-69a7391c355a
# ╟─97389744-a10b-498c-935a-36e4a1a8d34d
# ╟─95beb5fc-4ce2-4f3d-bae9-d341a8279790
# ╠═58ffbdff-bf65-4346-989f-0734eb03a82b
# ╠═7da5d3f5-21db-48ae-8ac3-ceea81b10b60
# ╟─1199cda4-aba2-423e-bc7b-5258f4b9ee9b
# ╟─1bb17e59-f5b9-4a16-8b93-7960ab3b88ab
# ╟─81375947-0ff8-4ae9-9f9c-e340b6516424
# ╠═6a8c2c06-b6c7-4c7b-9119-11d3558e9af9
# ╟─c7b4f50d-0c83-43dc-8980-0a8af08dd050
# ╠═db6fef20-7a71-4fd7-86b7-26146aec089f
# ╠═6d984c89-acb1-49ff-b27c-c91b510c8714
# ╠═d7209426-dd4c-4cde-adbe-59636f9490bd
# ╠═417d1646-4fd1-4985-b367-a231a5bd2e1c
# ╟─f0a8db15-4f76-4a0e-8d41-1027f379b29f
# ╟─b41eb5db-be21-4f3c-b808-9885cebea42e
# ╠═e25a29bf-9a1a-489e-9892-07204c44ab3b
# ╟─10fa7add-a009-4d66-b4bc-9e7761f53c44
# ╟─edd4804c-5e9d-4677-a41e-3a5533567978
# ╠═b532e032-8fb2-4def-b6e1-0750c264c7c2
# ╠═f37df05b-6bc9-42b5-87d6-450f16236d09
# ╠═1f5d54f8-bceb-41a3-b620-d9098f9a8dbb
# ╟─8a9d80ba-713a-4102-85c6-7a3807738c40
# ╠═e9b4362b-cd35-4385-b9b9-4e155d2a1729
# ╟─a632cc79-ec51-41e2-846a-ee6a078ce093
# ╟─d4a41154-7154-4e78-8ae2-b0d6056ff5ce
# ╠═3254e3bd-5d9a-4448-97bd-45655c40c714
# ╠═751534bc-a2f4-4499-89c0-89e1046b9e6e
# ╠═b829503b-ecb9-4b25-aa9b-3c6871f9a21d
# ╟─3b094dca-a532-4c7e-bc64-815d8c656b3b
# ╟─b67252c7-2fad-4553-a248-c54ca492f638
# ╠═ebcd776d-2311-45b5-9826-38a2c5e266eb
# ╟─3354324a-86a0-4686-a430-5086617dd096
# ╟─f31ce7df-5e05-4cd2-b811-e0a40ba53d2b
# ╟─6e9afe84-1b3c-43f6-9227-f4200d4b366a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
