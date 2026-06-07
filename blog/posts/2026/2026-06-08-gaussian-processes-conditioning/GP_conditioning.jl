### A Pluto.jl notebook ###
# v0.20.20

#> [frontmatter]
#> title = "Condition Gaussian Processes on (nearly) everything"
#> 
#>     [[frontmatter.author]]
#>     name = "Michiel Stock"
#>     url = "https://michielstock.github.io/"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ b5981d14-5cdd-11f1-acfa-4f73362e0076
using Plots, PlutoUI, LinearAlgebra, Distributions, SpecialFunctions, SparseArrays

# ╔═╡ 0782886e-870e-4a15-92dc-2171b8655916
using ForwardDiff

# ╔═╡ 41b69c32-993b-4c78-b826-c264974e23b2
using BenchmarkTools

# ╔═╡ 11d2d8f4-c43d-4b85-9585-efe3943772a5
using Colors

# ╔═╡ 6630dfd7-d1f4-4e4c-8c79-7915097d9c71
md"""

# Condition Gaussian processes on nearly everything

[Michiel Stock](mailto:michielfmstock@gmail.com)

## Introduction

The Bayesian viewpoint is a highly instructive way to think about machine learning (and intelligence in general). A Gaussian process (GP) is perhaps one of the most elegant machine learning methods, providing a way to perform Bayesian inference over functions, and you can condition on how these functions should behave to obtain a posterior process. Powerful, it might be, but the mathematics works out nicely only in the linear/Gaussian regime, where the distributions are conjugate. For example, regression is easy, while classification (where the functions typically map to probabilities) is substantially harder, requiring more advanced methods such as the Laplace approximation, variational inference, or MCMC.

I recently came across a nice paper by Moss et al. (2026) that shows how to condition a GP on nearly any event that can be encoded as a PDF. For example, you can constrain the functions to be positive, monotone, or convex, or to satisfy a differential equation. They call their framework FLOWGP, as it is based on stochastic flow models. Their conditioning framework includes a method for sampling from a prior GP and conditioning on both the data D (traditional linear-Gaussian constraints) and the nonlinear, non-Gaussian likelihoods C. 

Let us explore here how and whether this works.

"""

# ╔═╡ 40c167e8-39c7-41b9-a48f-0f119f835f3f
md"""

## GP crash course

A GP is rather simple: it is a stochastic process in which any finite set of point evaluations follows a multivariate normal (MVN) distribution. Practically, one can define a GP prior distribution by specifying a mean function $m(x)$[^1] and a positive definite covariance or kernel function $k(x,x')$, which determines the correlation between the points (or the variance via $k(x,x)$).

For example, we define a simple radial basis kernel for smooth, nonlinear functions. 
"""

# ╔═╡ fd1fefc7-90e4-4013-8905-f510928607f2
k(x, y; γ=1.0, degree=2, A=1) = A*exp(-norm(x .- y)^degree/γ^2)

# ╔═╡ ecb50b98-001a-4b79-913f-6234466bba3d
xsteps = 0:0.05:10

# ╔═╡ d35e31c3-ba4f-4406-a3df-0db91e86b046
n = length(xsteps)

# ╔═╡ fe1c2dff-fa63-4c60-b6b6-60b1d5725b79
K = [k(x, y, A=1) for x in xsteps, y in xsteps] + 1e-10I

# ╔═╡ 6e924c4e-884b-4d89-8f41-c9ecb296d4f8
m = zeros(n)

# ╔═╡ fe7a3337-1723-4a00-b152-535b6596edb2
prior_GP = MultivariateNormal(m, K)

# ╔═╡ e256ef46-1812-40eb-ae95-7c70fd1b6dab
function condition(prior_dist, y, L, Λ, b=0 )
	μ = prior_dist.μ
	Σ = prior_dist.Σ
	Σ̃ = Symmetric(Σ - Σ * L' * (inv(L * Σ * L' + Λ) * L ) * Σ)
	μ̃ = Σ̃ * (L' * inv(Λ) * (y .- b) + inv(Σ) *  μ)
	return MvNormal(μ̃, Σ̃)
end

# ╔═╡ a5027ced-5125-4e9c-977f-b99ed3069cf7
target(x) = 0.2x * sin(2x) + 1.2

# ╔═╡ 57d91f84-7640-4820-a3e6-f8c719a0c523
Isample = [50:10:100..., 190]

# ╔═╡ 1f679fae-f4b0-4cbf-8584-31624cf90ea2
xobs = xsteps[Isample]

# ╔═╡ e9260678-e3a3-4724-9d60-5171434ad643
nobs = length(xobs)

# ╔═╡ fccc611a-ec05-4d34-981f-7f025caf5f8a
σ = 0.1

# ╔═╡ 2a6b8e8e-1253-4bc0-9c37-2472f09984ee
yobs = target.(xobs) + σ .* randn(nobs)

# ╔═╡ 185985e9-ba67-41a7-87fe-3994cea056c1
begin
	L = spzeros(Bool, nobs, n)  # L encodes which elements of f are observed
	for (i, j) in zip(1:nobs, Isample)
		L[i, j]=true
	end
end

# ╔═╡ f29f9dd7-ab2c-4fb6-a668-c558c4e929fb
L

# ╔═╡ 9321df8c-fb3a-492c-94e2-f4d7465ac725
Λ = σ^2 * I

# ╔═╡ a93f902a-309f-4f50-9929-0cff5e7ede86
Σ̃ = K - K * L' * ((L * K * L' + Λ) \ L ) * K

# ╔═╡ 63a94b91-4cb3-4155-8f77-f2680e9fbcbb
μ̃ = Σ̃ * (L' * (Λ \ (yobs .- 0)) + K \ m)

# ╔═╡ f070960f-7e81-4e23-aeff-f8ce83bd0a3e
post_GP = condition(prior_GP, yobs, L, Λ)

# ╔═╡ c6df248e-8ae2-4f70-bcbe-0dfc85525328
md"""
## Gaussian processes as diffusion models

odels

In the paper, the authors frame a GP sample as a [diffusion model](https://en.wikipedia.org/wiki/Diffusion_model). The forward process assumes a sample ${f}_0$ and keeps adding noise to it to obtain ${f}_1\sim N(0, I)$. Diffusion models learn the inverse map, where one starts from a noise sample $\mathbf{z}\sim N(0, I)$ and one adds model-guided noise to obtain a sample of the target distribution (e.g., an image of an [avocado armchair](https://www.technologyreview.com/2021/01/05/1015754/avocado-armchair-future-ai-openai-deep-learning-nlp-gpt3-computer-vision-common-sense/)).

For Gaussian processes, the denoising process can be seen as simulating a stochastic differential equation, which can be generated by numerically integrating the following ODE, departing from a random sample:

$$\frac{df_t}{dt} = -\tfrac{1}{2}\beta(t)\big(f_t + s(f_t,t)\big)$$


Here, $\beta(t)$ is related to the process of how noise is added. For a vanilla GP, the guidance function $s(f_t,t)$ is fully determined by the GP. One can actually analytically solve the ODE. Things become more interesting when trying to guide with a non-Gaussian PDF $p(C\mid {f})$, for which the guide is the gradient of the log-likelihood:

$$s(f_t,t) = \nabla_{f_t}\log p(C\mid f_t, D)\,.$$

Two tricks the authors develop to make this work:

* This gradient is, in general, not possible to evaluate, as we need to condition also on D. To this end, the authors take some Monte Carlo estimates (five suffice, but it works with even a single one) of the GP and create a self-normalised importance estimator.
* Using diffusion for $p(D\mid f_0)$ and $p(C\mid f_0)$ makes the problem stiff. Before denoising, the authors whiten the sample, i.e., remove the GP posterior signal. The diffusion is done in whitened space, and at the end, they transform back. This greatly improves the structure of the denoising problem, making it rather insensitive to the specific hyperparameter settings (which I could confirm experimentally)!

"""

# ╔═╡ d9b6a2c4-ee18-46d9-941f-e7d273e6e730
z = randn(n)

# ╔═╡ 585f4824-e277-4071-9ca9-5a1ffbbb86e2
m_cond_y = post_GP.μ 

# ╔═╡ 3768fc4d-1c54-4731-8ee3-0aad86335685
K_cond_y = post_GP.Σ

# ╔═╡ 753d08a2-cdac-4876-a05b-96a7a69587e8
@bind t Slider(0:0.05:1, show_value=true)

# ╔═╡ 8c172f1d-c05e-4c8b-9c35-b22d18341fbe
md"""

## Conditioning using guidance

"""

# ╔═╡ 79480e1c-49a7-4afe-be3b-3211cd7e61c6
# ensures positivity of the functions
log_p_C(f₀; ν=1e-3) = sum(logerfc.(-(f₀ ./ ν) ./ sqrt(2)))

# ╔═╡ d71f84d7-8075-44ac-9ede-cb7d7d82a743
α(t; β₀=1e-5, β₁=10.0) = exp(-0.5 * β₀ * t - 0.25 * (β₁ - β₀) * t^2)

# ╔═╡ 1173dd83-7f4c-4554-ad19-af89b421cbcd
fₜ(t) = t < 1 ? α(t) * m_cond_y   + sqrt(α(t)^2 * K_cond_y + (1-α(t)^2) * I) * z : z

# ╔═╡ e9178326-e7c0-495c-b708-53c5580a78bb
begin
	plot(xsteps, fₜ(t))
	scatter!(xobs, yobs, label="datapoints")
	title!("GP diffusion t=$t")
end

# ╔═╡ 720aefbb-a42b-4d29-b1ba-794a6cf068d6
β(t; β₀=1e-5, β₁=10.0) = β₀ + (β₁ - β₀) * t

# ╔═╡ d9739f24-1113-424c-a005-9947a3982e17
function smooth_clip!(v; v_max = 100.0)
	v_norm = norm(v)
	
	v .*= v_max * tanh(v_norm / v_max) / (v_norm + 1e-8)
	return v
end

# ╔═╡ 7b8c5e1c-b5f7-45e3-9bac-36fb8e24d892
function logsumexp(x; γ=1)
    c = maximum(x)
    return c + γ * log(sum(exp.((x .- c)/γ)))
end

# ╔═╡ a83c4015-9686-4710-b08d-3529ed7a5b81
"""
	sample_conditional(m, K, log_p_C; kwargs)

Conditions a sample of a GP with mean `m` and covariance `K`
according to log-likelihood constraint `log_p_C`.

"""
function sample_conditional(m, K, log_p_C;
			S=5,                 # number of Monte Carlo samples for gradient
			n_euler=100,         # number of Euler steps
			jitter=1e-10,        # jitter for Cholesky decomposition
			v_max=100)           # vmax for gradient clipping
	n = length(m)
	# random perturbations
	Z = randn(n, S)  # share always the same noise for each step
	f̂ = randn(n)
	F_samples = similar(Z)
	# stuff for whitening
	C = cholesky(Symmetric(K + jitter*I))
	L = C.L 
	# define log_p_C and its gradient on whitened vector 
	log_p_C_whitened = f̂ -> log_p_C(L * f̂ .+ m)
	∇log_p_C_whitened = f̂ -> ForwardDiff.gradient(log_p_C_whitened, f̂)
	s = zeros(n)  # gradient for conditioning
	# euler steps
	tsteps = range(1, 0, length=n_euler)
	for t in tsteps
		αₜ = α(t)  # drift coefficient
		σₜ = √(1 - αₜ^2)
		# sample in whitened space
		F_samples .= αₜ .* f̂ .+ σₜ .* Z
		# compute gradient using MC
		ℓ = [log_p_C_whitened(F_samples[:,i]) for i in 1:S]
		# turn in weights (log-sum-exp trick)
		w = exp.(ℓ .- logsumexp(ℓ))
		s .= 0.0
		for (i, wᵢ) in enumerate(w)
			s .+= ∇log_p_C_whitened(F_samples[:,i]) * w[i]
		end
		s .*= - β(t) * αₜ / 2
		smooth_clip!(s; v_max)   # soft clip the gradients
		# update
		f̂ .+= s * step(tsteps)
	end
	# unwhiten
	f̂ .= L * f̂ .+ m
	return f̂	
end

# ╔═╡ 7eaef8c6-8649-4324-b99b-0a131cb7f7d8
f̂ = sample_conditional(m_cond_y, K_cond_y, log_p_C)

# ╔═╡ 70b0e631-4b10-4ca3-bd2d-d03b348de65f
@benchmark sample_conditional(m_cond_y, K_cond_y, log_p_C)

# ╔═╡ dfc422d5-a997-4fd0-97fe-67b578e4339f
md"""
## Easy GP classification

For this toy example, you can see that we can sample from the GP prior, where $f$ is a probit of the class probabilities. Such a Bayesian classifier is very useful, for example, in Bayesian optimisation with binary labels (e.g., designing proteins that are active or not), where one has to separate uncertainty between aleatoric (due to irreducible noise and fuzzy class boundaries) and epistemic (due to an incomplete model) uncertainty (Fauvel & Chalk, 2021).


"""

# ╔═╡ 068fad6f-193e-4c8c-b90a-9ea7a814eb01
xsteps

# ╔═╡ 1ddfc62c-73fd-4332-8d62-f709445252e9
clf_data = [
	(5, true),
	(7, true),
	(8, true),
	(11, true),
	(15, true),
	(76, true),
	(78, false),
	(79, false),
	(81, true),
	(82, false),
	(84, true),
	(169, false),
	(181, false),
	(183, false),
	(185, false),
	(187, false),
]

# ╔═╡ b52c6485-aac3-480d-848c-f02daec5171e
const Iclf, targetclf = collect.(zip(clf_data...))

# ╔═╡ 49f350f2-c12e-4b13-9b27-ecce1f9aa79a
sigmoid(f) = 1 / (1 + exp(-f))

# ╔═╡ 83fd010b-a59f-48b2-8719-c5d1af6c27f5
# cross-entropy on σ(f)
function log_C_clf(f)
	σ = fi -> 1 / (1 + exp(-fi))
	return sum(ti ? log(σ(f[i])) : log(1-σ(f[i])) for (i, ti) in zip(Iclf, targetclf))
end

# ╔═╡ 1bd3792a-03a7-4eb6-983a-cb458083af74
f̂[Iclf]

# ╔═╡ 8dfc97d1-5e2a-46a1-85d1-bcdb787f1f2f
Kclf = [k(x, y, A=2, γ=3) for x in xsteps, y in xsteps] + 1e-12I

# ╔═╡ 01603557-9f98-436b-823f-e5397e5119fa
f̂clf = sample_conditional(m, Kclf, log_C_clf)

# ╔═╡ 79a51fef-939f-49fb-9770-e416c65d8a12
md"""
## Other guidance functions
"""

# ╔═╡ f1b0a21f-253b-4f3d-a8c7-7915951a0d53
function log_p_convex(f; ν=1e-3)
    # Second differences: f[i+1] - 2f[i] + f[i-1] ≥ 0 for convexity
    Δ² = f[1:end-2] .- 2 .* f[2:end-1] .+ f[3:end]
    return sum(logerfc.(.-Δ² ./ (ν * sqrt(2))))
end

# ╔═╡ 71358683-16b9-49d4-bf83-2253b6c478a1
md"""
## Conclusions and potential

I enjoyed reading this paper and playing around with the ideas. The mathematical foundation is elegant and broad in scope, and it is surprisingly easy to get working in practice with very little tuning. My code could work for any constraint, thanks to autodiff, though it will need some speedup to scale to larger problems.

This work greatly extends/simplifies Bayesian and probabilistic modelling using Gaussian processes, without resorting to specific tricks such as variational inference. Constraints, physical laws, mechanistic structure, i.e., anything you can write down as a scoring function $\log p(C \mid f)$. It composes multiplicatively without needing specific inference schemes. It allows for moving GP beyond the world of the Gaussians.

What the authors briefly explore (and is almost certainly grounds for future research) is conditioning on LLM outputs, using them to encode inprecise or hard-to-quantify information. For example, it would be worthwhile in domains such as protein engineering with sparse experimental activity data and prior knowledge incorporated in foundation models. Here, a model such as ESMFold can serve as a prior for biological plausibility, which can be combined with experimental data in a Bayesian framework.

"""

# ╔═╡ e4d721eb-c1e3-4a52-a18f-9e28ebfea07a
md"## Helper functions"

# ╔═╡ fdf181b1-70a5-4f27-8ed1-627bb6c6f4b4
plots = Dict()  # for storing all my pretty plots

# ╔═╡ dd14c455-5449-4912-a6f6-9c236a6fecd8
plots["cov_prior"] = heatmap(xsteps, xsteps, K, title="Cov. of the prior process", xlab="x", ylab="x")

# ╔═╡ 18b07106-f391-4aae-9fcc-259fdf9641c3
begin
	plots["target"] = plot(xsteps, target, label="target function")
	scatter!(xobs, yobs, label="datapoints")
	xlabel!("x")
	title!("Target function + samples")
end

# ╔═╡ 73a3e72e-80eb-4563-8e09-878837ac6b12
plots["cov_posterior"] = heatmap(xsteps, xsteps, Σ̃, title="Cov. of the posterior process", xlab="x", ylab="x")

# ╔═╡ f896bc10-163a-4012-9a97-f3b8ff7a74fe
plots["mean_posterior"] = plot(xsteps, μ̃, xlab="x", title="Posterior mean", label="μ̃", lw=2)

# ╔═╡ f1f49949-f921-4180-aef6-de21c68ba9ac
plots["noise"] = plot(xsteps,z, xlab="x", label="Isotropic normal")

# ╔═╡ e78b000c-eb4d-4a0d-8e69-befcc827f646
plots["postive_constraint"] = plot(log_p_C, -4, 4, title="Likelihood positive", xlab="x", label="log p(f|C)")

# ╔═╡ fdfe7d5a-b06c-4c15-8e07-7f904aa0cf13
plots["alpha"] = plot(α, 0, 1, xlab="t", label="α(t)")

# ╔═╡ 1db7da65-ec26-467a-a944-2174d4bea666
plots["beta"] = plot(β, 0, 1, xlab="t", label="β(t)")

# ╔═╡ b4b06441-6155-43eb-bb29-829435579085
begin
	plots["prior_samples_convex"] = plot()
	for i in 1:5
		f̂ = sample_conditional(zeros(n), K, log_p_convex, S=5, n_euler=200)
		plot!(xsteps, f̂, alpha=0.7, label="")
	end
	#scatter!(xobs, yobs, label="datapoints")
	title!("Five prior samples | convex")
end

# ╔═╡ f6cddf5c-42f7-4778-9868-f677829064d9
# Catppuccin Latte (light)
const LATTE = (
    base      = colorant"#eff1f5",
    mantle    = colorant"#e6e9ef",
    crust     = colorant"#dce0e8",
    text      = colorant"#4c4f69",
    subtext   = colorant"#6c6f85",
    surface   = colorant"#ccd0da",
    overlay   = colorant"#9ca0b0",
    rosewater = colorant"#dc8a78",
    flamingo  = colorant"#dd7878",
    pink      = colorant"#ea76cb",
    mauve     = colorant"#8839ef",
    red       = colorant"#d20f39",
    maroon    = colorant"#e64553",
    peach     = colorant"#fe640b",
    yellow    = colorant"#df8e1d",
    green     = colorant"#40a02b",
    teal      = colorant"#179299",
    sky       = colorant"#04a5e5",
    sapphire  = colorant"#209fb5",
    blue      = colorant"#1e66f5",
    lavender  = colorant"#7287fd",
)

# ╔═╡ 6630d47c-46e5-4fb7-af7d-ac2b9bafce37
const C = LATTE  # or MOCHA, FRAPPE, MACCHIATO

# ╔═╡ 6a1f0b5a-7202-41de-9b2d-a044e8b8ada2
plots["samples_prior"] = plot(xsteps, rand(prior_GP, 10), label="", lw=2,
							  alpha=0.7, title="Prior samples", color=C.green)

# ╔═╡ cd6ede2b-a3bd-4edd-bf24-4d2f8d32269f
begin
	plots["sample_posterior"] = plot(xsteps, rand(post_GP, 5), label="posterior sample", lw=2)
	scatter!(xobs, yobs, label="datapoints", color=C.maroon)
	title!("Five posterior samples")
end

# ╔═╡ e771a528-726f-4fb5-a8fe-10b883f0a16f
let
	tvals = range(0, 1, length = 3)
	plots["gp_diffusion"] = plot(xsteps, fₜ.(tvals), 
								palette=cgrad(:roma, length(tvals)),
								 label=reshape(["t=$t" for t in tvals],1,:),
								 lw=2, alpha=0.7)
	scatter!(xobs, yobs, label="datapoints", color=C.red)
	xlabel!("x")
	title!("Diffusion from noise to posterior")
end

# ╔═╡ c2410b33-82be-41f2-a91a-59378b3550d8
begin
	plots["posterior_constraint_samples"] = plot()
	for i in 1:10
		f̂ = sample_conditional(m_cond_y, K_cond_y, log_p_C, S=5, n_euler=100)
		plot!(xsteps, f̂, alpha=0.7, label= i==1 ? "samples GP | D, f >0" : "",
			  color=C.lavender)
	end
	scatter!(xobs, yobs, label="datapoints", color=C.maroon)
	title!("Five posterior samples | non-negative")
end

# ╔═╡ 3d5e50e5-71a6-4025-8542-9f48a79d2437
plots["clf_data"] = scatter(xsteps[Iclf], targetclf, color=[t ? C.maroon : C.lavender for t in targetclf], label="label")

# ╔═╡ aa20902d-3be1-4b7f-b46a-edd68790502a
let
	n_samples = 10
	P = zeros(length(f̂clf), n_samples)
	plots["clf_post"] = scatter(xsteps[Iclf], targetclf, color=[t ? C.maroon : C.lavender for t in targetclf], label="label")
	
	for i in 1:n_samples
		f̂clf = sample_conditional(m, Kclf, log_C_clf)
		
		p = sigmoid.(f̂clf)
		P[:,i] .= p
		plot!(xsteps, p, label=i==1 ? "p(true|x, D) sample" : "", alpha=0.4, color=C.lavender)
	end
	plot!(xsteps, median(P, dims=2), label="median of p(true|x, D)")
	plot!(xsteps, std(P, dims=2), label="std of p(true|x, D)")
	title!("Posterior GP classification")
	plots["clf_post"]
end

# ╔═╡ 3d96b98c-383d-4d93-b64f-f6a23a888590
catppuccin_cycle = [C.blue, C.peach, C.green, C.mauve, C.red, 
                    C.teal, C.yellow, C.pink, C.sapphire, C.maroon]

# ╔═╡ 4047f582-c10a-4080-b6fc-679a552e9251
default(
    palette = catppuccin_cycle,
    background_color = C.base,
    foreground_color = C.text,
    foreground_color_axis = C.subtext,
    foreground_color_text = C.text,
    foreground_color_border = C.overlay,
    foreground_color_legend = C.text,
    gridcolor = C.surface,
	lw=2,
	xlab="x",
)

# ╔═╡ 3202c7de-73a5-4534-ae9d-666a083a3b07
const STDNORMAL = Normal()

# ╔═╡ 17428a45-44b1-4ed0-bb69-4d0af9113464
function log_p_monotone(f; ν=1e-3)
    Δ = diff(f)                     # discrete derivative
    return sum(logcdf(STDNORMAL, Δ ./ ν))
end

# ╔═╡ be915f6a-5e40-44e2-8625-d1fd0770c181
fmono = sample_conditional(zeros(n), K, log_p_monotone)

# ╔═╡ bfc4b560-442d-488f-90bc-7071679f9aca
begin
	plots["prior_samples_monotone"] = plot()
	for i in 1:5
		f̂ = sample_conditional(zeros(n), K, log_p_monotone, S=5, n_euler=100)
		plot!(xsteps, f̂, alpha=0.7, label="")
	end
	#scatter!(xobs, yobs, label="datapoints")
	title!("Five prior samples | monotone increasing")
end

# ╔═╡ 6816034a-8f65-44ed-9383-afdac65a984c
log_p_monotone_convex(t) = log_p_convex(t) + log_p_monotone(t)

# ╔═╡ 3bf7f1aa-2b9e-4f1a-b120-70f2e69f450b
begin
	plots["prior_samples_convex_montone"] = plot()
	for i in 1:5
		f̂ = sample_conditional(zeros(n), K, log_p_monotone_convex, S=5, n_euler=500)
		plot!(xsteps, f̂, alpha=0.7)
	end
	#scatter!(xobs, yobs, label="datapoints")
	title!("Five prior samples | convex & monotone")
end

# ╔═╡ 00c79842-aa95-4660-98a4-f55ef06738b9
TableOfContents()

# ╔═╡ abc3c32c-1696-43f6-a61c-085dfcc25f54
plots

# ╔═╡ b827cb92-2f33-48bb-a02f-feac3627cd24
begin
	for (name, plt) in plots
		savefig(plt, joinpath("figures", name *".svg"))
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[compat]
BenchmarkTools = "~1.8.0"
Colors = "~0.13.1"
Distributions = "~0.25.123"
ForwardDiff = "~1.3.2"
Plots = "~1.41.6"
PlutoUI = "~0.7.80"
SpecialFunctions = "~2.7.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "e645bfefdcc2805a068373b21a1e953096e42b43"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "PrecompileTools", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "9670d3febc2b6da60a0ae57846ba74670290653f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.8.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a21c5464519504e41e0cbc91f0188e8ca23d7440"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "fbcc7610f6d8348428f722ecbe0e6cfe22e672c6"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.123"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "66381d7059b5f3f6162f28831854008040a4e905"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.1+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "eef4c86803f47dcb61e9b8790ecaa96956fdd8ae"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.2"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "b7bfd56fa66616138dfe5237da4dc13bbd83c67f"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ee0585b62671ce88e48d3409733230b401c9775c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.22"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "7dd7173f7129a1b6f84e0f03e0890cd1189b0659"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.22+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "97bbca976196f2a1eb9607131cb108c69ec3f8a6"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d0205286d9eceadc518742860bf23f703779a3d6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.3+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0662b083e11420952f2e62e17eddae7fc07d5997"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "cb20a4eacda080e517e4deb9cfb6c7c518131265"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.6"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "fbc875044d82c113a9dee6fc14e16cf01fd48872"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.80"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "34f7e5d2861083ec7596af8b8c092531facf2192"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "8f528b0851b5b7025032818eb5abbeb8a736f853"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "fa95b3b097bcef5845c142ea2e085f1b2591e92c"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.7.1"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9cce64c0fdd1960b597ba7ecda2950b5ed957438"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.2+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4909eb8f1cbf6bd4b1c30dd18b2ead9019ef2fad"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.18.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e015f211ebb898c8180887012b938f3851e719ac"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.55+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╠═b5981d14-5cdd-11f1-acfa-4f73362e0076
# ╟─6630dfd7-d1f4-4e4c-8c79-7915097d9c71
# ╟─40c167e8-39c7-41b9-a48f-0f119f835f3f
# ╠═fd1fefc7-90e4-4013-8905-f510928607f2
# ╠═ecb50b98-001a-4b79-913f-6234466bba3d
# ╠═d35e31c3-ba4f-4406-a3df-0db91e86b046
# ╠═fe1c2dff-fa63-4c60-b6b6-60b1d5725b79
# ╟─dd14c455-5449-4912-a6f6-9c236a6fecd8
# ╠═6e924c4e-884b-4d89-8f41-c9ecb296d4f8
# ╠═fe7a3337-1723-4a00-b152-535b6596edb2
# ╟─6a1f0b5a-7202-41de-9b2d-a044e8b8ada2
# ╠═e256ef46-1812-40eb-ae95-7c70fd1b6dab
# ╠═a5027ced-5125-4e9c-977f-b99ed3069cf7
# ╠═57d91f84-7640-4820-a3e6-f8c719a0c523
# ╠═1f679fae-f4b0-4cbf-8584-31624cf90ea2
# ╠═e9260678-e3a3-4724-9d60-5171434ad643
# ╠═fccc611a-ec05-4d34-981f-7f025caf5f8a
# ╠═2a6b8e8e-1253-4bc0-9c37-2472f09984ee
# ╟─18b07106-f391-4aae-9fcc-259fdf9641c3
# ╠═185985e9-ba67-41a7-87fe-3994cea056c1
# ╠═f29f9dd7-ab2c-4fb6-a668-c558c4e929fb
# ╠═9321df8c-fb3a-492c-94e2-f4d7465ac725
# ╠═a93f902a-309f-4f50-9929-0cff5e7ede86
# ╠═73a3e72e-80eb-4563-8e09-878837ac6b12
# ╠═63a94b91-4cb3-4155-8f77-f2680e9fbcbb
# ╠═f896bc10-163a-4012-9a97-f3b8ff7a74fe
# ╠═f070960f-7e81-4e23-aeff-f8ce83bd0a3e
# ╠═cd6ede2b-a3bd-4edd-bf24-4d2f8d32269f
# ╟─c6df248e-8ae2-4f70-bcbe-0dfc85525328
# ╠═d9b6a2c4-ee18-46d9-941f-e7d273e6e730
# ╠═f1f49949-f921-4180-aef6-de21c68ba9ac
# ╠═585f4824-e277-4071-9ca9-5a1ffbbb86e2
# ╠═3768fc4d-1c54-4731-8ee3-0aad86335685
# ╠═753d08a2-cdac-4876-a05b-96a7a69587e8
# ╠═1173dd83-7f4c-4554-ad19-af89b421cbcd
# ╠═e9178326-e7c0-495c-b708-53c5580a78bb
# ╠═e771a528-726f-4fb5-a8fe-10b883f0a16f
# ╠═8c172f1d-c05e-4c8b-9c35-b22d18341fbe
# ╠═79480e1c-49a7-4afe-be3b-3211cd7e61c6
# ╠═e78b000c-eb4d-4a0d-8e69-befcc827f646
# ╠═d71f84d7-8075-44ac-9ede-cb7d7d82a743
# ╠═720aefbb-a42b-4d29-b1ba-794a6cf068d6
# ╠═fdfe7d5a-b06c-4c15-8e07-7f904aa0cf13
# ╠═1db7da65-ec26-467a-a944-2174d4bea666
# ╠═0782886e-870e-4a15-92dc-2171b8655916
# ╠═d9739f24-1113-424c-a005-9947a3982e17
# ╠═7b8c5e1c-b5f7-45e3-9bac-36fb8e24d892
# ╠═a83c4015-9686-4710-b08d-3529ed7a5b81
# ╠═7eaef8c6-8649-4324-b99b-0a131cb7f7d8
# ╠═41b69c32-993b-4c78-b826-c264974e23b2
# ╠═70b0e631-4b10-4ca3-bd2d-d03b348de65f
# ╠═c2410b33-82be-41f2-a91a-59378b3550d8
# ╟─dfc422d5-a997-4fd0-97fe-67b578e4339f
# ╠═068fad6f-193e-4c8c-b90a-9ea7a814eb01
# ╠═1ddfc62c-73fd-4332-8d62-f709445252e9
# ╠═b52c6485-aac3-480d-848c-f02daec5171e
# ╠═3d5e50e5-71a6-4025-8542-9f48a79d2437
# ╠═49f350f2-c12e-4b13-9b27-ecce1f9aa79a
# ╠═83fd010b-a59f-48b2-8719-c5d1af6c27f5
# ╠═1bd3792a-03a7-4eb6-983a-cb458083af74
# ╠═8dfc97d1-5e2a-46a1-85d1-bcdb787f1f2f
# ╠═01603557-9f98-436b-823f-e5397e5119fa
# ╟─aa20902d-3be1-4b7f-b46a-edd68790502a
# ╠═79a51fef-939f-49fb-9770-e416c65d8a12
# ╠═17428a45-44b1-4ed0-bb69-4d0af9113464
# ╠═be915f6a-5e40-44e2-8625-d1fd0770c181
# ╟─bfc4b560-442d-488f-90bc-7071679f9aca
# ╠═f1b0a21f-253b-4f3d-a8c7-7915951a0d53
# ╟─b4b06441-6155-43eb-bb29-829435579085
# ╠═6816034a-8f65-44ed-9383-afdac65a984c
# ╟─3bf7f1aa-2b9e-4f1a-b120-70f2e69f450b
# ╟─71358683-16b9-49d4-bf83-2253b6c478a1
# ╟─e4d721eb-c1e3-4a52-a18f-9e28ebfea07a
# ╠═fdf181b1-70a5-4f27-8ed1-627bb6c6f4b4
# ╠═11d2d8f4-c43d-4b85-9585-efe3943772a5
# ╠═f6cddf5c-42f7-4778-9868-f677829064d9
# ╠═6630d47c-46e5-4fb7-af7d-ac2b9bafce37
# ╠═3d96b98c-383d-4d93-b64f-f6a23a888590
# ╠═4047f582-c10a-4080-b6fc-679a552e9251
# ╠═3202c7de-73a5-4534-ae9d-666a083a3b07
# ╠═00c79842-aa95-4660-98a4-f55ef06738b9
# ╠═abc3c32c-1696-43f6-a61c-085dfcc25f54
# ╠═b827cb92-2f33-48bb-a02f-feac3627cd24
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
