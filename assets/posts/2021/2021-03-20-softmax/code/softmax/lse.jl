# This file was generated, do not modify it. # hide
function logsumexp(x; γ=1)
	m = maximum(x)
	return m + γ * log(sum(exp.((x .- m) ./ γ)))
end