# This file was generated, do not modify it. # hide
using Plots

heatmap(angle.(Z100), colorbar=false, color=:rainbow, ticks=false)

savefig(joinpath(@OUTPUT, "fractal.png")); # hide