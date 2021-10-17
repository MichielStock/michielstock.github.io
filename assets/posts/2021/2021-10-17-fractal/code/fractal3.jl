# This file was generated, do not modify it. # hide
# hideall
Z_f2 = applyiteratively.(2Z0, get_map(x->cosh(x) - 1); n=800)

heatmap(angle.(Z_f2), colorbar=false, color=:rainbow, ticks=false)

savefig(joinpath(@OUTPUT, "fractal3.png"));