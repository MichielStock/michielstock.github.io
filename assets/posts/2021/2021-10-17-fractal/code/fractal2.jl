# This file was generated, do not modify it. # hide
# hideall
Z_f2 = applyiteratively.(Z0, get_map(x->x^8+15x^4-16); n=500)

heatmap(angle.(Z_f2), colorbar=false, color=:rainbow, ticks=false)

savefig(joinpath(@OUTPUT, "fractal2.png"));