# This file was generated, do not modify it. # hide
using Symbolics

function get_map(f)
    # define variable
    @variables x
    # define derivative operator
    Dx = Differential(x)
    map = x - f(x) / Dx(f(x)) |> expand_derivatives
    # now we expand and compile to a Julia function
    update_expr = build_function(map, x)
    return eval(update_expr)
end

update = get_map(f)