# This file was generated, do not modify it. # hide
function applyiteratively(x, update; n=100)
    for i in 1:n
        x = update(x)
    end
    return x
end

applyiteratively(2.0, update)  # apply 100 times