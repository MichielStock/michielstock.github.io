# This file was generated, do not modify it. # hide
lower = -2 - 2im
upper = 2 + 2im

step = 0.5e-2

# generate a range of complex values
Z0 = [a+b*im for b in real(lower):step:real(upper),
                    a in imag(lower):step:imag(upper)]

# apply the update 100 times
Z100 = applyiteratively.(Z0, update);