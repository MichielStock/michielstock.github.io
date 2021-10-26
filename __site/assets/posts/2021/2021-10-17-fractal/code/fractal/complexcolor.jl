# This file was generated, do not modify it. # hide
using Colors

complex2col(z) = HSL(rad2deg(angle(z)), 1.0, abs(z)) |> RGB;