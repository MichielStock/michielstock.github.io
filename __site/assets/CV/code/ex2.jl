# This file was generated, do not modify it. # hide
for i in 1:5, j in 1:5
    print(" ", rpad("*"^i,5), lpad("*"^(6-i),5), j==5 ? "\n" : " "^4)
end