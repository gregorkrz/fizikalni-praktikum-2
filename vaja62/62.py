from uncertainties import ufloat

S = ufloat(7.7, 0.2)
P = ufloat(3.4, 0.2)
fob = ufloat(13.9, 0.3)
a = ufloat(18.5, 0.3)
b = ufloat(46.5, 0.5)

print(b/a, S/P)

#####

f1 = 0.51
f2 = 0.087
a = ufloat(12, 2)
q = ufloat(0.058, 0.005)
r = ufloat(1.5, 0.5) * 1e-3
N=4
x=0.02
M_izmerjena = (r/q)/ (N*x/a)
M = f1/f2 / (1-f1/a)
print(M, M_izmerjena)

####

fob = 0.087
fok = 0.058
d = ufloat(15.0, 1) * 1e-2
M_mikroskop = 0.25 * d / fob / fok
print(M_mikroskop)