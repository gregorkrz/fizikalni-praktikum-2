import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from uncertainties import ufloat, unumpy
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

d = ufloat(0.51, 0.01) / 100
r = 0.5 * ufloat(19, 0.2) / 100
S = np.pi * r ** 2
c = 2.998e8

e0_ref = 1 / (c*c*4*np.pi*1e-7)

data = {
    200: [4.3, 4.5, 4.5],
    500: [10.16, 10.5, 10.7],
    800: [10.8, 10.8, 9.5],
    900: [13.5, 13.4, 13],
    1000: [14.7, 15.4, 15.9],
    1060: [18.8, 19.0, 18.9],
    860: [16, 15.9,  15.0, 15.6],
    320: [7.66, 9.1, 8.8],
    240: [9.4, 9.6, 9.6],
    700: [10.8, 11.6, 11.4],
}
# key: mass in mg; values: list of measured treshold voltages

f = []
u = []
multiplier = 100

for k in data.keys():
    for i in range(len(data[k])):
        f.append(k)
        u.append(data[k][i] * multiplier)

data = pd.DataFrame({'m [mg]': f, 'U [V]': u})

func = lambda x, k, n: k * x + n
X =  np.array(f) * g * 1e-6
Y = np.array(u)**2
p, pcov = curve_fit(func, X, Y)
k = ufloat(p[0], np.sqrt(np.diag(pcov))[0])
_x = np.linspace(min(X), max(X), 2)
fig, ax = plt.subplots()
ax.plot(_x, func(_x, p[0], p[1]), "--", color="gray")
ax.plot(X, Y, ".")
ax.set_xlabel(r'F [N]')
ax.set_ylabel(r'$U^2 [V^2]$')
#plt.show()

e0 = 2*d*d/S/k