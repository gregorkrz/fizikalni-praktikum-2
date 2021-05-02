import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.constants import c, e
from scipy.optimize import curve_fit
from uncertainties import ufloat, unumpy
import numpy as np

files = os.listdir('data')
current_multiplier = 1e-13
Ums = {}
for f in files:
    print(f)
    model = lambda x, k, n: k*x + n
    data = pd.read_csv(os.path.join('data', f), delimiter=' ')
    x = data['U'].values
    y = data['I'].values * current_multiplier
    lmbd = float(f.split('.')[0])
    plt.plot(x, y, ".", label='λ = {} nm'.format(int(lmbd)))
    popt, pcov = curve_fit(model, x, y)
    perr = np.sqrt(np.diag(pcov))
    k, n = ufloat(popt[0], perr[0]), ufloat(popt[1], perr[1])
    Um = - n / k
    Ums[lmbd] = Um
    _x = np.linspace(min(x), max(x), 2)
    plt.plot(_x, model(_x, k.n, n.n), "--", color='gray')
plt.legend()
plt.grid()
plt.xlabel('U [V]')
plt.ylabel(r'I [A]')
plt.show()

energije = []
frekvence = []

for l in Ums.keys():
    print('λ={}nm | Um={}'.format(int(l), Ums[l]))
    Wk = e * Ums[l]
    lmbd = l * 1e-9
    nu = c / lmbd
    energije.append(Wk)
    frekvence.append(nu)

model = lambda x, k, n: k*x + n
popt, pcov = curve_fit(model, frekvence, unumpy.nominal_values(energije))
_x = np.linspace(min(frekvence), max(frekvence))
plt.errorbar(frekvence, unumpy.nominal_values(energije), yerr=unumpy.std_devs(energije), ecolor="gray", fmt=".")
plt.plot(_x, model(_x, popt[0], popt[1]), "--", color='lightgray')
plt.xlabel(r'$\nu [Hz] $')
plt.ylabel(r'$W_k [J] $')
plt.grid()
plt.show()

errs = np.sqrt(np.diag(pcov))
Planck_const = ufloat(popt[0], errs[0])
A_i = -1*ufloat(popt[1], errs[1]) / e # A_i [eV]