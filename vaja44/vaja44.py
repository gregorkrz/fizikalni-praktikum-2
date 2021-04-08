from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import g
import pandas as pd
import numpy as np

data = pd.read_csv('data44.dat', sep=' ')
I_err, m_err = 1, 0.01
l, a, b = ufloat(2.2, 0.2) * 1e-2, ufloat(2, 0.1) * 1e-2, ufloat(9, 1) * 1e-3
S = a * b
d = data.copy()
d.columns = ['I [mA]', 'm [g]']

data['I'] = data['I'] * 1e-3
data['m'] = data['m'] * 1e-3
data['F'] = data['m'] * g
multiplier = 1e5
datalen = len(data['I'].values)
func = lambda x, k, n: k * x + n
popt, pcov = curve_fit(func, data['I'].values, data['F'].values)
k = ufloat(popt[0], np.sqrt(np.diag(pcov))[0])
_x = np.linspace(min(data['I'].values), max(data['I'].values), 3)
plt.plot(data['I'].values, data['F'].values * multiplier, ".")
plt.plot(_x, func(_x, popt[0], popt[1]) * multiplier, "--", color="gray")
plt.errorbar(data['I'].values, data['F'].values * multiplier,
            xerr=[I_err * 1e-3]*datalen, yerr=[m_err * g * 1e-3 * multiplier]*datalen,
            ecolor='black', elinewidth=3, fmt='none')
plt.xlabel('I [A]')
plt.ylabel(r'$F [10^{-5} N]$')
plt.show()

B = k / l
magflux = B * S