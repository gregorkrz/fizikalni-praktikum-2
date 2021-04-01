import pandas as pd
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from scipy.optimize import curve_fit
import statsmodels.api as sm

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


data = pd.read_csv('data45.dat', delimiter=' ')
r, N = ufloat(0.12, 0.02) / 2, 3
ročica = ufloat(0.225, 0.003) / 2
R_h, N_h = 0.2, 154
F_err = 0.05e-3
data['F'] = data['F'] * 1e-3
data['I'] = data['I'] * 1e-3
data['I_h'] = data['I_h'] * 1e-3
data1 = data.copy()
data1.columns = ['F [N]', 'I [A]', 'I_H [A]']
multiplier = 1e3

def M(iih, mi0):
    res = (4/5)**1.5 * mi0 * N_h * N * iih[0] * iih[1] * np.pi * r**2 / R_h
    return unumpy.nominal_values(res)

navor = unumpy.uarray(data['F'].values, F_err) * ročica

x = np.linspace(min(data['I']), max(data['I']), 20)
y = np.linspace(min(data['I_h']), max(data['I_h']), 20)

p0 = 1e-6
popt, pcov = curve_fit(M, (data['I'].values, data['I_h'].values), unumpy.nominal_values(navor), p0,
                       sigma=unumpy.std_devs(navor), absolute_sigma=True)
mi0_nonlinear = ufloat(popt[0], np.sqrt(pcov[0]))

x, y = np.meshgrid(x, y)
z = M((x, y), mi0_nonlinear.n)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['I'].values, data['I_h'].values, unumpy.nominal_values(navor) * multiplier, color="black")
ax.plot_wireframe(x, y, unumpy.nominal_values(z) * multiplier)
ax.set_xlabel("$I [A]$")
ax.set_ylabel("$I_H [A]$")
ax.set_zlabel("$M [10^{-3}Nm]$")

const_I_h = 2.130
lindata = data[data['I_h'] == const_I_h]
linnavor = unumpy.uarray(lindata['F'].values, F_err) * ročica
popt, pcov = curve_fit(M, (lindata['I'].values, lindata['I_h'].values), unumpy.nominal_values(linnavor),
                       p0, sigma=unumpy.std_devs(linnavor), absolute_sigma=True)
mi0_linear = ufloat(popt[0], np.sqrt(pcov[0]))

figl, axl = plt.subplots()
axl.errorbar(lindata['I'].values, unumpy.nominal_values(linnavor) * multiplier, yerr=unumpy.std_devs(linnavor) * multiplier, ecolor="gray", fmt=".")
x = np.linspace(min(lindata['I'].values), max(lindata['I'].values), 5)
axl.plot(x, M((x, const_I_h), mi0_linear) * multiplier, '--')
axl.set_xlabel("$I [A]$")
axl.set_ylabel("$M [10^{-3}Nm]$")
axl.set_title("M(I) pri $I_H = {} A$".format(const_I_h))
'''
ols = sm.OLS(y, X)
ols_result = ols.fit()
# Now you have at your disposition several error estimates, e.g.
# and covariance estimates
ols_result.cov_HC0
'''