import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from uncertainties import unumpy, ufloat
from scipy.optimize import curve_fit

data = pd.read_csv('data43.dat', delimiter=' ')
displaydata = data.copy()
displaydata.columns = ["R [Ω]", "U1 (pp) [V]", "U2 (pp) [mV]", "št. razdelkov"]

data['U1'] = data['U1'] / 2 # izmerjena je V(peak-to-peak)
data['U2'] = data['U2'] / 2 * 1e-3

cdata = np.array([ # umeritvena tabela kondenzatorja
    [20, 16],
    [30, 31],
    [40, 50],
    [50, 65],
    [60, 87],
    [70, 110],
    [80, 145],
    [90, 185],
    [120, 320],
    [130, 380],
    [140, 445],
    [160, 580],
    [180, 710],
    [200, 860]
])

x = cdata[:, 0]
y = cdata[:, 1]

count_to_pF = np.poly1d(np.polyfit(x, y, 3))

ufig, uax = plt.subplots()
uax.plot(x, y, ".", label='podatki iz umeritvene tabele')
_x = np.linspace(min(x), max(x), 1000) # umeritvena krivulja
uax.plot(_x, count_to_pF(_x), "--", label='polinom 3. reda')
uax.legend()
uax.set_xlabel('count (število razdelkov na kondenzatorju)')
uax.set_ylabel('C [pF]')
uax.grid(color='lightgray')
#ufig.show()

#C_0 = ufloat(count_to_pF(173), 10) * 1e-12
C = count_to_pF(data['count'].values) * 1e-12
y = (data['U2'] / data['U1']).values
r = data['R'].values
w = np.where(r == 0)
freq = 560e3

def resonancna(x, a, b):
    return a / np.sqrt((1-x)**2 + b**2 * x**2)
    
def reson_from_C(c, L, a, b):
    rat = freq * 2 * np.pi * np.sqrt(L * c)
    return resonancna(rat, a, b)

p0 = (0.0001, 0.01, 0.01)

popt, pcov = curve_fit(reson_from_C, C[w], y[w], p0)
L_0 = popt[0]

L_0_ufloat = ufloat(L_0, np.sqrt(np.diag(pcov))[0])
#_x = np.linspace(min(C[w]), max(C[w]), 1000)
#plt.plot(C[w], y[w], ".")
#plt.plot(_x, reson_from_C(_x, popt[0], popt[1], popt[2]))
#plt.show()
#L_0 = (1 / (C_0 * 4 * np.pi ** 2 * freq ** 2))


freq0 = 1 / (2 * np.pi * (L_0 * C) ** 0.5)
data['freq_ratio'] = freq / freq0

count_to_pF(data['count'].values)

x = unumpy.nominal_values(data['freq_ratio'].values)
y = (data['U2'] / data['U1']).values
r = data['R'].values
r_txt = np.array(data['R'].astype('str') + ' ohm')
#r_str = r.astype('str') + ' ohm'


ohms = np.sort(np.unique(r))

params = {}

fig2, ax2 = plt.subplots()

for o in ohms:
    w = np.where(r == o)
    _x = np.linspace(min(x[w]), max(x[w]), 3000)
    p0 = (0.02, 0.02)
    popt, pcov = curve_fit(resonancna, x[w], y[w], p0)
    model = lambda x: resonancna(x, popt[0], popt[1])
    params[o] = popt
    ax2.plot(_x, model(_x), '--', color='gray')


#plt.plot(unumpy.nominal_values(data['freq_ratio'].values), data['U2'] / data['U1'], ".")
ax2.errorbar(x, y, xerr=(5/560)*x, ecolor='black', elinewidth=0.5,  fmt='none')
ax2.grid(color='lightgray')
sns.scatterplot(x, y, hue=r, ax=ax2)
ax2.axes.get_legend().set_title(r'R[$\Omega$]')
ax2.set_xlim(left=0.8, right=1.15)
ax2.set_xlabel(r'$\nu / \nu_0$')
ax2.set_ylabel(r'$U/U_0$')
#fig2.show()