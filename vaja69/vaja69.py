import pandas as pd
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

## 1. del
d1 = pd.read_csv('data1.txt', header=None, sep='\t', decimal=',')
t, cnt = d1[0].values, d1[1].values
avg = np.mean(cnt)
stp = np.sqrt(avg)
std = np.std(cnt)

d1.columns='t[min]','N'
plt.plot(t, cnt)
plt.xlabel('t[min]')
plt.ylabel('N')
plt.show()


## 2. del
d2 = {}
lines = open('data2.txt', 'r').readlines()
n = 0
for l in lines:
    if l.startswith('n='):
        n = int(l.split('=')[1])
        d2[n] = []
    else:
        s = l.split('\t')
        d2[n].append([float(s[0]), float(s[1])])

avg = []
ozadje = 0
for n in d2.keys():
    df = pd.DataFrame(d2[n])
    n_display = str(n) + ' ploščic'
    if n < 0: n_display = 'ozadje'
    print('Meritve N po minutah za', n_display, ':', str(list(df[1].values)))
    av, st =  np.average(df[1].values), np.std(df[1].values)
    if n < 0:
        ozadje = (av, st, np.sqrt(av))
    else:
        avg.append([n, av, st, np.sqrt(av)])

avg = np.array(avg)
ploscica = ufloat(2, 0.6) * 1e-3 # debelina ploščice
d = avg[:, 0] * ploscica
background = ufloat(ozadje[0], ozadje[1]) # sevanje ozadja
phi = unumpy.uarray(avg[:, 1], avg[:, 2]) - background
plt.errorbar(unumpy.nominal_values(d), unumpy.nominal_values(phi),
    xerr=unumpy.std_devs(d), yerr=unumpy.std_devs(phi), ecolor="gray", fmt=".", label='meritve')
plt.yscale('log')
phi_model = lambda x, mu, phi0: phi0 * np.exp(-mu*x)
popt, pcov = curve_fit(phi_model, unumpy.nominal_values(d), unumpy.nominal_values(phi))
_x = np.linspace(min(unumpy.nominal_values(d)), max(unumpy.nominal_values(d)), 1000)
_y = phi_model(_x, popt[0], popt[1])

mu = ufloat(popt[0], np.sqrt(np.diag(pcov))[0])

plt.plot(_x, _y, "--", label=r'model $\phi = \phi_0 e^{-\mu d}$')
plt.xlabel(r'$d [m]$')
plt.ylabel(r'$\phi [min^{-1}]$')
plt.legend()
plt.show()

#plt.clf()
## 3. del
values = []
files = ['data31.txt', 'data33.txt', 'data34.txt']
for f in files:
    data = pd.read_csv(f, sep='\t', decimal=',', header=None)
    vals = data[2].values[np.where(data[2].values != ' ')]
    vals = np.array([i.replace(',', '.') for i in vals]).astype(np.float64)
    values.append(vals)
v = np.concatenate(values)
mu1 = 1/np.mean(v)
plt.hist(v, density=True, bins=140)
_x = np.linspace(min(v), max(v), 10000)
_y = mu1 * np.exp(-mu1*_x)
plt.text(0, 10, r'$\mu={}$'.format(round(mu1, 1)), fontsize=10)
plt.ylabel('verjetnostna gostota')
plt.xlabel('t[s]')
plt.plot(_x, _y, label=r'model verjetnostne gostote $\mu e^{-\mu t}$')
plt.show()
#plt.clf()

# 3. del z generiranimi podatki
a = pd.read_csv('times.txt')
ts = (a-a.shift(1)).dropna().values.flatten()
mu2 = 1/np.mean(ts)
_x1 = np.linspace(min(ts), max(ts), 10000)
_y1 = mu2 * np.exp(-mu2*_x1)
plt.hist(ts, density=True, bins=10)
plt.text(0, 10, r'$\mu={}$'.format(round(mu2, 1)), fontsize=10)
plt.plot(_x1, _y1, label=r'model verjetnostne gostote $\mu e^{-\mu t}$')
plt.legend()
plt.show()