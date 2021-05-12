import pandas as pd
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
mili = 1e-3
prazn = pd.read_csv('data_praznjenje.dat', sep=' ')
poln = pd.read_csv('data_polnjenje.dat', sep=' ')
prazn['t'] = prazn['t'] * mili
poln['t'] = poln['t'] * mili

model_prazn = lambda t, x0, _tau: x0*np.exp(-t/_tau)
model_poln = lambda t, x0, _tau: x0*(1-np.exp(-t/_tau))

popt, pcov = curve_fit(model_prazn, prazn['t'], prazn['U'], p0=(12.0, 1e-2))
e = np.sqrt(np.diag(pcov))
U0, tau = ufloat(popt[0], e[0]), ufloat(popt[1], e[1])
l = len(prazn['t'].values)
xe, ye = 2e-3, 0.4
plt.errorbar(prazn['t'].values, prazn['U'].values, xerr=[xe]*l, yerr=[ye]*l, fmt=".", label='meritve')
plt.grid()
_t = np.linspace(min(prazn['t']), max(prazn['t']), 1000)
plt.plot(_t, model_prazn(_t, U0.n, tau.n), "--")
plt.legend()
plt.xlabel('t[s]')
plt.yscale('log')
plt.ylabel('U[V]')
plt.show()

C1 = ufloat(0.25, 0.1*0.25) * 1e-6
R1 = ufloat(39, 1) * 1000
mtau1 = C1 * R1 * 1000
mtau = tau * 1000
print('τ(izračunani)= RC =', mtau1, 'ms')
print('τ(izmerjeni)=', mtau, 'ms')


popt, pcov = curve_fit(model_poln, poln['t'].values, poln['U'].values, p0=(3.0, 0.4))
e = np.sqrt(np.diag(pcov))
Un, tau = ufloat(popt[0], e[0]), ufloat(popt[1], e[1])
R2 = 2.7e6

U_izvir = 12.0
Rn = R2 / (U_izvir / Un - 1)
alpha = ( R2/Rn + 1)
mtau2 = R2 * C1 * 1000 / alpha
mtau = tau * 1000

poln_err_t = 20e-3
poln_err_U = 130e-3

print('τ(izračunani)= RC/α =', mtau2, 'ms')
print('τ(izmerjeni)=', mtau, 'ms')
pt, pu = poln['t'].values, poln['U'].values,
plt.errorbar(pt, pu, xerr=[poln_err_t]*len(pt) , yerr=[poln_err_U]*len(pt) , ecolor="gray", fmt=".", label='meritve')
plt.grid()
_t = np.linspace(min(poln['t']), max(poln['t']), 1000)
plt.plot(_t, model_poln(_t, Un.n, tau.n), "--")
plt.legend()
plt.xlabel('t[s]')
plt.ylabel('U[V]')

plt.show()


nihanje = pd.read_csv('data_nihanje.dat', sep=' ')
model_nih = lambda t, a0, beta: a0 * np.exp(-beta*t)
casi_err = 1e-3
ampl_err = 0.09

N = 12
t = ufloat(40e-3, casi_err)
t0 = t/N
print(N, 'nihajev v', t, 's')

omega = 2*np.pi/t0

casi = nihanje['N'].values * t0.n
ampl = nihanje['A'].values
popt, pcov = curve_fit(model_nih, casi, ampl)
a0, a9 = ufloat(ampl[0], ampl_err), ufloat(ampl[-1], ampl_err)
beta_2 = 1/9/t0*unumpy.log(a0/a9)

beta = ufloat(popt[1], beta_2.s) # uporabimo napako od beta2 !

plt.errorbar(casi, ampl, xerr=[casi_err]*len(casi) , yerr=[ampl_err]*len(casi) , ecolor="gray", fmt=".", label='meritve')
_t = np.linspace(min(casi), max(casi), 1000)
plt.plot(_t, model_nih(_t, popt[0], beta.n), "--")
plt.grid()
plt.xlabel('t[s]')
plt.ylabel('amplituda [V]')
plt.yscale('log')
plt.show()

L = 1.227 #H
R = 138 #Ohm
U=12
beta_1 = R/(2*L)
omega_0_1 = (1/(L*C1))**0.5
omega_1 = (omega_0_1**2 - beta_1**2)**0.5

print('ω(izmerjeni)=2π/t0=',omega,'/s')
print('ω0(izrač. lastna frek.)=', omega_0_1,'/s')
print('ω(izrač.)=sqrt((ω0)^2-beta^2)=',omega_1,'/s')

print('β(teor.)=R/(2L)=',beta_1,'/s')
print('β(izmerjeni)=',beta,'/s')
print('I_0=U/R', 1e3 * U / R , 'mA')