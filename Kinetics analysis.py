# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
from uncertainties import ufloat
from uncertainties.umath import *

# Define physical constants
R = 1.98720425864083  # Gas constant in cal K−1 mol−1
k_B = 1.38066E-23      # Boltzmann constant in J/K
h = 6.6260755E-34      # Planck constant in J·s

# Define linear function for Arrhenius and Eyring fits
def linearFunc(x, intercept, slope):
    y = intercept + slope * x
    return y

# Calculate activation enthalpy from slope (with uncertainty)
def calc_enthalpy(slope, std):
    H = -ufloat(slope, std) * R
    return H

# Calculate activation entropy from intercept (with uncertainty)
def calc_entropy(intersept, std):
    S = R * (ufloat(intersept, std) - np.log(k_B / h))
    return S

# Calculate Gibbs free energy from enthalpy and entropy at a given temperature
def calc_gibbs(H, S, T=298.15):
    G = H - (T * S)
    return G

# Define temperature range in reciprocal K (1/T) for Eyring/Arrhenius plots
xdata = np.array([1/(273.15+i) for i in (40, 50, 60, 70)])  # 1/T in K⁻¹

# Experimental rate constants and their standard deviations (H and D)
k_1_H = np.array([1.18E-05, 4.29E-05, 1.52E-04, 5.32E-04])
d_1_H = np.array([2.10E-07, 9.28E-07, 5.75E-06, 5.90E-06])

k_1_D = np.array([1.30E-06, 6.42E-06, 2.51E-05, 1.04E-04])
d_1_D = np.array([3.08E-08, 3.70E-07, 8.43E-07, 6.48E-06])

# Calculate kinetic isotope effect (KIE) as ln(k_H / k_D) with uncertainties
KIE = [log(ufloat(k_1_H[i], d_1_H[i]) / ufloat(k_1_D[i], d_1_D[i])) for i in range(len(k_1_H))]

# Print KIE values in normal exponential form
print([exp(i) for i in KIE])

# Plotting parameters
markersize = 7
alpha = 0.5
custom_palette = sns.color_palette("mako_r", 6)

# Fit ln(KIE) vs 1/T
a_fit_k1, cov_k1 = curve_fit(linearFunc, xdata, [i.nominal_value for i in KIE], 
                             sigma=[i.std_dev for i in KIE], absolute_sigma=True)

inter_k1 = a_fit_k1[0]
slope_k1 = a_fit_k1[1]
yfit_k1 = inter_k1 + slope_k1 * xdata  # Generate y values from best fit

# Plot ln(KIE) data and fit
plt.plot(xdata*1000, yfit_k1, color = custom_palette[0], linestyle='-', linewidth=2.0)


plt.errorbar(xdata*1000, [i.nominal_value for i in KIE], yerr = [i.std_dev for i in KIE], linestyle='None', 
             color = custom_palette[1], ecolor = 'black',
              label='KIE forward', alpha=alpha, marker='o', capsize=3, capthick=1, markersize=markersize)

# Configure plot appearance
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma']
plt.legend(prop={'size': 14})
plt.xlabel("1000/T (1000/K)", fontsize=14)
plt.ylabel("ln(KIE)", fontsize=14)
plt.ylim(-1, 3.5)
plt.gca().minorticks_on()
plt.tick_params(axis='both', which='both', direction='in', length=7, labelsize=12, colors='black')
plt.tick_params(which='minor', length=4)
plt.rc('grid', linestyle="-", color='grey', linewidth=0.1, alpha=0.5)
plt.grid(True)

# Save figure
plt.savefig('KIE_Plot_with_error.png', dpi=300)

########################################## Arrhenius section ##########################################

# Calculate ln(k) and std dev for H
ydata=[]
d_y=[]
for i in range(len(k_1_H)):
    value = log(ufloat(k_1_H[i], d_1_H[i]))
    ydata.append(value.nominal_value)
    d_y.append(value.std_dev)

ydata_k1 = np.array(ydata)
d_y_k1 = np.array(d_y)

# Same for D
ydata=[]
d_y=[]
for i in range(len(k_1_D)):
    value = log(ufloat(k_1_D[i], d_1_D[i]))
    ydata.append(value.nominal_value)
    d_y.append(value.std_dev)

ydata_k1_D = np.array(ydata)
d_y_k1_D = np.array(d_y)

# Fit Arrhenius plot (ln(k) vs 1/T) for H
a_fit_k1, cov_k1 = curve_fit(linearFunc, xdata, ydata_k1, sigma=d_y_k1, absolute_sigma=True)
inter_k1 = a_fit_k1[0]
slope_k1 = a_fit_k1[1]
d_inter_k1 = np.sqrt(cov_k1[0][0])
d_slope_k1 = np.sqrt(cov_k1[1][1])

# Calculate activation energy and pre-exponential factor
E_a_k1 = -ufloat(slope_k1, d_slope_k1) * R 
A_k1 = exp(ufloat(inter_k1, d_inter_k1))

# Repeat fit for D
a_fit_k1_D, cov_k1_D = curve_fit(linearFunc, xdata, ydata_k1_D, sigma=d_y_k1_D, absolute_sigma=True)
inter_k1_D = a_fit_k1_D[0]
slope_k1_D = a_fit_k1_D[1]
d_inter_k1_D = np.sqrt(cov_k1_D[0][0])
d_slope_k1_D = np.sqrt(cov_k1_D[1][1])

E_a_k1_D = -ufloat(slope_k1_D, d_slope_k1_D) * R
A_k1_D = exp(ufloat(inter_k1_D, d_inter_k1_D))

# Calculate Arrhenius prefactor ratio and ln of that ratio
A_ratio = ufloat(A_k1.nominal_value, A_k1.std_dev) / ufloat(A_k1_D.nominal_value, A_k1_D.std_dev)
print('A_H/A_D_k1_inv:', A_ratio)
print('ln(A_H/A_D)_k1_inv:', log(A_ratio))

# Difference in activation energy in kcal/mol
print('E_a_D - E_a_H_k1_inv:', (ufloat(E_a_k1_D.nominal_value, E_a_k1_D.std_dev) - ufloat(E_a_k1.nominal_value, E_a_k1.std_dev)) / 1000)

# Generate fitted ln(k) lines
yfit_k1 = inter_k1 + slope_k1 * xdata
yfit_k1_D = inter_k1_D + slope_k1_D * xdata

print("Arrhenius analysis is successfully done")

########################################## Eyring section ##########################################
print()
print("Start of Eyring analysis")

# ln(k/T) vs 1/T for H
ydata=[]
d_y=[]
for i in range(len(k_1_H)):
    value = log(ufloat(k_1_H[i], d_1_H[i]) * xdata[i])  # log(k/T)
    ydata.append(value.nominal_value)
    d_y.append(value.std_dev)

ydata_k1 = np.array(ydata)
d_y_k1 = np.array(d_y)

# Fit Eyring plot
a_fit_k1, cov_k1 = curve_fit(linearFunc, xdata, ydata_k1, sigma=d_y_k1, absolute_sigma=True)
inter_k1 = a_fit_k1[0]
slope_k1 = a_fit_k1[1]
d_inter_k1 = np.sqrt(cov_k1[0][0])
d_slope_k1 = np.sqrt(cov_k1[1][1])

# Thermodynamic parameters from Eyring plot
H_k1 = calc_enthalpy(slope_k1, d_slope_k1)
print('.........Protium data...........')
print('H:', H_k1/1000)
S_k1 = calc_entropy(inter_k1, d_inter_k1)
print('S_k1:', S_k1)
G = calc_gibbs(H_k1, S_k1)
print('G k1_inv:', G/1000)

# ln(k/T) vs 1/T for D
ydata=[]
d_y=[]
for i in range(len(k_1_D)):
    value = log(ufloat(k_1_D[i], d_1_D[i]) * xdata[i])
    ydata.append(value.nominal_value)
    d_y.append(value.std_dev)
    
ydata_k1_D = np.array(ydata)
d_y_k1_D = np.array(d_y)

# Fit Eyring plot
a_fit_k1_D, cov_k1_D = curve_fit(linearFunc, xdata, ydata_k1_D, sigma=d_y_k1_D, absolute_sigma=True)
inter_k1_D = a_fit_k1_D[0]
slope_k1_D = a_fit_k1_D[1]
d_inter_k1_D = np.sqrt(cov_k1_D[0][0])
d_slope_k1_D = np.sqrt(cov_k1_D[1][1])

# Thermodynamic parameters for D
H_k1_D = calc_enthalpy(slope_k1_D, d_slope_k1_D)
print('........Deuterium data..........')
print('H:', H_k1_D/1000)
S_k1_D = calc_entropy(inter_k1_D, d_inter_k1_D)
print('S k1_inv:', S_k1_D)
G = calc_gibbs(H_k1_D, S_k1_D)
print('G k1_inv:', G/1000)

# Generate fits for Eyring plot
yfit_k1 = inter_k1 + slope_k1 * xdata
yfit_k1_D = inter_k1_D + slope_k1_D * xdata

# Plotting the Eyring plots
markersize = 7
alpha = 0.5
custom_palette = sns.color_palette("mako_r", 4)

print(yfit_k1)
print(ydata_k1)

plt.plot(xdata*1000, yfit_k1, color = custom_palette[0], linestyle='-', linewidth=2.0)
plt.errorbar(xdata*1000, ydata_k1, yerr = d_y_k1, linestyle='None', color = custom_palette[1], ecolor = 'black',
             label='$k_{1}$ H', alpha=alpha, marker='o', capsize=3, capthick=1, markersize=markersize)

plt.plot(xdata*1000, yfit_k1_D, color = custom_palette[2], linestyle='-', linewidth=2.0)
plt.errorbar(xdata*1000, ydata_k1_D, yerr = d_y_k1_D, linestyle='None', color = custom_palette[3], ecolor = 'black',
             label='$k_{1}$ D', alpha=alpha, marker='o', capsize=3, capthick=1, markersize=markersize)

# Configure appearance
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma']
plt.legend(prop={'size': 14})
plt.xlabel("1000/T (1000/K)", fontsize=14)
plt.ylabel("ln(k)", fontsize=14)
plt.gca().minorticks_on()
plt.tick_params(axis='both', which='both', direction='in', length=7, labelsize=12, colors='black')
plt.tick_params(which='minor', length=4)
plt.rc('grid', linestyle="-", color='grey', linewidth=0.1, alpha=0.5)
plt.grid(True)

# Save figure
plt.savefig('Plot_with_error.png',dpi=300)
plt.show()
