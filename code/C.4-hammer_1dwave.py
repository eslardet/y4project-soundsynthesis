"""
Finite Difference Scheme for 1D Wave equation
Fixed boundary conditions
Coupled with implicit hammer force term
"""

import numpy as np
from scipy.sparse import diags
from scipy.io.wavfile import write
from scipy import optimize
import matplotlib.pyplot as plt
from math import pi, floor

# Global parameters
SR = 32000          # sample rate (Hz)
TF = 1              # duration of simulation (s)
L = 0.657           # length of string (m)
d = 1e-3            # string diammeter (m)
rho = 7850          # string density (kg/m^3)
T_0 = 741           # string tension (N)
x_o = 0.3*L         # observation point (m)
la = 1              # courant number (lambda)

xH0 = -0.01         # hammer initial displacement (m)
vH0 = 1.5           # hammer initial velocity (m/s)
xH = 0.079          # striking location (m)
MH = 8.71e-3        # hammer mass (kg)
KH = 5.84e9         # hammer stiffness parameter (N/m^p)
p = 2.418           # hammer stiffness nonlinearity exponent

# Derived parameters
Ar = pi * d**2 / 4          # string cross-sectional area (m^2)
c = np.sqrt(T_0/(rho*Ar))   # wave speed (m/s)
k = 1/SR                    # time step (s)
NF = floor(SR*TF)           # number of samples in simulation

# Stability condition/ scheme parameters
h = c*k/la
N = floor(L/h)
h = L/N
la = c*k/h

lH = floor(xH/h)        # for spreading operator epsilon

# Readout interpolation parameters
l_o = floor(N*x_o)      # rounded grid index for readout
frac = x_o/h - l_o      # fractional part of readout location

# Initialise grid functions and output
uH2 = xH0               # hammer at n-1
uH1 = xH0 + k*vH0       # hammer at n

u2 = np.zeros(N+1)      # string at n-2
u1 = np.zeros(N+1)      # string at n-1
u = np.zeros(N+1)       # string at n

out = np.zeros(NF)      # output at observation point
f = np.zeros(NF)        # force
Et = np.zeros(NF-2)     # total energy

# Create update matrices
Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(N-1, N-1)).toarray() # h^2 * Dxx
B = 2*np.eye(N-1) + la**2 * Dxx

# Create energy matrices 
Dx_E = np.delete(diags([-1, 1], [-1, 0], shape=(N,N)).toarray(), N-1, 1) # h * Dx+

# Functions phi and G and their derivates for Newton-Raphson method
def phi(x):
    if x > 0:
        return KH/(p+1) * x**(p+1)
    else:
        return 0
    
def phidx(x):
    if x > 0:
        return KH * x**p
    else:
        return 0

def G(x):
    return Ga/x * (phi(x + Dn2) - phi(Dn2)) + x + Gb

def Gdx(x):
    return 1 + Ga/x * phidx(x + Dn2) + Ga/x**2 * (phi(Dn2) - phi(x + Dn2))

Ga = k**2 * (1/MH + 1/(rho*Ar*h))   # constant parameter in G
r1 = xH0    # initial guess for Newton-Raphson

#### Main loop ####
for n in range(2, NF):
    # Find knowns in G
    Dn2 = uH2 - u2[lH]
    Dn1 = uH1 - u1[lH]
    Gb = 2*uH2 - 2*uH1 + B[lH-1,:]@u1[1:N] - 2*u2[lH]
    
    # Perform Newton-Raphson method to find the root r
    r = optimize.newton(G, r1, fprime=Gdx)
    
    # Find Delta^{n+1} from r
    Dn = r + Dn2
    
    # Calculate force at nth time step
    F = (phi(Dn) - phi(Dn2))/r
    f[n] = F
    
    # Hammer update
    uH = 2*uH1 - uH2 - k**2/MH * F
    
    # String update
    u[1:N] = B@u1[1:N] - u2[1:N]
    u[lH] += k**2/(h*rho*Ar) * F

    # Output at observation point
    out[n] = (1-frac)*u[l_o] + frac*u[l_o+1]

    # Energy calculation and update
    Ek = rho*Ar*h/(2*k**2) * np.transpose(u1-u2) @ (u1-u2) # kinetic
    Ep = T_0/(2*h) * np.transpose(Dx_E@u1[1:N]) @ (Dx_E@u2[1:N]) # potential
    Eh = MH/(2*k**2) * (uH1 - uH2)**2 + 1/2 * (phi(Dn1) + phi(Dn2)) # hammer
    Et[n-2] = Ek + Ep + Eh

    # Update grid functions and hammer parameters
    u2 = u1.copy()
    u1 = u.copy()
    uH2 = uH1
    uH1 = uH
    r1 = r

# Make sound audible
out2 = out / abs(np.max(out))

# Play sound
write("hammer_1dwave.wav", SR, out2)

# Plot output waveform at observation point and force
fig, ax = plt.subplots(2, 1)
plt.subplots_adjust(hspace = 1)

plt.sca(ax[0])
plt.plot(np.arange(0,NF)*k, out)
plt.sca(ax[1])
plt.plot(np.arange(0,NF)*k, f)
plt.show()

# Plot energy
plt.plot(np.arange(2,NF)*k, Et)
plt.show()

# Plot energy variation
E_var = np.zeros(NF-2)
for n in range(1, NF-2):
    E_var[n] = (Et[n] - Et[0])/ Et[0]

plt.plot(np.arange(0, NF-2)*k, E_var, '.r', markersize = 2)
plt.show()