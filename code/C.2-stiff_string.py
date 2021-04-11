""" 
Finite Difference Scheme for Stiff String
Clamped/ simply supported boundary conditions
Raised cosine initial condition
"""

import numpy as np
from scipy.sparse import diags
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from math import pi, floor

# Scheme options
bc = 1              # boundary condition type 
                    # 0: clamped, 1: simply supported

# Global parameters
SR = 32000          # sample rate (Hz)
TF = 1              # duration of simulation (s)
L = 0.657           # length of string (m)
d = 1e-3            # string diammeter (m)
rho = 7850          # string density (kg/m^3)
T_0 = 741           # string tension (N)
E = 2.02e11         # Young's modulus (Pa)
ctr = 0.5*L         # cosine centre location (m)
wid = 0.1*L         # cosine width (m)
u0 = 0.01           # max initial displacement (m)
v0 = 0              # max initial velocity (m/s)
x_o = 0.3*L         # observation point (m)

# Derived parameters
Ar = pi * d**2 / 4          # string cross-sectional area (m^2)
I = pi * d**4 / 64          # area moment of inertia (m^4)
c = np.sqrt(T_0/(rho*Ar))   # wave speed (m/s)
K = np.sqrt(E*I/(rho*Ar))   # stiffness parameter
k = 1/SR                    # time step (s)
NF = floor(SR*TF)           # number of samples in simulation

# Stability condition/ scheme parameters
h = np.sqrt((c**2 * k**2 + np.sqrt(c**4 * k**4 + 16*K**2 * k**2)) / 2)
N = floor(L/h)
h = L/N
la = c*k/h
mu = K*k/(h**2)

# Readout interpolation parameters
l_o = floor(N*x_o)      # rounded grid index for readout
a = x_o/h - l_o         # fractional part of readout location

# Create raised cosine
xax = np.transpose(np.arange(0,N+1))*h
ind = np.sign(np.maximum(-(xax-ctr-wid/2)*(xax-ctr+wid/2), 0))
rc = 0.5*ind*(1+np.cos(2*pi*(xax-ctr)/wid))

# Initialise grid functions and output
u2 = u0*rc              # string at n-2
u1 = (u0+k*v0)*rc       # string at n-1
u = np.zeros(N+1)       # string at n

out = np.zeros(NF)      # output at observation point
Et = np.zeros(NF-2)     # total energy

# Create update matrices
if bc == 0:
    Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(N+1, N+1)).toarray()
    D4x = np.delete(np.delete(Dxx@Dxx, [0,1,N-1,N], 0), [0,1,N-1,N], 1) # h^4 * Dxxxx
    Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(N-3, N-3)).toarray() # h^2 * Dxx

    B = la**2 * Dxx - mu**2 * D4x + 2*np.eye(N-3)

if bc == 1:
    Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(N-1, N-1)).toarray() # h^2 * Dxx

    B =  la**2 * Dxx - mu**2 * Dxx@Dxx + 2*np.eye(N-1)

# Energy matrices
if bc == 0:
    Dx_E = np.delete(diags([-1, 1], [-1, 0], shape=(N-2, N-2)).toarray(), N-3, 1) # h * Dx+
    Dxx_E = np.delete(diags([1, -2, 1], [-1, 0, 1], shape=(N-1, N-1)).toarray(), [0, N-2], 1) # h^2 * Dxx

if bc == 1:
    Dx_E = np.delete(diags([-1, 1], [-1, 0], shape=(N, N)).toarray(), N-1, 1) # h * Dx+
    Dxx_E = diags([-1, 2, -1], [-1, 0, 1], shape=(N-1, N-1)).toarray() # h^2 * Dxx

# Left and right end points for non-zero grid points in u
if bc == 0:
    ul, ur = 2, N-1
if bc == 1:
    ul, ur = 1, N

#### Main loop ####
for n in range(2, NF):
    # Scheme string update
    u[ul:ur] = B@u1[ul:ur] - u2[ul:ur]

    # Output at observation point
    out[n] = (1-a)*u[l_o] + a*u[l_o+1]

    # Energy calculation and update
    Ek = rho*Ar*h/(2*k**2) * np.transpose(u1-u2) @ (u1-u2) # kinetic
    Ep = T_0/(2*h) * np.transpose(Dx_E@u1[ul:ur]) @ (Dx_E@u2[ul:ur]) + \
        E*I/(2*h**3) * np.transpose(Dxx_E@u1[ul:ur]) @ (Dxx_E@u2[ul:ur]) # potential
    Et[n-2] = Ek + Ep # total

    # Update grid functions
    u2 = u1.copy()
    u1 = u.copy()

# Play sound
write("stiff_string.wav", SR, out)

# Plot output waveform at observation point
plt.plot(np.arange(0, NF)*k, out)
plt.show()

# Plot energy
plt.plot(np.arange(2,NF)*k, Et)
plt.show()

# Plot normalised energy variation
E_var = np.zeros(NF-2)
for n in range(1, NF-2):
    E_var[n] = (Et[n] - Et[0])/ Et[0]

plt.plot(np.arange(0, NF-2)*k, E_var, '.r', markersize = 2)
plt.show()