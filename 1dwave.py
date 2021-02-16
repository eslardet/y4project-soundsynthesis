""" 
Finite Difference Scheme for 1D Wave equation
Fixed boundary conditions
Raised cosine initial condition
"""

import numpy as np
from scipy.sparse import diags
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from math import pi, sin, floor

# Global parameters
SR = 32000                  # sample rate (Hz)
rho = 7850                  # string density (kg/m^3)
d = 1*10**(-3)              # string diammeter (m)
T_0 = 741                   # string tension (N)
L = 0.657                   # length of string (m)
TF = 1                      # duration of simulation (s)
ctr = 0.5*L                 # cosine centre location (m)
wid = 0.1*L                 # cosine width (m)
u0 = 0.01                   # max initial displacement (m)
v0 = 0                      # max initial velocity (m/s)
x_o = 0.3*L                 # observation point (m)
la = 1                      # courant number (lambda)

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

# Readout interpolation parameters
l_o = floor(N*x_o)          # rounded grid index for readout
a = x_o/h - l_o             # fractional part of readout location

# Create raised cosine
xax = np.transpose(np.arange(0,N+1))*h
ind = np.sign(np.maximum(-(xax-ctr-wid/2)*(xax-ctr+wid/2), 0))
rc = 0.5*ind*(1+np.cos(2*pi*(xax-ctr)/wid))

# Initialise grid functions and output
u2 = u0*rc
u1 = (u0+k*v0)*rc
u = np.zeros(N+1)
out = np.zeros(NF)

# Create update matrices
Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(N-1, N-1)).toarray() # h^2 * Dxx
B = 2*np.eye(N-1) + la**2 * Dxx

# To store total energy
Et = np.zeros(NF-2)

# Create energy matrices
Dx = np.delete(diags([-1, 1], [-1, 0], shape=(N,N)).toarray(), N-1, 1) # h * Dx+


#### Main loop ####
for n in range(2, NF):
    # Scheme string update
    u[1:N] = B@u1[1:N] - u2[1:N]

    # Output at observation point
    out[n] = (1-a)*u[l_o] + a*u[l_o+1] 

    # Energy calculation and update
    Ek = rho*Ar*h/(2*k**2) * np.transpose(u1-u2) @ (u1-u2)
    Ep = T_0/(2*h) * np.transpose(Dx@u1[1:N]) @ (Dx@u2[1:N])
    Et[n-2] = Ek + Ep
    
    # Update grid functions
    u2 = u1.copy()                      
    u1 = u.copy()


# Play sound
write("1dwave.wav", SR, out)

# Plot output waveform at observation point
plt.plot(np.arange(0,NF)*k, out)
plt.show()

# Plot energy
plt.plot(np.arange(2,NF)*k, Et)
plt.show()

# Plot variation from initial energy
E_var = np.zeros(NF-2)
for n in range(1, NF-2):
    E_var[n] = (Et[n] - Et[0])/ Et[0]

plt.plot(np.arange(0, NF-2)*k, E_var, '.r', markersize = 2)
plt.show()
