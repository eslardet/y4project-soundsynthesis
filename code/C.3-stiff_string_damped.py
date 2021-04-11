""" 
Explicit/ Implicit Finite Difference Scheme for Stiff String with 2 damping parameters
Simply supported boundary conditions
Raised cosine initial condition
"""

import numpy as np
from scipy.sparse import diags
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from math import pi, sin, floor

# Scheme options
scheme = 0          # scheme type
                    # 0: explicit, 1: implicit

# Global parameters
SR = 32000          # sample rate (Hz)
TF = 1              # duration of simulation (s)
L = 0.657           # length of string (m)
d = 1e-3            # string diammeter (m)
rho = 7850          # string density (kg/m^3)
T_0 = 741           # string tension (N)
E = 2.02e11         # Young's modulus (Pa)
sig0 = 0.18         # non-frequency dependent damping paramter (s^-1)
sig1 = 2.6e-9       # frequency dependent damping paramter (s)
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
k = 1/SR                    # time step
NF = floor(SR*TF)           # duration of simulation (samples)

# Stability condition/ scheme parameters
if scheme == 0:
    h = np.sqrt((c**2 * k**2 + 4*sig1*c**2*k + np.sqrt((c**2*k**2 + 4*sig1*c**2*k)**2 + 16*K**2 * k**2)) / 2)
if scheme == 1:
    h = np.sqrt((c**2 * k**2 + np.sqrt(c**4 * k**4 + 16*K**2 * k**2)) / 2)
N = floor(L/h)
h = L/N
la = c*k/h
mu = K*k/(h**2)

# Readout interpolation parameters
l_o = floor(N*x_o)      # rounded grid index for readout
frac = x_o/h - l_o      # fractional part of readout location

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
Qt = 0                  # cumulative energy loss
comp = 0                # for kahan summation

# Create update matrices
Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(N-1, N-1)).toarray() # h^2 * Dxx

if scheme == 0:
    B = (2*np.eye(N-1) + la**2 * Dxx - mu**2 * Dxx@Dxx + 2*sig1*la**2/k * Dxx) / (1 + sig0*k)
    C = ((1 - sig0*k)*np.eye(N-1) + 2*sig1*la**2/k * Dxx) / (1 + sig0*k)
if scheme == 1:
    A = (1 + sig0*k)*np.eye(N-1) - sig1*la**2/k * Dxx
    B = 2*np.eye(N-1) + la**2 * Dxx - mu**2 * Dxx@Dxx
    C = (1 - sig0*k)*np.eye(N-1) + sig1*la**2/k * Dxx

# Energy matrices
Dx_E = np.delete(diags([-1, 1], [-1, 0], shape=(N, N)).toarray(), N-1, 1) # h * Dx+
Dxx_E = diags([-1, 2, -1], [-1, 0, 1], shape=(N-1, N-1)).toarray() # h^2 * Dxx

def thomas(a, b, c, d):
    """
    Thomas Algorithm to solve linear system Ax = d
    a, b, c are vectors containing the sub-diagonal, diagonal and super-diagonal entries of the tri-diagonal matrix A
    d is the right vector
    """
    n = len(b)
    cc= np.zeros(n-1)
    dd= np.zeros(n)
    x = np.zeros(n)
    cc[0] = c[0] / b[0]
    dd[0] = d[0] / b[0]

    for i in range(1, n):
        w = b[i] - a[i-1]*cc[i-1]
        if i != n-1:
            cc[i] = c[i]/w
        dd[i] = (d[i]-a[i-1]*dd[i-1])/w
    x[n-1] = dd[n-1]
    for i in range(n-2, -1, -1):
        x[i] = dd[i] - cc[i]*x[i+1]

    return x

# Vectors required for the Thomas algorithm
if scheme == 1:
    a2 = np.diagonal(A, offset=-1).copy()
    b2 = np.diagonal(A).copy()
    c2 = np.diagonal(A, offset=1).copy()

#### Main loop ####
for n in range(2, NF):
    # Scheme string update
    if scheme == 0:
        u[1:N] = B@u1[1:N] - C@u2[1:N]
    if scheme == 1:
        u[1:N] = thomas(a2, b2, c2, B@u1[1:N] - C@u2[1:N])
     
    # Output at observation point
    out[n] = (1-frac)*u[l_o] + frac*u[l_o+1]

    # Energy calculation and update
    Ek = rho*Ar*h/(2*k**2) * np.transpose(u1-u2) @ (u1-u2) # kinetic
    Ep = T_0 / (2*h) * np.transpose(Dx_E@u1[1:N]) @ (Dx_E@u2[1:N]) + \
        E * I / (2*h**3) * np.transpose(Dxx_E@u1[1:N]) @ (Dxx_E@u2[1:N]) # potential
    if scheme == 0:
        Q = rho * Ar * sig0 * h / (2*k**2) * np.transpose(u-u2) @ (u-u2) + \
            T_0 * sig1 / (h * k**2) * np.transpose(Dx_E@(u[1:N] - u2[1:N])) @ (Dx_E@(u1[1:N] - u2[1:N])) # loss
    if scheme == 1:
        Q = rho * Ar * sig0 * h / (2*k**2) * np.transpose(u-u2) @ (u-u2) + \
            T_0 * sig1 / (2*h*k**2) * np.transpose(Dx_E@(u[1:N] - u2[1:N])) @ (Dx_E@(u[1:N] - u2[1:N])) # loss
    Et[n-2] = Ek + Ep + k*Qt # total
    Qt += Q # cumulative energy

    # Update grid functions
    u2 = u1.copy()
    u1 = u.copy()
    
# Play sound
write("stiff_string_damped.wav", SR, out)

# Plot output waveform
plt.plot(np.arange(0, NF)*k, out)
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