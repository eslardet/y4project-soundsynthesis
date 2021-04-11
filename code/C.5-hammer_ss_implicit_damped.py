"""
Finite Difference Scheme for Stiff String with 2 damping parameters
Simply supported boundary conditions
Coupled with implicit hammer force term
"""

import numpy as np
from scipy.sparse import diags
from scipy.io.wavfile import write
from scipy import optimize
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from math import pi, sin, floor

# Global parameters
SR = 32000          # sample rate (Hz)
TF = 5              # duration of simulation (s)
L = 0.657           # length of string (m)
d = 1e-3            # string diammeter (m)
rho = 7850          # string density (kg/m^3)
T_0 = 741           # string tension (N)
E = 2.02e11         # Young's modulus (Pa)
sig0 = 0.18         # non-frequency dependent damping paramter (s^-1)
sig1 = 2.6e-9       # frequency dependent damping paramter (s)
x_o = 0.3*L         # observation point (m)

xH0 = -0.01         # hammer initial displacement (m)
vH0 = 1.5           # hammer initial velocity (m/s)
xH = 0.079          # striking location (m)
MH = 8.71e-3        # hammer mass (kg)
KH = 5.84e9         # hammer stiffness parameter (N/m^p)
p = 2.418           # hammer stiffness nonlinearity exponent

# Derived parameters
Ar = pi * d**2 / 4          # string cross-sectional area (m^2)
I = pi * d**4 / 64          # area moment of inertia
c = np.sqrt(T_0/(rho*Ar))   # wave speed
K = np.sqrt(E*I/(rho*Ar))   # stiffness parameter
k = 1/SR                    # time step
NF = floor(SR*TF)           # duration of simulation (samples)

# Stability condition/ scheme parameters
h = np.sqrt((c**2 * k**2 + np.sqrt(c**4 * k**4 + 16*K**2 * k**2)) / 2)
N = floor(L/h)
h = L/N
la = c*k/h
mu = K*k/(h**2)

# For spreading operator epsilon
lH = floor(xH/h)
eps = np.zeros(N-1)
eps[lH-1] = 1/h

# Readout interpolation parameters
l_o = floor(N*x_o)          # rounded grid index for readout
frac = x_o/h - l_o          # fractional part of readout location

# Initialise grid functions and output
uH2 = xH0               # hammer at n-1
uH1 = xH0 + k*vH0       # hammer at n

u2 = np.zeros(N+1)      # string at n-2
u1 = np.zeros(N+1)      # string at n-1
u = np.zeros(N+1)       # string at n

out = np.zeros(NF)      # output at observation point
f = np.zeros(NF)        # force
Et = np.zeros(NF-2)     # total energy
Qt = 0                  # cumulative energy loss
comp = 0                # for kahan summation

# Create update matrices
Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(N-1, N-1)).toarray() # h^2 * Dxx

A = (1 + sig0*k)*np.eye(N-1) - sig1*la**2/k * Dxx
B = 2*np.eye(N-1) + la**2 * Dxx - mu**2 * Dxx@Dxx
C = (1 - sig0*k)*np.eye(N-1) + sig1*la**2/k * Dxx

# Create energy matrices
Dx_E = np.delete(diags([-1, 1], [-1, 0], shape=(N, N)).toarray(), N-1, 1) # h * Dx+
Dxx_E = diags([-1, 2, -1], [-1, 0, 1], shape=(N-1, N-1)).toarray() # h^2 * Dxx

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

def thomas(a, b, c, d):
    """
    Thomas Algorithm to solve linear system Ax = d
    a, b, c are vectors containing the sub-diagonal, diagonal and super-diagonal entries of the tri-diagonal matrix A
    d is the right vector
    """
    n = len(b)
    cc= np.zeros(n-1)
    dc= np.zeros(n)
    x = np.zeros(n)
    cc[0] = c[0] / b[0]
    dc[0] = d[0] / b[0]

    for i in range(1, n):
        w = b[i] - a[i-1]*cc[i-1]
        if i != n-1:
            cc[i] = c[i]/w
        dc[i] = (d[i]-a[i-1]*dc[i-1])/w
    x[n-1] = dc[n-1]
    for i in range(n-2, -1, -1):
        x[i] = dc[i] - cc[i]*x[i+1]

    return x

# Vectors required for the Thomas algorithm
a2 = np.diagonal(A, offset=-1).copy()
b2 = np.diagonal(A).copy()
c2 = np.diagonal(A, offset=1).copy()

# Find the solution of Ax=epsilon
Ainv_eps = thomas(a2, b2, c2, eps)
# Constant parameter in G
Ga = k**2 * (1/MH + Ainv_eps[lH-1]/(rho*Ar))
# Initial guess for Newton-Raphson
r1 = xH0

#### Main loop ####
for n in range(2, NF):
    # String update (without force)
    u[1:N] = thomas(a2, b2, c2, B@u1[1:N] - C@u2[1:N])
    
    # Find knowns in G
    Dn2 = uH2 - u2[lH]
    Dn1 = uH1 - u1[lH]
    Gb = 2*uH2 - 2*uH1 - u2[lH] + u[lH]
    
    # Perform newton method to find the root r
    r = optimize.newton(G, r1, fprime=Gdx)
    
    # Find Delta^{n+1} from r
    Dn = r + Dn2
    
    # Calculate force at nth time step
    F = (phi(Dn) - phi(Dn2))/r
    f[n] = F
    
    # Hammer update
    uH = 2*uH1 - uH2 - k**2/MH * F
    
    # Add force to string update
    u[1:N] += k**2/(rho*Ar) * Ainv_eps * F

    # Output at observation point
    out[n] = (1-frac)*u[l_o] + frac*u[l_o+1]

    # Energy calculation and update
    Ek = rho*Ar*h/(2*k**2) * np.transpose(u1-u2) @ (u1-u2) # kinetic
    Ep = T_0 / (2*h) * np.transpose(Dx_E@u1[1:N]) @ (Dx_E@u2[1:N]) + \
        E * I / (2*h**3) * np.transpose(Dxx_E@u1[1:N]) @ (Dxx_E@u2[1:N]) # potential
    Eh = MH/(2*k**2) * (uH1 - uH2)**2 + 1/2 * (phi(Dn1) + phi(Dn2)) # hammer
    Q = rho * Ar * sig0 * h / (2*k**2) * np.transpose(u-u2) @ (u-u2) + \
        T_0 * sig1 / (2*h*k**2) * np.transpose(Dx_E@(u[1:N] - u2[1:N])) @ (Dx_E@(u[1:N] - u2[1:N])) # loss
    Et[n-2] = Ek + Ep + Eh + k*Qt # total
    Qt += Q # cumulative energy

    # Update grid functions and hammer parameters
    u2 = u1.copy()
    u1 = u.copy()
    uH2 = uH1
    uH1 = uH
    r1 = r

# Make sound audible
out2 = out / abs(np.max(out))

plt.plot(np.arange(0,NF)*k, out)
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.show()

# Play sound
write("hammer_ss_implicit_damped.wav", SR, out2)

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

# Plot normalised energy variation
E_var = np.zeros(NF-2)
for n in range(1, NF-2):
    E_var[n] = (Et[n] - Et[0])/ Et[0]

plt.plot(np.arange(0, NF-2)*k, E_var, '.r', markersize = 2)
plt.show()

# Plot frequency spectrum (amplitude (dB) vs frequency (Hz))
yf = fft(out)
xf = fftfreq(NF, k)[:NF//2]
plt.plot(xf, 20*np.log10(np.abs(yf[0:NF//2])))
plt.show()