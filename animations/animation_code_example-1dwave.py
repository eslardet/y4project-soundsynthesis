""" 
Finite Difference Scheme for 1D Wave equation
Fixed boundary conditions
Raised cosine initial condition
"""

import numpy as np
from scipy.sparse import diags
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from matplotlib import animation
from math import pi, floor

# Global parameters
SR = 32000                  # sample rate (Hz)
rho = 7850                  # string density (kg/m^3)
d = 1*10**(-3)              # string diammeter (m)
T_0 = 741                   # string tension (N)
L = 0.657                   # length of string (m)
TF = 1                      # duration of simulation (s)
ctr = 0.7*L                 # cosine centre location (m)
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

# Create raised cosine
xax = np.transpose(np.arange(0,N+1))*h
ind = np.sign(np.maximum(-(xax-ctr-wid/2)*(xax-ctr+wid/2), 0))
rc = 0.5*ind*(1+np.cos(2*pi*(xax-ctr)/wid))

# Initialise grid functions and output
u2 = u0*rc              # string at n-2
u1 = (u0+k*v0)*rc       # string at n-1
u = np.zeros(N+1)       # string at n

# Create update matrices
Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(N-1, N-1)).toarray() # h^2 * Dxx
B = 2*np.eye(N-1) + la**2 * Dxx


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

def nextu():
    global u1, u2
    u[1:N] = B@u1[1:N] - u2[1:N]        # scheme string update
    u2 = u1.copy()
    u1 = u.copy()


def animate(n):
    nextu()                             # get next grid function
    ax.clear()
    ax.set_ylim([-u0,u0])
    ax.set_title("t = " + str(np.round(n/NF*TF, 4)))
    ax.set_xlabel("String location (m)")
    ax.set_ylabel("Displacement (m)")
    ax.plot(np.arange(0,N+1)*h, u)      # plot string
    return ax

# Animation of string
ani = animation.FuncAnimation(fig, animate, frames=100, interval=10)
ani.save("1dwave.gif")
# plt.show()