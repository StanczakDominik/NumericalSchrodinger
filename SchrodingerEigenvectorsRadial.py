#Schrodinger radial equation with a Coulomb solved by finding eigenvectors of the hamiltonian
from __future__ import division

import numpy as np
import scipy
import scipy.integrate
import scipy.linalg
import scipy.misc
import scipy.special
import pylab as plt

l=80
N=1024
V0=1
a=1
angularl=1

x1=5
x2=12
def v(x):
    return -1/x+hbar*hbar*angularl*(angularl+1)/2/m/x**2
m=1
w=1
hbar=1
X, dx=np.linspace(0.001, l, N, retstep=True)
V = np.zeros_like(X)
for i in range(len(V)):
    print(X[i])
    V[i] = v(X[i])
H=np.diag(V)

for i in range(N):
    H[i,i]+=hbar*hbar/m/dx/dx
    if(i>0):
        H[i-1, i]-= 0.5*hbar*hbar/m/dx/dx
    if i<N-1:
        H[i+1, i]-= 0.5*hbar*hbar/m/dx/dx
print(H)

E, psi_E = scipy.linalg.eigh(H)
plt.plot(np.linspace(0,N,N)[:400], E[:400])
plt.title("Energies for the first $400$ states")
plt.xlabel("State index")
plt.ylabel("Energy")
plt.savefig("RadialEquationCoulombPotentialEigenvectorsEnergies.png")
plt.show()

for n in range(5):
    psi=psi_E[:,n]/X  # psi_E is the solution to the f(r) part of the f(r)/R radial part
    norm = np.sqrt(scipy.integrate.simps(psi**2, X))
    psi/=norm
    plt.plot(X, psi, label="$n=%i$" % n)
plt.grid()
plt.title("Radial wavefunctions for the Coulomb Potential")
plt.xlabel("Radius $r$")
plt.ylabel("Normalized wavefunction amplitude $\psi$")
plt.legend()
plt.savefig("RadialEquationCoulombPotentialEigenvectorsWavefunctions.png")
plt.show()