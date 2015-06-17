#Schrodinger 1D equation solved by finding eigenvectors for a gaussian potential
import numpy as np
import scipy
import scipy.integrate
import scipy.linalg
import scipy.misc
import scipy.special
import pylab as plt

l=10
N=1024

def v(x):
	return np.exp(-x*x)
m=1
w=1
hbar=1
X, dx=np.linspace(-l, l, N, retstep=True)

H=np.diag(v(X))

for i in range(N):
    H[i,i]+=hbar*hbar/m/dx/dx
    if(i>0):
        H[i-1, i]-= 0.5*hbar*hbar/m/dx/dx
    if i<N-1:
        H[i+1, i]-= 0.5*hbar*hbar/m/dx/dx
print(H)

E, psi_E = scipy.linalg.eigh(H)
plt.plot(np.linspace(0,N,N)[:400], E[:400], label="Calculated energies")
plt.title("Energies for the first $400$ states")
plt.xlabel("State index")
plt.ylabel("Energy")
plt.legend()
plt.savefig("GaussianPotentialEigenvectorsEnergies.png")
plt.show()

for n in [0, 1, 2,3,4,5,6,7,8]:
    psi=psi_E[:,n]
    norm = np.sqrt(scipy.integrate.simps(psi**2, X))
    psi/=norm
    plt.plot(X, psi, label="$n=%i$" % n)

    plt.title("The gaussian potential wavefunction #" + str(n))
    plt.xlabel("$x$ position")
    plt.ylabel("Normalized wavefunction amplitude $\psi$")
    plt.grid()
    plt.legend()
    plt.savefig(str(n) + "Gaussian potential eigenvectors.png")
    plt.show()