#Schrodinger 1D equation solved by finding eigenvectors of the hamiltonian
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
	return m*w*x*x/2
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
plt.plot(np.linspace(0,N,N)[:400], hbar*w*(np.linspace(0,N,N)[:400]+0.5), label="Exact harmonic oscillator result")
plt.title("Energies for the first $400$ states")
plt.xlabel("State index")
plt.ylabel("Energy")
plt.legend()
plt.savefig("harmonicOscillatorEigenvectorsEnergies.png")
plt.show()

for n in [0, 1, 2,3,4,5,6,7,8]:
    psi=psi_E[:,n]
    norm = np.sqrt(scipy.integrate.simps(psi**2, X))
    psi/=norm
    plt.plot(X, psi, label="$n=%i$" % n)

    # exact wavefunction for the harmonic oscillator
    plt.plot(X, (m*w/hbar/np.pi)**0.25/np.sqrt(2**n*scipy.misc.factorial(n))*np.exp(-m*w*X*X/2/hbar)*scipy.special.hermite(n)(np.sqrt(m*w/hbar)*X), 'x', label="exact, $n=%i$" %n)

    plt.title("The harmonic oscillator wavefunction #" + str(n))
    plt.xlabel("$x$ position")
    plt.ylabel("Normalized wavefunction amplitude $\psi$")
    plt.grid()
    plt.legend()
    plt.savefig(str(n) + "harmonicOscillatorEigenvectors.png")
    plt.show()