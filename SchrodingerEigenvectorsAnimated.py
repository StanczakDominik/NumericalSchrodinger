# Schrodinger 1D equation solved by finding eigenvectors of the hamiltonian
import numpy as np
import scipy
import scipy.integrate
import scipy.linalg
import scipy.misc
import scipy.special
import pylab as plt

l = 10
N = 1024


def v(x):
    return m * w * x * x / 2


m = 1
w = 1
hbar = 1
X, dx = np.linspace(-l, l, N, retstep=True)

H = np.diag(v(X))

for i in range(N):
    H[i, i] += hbar * hbar / m / dx / dx
    if i > 0:
        H[i - 1, i] -= 0.5 * hbar * hbar / m / dx / dx
    if i < N - 1:
        H[i + 1, i] -= 0.5 * hbar * hbar / m / dx / dx

E, psi_E = scipy.linalg.eigh(H)
E, psi_E = E[:100], psi_E[:, :100]
for n in range(100):
    psi = psi_E[:, n]
    norm = np.sqrt(scipy.integrate.simps(psi ** 2, X))
    psi /= norm


def psi_t_parts(psi_input, energy, t_input):
    psi_complex = psi_input * np.exp(-1j * energy * t_input / hbar)
    return np.real(psi_complex), np.imag(psi_complex)


psi_list = psi_E[:, :5]
E_list = E[:5]

for t in range(10):
    t /= 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    psi_total_real = np.zeros_like(X)
    psi_total_imag = np.zeros_like(X)
    for index in range(5):
        psi = psi_list[:, index]
        psi_r, psi_i = psi_t_parts(psi, E_list[index], t)
        psi_total_real += psi_r
        psi_total_imag += psi_i
        ax.plot(X, psi_r, psi_i, label="$n=%i$" % index)
    ax.plot(X, psi_r, psi_i, label="full wavefunction")
    ax.set_xlabel("$x$ position")
    ax.set_ylabel("Real part")
    ax.set_zlabel("Imaginary part")
    ax.set_xlim3d(-l, l)
    ax.grid()
    ax.legend()
    plt.show()
