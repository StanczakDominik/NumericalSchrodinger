# Schrodinger 1D equation solved by finding eigenvectors of the hamiltonian
import numpy as np
import scipy
import scipy.integrate
import scipy.linalg
import scipy.misc
import scipy.special
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

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


N_lines=10
psi_list = psi_E[:, :N_lines]
E_list = E[:N_lines]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lines=[ax.plot(X, np.zeros_like(X), np.zeros_like(X))[0] for n in range(N_lines)]

def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return lines

frame_number=5000

def animate(i, lines, dummy):
    t=4*np.pi*i/frame_number
    for index in range(N_lines):
        line=lines[index]
        psi = psi_list[:, index]
        psi_r, psi_i = psi_t_parts(psi, E_list[index], t)
        line.set_data(X,psi_r)
        line.set_3d_properties(psi_i)
    return lines
ax.set_xlabel("$x$ position")
ax.set_ylabel("Real part")
ax.set_zlabel("Imaginary part")
ax.set_xlim3d(-l, l)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)
ax.grid()
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frame_number, fargs=(lines, "dummy"), interval=10, blit=True)

#saves mp4 file
#mywriter = animation.MencoderWriter()
#anim.save('wavefunction_animation.mp4', writer=mywriter, extra_args=['-vcodec', 'libx264'])
plt.show()