# run via python 2.7

#solves infinite well schrodinger equation in the time domain
from __future__ import division
import numpy as np

from numpy import copy

def banded(Aa,va,up,down):

    # Copy the inputs and determine the size of the system
    A = copy(Aa)
    v = copy(va)
    N = len(v)

    # Gaussian elimination
    for m in range(N):

        # Normalization factor
        div = A[up,m]

        # Update the vector first
        v[m] /= div
        for k in range(1,down+1):
            if m+k<N:
                v[m+k] -= A[up+k,m]*v[m]

        # Now normalize the pivot row of A and subtract from lower ones
        for i in range(up):
            j = m + up - i
            if j<N:
                A[i,j] /= div
                for k in range(1,down+1):
                    A[i+k,j] -= A[up+k,m]*A[i,j]

    # Backsubstitution
    for m in range(N-2,-1,-1):
        for i in range(up):
            j = m + up - i
            if j<N:
                v[m] -= A[i,j]*v[j]

    return v

from visual import rate, curve, display, color

N=1000
m=9.109e-31
L=1e-8
hbar=1.054571726e-34
h=1e-18
a=L/N
a1=1+1j*h*hbar/(2.*m*a*a)
a2=-1j*h*hbar/(4.*m*a*a)
b1=1-1j*h*hbar/(2.*m*a*a)
b2=1j*h*hbar/(4.*m*a*a)

A=np.zeros([3,N], complex)
A[0,:]=a2
A[1,:]=a1
A[2,:]=a2
x=np.arange(0,L,a)

x0=0.5*L
sigma=1e-10
kappa=5e10
phi=np.exp(-(x-x0)**2/(2.*sigma**2))*np.exp(1j*kappa*x)

scale=1e-9
display(center=[L/2, 0,0])
krzywa = np.zeros([N,3])
krzywa[:,0] = x

krzywa[:,1] = phi.real*scale
krzywa[:,2] = phi.imag*scale
real=np.zeros([N,3])
real[:,0]=x
real[:,1]=phi.real*scale
#real[:,1]=abs(phi)*abs(phi)*scale          #probability distribution (non-normalized)
real[:,2]=0

imag=np.zeros([N,3])

imag[:,0]=x
imag[:,1]=0
imag[:,2]=phi.imag*scale
l=curve(pos=krzywa,radius=0.01*scale)
lr=curve(pos=real,radius=0.0025*scale, color=color.blue)
li=curve(pos=imag,radius=0.0025*scale, color=color.orange)
def UpdateCurve():
    global l, krzywa, lp, real, li, imag
    krzywa[:,1] = real[:,1]= phi.real*scale
    #real[:,1]=abs(phi)*abs(phi)*scale
    krzywa[:,2] = imag[:,2] = phi.imag*scale
    l.pos=krzywa
    lr.pos=real
    li.pos=imag

def CrankNicolsonStep(phi):
    v=np.zeros([N], complex)
    v[0]=b1*phi[0]+b2*phi[1]
    v[N-1]=b1*phi[N-2]+b2*phi[N-1]
    for i in range(1,N-2):
        v[i]=b1*phi[i]+b2*(phi[i+1]+phi[i-1])
    result = banded(A,v,1,1)
    return result


while(1):
    phi=CrankNicolsonStep(phi)
    UpdateCurve()
    rate(30)
