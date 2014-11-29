import sys
sys.path.append('..');

import MongeAmpere as ma
import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.tri as tri

n = 128;
[x,y] = np.meshgrid(np.linspace(-1,1,n),
                    np.linspace(-1,1,n));
Nx = n*n;
x = np.reshape(x,(Nx));
y = np.reshape(y,(Nx));
X = np.vstack([x,y]).T;
T = ma.delaunay_2(X,np.zeros(Nx));
img = sp.misc.lena();
#img = sp.misc.imread("calif.jpg", flatten=True);
mu = sp.misc.imresize(img, (n,n));
mu = sp.misc.imrotate(mu, 180);
mu = np.reshape(mu,(Nx));
mu = 255-np.asarray(mu, dtype=float);
dens = ma.Density_2(X,mu,T);
print(dens.mass());

N = 20000;
Y = 2*np.random.rand(N,2) - np.ones((N,2));
nu = np.ones(N);
nu = (dens.mass() / np.sum(nu)) * nu;


w = np.zeros(N);
plt.figure(figsize=(10,10),facecolor="white")
for i in xrange(1,10):
    [Z,m] = ma.lloyd_2(dens, Y, w);
    Y = Z
    w = ma.optimal_transport_2(dens,Y,nu);
    plt.cla();
    plt.gca().set_aspect('equal')
    plt.scatter(Y[:,0], Y[:,1], s=2);
    plt.pause(.01)
plt.pause(10)


