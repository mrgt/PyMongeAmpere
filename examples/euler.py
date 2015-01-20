import sys
sys.path.append('..');

import MongeAmpere as ma
import numpy as np
import scipy as sp
import scipy.optimize as opt
import scipy.spatial as spt
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.tri as tri

def segments(X,Y,color=(1,0,0,0)):
    N = X.shape[0]
    z = np.zeros((N,2,2));
    z[:,0,:] = X
    z[:,1,:] = Y
    lc = LineCollection(z)
    lc.set_color(color)
    plt.gca().add_collection(lc)

def project_on_incompressible(dens,Z):
    N = Z.shape[0]
    nu = np.ones(N) * dens.mass()/N;
    w = ma.optimal_transport_2(dens, Z, nu, verbose=True)
    return dens.lloyd(Z,w)[0];

def project_mean_on_incompressible(dens, X,Y):
    return project_on_incompressible(dens,(X+Y)/2)

def projection_smooth(dens,Xt):
    nt = len(Xt)
    for i in xrange(1,nt-1):
        Xt[i] = project_mean_on_incompressible(dens,Xt[i-1],Xt[i+1])
    return Xt
def gen_disk(k):
    t = np.linspace(0,2*np.pi,k+1);
    t = t[0:k]
    return np.vstack([np.cos(t),np.sin(t)]).T;

# k = log_2(n time steps)
# nsmooth = number of smoothing steps after adding intermediate timesteps
def euler_solve(dens,X,Y,k=2,nsmooth=1):
    Xt = [X,Y]
    for j in xrange(k):
        Yt = list()
        print("ADDING TIMESTEPS")
        for i in xrange(len(Xt)-1):
            Yt.append(Xt[i])
            Yt.append(project_mean_on_incompressible(dens,Xt[i],Xt[i+1]))
        Yt.append(Xt[-1]);
        print("SMOOTHING")
        Xt = Yt
        for i in xrange(nsmooth):
            Xt = projection_smooth(dens,Xt)
    return Xt
