import sys
sys.path.append('..');

import MongeAmpere as ma
import numpy as np

# source: uniform measure on the square with sidelength 2
X = np.array([[-1,-1],
              [1, -1],
              [1, 1],
              [-1,1]], dtype=float);
T = ma.delaunay_2(X,np.zeros(4));
print T
mu = np.ones(4)/4;
dens = ma.Density_2(X,mu,T);
print "mass=%g"%dens.mass()

# target is a random set of points, with random weights
N = 1000;
Y = np.random.rand(N,2)/2;
nu = 10+np.random.rand(N);
nu = (dens.mass() / np.sum(nu)) * nu;

# print "mass(nu) = %f" % sum(nu)
# print "mass(mu) = %f" % dens.mass()

# 
ma.optimal_transport_2(dens,Y,nu)
