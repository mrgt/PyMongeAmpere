# PyMongeAmpere
# Copyright (C) 2014 Quentin Merigot, CNRS
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import sys
sys.path.append('..');
import os
import MongeAmpere as ma
import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# generate uniform points in the disk, and then a Gaussian density
Nx = 30;
t = np.linspace(0,2*np.pi,Nx+1);
t = t[0:Nx]
disk = np.vstack([np.cos(t),np.sin(t)]).T;
X = ma.optimized_sampling_2(ma.Density_2(disk), 1000, verbose=True);
sigma = 7;
mu = np.exp(-sigma*(np.power(X[:,0],2) + np.power(X[:,1],2)));
dens = ma.Density_2(X,mu);

N = 5000;
Y = dens.random_sampling(N);
nu = np.ones(N);
nu = (dens.mass() / np.sum(nu)) * nu;

w = np.zeros(N);
for i in xrange(1,5):
    [Z,m] = dens.lloyd(Y, w);
    Y = Z;

plt.figure(figsize=(10,10),facecolor="white")
for i in xrange(1,10):
    w = ma.optimal_transport_2(dens,Y,nu);
    [Z,m] = dens.lloyd(Y, w);
    Y = Z
    plt.cla();
    plt.gca().set_aspect('equal')
    plt.scatter(Y[:,0], Y[:,1], s=1);
    plt.axis([-1,1,-1,1])
    plt.axis('off')
    plt.pause(.01)
plt.pause(10)

# np.savetxt(cloudname, Y, fmt='%g')
