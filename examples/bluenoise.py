# PyMongeAmpere
# Copyright (C) 2014 Quentin Mérigot, CNRS
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

n = 256;
[x,y] = np.meshgrid(np.linspace(-1,1,n),
                    np.linspace(-1,1,n));
Nx = n*n;
x = np.reshape(x,(Nx));
y = np.reshape(y,(Nx));
X = np.vstack([x,y]).T;
T = ma.delaunay_2(X,np.zeros(Nx));

# load image, and resize/fix it for the stippling procedure
if len(sys.argv) > 1:
    name = sys.argv[1]
    cloudname = os.path.splitext(name)[0] + '.cloud';
    img = sp.misc.imread(name, flatten=True);
else:
    cloudname = "lena.cloud";
    img = sp.misc.lena();
mu = sp.misc.imresize(img, (n,n));
mu = sp.misc.imrotate(mu, 180);
mu = np.reshape(mu,(Nx));
mu = 255-np.asarray(mu, dtype=float);
dens = ma.Density_2(X,mu,T);

N = 40000;
Y = dens.random_sampling(N);
nu = np.ones(N);
nu = (dens.mass() / np.sum(nu)) * nu;


# smoothen point cloud
w = np.zeros(N);
for i in xrange(1,5):
    [Z,m] = ma.lloyd_2(dens, Y, w);
    Y = Z

ww=10
plt.figure(figsize=(ww,ww),facecolor="white")
surf=(ww*100)*(ww*100);
# N*s/surf = dens.mass()/2^2
avg_gray = dens.mass()/(4*255);
print avg_gray
S = (surf/N)*avg_gray
S = S/4
print S

for i in xrange(1,5):
    w = ma.optimal_transport_2(dens,Y,nu);
    [Z,m] = ma.lloyd_2(dens, Y, w);
    Y = Z
    plt.cla();
    plt.gca().set_aspect('equal')
    plt.scatter(Y[:,0], Y[:,1], s=S);
    plt.axis([-1,1,-1,1])
    plt.axis('off')
    plt.pause(.01)
plt.pause(10)

# np.savetxt(cloudname, Y, fmt='%g')
