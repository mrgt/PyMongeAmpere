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

import MongeAmpere as ma
import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.tri as tri

X = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float);
mu = np.ones(4);
dens = ma.Density_2(X,mu);

# target is a random set of points
N = 100;
Y = np.random.rand(N,2);
w = np.zeros(N);

for i in xrange(1,50):
    [Z,m] = dens.lloyd(Y, w);
    Y=Z;
    plt.cla();
    T = ma.delaunay_2(Y,w);
    triang = tri.Triangulation(Y[:,0], Y[:,1], T);
    plt.gca().set_aspect('equal')
    plt.triplot(triang, 'bo-')
    plt.pause(.01)
