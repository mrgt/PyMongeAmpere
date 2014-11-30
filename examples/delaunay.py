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
import matplotlib.pyplot as plt
import matplotlib.tri as tri

N = 1000;
X = np.random.rand(N,2);
w = np.zeros(N);
T = ma.delaunay_2(X,w);
triang = tri.Triangulation(X[:,0], X[:,1], T);

plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(triang, 'bo-')
plt.title('triplot of Delaunay triangulation')
plt.show();
#print(T);
