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
