import sys
import os
sys.path.append('..');

import MongeAmpere as ma
import numpy as np
import matplotlib.pyplot as plt


def draw_laguerre_cells(dens,Y,w):
    E = dens.restricted_laguerre_edges(Y,w)
    nan = float('nan')
    N = E.shape[0]
    x = np.zeros(3*N)
    y = np.zeros(3*N)
    a = np.array(range(0,N))
    x[3*a] = E[:,0]
    x[3*a+1] = E[:,2]
    x[3*a+2] = nan
    y[3*a] = E[:,1]
    y[3*a+1] = E[:,3]
    y[3*a+2] = nan
    plt.plot(x,y,color=[1,0,0],linewidth=1,aa=True)

dens = ma.Density_2(np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.]]))
N = 40
draw_laguerre_cells(dens,np.random.rand(N,2),np.zeros(N))
plt.show()

