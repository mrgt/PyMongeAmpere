import sys
sys.path.append('..');

import MongeAmpere as ma
import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.tri as tri

bbox = [0.,0.,1.,1.];
dens = ma.Periodic_density_2(bbox);

N = 10000;
Z = dens.random_sampling(N)
nu = np.ones(N)/N;

dt = 0.2
t = dt
for i in xrange(1,50):
    print("ITERATION %d" % i)
    w = ma.optimal_transport_2(dens, Z, nu, verbose=True)
    C = dens.lloyd(Z, w)[0]
    Z = dens.to_fundamental_domain(Z+(dt/t)*(Z-C))
    t = t+dt
    plt.cla()
    plt.scatter(Z[:,0], Z[:,1], color='blue', s=1);
    plt.axis([0,1,0,1])
    plt.pause(.5);

