import sys
sys.path.append('../lib/');

import MongeAmperePP as ma
import numpy as np
import scipy as sp
import scipy.optimize as opt
from MongeAmperePP import Density_2, lloyd_2, delaunay_2

def kantorovich_2(dens,Y,nu,w):
    N = len(nu);
    [f,m,h] = ma.kantorovich_2(dens, Y, w);
    # compute the linear part of the optimal transport functional
    # and update the gradient accordingly
    f = f - np.dot(w,nu);
    g = m - nu;
    H = sp.sparse.csr_matrix(h,shape=(N,N))
    return f,m,g,H;


# The hessian of Kantorovich's functional is not invertible, because
# its kernel contains constant vectors. Removing the last line and
# column of the matrix before performing the resolution of the linear
# system solves this issue.
def solve_graph_laplacian(H,g):
    N = len(g);
    Hs = H[0:(N-1),0:(N-1)]
    gs = g[0:N-1]
    ds = sp.sparse.linalg.spsolve(Hs,gs);
    d = np.hstack((ds,[0]));
    return d;

def optimal_transport_2(dens, Y, nu, w0 = [0], eps_g=1e-7, maxit=100):
    # if no initial guess is provided, start with zero, and compute
    # function value, gradient and hessian
    N = Y.shape[0];
    if len(w0) == N:
        w = w0;
    else:
        w = np.zeros(N);
    [f,m,g,H] = kantorovich_2(dens, Y, nu, w);

    # we impose a minimum weighted area for Laguerre cells during the
    # execution of the algorithm:
    # eps0 = min(minimum of cells areas at beginning,
    #           minimum of target areas).
    eps0 = min(min(m),min(nu))/2;
    it = 0;

    print eps0
    while (np.linalg.norm(g) > eps_g and it <= maxit):
        d = solve_graph_laplacian(H,-g)

        # choose the step length by a simple backtracking, ensuring
        # the invertibility (up to the invariance under the addition
        # of a constant) of the hessian at the next point
        alpha = 1;
        w0 = w;
        n0 = np.linalg.norm(g);
        nlinesearch = 0;
        while True:
            w = w0 + alpha * d;
            [f,m,g,H] = kantorovich_2(dens, Y, nu, w);
            if (min(m) >= eps0 and
                np.linalg.norm(g) <= (1-alpha/2)*n0):
                break;
            alpha *= .5;
        print ("it %d: f=%g |g|=%g t=%g"
               % (it, f,np.linalg.norm(g),alpha));
        it = it+1;
    return w
