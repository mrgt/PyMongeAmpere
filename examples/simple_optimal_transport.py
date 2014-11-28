import pdb
import sys
sys.path.append('../lib/');

import MongeAmpere as ma
import numpy as np
import scipy as sp
import scipy.optimize as opt

# source: uniform measure on the square with sidelength 2
X = np.array([[-1,-1],
              [1, -1],
              [1, 1],
              [-1,1]], dtype=float);
mu = np.ones(4)/4;
dens = ma.Density(X,mu);

# target is a random set of points, with random weights
N = 10000;
Y = np.random.rand(N,2);
#nu = 10+np.random.rand(N);
nu = np.ones(N);
nu = (dens.mass() / np.sum(nu)) * nu;

# print "mass(nu) = %f" % sum(nu)
# print "mass(mu) = %f" % dens.mass()

# 
eps_g = 1e-7;
maxit = 1000;

if False:
    I = np.array([0,1,2,2]);
    J = np.array([0,1,1,2]);
    S = np.array([.1,.3,.1,.4]);
    h = sp.sparse.coo_matrix((S, (I,J)), shape=(3,3));
    b = np.array([.1,.1,3]);
    dd = ma.solve_spqr(h,b);
    d = sp.sparse.linalg.lsqr(h,b)[0];
    print d
    print dd
    sys.exit(0)

def newton(f, x):
    fe = lambda x: f(x)[0];  # function evaluation only
    fp = lambda x: f(x)[1];  # gradient evaluation

    for it in xrange(1,100):
        [fval,grad,hess] = f(x);
        ng = np.linalg.norm(grad);
        if ng < eps_g:
            break
        #d = ma.solve_spqr(hess,-grad);
        d = -sp.sparse.linalg.lsqr(hess,grad)[0];
        print d
        t = sp.optimize.line_search(fe, fp, x, d, grad, fval)[0];
        x = x + t*d;
        print "it %d: f=%f |g|=%f t=%f" % (it, fval, ng, t)

def optimal_transport_functional(dens,Y,nu,w):
    [f,g,h] = ma.kantorovich(dens, Y, w);
    # compute the linear part of the optimal transport functional
    # and update the gradient accordingly
    f = f - np.dot(w,nu);
    g = g - nu;
    H = sp.sparse.coo_matrix(h,shape=(N,N))
    np.dot(H,-g)
    return f,g,H;

# w = np.zeros(N);
# newton(lambda w: optimal_transport_functional(dens,Y,nu,w), w);

w = np.zeros(N);
for it in xrange(0,maxit):
    [f,g,h] = ma.kantorovich(dens, Y, w);

    # compute the linear part of the optimal transport functional
    # and update the gradient accordingly
    f = f - np.dot(w,nu);
    g = g - nu;

    H = sp.sparse.coo_matrix(h,shape=(N,N))
    d = ma.solve_spqr(H,-g)
    #d = sp.sparse.linalg.lsqr(H,-g,atol=0,btol=0,conlim=0)[0]
    ng = np.linalg.norm(g);

    if ng < eps_g:
        break

    # find suitable stepsize
    t = 1.;
    found = False;
    for i in xrange(0,10):
        ww = w + t*d;
        gg = ma.kantorovich(dens,Y,ww)[1];
        if np.amin(gg) > 1e-9:
            found = True;
            break
        # try smaller stepsize
        t = t/2 
    print "it %d: f=%f |g|=%f t=%f" % (it, f, ng, t)
    w = ww

print np.linalg.norm(g)



