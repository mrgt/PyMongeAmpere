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
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../lib/');

import MongeAmperePP as ma
import numpy as np
import scipy as sp
import scipy.optimize as opt

def delaunay_2(X,w=None):
    if w==None:
        w = np.zeros(X.shape[0]);
    return ma.delaunay_2(X,w);

def lloyd_2(dens,X,w=None):
    if w==None:
        w = np.zeros(X.shape[0]);
    return ma.lloyd_2(dens,X,w);

class Density_2 (ma.Density_2):
    def __init__(self, X, f=None, T=None):
        """
        This function creates a density which is piecewise linear on the
        triangulation T of X, and whose values at the vertices X is
        given by f. If f is not given, it is assumed to be constant
        and equal to one. If the triangulation is missing, the
        Delaunay triangulation of X is used. In particular, if only X
        is provided, the density is uniform on the convex hull of X
        (and vanishes outside this convex hull).

        Args: 
            X(array): input points, described by a Nx2 array
            f(array): values of the density at X, described by a Nx1 array (optional)
            T(array): triangulation of X, described by a Nx3 array of integers,
                      corresponding to row indices in X.
        """

        # by default, the density is uniform over the convex hull of X
        if f == None:
            f = np.ones(X.shape[0])
        if T == None:
            T = delaunay_2(X)
        ma.Density_2.__init__(self, X,f,T);

    @classmethod
    def from_image(cls,im,bbox=None):
    	"""
    	This function constructs a density from an image. 
    	
    	Args:
    		im: source image
    		bbox: box containing the points
    	"""
        if bbox==None:
            bbox = [-1,1,-1,1];
        h = im.shape[0];
        w = im.shape[1];
        [x,y] = np.meshgrid(np.linspace(bbox[0],bbox[1],w),
                            np.linspace(bbox[2],bbox[3],h));
        Nx = w*h;
        x = np.reshape(x,(Nx));
        y = np.reshape(y,(Nx));
        z = np.reshape(im,(Nx));
        return cls(np.vstack([x,y]).T,z)

        
    def optimized_sampling(self, N, niter=1,verbose=False):
        """
        This function constructs an optimized sampling of the density,
        combining semi-discrete optimal transport to determine the size
        of Voronoi cells with Lloyd's algorithm to relocate the points
        at the centroids.

        See: Blue Noise through Optimal Transport
             de Goes, Breeden, Ostromoukhov, Desbrun
             ACM Transactions on Graphics 31(6)

        Args:
            N: number of points in the constructed sample
            niter: number of iterations of optimal transport (default at 1)
            verbose: display informations on the iterations

        Returns:
            A numpy array with N rows and 2 columns containing the coordinates of
            the optimized sample points.
        """
        Y = self.random_sampling(N);
        nu = np.ones(N);
        nu = (self.mass() / np.sum(nu)) * nu;
        
        w = np.zeros(N);
        for i in xrange(1,5):
            Y = lloyd_2(self, Y, w)[0];
        for i in xrange(0,niter):
            if verbose:
                print "optimized_sampling, step %d" % (i+1)
            w = optimal_transport_2(self,Y,nu,verbose=verbose);
            Y = lloyd_2(self, Y, w)[0];
        return Y

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
    # solve the linear system Hs*ds = gs using a Cholesky decomposition
    ds = ma.solve_cholesky(Hs.tocoo(),gs);
    d = np.hstack((ds,[0]));
    return d;

def optimal_transport_2(dens, Y, nu, w0 = [0], eps_g=1e-7,
                        maxit=100, verbose=False):
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

    # check that eps0 is not zero, in which case the damped Newton
    # algorithm won't converges (the Hessian is non-invertible)
    assert (eps0 > 0);
    if eps0 <= 0:
        if min(m) <= 0:
            ii = np.argmin();
            raise ValueError("optimal_transport_2: minimum cell area is zero; "
                             "this is because the cell %d corresponding to the "
                             " point (%g,%g) is empty" %
                             (ii, Y[ii,0], Y[ii,1]));
        else: # in this case min(mu) == 0
            ii = np.argmin();
            raise ValueError("optimal_transport_2: minimum cell area is zero; "
                             "this is because the target cell masses vanishes, nu[%d] == 0" % ii);

    # warn the user if the smallest cell has a mass close to zero, as
    # it will likely lead to numerical instabilities.
    if eps0 < 1e-10:
        print ("optimal_transport_2: minimum cell area is small eps0=%g\n", eps0)

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
        if verbose:
            print ("it %d: f=%g |g|=%g t=%g"
                   % (it, f,np.linalg.norm(g),alpha));
        it = it+1;
    return w
