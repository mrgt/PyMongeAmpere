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

# FIXME: we need to find a nice way to detect the path to MongeAmperePP
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../lib/');
sys.path.append('../lib/');

import MongeAmperePP as ma
import numpy as np
import numpy.matlib
import scipy as sp
import scipy.optimize as opt
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def delaunay_2(X,w=None):
    if w is None:
        w = np.zeros(X.shape[0]);
    return ma.delaunay_2(X,w);

def kantorovich_2(dens,Y,nu,w):
    N = len(nu);
    [f,m,h] = ma.kantorovich_2(dens, Y, w);
    # compute the linear part of the optimal transport functional
    # and update the gradient accordingly
    f = f - np.dot(w,nu);
    g = m - nu;
    H = sp.sparse.csr_matrix(h,shape=(N,N))
    return f,m,g,H;

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
        if f is None:
            f = np.ones(X.shape[0])
        if T is None:
            T = delaunay_2(X)
        self.vertices = X.copy()
        self.triangles = T.copy()
        self.values = f.copy()            
        ma.Density_2.__init__(self, X,f,T);

    @classmethod
    def from_image(cls,im,bbox=None):
    	"""
    	This function constructs a density from an image. 
    	
    	Args:
    		im: source image
    		bbox: box containing the points
    	"""
        if bbox is None:
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

    def kantorovich(self,Y,nu,w):
        N = len(nu);
        [f,m,h] = ma.kantorovich_2(self, Y, w);
        # compute the linear part of the optimal transport functional
        # and update the gradient accordingly
        f = f - np.dot(w,nu);
        g = m - nu;
        H = sp.sparse.csr_matrix(h,shape=(N,N))
        return f,m,g,H;

    def lloyd(self,X,w=None):
        if w is None:
            w = np.zeros(X.shape[0]);
        return ma.lloyd_2(self,X,w);

    def moments(self,X,w=None):
        if w is None:
            w = np.zeros(X.shape[0]);
        return ma.moments_2(self,X,w);
        
class Periodic_density_2 (ma.Density_2):
    def __init__(self, bbox):
        self.x0 = np.array([bbox[0],bbox[1]]);
        self.x1 = np.array([bbox[2],bbox[3]]);
        self.u = self.x1 - self.x0;
        X = np.array([[bbox[0],bbox[1]],
                      [bbox[0],bbox[3]],
                      [bbox[2],bbox[1]],
                      [bbox[2],bbox[3]]])
        f = np.ones(X.shape[0])
        T = delaunay_2(X)
        ma.Density_2.__init__(self, X,f,T)

    def to_fundamental_domain(self,Y):
        N = Y.shape[0];
        Y = (Y - np.tile(self.x0,(N,1))) / np.tile(self.u,(N,1)); 
        Y = Y - np.floor(Y);
        Y = np.tile(self.x0,(N,1)) + Y * np.tile(self.u,(N,1));
        return Y;

    # FIXME
    def kantorovich(self,Y,nu,w):
        N = len(nu);

        # create copies of the points, so as to cover the neighborhood
        # of the fundamental domain.
        Y0 = self.to_fundamental_domain(Y)
        x = self.u[0]
        y = self.u[1]
        v = np.array([[0,0],
                      [x,0], [-x,0], [0,y], [0,-y],
                      [x,y], [-x,y], [x,-y], [-x,-y]]);
        Yf = np.zeros((9*N,2))
        wf = np.hstack((w,w,w,w,w,w,w,w,w));
        for i in xrange(0,9):
            Nb = N*i; Ne = N*(i+1)
            Yf[Nb:Ne,:] = Y0 + np.tile(v[i,:],(N,1))

        # sum the masses of each "piece" of the Voronoi cells
        [f,mf,hf] = ma.kantorovich_2(self, Yf, wf);

        m = np.zeros(N);
        for i in xrange(0,9):
            Nb = N*i; Ne = N*(i+1);
            m += mf[Nb:Ne]

        # adapt the Hessian by correcting indices of points. we use
        # the property that elements that appear multiple times in a
        # sparse matrix are summed
        h = (hf[0], (np.mod(hf[1][0], N), np.mod(hf[1][1], N)))

        # remove the linear part of the function
        f = f - np.dot(w,nu);
        g = m - nu;
        H = sp.sparse.csr_matrix(h,shape=(N,N))
        return f,m,g,H;

    def lloyd(self,Y,w=None):
        if w is None:
            w = np.zeros(Y.shape[0]);
        N = Y.shape[0];
        Y0 = self.to_fundamental_domain(Y)

        # create copies of the points, so as to cover the neighborhood
        # of the fundamental domain.
        x = self.u[0]
        y = self.u[1]
        v = np.array([[0,0],
                      [x,0], [-x,0], [0,y], [0,-y],
                      [x,y], [-x,y], [x,-y], [-x,-y]]);
        Yf = np.zeros((9*N,2))
        wf = np.hstack((w,w,w,w,w,w,w,w,w));
        for i in xrange(0,9):
            Nb = N*i; Ne = N*(i+1)
            Yf[Nb:Ne,:] = Y0 + np.tile(v[i,:],(N,1))

        # sum the moments and masses of each "piece" of the Voronoi
        # cells
        [mf,Yf,If] = ma.moments_2(self, Yf, wf);

        Y = np.zeros((N,2));
        m = np.zeros(N);
        for i in xrange(0,9):
            Nb = N*i; Ne = N*(i+1);
            m += mf[Nb:Ne]
            ww = np.tile(mf[Nb:Ne],(2,1)).T
            Y += Yf[Nb:Ne,:] - ww * np.tile(v[i,:],(N,1))

        # rescale the moments to get centroids
        Y /= np.tile(m,(2,1)).T
        #Y = self.to_fundamental_domain(Y);
        return (Y,m)

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
	# if no initial guess is provided, start with a pre-estimated solution.
	# Then compute function value, gradient and hessian
    N = Y.shape[0];
    if len(w0) == N:
        w = w0;
    else:
        w = np.zeros(N);
    
    [f,m,g,H] = dens.kantorovich(Y, nu, w);

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
            raise ValueError("optimal_transport_2: minimum cell area is"
                             "zero; this is because the cell %d "
                             " corresponding to the point (%g,%g) is "
                             " empty" % (ii, Y[ii,0], Y[ii,1]));
        else: # in this case min(mu) == 0
            ii = np.argmin();
            raise ValueError("optimal_transport_2: minimum cell area "
                             "is zero; "
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
            [f,m,g,H] = dens.kantorovich(Y, nu, w);
            if (min(m) >= eps0 and
                np.linalg.norm(g) <= (1-alpha/2)*n0):
                break;
            alpha *= .5;
        if verbose:
            print ("it %d: f=%g |g|=%g t=%g"
                   % (it, f,np.linalg.norm(g),alpha));
        it = it+1;
    return w

from PIL import Image
def to_grayscale(I):
    I8 = np.minimum(255.0*I, 255).astype(np.uint8)
    return Image.fromarray(I8)
def to_rgb(R,G,B):
    R8 = np.minimum(255.0*R, 255).astype(np.uint8)
    G8 = np.minimum(255.0*G, 255).astype(np.uint8)
    B8 = np.minimum(255.0*B, 255).astype(np.uint8)
    return Image.fromarray(np.dstack((R8,G8,B8)))

def laguerre_diagram_to_image(dens, Y, w, colors, bbox, ww, hh):
    nc = colors.shape[1]
    A = ma.rasterize_2(dens, Y, w, colors, bbox[0], bbox[1], bbox[2], bbox[3], ww, hh);

    if (nc == 1):
        img = to_grayscale(A[0].T)
        return img
    elif (nc == 3):
        img = to_rgb(A[0].T,A[1].T,A[2].T)
    else:
        raise ValueError("laguerre_diagram_to_image: number of color channels should be 1 or 3")
    return img

        
def to_grayscale(I):
    I8 = np.minimum(255.0*I, 255).astype(np.uint8)
    return Image.fromarray(I8)
def to_rgb(R,G,B):
    R8 = np.minimum(255.0*R, 255).astype(np.uint8)
    G8 = np.minimum(255.0*G, 255).astype(np.uint8)
    B8 = np.minimum(255.0*B, 255).astype(np.uint8)
    return Image.fromarray(np.dstack((R8,G8,B8)))

def optimized_sampling_2(dens, N, niter=1,verbose=False):
    """
    This functions constructs an optimized sampling of the density,
    combining semi-discrete optimal transport to determine the size
    of Voronoi cells with Lloyd's algorithm to relocate the points
    at the centroids.

    See: Blue Noise through Optimal Transport
         de Goes, Breeden, Ostromoukhov, Desbrun
         ACM Transactions on Graphics 31(6)

    Args:
        N: number of points in the constructed sample
        niter: number of iterations of optimal transport (default: 1)
        verbose: display informations on the iterations
    
    Returns:
        A numpy array with N rows and 2 columns containing the
        coordinates of the optimized sample points.
    """
    Y = dens.random_sampling(N);
    nu = np.ones(N);
    nu = (dens.mass() / np.sum(nu)) * nu;
        
    w = np.zeros(N);
    for i in xrange(1,5):
        Y = dens.lloyd(Y, w)[0];
    for i in xrange(0,niter):
        if verbose:
            print "optimized_sampling, step %d" % (i+1)
        w = optimal_transport_2(dens,Y,nu,verbose=verbose);
        Y = dens.lloyd(Y, w)[0];
    return Y
    
def optimal_transport_presolve_2(Y, X, Y_w=None, X_w=None):
	"""
	This function calculates first estimation of the potential.
	
	Parameters
	----------
	Y : 2D array
		Target samples
	Y_w : 1D array
		Weights associated to Y
	X : 2D array
		Source samples
	X_w : 1D array
		Weights asociated to X
		
	Returns
	-------
	psi0 : 1D array
		Convex estimation of the potential. Its gradient
		send Y convex hull into X convex hull.	
	"""
	
	if X_w is None:
		X_w = np.ones(X.shape[0])
	if Y_w is None:
		Y_w = np.ones(Y.shape[0])
	
	bary_X = np.average(X,axis=0,weights=X_w)
	bary_Y = np.average(Y,axis=0,weights=Y_w)
	
	# Y circum circle radius centered on bary_Y
	r_Y = furthest_point(Y, bary_Y)

	X_hull = ConvexHull(X)
	points = X_hull.points
	simplices = X_hull.simplices
	
	# Search of the largest inscribed circle
	# centered on Y barycentre
	dmin = distance_point_line(points[simplices[0][0]], points[simplices[0][1]], bary_X)
	for simplex in simplices:
		d = distance_point_line(points[simplex[0]], points[simplex[1]], bary_X)
		if d < dmin:
			dmin = d
	# Y inscribed circle radius centered on bary_Y		
	r_X = dmin
	
	ratio = r_X / r_Y

	psi_tilde0 = 0.5 * ratio * (np.power(Y[:,0]-bary_Y[0],2)+np.power(Y[:,1]-bary_Y[1],2)) + bary_X[0]*(Y[:,0]) + bary_X[1]*(Y[:,1])

	psi0 = np.power(Y[:,0],2) + np.power(Y[:,1],2) - 2*psi_tilde0
	
	return psi0

	
def distance_point_line(m, n, pt):
	"""
	Computes the distance between the line generated 
	by segment MN and the point pt, in a space of arbitrary
	dimension dim.
	
	Parameters
	----------
	m : array (dim,)
		Point generating the line
	n : array (dim,)
		Second point generating the line
	pt : array (dim,)
		Point we calculate the distance from the line
		
	Returns
	-------
	dist : real
		distance between line (MN) and pt
	
	"""
	u = n - m			# Direction vector
	Mpt = pt - m
	norm_u = np.linalg.norm(u)
	dist = np.linalg.norm(Mpt - (np.inner(Mpt,u)/(norm_u*norm_u))*u)
	return dist
		

def furthest_point(cloud, a):
	"""
	Computes the distance between a and the furthest point
	in cloud, in a space of arbitrary dimension dim.
	
	Parameters
	----------
	cloud : (N,dim) array
		Point cloud
	a : (,dim) array
		Point
	
	Returns
	-------
	distance : real
		distance between a and the furthest point in cloud.
	"""
	assert(a.shape[0] == cloud.shape[1])
	N = np.shape(cloud)[0]
	dim = np.shape(cloud)[1]
	tmp = np.matlib.repmat(a,N,1)
	dist = np.linalg.norm(tmp-cloud, axis=1)
	return np.max(dist)
