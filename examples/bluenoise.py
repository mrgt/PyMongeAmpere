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
import os
import MongeAmpere as ma
import numpy as np
import scipy as sp
import scipy.misc
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# load image, and resize/fix it for the stippling procedure
if len(sys.argv) > 1:
    name = sys.argv[1]
    cloudname = os.path.splitext(name)[0] + '.cloud';
    img = sp.misc.imread(name, flatten=True);
else:
    cloudname = "ascent.cloud";
    img = sp.misc.ascent();

n = 256;
mu = sp.misc.imresize(img, (n,n));
mu = sp.misc.imrotate(mu, 180);
mu = 255-np.asarray(mu, dtype=float);
dens = ma.Density_2.from_image(mu,[0,1,0,1]);

N = 20000;
Y = dens.random_sampling(N);
nu = np.ones(N);
nu = (dens.mass() / np.sum(nu)) * nu;

# smoothen point cloud
w = np.zeros(N);
for i in xrange(1,3):
    [Z,m] = dens.lloyd(Y, w);
    Y = Z

# Set image to full window
ww=10
plt.figure(figsize=(ww,ww),facecolor="white")
ax = plt.Axes(plt.gcf(),[0,0,1,1],yticks=[],xticks=[],frame_on=False)
plt.gcf().delaxes(plt.gca())
plt.gcf().add_axes(ax)

# Compute surface area of each dot, so that the total amount of gray
# is coherent with the gray level of the picture
#
# N*s/surf = dens.mass(), where surf is the area of the image in points
surf=(ww*72)*(ww*72); # 72 points per inch
avg_gray = dens.mass()/255;
S = (surf/N)*avg_gray

for i in xrange(1,5):
    w = ma.optimal_transport_2(dens,Y,nu,verbose=True);
    [Z,m] = dens.lloyd(Y, w);
    Y = Z
    plt.cla();
    plt.gca().set_aspect('equal')
    plt.scatter(Y[:,0], Y[:,1], s=S);
    plt.axis([0,1,0,1])
    plt.axis('off')
    plt.pause(.01)
plt.pause(10)

np.savetxt(cloudname, Y, fmt='%g')
