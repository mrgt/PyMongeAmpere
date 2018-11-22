# PyMongeAmpere
Python interface for a 2D optimal transport / Monge Ampère solver using Laguerre diagrams

## Getting started

First of all, you have to download [MongeAmpere](https://github.com/mrgt/MongeAmpere). Note that it's not necessary to compile it. Then go into the same directory as MongeAmpere and type in a terminal:

``` sh
git clone https://github.com/mrgt/MongeAmpere.git
git clone https://github.com/mrgt/PyMongeAmpere.git
cd PyMongeAmpere && git submodule update --init --recursive
```

### Dependencies
+ Eigen3
+ CImg
+ CGAL 4.11
+ Python 2.7
+ Boost-python
+ NumPy
+ Scipy
+ Matplotlib
+ Pillow (not in Homebrew, install via pip)
Some help on installing them can be found [here](https://github.com/mrgt/PyMongeAmpere/wiki/InstallingDependencies)

### Compiling
Once you have installed all the dependencies, building is rather straightforward. Create a out-of-source build folder and then configure (using cmake) and build :

``` sh
cd /path/to/PyMongeAmpere/..
mkdir PyMongeAmpere-build && cd PyMongeAmpere-build
cmake ../PyMongeAmpere
make
```

NB: If you encounter problem during the building step, you can use the ccmake tool, enabling you to modify easily the paths to the dependencies and the compilation options. On OSX, linking the X11 librairies may be a problem during the compilation step, you must then check (with the toggle command in ccmake) that all the X11 related paths to include and libs are the same (something like /usr/local/... or /opt/X11/).

## Examples

The directory `examples/` contains several examples on how to use the PyMongeAmpere API. To check that everything works properly you can try the following command, which by default produces and displays an optimized sampling of the scipy.misc.ascent() picture:

```sh
cd examples
python /path/to/PyMongeAmpere-build/bluenoise.py
```
