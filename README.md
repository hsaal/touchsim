# TouchSim
A Python implementation of the TouchSim model to simulate peripheral tactile responses. For details on the model, see Saal et al., PNAS, 2017.

## Installation
The touchsim package requires Python 3.6 or higher to run. It also requires *numpy*, *scipy*, *skikit-image*, *numba*, and *matplotlib*. Additionally, *holoviews* is required to use the simulation's inbuilt plotting functions.

If using conda for package management, the following command creates a new environment *ts* with all dependencies installed:
```conda env create -f environment.yml```

To install a static version of the package, use ```python setup.py install```. To install the package in development mode (such that updates to the source directory are reflected in the installed package), use ```python setup.py develop```.

If *pytest* is installed, running `pytest` from the base directory will test whether the package is working as intended.

## Using the package
Examples of how to use the model and its plotting functions are given as Jupyter notebooks in the base directory, see the [general tutorial](./touchsim_demo.ipynb), [plotting demo](./touchsim_plotting.ipynb), and [list of overloaded functions](./touchsim_shortcuts.ipynb).
