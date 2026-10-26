# TouchSim
A Python implementation of the TouchSim model to simulate peripheral tactile responses. For details on the model, please see [Saal, Delhaye et al., PNAS, 2017](https://doi.org/10.1073/pnas.1704856114).

## Try it online

Click on the badge below to open a fully functional [tutorial notebook](./touchsim_demo.ipynb) in your browser using myBinder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hsaal/touchsim/master?filepath=touchsim_demo.ipynb)

*Note:* The notebook usually starts in less than 30 seconds, however it can take around 10 minutes if the repository has recently been updated and a new environment needs to be created (evident if the log appears stuck on 'Solving environment').

## Installation
The touchsim package requires Python 3.9 or higher to run. It also requires *numpy*, *scipy*, *scikit-image*, *numba*, and *matplotlib*.

If using conda for package management, the following command creates a new environment *ts* with all dependencies installed:
```conda env create -f environment.yml```

To install the latest version directly from GitHub, use
```pip install git+https://github.com/hsaal/touchsim```

Alternatively, from a local copy of the repository, run `pip install .` from the base directory. To install the package in development (editable) mode, such that updates to the source directory are reflected in the installed package, use `pip install -e .` instead.

If *pytest* is installed, running `pytest` from the base directory will test whether the package is working as intended.

## Using the package
Examples of how to use the model and its plotting functions are given as Jupyter notebooks in the base directory, see the [general tutorial](./touchsim_demo.ipynb), [plotting demo](./touchsim_plotting.ipynb), and [list of overloaded functions](./touchsim_shortcuts.ipynb).

## Citation
If you use TouchSim in your research, please cite:

> Saal HP, Delhaye BP, Rayhaun BC, Bensmaia SJ (2017). Simulating tactile signals from the whole hand with millisecond precision. *Proceedings of the National Academy of Sciences* 114(28): E5693–E5702. https://doi.org/10.1073/pnas.1704856114

## License
TouchSim is released under the [MIT License](./LICENSE).
