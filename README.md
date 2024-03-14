# Driving force extension method - hands-on tutorial

This short tutorial is used at the [ChiMaD Phase Field Workshop](https://github.com/usnistgov/pfhub/wiki#Workshops) XVI, March 19-21, 2024. The goal is to provide an introduction to the driving force extension method in our recent paper:

Jin Zhang, Alexander F. Chadwick, David L. Chopp, Peter W. Voorhees, "Phase Field Modeling with Large Driving Forces," npj Computational Materials, 9 (2023) 166. doi: [10.1038/s41524-023-01118-0](https://doi.org/10.1038/s41524-023-01118-0)

Here, a 1D KKS model is used for demonstration. Several assumptions are made for simplification so that we can focus on the problem of the large driving force and the driving force extension method.

## Driving force extension method

The driving force extension method is used to solve the stability problem resulting from a large driving force in phase field modeling. 

Phase field equation typically has the following form
```math
\tau\frac{\partial \phi}{\partial t} = \kappa \nabla^2{\phi} - m g'(\phi) - p'(\phi) F,
```
where $F$ is the driving force. If the driving force is much larger than the surface energy $F \gg \sigma/l$ (here, $\sigma$ is surface energy, $l$ is diffusion interface width), phase field simulation can suffer numerical instability. To solve this problem, a smaller interface width $l$ (and hence a smaller grid size) is needed. However, this limits the system size that one can simulate.

To solve the problem of the large driving force but still use a large grid size, the driving force extension method introduces a simple modification of the phase field equation
```math
\tau\frac{\partial \phi}{\partial t} = \kappa \nabla^2{\phi} - m g'(\phi) - p'(\phi) \mathcal{P}(F),
```
where $\mathcal{P}$ projects $F$ to a constant perpendicular to the interface:
```math
\mathcal{P}(F(\vec{x}, t)) = F(\vec{x}_\Gamma, t).
```
Here, $\vec{x}_\Gamma$ is the closest point on the interface $\Gamma$:
```math
\vec{x}_\Gamma(\vec{x}) = \{\vec{y} : \min_{\vec{y}\in \Gamma} |{\vec{y}-\vec{x}}|\}
```

The algorithm for the projection $\mathcal{P}(F)$ is well-developed in the level-set community and is called velocity-extension.

## Using the code

You will need *Python 3* installed. 

Also, you need *numpy* and *matplotlib*. Install them by

```bash
pip install numpy matplotlib
```

To use the jupyter notebook, you need to install *jupyter* by
```bash
pip install jupyterlab
```

To open the jupyter notebook, run in the terminal

```bath
jupyter-lab
```

A browser window should automatically open.

### Files

- driving_force_extension.ipynb : tutorial notebook

- driving_force_extension.py : python script in case you don't have jupyter


### Functions in the code

- dg : derivative of the double well (times 0.5)
- laplacian_1d : Laplacian (cell-based)
- apply_bc : apply boundary condition
- velext_simple : a simple velocity extension in 1D
- velext_fim : a general velocity extension in 1D
- kks1d : the KKS model

## Citing our work
If you use the driving force extension method in your work, please cite our paper

> Zhang, J., Chadwick, A.F., Chopp, D.L. et al. Phase field modeling with large driving forces. npj Comput Mater 9, 166 (2023).

You can use the BibTeX

```
@article{Zhang2023,
  title={Phase field modeling with large driving forces},
  author={Zhang, Jin and Chadwick, Alexander F and Chopp, David L and Voorhees, Peter W},
  journal={npj Computational Materials},
  volume={9},
  number={1},
  pages={166},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

```

----

For questions, comments, suggestions, or bug reports, email me at [jzhang\@northwestern.edu](mailto:jzhang@northwestern.edu) or visit [github](https://github.com/jijn/dfe_tutorial).
