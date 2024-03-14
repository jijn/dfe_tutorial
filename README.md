# Driving force extension method - hands-on tutorial

This repository is used at the [ChiMaD Phase Field Workshop](https://github.com/usnistgov/pfhub/wiki#Workshops) XVI, March 19-21 2024. The goal is to provide an introduction to the driving force extension method in our recent paper:

J. Zhang, A. F. Chadwick, D. L. Chopp, P. W. Voorhees, "Phase Field Modeling with Large Driving Forces," npj Computational Materials, 9 (2023) 166. doi: [10.1038/s41524-023-01118-0](https://doi.org/10.1038/s41524-023-01118-0)

Here, a 1D KKS model is used for demonstration. Several assumptions are introduced for simplification.

## Files

- driving_force_extension.ipynb : tutorial notebook

- driving_force_extension.py : python script in case you don't have jupyter

## Requirement

You will need *Python 3* installed. 

Also, you need *numpy* and *matplotlib*. Install them by

```bash
pip install numpy matplotlib
```

To use the jupyter notebook, you need to install jupyter by
```bash
pip install jupyterlab
```

## Open the jupyter notebook

Compile the code:

```bath
jupyter-lab
```

A browser window should automatically open.

## Driving force extension method

Phase field equation typically has the following form
$$
\tau\frac{\partial \phi}{\partial t} = \kappa \nabla^2{\phi} - m g'(\phi) - p'(\phi) F
$$
where $F$ is the driving force. Phase field simulations with a large driving force and a large grid size can be unstable.

To solve this problem, the driving force extension method introduces a simple modification of the equation
$$
\tau\frac{\partial \phi}{\partial t} = \kappa \nabla^2{\phi} - m g'(\phi) - p'(\phi) \mathcal{P}(F),
$$
where $\mathcal{P}$ project $F$ to a constant perpendicular to the interface:
$$
\mathcal{P}(F(\vec{x}, t)) = F(\vec{x}_\Gamma, t).
$$
Here, $\vec{x}_\Gamma$ is the closest point on the interface $\Gamma$:
$$
\vec{x}_\Gamma(\vec{x}) = \{\vec{y} : \min_{\vec{y}\in \Gamma} |{\vec{y}-\vec{x}}|\}
$$

## Functions in the code

- dg : derivative of the double well (times 0.5)
- laplacian_1d : Laplacian (cell-based)
- apply_bc : apply boundary condition
- velext_simple : a simple velocity extension in 1D
- velext_fim : a general velocity extension in 1D
- kks1d : the KKS model
