# fiberoripy

This python package provides basic functionality and tools for fiber orientations and
closure models.

## Installation

```
pip install fiberoripy              # library
pip install "fiberoripy[examples]"  # plus interactive plotting for the notebooks
```

## Quickstart

```python
import numpy as np
from scipy.integrate import solve_ivp

from fiberoripy.closures import IBOF_closure, compute_closure
from fiberoripy.orientation import folgar_tucker_ode, integrate_ori_ode

# Close a second-order orientation tensor to fourth order.
A = compute_closure(np.diag([0.7, 0.2, 0.1]), "IBOF")


# Evolve an orientation state in simple shear.
def L(t):
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


t = np.linspace(0.0, 100.0, 500)
solution = solve_ivp(
    integrate_ori_ode,
    (t.min(), t.max()),
    (np.eye(3) / 3.0).ravel(),
    t_eval=t,
    args=(L, IBOF_closure, folgar_tucker_ode, {"xi": 1.0, "Ci": 0.01}),
)
a = solution.y.T.reshape(-1, 3, 3)
```

Closures also accept stacked input of shape `(N, 3, 3)` or `(N, 3, 3, 3, 3)`.

## Citing

Please cite the archived release,
[10.5281/zenodo.4679755](https://doi.org/10.5281/zenodo.4679755).
