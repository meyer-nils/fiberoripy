# Contributing to fiberoripy

Bug reports, new models and closures, and documentation fixes are all welcome.

## Setup

```
git clone https://github.com/meyer-nils/fiberoripy.git
cd fiberoripy
pip install -e ".[test,examples]"
pre-commit install
pytest
```

`pre-commit run --all-files` checks the whole repository. CI runs the test suite on
Python 3.11 to 3.14.

Documentation is built by `.github/workflows/docs.yml`; its API reference comes from
the docstrings, so new models and closures appear on their own. Preview it locally
with `pip install -e ".[docs]"` and `zensical serve -o`.

## Conventions

- Linting and formatting are handled by [ruff](https://docs.astral.sh/ruff/)
  (88 characters). Notebook outputs are stripped by `nbstripout`.
- [numpydoc](https://numpydoc.readthedocs.io/) docstrings, since the API reference is
  generated from them. Include a `References` section with a DOI for published work.
- Orientation models use the signature `model(a, A, D, W, xi, **kwargs)` and return
  the rate of the second-order tensor, so they work with `integrate_ori_ode`.
- Closures must accept a stack of tensors, i.e. shape `(N, 3, 3)` or
  `(N, 3, 3, 3, 3)`. Use `...` subscripts in `np.einsum` so leading batch dimensions
  broadcast, and attach the ellipsis to the *tensor* operand, never to a fixed one
  such as `np.eye(3)`.

## Adding a closure

1. Implement it in `fiberoripy/closures.py`.
2. Add its name to `_FOT2_CLOSURES` or `_FOT4_CLOSURES` and to the matching dispatch.
3. Add it to the parametrised lists in `tests/test_closure.py`, batched tests included.
4. List it in the README under the exact name the code accepts.

When adding an example notebook, add a card for it to `docs/examples.md`. Notebooks
in any subdirectory of `examples/` are rendered automatically.

## Pull requests

Open them against `master`, keep the tests green, and add an entry to the
`Unreleased` section of [CHANGELOG.md](CHANGELOG.md).
