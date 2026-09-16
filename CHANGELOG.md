# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Tests for the fitting, plotting, aspect ratio and constants modules, and for the
  error paths of the closures, taking statement coverage to 100%. CI measures
  coverage, fails below 95% and reports a table in the job summary.
- The `constants` module is now part of the API reference.
- CI builds the distribution and runs the test suite against the installed wheel,
  so a file missing from the wheel or sdist fails a pull request rather than a
  release.

### Changed
- Enabled the bugbear, pyupgrade and ruff-specific lint rules, and fixed what they
  reported: `%`-formatting in the fitting example, redundant parentheses, and list
  concatenation in the tests.
- `fit_optimal_params` and `compute_error` pair parameter names with values using
  `zip(..., strict=True)`, so a mismatched `keys`/`values` pair raises instead of
  silently fitting fewer parameters.
- Each example notebook now has a heading that identifies it; three shared
  "Testing re-orientation in shearflow" and two more were also duplicates.

### Changed
- Shape validation in `assert_fot_properties` and `assert_fot4_properties` raises
  `ValueError` with the offending shape instead of using a bare `assert`, which is
  stripped under `python -O`.
- Closure dispatch uses lookup tables, so the supported closure names are recorded
  in one place rather than in both a name set and a chain of `if` statements.

## [1.4.0] - 2026-09-16

### Added
- Batched-input tests for every closure.
- `test` and `examples` extras; `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`.
- README quickstart and documentation of the fourth- to sixth-order closures.
- Dependabot configuration for GitHub Actions.
- An optional `rng` argument on `get_random_tensor_pair` for reproducible
  sampling.
- `fiberoripy.__version__`, read from the installed package metadata.

### Changed
- Minimum Python is now 3.11; CI covers 3.11-3.14.
- `pytest` and `ipympl` moved from runtime dependencies into extras.
- Runtime dependencies gained tested lower bounds (`numpy>=1.23.3`, `scipy>=1.9.2`,
  `matplotlib>=3.6`).
- License metadata migrated to a PEP 639 SPDX expression.
- Random sampling uses `numpy.random.Generator` instead of the legacy global
  `numpy.random` functions; the test tensors are now seeded and reproducible.
- `get_random_tensor_pair` takes `seed=None` rather than evaluating the default
  isotropic tensor at import time.
- Actions updated to current majors; PyPI releases use Trusted Publishing.
- Replaced black, isort and flake8 with ruff, and added a CI lint job.
- Unified docstrings on the numpydoc style, now enforced by ruff.
- Documentation moved from Sphinx on Read the Docs to Zensical on GitHub Pages at
  <https://meyer-nils.github.io/fiberoripy>. The API reference is generated from the
  docstrings by mkdocstrings, and all example notebooks are executed and rendered
  into a gallery.

### Fixed
- Orientation models used `np.linalg.eig`, which returns `complex128`
  unconditionally as of NumPy 2.5.0 and broke every spectral-decomposition model on
  Python 3.12+. They now use `np.linalg.eigh`.
- `SQC` and the fourth- to sixth-order `LINEAR`/`HYBRID` closures raised on batched
  input despite documenting support for it.
- The `fiberoripy` command pointed at a non-existent examples directory and emitted
  `SyntaxWarning` on Python 3.12+.
- README: closure `SIC` renamed to `SIQ` (the accepted name), stale repository links.
- `compute_closure_FOT2`, `compute_closure_FOT4` and `sample_circle` returned
  `None` for an unrecognised argument instead of raising.
- The published API reference omitted the `closures`, `fit` and `tensorplot`
  modules entirely; all three are now documented.

## [1.3.0] - 2025-09-22
- Added fourth- to sixth-order closures (`LINEAR`, `QUADRATIC`, `HYBRID`).
- `compute_closure` dispatches on input order and reports unsupported closures.

## [1.2.1] - 2024-12-06
- Fixed Read the Docs configuration and documentation requirements.

## [1.2.0] - 2024-12-06
- Added a Mori-Tanaka based modified Jeffery equation.
- Added `integrate_ori_ode`, a wrapper for SciPy solvers using the 3x3 representation.
- Moved packaging from `setup.py` to `pyproject.toml`; fixed fitting procedures.

## [1.1.4] - 2024-07-01
- Broadcastable `SQC` closure and one-dimensional `SIQ` closure.

Earlier releases: see the [tags](https://github.com/meyer-nils/fiberoripy/tags).

[Unreleased]: https://github.com/meyer-nils/fiberoripy/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/meyer-nils/fiberoripy/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/meyer-nils/fiberoripy/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/meyer-nils/fiberoripy/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/meyer-nils/fiberoripy/compare/v1.1.4...v1.2.0
[1.1.4]: https://github.com/meyer-nils/fiberoripy/compare/v1.1.3...v1.1.4
