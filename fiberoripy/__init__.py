"""Fiber orientation models and closures."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fiberoripy")
except PackageNotFoundError:  # pragma: no cover - not installed, e.g. a source tree
    __version__ = "unknown"

__all__ = ["__version__"]
