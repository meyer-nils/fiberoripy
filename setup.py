# -*- coding: utf-8 -*-
"""Setup file."""
import setuptools

setuptools.setup(
    name="fiberpy",
    version="0.0.1",
    author="Nils Meyer",
    author_email="nils.meyer@kit.edu",
    description="Fiber orientation models",
    packages=["fiberpy"],
    package_dir={"fiberpy": "fiberpy"},
    include_package_data=True,
    install_requires=["numpy", "matplotlib", "scipy"],
)
