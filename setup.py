# -*- coding: utf-8 -*-
"""Setup file."""
import setuptools

setuptools.setup(
    name="fiberpy",
    version="0.0.1",
    author="Nils Meyer",
    author_email="nils.meyer@kit.edu",
    description="Fiber orientation models and closures",
    long_description=open("README.md").read(),
    packages=["fiberpy"],
    package_dir={"fiberpy": "fiberpy"},
    install_requires=["numpy", "matplotlib", "scipy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
