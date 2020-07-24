# -*- coding: utf-8 -*-
"""Setup file."""
import setuptools

setuptools.setup(
    name="fiberoripy",
    version="1.0.0",
    author="Nils Meyer",
    author_email="nils.meyer@kit.edu",
    description="Fiber orientation models and closures",
    long_description=open("README.md").read(),
    packages=["fiberoripy"],
    package_dir={"fiberoripy": "fiberoripy"},
    install_requires=["numpy", "matplotlib", "scipy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
