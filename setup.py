# -*- coding: utf-8 -*-
"""Setup file."""
from os import path

import setuptools

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="fiberoripy",
    version="1.0.5",
    author="Nils Meyer",
    author_email="nils.meyer@kit.edu",
    description="Fiber orientation models and closures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    scripts=["bin/fiberoripy"],
    packages=["fiberoripy"],
    url="https://github.com/nilsmeyerkit/fiberoripy",
    package_dir={"fiberoripy": "fiberoripy"},
    package_data={"fiberoripy": ["examples/"]},
    install_requires=["numpy", "matplotlib", "scipy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
