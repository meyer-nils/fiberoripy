# -*- coding: utf-8 -*-
"""Setup file."""
from os import path, walk

import setuptools

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def package_files(directory):
    """List files of a directory that should be included by pip.

    Parameters
    ----------
    directory : str
        Included directory.

    Returns
    -------
    list of str
        Filenames to be included.

    """
    paths = []
    for (p, dir, filenames) in walk(directory):
        for filename in filenames:
            paths.append(path.join("..", p, filename))
    return paths


setuptools.setup(
    name="fiberoripy",
    version="1.0.10",
    author="Nils Meyer",
    author_email="nils.meyer@kit.edu",
    description="Fiber orientation models and closures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": ["fiberoripy = fiberoripy.__main__:main"]
    },
    packages=["fiberoripy"],
    url="https://github.com/nilsmeyerkit/fiberoripy",
    package_dir={"fiberoripy": "fiberoripy"},
    package_data={"fiberoripy": package_files("examples")},
    install_requires=["numpy", "matplotlib", "scipy", "pytest"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
