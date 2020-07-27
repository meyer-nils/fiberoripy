# -*- coding: utf-8 -*-
"""Explanation and hint for examples."""
import os

import fiberoripy


def main(args=None):
    """Show usage information."""
    path = os.path.dirname(os.path.dirname(fiberoripy.__file__))

    print("Copyright (c) 2020 Nils Meyer")
    print(
        "Fiberoripy is a python package that provides fiber orientation models"
        " and closure models."
    )
    print("Check out the examples at:")
    print(os.path.join(path, "examples"))


if __name__ == "__main__":
    main()
