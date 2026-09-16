import numpy as np

import fiberoripy
from fiberoripy.__main__ import main
from fiberoripy.constants import COMPS


def test_version_is_exposed():
    """Users should be able to report which version they are running."""
    assert isinstance(fiberoripy.__version__, str)
    assert fiberoripy.__version__


def test_comps_indexes_a_flattened_tensor():
    """`COMPS` maps a component name onto its index in a raveled 3x3 tensor."""
    a = np.arange(9.0).reshape((3, 3))
    for name, index in COMPS.items():
        row, column = int(name[1]) - 1, int(name[2]) - 1
        assert a.ravel()[index] == a[row, column]


def test_cli_prints_where_to_find_the_examples(capsys):
    """`fiberoripy` is a pointer to the documentation, so it must print the link."""
    main()
    out = capsys.readouterr().out
    assert "fiberoripy" in out.lower()
    assert "https://github.com/meyer-nils/fiberoripy" in out
