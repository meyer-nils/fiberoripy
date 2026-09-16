# Raw string: the ASCII art contains backslashes that must not be read as escapes.
BANNER = r"""  ______ _ _                ____       _ _____
 |  ____(_) |              / __ \     (_)  __ \
 | |__   _| |__   ___ _ __| |  | |_ __ _| |__) |   _
 |  __| | | '_ \ / _ \ '__| |  | | '__| |  ___/ | | |
 | |    | | |_) |  __/ |  | |__| | |  | | |   | |_| |
 |_|    |_|_.__/ \___|_|   \____/|_|  |_|_|    \__, |
                                                __/ |
                                               |___/
"""


def main(args=None):
    """Show usage information."""
    print(
        BANNER + "\n"
        "Copyright (c) 2026 "
        "Nils Meyer, Constantin Krauß, Louis Schreyer, Julian Bauer, "
        "Johannes Mitsch\n\n"
        "Fiberoripy is a python package that provides fiber orientation models"
        " and closures for fourth order orientation tensors.\n\n"
        "Check out the examples at: "
        "https://github.com/meyer-nils/fiberoripy/tree/master/examples"
    )


if __name__ == "__main__":
    main()
