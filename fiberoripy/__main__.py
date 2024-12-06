import os

import fiberoripy


def main(args=None):
    """Show usage information."""
    path = os.path.dirname(os.path.dirname(fiberoripy.__file__))

    print(
        "  ______ _ _                ____       _ _____       \n"
        " |  ____(_) |              / __ \     (_)  __ \      \n"
        " | |__   _| |__   ___ _ __| |  | |_ __ _| |__) |   _ \n"
        " |  __| | | '_ \ / _ \ '__| |  | | '__| |  ___/ | | |\n"
        " | |    | | |_) |  __/ |  | |__| | |  | | |   | |_| |\n"
        " |_|    |_|_.__/ \___|_|   \____/|_|  |_|_|    \__, |\n"
        "                                                __/ |\n"
        "                                               |___/ \n"
        "Copyright (c) 2024 "
        "Nils Meyer, Constantin Krauß, Louis Schreyer, Julian Bauer\n\n"
        "Fiberoripy is a python package that provides fiber orientation models"
        " and closures for fourth order orientation tensos.\n\n"
        f"Check out the examples at: {os.path.join(path, 'examples')}"
    )


if __name__ == "__main__":
    main()
