"""Setup file."""
from distutils.core import setup

setup(
    name='fiberpy',
    version='0.1dev',
    author="Nils Meyer",
    author_email="nils.meyer@kit.edu",
    packages=['fiberpy', ],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),
    )
