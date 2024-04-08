import sys
import os
from distutils.core import setup

setup(name='prog-shape',
      version='0.01',
      packages=[''],
      package_dir={'': '.'},
      install_requires=['numpy', 'scipy', 'scikit-fda', 'multiprocess'],
      py_modules=['prog_shape'])