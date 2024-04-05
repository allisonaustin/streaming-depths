import sys
import os
from distutils.core import setup

setup(name='inc-shape',
      version='0.01',
      packages=[''],
      package_dir={'': '.'},
      install_requires=['numpy', 'scipy', 'scikit-fda', 'matplotlib', 'multiprocess'],
      py_modules=['inc_shape'])