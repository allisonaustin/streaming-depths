import sys
import os
from distutils.core import setup

setup(name='streaming-elastic',
      version='0.01',
      packages=[''],
      package_dir={'': '.'},
      install_requires=['numpy', 'scipy', 'scikit-fda', 'multiprocess', 'joblib', 'optimum_reparam_N', 'optimum_reparamN2'],
      py_modules=['streaming_elastic'])