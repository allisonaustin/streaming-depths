# streaming-depths
Incremental and progressive elastic depths for outlier detection of time series data. 

## About 
Elastic depths introduced in "Elastic Depths for Detecting Shape Anomalies in Functional Data" [Harris et al, 2020]. It is available in fdasrsf python package (https://fdasrsf-python.readthedocs.io/en/latest/)
The source code can be found here: https://github.com/jdtuck/fdasrsf_python. 

This implementation incrementally updates new incoming time points and progressively updates (with or without approximation) the new time-series (i.e., functions) for fast streaming input data. 
 
## Requirements
Python3, Numpy, Cython, cffi, multiprocess, scipy, scikit-fda

Note: Tested on macOS Sonoma.

## Setup 

``` pip install -r requirements.txt ```

### To install incremental elastic depths: 

` cd ./elastic_depth/ `

```python setup.py install```

### Usage: 
See examples/ for detailed usage.