import numpy as np
import scipy.optimize as optimize
import multiprocess as mp

class ProgressiveDepth():
    '''
    Progressive Version of Elastic Depths.
    Non-incremental version is introduced in "Elastic Depths for 
    Detectng Shape Anomalies in Functional Data" [Harris et al, 2020].
    Parameters
    ----------
    None
    Attributes
    ----------
    
    ----------
    '''
    def __init__(self, prog_ed=None, F=None, k=1.5, p=0.95):

        if prog_ed == None:
            self.F = F # data matrix of functional time series
            self.depths = [] # amplitude depths for all functions
            self.labels = []
            self.k = k
            self.p = p

        else:
            self.F = prog_ed.F
            self.depths = prog_ed.depths
            self.labels = prog_ed.labels
            self.k = prog_ed.k
            self.p = prog_ed.p

    def SRSF(self, f):
        '''
        Computing square root slope velocity function of f.
            Finds distance between curves and functional data. 
        Parameters
        ----------
        f
        Returns
        -------
        f' / sqrt(|f'|)
        '''
        f_prime = np.gradient(f.flatten())
        return f_prime * np.sqrt(np.abs(f_prime) + 1e-16)

    def distance(self, gamma, f, g):
        return np.linalg.norm(self.SRSF(f) - self.SRSF(g * gamma), ord=2)

    def amplitudeOutlyingness(self, f, P):
        '''
        Computing amplitude outlyingess for the function f in P
        Parameters
        ----------
        f
        P: array of all functions
        Returns
        -------
        percent outlying the distance of f from random function g
        '''
        g = P[np.random.choice(len(P))]

        y = optimize.minimize(
            fun=self.distance, 
            x0=0.0, 
            args=(f, g),
            bounds=[(0, 1)] 
        )
        
        amplitudeDistance = self.distance(y.x, f, g)
        
        # amplitude outlyingness for the current gamma
        outlyingness = np.percentile(amplitudeDistance, 50)
        return outlyingness

    def amplitudeDepth(self, f, P):
        return (1 + self.amplitudeOutlyingness(f,P))**-1

    def getAmplitudeOutliers(self, F):
        '''
        Computing amplitude depths for each function. Median of the amplitude distances between f and all f[0],..,f[n]
        Parameters
        ----------
        F: array-like, shape(n_dimensions, n_time_points, 1)
            Multidimensional function data discretized with even intervals.
        Returns
        -------
        self
        '''
        with mp.Pool() as pool:
            depths_new = pool.map(lambda i: self.amplitudeDepth(F[i], self.F), range(len(F)))

        self.depths.extend(depths_new)

        iqr = max(self.depths) - np.median(self.depths)
        c = np.median(self.depths) - self.k * iqr 
        q = np.percentile(self.depths, (1-self.p) * 100)

        with mp.Pool() as pool:
            self.labels = pool.map(lambda d: 'outlier' if d < min(c, q) else 'not outlier', self.depths)

        return self