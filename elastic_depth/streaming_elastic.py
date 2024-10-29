import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapz, cumtrapz
from joblib import Parallel, delayed
import scipy.optimize as optimize
import optimum_reparamN2 as orN2
import optimum_reparam_N as orN
import multiprocess as mp

class StreamingDepth():
    '''
    Incremental and Progressive Version of Elastic Depths
    Non-incremental version is introduced in "Elastic Depths for Detecting Shape Anomalies in Functional Data" [Harris et al, 2020]
    Parameters
    ----------
    inc_ed, F, k, p, threshold
    Attributes
    ----------
    
    ----------
    '''
    def __init__(self, F=None, k=1.5, threshold=0.5, n_inc=0):
        self.F = F # data matrix of functional time series
        self.time = np.linspace(0, 1, F.shape[0]) if F is not None else None
        self.depths = [] # elastic depths for all functions
        self.labels = [] # anomalous labels for each function
        self.k = k
        self.threshold = threshold
        self.n_inc = n_inc

    def warp_f_gamma(self, time, f, gam):
        """
        warps a function f by gam

        :param time vector describing time samples
        :param q vector describing srsf
        :param gam vector describing warping function

        :rtype: numpy ndarray
        :return f_temp: warped srsf

        """
        f_temp = np.interp((time[-1] - time[0]) * gam + time[0], time, f)

        return f_temp
    
    def warp_q_gamma(self, time, q, gam):
        """
        warps a srsf q by gam

        :param time vector describing time samples
        :param q vector describing srsf
        :param gam vector describing warping function

        :rtype: numpy ndarray
        :return q_temp: warped srsf

        """
        M = gam.size
        gam_dev = np.gradient(gam, 1 / np.double(M - 1))
        tmp = np.interp((time[-1] - time[0]) * gam + time[0], time, q)

        q_temp = tmp * np.sqrt(gam_dev)

        return q_temp
    
    def gradient_spline(self, time, f, smooth=False):
        """
        This function takes the gradient of f using b-spline smoothing

        :param time: vector of size N describing the sample points
        :param f: numpy ndarray of shape (M,N) of M functions with N samples
        :param smooth: smooth data (default = F)

        :rtype: tuple of numpy ndarray
        :return f0: smoothed functions functions
        :return g: first derivative of each function
        :return g2: second derivative of each function

        """
        M = f.shape[0]

        if f.ndim > 1:
            N = f.shape[1]
            f0 = np.zeros((M, N))
            g = np.zeros((M, N))
            g2 = np.zeros((M, N))
            for k in range(0, N):
                if smooth:
                    spar = time.shape[0] * (.025 * np.fabs(f[:, k]).max()) ** 2
                else:
                    spar = 0
                tmp_spline = UnivariateSpline(time, f[:, k], s=spar)
                f0[:, k] = tmp_spline(time)
                g[:, k] = tmp_spline(time, 1)
                g2[:, k] = tmp_spline(time, 2)
        else:
            if smooth:
                spar = time.shape[0] * (.025 * np.fabs(f).max()) ** 2
            else:
                spar = 0
            tmp_spline = UnivariateSpline(time, f, s=spar)
            f0 = tmp_spline(time)
            g = tmp_spline(time, 1)
            g2 = tmp_spline(time, 2)

        return f0, g, g2
    
    def f_to_srsf(self, f, time, smooth=False):
        """
        converts f to a square-root slope function (SRSF)

        :param f: vector of size N samples
        :param time: vector of size N describing the sample points

        :rtype: vector
        :return q: srsf of f

        """
        eps = np.finfo(np.double).eps
        f0, self.g, g2 = self.gradient_spline(time, f, smooth)
        q = self.g / np.sqrt(np.fabs(self.g) + eps)
        return q
    
    def srsf_to_f(self, q, time, f0=0.0):
        """
        converts q (srsf) to a function

        :param q: vector of size N samples of srsf
        :param time: vector of size N describing time sample points
        :param f0: initial value

        :rtype: vector
        :return f: function

        """
        integrand = q*np.fabs(q)
        f = f0 + cumtrapz(integrand,time,initial=0)
        return f
    
    def optimum_reparam(self, q1, time, q2, method="DP2", lam=0.0, penalty="roughness", grid_dim=7):
        """
        calculates the warping to align srsf q2 to q1

        :param q1: vector of size N or array of NxM samples of first SRSF
        :param time: vector of size N describing the sample points
        :param q2: vector of size N or array of NxM samples samples of second SRSF
        :param method: method to apply optimization (default="DP2") options are "DP","DP2","RBFGS"
        :param lam: controls the amount of elasticity (default = 0.0)
        :param penalty: penalty type (default="roughness") options are "roughness", "l2gam", 
                        "l2psi", "geodesic". Only roughness implemented in all methods. To use
                        others method needs to be "RBFGS"
        :param grid_dim: size of the grid, for the DP2 method only (default = 7)

        :rtype: vector
        :return gam: describing the warping function used to align q2 with q1

        """

        if penalty == "l2gam" and (method == "DP" or method == "DP2"):
            raise Exception('penalty not implemented')
        if penalty == "l2psi" and (method == "DP" or method == "DP2"):
            raise Exception('penalty not implemented')
        if penalty == "geodesic" and (method == "DP" or method == "DP2"):
            raise Exception('penalty not implemented')
        
        if method == "DP2":
            if q1.ndim == 1 and q2.ndim == 1:
                gam = orN2.coptimum_reparam(np.ascontiguousarray(q1), time,
                                            np.ascontiguousarray(q2), lam, grid_dim)

            if q1.ndim == 1 and q2.ndim == 2:
                gam = orN2.coptimum_reparamN(np.ascontiguousarray(q1), time,
                                            np.ascontiguousarray(q2), lam, grid_dim)

            if q1.ndim == 2 and q2.ndim == 2:
                gam = orN2.coptimum_reparamN2(np.ascontiguousarray(q1), time,
                                            np.ascontiguousarray(q2), lam, grid_dim)
            
        else:
            raise Exception('Invalid Optimization Method')

        return gam
    
    def elastic_distance(self, f1, f2, time, method="DP2", lam=0.0):
        """"
        calculates the distances between function, where f1 is aligned to
        f2. In other words
        calculates the elastic distances

        :param f1: vector of size N
        :param f2: vector of size N
        :param time: vector of size N describing the sample points
        :param method: method to apply optimization (default="DP2") options are "DP","DP2","RBFGS"
        :param lam: controls the elasticity (default = 0.0)

        :rtype: scalar
        :return Dy: amplitude distance
        :return Dx: phase distance

        """
        q1 = self.f_to_srsf(f1, time)
        q2 = self.f_to_srsf(f2, time)

        gam = self.optimum_reparam(q1, time, q2, method, lam)
        fw = self.warp_f_gamma(time, f2, gam)
        qw = self.warp_q_gamma(time, q2, gam)

        Dy = np.sqrt(trapz((qw - q1) ** 2, time))
        M = time.shape[0]

        time1 = np.linspace(0,1,M)
        binsize = np.mean(np.diff(time1))
        psi = np.sqrt(np.gradient(gam,binsize))
        q1dotq2 = trapz(psi, time1)
        if q1dotq2 > 1:
            q1dotq2 = 1
        elif q1dotq2 < -1:
            q1dotq2 = -1

        Dx = np.real(np.arccos(q1dotq2))

        return Dy, Dx

    def distmat(self, f, f1, time, idx, method):
        N = f.shape[1]
        dp = np.zeros(N)
        da = np.zeros(N)
        for jj in range(N):
            Dy,Dx = self.elastic_distance(f[:,jj], f1, time, method)

            da[jj] = Dy
            dp[jj] = Dx
        
        return(da, dp)

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
        return f_prime / np.sqrt(np.abs(f_prime) + 1e-16)

    def elastic_depth(self, method="DP2", lam=0.0, parallel=True):
        """
        Calculates the elastic depth between functions in matrix f

        :param f: matrix of size MxN (M time points for N functions)
        :param time: vector of size M describing the sample points
        :param method: method to apply optimization (default="DP2") options are "DP","DP2","RBFGS"
        :param lam: controls the elasticity (default = 0.0)

        :rtype: scalar
        :return amp: amplitude depth
        :return phase: phase depth

        """
        obs, fns = self.F.shape

        amp_dist = np.zeros((fns,fns))
        phs_dist = np.zeros((fns,fns))

        if parallel:
            out = Parallel(n_jobs=-1)(delayed(self.distmat)(self.F, self.F[:, n], self.time, n, method) for n in range(fns))
            for i in range(0, fns):
                amp_dist[i, :] = out[i][0]
                phs_dist[i, :] = out[i][1]
        else:
            for i in range(0, fns):
                amp_dist[i, :], phs_dist[i, :] = self.distmat(self.F, self.F[:, i], self.time, i, method)
        
        amp_dist = amp_dist + amp_dist.T
        phs_dist = phs_dist + phs_dist.T

        amp = 1 / (1 + np.median(amp_dist,axis=0))
        phase = 1 / (1 + np.median(phs_dist,axis=0))
        phase = ((2+np.pi)/np.pi) * (phase - 2/(2+np.pi))

        self.depths = {'amplitude': amp, 'phase': phase}

        return amp, phase

    def elastic_outliers(self, compute_depths=True):
        '''
        Computing outliers from elastic depths
        Parameters
        ----------
        None
        Returns
        -------
        self
        '''
        if compute_depths:
            amp_depth, phs_depth = self.elastic_depth()

        amp = self.depths['amplitude']
        phs = self.depths['phase']

        amp_100 = np.max(amp)
        phs_100 = np.max(phs)

        amp_50 = np.percentile(amp, 50)
        phs_50 = np.percentile(phs, 50)

        amp_iqr = amp_100 - amp_50 
        phs_iqr = phs_100 - phs_50

        amp_lim = max(amp_50 - self.k * amp_iqr, 0)
        phs_lim = max(phs_50 - self.k * phs_iqr, 0)

        amp_thre = np.percentile(amp, self.threshold * 100)
        phs_thre = np.percentile(phs, self.threshold * 100)

        amp_out = (amp < amp_lim) & (amp < amp_thre)
        phs_out = (phs < phs_lim) & (phs < phs_thre)

        self.labels = {'amp': amp_out, 'phs': phs_out}

        return self
        

    def prog_update(self, f, method="DP2", lam=0.0, parallel=True):
        '''
        Updating depths with new function f. 
        Computes elastic depth for new function f.
        Determines anomalous label of new depth measure wrt existing depths in F.
        TODO: Checks whether recomputation of centralness is needed.
        Parameters
        ----------
        f: array-like, shape(1,n_time_points)
        Returns
        -------
        self
        '''
        obs, fns = self.F.shape

        amp_dist = np.zeros(fns)
        phs_dist = np.zeros(fns)

        # computing elastic distances 
        amp_dist, phs_dist = self.distmat(self.F, f, self.time, 0, method)

        amp_new = 1 / (1 + np.median(amp_dist,axis=0))
        phase_new = 1 / (1 + np.median(phs_dist,axis=0))
        phase_new = ((2+np.pi)/np.pi) * (phase_new - 2/(2+np.pi))

        # computing outlier label for new amp/phase depth
        amp_depths = self.depths['amplitude']
        phs_depths = self.depths['phase']

        amp_100 = np.max(amp_depths)
        phs_100 = np.max(phs_depths)

        amp_50 = np.percentile(amp_depths, 50)
        phs_50 = np.percentile(phs_depths, 50)

        amp_iqr = amp_100 - amp_50 
        phs_iqr = phs_100 - phs_50

        amp_lim = max(amp_50 - 1.5 * amp_iqr, 0)
        phs_lim = max(phs_50 - 1.5 * phs_iqr, 0)

        amp_thre = np.percentile(amp_depths, 0.5 * 100)
        phs_thre = np.percentile(phs_depths, 0.5 * 100)

        amp_out = (amp_new < amp_lim) & (amp_new < amp_thre)
        phs_out = (phase_new < phs_lim) & (phase_new < phs_thre)

        # updating labels
        self.labels['amp'] = np.append(self.labels['amp'], amp_out)
        self.labels['phs'] = np.append(self.labels['phs'], phs_out)

        # updating depths
        self.depths['amplitude'] = np.append(self.depths['amplitude'], amp_new)
        self.depths['phase'] = np.append(self.depths['phase'], phase_new)

        return self
    
    def inc_update(self, x, threshold=0.8, method="DP2", lam=0.0, parallel=True):
        """
        Updating elastic depths with new time points.
        TODO: Checks whether recomputation of centralness is needed.
        Parameters
        ----------
        x: array-like, shape(n_dimensions, 1)
            A new time point of multidimensional function data that discretized
            with even intervals.
        Returns
        -------
        self
        """
        F_inc = np.vstack((self.F, x)) 
        obs, fns = self.F.shape
        obs2, fns2 = F_inc.shape
        
        t1 = np.linspace(0, 1, self.F.shape[0])
        t2 = np.linspace(0, 1, F_inc.shape[0])

        # iterating through the functions
        for i in range(fns):
            f1 = self.F[:,i]
            f2 = F_inc[:,i]

            q1 = self.f_to_srsf(f1, t1) # previous without new point
            q2 = self.f_to_srsf(f2, t2) # with new point

            gamma = self.optimum_reparam(q1, t2, q2, method, lam)
            q1_aligned = np.interp(t2, np.linspace(0, 1, len(q1)), q1)

            q_diff = np.linalg.norm(q2-q1_aligned)
            print(q_diff)

            if q_diff > self.threshold:
                continue # significant change, recompute depth
            else:
                continue

        # updating F
        self.F = F_inc
        return self