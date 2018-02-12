""" This file defines general utility functions and classes. """
import numpy as np
import scipy.ndimage as sp_ndimage

class BundleType(object):
    """
    This class bundles many fields, similar to a record or a mutable
    namedtuple.
    """
    def __init__(self, variables):
        for var, val in variables.items():
            object.__setattr__(self, var, val)

    # Freeze fields so new ones cannot be set.
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError("%r has no attribute %s" % (self, key))
        object.__setattr__(self, key, value)

class IterationData(BundleType):
    """ Collection of iteration variables. """
    def __init__(self):
        variables = {
            'sample_list': None,  # List of samples for the current iteration.
            'traj_info': None,  # Current TrajectoryInfo object.
            'cost_info': None,  # Current TrajectoryInfo object.
            'pol_info': None,  # Current PolicyInfo object.
            'traj_distr': None,  # Initial trajectory distribution.
            'new_traj_distr': None, # Updated trajectory distribution.
            'cs': None,  # Sample costs of the current iteration.
            'step_mult': 1.0,  # KL step multiplier for the current iteration.
            'eta': 1.0,  # Dual variable used in LQR backward pass.
        }
        BundleType.__init__(self, variables)

class TrajectoryInfo(BundleType):
    """ Collection of trajectory-related variables. """
    def __init__(self, config):

        T           = config['agent']['T']
        dU          = config['agent']['dU']
        dX          = config['agent']['dX']

        variables   = {
                        'fx':               np.zeros((T, dX, dX)),
                        'fu':               np.zeros((T, dX, dU)),
                        'fuu':              np.zeros((T, dU, dU)),
                        'fux':              np.zeros((T, dU, dX)),
                        'action':           np.zeros((T, dU)),
                        'nominal_action':   np.zeros((T, dU)),
                        'delta_action':     np.zeros((T, dU)),
                        'state':            np.zeros((T, dX)),
                        'nominal_state':    np.zeros((T, dX)),
                        'delta_state':      np.zeros((T, dX)),
                        'delta_state_plus': np.zeros((T, dX)),
                        'gu':               np.zeros((T, dU)),  # open loop gain
                        'noise_covar':      np.zeros((T)),
                        'Gu':               np.zeros((T, dU, dX))   # closed loop gain
                    }
        BundleType.__init__(self, variables)


class CostInfo(BundleType):
    """ Collection of stage cost-related variables. """
    def __init__(self, config):

        T           = config['agent']['T']
        dU          = config['agent']['dU']
        dX          = config['agent']['dX']

        variables   = {
                        'l':                np.zeros((T)),
                        'l_nom':            np.zeros((T)),
                        'lx':               np.zeros((T, dX)),
                        'lu':               np.zeros((T, dU)),
                        'lux':              np.zeros((T, dU, dX)),
                        'lxx':              np.zeros((T, dX, dX)),
                        'luu':              np.zeros((T, dU, dU)),
                        'Qx':               np.zeros((T, dX)),
                        'Qu':               np.zeros((T, dU)),
                        'Qxx':              np.zeros((T, dX, dX)),
                        'Qux':              np.zeros((T, dU, dX)),
                        'Quu':              np.zeros((T, dU, dU)),
                        'Qx':               np.zeros((T, dX)),
                        'Qu_tilde':         np.zeros((T, dU)),
                        'Qux_tilde':        np.zeros((T, dU, dX)),
                        'Quu_tilde':        np.zeros((T, dU, dU)),
                        'V':                np.zeros((T)),
                        'Vx':               np.zeros((T, dX)),
                        'Vxx':              np.zeros((T, dX, dX)),
                    }
        BundleType.__init__(self, variables)

def extract_condition(hyperparams, m):
    """
    Pull the relevant hyperparameters corresponding to the specified
    condition, and return a new hyperparameter dictionary.
    """
    return {var: val[m] if isinstance(val, list) else val
            for var, val in hyperparams.items()}

def generate_noise(T, dU, agent):
    """
    Generate a T x dU gaussian-distributed random walk noise vector. This will
    approximately have mean 0 and variance 1, ignoring smoothing.

    Args:
        T: Number of time steps.
        dU: Dimensionality of actions.
    agent:
        smooth: Whether or not to perform smoothing of noise.
        var : If smooth=True, applies a Gaussian filter with this
            variance.
        renorm : If smooth=True, renormalizes data to have variance 1
            after smoothing.
    """

    smooth      = agent['smooth_noise']
    var         = agent['smooth_noise_var']
    renorm      = agent['smooth_noise_renormalize']
    random_walk = agent['smooth_random_walk']

    noise       = np.random.randn(T, dU)

    if smooth:
        # Smooth noise. This violates the controller assumption, but
        # might produce smoother motions.
        for i in range(dU):
            noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)

        if random_walk:
            noise_cum = np.cumsum(noise, axis=1)

            # Determine the time evolution of the mean square distance.
            noise_cum_sq = noise_cum**2
            mean_sq_noise = np.mean(noise_cum_sq, axis=0) 

            noise = noise_cum_sq 
                  
        if renorm:
            variance = np.var(noise, axis=0)
            noise = noise / np.sqrt(variance)
    return noise