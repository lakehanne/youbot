""" This file defines general utility functions and classes. """
import numpy as np

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

class TrajectoryInfo(BundleType):
    """ Collection of trajectory-related variables. """
    def __init__(self, config):

        T           = config['agent']['T']
        dU          = config['agent']['dU']
        dX          = config['agent']['dX']
        euler_iter  = config['agent']['euler_iter']

        variables   = {
                        'V':                np.zeros((T)),
                        'Vx':               np.zeros((T, dX)),
                        'Vxx':              np.zeros((T, dX, dX)),
                        'Qx':               np.zeros((T, dX)),
                        'Qu':               np.zeros((T, dU)),
                        'Qxx':              np.zeros((T, dX, dX)),
                        'Qux':              np.zeros((T, dU, dX)),
                        'Quu':              np.zeros((T, dU, dU)),
                        'Qx':               np.zeros((T, dX)),
                        'Qu_tilde':         np.zeros((T, dU)),
                        'Qux_tilde':        np.zeros((T, dU, dX)),
                        'Quu_tilde':        np.zeros((T, dU, dU)),
                        'fx':               np.zeros((T, dX, dX)),
                        'fu':               np.zeros((T, dX, dU)),
                        'fuu':              np.zeros((T, dU, dU)),
                        'fux':              np.zeros((T, dU, dX)),
                        'action':           np.zeros((T, dU)),
                        'action_nominal':   np.zeros((T, dU)),
                        'delta_action':     np.zeros((T, dU)),
                        'state':            np.zeros((T, dX)),
                        'nominal_state':    np.zeros((T, dX)),
                        'nominal_state_':   np.zeros((euler_iter, dX)),
                        'delta_state':      np.zeros((T, dX)),
                        'delta_state_plus': np.zeros((T, dX)),
                        'gu':               np.zeros((T, dU)),  # open loop gain
                        'noise_covar':      np.zeros((T)),
                        'Gu':               np.zeros((T, dU, dX))   # closed loop gain
                    }
        BundleType.__init__(self, variables)


def extract_condition(hyperparams, m):
    """
    Pull the relevant hyperparameters corresponding to the specified
    condition, and return a new hyperparameter dictionary.
    """
    return {var: val[m] if isinstance(val, list) else val
            for var, val in hyperparams.items()}
