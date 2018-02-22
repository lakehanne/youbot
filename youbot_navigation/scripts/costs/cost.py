""" This file defines the base cost class. """
import abc


class Cost(object):
    """ Cost superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        self.config = hyperparams

    @abc.abstractmethod
    def eval(self, **kwargs):
        """
        Evaluate cost function and derivatives.
        Args:
            kwargs:  A single sample's state and action params.
        """
        raise NotImplementedError("Must be implemented in subclass.")
