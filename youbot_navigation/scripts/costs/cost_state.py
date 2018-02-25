import numpy as np
import numpy.linalg as LA
import copy

from .cost import Cost
from scripts.costs.config import COST_STATE

class CostState(Cost):
    """docstring for CostState"""
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_STATE)
        config.update(hyperparams)
        Cost.__init__(self, hyperparams)


    def eval(self, **kwargs):

        T             = self.config['agent']['T']
        dU            = self.config['agent']['dU']
        dV            = self.config['agent']['dV']
        dX            = self.config['agent']['dX']

        x             = kwargs['x'] 
        xstar         = kwargs['xstar'] 
        alpha         = kwargs['alpha']
        wx            = kwargs['wx']

        # expand the state        
        state_diff    = x - xstar # this is correct. see matlab code
        xs_exp        = np.expand_dims(x-xstar, axis=1) 

        lu            = np.zeros((dU))
        lv            = np.zeros((dV))
        lux           = np.zeros((dU, dX))
        luu           = np.zeros((dU, dU))
        luv           = np.zeros((dU, dV))
        lvx           = np.zeros((dV, dX))
        lvv           = np.zeros((dV, dV))

        stage_term    = xs_exp.T.dot(np.diag(wx)).dot(xs_exp)

        l             = np.sqrt(alpha + stage_term)
        lx            = (wx * state_diff) / l
        lxx           = (alpha * wx) / (l ** 3)
        lxx           = np.diag(lxx)

        print('l: {}, \nlx: {}, \nlxx: {}'.format(l, lx, lxx))

        return l, lx, lu, lv, lux, lvx, lxx, luu, luv, lvv