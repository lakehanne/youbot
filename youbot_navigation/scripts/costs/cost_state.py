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
        l21_const     = kwargs['l21_const']
        state_penalty = kwargs['state_penalty']

        # expand the state        
        state_diff    = x - xstar
        x_exp         = np.expand_dims(x, axis=1)  # this is correct. see matlab code
        xs_exp        = np.expand_dims(x-xstar, axis=1) 

        lu            = np.zeros((dU))
        lux           = np.zeros((dU, dX))
        luu           = np.zeros((dU, dU))

        stage_term    = 0.5 * np.expand_dims(state_penalty*x, axis=1)\
                        .T.dot(np.expand_dims(state_penalty*x, axis=1))
        l1l2_term     = np.sqrt(l21_const + xs_exp.T.dot(xs_exp))

        l             = stage_term + l1l2_term
        lx            = state_penalty * x + state_diff /(l1l2_term)

        # lxx_t1        = l21_const
        # lxx_t2_top    = np.diag(state_penalty) * l1l2_term ** 3 
        # lxx_t2_bot    = l1l2_term**3
        # lxx           = (lxx_t1 + lxx_t2_top)/lxx_t2_bot

        # lxx           = (l21_const + state_penalty * l1l2_term)/(l1l2_term)
        # stage_term    = 0.5 * LA.norm(state_penalty * x)**2
        # l1l2_term     = np.sqrt(l21_const +   xs_exp.T.dot(xs_exp))
        # l             = stage_term + l1l2_term
        # lx            = state_penalty * x + state_diff/(l1l2_term) 
        lxx_t1        = np.diag(state_penalty)
        lxx_t2_top    = l1l2_term - (state_diff**2) / l1l2_term
        lxx_t2_bot    = state_diff**2
        lxx_t2        = lxx_t2_top / lxx_t2_bot
        lxx           = lxx_t1 + np.diag(lxx_t2.squeeze())
        # print('lxx: ', lxx, '\n', 'lx: ', lx)

        return l, lx, lu, lux, lxx,  luu