import numpy as np
import numpy.linalg as LA
import copy
from scripts.costs.config import COST_ACTION

from .cost import Cost

class CostAction(Cost):
    """docstring for CostAction"""
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_ACTION)
        config.update(hyperparams)
        Cost.__init__(self, hyperparams)

    def eval(self, **kwargs):

        T             = self.config['agent']['T']
        dU            = self.config['agent']['dU']
        dV            = self.config['agent']['dV']
        dX            = self.config['agent']['dX']

        u             = kwargs['u'] 
        l21_const     = kwargs['l21_const']
        action_penalty= kwargs['action_penalty']

        u_exp = np.expand_dims(u, axis=1)

        lx            = np.zeros((dX))
        lux           = np.zeros((dU, dX))
        lxx           = np.zeros((dX, dX))

        # l             = LA.norm(action_penalty * l21_const **2 * np.cosh( (u/l21_const) - 1))
        # lu            = action_penalty * l21_const* np.sinh(u/l21_const)
        # luu           = np.diag(np.cosh(u/l21_const))

        l             = 0.5 * action_penalty[0] * u_exp.T.dot(u_exp)
        lu            = action_penalty * u
        luu           = np.eye(dU)  
        # print('luu: ', luu)      


        return l, lx, lu, lux, lxx, luu