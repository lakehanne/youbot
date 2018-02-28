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

        u             = kwargs['u'] 
        v             = kwargs['v'] 

        wu            = self.config['all_costs']['wu']
        wv            = self.config['all_costs']['wv']
        alpha         = self.config['all_costs']['alpha']
        gamma         = self.config['all_costs']['gamma']

        u_exp         = np.expand_dims(u, axis=1)
        v_exp         = np.expand_dims(v, axis=1)
        wu_exp        = np.expand_dims(wu, axis=1)
        wv_exp        = np.expand_dims(wv, axis=1)

        T             = self.config['agent']['T']
        dU            = self.config['agent']['dU']
        dV            = self.config['agent']['dV']
        dX            = self.config['agent']['dX']

        lx            = np.zeros((dX))
        lux           = np.zeros((dU, dX))
        luv           = np.zeros((dU, dV))
        lvx           = np.zeros((dV, dX))
        lxx           = np.zeros((dX, dX))

        l             = (alpha**2) * (np.cosh(wu_exp.T.dot(u_exp)-1) \
                        - gamma*np.cosh(wv_exp.T.dot(v_exp)-1))
        l             = l.squeeze() if l.ndim > 1 else l
        
        lu            = (alpha**2) * wu * np.sinh(wu*u)
        lv            = -gamma*(alpha**2) * wv * np.sinh(wv*v)
        luu           = np.diag((alpha**2) * (wu**2) * np.cosh(wu*u))
        lvv           = np.diag(-(alpha**2) * gamma * (wv**2) * np.cosh(wv*v))


        # print('l: {}, \nlu: {}, \nluu: {}'.format(l, lu, luu)) 
        # print('lv: {}, \nlvv: {}, \nluv: {}'.format(lv, lvv, luv)) 

        return l, lx, lu, lv, lux, lvx, lxx, luu, luv, lvv