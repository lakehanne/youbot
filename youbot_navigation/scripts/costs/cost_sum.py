import numpy as np
import copy

from .cost import Cost
from scripts.costs.config import COST_SUM

from scripts.costs import CostAction, CostState

class CostSum(Cost):
    """docstring for CostSum"""
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_SUM)
        config.update(hyperparams)
        Cost.__init__(self, config)

        self._costs = []

        self._weights = self.config['all_costs']['weights']

        for cost in self.config['all_costs']['costs']:
            self._costs.append(cost['type'](config))

    def eval(self, **kwargs):
        T             = self.config['agent']['T']
        dU            = self.config['agent']['dU']
        dV            = self.config['agent']['dV']
        dX            = self.config['agent']['dX']

        l, lx, lu, lux, lxx,  luu = self._costs[0].eval(**kwargs)

        # print('action l: ', l)
        # Compute weighted sum of each cost value and derivatives.
        weight = self._weights[0]

        for i in range(1, len(self._costs)):
            pl, plx, plu, plux, plxx, pluu  = self._costs[i].eval(**kwargs)
            weight = self._weights[i]

            # print(self._costs[i])
            # print('lxx: {}, plxx: {}'.format(lxx.shape, plxx.shape))

            l   = l + pl * weight
            lx  = lx + plx * weight
            lu  = lu + plu * weight
            lxx = lxx + plxx * weight
            luu = luu + pluu * weight
            lux = lux + plux * weight  

        # print('state l: ', l)

        return l, lx, lu, lux, lxx,  luu