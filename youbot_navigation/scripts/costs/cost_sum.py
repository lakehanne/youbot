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

        l, lx, lu, lv, lux, lvx, lxx, luu, luv, lvv \
                = self._costs[0].eval(**kwargs)

        # Compute weighted sum of each cost value and derivatives.
        weight = self._weights[0]

        for i in range(1, len(self._costs)):
            l, lx, lu, lv, lux, lvx, lxx, luu, luv, lvv \
                 = self._costs[i].eval(**kwargs)
            weight = self._weights[i]

            l   = l + pl * weight
            lx  = lx + plx * weight
            lu  = lu + plu * weight
            lv  = lv + plv * weight
            luu = luu + pluu * weight
            luv = luv + pluv * weight
            lvv = lvv + plvv * weight
            lux = lux + plux * weight 
            lvx = lvx + plvx * weight  
            lxx = lxx + plxx * weight

        return l, lx, lu, lv, lux, lvx, lxx, luu, luv, lvv