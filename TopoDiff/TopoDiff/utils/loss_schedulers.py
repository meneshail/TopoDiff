from typing import Any
import torch
import logging

from numbers import Number

logger = logging.getLogger("TopoDiff.utils.scheduler")


class LossScheduler:
    def __init__(self, config_weight):
        self.config = config_weight

        if self.config.schedule is None and isinstance(config_weight.weight, Number):  # constant
            print('Mode: constant weight')
            self.mode = 0
            self.weight = config_weight.weight
        elif self.config.schedule.mode == 'cold start':
            print('Mode: cold start')
            self.mode = 2

            self.cold_start_epoch = self.config.schedule.cold_start_epoch
            self.warm_up_epoch = self.config.schedule.warm_up_epoch
            self.weight_min = self.config.schedule.weight_min
            self.weight_max = self.config.schedule.weight_max
        else:
            raise NotImplementedError(f"mode: {self.config.schedule.mode}")
        
    def __call__(self, epoch):
        if self.mode == 0:
            return self.weight
        elif self.mode == 2:
            if epoch < self.cold_start_epoch:
                return self.weight_min
            elif epoch < self.warm_up_epoch:
                return (epoch - self.cold_start_epoch) / (self.warm_up_epoch - self.cold_start_epoch) * (self.weight_max - self.weight_min) + self.weight_min
            else:
                return self.weight_max
        else:
            raise NotImplementedError

        