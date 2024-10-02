import logging
from typing import Optional, Sequence, List, Any

import ml_collections as mlc
import numpy as np
import torch

from myopenfold.np import residue_constants as rc

import logging

logger = logging.getLogger("TopoDiff.data.data_modules")

    
class UnconditionalSamplingDummyCollator:
    def __call__(self, prots):
        return prots[0]

class LatentContionalSamplingDataset2(torch.utils.data.Dataset):
    def __init__(
            self,
            condition_dict: dict,
            **kwargs,
    ):
        super(LatentContionalSamplingDataset2, self).__init__()

        condition_keys = list(condition_dict.keys())
        n_sample = len(condition_dict[condition_keys[0]])

        for k in condition_keys:
            assert len(condition_dict[k]) == n_sample

        self.condition_dict = condition_dict
        self.n_sample = n_sample

    def __len__(self):
        return self.n_sample
    
    def __getitem__(self, idx):
        condition_dict = {k: self.condition_dict[k][idx] for k in self.condition_dict}
        condition_dict['sample_idx'] = idx
        return condition_dict