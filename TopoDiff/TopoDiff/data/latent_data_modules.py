from functools import partial
import json
import os
import pickle
from typing import Optional, Sequence, List, Any

import ml_collections as mlc
import numpy as np
import torch

from myopenfold.utils.tensor_utils import tensor_tree_map, dict_multimap

import traceback
import logging

logger = logging.getLogger("TopoDiff.data.latent_data_modules")

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self,
                 config: mlc.ConfigDict,
                 latent_data: torch.Tensor = None,
                 latent_data_path: str = None,
                 epoch = 0
                 ):
        """
        Args:
            config (mlc.ConfigDict): 
                data config
            latent_data (torch.Tensor): [*, latent_dim]
                The latent data
            latent_data_path (str): 
                path to the latent data
        """
        super().__init__()

        if latent_data is None and latent_data_path is None:
            raise ValueError('Either `latent_data` or `latent_data_path` must be provided')
        if latent_data is not None and latent_data_path is not None:
            raise ValueError('Only one of `latent_data` or `latent_data_path` must be provided')
        
        self.config = config
        self.latent_data = latent_data
        self.latent_data_path = latent_data_path

        self.epoch = epoch
        self.gen = torch.Generator().manual_seed(self.epoch)
        
        if self.latent_data is None:
            self.latent_data = torch.load(self.latent_data_path)
        
        self.latent_dim = self.latent_data.shape[-1]
        self.latent_T = self.config.common.timestep.T

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.gen = torch.Generator().manual_seed(self.epoch)

    def __len__(self):
        return self.latent_data.shape[0]
    
    def __getitem__(self, idx):
        feat = {}

        feat['timestep'] = torch.randint(1, self.latent_T+1, (), generator = self.gen)
        feat['sample_idx'] = torch.tensor(idx, dtype = torch.long)
        feat['latent_gt'] = self.latent_data[idx]

        return feat
    
class LatentCollator:
    def __call__(self, prots):
        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, prots)

        


