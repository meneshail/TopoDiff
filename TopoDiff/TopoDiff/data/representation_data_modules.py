from functools import partial
import json
import os
import pickle
from typing import Optional, Sequence, List, Any, Literal

import ml_collections as mlc
import pandas as pd
import numpy as np
import torch

from myopenfold.utils.tensor_utils import tensor_tree_map, dict_multimap

from TopoDiff.data.encoder_transform import coords_to_dict, add_noise

import traceback
import logging

logger = logging.getLogger("TopoDiff.data.representation_data_modules")

def process_encoder_feature_no_wrap(coords, config_encoder_feat, extra_config = None, generator = None):
    """
    Process features for topology encoder.

    Args:
        coord: [N_res, 3]
        
    Returns:
        encoder_feats: [N_res, N_c]
            preprocessed features for encoder model
        encoder_coords: [N_res, 3]
            coordinates for encoder model
        encoder_mask: [N_res]
            mask for encoder model
        encoder_adj_mat [N_res, N_res]
            adjacency matrix for encoder model
    """
    feat = {}

    # add noise to the coordinates
    if (config_encoder_feat.add_noise.enabled and 
        not (extra_config is not None and extra_config['encoder_no_noise'])):
        coords = add_noise(coords, config_encoder_feat.add_noise, generator = generator)  
    
    # get graph features
    encoder_feat = coords_to_dict(coords)
    feat.update(encoder_feat)

    # add mask
    feat['encoder_mask'] = torch.ones(encoder_feat['encoder_feats'].size(0), dtype = torch.bool)

    return feat

def stack_fn(x):
    return torch.stack(x, dim=0) if isinstance(x[0], torch.Tensor) else x

class StructureRepresentationCollator:
    def __init__(self, config_data = None, force_pad_size = None, pad_in_collator = False):
        self.force_pad_size = force_pad_size

        if (config_data is not None and config_data.pad_in_collator.enabled) or pad_in_collator:
            self.pad_in_collator = True
        else:
            self.pad_in_collator = False

    def __call__(self, prots):
        if self.pad_in_collator:
            if 'encoder_feats' in prots[0]:
                max_len_encoder = max([p['encoder_feats'].shape[0] for p in prots]) if self.force_pad_size is None else self.force_pad_size
                for prot in prots:
                    prot['encoder_feats'] = torch.nn.functional.pad(prot['encoder_feats'], (0, 0, 0, max_len_encoder - prot['encoder_feats'].shape[0]))
                    prot['encoder_coords'] = torch.nn.functional.pad(prot['encoder_coords'], (0, 0, 0, max_len_encoder - prot['encoder_coords'].shape[0]))
                    prot['encoder_mask'] = torch.nn.functional.pad(prot['encoder_mask'], (0, max_len_encoder - prot['encoder_mask'].shape[0]))
                    prot['encoder_adj_mat'] = torch.nn.functional.pad(prot['encoder_adj_mat'], (0, max_len_encoder - prot['encoder_adj_mat'].shape[0], 0, max_len_encoder - prot['encoder_adj_mat'].shape[0]))
        
        return dict_multimap(stack_fn, prots) 
    

class StructureRepresentationMinimalDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_info: pd.DataFrame,
                 data_dict: dict,
                 config: mlc.ConfigDict,
                 extra_config = None,
                 epoch = None,
                 base_seed = 42,
    ):
        super().__init__()

        self.data_info = data_info
        self.data_dict = data_dict
        self.config = config

        self.extra_config = extra_config

        self.epoch = epoch
        self.base_seed = base_seed

        self.keys = sorted(self.data_info['key'].tolist())
        
        self.set_epoch(self.epoch)

    def __len__(self):
        return len(self.keys)

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.epoch is not None:
            self.gen = torch.Generator().manual_seed(self.epoch + self.base_seed)
        else:
            self.gen = None

    def __getitem__(self, idx):
        key = self.keys[idx]
        coord = torch.tensor(self.data_dict[key])

        # process features
        feat = process_encoder_feature_no_wrap(coord, self.config.encoder_feat, self.extra_config, generator = self.gen)

        feat['id'] = key
        feat['sample_idx'] = torch.tensor(idx, dtype = torch.long)
        feat['length'] = torch.tensor(feat['encoder_mask'].shape[0], dtype = torch.long)

        return feat
                




        

                 