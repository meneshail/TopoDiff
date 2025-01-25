import os
import traceback
import logging
import math
import ml_collections as mlc
import json

import pickle

import numpy as np
import pandas as pd

import torch

# data
from TopoDiff.data import data_transforms

logger = logging.getLogger("TopoDiff.data.feature_pipeline")

def transform_fns(common_cfg,
                mode_cfg,
                extra_config,
                ):
    crop_feats = dict(common_cfg.feat)
    rename_dict = common_cfg.feat_rename
    dtype_dict = common_cfg.feat_type

    transforms = [
        data_transforms.random_crop_to_size(mode_cfg.crop_size, crop_feats)
    ]

    transforms.extend([
        # sequence feat
        data_transforms.make_seq_feat(common_cfg.masked_sequence),
        data_transforms.update_seq_idx(),

        # preprocessed structure specific transforms
        data_transforms.get_fake_atom14_from_atom37,

        # structure relevant transforms
        data_transforms.get_chi_angles(),
        data_transforms.center_backbone_frames(common_cfg.coordinate_shift),
        data_transforms.add_timestep(common_cfg.timestep),
        data_transforms.add_frame_mask(common_cfg.partial_fixed), 
        data_transforms.get_coord(common_cfg.coord)
    ])

    # optional feature transforms
    if common_cfg.topo_conditioning.enabled:
        transforms.append(data_transforms.add_topo_conditiona(common_cfg.topo_conditioning))

    if common_cfg.encoder_feat.enabled:
        transforms.append(data_transforms.process_encoder_feature(common_cfg.encoder_feat,
            extra_config,
        ))

    transforms.append(data_transforms.select_feat(list(crop_feats), rename_dict, dtype_dict))

    return transforms

def np_to_tensor_dict(feat):
    for key in feat:
        if isinstance(feat[key], np.ndarray):
            feat[key] = torch.tensor(feat[key])

    return feat

class PreprocessedFeaturePipeline:
    def __init__(self, 
                config,
                ):
        self.config = config

    def process_preprocessed_features(self, feat, extra_config = None):
        # convert np arrays to tensors
        feat = np_to_tensor_dict(feat)

        transforms = transform_fns(self.config.common, self.config.train, extra_config)
        for transform in transforms:
            feat = transform(feat)

        return feat