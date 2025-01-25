import os
import logging
import json
from typing import Optional, Sequence, List, Any

from functools import partial
import ml_collections as mlc
import numpy as np
import torch

from myopenfold.utils.tensor_utils import tensor_tree_map, dict_multimap
from myopenfold.np import residue_constants as rc

from TopoDiff.data import feature_pipeline

import logging

logger = logging.getLogger("TopoDiff.data.data_modules")

class UnconditionalSamplingDummyCollator:
    def __call__(self, prots):
        return prots[0]

class UnconditionalSamplingDummyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 n_samples: int,
                 length_list = 150):
        super(UnconditionalSamplingDummyDataset, self).__init__()

        self.n_samples = n_samples

        if isinstance(length_list, int):
            self.length_list = [length_list] * n_samples
        elif isinstance(length_list, list):
            assert len(length_list) == n_samples
            self.length_list = length_list
        else:
            raise ValueError("Invalid length_list type")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (idx, self.length_list[idx])

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

class TopoDiffDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, config, stage="train", seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage

        if seed is None:
            logger.info("Using default seed 42")
            seed = 42
        self.seed = seed

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        self.generator = generator

    def _add_batch_properties(self, batch):
        self_condition_step = torch.randint(0, self.config.common.max_self_condition_step + 1, (), generator=self.generator) 
        self_condition_step = self_condition_step.clamp_max(torch.min(batch['timestep']) - 1)
        batch['self_condition_step'] = torch.ones_like(batch['timestep']) * self_condition_step
        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)

class TopoDiffBatchCollator:
    def __init__(self, config_data = None, force_pad_size = None, pad_in_collator = False):
        self.force_pad_size = force_pad_size

        if pad_in_collator or (config_data is not None and config_data.common.pad_in_collator.enabled):
            self.pad_in_collator = True
        else:
            self.pad_in_collator = False

        self.stack_fn = partial(torch.stack, dim=0)
        self.map_fn = dict_multimap

    def __call__(self, prots):
        if self.pad_in_collator:
            if 'encoder_feats' in prots[0]:
                max_len_encoder = max([p['encoder_feats'].shape[0] for p in prots]) if self.force_pad_size is None else self.force_pad_size
                for prot in prots:
                    prot['encoder_feats'] = torch.nn.functional.pad(prot['encoder_feats'], (0, 0, 0, max_len_encoder - prot['encoder_feats'].shape[0]))
                    prot['encoder_coords'] = torch.nn.functional.pad(prot['encoder_coords'], (0, 0, 0, max_len_encoder - prot['encoder_coords'].shape[0]))
                    prot['encoder_mask'] = torch.nn.functional.pad(prot['encoder_mask'], (0, max_len_encoder - prot['encoder_mask'].shape[0]))
                    prot['encoder_adj_mat'] = torch.nn.functional.pad(prot['encoder_adj_mat'], (0, max_len_encoder - prot['encoder_adj_mat'].shape[0], 0, max_len_encoder - prot['encoder_adj_mat'].shape[0]))
                    if 'encoder_frame_gt' in prot:
                        prot['encoder_frame_gt'] = torch.nn.functional.pad(prot['encoder_frame_gt'], (0, 0) * (prot['encoder_frame_gt'].ndim - 1) +
                                                                           (0, max_len_encoder - prot['encoder_frame_gt'].shape[0]))

            max_len_decoder = max([p['seq_idx'].shape[0] for p in prots]) if self.force_pad_size is None else self.force_pad_size
            used_keys = [k for k in ['seq_idx', 'seq_mask', 'seq_type', 'seq_feat', 'frame_gt', 'frame_gt_mask', 'frame_mask', 
                                        'contact_gt', 'contact_gt_mask', 'coord_gt', 'coord_gt_mask',
                                        'motif_mask', 'motif_feat'] if k in prots[0]]
            for prot in prots:
                for key in used_keys:
                    prot[key] = torch.nn.functional.pad(prot[key], (0, 0) * (prot[key].ndim - 1) + (0, max_len_decoder - prot[key].shape[0]))

        return self.map_fn(self.stack_fn, prots)

class TopoDiffTopologyDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_dir: str,
        config: mlc.ConfigDict,
        topo_data_cache_path: Optional[str] = None,
        mode: str = "train", 
        rank = None,
        epoch = 0,
        base_seed = 280421310721,
        extra_config = None,
        **kwargs,
    ):
        """
            Args:
                data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                config:
                    A dataset config object. See openfold.config
                topo_data_cache_path:
                    Path to cache of data_dir generated by
                    scripts/generate_chain_data_cache.py
                mode:
                    currently we only implemented "train"
        """
        super(TopoDiffTopologyDataset, self).__init__()
        self.data_dir = data_dir

        self.config = config

        self.mode = mode

        self.rank = rank if rank is not None else 0
        self.epoch = epoch
        self.base_seed = base_seed

        self.extra_config = extra_config
        
        with open(topo_data_cache_path, "r") as fp:
            self.topo_data_cache = json.load(fp)
        assert isinstance(self.topo_data_cache, dict)
        self._topo_ids = sorted(list(self.topo_data_cache.keys()))
       
        # create a dictionary mapping chain ids to indices
        self._topo_id_to_idx_dict = {
            topo_id: i for i, topo_id in enumerate(self._topo_ids)
        }

        self.feature_pipeline = feature_pipeline.PreprocessedFeaturePipeline(config)

    def topo_id_to_idx(self, chain_id):
        return self._topo_id_to_idx_dict[chain_id]

    def idx_to_topo_id(self, idx):
        return self._topo_ids[idx]

    def __getitem__(self, idx):
        topo_id = self._topo_ids[idx]

        torch.manual_seed(self.base_seed + self.epoch + idx)
        np.random.seed((self.base_seed + self.epoch + idx ) % 2**32)

        path = os.path.join(self.data_dir, topo_id[1:3], topo_id + '.pt')
        data = torch.load(path)

        feats = self.feature_pipeline.process_preprocessed_features(
            data,
            extra_config = self.extra_config,
        )
        feats["batch_idx"] = torch.tensor(
            idx ,  
            dtype=torch.int64,
            device=feats["seq_idx"].device)

        return feats

    def __len__(self):
        return len(self._topo_ids)