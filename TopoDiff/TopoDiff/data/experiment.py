import os

import torch
import numpy as np

import mdtraj as md
from tmtools import tm_align

class TaskAllocationDummyDataset(torch.utils.data.Dataset):
    """
    Dataset for task or data simple allocation.
    """
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

class TMAlignDataset(torch.utils.data.Dataset):
    def __init__(self, coord_dict_1, 
                 coord_dict_2 = None,
                 key_1 = None,
                 key_2 = None,
                 idx_list = None,
                 log = False):
        self.coord_dict_1 = coord_dict_1
        self.coord_dict_2 = coord_dict_2
        self.log = log

        if coord_dict_2 is None:
            self.mode = 'inner'
            self._log('mode: inner')
            if key_1 is not None:
                self._log('Using provided key_1')
                self.key_1 = key_1
                assert set(self.key_1) <= set(coord_dict_1.keys())
            else:
                self._log('Using sorted(coord_dict_1.keys()) as key_1')
                self.key_1 = sorted(list(coord_dict_1.keys()))
            self.key_2 = None

            if idx_list is not None:
                self._log('Using provided idx_list')
                self.idx_list = idx_list
            else:
                self._log('Using all combinations of key_1 as idx_list')
                self.idx_list = []
                for i in range(len(self.key_1)):
                    for j in range(i, len(self.key_1)):
                        self.idx_list.append((i, j))

        else:
            self.mode = 'outer'
            self._log('mode: outer')

            if key_1 is not None:
                self._log('Using provided key_1')
                self.key_1 = key_1
                assert set(self.key_1) <= set(coord_dict_1.keys())
            else:
                self._log('Using sorted(coord_dict_1.keys()) as key_1')
                self.key_1 = sorted(list(coord_dict_1.keys()))
            
            if key_2 is not None:
                self._log('Using provided key_2')
                self.key_2 = key_2
                assert set(self.key_2) <= set(coord_dict_2.keys())
            else:
                self._log('Using sorted(coord_dict_2.keys()) as key_2')
                self.key_2 = sorted(list(coord_dict_2.keys()))

            if idx_list is not None:
                self._log('Using provided idx_list')
                self.idx_list = idx_list
            else:
                self._log('Using all combinations of key_1 and key_2 as idx_list')
                self.idx_list = []
                for i in range(len(self.key_1)):
                    for j in range(len(self.key_2)):
                        self.idx_list.append((i, j))
        
    def _log(self, msg):
        if self.log:
            print(msg)

    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, idx):
        if self.mode == 'inner':
            i, j = self.idx_list[idx]
            key_1 = self.key_1[i]
            key_2 = self.key_1[j]
            coords_1 = self.coord_dict_1[key_1]
            coords_2 = self.coord_dict_1[key_2]
            seq_1 = 'A' * len(coords_1)
            seq_2 = 'A' * len(coords_2)
        elif self.mode == 'outer':
            i, j = self.idx_list[idx]
            key_1 = self.key_1[i]
            key_2 = self.key_2[j]
            coords_1 = self.coord_dict_1[key_1]
            coords_2 = self.coord_dict_2[key_2]
            seq_1 = 'A' * len(coords_1)
            seq_2 = 'A' * len(coords_2)
        else:
            raise ValueError('mode not supported')
        
        res = tm_align(coords_1, coords_2, seq_1, seq_2)

        res_dict = {}
        res_dict['i'] = i
        res_dict['j'] = j
        res_dict['key_1'] = key_1
        res_dict['key_2'] = key_2
        res_dict['t'] = res.t
        res_dict['u'] = res.u
        res_dict['tm_norm_chain1'] = res.tm_norm_chain1
        res_dict['tm_norm_chain2'] = res.tm_norm_chain2

        return res_dict

class ReadCoordDataset(TaskAllocationDummyDataset):
    def __getitem__(self, idx):
        data_info = self.data_list[idx]

        pdb_path = data_info['pdb_path']
        traj = md.load_pdb(pdb_path)
        ca_idx = traj.top.select('name CA')
        ca_coord = traj.xyz[0, ca_idx] * 10

        data_info['ca_coord'] = ca_coord
        return data_info