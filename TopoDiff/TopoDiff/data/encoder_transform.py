from typing import Tuple, List, Callable, Any, Dict, Sequence, Optional

import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import normalize

from math import ceil

pos_embed_dim = 64
pos_embed_freq_inv = 2000

class SinusoidalPositionalEncoding(torch.nn.Module):
    """
    Copied from progres
    """
    def __init__(self, channels):
        super().__init__()
        channels = int(ceil(channels / 2) * 2)
        inv_freq = 1.0 / (pos_embed_freq_inv ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        sin_inp_x = torch.einsum("...i,j->...ij", tensor, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x

pos_embedder = SinusoidalPositionalEncoding(pos_embed_dim)

def coords_to_dict(coords, contact_dist = 10.0):
    """
    Modified from progres
    """
    n_res = len(coords)
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords)
    if coords.size(1) != 3:
        raise ValueError("coords must be, or must be convertible to, a tensor of shape (nres, 3)")
    dmap = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0),
                       compute_mode="donot_use_mm_for_euclid_dist")
    contacts = (dmap <= contact_dist).squeeze(0).bool()
    # edge_index = contacts.to_sparse().indices()

    degrees = contacts.sum(dim=0)
    norm_degrees = (degrees / degrees.max()).unsqueeze(1)
    term_features = [[0.0, 0.0] for _ in range(n_res)]
    term_features[ 0][0] = 1.0
    term_features[-1][1] = 1.0
    term_features = torch.tensor(term_features)

    # The tau torsion angle is between 4 consecutive Cα atoms, we assign it to the second Cα
    # This feature breaks mirror invariance
    vec_ab = coords[1:-2] - coords[ :-3]
    vec_bc = coords[2:-1] - coords[1:-2]
    vec_cd = coords[3:  ] - coords[2:-1]
    cross_ab_bc = torch.cross(vec_ab, vec_bc, dim=1)
    cross_bc_cd = torch.cross(vec_bc, vec_cd, dim=1)
    taus = torch.atan2(
        (torch.cross(cross_ab_bc, cross_bc_cd, dim=1) * normalize(vec_bc, dim=1)).sum(dim=1),
        (cross_ab_bc * cross_bc_cd).sum(dim=1),
    )
    taus_pad = torch.cat((
        torch.tensor([0.0]),
        taus / torch.pi, # Convert to range -1 -> 1
        torch.tensor([0.0, 0.0]),
    )).unsqueeze(1)

    pos_embed = pos_embedder(torch.arange(1, n_res + 1))
    x = torch.cat((norm_degrees, term_features, taus_pad, pos_embed), dim=1)
    # data = Data(x=x, edge_index=edge_index, coords=coords)
    data = {
        'encoder_feats': x,
        'encoder_coords': coords,
        'encoder_adj_mat': contacts,
    }
    return data

def add_noise(coord, config_noise, generator = None):
    """
    Args:
        coord [N_res, 3]
    
    Returns:
        coord (noised) [N_res, 3]
    """
    if config_noise.type == 'gaussian':
        noise = torch.randn(coord.shape, generator = generator) * config_noise.std
    else:
        raise NotImplementedError
    
    coord += noise
    return coord