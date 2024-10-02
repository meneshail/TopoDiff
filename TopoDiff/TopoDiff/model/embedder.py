import math
import logging

import torch
import torch.nn as nn
from typing import Tuple, Optional

from myopenfold.model.primitives import Linear
from myopenfold.utils.tensor_utils import add

# utils
from TopoDiff.utils.debug import log_var

# functional
from TopoDiff.model.functional import get_timestep_embedding

from functools import partial

logger = logging.getLogger("TopoDiff.model.embedder2")

def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class InputEmbedder2(nn.Module):
    """The main embedding module.

    """
    def __init__(self,
                 config_embedder,
                 depth = 0,
                 log = False,
                 **kwargs):
        super(InputEmbedder2, self).__init__()

        self.config = config_embedder
        self.depth = depth
        self.log = log

        self.eps = config_embedder.eps
        self.inf = config_embedder.inf

        self.tf_dim = config_embedder['tf_dim']
        self.pos_emb_dim = config_embedder['pos_emb_dim']
        self.time_emb_dim = config_embedder['time_emb_dim']
        self.embed_fixed = config_embedder['embed_fixed']
        if self.embed_fixed:
            self.time_proj_dim = self.time_emb_dim + 1
        else:
            self.time_proj_dim = self.time_emb_dim

        self.time_emb_max_positions = config_embedder['time_emb_max_positions']
        self.pos_emb_max_positions = config_embedder['pos_emb_max_positions']

        if self.config.recyc_struct:
            self.recyc_struct = True
            self.recyc_struct_min_bin = config_embedder['recyc_struct_min_bin']
            self.recyc_struct_max_bin = config_embedder['recyc_struct_max_bin']
            self.recyc_struct_no_bin = config_embedder['recyc_struct_no_bin']
        else:
            self.recyc_struct = False

        self.c_s = config_embedder['c_s']
        self.c_z = config_embedder['c_z']

        # feature embedder
        self.time_embedder = partial(get_timestep_embedding, 
                                     embedding_dim = self.time_emb_dim,
                                     max_positions = self.time_emb_max_positions)
        self.pos_embedder = partial(get_index_embedding,
                                   embed_size = self.pos_emb_dim,
                                   max_len = self.pos_emb_max_positions)

        # node projection and embedding
        self.node_tf_projection = nn.Linear(self.tf_dim, self.c_s)
        self.node_time_projection = nn.Linear(self.time_proj_dim, self.c_s)
        self.node_pos_projection = nn.Linear(self.pos_emb_dim, self.c_s)

        self.node_embedder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.c_s, self.c_s),
            nn.ReLU(),
            nn.Linear(self.c_s, self.c_s),
            nn.LayerNorm(self.c_s)
        )

        # edge projection and embedding
        self.edge_tf_projection = nn.Linear(self.tf_dim * 2, self.c_z)
        self.edge_time_projection = nn.Linear(self.time_proj_dim * 2, self.c_z)
        self.edge_pos_projection = nn.Linear(self.pos_emb_dim, self.c_z)
        if self.recyc_struct:
            self.edge_recyc_struct_projection = nn.Linear(self.recyc_struct_no_bin, self.c_z)

        self.edge_embedder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
            nn.LayerNorm(self.c_z)
        )

        if self.config.topo_embedder.enabled:
            if self.config.topo_embedder.type == 'continuous':
                self.topo_mode = 2
                if self.config.topo_embedder.embed_mask:
                    embedder_dim = self.config.topo_embedder.embed_dim + 1
                else:
                    embedder_dim = self.config.topo_embedder.embed_dim
                self.node_topo_projection = torch.nn.Sequential(
                    Linear(embedder_dim, self.c_s),
                    nn.GELU(),
                    Linear(self.c_s, self.c_s, init="final"),
                    nn.LayerNorm(self.c_s)
                )
                self.edge_topo_projection = torch.nn.Sequential(
                    Linear(embedder_dim, self.c_z),
                    nn.GELU(),
                    Linear(self.c_z, self.c_z, init="final"),
                    nn.LayerNorm(self.c_z)
                )
            elif self.config.topo_embedder.type == 'continuous_v2':
                self.topo_mode = 3
                if self.config.topo_embedder.embed_mask:
                    embedder_dim = self.config.topo_embedder.embed_dim + 1
                else:
                    embedder_dim = self.config.topo_embedder.embed_dim

                self.node_topo_shift = Linear(embedder_dim, self.c_s, init="final")
                self.node_topo_scale = Linear(embedder_dim, self.c_s, init="gating")
                self.edge_topo_shift = Linear(embedder_dim, self.c_z, init="final")
                self.edge_topo_scale = Linear(embedder_dim, self.c_z, init="gating")
            else:
                raise NotImplementedError('Other topology embedding methods are not implemented yet.')
        else:
            self.topo_mode = 0

    def _log(self, text, tensor = 'None'):
        if self.log:
            log_var(text, tensor, depth = self.depth)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        """From SE3 diffusion
        """
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float()                                               #  .reshape([num_batch, num_res**2, -1])
    
    def _cross_concat_virtual(self, feats_1d_res, feats_1d_virtual, num_res, num_virtual):
        """
        """
        return torch.cat([
            torch.tile(feats_1d_res[:, :, None, :], (1, 1, num_virtual, 1)),
            torch.tile(feats_1d_virtual[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float()       
    
    def forward(self,
                feat,
                prev,
                inplace_safe: bool = False,
                is_first: bool = False,
                is_last: bool = False,
                frame_noised = None,
                ):
        """Embeds the input features.
            feat:
                Batched input features.
                    seq_idx: [B, N_res] 
                        tensor of residue indices
                    seq_feat: [B, N_res, n_ft] 
                        tensor of residue 1d features (21  one-hot + 1 chain idx)

                    timestep: [B,]
                        tensor of timestep
                    frame_mask: [B, N_res]
                        The mask of the sequence, masked frames will be fixed and set timestep = 0
                    
            prev:
                Embedding of previous iteration (self-conditioning)
                frame_trans_t: [B, N_res, 3]
                    The denoised CA coordinates at time step = t
        """

        batch_dim, n_res = feat['seq_idx'].shape
        device = feat['seq_idx'].device

        output_feat_dict = {}

        ##############################  position  ##############################
        rel_seq_idx = feat['seq_idx'][:, :, None] - feat['seq_idx'][:, None, :]

        seq_emb = self.node_pos_projection(self.pos_embedder(feat['seq_idx']))
        edge_emb = self.edge_pos_projection(self.pos_embedder(rel_seq_idx))

        ##############################  topology  ##############################
        if self.config.topo_embedder.enabled:
            if self.topo_mode == 2:
                if not self.config.topo_embedder.embed_mask:
                    seq_emb = add(seq_emb, self.node_topo_projection(feat['latent_z'])[:, None], inplace=inplace_safe)
                    edge_emb = add(edge_emb, self.edge_topo_projection(feat['latent_z'])[:, None, None], inplace=inplace_safe)
                else:
                    latent_mask = feat['latent_mask'][..., None]
                    latent_z = torch.cat([feat['latent_z'] * latent_mask, latent_mask], dim = -1)
                    seq_emb = add(seq_emb, self.node_topo_projection(latent_z)[:, None], inplace=inplace_safe)
                    edge_emb = add(edge_emb, self.edge_topo_projection(latent_z)[:, None, None], inplace=inplace_safe)

                if is_last:
                    output_feat_dict['seq_top_emb'] = seq_emb.clone()

        ##############################  seq feat  ##############################
        seq_feat_2d = self._cross_concat(feat['seq_feat'], batch_dim, n_res)

        seq_emb = add(seq_emb, self.node_tf_projection(feat['seq_feat']), inplace=inplace_safe)
        edge_emb = add(edge_emb, self.edge_tf_projection(seq_feat_2d), inplace=inplace_safe)

        ##############################  timestep  ##############################
        timestep_per_res = torch.tile(feat['timestep'][:, None], (1, n_res)) * feat['frame_mask']
        timestep_per_res_flat = timestep_per_res.reshape(-1)

        timestep_emb = self.time_embedder(timestep_per_res_flat).reshape((batch_dim, n_res, -1))
        if self.embed_fixed:
            timestep_emb = torch.cat([timestep_emb, feat['frame_mask'][:, :, None]], dim = -1)
        timestep_emb_2d = self._cross_concat(timestep_emb, batch_dim, n_res)

        seq_emb = add(seq_emb, self.node_time_projection(timestep_emb), inplace=inplace_safe)
        edge_emb = add(edge_emb, self.edge_time_projection(timestep_emb_2d), inplace=inplace_safe)
        

        ##############################  recyc struct  ##############################
        # Adapted from recycling embedder of AlphaFold2
        if self.recyc_struct:
            if prev is None:
                x = torch.zeros((batch_dim, n_res, 3), device=device, dtype=seq_emb.dtype)
            else:
                x = prev[2]

            bins = torch.linspace(
                self.recyc_struct_min_bin,
                self.recyc_struct_max_bin,
                self.recyc_struct_no_bin,
                dtype=edge_emb.dtype,
                device=edge_emb.device,
                requires_grad=False,
            )
            squared_bins = bins ** 2
            upper = torch.cat([squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1)

            pair_dis = torch.sum((x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True)

            pair_idx = ((pair_dis > squared_bins) * (pair_dis <= upper)).type(x.dtype)

            edge_emb = add(edge_emb, self.edge_recyc_struct_projection(pair_idx), inplace=inplace_safe)


        ##############################  embedder  ##############################
        seq_emb = self.node_embedder(seq_emb)
        edge_emb = self.edge_embedder(edge_emb)

        ##############################  topology - AdaGN(LN) version ##############################
        if self.config.topo_embedder.enabled:
            if self.topo_mode == 3:
                if self.config.topo_embedder.embed_mask:
                    latent_mask = feat['latent_mask'][..., None]
                    latent_z = torch.cat([feat['latent_z'] * latent_mask, latent_mask], dim = -1)
                else:
                    latent_z = feat['latent_z']

                seq_emb = seq_emb * self.node_topo_scale(latent_z)[:, None] + self.node_topo_shift(latent_z)[:, None]

                edge_emb = edge_emb * self.edge_topo_scale(latent_z)[:, None, None] + self.edge_topo_shift(latent_z)[:, None, None]


        return seq_emb, edge_emb, output_feat_dict



    
