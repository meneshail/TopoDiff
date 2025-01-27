import logging
import torch
from torch.nn import Dropout, Identity, Linear, Sequential, SiLU
from torch.nn.functional import normalize

from TopoDiff.utils.encoder_tensor_utils import batched_index_select

from TopoDiff.utils.debug import print_shape, log_var

import torch.nn as nn
from myopenfold.utils.rigid_utils import Rotation, Rigid
from myopenfold.model.structure_module import InvariantPointAttention
from TopoDiff.model.backbone import EdgeTransition

logger = logging.getLogger("TopoDiff.model.encoder")

# From https://github.com/lucidrains/egnn-pytorch
class EGNN(torch.nn.Module):
    def __init__(
        self,
        config_encoder = None,
        dim = None,
        m_dim = None,
        hidden_egnn_dim = None,
        hidden_edge_dim = None,
        dropout = 0.0,
        init_eps = 1e-3,
    ):
        super().__init__()

        self.config = config_encoder

        if self.config is None:
            self.dim = dim
            self.m_dim = m_dim
            self.dropout = dropout
            self.hidden_edge_dim = hidden_edge_dim
            self.hidden_egnn_dim = hidden_egnn_dim
        else:
            self.dim = self.config.hidden_dim
            self.m_dim = self.config.m_dim
            self.dropout = self.config.dropout
            self.hidden_edge_dim = self.config.hidden_edge_dim
            self.hidden_egnn_dim = self.config.hidden_egnn_dim
                
        self.edge_input_dim = (self.dim * 2) + 1
        dropout = Dropout(self.dropout) if self.dropout > 0 else Identity()

        self.edge_mlp = Sequential(
            Linear(self.edge_input_dim, self.hidden_edge_dim),
            dropout,
            SiLU(),
            Linear(self.hidden_edge_dim, self.m_dim),
            SiLU(),
        )

        self.node_mlp = Sequential(
            Linear(self.dim + self.m_dim, self.dim * 2),
            dropout,
            SiLU(),
            Linear(self.dim * 2, self.dim),
        )

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {Linear}:
            torch.nn.init.normal_(module.weight, std=self.init_eps)

    def forward(self, feats, coors, mask, adj_mat):
        """
        Args:
            feats [B, N_res, N_c]
            coors [B, N_res, 3]
            mask [B, N_res]
            adj_mat [B, N_res, N_res]
        """
        b, n, d, device = *feats.shape, feats.device

        rel_coors = torch.unsqueeze(coors, dim = -2) - torch.unsqueeze(coors, dim = -3)
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        i = j = n

        ranking = rel_dist[..., 0].clone()
        rank_mask = mask[:, :, None] * mask[:, None, :]
        ranking.masked_fill_(~rank_mask, 1e5)

        num_nearest = int(adj_mat.float().sum(dim=-1).max().item())
        valid_radius = 0

        self_mask = torch.eye(n, device=device, dtype=torch.bool)[None]

        adj_mat = adj_mat.masked_fill(self_mask, False)
        ranking.masked_fill_(self_mask, -1.)
        ranking.masked_fill_(adj_mat, 0.)

        nbhd_ranking, nbhd_indices = ranking.topk(num_nearest, dim=-1, largest=False)
        nbhd_mask = nbhd_ranking <= valid_radius

        rel_coors = batched_index_select(rel_coors, nbhd_indices, dim=2)
        rel_dist = batched_index_select(rel_dist, nbhd_indices, dim=2)

        j = num_nearest
        feats_j = batched_index_select(feats, nbhd_indices, dim=1)
        feats_i = torch.unsqueeze(feats, dim = -2) # rearrange(feats, "b i d -> b i () d")
        feats_i, feats_j = torch.broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)
        m_ij = self.edge_mlp(edge_input)

        mask_i = mask[..., None]
        mask_j = batched_index_select(mask, nbhd_indices, dim = 1)
        mask = (mask_i * mask_j) & nbhd_mask

        m_ij_mask = mask[..., None]
        m_ij = m_ij.masked_fill(~m_ij_mask, 0.)
        m_i = m_ij.sum(dim=-2)

        node_mlp_input = torch.cat((feats, m_i), dim = -1)
        node_out = self.node_mlp(node_mlp_input) + feats

        return node_out, coors

# Based on https://github.com/vgsatorras/egnn/blob/main/qm9/models.py
class Model_0(torch.nn.Module):
    def __init__(self, config_encoder, depth = 0, log = False):
        super().__init__()

        self.depth = depth
        self.log = log

        self.feature_dim  = config_encoder.feature_dim
        self.hidden_dim   = config_encoder.hidden_dim
        self.n_layers     = config_encoder.n_layers

        self.layer_type = config_encoder.layer_type

        self.hidden_egnn_dim = config_encoder.hidden_egnn_dim
        self.hidden_edge_dim = config_encoder.hidden_edge_dim

        self.ipa_config = config_encoder.ipa

        self.dropout      = config_encoder.dropout
        self.dropout_final = config_encoder.dropout_final
        self.embedding_size = config_encoder.embedding_size
        self.eps = config_encoder.eps
        self.normalize = config_encoder.normalize
        self.final_init = config_encoder.final_init
        self.reduce = config_encoder.reduce
        self.transformer_config = None if not config_encoder.transformer.enable else config_encoder.transformer
        
        self.node_enc = Linear(self.feature_dim, self.hidden_dim)

        if self.layer_type == 'egnn':  
            self.layers = torch.nn.ModuleList()
            for i in range(self.n_layers):
                self.layers.append(EGNN(
                    dim=self.hidden_dim,
                    m_dim=self.hidden_egnn_dim,
                    hidden_egnn_dim=self.hidden_egnn_dim,
                    hidden_edge_dim=self.hidden_edge_dim,
                    dropout=self.dropout,
                ))
        elif self.layer_type == 'ipa':
            logger.info('Using IPA encoder')
            assert self.ipa_config.c_s == self.hidden_dim
            self.layer_dict = nn.ModuleDict()
            # self.edge_enc = Linear(self.hidden_dim * 2 + 1, self.ipa_config.c_z)
            self.edge_enc = EdgeTransition(
                node_embed_size=self.hidden_dim,
                edge_embed_in=1,
                edge_embed_out=self.ipa_config.c_z,
                )
            
            for i in range(self.n_layers):
                self.layer_dict[f'ipa_{i}'] = InvariantPointAttention(
                    c_s = self.ipa_config.c_s,
                    c_z=self.ipa_config.c_z,
                    c_hidden=self.ipa_config.c_hidden,
                    no_heads=self.ipa_config.no_heads,
                    no_qk_points=self.ipa_config.no_qk_points,
                    no_v_points=self.ipa_config.no_v_points,
                    inf=self.ipa_config.inf,
                    eps=self.ipa_config.eps,
                )
                self.layer_dict[f'ipa_ln_{i}'] = nn.LayerNorm(self.hidden_dim)

                self.layer_dict['ffn_%d' % i] = Sequential(
                    Linear(self.hidden_dim, self.hidden_dim * 2),
                    Dropout(self.dropout) if self.dropout > 0 else Identity(),
                    SiLU(),
                    Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                )

                if i != self.n_layers - 1:
                    self.layer_dict[f'edge_transition_{i}'] = EdgeTransition(
                        node_embed_size=self.hidden_dim,
                        edge_embed_in=self.ipa_config.c_z,
                        edge_embed_out=self.ipa_config.c_z,
                        )
        else:
            raise ValueError(f'Unknown layer type {self.layer_type}')
                
        self.node_dec = torch.nn.Sequential(
            Linear(self.hidden_dim, self.hidden_dim),
            Dropout(self.dropout) if self.dropout > 0 else Identity(),
            SiLU(),
            Linear(self.hidden_dim, self.hidden_dim),
        )
        self.graph_dec = torch.nn.Sequential(
            Linear(self.hidden_dim, self.hidden_dim),
            Dropout(self.dropout) if self.dropout > 0 else Identity(),
            SiLU(),
            Dropout(self.dropout_final) if self.dropout_final > 0 else Identity(),
            Linear(self.hidden_dim, self.embedding_size),
        )

        # initialize final layer
        if self.final_init:
            with torch.no_grad():
                self.graph_dec[-1].weight.fill_(1e-3)
                self.graph_dec[-1].bias.fill_(0)

        # transformer
        if self.transformer_config is not None:
            if self.transformer_config.version == 1:
                self.tfmr_version = 1
                transformer_layer = torch.nn.TransformerEncoderLayer(
                    d_model = self.hidden_dim,
                    nhead = self.transformer_config.n_heads,
                    dim_feedforward = self.hidden_dim,
                    dropout = self.transformer_config.dropout,
                    batch_first = True,
                    norm_first = False
                    )
                self.transformer = torch.nn.TransformerEncoder(
                    encoder_layer = transformer_layer,
                    num_layers = self.transformer_config.n_layers,
                    )
            elif self.transformer_config.version == 2:
                self.tfmr_version = 2
                transformer_layer = torch.nn.TransformerEncoderLayer(
                    d_model = self.hidden_dim,
                    nhead = self.transformer_config.n_heads,
                    dim_feedforward = self.hidden_dim,
                    dropout = self.transformer_config.dropout,
                    batch_first = True,
                    norm_first = False
                    )
                self.transformer = torch.nn.TransformerEncoder(
                    encoder_layer = transformer_layer,
                    num_layers = self.transformer_config.n_layers,
                    )
                self.transformer_post = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                with torch.no_grad():
                    self.transformer_post.weight.fill_(1e-3)
                    self.transformer_post.bias.fill_(0)
            else:
                raise ValueError(f'Unknown transformer version {self.transformer_config.version}')
        else:
            self.tfmr_version = None

        # global pooling
        if self.reduce == 'sum':
            self.reduce_idx = 1
            # self.reduce_fn = torch.sum
        elif self.reduce == 'mean':
            self.reduce_idx = 2
            # self.reduce_fn = torch.mean
        else:
            raise ValueError(f'Unknown reduce function {self.reduce}')

    def forward(self, data): 
        if isinstance(data, dict):
            mode = 'dict'
            device = data["encoder_feats"].device
            feats = data["encoder_feats"]
            coords = data["encoder_coords"]
            adj_mat = data["encoder_adj_mat"]
            mask = data["encoder_mask"]
            if self.layer_type == 'ipa':
                if data['encoder_frame_gt'].shape[-1] == 7:
                    rigids = Rigid.from_tensor_7(data['encoder_frame_gt']).scale_translation(1. / self.ipa_config.trans_scale_factor)
                else:
                    rigids = Rigid.from_tensor_4x4(data['encoder_frame_gt']).scale_translation(1. / self.ipa_config.trans_scale_factor)
        else:
            raise NotImplementedError(f'Unknown input type {type(data)}')

        feats = self.node_enc(feats)

        if self.layer_type == 'egnn':
            for layer in self.layers:
                feats, coords = layer(feats, coords, mask, adj_mat)
        elif self.layer_type == 'ipa':
            edge_feat = self.edge_enc(feats, adj_mat[..., None])
            for i in range(self.n_layers):
                ipa_out = self.layer_dict[f'ipa_{i}'](feats, edge_feat, rigids, mask.to(feats.dtype))
                # feats = feats * mask[..., None]
                feats = self.layer_dict[f'ipa_ln_{i}'](feats + ipa_out)
                feats = self.layer_dict[f'ffn_{i}'](feats)
                if i != self.n_layers - 1:
                    edge_feat = self.layer_dict[f'edge_transition_{i}'](feats, edge_feat)

        if self.tfmr_version is not None:
            if self.tfmr_version == 1:
                feats = self.transformer(feats, src_key_padding_mask=~mask)
            elif self.tfmr_version == 2:
                feats = feats + self.transformer_post(self.transformer(feats, src_key_padding_mask=~mask))
        
        feats = self.node_dec(feats)

        # reduce
        if self.reduce_idx == 1:
            graph_feats = torch.sum(feats * mask.unsqueeze(-1), dim=1)
        elif self.reduce_idx == 2:
            graph_feats = torch.sum(feats * mask.unsqueeze(-1), dim=1) / torch.sum(mask, dim=-1, keepdim=True)

        out = self.graph_dec(graph_feats)

        if self.normalize:
            out = normalize(out, dim=1, eps=self.eps)

        return out
    
Encoder_v1 = Model_0