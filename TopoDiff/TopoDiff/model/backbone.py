import torch
import torch.nn as nn
from typing import Tuple, Optional

#. model
from TopoDiff.model.structure_module import StructureModuleHelper

from myopenfold.model.primitives import Linear
from myopenfold.model.structure_module import InvariantPointAttention, StructureModuleTransition, BackboneUpdate, AngleResnet

#. utils
from myopenfold.utils.rigid_utils import Rotation, Rigid
from myopenfold.utils.tensor_utils import dict_multimap, add

#. np
from myopenfold.np import residue_constants as rc

#. debug
from TopoDiff.utils.debug import log_var

class EdgeTransition(nn.Module):
    """From SE3 diffusion
    """
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
                batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed


class Backbone2(nn.Module):
    """
        Backbone network.
    """

    def __init__(self, config_backbone, depth = 0, log = False):
        super(Backbone2, self).__init__()

        self.depth = depth
        self.log = log

        self.config = config_backbone

        self.predict_torsion = self.config.predict_torsion
        self.reconstruct_backbone_atoms = self.config.reconstruct_backbone_atoms
        self.reconstruct_CB = self.config.reconstruct_CB
        self.torsion_index = self.config.torsion_index

        self.trans_scale_factor = self.config.trans_scale_factor
        self.epsilon = self.config.epsilon
        self.inf = self.config.inf
        self.no_blocks = self.config.no_blocks

        if self.reconstruct_backbone_atoms:
            self.structure_helper = StructureModuleHelper()

        self.trunk = nn.ModuleDict()

        for i in range(self.no_blocks):

            self.trunk['ipa_%d' % i] = InvariantPointAttention(
                c_s = self.config.ipa.c_s,
                c_z = self.config.ipa.c_z,
                c_hidden = self.config.ipa.c_hidden,
                no_heads=self.config.ipa.no_heads,
                no_qk_points=self.config.ipa.no_qk_points,
                no_v_points=self.config.ipa.no_v_points,
                inf=self.config.ipa.inf,
                eps=self.config.ipa.eps
            )
            self.trunk[f'ipa_ln_%d' % i] = nn.LayerNorm(self.config.c_s)
            self.trunk[f'skip_embed_%d' % i] = Linear(
                self.config.c_s,
                self.config.c_skip,
                init="final"
            )
            tfmr_in = self.config.c_s + self.config.c_skip
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self.config.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_%d' % i] = torch.nn.TransformerEncoder(
                tfmr_layer, self.config.seq_tfmr_num_layers)
            self.trunk[f'seq_tfmr_post_%d' % i] = Linear(tfmr_in, self.config.c_s, init="final")
            self.trunk[f'node_transition_%d' % i] = StructureModuleTransition(
                c=self.config.c_s,
                num_layers=self.config.no_seq_transition_layers,
                dropout_rate=self.config.seq_transition_dropout_rate,
            )
            self.trunk[f'bb_update_%d' % i] = BackboneUpdate(
                c_s=self.config.c_s
            )

            if i < self.no_blocks - 1:
                self.trunk[f'edge_transition_%d' % i] = EdgeTransition(
                    node_embed_size=self.config.c_s,
                    edge_embed_in=self.config.c_z,
                    edge_embed_out=self.config.c_z,
                )
        
        self.angle_resnet = AngleResnet(
            self.config.c_s,
            self.config.angle_c_resnet,
            self.config.angle_no_resnet_blocks,
            self.config.no_angles,
            self.epsilon,
        )

    def _log(self, text, tensor = 'None'):
        if self.log:
            log_var(text, tensor, depth = self.depth)

    def forward(self,
                feat,
                frame_noised,
                seq_emb,
                pair_emb,
                emb_dict = None,
                inplace_safe = False):
        """
            feat:
                Batched input data.
                Required features:
                    seq_mask: [*, N_res]
            
            frame_noised: (R, T) ([*, N_res, 3, 3], [*, N_res, 3])
                The noised rigid transformations at time step = t

            seq_emb: [*, N_res, C_m]
                The 1d sequence embedding

            pair_emb: [*, N_res, N_res, C_z]
                The 2d pair embedding
        """

        seq_mask = feat["seq_mask"].to(dtype = seq_emb.dtype)
        pair_mask = (seq_mask[..., None] * seq_mask[..., None, :]).to(dtype = pair_emb.dtype)
        diffuse_mask = feat['frame_mask']

        ################################### Structure Module ###################################
        seq_emb_initial = seq_emb * seq_mask[..., None]
        seq_emb = seq_emb_initial

        # initialize the rigid body transformations
        curr_rigids = Rigid.from_tensor_7(frame_noised).scale_translation(1. / self.trans_scale_factor)

        outputs = []

        for i in range(self.no_blocks):
            # 1. Invariant Point Attention
            ipa_emb = self.trunk[f'ipa_%d' % i](
                seq_emb,
                pair_emb,
                curr_rigids,
                seq_mask,
            )
            ipa_emb *= seq_mask[..., None]

            # 2. Linear transformation
            seq_emb = self.trunk[f'ipa_ln_%d' % i](seq_emb + ipa_emb)

            seq_tfmr_in = torch.cat([
                seq_emb, self.trunk[f'skip_embed_%d' % i](seq_emb_initial)
            ], dim=-1)

            # 3. Seq Transformer
            seq_tfmr_out = self.trunk[f'seq_tfmr_%d' % i](
                seq_tfmr_in, src_key_padding_mask = 1 - seq_mask)
            
            # 4. Linear transformation
            seq_emb = seq_emb + self.trunk[f'seq_tfmr_post_%d' % i](seq_tfmr_out)

            # 5. Seq Transition
            seq_emb = self.trunk[f'node_transition_%d' % i](seq_emb)
            seq_emb = seq_emb * seq_mask[..., None]

            # 6. Backbone Update
            rigid_update = self.trunk[f'bb_update_%d' % i](seq_emb)
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, update_mask = diffuse_mask)

            # 7. Edge Transition
            if i < self.no_blocks - 1:
                pair_emb = self.trunk[f'edge_transition_%d' % i](seq_emb, pair_emb)
                pair_emb *= pair_mask[..., None]

            # save intermediate results
            scaled_rigids = curr_rigids.scale_translation(self.trans_scale_factor)
            preds = {
                'frames' : scaled_rigids.to_tensor_7(),
            }

            if self.predict_torsion and i == self.no_blocks - 1:
                unnormalized_angles, angles = self.angle_resnet(seq_emb, seq_emb_initial)
            
                if self.reconstruct_backbone_atoms:
                    backbone_positions = self.structure_helper.reconstruct_backbone_position_wrap(
                        frame_pred=scaled_rigids,
                        torsion_sin_cos=angles,
                        torsion_index=self.torsion_index,
                        seq_type=None,
                        include_CB=self.reconstruct_CB
                    )
            
            outputs.append(preds)

        ################################### pack result ###################################
        result_dict = {}
        result_dict['sm_result'] = dict_multimap(torch.stack, outputs)
        result_dict["seq_emb"] = seq_emb
        result_dict['single_emb'] = seq_emb
        result_dict["pair_emb"] = pair_emb
        result_dict['frame_hat'] = result_dict['sm_result']['frames'][-1]
        if self.predict_torsion:
            result_dict['torsion_angles'] = unnormalized_angles
            result_dict['torsion_angles_normalized'] = angles
        if self.reconstruct_backbone_atoms:
            result_dict['backbone_positions'] = backbone_positions

        return result_dict