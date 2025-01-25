from functools import partial
import logging

import torch
import torch.nn as nn

from TopoDiff.model.encoder import Encoder_v1
from TopoDiff.model.diffuser.diffuser import SE3Diffuser
from TopoDiff.model.embedder import InputEmbedder2
from TopoDiff.model.backbone import Backbone2
from TopoDiff.model.aux_head import AuxiliaryHeads
from TopoDiff.model.structure_module import StructureModuleHelper

from myopenfold.utils.tensor_utils import tensor_tree_map
from myopenfold.utils.rigid_utils import Rotation, Rigid
from TopoDiff.utils.debug import print_shape, log_var

from TopoDiff.data.data_transforms import make_one_hot


logger = logging.getLogger("TopoDiff.model.diffusion")

class Diffusion(nn.Module):

    def __init__(self, config, depth = 0, log = False) -> None:
        super().__init__()

        self.depth = depth
        self.log = log

        self.config = config

        if 'Encoder' not in config.Global or config.Global.Encoder is None:
            logger.info('Not using topology encoder.')
            self.encoder = None
        elif config.Global.Encoder == 'Encoder_v1':
            self.encoder = Encoder_v1(config.Encoder_v1, depth = self.depth + 1, log = self.log)
            self.encoder_config = config.Encoder_v1
        else:
            raise NotImplementedError(f'Encoder {config.Global.Encoder} not implemented')

        self.diffuser = SE3Diffuser(config.Diffuser, depth = self.depth + 1, log = self.log)

        if config.Global.Embedder == 'Embedder_v2':
            self.embedder = InputEmbedder2(config.Embedder_v2, depth = self.depth + 1, log = self.log)
            self.latent_dim = self.config.Embedder_v2.topo_embedder.embed_dim
        else:
            raise NotImplementedError(f'Embedder {config.Global.Embedder} not implemented')
        
        if config.Global.Backbone == 'Backbone_v2':
            self.backbone = Backbone2(config.Backbone_v2, depth = self.depth + 1, log = self.log)
        else:
            raise NotImplementedError(f'Backbone {config.Global.Backbone} not implemented')

        self.aux_heads = AuxiliaryHeads(config.Aux_head, depth = self.depth + 1, log = self.log)

        self.dummy_param = nn.Parameter(torch.empty(0), requires_grad = False)

    @property
    def device(self):
        return self.dummy_param.device

    def _log(self, text, tensor = 'None'):
        if self.log:
            log_var(text, tensor, depth = self.depth)


    def forward(self, feat: dict, _return_feat = False):
        """
        Args:
            feat:
                Batched input data.
                Required features:
                    frame_mask: [*, N_res]
                        The mask of the sequence, masked frames will be fixed
                    frame_gt: [*, N_res, 4, 4]
                        The rigid transformation at time step = 0, in 4x4 tensor format
                    frame_gt_mask: [*, N_res]
                        The mask of the ground truth rigid transformation (use for computing loss) 
                    time_step: [*,]
                        The time step to be forward sampled (should include the self-conditioning step)
                    self-condition_step: torch.int

                Optional features (VAE version):
                    encoder_feats: [*, N_res, C_encoder]
                        The input features of the encoder
                    encoder_coords: [*, N_res, 3]
                        The input CA coordinates of the encoder
                    encoder_mask: [*, N_res]
                        The mask of the encoder input
                    encoder_adj_mat: [*, N_res, N_res]
                        The adjacency matrix of the encoder input
        """

        if self.encoder is not None:
            with torch.set_grad_enabled(self.encoder_config.trainable):
                # encoder
                #. [B, latent_dim * 2]
                enc_output = self.encoder(feat)

                # reparameterization
                #. [B, latent_dim]
                latent_mu = enc_output[:, :self.encoder_config.latent_dim]
                #. [B, latent_dim]
                latent_sigma = torch.exp(enc_output[:, self.encoder_config.latent_dim:]) + self.encoder_config.eps
                #. [B, latent_dim]
                latent_z = latent_mu + latent_sigma * torch.randn_like(latent_mu) * self.encoder_config.temperature

                # pack feature
                feat['latent_z'] = latent_z

        with torch.set_grad_enabled(False):
            # forward sample
            frame_noised = self.diffuser.forward_sample_marginal(feat['frame_gt'], feat['timestep'], feat['frame_mask'], intype = 'tensor_4x4', outtype = 'tensor_7')
            
            is_first_tag = True
            prev = None
            # self-conditioning
            for i in range(feat['self_condition_step'][0]):
                res_dict = self.sample_step(feat, frame_noised, prev, reverse_sample = True, inplace_safe = True, is_first = is_first_tag, intype = 'tensor_7', outtype = 'tensor_7')
                seq_emb, pair_emb, frame_hat, frame_denoised = res_dict['seq_emb'], res_dict['pair_emb'], res_dict['frame_hat'], res_dict['frame_denoised']

                # reference management
                frame_hat_trans = frame_hat[..., 4:]
                prev = [seq_emb, pair_emb, frame_hat_trans]
                frame_noised = frame_denoised
                del res_dict, seq_emb, pair_emb, frame_hat, frame_denoised

                feat['timestep'] -= 1
                is_first_tag = False
        
                torch.cuda.empty_cache()
        
        # the grad-computing step
        # according to https://github.com/pytorch/pytorch/issues/65766#issuecomment-932511164, I should add this.
        torch.cuda.empty_cache()
        if torch.is_autocast_enabled():
            torch.clear_autocast_cache()

        res_dict = self.sample_step(feat, frame_noised, prev, reverse_sample = False, inplace_safe = False,
                                    is_first = is_first_tag, is_last = True)
    
        torch.cuda.empty_cache()

        # pack encoder result
        if self.encoder is not None:
            res_dict['latent_mu'] = latent_mu
            res_dict['latent_logvar'] = enc_output[:, self.encoder_config.latent_dim:]
            res_dict['latent_z'] = latent_z
        
        # auxiliary prediction tasks
        res_dict.update(self.aux_heads(res_dict, feat))

        torch.cuda.empty_cache()

        if _return_feat:
            return res_dict, feat

        return res_dict

    def sample_step(self, feat: dict, frame_noised, prev = None, reverse_sample = False, inplace_safe = False, is_first = False, is_last = False,
                    intype = 'tensor_7', outtype = 'tensor_7',
                    translation_reverse_strategy_override = None, 
                    rotation_reverse_strategy_override = None,
                    rotation_reverse_noise_scale_override = None,
                    rotation_reverse_score_scale_override = None,
    ):
        result = {}
        
        # embed the input
        seq_emb, pair_emb, emb_output = self.embedder(feat, prev, inplace_safe, is_first, is_last)

        # predict the ground truth frame
        backbone_output = self.backbone(feat, frame_noised, seq_emb, pair_emb, emb_dict = emb_output, inplace_safe = inplace_safe)

        #. if it is not the last layer, we only need to return these features
        if not is_last:
            result['frame_hat'] = backbone_output['frame_hat']
            result['seq_emb'] = backbone_output['seq_emb']
            result['pair_emb'] = backbone_output['pair_emb']
        else:
            result.update(emb_output)


            result.update(backbone_output)
        del backbone_output

        if reverse_sample:
            # reverse sample
            frame_denoised = self.diffuser.reverse_sample(frame_noised, result['frame_hat'], feat['timestep'], feat['frame_mask'], 
                                                          intype = intype, outtype = outtype,
                                                          translation_reverse_strategy_override = translation_reverse_strategy_override,
                                                          rotation_reverse_strategy_override = rotation_reverse_strategy_override,
                                                          rotation_reverse_noise_scale_override = rotation_reverse_noise_scale_override,
                                                          rotation_reverse_score_scale_override = rotation_reverse_score_scale_override,
                                                          )
            result['frame_denoised'] = frame_denoised

        return result
    

    def _init_feat(self, num_samples = 1, num_res = 150, timestep = 200):
        feat = {}
        feat['timestep'] = torch.ones(num_samples, dtype = torch.long) * timestep
        has_break = torch.zeros(num_samples, num_res, 1, dtype = torch.long)
        seq_type = torch.ones(num_samples, num_res, dtype = torch.long) * 20
        feat['seq_type'] = seq_type
        feat['seq_feat'] = torch.cat([has_break, make_one_hot(seq_type, 21)], dim = -1).float()
        feat['seq_idx'] = torch.arange(num_res, dtype = torch.long).unsqueeze(0).repeat(num_samples, 1)
        feat['seq_mask'] = torch.ones(num_samples, num_res, dtype = torch.bool)
       
        feat['frame_mask'] = torch.ones(num_samples, num_res, dtype = torch.bool)
        feat['frame_gt'] = torch.zeros(num_samples, num_res, 4, 4, dtype = torch.float)
        feat['frame_gt'][..., :3, :3] = torch.eye(3, dtype = torch.float)
        feat['frame_gt'][:, 3, 3] = 1.0

        return feat   
    
    def _prepare_feat(self, latent, num_res, timestep = 200):
        num_samples = 1
        device = self.device

        latent = latent.to(self.device)[None].float()

        feat = {'batch_idx': torch.zeros(num_samples, dtype = torch.long)}
        # initialize features
        feat.update(self._init_feat(num_samples = num_samples, num_res = num_res, timestep = timestep))
        # initialize latent features
        feat['latent_z'] = latent
        feat['latent_mask'] = torch.ones(num_samples, dtype = torch.float32)
        feat = tensor_tree_map(lambda x: x.to(device), feat)
                
        # forward sample
        frame_noised = self.diffuser.forward_sample_marginal(feat['frame_gt'], feat['timestep'], feat['frame_mask'], intype = 'tensor_4x4', outtype = 'tensor_7')

        return feat, frame_noised

    def sample_latent_conditional(self, latent, num_res, return_traj = False, timestep = 200, 
                             return_frame = False, return_position = False, reconstruct_position = False,
                             translation_reverse_strategy_override = None, 
                             rotation_reverse_strategy_override = None,
                             rotation_reverse_noise_scale_override = None,
                             rotation_reverse_score_scale_override = None,
                             force_mask_latent = False,
                             **kwargs):
        """Conditional generation of samples.

        feat(Minimal):
            - timestep [*]
            - seq_type [*, N_res]
            - seq_idx [*, N_res]
            - seq_mask [*, N_res]
            - frame_mask [*, N_res]
            - seq_feat [*, N_res, 22]
        """
        if 'num_samples' in kwargs:
            logger.warning('`num_samples` is deprecated. Will be set to 1 in all cases.')
        num_samples = 1
        device = self.device

        if return_traj:
            if not return_frame and not reconstruct_position:
                raise ValueError('return_traj is True but return_frame and reconstruct_position are both False.')
            if return_frame:
                frame_noised_record = []
                frame_hat_record = []
            if return_position:
                if reconstruct_position:
                    self._prepare_helper()
                coord_noised_record = []
                coord_hat_record = []

        with torch.no_grad():
            latent = latent.to(self.device)[None].float()

            feat = {'batch_idx': torch.zeros(num_samples, dtype = torch.long)}
            #. initialize features
            feat.update(self._init_feat(num_samples = num_samples, num_res = num_res, timestep = timestep))
            #. initialize latent features
            feat['latent_z'] = latent
            if not force_mask_latent:
                feat['latent_mask'] = torch.ones(num_samples, dtype = torch.float32)
            else:
                feat['latent_mask'] = torch.zeros(num_samples, dtype = torch.float32)
            feat = tensor_tree_map(lambda x: x.to(device), feat)
                
            # forward sample
            frame_noised = self.diffuser.forward_sample_marginal(feat['frame_gt'], feat['timestep'], feat['frame_mask'], intype = 'tensor_4x4', outtype = 'tensor_7')

            if return_traj:
                if return_frame:
                    frame_noised_record.append(frame_noised.detach().cpu())
                if return_position:
                    if reconstruct_position:
                        coord_noised = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                            frame_pred = frame_noised, seq_type = feat['seq_type'], intype='tensor_7'
                        )
                    else:
                        raise NotImplementedError('Not implemented yet.')
                    coord_noised_record.append(coord_noised.detach().cpu())

            is_first_tag = True
            prev = None

            # self-conditioning
            for i in range(timestep):
                res_dict = self.sample_step(feat, frame_noised, prev, reverse_sample = True, inplace_safe = True, is_first = is_first_tag, intype = 'tensor_7', outtype = 'tensor_7',
                                    translation_reverse_strategy_override = translation_reverse_strategy_override,
                                    rotation_reverse_strategy_override = rotation_reverse_strategy_override,
                                    rotation_reverse_noise_scale_override = rotation_reverse_noise_scale_override,
                                    rotation_reverse_score_scale_override = rotation_reverse_score_scale_override,
                                    )
                seq_emb, pair_emb, frame_hat, frame_denoised = res_dict['seq_emb'], res_dict['pair_emb'], res_dict['frame_hat'], res_dict['frame_denoised']

                # reference management
                frame_hat_trans = frame_hat[..., 4:]
                prev = [seq_emb, pair_emb, frame_hat_trans]
                frame_noised = frame_denoised
                feat['timestep'] -= 1
                is_first_tag = False

                if return_traj:
                    if return_frame:
                        frame_hat_record.append(frame_hat.detach().cpu())
                        frame_noised_record.append(frame_denoised.detach().cpu())
                    if return_position:
                        if reconstruct_position:
                            coord_hat = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                                frame_pred = frame_hat, seq_type = feat['seq_type'], intype='tensor_7'
                            )
                            coord_denoised = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                                frame_pred = frame_denoised, seq_type = feat['seq_type'], intype='tensor_7'
                            )
                        coord_noised_record.append(coord_denoised.detach().cpu())
                        coord_hat_record.append(coord_hat.detach().cpu())

                del res_dict, seq_emb, pair_emb      # frame_hat, frame_denoised

        result = {}
        result['frame_denoised'] = frame_noised[0].cpu()
        result['frame_hat'] = frame_hat[0].cpu()

        #. concat along the first dimension and reverse the order
        if return_traj:
            if return_frame:
                result['frame_noised_record'] = torch.cat(frame_noised_record[::-1], dim = 0)
                result['frame_hat_record'] = torch.cat(frame_hat_record[::-1], dim = 0)
            if return_position:
                result['coord_noised_record'] = torch.cat(coord_noised_record[::-1], dim = 0)
                result['coord_hat_record'] = torch.cat(coord_hat_record[::-1], dim = 0)
            result['noised_timestep_record'] = torch.arange(timestep+1, dtype = torch.long, device = device)
            result['hat_timestep_record'] = torch.arange(timestep, dtype = torch.long, device = device)

        if reconstruct_position:
            coord_denoised = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                frame_pred = frame_denoised, seq_type = feat['seq_type'], intype='tensor_7'
            )
            coord_hat = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                frame_pred = frame_hat, seq_type = feat['seq_type'], intype='tensor_7'
            )
            result['coord_denosied'] = coord_denoised[0].cpu()
            result['coord_hat'] = coord_hat[0].cpu()

        return result
    
    def sample_latent_conditional_from_feat(self, feat, frame_noised,
                                            return_traj = False, timestep = 200, 
                                            return_frame = False, return_position = False, reconstruct_position = False, 
                                            translation_reverse_strategy_override = None, 
                                            rotation_reverse_strategy_override = None,
                                            rotation_reverse_noise_scale_override = None,
                                            rotation_reverse_score_scale_override = None,
                                            **kwargs):
        """Conditional generation of samples from initialized features.

        feat(Minimal):
            - timestep [*]
            - seq_type [*, N_res]
            - seq_idx [*, N_res]
            - seq_mask [*, N_res]
            - frame_mask [*, N_res]
            - seq_feat [*, N_res, 22]
        """
        if 'num_samples' in kwargs:
            logger.warning('`num_samples` is deprecated. Will be set to 1 in all cases.')
        device = self.device

        if return_traj:
            if not return_frame and not reconstruct_position:
                raise ValueError('return_traj is True but return_frame and reconstruct_position are both False.')
            if return_frame:
                frame_noised_record = []
                frame_hat_record = []
            if return_position:
                if reconstruct_position:
                    self._prepare_helper()
                coord_noised_record = []
                coord_hat_record = []


        with torch.no_grad():
            if return_traj:
                if return_frame:
                    frame_noised_record.append(frame_noised.detach().cpu())
                if return_position:
                    if reconstruct_position:
                        coord_noised = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                            frame_pred = frame_noised, seq_type = feat['seq_type'], intype='tensor_7'
                        )
                    else:
                        raise NotImplementedError('Not implemented yet.')
                    coord_noised_record.append(coord_noised.detach().cpu())

                is_first_tag = True
                prev = None

            # self-conditioning
            for i in range(timestep):
                res_dict = self.sample_step(feat, frame_noised, prev, reverse_sample = True, inplace_safe = True, is_first = is_first_tag, intype = 'tensor_7', outtype = 'tensor_7',
                                    translation_reverse_strategy_override = translation_reverse_strategy_override,
                                    rotation_reverse_strategy_override = rotation_reverse_strategy_override,
                                    rotation_reverse_noise_scale_override = rotation_reverse_noise_scale_override,
                                    rotation_reverse_score_scale_override = rotation_reverse_score_scale_override)
                seq_emb, pair_emb, frame_hat, frame_denoised = res_dict['seq_emb'], res_dict['pair_emb'], res_dict['frame_hat'], res_dict['frame_denoised']

                # reference management
                frame_hat_trans = frame_hat[..., 4:]
                prev = [seq_emb, pair_emb, frame_hat_trans]
                frame_noised = frame_denoised
                feat['timestep'] -= 1
                is_first_tag = False

                if return_traj:
                    if return_frame:
                        frame_hat_record.append(frame_hat.detach().cpu())
                        frame_noised_record.append(frame_denoised.detach().cpu())
                    if return_position:
                        if reconstruct_position:
                            coord_hat = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                                frame_pred = frame_hat, seq_type = feat['seq_type'], intype='tensor_7'
                            )
                            coord_denoised = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                                frame_pred = frame_denoised, seq_type = feat['seq_type'], intype='tensor_7'
                            )
                        coord_noised_record.append(coord_denoised.detach().cpu())
                        coord_hat_record.append(coord_hat.detach().cpu())

                del res_dict, seq_emb, pair_emb      # frame_hat, frame_denoised

        result = {}
        result['frame_denoised'] = frame_noised[0].cpu()
        result['frame_hat'] = frame_hat[0].cpu()

        # concat along the first dimension and reverse the order
        if return_traj:
            if return_frame:
                result['frame_noised_record'] = torch.cat(frame_noised_record[::-1], dim = 0)
                result['frame_hat_record'] = torch.cat(frame_hat_record[::-1], dim = 0)
            if return_position:
                result['coord_noised_record'] = torch.cat(coord_noised_record[::-1], dim = 0)
                result['coord_hat_record'] = torch.cat(coord_hat_record[::-1], dim = 0)
            result['noised_timestep_record'] = torch.arange(timestep+1, dtype = torch.long, device = device)
            result['hat_timestep_record'] = torch.arange(timestep, dtype = torch.long, device = device)

        if reconstruct_position:
            coord_denoised = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                frame_pred = frame_denoised, seq_type = feat['seq_type'], intype='tensor_7'
            )
            coord_hat = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                frame_pred = frame_hat, seq_type = feat['seq_type'], intype='tensor_7'
            )
            result['coord_denosied'] = coord_denoised[0].cpu()
            result['coord_hat'] = coord_hat[0].cpu()

        return result

    
    def _prepare_helper(self):
        """Prepare StructureModuleHelper for reconstructing position."""
        if not hasattr(self, 'helper'):
            self.helper = StructureModuleHelper().to(self.device)
        return

    def encode_topology(self, feat: dict):
        """
        Args:
            feat:
                features for VAE encoder:
                    encoder_feats: [*, N_res, C_encoder]
                        The input features of the encoder
                    encoder_coords: [*, N_res, 3]
                        The input CA coordinates of the encoder
                    encoder_mask: [*, N_res]
                        The mask of the encoder input
                    encoder_adj_mat: [*, N_res, N_res]
                        The adjacency matrix of the encoder input
        """
        assert hasattr(self, 'encoder_config'), '`encoder_config` is not defined.'
        assert hasattr(self, 'encoder'), '`encoder` is not defined.'

        with torch.set_grad_enabled(self.encoder_config.trainable):
            # encoder
            #. [B, latent_dim * 2]
            enc_output = self.encoder(feat)

            # reparameterization
            #. [B, latent_dim]
            latent_mu = enc_output[:, :self.encoder_config.latent_dim]
            #. [B, latent_dim]
            latent_sigma = torch.exp(enc_output[:, self.encoder_config.latent_dim:]) + self.encoder_config.eps
            #. [B, latent_dim]
            latent_z = latent_mu + latent_sigma * torch.randn_like(latent_mu) * self.encoder_config.temperature

        # pack encoder result
        res_dict = {}
        res_dict['latent_mu'] = latent_mu
        res_dict['latent_logvar'] = enc_output[:, self.encoder_config.latent_dim:]
        res_dict['latent_z'] = latent_z

        return res_dict

    def _init_latent(self, num_samples = 1):
        feat = {}
        if self.latent_dim is not None:
            feat['latent_z'] = torch.zeros(num_samples, self.latent_dim, dtype = torch.float32)
        feat['latent_mask'] = torch.zeros(num_samples, dtype = torch.float32)
        return feat

    def sample_unconditional(self, feat = None, init_feat = True, return_traj = False, num_res = 150, timestep = 200, 
                             return_frame = False, return_position = False, reconstruct_position = False, 
                             init_latent = False, 
                             translation_reverse_strategy_override = None, 
                             rotation_reverse_strategy_override = None,
                             rotation_reverse_noise_scale_override = None,
                             rotation_reverse_score_scale_override = None,
                             **kwargs):
        """Unconditional generation of samples.

        feat(Minimal):
            - timestep [*]
            - seq_type [*, N_res]
            - seq_idx [*, N_res]
            - seq_mask [*, N_res]
            - frame_mask [*, N_res]
            - seq_feat [*, N_res, 22]

            (maybe necessary)
            - latent_z [*, latent_dim]
            - latent_mask [*,]
        """
        if 'num_samples' in kwargs:
            logger.warning('`num_samples` is deprecated. Will be set to 1 in all cases.')
        num_samples = 1
        device = self.device

        if return_traj:
            if not return_frame and not reconstruct_position:
                raise ValueError('return_traj is True but return_frame and reconstruct_position are both False.')
            if return_frame:
                frame_noised_record = []
                frame_hat_record = []
            if return_position:
                if reconstruct_position:
                    self._prepare_helper()
                coord_noised_record = []
                coord_hat_record = []

        with torch.no_grad():
            if feat is None or init_feat:
                if feat is None:
                    feat = {'batch_idx': torch.zeros(num_samples, dtype = torch.long)}
                #. initialize features
                feat.update(self._init_feat(num_samples = num_samples, num_res = num_res, timestep = timestep))
                #. initialize latent features
                if init_latent:
                    feat.update(self._init_latent(num_samples = num_samples))
                feat = tensor_tree_map(lambda x: x.to(device), feat)
                
            # forward sample
            frame_noised = self.diffuser.forward_sample_marginal(feat['frame_gt'], feat['timestep'], feat['frame_mask'], intype = 'tensor_4x4', outtype = 'tensor_7')

            if return_traj:
                if return_frame:
                    frame_noised_record.append(frame_noised.detach().cpu())
                if return_position:
                    if reconstruct_position:
                        coord_noised = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                            frame_pred = frame_noised, seq_type = feat['seq_type'], intype='tensor_7'
                        )
                    else:
                        raise NotImplementedError('Not implemented yet.')
                    coord_noised_record.append(coord_noised.detach().cpu())

            is_first_tag = True
            prev = None

            if 'sample_idx' in feat:
                sample_idx = feat['sample_idx']
            else:
                sample_idx = torch.zeros(1, dtype = torch.long)

            for i in range(timestep):
                res_dict = self.sample_step(feat, frame_noised, prev, reverse_sample = True, inplace_safe = True, is_first = is_first_tag, intype = 'tensor_7', outtype = 'tensor_7',
                                    translation_reverse_strategy_override = translation_reverse_strategy_override,
                                    rotation_reverse_strategy_override = rotation_reverse_strategy_override,
                                    rotation_reverse_noise_scale_override = rotation_reverse_noise_scale_override,
                                    rotation_reverse_score_scale_override = rotation_reverse_score_scale_override,
                                    )
                seq_emb, pair_emb, frame_hat, frame_denoised = res_dict['seq_emb'], res_dict['pair_emb'], res_dict['frame_hat'], res_dict['frame_denoised']

                frame_hat_trans = frame_hat[..., 4:]
                prev = [seq_emb, pair_emb, frame_hat_trans]
                frame_noised = frame_denoised
                feat['timestep'] -= 1
                is_first_tag = False

                if return_traj:
                    if return_frame:
                        frame_hat_record.append(frame_hat.detach().cpu())
                        frame_noised_record.append(frame_denoised.detach().cpu())
                    if return_position:
                        if reconstruct_position:
                            coord_hat = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                                frame_pred = frame_hat, seq_type = feat['seq_type'], intype='tensor_7'
                            )
                            coord_denoised = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                                frame_pred = frame_denoised, seq_type = feat['seq_type'], intype='tensor_7'
                            )
                        coord_noised_record.append(coord_denoised.detach().cpu())
                        coord_hat_record.append(coord_hat.detach().cpu())

                del res_dict, seq_emb, pair_emb      # frame_hat, frame_denoised

        result = {}
        result['frame_denoised'] = frame_noised[0].cpu()
        result['frame_hat'] = frame_hat[0].cpu()

        #. concat along the first dimension and reverse the order
        if return_traj:
            if return_frame:
                result['frame_noised_record'] = torch.cat(frame_noised_record[::-1], dim = 0)
                result['frame_hat_record'] = torch.cat(frame_hat_record[::-1], dim = 0)
            if return_position:
                result['coord_noised_record'] = torch.cat(coord_noised_record[::-1], dim = 0)
                result['coord_hat_record'] = torch.cat(coord_hat_record[::-1], dim = 0)
            result['noised_timestep_record'] = torch.arange(timestep+1, dtype = torch.long, device = device)
            result['hat_timestep_record'] = torch.arange(timestep, dtype = torch.long, device = device)

        if reconstruct_position:
            coord_denoised = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                frame_pred = frame_denoised, seq_type = feat['seq_type'], intype='tensor_7'
            )
            coord_hat = self.helper.reconstruct_backbone_position_without_torsion_wrap(
                frame_pred = frame_hat, seq_type = feat['seq_type'], intype='tensor_7'
            )
            result['coord_denosied'] = coord_denoised[0].cpu()
            result['coord_hat'] = coord_hat[0].cpu()

        return result   
