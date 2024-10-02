import os
import torch
from torch import nn
import numpy as np

from TopoDiff.utils import so3_utils as so3

from TopoDiff.model.diffuser.rotation_diffuser import SO3Diffuser
from TopoDiff.model.diffuser.translation_diffusor import CartesianDiffuser

from myopenfold.utils import rigid_utils as rigid_utils

class SE3Diffuser(nn.Module):
    def __init__(self, config_diffuser, depth = 0, log = False):
        super(SE3Diffuser, self).__init__()

        self.depth = depth
        self.log = log
        
        self.config = config_diffuser
        self.cartesian_diffuser = CartesianDiffuser(self.config.Cartesian, depth = self.depth + 1, log = self.log)
        self.so3_diffuser = SO3Diffuser(self.config.SO3, depth = self.depth + 1, log = self.log)

        self.trans_scale_factor = self.config.Global.trans_scale_factor


    def forward_sample(self, rigid, t, rigid_mask = None, intype = 'rigid', outtype = 'rigid'):
        """sample from the forward distrbution. t-1 -> t

        Args:
            rigid (Rigid or torch.Tensor): [*, N_res]
                The Rigid transformation to be diffused at time step = t-1
            t (torch.Tensor): [*,]
                The time step
            rigid_mask (torch.Tensor): [*, N_res]
                The mask of the rigid

        Returns:
            rigid: [*, N_res]
                The diffused rigid transformation at time step = t
        """
        # get R & T
        rot, trans = self.convert_input_to_RT(rigid, intype)

        if rigid_mask is None:
            rigid_mask = torch.ones_like(rot[..., 0, 0], dtype=torch.bool)
        if rigid_mask.dtype != torch.bool:
            rigid_mask = rigid_mask.to(dtype = torch.bool)

        # sample from the forward distribution
        ## translation
        trans_noised = self.cartesian_diffuser.forward_sample(trans, t, rigid_mask)

        ## rotation
        rot_noised = self.so3_diffuser.forward_sample(rot, t, rigid_mask)

        # get the diffused rigid transformation
        rigid_noised = self.convert_RT_to_output((rot_noised, trans_noised), outtype)
        return rigid_noised

    
    def forward_sample_marginal(self, rigid, t, rigid_mask = None, intype = 'rigid', outtype = 'rigid'):
        """sample from the forward distrbution. 0 -> t

        Args:
            rigid (Rigid or torch.Tensor): [*, N_res]
                The Rigid transformation to be diffused at time step = 0
            t (torch.Tensor): [*,]
                The time step
            rigid_mask (torch.Tensor): [*, N_res]
                The mask of the rigid

        Returns:
            rigid: [*, N_res]
                The diffused rigid transformation at time step = t
        """
        # get R & T
        rot, trans = self.convert_input_to_RT(rigid, intype)

        if rigid_mask is None:
            rigid_mask = torch.ones_like(rot[..., 0, 0], dtype=torch.bool)
        if rigid_mask.dtype != torch.bool:
            rigid_mask = rigid_mask.to(dtype = torch.bool)

        # sample from the forward distribution
        ## translation
        trans_noised = self.cartesian_diffuser.forward_sample_marginal(trans, t, rigid_mask)

        ## rotation
        rot_noised = self.so3_diffuser.forward_sample_marginal(rot, t, rigid_mask)

        # get the diffused rigid transformation
        rigid_noised = self.convert_RT_to_output((rot_noised, trans_noised), outtype)
        return rigid_noised
    
    def sample_from_noise(self, rigid, rigid_mask = None, intype = 'rigid', outtype = 'rigid'):
        if rigid_mask is None and rigid is not None:
            raise ValueError('rigid_mask should be provided when rigid is not None')
        
        if rigid is None:
            rot, trans = None, None
        else:
            rot, trans = self.convert_input_to_RT(rigid, intype)

        if rigid_mask is None:
            rigid_mask = torch.ones_like(rot[..., 0, 0], dtype=torch.bool)
        if rigid_mask.dtype != torch.bool:
            rigid_mask = rigid_mask.to(dtype = torch.bool)
        
        trans_noise = self.cartesian_diffuser.sample_from_noise(trans, rigid_mask)

        rot_noise = self.so3_diffuser.sample_from_noise(rot, rigid_mask)

        rigid_noise = self.convert_RT_to_output((rot_noise, trans_noise), outtype)
        return rigid_noise
    

    def reverse_sample(self, rigid_t, rigid_0_hat, t, rigid_mask = None, intype = 'rigid', outtype = 'rigid', 
                       translation_reverse_strategy_override = None, 
                       rotation_reverse_strategy_override = None,
                       rotation_reverse_noise_scale_override = None,
                       rotation_reverse_score_scale_override = None):
        """Sample from the reverse distribution. t -> t-1.

        Args:
            rigid_t (Rigid or torch.Tensor): [*, N_res]
                The Rigid transformation at time step = t
            rigid_0_hat (Rigid or torch.Tensor): [*, N_res]
                The Rigid transformation at time step = 0
            t (torch.Tensor): [*,]
                The time step
            rigid_mask (torch.Tensor): [*, N_res]
                The mask of the rigid
        
        Returns:
            rigid: [*, N_res]
                The diffused rigid transformation at time step = t-1
        """
        # get R T
        rot_t, trans_t = self.convert_input_to_RT(rigid_t, intype)
        rot_0_hat, trans_0_hat = self.convert_input_to_RT(rigid_0_hat, intype)

        #. rot_noised [*, N_res]
        if rigid_mask is None:
            rigid_mask = torch.ones_like(rot_t[..., 0, 0], dtype=torch.bool)
        if rigid_mask.dtype != torch.bool:
            rigid_mask = rigid_mask.to(dtype = torch.bool)

        #. sample from the reverse distribution
        ## translation
        trans_denoised = self.cartesian_diffuser.reverse_sample(trans_t, trans_0_hat, t, rigid_mask, reverse_strategy_override = translation_reverse_strategy_override)

        ## rotation
        rot_denoised = self.so3_diffuser.reverse_sample(rot_t, rot_0_hat, t, rigid_mask, 
                                                        reverse_strategy_override = rotation_reverse_strategy_override,
                                                        reverse_noise_scale_override = rotation_reverse_noise_scale_override,
                                                        reverse_score_scale_override = rotation_reverse_score_scale_override,)

        # get the denoised rigid transformation
        rigid_denoised = self.convert_RT_to_output((rot_denoised, trans_denoised), outtype)

        return rigid_denoised


    def convert_input_to_RT(self, rigid, intype = 'rigid'):
        if intype == 'rigid':
            rot, trans = rigid.to_tensor_RT()
        elif intype == 'tensor_4x4':
            rot = rigid[..., :3, :3]
            trans = rigid[..., :3, -1]
        elif intype == 'tensor_7':
            rot, trans = rigid_utils.Rigid.from_tensor_7(rigid).to_tensor_RT()
        elif intype == 'tuple':
            rot, trans = rigid
        else:
            raise NotImplementedError

        trans = trans / self.trans_scale_factor

        return rot, trans
    

    def convert_RT_to_output(self, rigid, outtype = 'rigid'):
        rot, trans = rigid
        trans = trans * self.trans_scale_factor

        if outtype == 'rigid':
            rigid = rigid_utils.Rigid.from_tensor_RT(rot, trans)
        elif outtype == 'tensor_4x4':
            rigid = torch.zeros(*trans.shape[:-1], 4, 4)
            rigid[..., :3, :3] = rot
            rigid[..., :3, -1] = trans
            rigid[..., -1, -1] = 1
        elif outtype == 'tensor_7':
            rigid = rigid_utils.Rigid.from_tensor_RT(rot, trans).to_tensor_7()
        elif outtype == 'tuple':
            rigid = rot, trans
        else:
            raise NotImplementedError

        return rigid

