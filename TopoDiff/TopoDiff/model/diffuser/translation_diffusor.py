import os
import torch
from torch import nn
import numpy as np

class CartesianDiffuser(nn.Module):
    def __init__(self, config, log = False, depth = 0):
        super(CartesianDiffuser, self).__init__()

        self.config = config
        self.log = log
        self.depth = depth

        self.reverse_strategy = config.reverse_strategy

        self._precompute_time_schedule()


    def forward_sample(self, coord, t, coord_mask, return_delta = False):
        """Sample from the forward distribution. t-1 -> t.
        
        Args:
            coord (torch.Tensor): [*, N_res, 3]
                Cartesian coordinates at time_step = t - 1.
            coord_mask (torch.Tensor): [*, N_res]
                The mask of the coordinates.
            t (torch.Tensor): [*,]
                The current time step.
            
        Returns:
            torch.Tensor: [*, N_res, 3]
                The sampled coordinates at time_step = t.

        """

        if coord_mask.dtype != torch.bool:
            coord_mask = coord_mask.bool()

        #. [*, 1, 1]
        alpha_sqrt = torch.sqrt(self.alpha_schedule[t][..., None, None])
        beta_sqrt = torch.sqrt(self.beta_schedule[t][..., None, None])

        #. [*, N_res, 3]
        e = torch.randn_like(coord)

        #. [*, N_res, 3]
        # coord_mask = coord_mask.float()
        coord_t = coord * (~coord_mask[..., None]) + (alpha_sqrt * coord + beta_sqrt * e) * coord_mask[..., None]

        if return_delta:
            delta = (alpha_sqrt * coord + beta_sqrt * e) - coord
            return coord_t, delta

        return coord_t
    

    #. for test
    def forward_sample_recursive(self, coord, T, coord_mask):
        delta_list = []
        for i in range(1, T+1):
            coord, delta = self.forward_sample(coord, i, coord_mask)
            delta_list.append(delta)
        return coord, delta_list


    def forward_sample_marginal(self, coord, t, coord_mask):
        """Sample from the marginal forward distribution. 0 -> t.

        Args:
            coord (torch.Tensor): [*, N_res, 3]
                Cartesian coordinates at time_step = 0.
            coord_mask (torch.Tensor): [*, N_res]
                The mask of the coordinates.
            t (torch.Tensor): [*,]
                The current time step.

        Returns:
            torch.Tensor: [*, N_res, 3]
                The sampled coordinates at time_step = t.

        """

        if coord_mask.dtype != torch.bool:
            coord_mask = coord_mask.bool()

        #. [*, 1, 1]
        alpha_tilde = self.alpha_tilde_schedule[t][..., None, None]
        alpha_tilde_sqrt = torch.sqrt(alpha_tilde)
        alpha_tilde_rev_sqrt = torch.sqrt(1 - alpha_tilde)

        #. [*, N_res, 3]
        e = torch.randn_like(coord)

        #. [*, N_res, 3]
        coord_t = coord * (~coord_mask[..., None]) + (alpha_tilde_sqrt * coord + alpha_tilde_rev_sqrt * e) * coord_mask[..., None]

        return coord_t
    
    def sample_from_noise(self, coord, coord_mask):
        """Sample from the noise distribution.

        Args:
            coord (torch.Tensor): [*, N_res, 3] (could be None)
                Cartesian coordinates at time_step = 0.
            coord_mask (torch.Tensor): [*, N_res]
                The mask of the coordinates.
        
        Returns:
            torch.Tensor: [*, N_res, 3]
                The sampled coordinates.
        """

        if coord_mask.dtype != torch.bool:
            coord_mask = coord_mask.bool()
        
        if coord is None:
            coord = torch.zeros(coord_mask.shape + (3,), dtype = torch.float32, device = coord_mask.device)
        
        t = torch.ones(coord_mask.shape[:-1], dtype = torch.long) * self.config.T

        coord_t = self.forward_sample_marginal(coord, t, coord_mask)
        
        return coord_t

    def reverse_sample(self, coord_t, coord_0_hat, t, coord_mask, reverse_strategy_override = None):
        """Sample from the reverse distribution. t -> t-1.

        Args:
            coord_t (torch.Tensor): [*, N_res, 3]
                Cartesian coordinates at time_step = t.
            coord_0_hat (torch.Tensor): [*, N_res, 3]
                The estimated coordinates at time_step = 0.
            coord_mask (torch.Tensor): [*, N_res]
                The mask of the coordinates.
            t (torch.Tensor): [*,]
                The current time step.

        Returns:
            torch.Tensor: [*, N_res, 3]
                The sampled coordinates at time_step = t - 1.

        """
        if coord_mask.dtype != torch.bool:
            coord_mask = coord_mask.bool()

        reverse_strategy = self.reverse_strategy if reverse_strategy_override is None else reverse_strategy_override

        if reverse_strategy == 'DDPM':

            #. [*, 1, 1]
            x0_ratio = self.rev_x0_ratio_schedule[t][..., None, None]
            xt_ratio = self.rev_xt_ratio_schedule[t][..., None, None]
            beta_tilde_sqrt = torch.sqrt(self.beta_tilde_schedule[t][..., None, None])

            #. [*, N_res, 3]
            e = torch.randn_like(coord_t)

            #. [*, N_res, 3]
            coord = coord_t * (~coord_mask[..., None]) + (x0_ratio * coord_0_hat + xt_ratio * coord_t + beta_tilde_sqrt * e) * coord_mask[..., None]

        elif reverse_strategy == 'DDIM':

            #. [*, 1, 1]
            x0_ratio = (torch.sqrt(self.alpha_tilde_schedule[t-1]) - 
                        torch.sqrt((1 - self.alpha_tilde_schedule[t-1]) * self.alpha_tilde_schedule[t] / (1 - self.alpha_tilde_schedule[t])))[..., None, None]
            xt_ratio = torch.sqrt((1 - self.alpha_tilde_schedule[t-1]) / (1 - self.alpha_tilde_schedule[t]))[..., None, None]

            #. [*, N_res, 3]
            coord = coord_t * (~coord_mask[..., None]) + (x0_ratio * coord_0_hat + xt_ratio * coord_t) * coord_mask[..., None]

        return coord


    def _precompute_time_schedule(self):
        """The variance schdule.
        """
        #. [T + 1]
        alpha_schedule = torch.cat([torch.tensor([1.0]), torch.linspace(self.config.alpha_1, self.config.alpha_T, self.config.T)])
        #. [T + 1]
        alpha_tilde_schedule = torch.cumprod(alpha_schedule, dim = 0)

        #. [T + 1]
        beta_schedule = 1 - alpha_schedule
        #. [T + 1]
        beta_tilde_schedule = beta_schedule.clone()
        beta_tilde_schedule[1:] = beta_tilde_schedule[1:] * (1 - alpha_tilde_schedule[:-1]) / (1 - alpha_tilde_schedule[1:])

        #. [T + 1]
        rev_x0_ratio_schedule = torch.cat([torch.tensor([1.]), torch.sqrt(alpha_tilde_schedule[:-1]) * beta_schedule[1:] / (1 - alpha_tilde_schedule[1:])])
        #. [T + 1]
        rev_xt_ratio_schedule = torch.cat([torch.tensor([0.]), torch.sqrt(alpha_schedule[1:]) * (1 - alpha_tilde_schedule[:-1]) / (1 - alpha_tilde_schedule[1:])])

        self.register_buffer('alpha_schedule', alpha_schedule)
        self.register_buffer('alpha_tilde_schedule', alpha_tilde_schedule)
        self.register_buffer('beta_schedule', beta_schedule)
        self.register_buffer('beta_tilde_schedule', beta_tilde_schedule)
        self.register_buffer('rev_x0_ratio_schedule', rev_x0_ratio_schedule)
        self.register_buffer('rev_xt_ratio_schedule', rev_xt_ratio_schedule)