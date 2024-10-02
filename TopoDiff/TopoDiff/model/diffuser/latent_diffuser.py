import os
import torch
from torch import nn
import numpy as np

from typing import Literal

class LatentDiffuser(nn.Module):
    def __init__(self, config, log = False, depth = 0):
        super(LatentDiffuser, self).__init__()

        self.config = config
        self.log = log
        self.depth = depth

        self.pred_type = self.config.pred_type
        self.reverse_type = self.config.reverse_type

        self._precompute_time_schedule()

    def forward_sample(self, x_t_1, t, result_eps = False):
        """Sample from the forward distribution. t-1 -> t

        Args:
            x_t_1 (torch.Tensor): [*, C]
                latent at timestep = t - 1
            t (torch.Tensor): [*]
                the current timestep
            result_eps (bool, optional):
                whether to return eps. Defaults to False.

        Returns:
            x_t (torch.Tensor): [*, C]
                latent at timestep = t
        """
        #. [*, 1]
        alpha_sqrt = torch.sqrt(self.alpha_schedule[t])[..., None]
        #. [*, 1]
        beta_sqrt = torch.sqrt(self.beta_schedule[t])[..., None]

        #. [*, C]
        eps = torch.randn_like(x_t_1)

        #. [*, C]
        x_t = alpha_sqrt * x_t_1 + beta_sqrt * eps

        if result_eps:
            return x_t, eps
        return x_t
    
    def forward_sample_recursive(self, x_0, T, result_eps = False):
        """Sample from the forward distribution. 0 -> t

        Args:
            x_0 (torch.Tensor): [*, C]
                latent at timestep = 0
            T (int):
                the current timestep
            result_eps (bool, optional):
                whether to return eps. Defaults to False.
        
        Returns:
            x_t (torch.Tensor): [*, C]
                latent at timestep = t
        """
        eps_list = []
        for T_i in range(1, T + 1):
            t_i = torch.ones_like(x_0[..., 0], dtype = torch.long) * T_i
            x_0, eps = self.forward_sample(x_0, t_i, result_eps = True)
            eps_list.append(eps)
        
        if result_eps:
            return x_0, eps_list
        return x_0


    def forward_sample_marginal(self, x_0, t, return_eps = False, print_coef = False):
        """Sample from the forward marginal distribution. 0 -> t

        Args:
            x_0 (torch.Tensor): [*, C]
                latent at timestep = 0
            t (torch.Tensor): [*]
                the current timestep
            return_eps (bool, optional):
                whether to return eps. Defaults to False.
        
        Returns:
            x_t (torch.Tensor): [*, C]
                latent at timestep = t
        """
        #. [B, 1]
        alpha_tilde_sqrt = self.sqrt_alpha_tilde_schedule[t][..., None]
        #. [B, 1]
        alpha_tilde_rev_sqrt = self.sqrt_one_minus_alphas_tilde_schedule[t][..., None]

        #. [B, C]
        eps = torch.randn_like(x_0)

        if print_coef:
            print('alpha_tilde_sqrt', alpha_tilde_sqrt)
            print('alpha_tilde_rev_sqrt', alpha_tilde_rev_sqrt)

        #. [B, C]
        x_t = alpha_tilde_sqrt * x_0 + alpha_tilde_rev_sqrt * eps

        if return_eps:
            return x_t, eps
        return x_t
    
    def reverse_sample(self, pred, x_t, t, reverse_type: Literal['x_0', 'eps-1', 'eps-2'] = None, result_eps = False):
        """Sample from the reverse distribution. t -> t - 1

        Args:
            x_t (torch.Tensor): [*, C]
                latent at timestep = t
            pred (torch.Tensor): [*, C]
                the predicted variable, either x_0 or eps
            t (torch.Tensor): [*]
                the current timestep
            reverse_type (Literal['x_0', 'eps-1', 'eps-2'], optional):
                the type of reverse sampling. Defaults to 'x_0'.
            result_eps (bool, optional):
                whether to return eps. Defaults to False.

        Returns:
            torch.Tensor: [*, C]
                The sampled latent at timestep = t - 1
        """
        if reverse_type is None:
            reverse_type = self.reverse_type

        #. [*, C]
        if reverse_type == 'x_0':
            posterior_mean = self._compute_posterior_mean_x0_xt(pred, x_t, t)
        elif reverse_type == 'eps-1':
            posterior_mean = self._compute_posterior_mean_eps_xt(pred, x_t, t)
        elif reverse_type == 'eps-2':
            pred_x0 = self._compute_pred_x0_eps_xt(pred, x_t, t)
            posterior_mean = self._compute_posterior_mean_x0_xt(pred_x0, x_t, t)
        else:
            raise NotImplementedError
        
        #. [*, 1]
        beta_tilde_sqrt = self.sqrt_beta_tilde_schedule[t][..., None]

        #. [*, C]
        eps = torch.randn_like(x_t)

        #. [*, C]
        x_t_1 =  posterior_mean + beta_tilde_sqrt * eps

        if result_eps:
            return x_t_1, eps
        return x_t_1

    def _compute_posterior_mean_x0_xt(self, pred_x0, x_t, t):
        """Compute the posterior mean given pred_x0 and x_t

        Args:
            pred_x0 (torch.Tensor): [*, C]
                the predicted x_0
            x_t (torch.Tensor): [*, C]
                latent at timestep = t
            t (torch.Tensor): [*]
                the current timestep

        Returns:
            torch.Tensor: [*, C]
                The posterior mean
        """
        #. [*, 1]
        rev_x0_ratio = self.rev_x0_ratio_schedule[t][..., None]
        #. [*, 1]
        rev_xt_ratio = self.rev_xt_ratio_schedule[t][..., None]

        #. [*, C]
        posterior_mean = rev_x0_ratio * pred_x0 + rev_xt_ratio * x_t

        return posterior_mean

    def _compute_posterior_mean_eps_xt(self, pred_eps, x_t, t):
        """Compute the posterior mean given pred_eps and x_t

        Args:
            pred_eps (torch.Tensor): [*, C]
                the predicted eps
            x_t (torch.Tensor): [*, C]
                latent at timestep = t
            t (torch.Tensor): [*]
                the current timestep

        Returns:
            torch.Tensor: [*, C]
                The posterior mean
        """
        #. [*, 1]
        rev_eps_ratio = self.rev_eps_ratio_schedule[t][..., None]
        #. [*, 1]
        sqrt_recip_alpha = self.sqrt_recip_alpha_schedule[t][..., None]

        #. [*, C]
        posterior_mean = sqrt_recip_alpha * x_t - rev_eps_ratio * pred_eps

        return posterior_mean

    def _compute_pred_x0_eps_xt(self, pred_eps, x_t, t):
        """Compute the pred_x0 given pred_eps and x_t

        Args:
            pred_eps (torch.Tensor): [*, C]
                the predicted eps
            x_t (torch.Tensor): [*, C]
                latent at timestep = t
            t (torch.Tensor): [*]
                the current timestep
        
        Returns:
            torch.Tensor: [*, C]
                The pred_x0
        """  
        #. [*, 1]
        sqrt_recip_alpha_tilde = self.sqrt_recip_alpha_tilde_schedule[t][..., None]
        #. [*, 1]
        sqrt_recip_alpha_tilde_m1 = self.sqrt_recip_alpha_tilde_m1_schedule[t][..., None]

        #. [*, C]
        pred_x0 = sqrt_recip_alpha_tilde * x_t - sqrt_recip_alpha_tilde_m1 * pred_eps

        return pred_x0

    def _compute_pred_eps_x0_xt(self, pred_x0, x_t, t):
        """Compute the pred_eps given pred_x0 and x_t

        Args:
            pred_x0 (torch.Tensor): [*, C]
                the predicted x_0
            x_t (torch.Tensor): [*, C]
                latent at timestep = t
            t (torch.Tensor): [*]
                the current timestep
        
        Returns:
            torch.Tensor: [*, C]
                The pred_eps
        """
        #. [*, 1]
        alpha_tilde = self.alpha_tilde_schedule[t][..., None]
        xt_coef = 1 / torch.sqrt(1 - alpha_tilde)
        #. [*, 1]
        x0_coef = 1 / self.sqrt_recip_alpha_tilde_m1_schedule[t][..., None]

        #. [*, C]
        pred_eps = xt_coef * x_t - x0_coef * pred_x0

        return pred_eps


    def _precompute_time_schedule(self):
        """The variance schdule.
        """
        if self.config.schedule == 'linear':
            #. [T + 1]
            alpha_schedule = torch.cat([torch.tensor([1.0]), torch.linspace(1 - self.config.beta_1, 1 - self.config.beta_T, self.config.T)])

            #. [T + 1]
            alpha_tilde_schedule = torch.cumprod(alpha_schedule, dim = 0)
        else:
            raise NotImplementedError

        #. [T + 1]
        beta_schedule = 1 - alpha_schedule

        #. std of posterior mean 
        #. [T + 1]
        beta_tilde_schedule = beta_schedule.clone()
        beta_tilde_schedule[1:] = beta_tilde_schedule[1:] * (1 - alpha_tilde_schedule[:-1]) / (1 - alpha_tilde_schedule[1:])
        sqrt_beta_tilde_schedule = torch.sqrt(beta_tilde_schedule)

        #. x_0 coef of posterior mean
        #. [T + 1]
        rev_x0_ratio_schedule = torch.cat([torch.tensor([1.]), torch.sqrt(alpha_tilde_schedule[:-1]) * beta_schedule[1:] / (1 - alpha_tilde_schedule[1:])])
        #. x_t coef of posterior mean
        #. [T + 1]
        rev_xt_ratio_schedule = torch.cat([torch.tensor([0.]), torch.sqrt(alpha_schedule[1:]) * (1 - alpha_tilde_schedule[:-1]) / (1 - alpha_tilde_schedule[1:])])

        #. coef of forward maginal mean
        #. [T + 1]
        sqrt_alpha_tilde_schedule = torch.sqrt(alpha_tilde_schedule)
        #. coef of forward maginal std
        #. [T + 1]
        sqrt_one_minus_alphas_tilde_schedule = torch.sqrt(1.0 - alpha_tilde_schedule)

        #. coef for x_t when predicting x_0 with eps
        #. [T + 1]
        sqrt_recip_alpha_tilde_schedule = torch.sqrt(1.0 / alpha_tilde_schedule)
        #. coef for eps when predicting x_0 with x_t
        #. [T + 1]
        sqrt_recip_alpha_tilde_m1_schedule = torch.sqrt(1.0 / alpha_tilde_schedule - 1.0)

        #. coef for x_t when directly computing posterior mean with eps
        #. [T + 1]
        sqrt_recip_alpha_schedule = torch.sqrt(1.0 / alpha_schedule)
        #. coef for eps when directly computing posterior mean with x_t
        #. [T + 1]
        rev_eps_ratio_schedule = torch.cat([torch.tensor([0.]), beta_schedule[1:] / torch.sqrt(1.0 - alpha_tilde_schedule[1:]) * sqrt_recip_alpha_schedule[1:] ])


        self.register_buffer('alpha_schedule', alpha_schedule)
        self.register_buffer('alpha_tilde_schedule', alpha_tilde_schedule)
        self.register_buffer('beta_schedule', beta_schedule)
        # self.register_buffer('beta_tilde_schedule', beta_tilde_schedule)
        self.register_buffer('sqrt_beta_tilde_schedule', sqrt_beta_tilde_schedule)

        self.register_buffer('rev_x0_ratio_schedule', rev_x0_ratio_schedule)
        self.register_buffer('rev_xt_ratio_schedule', rev_xt_ratio_schedule)

        self.register_buffer('sqrt_alpha_tilde_schedule', sqrt_alpha_tilde_schedule)
        self.register_buffer('sqrt_one_minus_alphas_tilde_schedule', sqrt_one_minus_alphas_tilde_schedule)

        self.register_buffer('sqrt_recip_alpha_tilde_schedule', sqrt_recip_alpha_tilde_schedule)
        self.register_buffer('sqrt_recip_alpha_tilde_m1_schedule', sqrt_recip_alpha_tilde_m1_schedule)

        self.register_buffer('sqrt_recip_alpha_schedule', sqrt_recip_alpha_schedule)
        self.register_buffer('rev_eps_ratio_schedule', rev_eps_ratio_schedule)


    




