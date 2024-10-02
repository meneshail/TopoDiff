import logging

import torch
import torch.nn as nn

#. model
from TopoDiff.model.latent_backbone import MLPSkipNet
from TopoDiff.model.diffuser.latent_diffuser import LatentDiffuser

#. debug utils
from TopoDiff.utils.debug import print_shape, log_var

logger = logging.getLogger("TopoDiff.model.latent_diffusion")


class LatentDiffusion(nn.Module):

    def __init__(self, config_latent_diffusion, depth = 0, log = False):
        super().__init__()

        self.depth = depth
        self.log = log

        self.config = config_latent_diffusion
        self.data_config = config_latent_diffusion.Data
        self.model_config = config_latent_diffusion.Model

        self.latent_dim = self.config.Global.latent_dim
        self.latent_T = self.config.Global.T

        self.diffuser = LatentDiffuser(self.model_config.Diffuser)

        if self.model_config.Global.Backbone == 'MLPSkipNet':
            self.backbone = MLPSkipNet(self.model_config.MLPSkipNet)
        else:
            raise NotImplementedError

        self.dummy_param = nn.Parameter(torch.empty(0), requires_grad = False)

        if self.data_config.common.normalize.enabled:
            assert self.data_config.common.normalize.mu is not None and self.data_config.common.normalize.mu.shape == (self.latent_dim,)
            assert self.data_config.common.normalize.sigma is not None and self.data_config.common.normalize.sigma.shape == (self.latent_dim,)
            self.norm_mu = nn.Parameter(self.data_config.common.normalize.mu, requires_grad = False)
            self.norm_sigma = nn.Parameter(self.data_config.common.normalize.sigma, requires_grad = False)

    @property
    def device(self):
        return self.dummy_param.device

    def _log(self, text, tensor = 'None'):
        if self.log:
            log_var(text, tensor, depth = self.depth)

    def normalize(self, latent):
        n_dim = latent.dim()
        return (latent - self.norm_mu[(None,) * (n_dim - 1)]) / self.norm_sigma[(None,) * (n_dim - 1)]

    def denormalize(self, latent):
        n_dim = latent.dim()
        return latent * self.norm_sigma[(None,) * (n_dim - 1)] + self.norm_mu[(None,) * (n_dim - 1)]

    def forward(self, feat):
        """
        Args:
            feat:
                Required features:
                    latent_gt: [*, latent_dim]
                        The ground truth latent
                    time_step: [*,]
                        The time step to be forward sampled
        """
        latent_gt = feat['latent_gt']
        if self.data_config.common.normalize.enabled:
            with torch.set_grad_enabled(False):
                latent_gt = self.normalize(latent_gt)

        with torch.set_grad_enabled(False):
            # forward sample
            #. [*, latent_dim]
            latent_noised, latent_eps = self.diffuser.forward_sample_marginal(latent_gt, feat['timestep'], return_eps = True)

        
        # predict from noised latent
        #. [*, latent_dim]
        pred = self.backbone(latent_noised, feat['timestep'])

        # pack result
        result = {}
        result['latent_noised'] = latent_noised
        result['latent_eps'] = latent_eps
        result['pred'] = pred
        result['latent_gt'] = latent_gt

        return result


    def _init_feat(self, n_sample = 8, timestep = None):
        """
        Initialize features for sampling
        """
        if timestep is None:
            timestep = self.latent_T
        feat = {}
        feat['timestep'] = torch.ones((n_sample,), device = self.device, dtype=torch.long) * timestep

        latent_noised = torch.zeros((n_sample, self.latent_dim), device = self.device)
        latent_noised = self.diffuser.forward_sample_marginal(latent_noised, feat['timestep'])
        feat['latent_noised'] = latent_noised

        return feat


    def sample(self, n_sample = 8, timestep = None):
        """
        Sample latent from prior
        
        Args:
            n_sample: int
                The number of samples to be generated
            timestep: int
                The timestep to be sampled
        
        Returns:
            latent: [n_sample, latent_dim]
                The sampled latent
        """
        if timestep is None:
            timestep = self.latent_T

        # init feat
        feat = self._init_feat(n_sample, timestep)
        latent_noised = feat['latent_noised']
        latent_timestep = feat['timestep']

        # sample
        for i in range(timestep):
            #. [n_sample, latent_dim]
            pred = self.backbone(latent_noised, latent_timestep)

            #. [n_sample, latent_dim]
            latent_denoised = self.diffuser.reverse_sample(pred, latent_noised, latent_timestep)

            # update
            latent_timestep -= 1
            latent_noised = latent_denoised

        result = {}
        if self.data_config.common.normalize.enabled:
            result['latent_sample_normalized'] = latent_denoised
            result['latent_sample'] = self.denormalize(latent_denoised)
        else:
            result['latent_sample'] = latent_denoised

        return result

    def sample_from_t(self, latent, timestep):
        """
        Continue sample from a timestep t.
        """
        n_sample = latent.shape[0]

        latent_timestep= torch.ones((n_sample,), device = self.device, dtype=torch.long) * timestep
        latent_noised = self.normalize(latent)

        for i in range(timestep):
            #. [n_sample, latent_dim]
            pred = self.backbone(latent_noised, latent_timestep)

            #. [n_sample, latent_dim]
            latent_denoised = self.diffuser.reverse_sample(pred, latent_noised, latent_timestep)

            # update
            latent_timestep -= 1
            latent_noised = latent_denoised

        result = {}
        if self.data_config.common.normalize.enabled:
            result['latent_sample_normalized'] = latent_denoised
            result['latent_sample'] = self.denormalize(latent_denoised)
        else:
            result['latent_sample'] = latent_denoised

        return result







