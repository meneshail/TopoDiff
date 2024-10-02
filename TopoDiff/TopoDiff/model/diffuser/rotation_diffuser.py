import os
import torch
from torch import nn
import numpy as np

from TopoDiff.utils import so3_utils as so3

class SO3Diffuser(nn.Module):
    def __init__(self, config, log = False, depth = 0, debug = False):
        super(SO3Diffuser, self).__init__()

        self.config = config
        self.log = log
        self.depth = depth
        self.debug = debug

        self.reverse_strategy = self.config.reverse_strategy
        # print('Using reverse strategy: %s' % (self.reverse_strategy))
        if self.reverse_strategy == 'hat_and_noise':
            self.reverse_parm = self.config[self.reverse_strategy]
        else:
            self.reverse_parm = self.config.reverse_parm[self.reverse_strategy]

        self._load_cache()
        self._precompute_time_schedule()

    def forward_sample(self, rot, t, rot_mask = None):
        """sample from the forward distrbution. t-1 -> t

        Args:
            rot (torch.Tensor): [*, N_res, 3, 3]
                The orientation in rotation matrix form at time_step = t - 1.
            rot_mask (torch.Tensor): [*, N_res]
                The mask of the rotation matrix.
            t (torch.Tensor): [*,]
                The current time step.

        Returns:
            torch.Tensor: [*, N_res, 3, 3]
                The orientation in rotation matrix form at time_step = t.
        """

        if rot_mask is None:
            rot_mask = torch.ones(rot.shape[:-2], dtype=torch.bool).to(rot.device)
            
        #. [*, N_res]
        sigma = self.sigma_interval_schedule[t][..., None].expand(rot.shape[:-2])

        #. [*, N_res, 3]
        noise_vec = self.sample_vec(sigma).to(rot.device)

        #. [*, N_res, 3, 3]
        noise_rot = so3.so3_Exp(noise_vec)

        #. [*, N_res, 3, 3]
        rot_t = rot * (~rot_mask[..., None, None]) + rot @ noise_rot * rot_mask[..., None, None]

        if self.debug:
            return rot_t, noise_rot
        return rot_t
    
    def forward_sample_marginal(self, rot, t, rot_mask = None):
        """sample from the forward distrbution. 0 -> t

        Args:
            rot (torch.Tensor): [*, N_res, 3, 3]
                The orientation in rotation matrix form at time_step = 0.
            rot_mask (torch.Tensor): [*, N_res]
                The mask of the rotation matrix.
            t (torch.Tensor): [*,]
                The current time step.

        Returns:
            torch.Tensor: [*, N_res, 3, 3]
                The orientation in rotation matrix form at time_step = t.
        """

        if rot_mask is None:
            rot_mask = torch.ones(rot.shape[:-2], dtype = torch.bool).to(rot.device)

        #. [*, N_res]
        # in torch 2.0, `t` need to be manually passed to cpu
        sigma = self.sigma_schedule[t.cpu()][..., None].expand(rot.shape[:-2])

        #. [*, N_res, 3]
        noise_vec = self.sample_vec(sigma).to(rot.device)

        #. [*, N_res, 3, 3]
        noise_rot = so3.so3_Exp(noise_vec)

        #. [*, N_res, 3, 3]
        rot_t = rot * (~rot_mask[..., None, None]) + rot @ noise_rot * rot_mask[..., None, None]

        if self.debug:
            return rot_t, noise_rot
        return rot_t
    
    def sample_from_noise(self, rot, rot_mask):
        """Sample from the noise distribution.

        Args:
            rot (torch.Tensor): [*, N_res, 3, 3] (could be Nonr)
                The orientation in rotation matrix form at time_step = 0.
            rot_mask (torch.Tensor): [*, N_res]
                The mask of the rotation matrix.

        Returns:
            torch.Tensor: [*, N_res, 3, 3]
                The orientation in rotation matrix form at time_step = t.
        """
        if rot_mask.dtype != torch.bool:
            rot_mask = rot_mask.bool()
        
        if rot is None:
            n_batch_dim = rot_mask.ndim
            rot = torch.eye(3)[(None,) * n_batch_dim].repeat(rot_mask.shape + (1, 1)).to(rot_mask.device)
        
        t = torch.ones(rot_mask.shape[:-1], dtype = torch.long) * self.config.T

        rot_t = self.forward_sample_marginal(rot, t, rot_mask)

        return rot_t

    def reverse_sample(self, rot_t, rot_0_hat, t, rot_mask = None, 
                       reverse_strategy_override = None,
                       reverse_noise_scale_override = None,
                       reverse_score_scale_override = None):
        """Sample from the reverse distribution. t -> t-1.
            This implementation is similar to 'Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models for Protein Structures'
            where each new orientation is sampled as `rot_0_hat` with some additional noise

        Args:
            rot_t (torch.Tensor): [*, N_res, 3, 3]
                The orientation in rotation matrix form at time_step = t.
            rot_0_hat (torch.Tensor): [*, N_res, 3, 3]
                The estimated orientation in rotation matrix form at time_step = 0.
            rot_mask (torch.Tensor): [*, N_res]
                The mask of the rotation matrix.
            t (torch.Tensor): [*,]
        
        Returns:
            torch.Tensor: [*, N_res, 3, 3]
                The orientation in rotation matrix form at time_step = t-1.
        """
        reverse_strategy = self.reverse_strategy if reverse_strategy_override is None else reverse_strategy_override
        if reverse_strategy == 'score_and_noise':
            noise_scale = reverse_noise_scale_override if reverse_noise_scale_override is not None else self.config.reverse_parm['score_and_noise']['noise_scale']
            score_scale = reverse_score_scale_override if reverse_score_scale_override is not None else self.config.reverse_parm['score_and_noise']['score_scale']

        if rot_mask is None:
            rot_mask = torch.ones(rot_t.shape[:-2], dtype=torch.bool)

        if reverse_strategy == 'hat_and_noise':

            #, not adding additional perturb for step t = 1 -> t = 0
            #. [*,] -> [*, N_res]
            # in torch 2.0, `t` need to be manually passed to cpu
            t_cpu = t.cpu()
            sigma = self.sigma_interval_schedule[t_cpu] * self.reverse_noise_schedule[t_cpu]
            sigma[t_cpu == 1] = 0
            sigma = sigma[..., None].expand(rot_t.shape[:-2])

            #. [*, N_res, 3]
            noise_vec = self.sample_vec(sigma).to(rot_t.device)

            #. [*, N_res, 3, 3]
            noise_rot = so3.so3_Exp(noise_vec)

            
            #. [*, N_res, 3, 3]
            rot = rot_t * (~rot_mask[..., None, None]) + rot_0_hat @ noise_rot * rot_mask[..., None, None]

        elif reverse_strategy == "hat_and_noise_2":
            t_cpu = t.cpu()
            sigma = self.sigma_schedule[t_cpu - 1] * self.reverse_noise_schedule[t_cpu]
            sigma[t_cpu == 1] = 0
            sigma = sigma[..., None].expand(rot_t.shape[:-2])

            #. [*, N_res, 3]
            noise_vec = self.sample_vec(sigma).to(rot_t.device)

            #. [*, N_res, 3, 3]
            noise_rot = so3.so3_Exp(noise_vec)

            #. [*, N_res, 3, 3]
            rot = rot_t * (~rot_mask[..., None, None]) + rot_0_hat @ noise_rot * rot_mask[..., None, None]
        
        elif reverse_strategy == 'score_and_noise':
            rot_t1 = self.reverse_sample_vectorized(
                rot_t, 
                rot_0_hat, 
                t, 
                noise_scale = noise_scale,
                score_scale = score_scale,
                eps = self.config.reverse_parm['score_and_noise']['eps'],
                Log_type = self.config.reverse_parm['score_and_noise']['Log_type'],
            )
            rot = rot_t * (~rot_mask[..., None, None]) + rot_t1 * rot_mask[..., None, None]
        
        else:
            raise NotImplementedError

        if self.debug:
            return rot, noise_rot
        return rot

    
    def _load_cache(self):
        # print('Loading cache from %s' % (self.config.cache_dir))

        self.omega_cache = np.concatenate([np.array([0]), np.load(os.path.join(self.config.cache_dir, 'omega_ary.npy'))])
        self.sigma_cache = np.concatenate([np.array([0]), np.load(os.path.join(self.config.cache_dir, 'sigma_ary%s.npy' % (self.config.suffix)))])
        self.cdf_cache = np.load(os.path.join(self.config.cache_dir, 'cdf_ary%s.npy' % (self.config.suffix)))
        n_sigma, n_omega = self.cdf_cache.shape

        self.score_norm_cache = np.load(os.path.join(self.config.cache_dir, 'score_norm%s.npy' % (self.config.suffix)))
        self.score_norm_cache = np.concatenate([self.score_norm_cache[:, [0]], self.score_norm_cache], axis = 1)
        self.score_norm_cache = np.concatenate([self.score_norm_cache[0, :][None], self.score_norm_cache], axis = 0)

        self.cdf_cache = np.concatenate([np.zeros((n_sigma, 1)), self.cdf_cache], axis = 1)
        self.cdf_cache = np.concatenate([np.ones((1, n_omega + 1)), self.cdf_cache], axis = 0)


    def _precompute_time_schedule(self):
        if self.config.schedule == 'linear':
            self.sigma_schedule = torch.cat([torch.tensor([0.]), torch.linspace(self.config.sigma_1, self.config.sigma_T, self.config.T)])
        elif self.config.schedule == 'log':
            self.sigma_schedule = torch.cat([torch.tensor([0.]), torch.exp(torch.linspace(np.log(self.config.sigma_1), np.log(self.config.sigma_T), self.config.T))])
        elif self.config.schedule == 'exp':
            self.sigma_schedule = torch.cat([torch.tensor([0.]), torch.log(torch.linspace(np.exp(self.config.sigma_1), np.exp(self.config.sigma_T), self.config.T))])
        else:
            raise NotImplementedError
        
        self.sigma_interval_schedule = torch.cat([torch.tensor([0.]), torch.sqrt(self.sigma_schedule[1:]**2 - self.sigma_schedule[:-1]**2)])
        if self.config.reverse_strategy == 'hat_and_noise':
            if self.config.hat_and_noise.noise_scale_schedule == 'linear':
                self.reverse_noise_schedule = torch.cat([torch.tensor([0.]), torch.linspace(self.config.hat_and_noise.noise_scale[0], self.config.hat_and_noise.noise_scale[1], self.config.T)])
        elif self.config.reverse_strategy == 'hat_and_noise_2':
            if self.config.reverse_parm.hat_and_noise_2.noise_scale_schedule == 'linear':
                self.reverse_noise_schedule = torch.cat([torch.tensor([0.]), torch.linspace(self.config.reverse_parm.hat_and_noise_2.noise_scale[0], self.config.reverse_parm.hat_and_noise_2.noise_scale[1], self.config.T)])
            else:
                raise NotImplementedError

    
    def _get_cdf_np(self, sigma):
        if np.isscalar(sigma):
            sigma = np.array(sigma)
        n_sigma = len(self.sigma_cache)

        #. [*size,] -> [n_sample,]
        sigma_reshaped = sigma.reshape(-1)

        #. sigma_ary -> [1, n_sigma]
        #. sigma -> [n_sample, 1]
        #. sample_idx -> [n_sample, n_sigma] -> [n_sample,]
        sigma_idx = np.sum(self.sigma_cache[None] < sigma_reshaped[..., None], axis=-1).clip(max = n_sigma - 1)

        #. [*size, n_omega]
        cdf_ary = np.take_along_axis(self.cdf_cache, sigma_idx[..., None].repeat(self.cdf_cache.shape[1], axis = -1), axis=0).reshape(*sigma.shape, -1)

        return cdf_ary

    def sample_omega_np(self, sigma):
        """Sample omega from IGSO3 distribution

        Args:
            sigma: ndarray with arbitrary shape
                Current sigma
            sigma_ary: [n_sigma]
                Array of sigma
            cdf_ary: [n_sigma, n_omega]
                Array of cdf
            omega_ary: [n_omega]
                Array of omega to sample
        
        Returns:
            omega: sahpe of `sigma`
        """
        if np.isscalar(sigma):
            sigma = np.array(sigma)
        n_sigma = len(self.sigma_cache)

        #. [*size,] -> [n_sample,]
        sigma_reshaped = sigma.reshape(-1)

        #. sigma_ary -> [1, n_sigma]
        #. sigma -> [n_sample, 1]
        #. sample_idx -> [n_sample, n_sigma] -> [n_sample,]
        sigma_idx = np.sum(self.sigma_cache[None] < sigma_reshaped[..., None], axis=-1).clip(max = n_sigma - 1)

        #. [n_sample,]
        rd = np.random.rand(*sigma_idx.shape)

        #. [n_sample,] -> [*size,]
        sample = np.array([np.interp(rd[i], self.cdf_cache[sigma_idx[i]], self.omega_cache) for i in range(len(sigma_idx))])
        sample = sample.reshape(sigma.shape)

        return sample


    def sample_vec_np(self, sigma):
        """Sample the rotation vector with omega sampled from IGSO3 distribution

        Args:
            sigma: ndarray with arbitrary shape
                Current sigma
            sigma_ary: [n_sigma]
                Array of sigma
            cdf_ary: [n_sigma, n_omega]
                Array of cdf
            omega_ary: [n_omega]
                Array of omega to sample
        
        Returns:
            vec: shape of [*sigma.shape, 3]

        """
        #. [*sigma.shape, 3]
        vec = np.random.randn(*(*sigma.shape, 3))
        vec /= np.linalg.norm(vec, axis=-1, keepdims=True)

        #. [*sigma.shape, 1]
        omega =  self.sample_omega_np(sigma)[..., None]
        return vec * omega
    
    
    def sample_omega(self, sigma):
        """Sample omega from IGSO3 distribution

        Args:
            sigma: [ndarray, tensor, scalr] with arbitrary shape
                Current sigma
            sigma_ary: [n_sigma]
                Array of sigma
            cdf_ary: [n_sigma, n_omega]
                Array of cdf
            omega_ary: [n_omega]
                Array of omega to sample
        
        Returns:
            omega: sahpe of `sigma`
        """
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.cpu().numpy()
        
        sample = self.sample_omega_np(sigma)

        return torch.from_numpy(sample).float()


    def sample_vec(self, sigma):
        """Sample the rotation vector with omega sampled from IGSO3 distribution

        Args:
            sigma: [ndarray, tensor, scalr] with arbitrary shape
                Current sigma
            sigma_ary: [n_sigma]
                Array of sigma
            cdf_ary: [n_sigma, n_omega]
                Array of cdf
            omega_ary: [n_omega]
                Array of omega to sample
        
        Returns:
            vec: shape of [*sigma.shape, 3]

        """
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.cpu().numpy()

        vec = self.sample_vec_np(sigma)
        return torch.from_numpy(vec).float()

    def score_norm(self, t, omega):
        """Compute the score norm based on the time step and angle

        Args:
            t: timestep
                [*,]
            omega: angles
                [*, N_res] or [*,]
        
        Cache:
            sigma_cache: sigma array
                [n_sigma]
            omega_cache: omega array
                [n_omega]
            score_norm_cache: score norm array
                [n_sigma, n_omega]
            
        Return:
            score_norm: score norm with the same shape as omega
                [*, N_res] or [*,]
        """
        n_sigma = len(self.sigma_cache)

        #. [*size] -> [n_sample,]
        t_reshaped = t.reshape(-1)
        #. [*size, N_res] -> [n_sample, N_res]
        omega_reshaped = omega.reshape(-1, omega.shape[-1])

        #. [n_sample]
        sigma = self.sigma_schedule[t_reshaped.cpu()].numpy()
        #. [n_sample]
        sigma_idx = np.sum(self.sigma_cache[None] < sigma[..., None], axis=-1).clip(max = n_sigma - 1)

        #. [n_sample, N_res]
        score_norm_t_reshaped = np.stack([
            np.interp(
                omega_reshaped[i],
                self.omega_cache,
                self.score_norm_cache[sigma_idx[i]],
            ) for i in range(len(sigma_idx))
        ],
        axis = 0)
        score_norm_t = score_norm_t_reshaped.reshape(*t.shape, -1)

        return score_norm_t

    def reverse_sample_vectorized(self, R_t, R_0, t, noise_scale = 1., score_scale = 1., eps = 1e-8, log = False, Log_type = 2):
        """Given R_t, t and the predicted R_0, sample R_{t-1} based on the computed score and specified noise level and score scale.
        this function is vectorized in the batch dimension
        Args:
            R_t: rotation matrix at time t
                [*, N_res, 3, 3]
            R_0: predicted rotation matrix at time 0
                [*, N_res, 3, 3]
            t: time step
                [*,]
            noise_level (optional): noise level
                scalar
            score_scale (optional): score scale
                scalar
            eps (optional): epsilon for numerical stability
                scalar

        Return:
            R_{t-1}: rotation matrix at time t-1
                [*, N_res, 3, 3]
        """
        device = R_t.device
        t_cpu = t.cpu()

        # compute the rotation vector from R_t to the predicted R_0
        #. [*, N_res, 3, 3]
        R_0t = R_t @ R_0.transpose(-1, -2)

        # use so3_Log_decompose
        #. [*, N_res, 3], [*, N_res]
        vec_0t, omega = so3.so3_Log_decompose(R_0t, eps=eps, type=Log_type)

        # compute the score
        #. [*, N_res]
        score_norm = torch.from_numpy(self.score_norm(t_cpu, omega.cpu())).to(device).float()
        #. [*, N_res, 3]
        score_vec = vec_0t * score_norm[..., None]

        # compute the squared drift
        #. [*, 1, 1]
        drift = (self.sigma_interval_schedule[t_cpu]).to(device)[..., None, None]

        # compute the reverse step of drift term
        #. [*, N_res, 3]
        drift_tangent_vec = drift**2 * score_vec * score_scale

        # sample the noise term
        #. [*, N_res, 3]
        noise_tangent_vec = drift * torch.randn_like(drift_tangent_vec) * noise_scale

        # compute the reverse step
        #. [*, 1, 1]
        t1_mask = (t == 1)[..., None, None]
        #. [*, N_res, 3]
        delta_tangent_vec = -vec_0t * omega[..., None] * t1_mask + (drift_tangent_vec + noise_tangent_vec) * (~t1_mask)
        
        # convert the tangent vector to rotation matrix
        #. [*, N_res, 3, 3]
        R_delta = so3.so3_Exp(delta_tangent_vec)

        # compute the rotation matrix at time t-1
        #. [*, N_res, 3, 3]
        R_t1 = R_delta @ R_t
        
        return R_t1
