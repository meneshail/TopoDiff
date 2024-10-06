import os
import sys
import logging
from tqdm import tqdm
import pickle

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TopoDiff.experiment.sampler")

#. model
from TopoDiff.model.diffusion import Diffusion
from TopoDiff.model.aux_head import SCHead
from TopoDiff.model.latent_diffusion import LatentDiffusion

#. config
from TopoDiff.config.config import model_config
from TopoDiff.config.latent_config import latent_model_config
from TopoDiff.config.head_config import pred_head_model_config

#. data
from TopoDiff.data.data_modules import LatentContionalSamplingDataset2, UnconditionalSamplingDummyCollator
from TopoDiff.data.structure import StructureBuilder

#. utils
from myopenfold.utils.tensor_utils import tensor_tree_map
from TopoDiff.utils.debug import print_shape

#. pdb
from myopenfold.np import protein

default_sample_config = {
    'label': 'default sampling',
    
    'mode_preset': None,

    'pred_sc': False,
    'min_sc': 0.0,
    'max_sc': np.Inf,
    'pred_novelty': False,
    'min_novelty': 0.0,
    'max_novelty': np.Inf,
    'pred_alpha': False,
    'min_alpha': 0.0,
    'max_alpha': np.Inf,
    'pred_beta': False,
    'min_beta': 0.0,
    'max_beta': np.Inf,
    'pred_coil': False,
    'min_coil': 0.0,
    'max_coil': np.Inf,
    
    # 'latent_batch_size': 10000,

    'length_epsilon': 0.2,
    'length_k': 1,

    'soft_prob': 0.1,

    # 'latent_seed': None,
    # 'structure_seed': None,

}

project_dir = os.path.dirname(os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Sampler:
    def __init__(self,

                 model_version,

                 output_dir = None,

                 rank = 0,
                 world_size = 1,

                 sample_config = None,

                 extra = None,
                ):
        """
        Args:
            'setting_diffusion'
                'ckpt_path': str
                'model_preset': str
            'setting_latent'
                'ckpt_path': str
                'model_preset': str
            'setting_sc_head'
                'ckpt_path': str
                'model_preset': str
                    if None, the config will be queried from the config of the base diffusion model

            'label_dict_path': str
                path to the label dict
            'label_dict': dict
                label dict

            'sample_config': dict
                sample_goal_list: list of tuple
                    each tuple is in the form of (length, repeat)
                gpu_list: list of int
                    list of gpu id

                'latent_seed': int
                    seed for sampling latent
                'structure_seed': int
                    seed for sampling structure

                length_epsilon: float
                    epsilon for length prediction
                length_k: int
                    k for length prediction
                
                pred_sc: bool
                    whether to predict sc and enforce as cutoff
                min_sc: float
                    minimum sc
                max_sc: float
                    maximum sc

                pred_novelty: bool
                    whether to predict novelty and enforce as cutoff
                min_novelty: float
                    minimum novelty
                max_novelty: float
                    maximum novelty

                pred_alpha: bool
                pred_beta: bool
                pred_coil: bool
                    
                latent_batch_size: int
                    batch size for latent sampling
        """
        self.output_dir = output_dir
        if self.output_dir is None:
            logger.warning('output_dir is not provided, will return latent and structure result in memory')
            if world_size > 1:
                raise ValueError('output_dir is not provided, cannot use multi-GPU')
             
        # self.config = sampling_config

        self.rank = rank
        self.world_size = world_size

        if self.rank != 0:
            # print('rank', self.rank, os.environ['LOCAL_RANK'])
            # print('world_size', self.world_size, os.environ['WORLD_SIZE'])
            assert self.rank == int(os.environ['LOCAL_RANK']), 'rank %d != os.environ[LOCAL_RANK] %d' % (self.rank, int(os.environ['LOCAL_RANK']))
            assert self.world_size == int(os.environ['WORLD_SIZE']), 'world_size %d != os.environ[WORLD_SIZE] %d' % (self.world_size, int(os.environ['WORLD_SIZE']))

        # init settings for specified version
        logger.debug('init settings for version %s' % model_version)
        self.model_version = model_version
        self._init_version()

        if sample_config is None:
            logger.debug('Sample config is not provided, will wait for user input')
        self.sample_config = sample_config

        # added for score sampling parameter
        # print(self.setting_diffusion)
        if 'sampling_parm' in self.setting_diffusion:
            logger.debug('Using custom diffusion sampling parameter: %s' % self.setting_diffusion['sampling_parm'])
            self.diffusion_sampling_parm = self.setting_diffusion['sampling_parm']
        else:
            logger.debug('Using default diffusion sampling parameter')
            self.diffusion_sampling_parm = None

        self.extra = {} if extra is None else extra

        self._init_multi_gpu()
        self._init_model()

        self.structure_builder = StructureBuilder()

    def _init_version(self):
        if self.model_version == 'v1_1_2':
            model_dir = os.path.join(project_dir, 'data', 'weights', 'v1_1_2')

            ########## diffusion ##########
            self.setting_diffusion = {
                'ckpt_path': os.path.join(model_dir, 'model.ckpt'),
                'model_preset': 'v1_1_2',
                'extra': [],
                'sampling_parm': {
                    'rotation_reverse_score_scale_override': 2,
                    'rotation_reverse_noise_scale_override': 0,
                },
            }

            ########## latent ##########
            self.setting_latent = {
                'model_preset': 'model_1',
            }

            ########## sc head ##########
            self.setting_sc_head = {'model_preset': 'candidate_1',}
            self.setting_novelty_head = {'model_preset': 'candidate_1',}
            self.setting_alpha_head = {'model_preset': 'candidate_1'}
            self.setting_beta_head = {'model_preset': 'candidate_1'}
            self.setting_coil_head = {'model_preset': 'candidate_1'}

        elif self.model_version == 'v1_1_1':
            model_dir = os.path.join(project_dir, 'data', 'weights', 'v1_1_1')

            ########## diffusion ##########
            self.setting_diffusion = {
                'ckpt_path': os.path.join(model_dir, 'model.ckpt'),
                'model_preset': 'v1_1_1',
                'extra': [],
                'sampling_parm': {
                    'rotation_reverse_score_scale_override': 3,
                    'rotation_reverse_noise_scale_override': 0,
                },
            }

            ########## latent ##########
            self.setting_latent = {
                'model_preset': 'model_1',
            }

            ########## sc head ##########
            self.setting_sc_head = {'model_preset': 'candidate_1',}
            self.setting_novelty_head = {'model_preset': 'candidate_1',}
            self.setting_alpha_head = {'model_preset': 'candidate_1'}
            self.setting_beta_head = {'model_preset': 'candidate_1'}
            self.setting_coil_head = {'model_preset': 'candidate_1'}

        else:
            raise NotImplementedError('model_version %s is not implemented' % self.model_version)


    def _init_multi_gpu(self):
        if self.world_size > 1:
            logger.info("Rank %d: Using multi-GPU, total gpus: %d" % (self.rank, self.world_size))
            self.use_multi_gpu = True
        else:
            logger.info("Using single-GPU")
            self.use_multi_gpu = False

    def _init_model(self):

        ckpt = torch.load(self.setting_diffusion['ckpt_path'], map_location='cpu', weights_only=False)

        logger.debug('Loading structure diffusion model...')
        self.config_diffusion = model_config(self.setting_diffusion['model_preset'],
                                             extra = self.setting_diffusion['extra'] if 'extra' in self.setting_diffusion else None)
        self.config_diffusion.Model.Diffuser.SO3.cache_dir = os.path.join(project_dir, 'TopoDiff', 'cache')
        ckpt_diffusion = ckpt['main_ckpt']
        self.model_diffusion = Diffusion(self.config_diffusion.Model)
        missing_keys, unexpected_keys = self.model_diffusion.load_state_dict(ckpt_diffusion, strict=False)
        if len(missing_keys) > 0:
            logger.warning('Missing keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0 and set(unexpected_keys) != set( ['helper.restype_backbone_rigid_group_default_frame', 'helper.restype_backbone_rigid_group_positions', 'backbone.structure_helper.restype_backbone_rigid_group_default_frame', 'backbone.structure_helper.restype_backbone_rigid_group_positions']):
            logger.warning('Unexpected keys: {}'.format(unexpected_keys))
        del ckpt_diffusion

        logger.debug('Loading latent diffusion model...')
        self.config_latent = latent_model_config(self.setting_latent['model_preset'])
        ckpt_latent = ckpt['latent_ckpt']
        self.config_latent.Data.common.normalize.mu = ckpt_latent['norm_mu']
        self.config_latent.Data.common.normalize.sigma = ckpt_latent['norm_sigma']
        self.model_latent = LatentDiffusion(self.config_latent)
        missing_keys, unexpected_keys = self.model_latent.load_state_dict(ckpt_latent, strict=False)
        if len(missing_keys) > 0:
            logger.warning('Missing keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            logger.warning('Unexpected keys: {}'.format(unexpected_keys))
        del ckpt_latent
        
        if self.setting_sc_head is not None:
            logger.debug('Loading designability prediction model...')
            if 'model_config' not in self.setting_sc_head or self.setting_sc_head['model_config'] is None:
                if 'model_preset' in self.setting_sc_head:
                    logger.debug('SC head: using model_preset %s for config' % self.setting_sc_head['model_preset'])
                    self.config_sc_head = pred_head_model_config(self.setting_sc_head['model_preset'],
                                                                extra = self.setting_sc_head['extra'] if 'extra' in self.setting_sc_head else None).Model
                else:
                    logger.warning('SC head: model_preset is not provided, using default config')
                    self.config_sc_head = self.config_diffusion.Model.Aux_head.SC
            else:
                self.config_sc_head = self.setting_sc_head['model_config']
            ckpt_sc_head = ckpt['sc_head_ckpt']
            self.model_sc = SCHead(self.config_sc_head)
            missing_keys, unexpected_keys = self.model_sc.load_state_dict(ckpt_sc_head, strict=False)
            if len(missing_keys) > 0:
                logger.warning('Missing keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                logger.warning('Unexpected keys: {}'.format(unexpected_keys))
            del ckpt_sc_head
        else:
            logger.info('No designability prediction model provided, skip designability prediction')

        if self.setting_novelty_head is not None:
            logger.debug('Loading novelty prediction model...')
            if 'model_config' not in self.setting_novelty_head or self.setting_novelty_head['model_config'] is None:
                if 'model_preset' in self.setting_novelty_head:
                    logger.debug('Novelty head: using model_preset %s for config' % self.setting_novelty_head['model_preset'])
                    self.config_novelty_head = pred_head_model_config(self.setting_novelty_head['model_preset'],
                                                                extra = self.setting_novelty_head['extra'] if 'extra' in self.setting_novelty_head else None).Model
                else:
                    logger.warning('Novelty head: model_preset is not provided, using default config')
                    self.config_novelty_head = self.config_diffusion.Model.Aux_head.SC
            else:
                self.config_novelty_head = self.setting_novelty_head['model_config']
            ckpt_novelty_head = ckpt['novelty_head_ckpt']
            self.model_novelty = SCHead(self.config_novelty_head)
            missing_keys, unexpected_keys = self.model_novelty.load_state_dict(ckpt_novelty_head, strict=False)
            if len(missing_keys) > 0:
                logger.warning('Missing keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                logger.warning('Unexpected keys: {}'.format(unexpected_keys))
            del ckpt_novelty_head
        else:
            logger.info('No novelty prediction model provided, skip novelty prediction')

        if self.setting_alpha_head is not None:
            logger.debug('Loading alpha prediction model...')
            if 'model_config' not in self.setting_alpha_head or self.setting_alpha_head['model_config'] is None:
                if 'model_preset' in self.setting_alpha_head:
                    logger.debug('Alpha head: using model_preset %s for config' % self.setting_alpha_head['model_preset'])
                    self.config_alpha_head = pred_head_model_config(self.setting_alpha_head['model_preset'],
                                                                extra = self.setting_alpha_head['extra'] if 'extra' in self.setting_alpha_head else None).Model
                else:
                    logger.warning('Alpha head: model_preset is not provided, using default config')
                    self.config_alpha_head = self.config_diffusion.Model.Aux_head.SC
            else:
                self.config_alpha_head = self.setting_alpha_head['model_config']
            ckpt_alpha_head = ckpt['alpha_head_ckpt']
            self.model_alpha = SCHead(self.config_alpha_head)
            missing_keys, unexpected_keys = self.model_alpha.load_state_dict(ckpt_alpha_head, strict=False)
            if len(missing_keys) > 0:
                logger.warning('Missing keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                logger.warning('Unexpected keys: {}'.format(unexpected_keys))
            del ckpt_alpha_head

        if self.setting_beta_head is not None:
            logger.debug('Loading beta prediction model...')
            if 'model_config' not in self.setting_beta_head or self.setting_beta_head['model_config'] is None:
                if 'model_preset' in self.setting_beta_head:
                    logger.debug('Beta head: using model_preset %s for config' % self.setting_beta_head['model_preset'])
                    self.config_beta_head = pred_head_model_config(self.setting_beta_head['model_preset'],
                                                                extra = self.setting_beta_head['extra'] if 'extra' in self.setting_beta_head else None).Model
                else:
                    logger.warning('Beta head: model_preset is not provided, using default config')
                    self.config_beta_head = self.config_diffusion.Model.Aux_head.SC
            else:
                self.config_beta_head = self.setting_beta_head['model_config']
            ckpt_beta_head = ckpt['beta_head_ckpt']
            self.model_beta = SCHead(self.config_beta_head)
            missing_keys, unexpected_keys = self.model_beta.load_state_dict(ckpt_beta_head, strict=False)
            if len(missing_keys) > 0:
                logger.warning('Missing keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                logger.warning('Unexpected keys: {}'.format(unexpected_keys))
            del ckpt_beta_head

        if self.setting_coil_head is not None:
            logger.debug('Loading coil prediction model...')
            if 'model_config' not in self.setting_coil_head or self.setting_coil_head['model_config'] is None:
                if 'model_preset' in self.setting_coil_head:
                    logger.debug('Coil head: using model_preset %s for config' % self.setting_coil_head['model_preset'])
                    self.config_coil_head = pred_head_model_config(self.setting_coil_head['model_preset'],
                                                                extra = self.setting_coil_head['extra'] if 'extra' in self.setting_coil_head else None).Model
                else:
                    logger.warning('Coil head: model_preset is not provided, using default config')
                    self.config_coil_head = self.config_diffusion.Model.Aux_head.SC
            else:
                self.config_coil_head = self.setting_coil_head['model_config']
            ckpt_coil_head = ckpt['coil_head_ckpt']
            self.model_coil = SCHead(self.config_coil_head)
            missing_keys, unexpected_keys = self.model_coil.load_state_dict(ckpt_coil_head, strict=False)
            if len(missing_keys) > 0:
                logger.warning('Missing keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                logger.warning('Unexpected keys: {}'.format(unexpected_keys))
            del ckpt_coil_head

        self.label_dict = ckpt['embedding_dict']
        if 'mode_preset' in ckpt:
            self.mode_preset_settings = ckpt['mode_preset']

        del ckpt

        self.model_diffusion.to(self.rank)
        self.model_latent.to(self.rank)
        self.model_latent.eval()

        self.latent_dim = self.config_diffusion.Model.Embedder_v2.topo_embedder.embed_dim

    def _process_sample_config(self):

        if 'sample_goal_list' not in self.sample_config:
            raise ValueError('Error: sample_goal_list is not provided in sample_config')
        num_total_sample = sum([goal[1] for goal in self.sample_config['sample_goal_list']])
        if num_total_sample > 100 and self.output_dir is None:
            raise ValueError('Error: the total number of samples is larger than 100, please provide output_dir to save the result')
        
        # set default values
        default_sample_config['latent_batch_size'] = min(10000, num_total_sample * 50)
        default_sample_config['latent_seed'] = np.random.randint(0, 147483647)
        default_sample_config['structure_seed'] = np.random.randint(0, 147483647)

        for k, v in default_sample_config.items():
            if k not in self.sample_config or self.sample_config[k] is None:
                self.sample_config[k] = v

        if self.sample_config['mode_preset'] is None:
            # when no preset mode is specified
            for prefix in ['min', 'max']:
                for attrib in ['sc', 'novelty', 'alpha', 'beta', 'coil']:
                    if np.isscalar(self.sample_config['%s_%s' % (prefix, attrib)]):
                        self.sample_config['%s_%s' % (prefix, attrib)] = [self.sample_config['%s_%s' % (prefix, attrib)]] * len(self.sample_config['sample_goal_list'])

        else:
            # when preset mode is specified, other cutoff settings can be overriden
            cur_mode_setting = self.mode_preset_settings[self.sample_config['mode_preset']]
            # print(cur_mode_setting)

            for attrib in ['sc', 'novelty', 'alpha', 'beta', 'coil']:
                self.sample_config['pred_%s' % attrib] = False
                for prefix in ['min', 'max']:
                    if '%s_%s' % (prefix, attrib) in cur_mode_setting:
                        length_cutoff, value_list = cur_mode_setting['%s_%s' % (prefix, attrib)]
                        length_cutoff = np.array(length_cutoff)
                        self.sample_config['pred_%s' % attrib] = True
                        self.sample_config['%s_%s' % (prefix, attrib)] = [value_list[np.sum(goal[0] > length_cutoff)] for goal in self.sample_config['sample_goal_list']]
                    else:
                        self.sample_config['%s_%s' % (prefix, attrib)] = [default_sample_config['%s_%s' % (prefix, attrib)]] * len(self.sample_config['sample_goal_list'])

        if 'soft_prob' in self.sample_config:
            self.soft_prob = self.sample_config['soft_prob']
        else:
            logger.warning('soft_prob is not provided, using default value 0.1')
            self.soft_prob = 0.1

        logger.info('final sampling configuration: %s' % self.sample_config)
    
    def run(self, sample_config_override = None):
        if sample_config_override is not None:
            self.sample_config = sample_config_override
        self._process_sample_config()

        with torch.no_grad():

            if self.rank == 0:
                # latent sampling
                logger.info('Start sampling latent...')
                self.rejection_sampling()
                logger.info('Finish sampling latent...')
                self.save_latent_result()
            
            if self.use_multi_gpu:
                # max 120 min
                dist.barrier()

            # structure sampling
            if self.output_dir is not None:
                self.load_latent_result()
            else:
                self.structure_cache = []
            logger.debug('Setting up sampling dataset and dataloader...')
            self.setup_sampling_dataset_and_dataloader()
            logger.info('Start sampling structure...')
            self.strcuture_sampling()
            logger.info('Finish sampling structure...')

            if self.use_multi_gpu:
                dist.barrier()

        if self.output_dir is None:
            return self.latent_cache, self.structure_cache

    def strcuture_sampling(self):
        pbar = tqdm(self.sampling_loader, disable = self.rank != 0)
        for batch_idx, batch in enumerate(pbar):
            
            idx = int(batch['idx'].item())
            length = int(batch['length'].item())
            latent = batch['latent'].to(self.rank)
            rep = int(batch['idx_in_length'].item())

            if self.sample_config['structure_seed'] is not None:
                sd = self.sample_config['structure_seed'] + idx
                self._set_seed(sd)
            else:
                sd = None

            pbar.set_description('Sampling structure for label %s, length %d, repeat %d' % (self.sample_config['label'], length, rep))

            prediction = self.model_diffusion.sample_latent_conditional(
                    latent = latent,
                    return_traj = True,
                    return_frame = False,
                    return_position = True,
                    reconstruct_position = True,
                    num_res = length,
                    timestep = 200,
                    rotation_reverse_score_scale_override = self.diffusion_sampling_parm['rotation_reverse_score_scale_override'] if self.diffusion_sampling_parm is not None and 'rotation_reverse_score_scale_override' in self.diffusion_sampling_parm else None,
                    rotation_reverse_noise_scale_override = self.diffusion_sampling_parm['rotation_reverse_noise_scale_override'] if self.diffusion_sampling_parm is not None and 'rotation_reverse_noise_scale_override' in self.diffusion_sampling_parm else None,
                    )
            coord37_record, coord37_mask = self.structure_builder.coord14_to_coord37(prediction['coord_hat'], trunc=True)
            prot_traj = self.structure_builder.get_coord_traj(coord37_record[None],
                                aa_mask=coord37_mask,
                                label_override='length %d, repeat %d' % (length, rep) if sd is None else 'length %d, repeat %d, seed %d' % (length, rep, sd),
                                default_res='G'
                                )
            if self.output_dir is not None:
                save_path = os.path.join(self.output_dir, 'length_%d' % length, 'sample_%d.pdb' % rep)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    f.write(protein.to_pdb(prot_traj[0]))
            else:
                self.structure_cache.append(protein.to_pdb(prot_traj[0]))


    def _init_sample_goal_list(self):
        self.num_total_sample = sum([goal[1] for goal in self.sample_config['sample_goal_list']])
        self.sampled_latent = torch.zeros((self.num_total_sample, self.latent_dim))
        self.sampled_length = torch.zeros((self.num_total_sample))
        self.sampled_length_pred = torch.zeros((self.num_total_sample))
        self.sampled_idx_in_length = torch.zeros((self.num_total_sample))
        self.sampled_sc_pred = torch.zeros((self.num_total_sample))
        self.sampled_novelty_pred = torch.zeros((self.num_total_sample))
        self.sampled_alpha_pred = torch.zeros((self.num_total_sample))
        self.sampled_beta_pred = torch.zeros((self.num_total_sample))
        self.sampled_coil_pred = torch.zeros((self.num_total_sample))
        self.num_total_sampled = 0

    def _sample_new_latent(self):
        latent_cache = self.model_latent.sample(n_sample = self.sample_config['latent_batch_size'])
        latent_cache = {k: v.to('cpu') for k, v in latent_cache.items()}

        # pred sc
        if self.sample_config['pred_sc']:
            sc_pred = self.model_sc(latent_cache['latent_sample'])
        else:
            sc_pred = torch.ones((self.sample_config['latent_batch_size'])) * -1

        if self.sample_config['pred_novelty']:
            novelty_pred = self.model_novelty(latent_cache['latent_sample'])
        else:
            novelty_pred = torch.ones((self.sample_config['latent_batch_size'])) * -1

        if self.sample_config['pred_alpha']:
            alpha_pred = self.model_alpha(latent_cache['latent_sample'])
        else:
            alpha_pred = torch.ones((self.sample_config['latent_batch_size'])) * -1

        if self.sample_config['pred_beta']:
            beta_pred = self.model_beta(latent_cache['latent_sample'])
        else:
            beta_pred = torch.ones((self.sample_config['latent_batch_size'])) * -1

        if self.sample_config['pred_coil']:
            coil_pred = self.model_coil(latent_cache['latent_sample'])
        else:
            coil_pred = torch.ones((self.sample_config['latent_batch_size'])) * -1

        # pred length
        dis_mat = torch.sum((latent_cache['latent_sample'][:, None, :] - self.label_dict['label_latent'][None, :, :])**2, dim=-1)
        min_val, min_idx = torch.topk(dis_mat, k=self.sample_config['length_k'], dim=-1, largest=False, sorted=True)
        length_pred = self.label_dict['label_length'][min_idx].float().mean(dim=-1).long()

        self.latent_cache_dict = {
            'latent': latent_cache['latent_sample'],
            'length_pred': length_pred,
            'sc_pred': sc_pred,
            'novelty_pred': novelty_pred,
            'alpha_pred': alpha_pred,
            'beta_pred': beta_pred,
            'coil_pred': coil_pred,
        }

        self.sample_idx = 0

    def _get_next_sample(self):
        if self.sample_idx >= self.sample_config['latent_batch_size']:
            logger.info('sample new latent')
            self._sample_new_latent()

        sample = {key: val[self.sample_idx] for key, val in self.latent_cache_dict.items()}
        self.sample_idx += 1
        self.num_total_sampled += 1

        return sample
    
    def rejection_sampling(self):
        if self.sample_config['latent_seed'] is not None:
            self._set_seed(self.sample_config['latent_seed'])

        self._init_sample_goal_list()
        self._sample_new_latent()

        with tqdm(total=self.num_total_sample) as pbar:
            num_toal_accepted = 0
            for i, goal_tup in enumerate(self.sample_config['sample_goal_list']):

                cur_min_sc = self.sample_config['min_sc'][i]
                cur_max_sc = self.sample_config['max_sc'][i]
                cur_min_novelty = self.sample_config['min_novelty'][i]
                cur_max_novelty = self.sample_config['max_novelty'][i]

                cur_min_alpha = self.sample_config['min_alpha'][i]
                cur_max_alpha = self.sample_config['max_alpha'][i]
                cur_min_beta = self.sample_config['min_beta'][i]
                cur_max_beta = self.sample_config['max_beta'][i]
                cur_min_coil = self.sample_config['min_coil'][i]
                cur_max_coil = self.sample_config['max_coil'][i]

                num_accepted = 0
                while num_accepted < goal_tup[1]:
                    pbar.set_description('Sampling latent for length %d, repeat %d' % (goal_tup[0], num_accepted))

                    sample = self._get_next_sample()
                    # print(sample)
                    if (sample['length_pred'] * (1 - self.sample_config['length_epsilon']) <= goal_tup[0] and
                    sample['length_pred'] * (1 + self.sample_config['length_epsilon']) >= goal_tup[0] ):
                        if (
                        (
                            self.sample_config['pred_sc'] is False or
                            (
                                sample['sc_pred'] >= cur_min_sc and
                                sample['sc_pred'] <= cur_max_sc
                            )
                        )
                        and
                        (
                            self.sample_config['pred_novelty'] is False or
                            (
                                sample['novelty_pred'] >= cur_min_novelty and
                                sample['novelty_pred'] <= cur_max_novelty
                            )
                        )
                        and
                        (
                            self.sample_config['pred_alpha'] is False or
                            (
                                sample['alpha_pred'] >= cur_min_alpha and
                                sample['alpha_pred'] <= cur_max_alpha
                            )
                        )
                        and
                        (
                            self.sample_config['pred_beta'] is False or
                            (
                                sample['beta_pred'] >= cur_min_beta and
                                sample['beta_pred'] <= cur_max_beta
                            )
                        )
                        and
                        (
                            self.sample_config['pred_coil'] is False or
                            (
                                sample['coil_pred'] >= cur_min_coil and
                                sample['coil_pred'] <= cur_max_coil
                            )
                        )) or torch.rand(()) <= self.soft_prob:
                            self.sampled_latent[num_toal_accepted] = sample['latent']
                            self.sampled_length[num_toal_accepted] = goal_tup[0]
                            self.sampled_length_pred[num_toal_accepted] = sample['length_pred']
                            self.sampled_sc_pred[num_toal_accepted] = sample['sc_pred']
                            self.sampled_novelty_pred[num_toal_accepted] = sample['novelty_pred']

                            self.sampled_alpha_pred[num_toal_accepted] = sample['alpha_pred']
                            self.sampled_beta_pred[num_toal_accepted] = sample['beta_pred']
                            self.sampled_coil_pred[num_toal_accepted] = sample['coil_pred']

                            self.sampled_idx_in_length[num_toal_accepted] = num_accepted
                            num_toal_accepted += 1
                            num_accepted += 1
                            pbar.update(1)
        
        logger.info('Successfully accepted %d latents' % num_toal_accepted)
        logger.info('Total sampled %d latents' % self.num_total_sampled)

    def save_latent_result(self):
        self.sampled_idx = torch.arange(self.num_total_sample)
        latent_dict = {
                'latent': self.sampled_latent,
                'length': self.sampled_length,
                'length_pred': self.sampled_length_pred,
                'sc_pred': self.sampled_sc_pred,
                'novelty_pred': self.sampled_novelty_pred,
                'alpha_pred': self.sampled_alpha_pred,
                'beta_pred': self.sampled_beta_pred,
                'coil_pred': self.sampled_coil_pred,
                'idx_in_length': self.sampled_idx_in_length,
                'idx': self.sampled_idx,
            }

        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
            save_path = os.path.join(self.output_dir, 'latent_dict.pt')
            torch.save(latent_dict, save_path)
            logger.info('Saving sample result to %s' % save_path)
        else:
            logger.warning('output_dir is not provided, will not save latent result')
            self.latent_cache = latent_dict    
            


    def load_latent_result(self):
        load_path = os.path.join(self.output_dir, 'latent_dict.pt')
        logger.debug('Loading sample result from %s' % load_path)
        latent_dict = torch.load(load_path, map_location='cpu', weights_only=False)
        self.sampled_latent = latent_dict['latent']
        self.sampled_length = latent_dict['length']
        self.sampled_length_pred = latent_dict['length_pred']
        self.sampled_sc_pred = latent_dict['sc_pred']
        self.sampled_novelty_pred = latent_dict['novelty_pred']
        self.sampled_alpha_pred = latent_dict['alpha_pred']
        self.sampled_beta_pred = latent_dict['beta_pred']
        self.sampled_coil_pred = latent_dict['coil_pred']
        self.sampled_idx_in_length = latent_dict['idx_in_length']
        self.sampled_idx = latent_dict['idx']
        self.num_total_sampled = self.sampled_latent.shape[0]
        
    def setup_sampling_dataset_and_dataloader(self):
        self.sampling_dataset = LatentContionalSamplingDataset2(
            condition_dict={
                'latent': self.sampled_latent,
                'length': self.sampled_length,
                'idx_in_length': self.sampled_idx_in_length,
                'idx': self.sampled_idx,
                'length_pred': self.sampled_length_pred,
                'sc_pred': self.sampled_sc_pred,
                'novelty_pred': self.sampled_novelty_pred,
                'alpha_pred': self.sampled_alpha_pred,
                'beta_pred': self.sampled_beta_pred,
                'coil_pred': self.sampled_coil_pred,
            }
        )

        sampler = torch.utils.data.distributed.DistributedSampler(self.sampling_dataset, 
                                                                  shuffle=False,
                                                                  num_replicas=self.world_size,
                                                                  rank=self.rank)
    
        self.sampling_loader = torch.utils.data.DataLoader(
            self.sampling_dataset,
            batch_size=1,
            collate_fn=UnconditionalSamplingDummyCollator(),
            num_workers=0,
            sampler=sampler,
        )


    def _set_seed(self, sd):
        torch.manual_seed(sd)
        np.random.seed(sd % (2**32))