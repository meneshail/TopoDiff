import os, sys
import time
from functools import partial
import logging
import pickle
from contextlib import nullcontext
from tqdm import tqdm
from collections import deque
import ml_collections as mlc
import copy

import argparse

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

#. model
from TopoDiff.model.latent_diffusion import LatentDiffusion
from TopoDiff.model.diffuser.latent_diffuser import LatentDiffuser

#. config
from TopoDiff.config.latent_config import latent_model_config

#. data
from TopoDiff.data.latent_data_modules import LatentDataset, LatentCollator

#. loss
from TopoDiff.utils.latent_loss import LatentLoss

#. utils
from myopenfold.utils.tensor_utils import tensor_tree_map, dict_multimap

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

project_dir = os.path.dirname(os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODEL_NAME = 'model_1'
MODEL_PRESENT = 'model_1'

DISABLE_EPOCH_LOG = True
SAVE_CUSTOM_LOG = False

default_latent_training_config = {
    "Base" : {
        "mode" : "train",
        "start_epoch" : 0,
        "max_epochs" : 1000,
        "ckpt_save_interval" : 1000,
        "log_to_main_interval" : 100,
        'log_to_main_metric_list': ['total_loss', 'recon'],
        "base_seed" : None,
        "root_dir" : None,
        'sampling_save_dir': None,
        "custom_log_save_dir" : None,
        "ckpt_save_dir" : None,
        "model_load_path": None,
        'state_load_path': None,
    },
    'Sampling':{
        'sampling_per_epoch': None,
        'timesteps': 200,
        "N_sample": 10000,
        "batch_size" : 128,
        'Sampling_base_seed': 424242,
    },
    "Strategy" : {
        "accelerator" : "gpu",
        "gpus" : None,
        "devices" : 1,
        "accumulate_grad_batches" : 1,
        "precision" : 32,
        'lr' : 3e-5,
        'eps': 1e-7,
    }, 
    "Model" : {
        "model_preset" : MODEL_PRESENT,
        "extra": [],
    },
    "Data" : {
        "train_data_path" : None,
        "sort_data" : True,
        "batch_size" : 256,
        "num_workers" : 4,
    }
}

def get_latent_train_config(args):
    train_cfg = copy.deepcopy(default_latent_training_config)
    train_cfg = mlc.ConfigDict(train_cfg)

    par_dir = args.outdir
    epoch = args.latent_epoch
    train_cfg.Base.root_dir = os.path.join(par_dir, 'latent', 'save', MODEL_NAME, 'epoch_%d' % epoch)
    train_cfg.Data.train_data_path = os.path.join(par_dir, 'save', 'embedding', 'epoch_%d.pkl' % epoch)
    assert os.path.exists(train_cfg.Data.train_data_path), 'train data path not exists'
    train_cfg.Base.ckpt_save_dir = os.path.join(par_dir, 'latent', 'save', MODEL_NAME, 'epoch_%d' % epoch, 'ckpt')
    os.makedirs(train_cfg.Base.ckpt_save_dir, exist_ok = True)
    train_cfg.Base.custom_log_save_dir = os.path.join(par_dir, 'latent', 'save', MODEL_NAME, 'epoch_%d' % epoch, 'custom_log')
    os.makedirs(train_cfg.Base.custom_log_save_dir, exist_ok = True)
    # train_cfg.Base.sampling_save_dir = os.path.join(par_dir, 'latent', 'save', MODEL_NAME, 'epoch_%d' % epoch, 'sampling')

    train_cfg.Strategy.gpus = list(map(int, args.gpu.split(',')))

    train_cfg.Base.base_seed = args.seed

    return train_cfg


def merge(outputs):
    if dist.is_initialized():
        dist.barrier()
        all_rank_outputs = [None for _ in range(dist.get_world_size())]    
        dist.all_gather_object(all_rank_outputs, outputs)
        return all_rank_outputs
    else:
        return [outputs]

def worker_init_fn(worker_id, constant = 0, rank = 0):
    logger.debug("Init: rank %d, worker_id %d.." % (rank, worker_id))
    return 

class MyLatentTrainer():
    def __init__(self, args, rank, world_size, log = False):
        self.args = args
        self.train_config = get_latent_train_config(args)
        self.config = latent_model_config(name = self.train_config.Model.model_preset,)

        self.rank = rank
        self.world_size = world_size
        self.log = log

        self.current_epoch = 0
        self.global_step = 0
        self.step_in_epoch = 0

        self.grad_accumulation_step = self.train_config.Strategy.accumulate_grad_batches

        self.loss = LatentLoss(config=self.config.Loss, log_parm=self.rank == 0)
        self.collator = LatentCollator()

        self._init()

        if 'gradient_clip_val' not in self.train_config.Strategy or 'gradient_clip_algorithm' not in self.train_config.Strategy:
            self.train_config.Strategy.gradient_clip_val = 1.0
            self.train_config.Strategy.gradient_clip_algorithm = 'norm'

        if dist.is_initialized():
            assert self.world_size == dist.get_world_size() 
            assert self.rank == dist.get_rank()
        else:
            assert self.world_size == 1
            assert self.rank == 0


    def _init(self):
        self._init_data()
        self._init_model()
        self._init_ddp()
        self._init_optimizer_and_scheduler()
        self._init_scaler()
        self._load_state()

    def _init_data(self):
        latent_data_dict = pickle.load(open(self.train_config.Data.train_data_path, 'rb'))

        if self.train_config.Data.sort_data:
            pos_dict = {}
            for pos, sample_idx in enumerate(latent_data_dict['sample_idx']):
                if sample_idx not in pos_dict:
                    pos_dict[sample_idx] = pos
            new_order = np.array([pos_dict[k] for k in sorted(pos_dict.keys())])
            latent_data = torch.from_numpy(latent_data_dict['latent_mu'][new_order]).float()
        else:
            latent_data = torch.from_numpy(latent_data_dict['latent_mu']).float()
        
        latent_bg_mu = latent_data.mean(dim=0)
        latent_bg_std = latent_data.std(dim=0)

        self.config.Data.common.normalize.mu = latent_bg_mu
        self.config.Data.common.normalize.sigma = latent_bg_std
        self.train_data = latent_data

    def _init_model(self):
        model = LatentDiffusion(config_latent_diffusion=self.config, log = self.log)
        if self.train_config.Base.model_load_path is not None:
            logger.info('Loading latent diffusion model from checkpoint {}'.format(self.train_config.Base.model_load_path))
            missing_keys, unexpected_keys = model.load_state_dict(torch.load(self.train_config.Base.model_load_path, map_location = 'cpu'), strict=False)

            if self.rank == 0:
                logger.info('Missing keys: {}'.format(missing_keys))
                logger.info('Unexpected keys: {}'.format(unexpected_keys))
                        
        model = model.to(self.rank)
        self.model = model

    
    def _init_ddp(self):
        if self.world_size > 1:
            logger.info("Rank %d: Using DDP, total gpus: %d" % (self.rank, self.world_size))
            self.use_ddp = True
            self.model = DDP(self.model, device_ids = [self.rank], find_unused_parameters = False)
        else:
            logger.info("Rank %d: Not using DDP" % self.rank)
            self.use_ddp = False

    def _init_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr = self.train_config.Strategy.lr, 
                                          eps = self.train_config.Strategy.eps,
                                          weight_decay=1e-8)    
        self.scheduler = None
                                
    def _init_scaler(self):
        if str(self.train_config.Strategy.precision) == '16':
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.use_amp = False
            self.scaler = None

    def _load_state(self):
        logger.info("In _load_state, rank %d" % (self.rank))
        if self.train_config.Base.state_load_path is not None:
            logger.info('Loading training state from checkpoint {}'.format(self.train_config.Base.state_load_path))
            state_dct = torch.load(self.train_config.Base.state_load_path, map_location = 'cpu')
            if 'optimizer' in state_dct:
                logger.info('Loading optimizer state...')
                self.optimizer.load_state_dict(state_dct['optimizer'])
                for group in self.optimizer.param_groups:
                    group['lr'] = self.train_config.Strategy.lr
            else:
                logger.info('No optimizer state found in checkpoint, using default optimizer state')
            if 'scaler' in state_dct and state_dct['scaler'] is not None and self.scaler is not None:
                logger.info('Loading scaler state...')
                self.scaler.load_state_dict(state_dct['scaler'])
            else:
                logger.info('No scaler state found in checkpoint, using default scaler state')
            if 'current_epoch' in state_dct:
                logger.info('Loading epoch...')
                self.current_epoch = state_dct['current_epoch']
                logger.info('Continue training from epoch %d' % self.current_epoch)
            else:
                logger.info('No epoch found in checkpoint, starting from epoch 1')
            
            if 'global_step' in state_dct:
                self.global_step = state_dct['global_step']
                logger.info('Loading global_step, starting from global_step {}'.format(self.global_step + 1))
            else:
                logger.info('No global_step found in checkpoint, starting from global_step 1')
            
        else:
            logger.info('No training state found, using default training state')

    def run(self):
        progress_bar = tqdm(total = self.train_config.Base.max_epochs, desc = "Training progress", disable = self.rank != 0)
        while self.current_epoch < self.train_config.Base.max_epochs:
            self.train_epoch()
            progress_bar.update(1)

        self._pack_ckpt()

    def _start_new_epoch(self):
        self.current_epoch += 1
        self.step_in_epoch = 0
        torch.manual_seed(self.train_config.Base.base_seed + self.current_epoch)
        np.random.seed(self.train_config.Base.base_seed + self.current_epoch)

        self.train_track_list = []
        self.bad_samples = []
        self.val_track_dict = {}

    def _start_new_step(self):
        self.step_in_epoch += 1
        self.global_step += 1

    def _set_seed(self, sd):
        torch.manual_seed(sd)
        np.random.seed(sd)

    def setup_dataset_and_dataloader(self):

        self.train_set = LatentDataset(
                latent_data = self.train_data,
                config = self.config.Data,
            )
        self.train_set.set_epoch(self.current_epoch)

        sampler = torch.utils.data.distributed.DistributedSampler(self.train_set,
                                                                  shuffle=True,
                                                                  seed = self.train_config.Base.base_seed,
                                                                  num_replicas = self.world_size,
                                                                  rank = self.rank,)
        sampler.set_epoch(self.current_epoch)

        self.train_loader =  torch.utils.data.DataLoader(
            self.train_set,
            batch_size = self.train_config.Data.batch_size,
            num_workers = self.train_config.Data.num_workers,
            pin_memory = True,
            sampler = sampler,
            collate_fn = self.collator,
        )

        self.total_step_in_epoch = len(self.train_loader)

    def train_epoch(self):
        self._start_new_epoch()

        self.setup_dataset_and_dataloader()

        self.model.train()

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc = "Training epoch %d" % (self.current_epoch), disable = DISABLE_EPOCH_LOG or self.rank != 0)):

            self._start_new_step()
            start = time.time()

            batch = tensor_tree_map(lambda x: x.to(self.rank), batch)
            is_grad_accumulation_step = self.step_in_epoch % self.grad_accumulation_step == 0
            batch_part = self.train_step(batch, batch_idx, is_grad_accumulation_step)
            end = time.time()

            self.train_step_end(batch_part)

        self.optimizer.zero_grad(set_to_none=True)

        if self.train_config.Sampling.sampling_per_epoch is not None and self.current_epoch % self.train_config.Sampling.sampling_per_epoch == 0:
            self.sampling_iter()

        self.train_epoch_end()

    def train_step(self, batch, batch_idx, is_grad_accumulation_step):
        my_context = self.model.no_sync if self.use_ddp and (not is_grad_accumulation_step) else nullcontext
        with my_context():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):

                prediction = self.model(batch)
                loss_cum, loss_dict = self.loss(batch, prediction, _return_breakdown=True)
                loss_cum = loss_cum / self.grad_accumulation_step 

                loss_cum = torch.mean(loss_cum)

                n_sample = batch['sample_idx'].shape[0]
                for key in loss_dict:
                    loss_dict[key] = np.mean(loss_dict[key])[None]
                loss_dict['step_in_epoch'] = np.ones(1, dtype = np.int32) * self.step_in_epoch
                loss_dict['gobal_step'] = np.ones(1, dtype = np.int32) * self.global_step

            if self.use_amp:
                self.scaler.scale(loss_cum).backward()
            else:
                loss_cum.backward()

            if is_grad_accumulation_step:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_config.Strategy.gradient_clip_val)

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
        
        return {'loss': loss_cum, 'loss_dict': loss_dict, 'batch_idx' : batch_idx}

    def train_epoch_end(self):
        if self.current_epoch % self.train_config.Base.ckpt_save_interval == 0:
            self._save_checkpoint(save_state=False)

        output_dicts_list = merge(self.train_track_list)

        if self.rank == 0 and SAVE_CUSTOM_LOG:
            self._save_record(output_dicts_list, epoch_end = True)

        if self.use_ddp:
            dist.barrier()

        return

    def train_step_end(self, batch_part):
        loss_dict = batch_part['loss_dict']
        self.train_track_list.append(loss_dict)

        if self.step_in_epoch % self.train_config.Base.log_to_main_interval == 0:
            loss_dict_list = merge(loss_dict)
            if self.rank == 0:
                self._log_to_main(loss_dict_list)

    def sampling_iter(self, label = None):

        self.model.eval()
        self._set_seed(self.current_epoch + self.train_config.Sampling.Sampling_base_seed)

        if self.rank == 0:
            with torch.no_grad():
                latent_sample_list = []
                for i in tqdm(range(0, self.train_config.Sampling.N_sample, self.train_config.Sampling.batch_size)):
                    n_sample = int(min(self.train_config.Sampling.batch_size, self.train_config.Sampling.N_sample - i))
                    latent_sample_dict = self.model.sample(n_sample = n_sample)
                    latent_sample_dict = tensor_tree_map(lambda x: x.cpu(), latent_sample_dict)
                    latent_sample_list.append(latent_sample_dict)

                latent_sample_dict = dict_multimap(lambda x: torch.cat(x, dim = 0), latent_sample_list)
        else:
            latent_sample_dict = None

        if self.use_ddp:
            dist.barrier()

        self.sampling_epoch_end(latent_sample_dict, label = label)
        return
    
    def sampling_epoch_end(self, latent_sample_dict, label = None):
        save_dir = os.path.join(self.train_config.Base.sampling_save_dir, 'epoch_%d' % self.current_epoch if label is None else 'epoch_%d_%s' % (self.current_epoch, label))
        
        if self.rank == 0:
            save_path = os.path.join(save_dir, 'sample.pkl')
            os.makedirs(save_dir, exist_ok = True)
            logger.info("Saving sample to %s" % save_path)
            torch.save(latent_sample_dict, save_path)

        if self.use_ddp:
            dist.barrier()

        return
    
    def _log_to_main(self, loss_dict_list):
        log_list = []
        for key in self.train_config.Base.log_to_main_metric_list:
            if key in loss_dict_list[0]:
                cur_loss = np.concatenate([loss_dict[key] for loss_dict in loss_dict_list], axis = 0)
                log_list.append('%s: %.4f' % (key, np.mean(cur_loss)))
        log_str = ', '.join(log_list)
        logger.info("Epoch %d, step %d: %s" % (self.current_epoch, self.step_in_epoch, log_str))

    def _save_record(self, output_dicts_list, epoch_end = False):
        n_rank = len(output_dicts_list)
        n_step = len(output_dicts_list[0])

        train_loss_dict_flatten = [dcts[i] for i in range(n_step) for dcts in output_dicts_list]
        train_record_dict = {}

        for key in train_loss_dict_flatten[0].keys():
            train_record_dict['train_' + key] = np.concatenate([d[key] for d in train_loss_dict_flatten])

        if epoch_end:
            train_record_dict.update(self.val_track_dict)

        log_path = os.path.join(self.train_config.Base.custom_log_save_dir, "epoch_%d.pkl" % self.current_epoch)
        with open(log_path, 'wb') as f:
            pickle.dump(train_record_dict, f)

    def _save_checkpoint(self, save_state = True, label = None):
        if self.rank == 0:
            label = "epoch_%d_%s" % (self.current_epoch, label) if label is not None else "epoch_%d" % self.current_epoch
            checkpoint_path = os.path.join(self.train_config.Base.ckpt_save_dir, '%s.pkl' % label)
            logger.info("Saving checkpoint to %s" % checkpoint_path)
            if self.use_ddp:
                torch.save(self.model.module.state_dict(), checkpoint_path)
            else:
                torch.save(self.model.state_dict(), checkpoint_path)

            if save_state:
                state_dict = {}
                state_dict['optimizer'] = self.optimizer.state_dict()
                state_dict['scheduler'] = self.scheduler.state_dict() if self.scheduler is not None else None
                state_dict['scaler'] = self.scaler.state_dict() if self.scaler is not None else None
                state_dict['current_epoch'] = self.current_epoch
                state_dict['global_step'] = self.global_step
                state_dict_path = os.path.join(self.train_config.Base.ckpt_save_dir, "%s_state.pkl" % label)
                torch.save(state_dict, state_dict_path)

    def _pack_ckpt(self):
        if self.rank == 0:
            logger.info('Packing checkpoint...')
            
            save_path = os.path.join(self.args.outdir, 'ckpt', 'epoch_%d.ckpt' % self.args.latent_epoch)
            structure_ckpt_path = os.path.join(self.args.outdir, 'save', 'ckpt', 'epoch_%d.pkl' % self.args.latent_epoch)
            embedding_path = os.path.join(self.args.outdir, 'save', 'embedding', 'epoch_%d.pkl' % self.args.latent_epoch)
            latent_ckpt_path = os.path.join(self.train_config.Base.ckpt_save_dir, 'epoch_%d.pkl' % self.current_epoch)
            latent_ref_path = os.path.join(project_dir, 'data', 'weights', 'latent_ref', 'ref.pt')

            ckpt_dict = {}
            # latent label
            latent_ref_dict = torch.load(latent_ref_path)
            with open(embedding_path, 'rb') as f:
                embedding_dict = pickle.load(f)
            latent_ref_dict['label_latent'] = torch.from_numpy(embedding_dict['latent_mu']).float()
            ckpt_dict['embedding_dict'] = latent_ref_dict

            # latent model
            latent_model_dict = torch.load(latent_ckpt_path, map_location = 'cpu')
            ckpt_dict['latent_ckpt'] = latent_model_dict

            # structure model
            structure_model_dict = torch.load(structure_ckpt_path, map_location = 'cpu')
            ckpt_dict['main_ckpt'] = structure_model_dict

            os.makedirs(os.path.dirname(save_path), exist_ok = True)
            torch.save(ckpt_dict, save_path)

            logger.info('Packing checkpoint for structure diffusion epoch %d, latent diffusion epoch %d, saved to %s' % (self.args.latent_epoch, self.current_epoch, save_path))