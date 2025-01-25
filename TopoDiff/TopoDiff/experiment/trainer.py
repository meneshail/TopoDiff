import os, sys
import time
from functools import partial
import logging
import pickle
from contextlib import nullcontext
from tqdm import tqdm
import ml_collections as mlc
import numpy as np
import copy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TopoDiff.experiment.trainer")

#. model
from TopoDiff.model import diffusion

#. loss
from TopoDiff.utils.loss import TopoDiffLoss

#. lr_schedulers
from TopoDiff.utils.lr_schedulers import TopoDiffLRScheduler

#. utils
from myopenfold.utils.tensor_utils import tensor_tree_map
from TopoDiff.utils.debug import print_shape

#. config
from TopoDiff.config.config import model_config

#. data
from TopoDiff.data.data_modules import TopoDiffDataLoader, TopoDiffBatchCollator, TopoDiffTopologyDataset
from TopoDiff.data.data_modules import UnconditionalSamplingDummyDataset, UnconditionalSamplingDummyCollator
from TopoDiff.data.structure import StructureBuilder

#. pdb
from myopenfold.np import protein

project_dir = os.path.dirname(os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

default_training_config = {
    "Base" : {
        "mode" : "train",
        "start_epoch" : 0,
        "max_epochs" : None,   # NOTE: need to set
        "ckpt_save_interval" : 25,
        'ckpt_state_save_interval': 25,
        "log_to_main_interval" : 25,
        'log_to_main_metric_list': ['translation','rotation', 'distogram', 'fape_backbone',  'kl_regularization', 'unscaled_loss', 'scaled_loss'],
        "base_seed" : None,  # NOTE: need to set
        "root_dir" : None,  # NOTE: need to set
        'sampling_save_dir': None,  # NOTE: need to set
        "custom_log_save_dir" : None,  # NOTE: need to set
        "embedding_save_per_epoch": None,  # NOTE: need to set
        'embedding_save_dir': None,  # NOTE: need to set
        "ckpt_save_dir" : None,  # NOTE: need to set
        "model_load_path": None,  # NOTE: need to set
        'state_load_path': None,
    },
    'Sampling':{
        'sampling_per_epoch': 25,
        'timesteps': 200,
        "N_sample_list": [4, 4, 4, 4, 4, 4],
        "N_length_list": [75, 100, 125, 150, 200, 250],
        "batch_size" : 1,
        "num_workers" : 3,
        'Sampling_base_seed': 424242,
    },
    "Strategy" : {
        "accelerator" : "gpu",
        "gpus" : None,  # NOTE: need to set
        "devices" : None, # NOTE: need to set
        "accumulate_grad_batches" : 1,
        "precision" : 32,
        'lr' : 3e-5,
        'start_decay_after_n_steps': 200000000,  # no decay
        'decay_every_n_steps': 100000,
        'warmup_no_steps': 2000,
        're_warmup': False,
        'eps': 1e-7,
    }, 
    "Model" : {
        "model_preset" : None,  # NOTE: need to set
        "extra": [],
    },
    "Data" : {
        "train_data_dir" : None,  # NOTE: need to set
        "train_data_cache_path" : None, # NOTE: need to set
        "batch_size" : None, # NOTE: need to set
        "num_workers" : 8,
    }
}

def get_train_config(args):
    train_cfg = copy.deepcopy(default_training_config)
    train_cfg = mlc.ConfigDict(train_cfg)

    train_cfg.Base.max_epochs = args.n_epoch
    train_cfg.Base.base_seed = args.seed
    train_cfg.Base.root_dir = args.outdir
    train_cfg.Base.sampling_save_dir = os.path.join(args.outdir, 'save', 'sampling')
    os.makedirs(train_cfg.Base.sampling_save_dir, exist_ok = True)
    train_cfg.Base.custom_log_save_dir = os.path.join(args.outdir, 'save', 'log')
    os.makedirs(train_cfg.Base.custom_log_save_dir, exist_ok = True)
    train_cfg.Base.embedding_save_dir = os.path.join(args.outdir, 'save', 'embedding')
    os.makedirs(train_cfg.Base.embedding_save_dir, exist_ok = True)
    train_cfg.Base.ckpt_save_dir = os.path.join(args.outdir, 'save', 'ckpt')
    os.makedirs(train_cfg.Base.ckpt_save_dir, exist_ok = True)
    train_cfg.Base.model_load_path = args.init_ckpt

    train_cfg.Strategy.gpus = args.gpu.split(',')
    train_cfg.Strategy.devices = len(args.gpu.split(','))
    
    train_cfg.Data.batch_size = args.batch_size

    if args.stage == 1:
        train_cfg.Model.model_preset = 'train_stage_1'
        train_cfg.Data.train_data_dir = os.path.join(project_dir, 'data', 'train_data', 'monomer_train')
        train_cfg.Data.train_data_cache_path = os.path.join(project_dir, 'data', 'train_data', 'info', 'monomer_train.json')
    elif args.stage == 2:
        train_cfg.Model.model_preset = 'train_stage_2'
        train_cfg.Data.train_data_dir = os.path.join(project_dir, 'data', 'train_data', 'monomer_train')
        train_cfg.Data.train_data_cache_path = os.path.join(project_dir, 'data', 'train_data', 'info', 'monomer_train.json') 
    elif args.stage == 3:
        train_cfg.Model.model_preset = 'train_stage_3'
        train_cfg.Data.train_data_dir = os.path.join(project_dir, 'data', 'train_data', 'cath_train')
        train_cfg.Data.train_data_cache_path = os.path.join(project_dir, 'data', 'train_data', 'info', 'cath_train.json')
        train_cfg.Base.embedding_save_per_epoch = train_cfg.Base.ckpt_save_interval
    else:
        raise ValueError('Invalid stage, should be 1, 2, or 3')

    return train_cfg

def merge(outputs):
    if dist.is_initialized():
        dist.barrier()
        dist.barrier()
        all_rank_outputs = [None for _ in range(dist.get_world_size())]    
        dist.all_gather_object(all_rank_outputs, outputs)
        return all_rank_outputs
    else:
        # logger.info("Not in distributed mode, returning..")
        return [outputs]

def worker_init_fn(worker_id, constant = 0, rank = 0):
    logger.debug("Init: rank %d, worker_id %d.." % (rank, worker_id))
    return 

class MyTrainer():
    def __init__(self, args, rank, world_size, log = False):
        self.train_config = get_train_config(args)
        self.config = model_config(
            name = self.train_config.Model.model_preset,
        )

        if args.stage == 3:
            self.use_encoder = True
        else:
            self.use_encoder = False

        self.config.Model.Diffuser.SO3.cache_dir = os.path.join(project_dir, 'TopoDiff', 'cache')
        self.rank = rank
        self.world_size = world_size
        self.log = log

        self.current_epoch = 0
        self.global_step = 0
        self.step_in_epoch = 0
        self.global_accum_step = 0

        self.grad_accumulation_step = self.train_config.Strategy.accumulate_grad_batches

        self.loss = TopoDiffLoss(self.config.Loss)
        self.collator = TopoDiffBatchCollator(self.config.Data)

        self.sampling_collator = UnconditionalSamplingDummyCollator()
        self.sb = StructureBuilder()

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
        self._init_model()
        self._init_ddp()
        self._init_optimizer_and_scheduler()
        self._init_scaler()
        self._load_state()

    def _init_model(self):
        model = diffusion.Diffusion(self.config.Model, log = self.log)
        if self.train_config.Base.model_load_path is not None:
            logger.info('Loading diffusion model from checkpoint {}'.format(self.train_config.Base.model_load_path))
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
        self.scheduler = TopoDiffLRScheduler(self.optimizer, verbose = False,
                                             warmup_no_steps=self.train_config.Strategy.warmup_no_steps,
                                             max_lr=self.train_config.Strategy.lr,
                                             start_decay_after_n_steps=self.train_config.Strategy.start_decay_after_n_steps,
                                             decay_every_n_steps=self.train_config.Strategy.decay_every_n_steps,
                                             )
        
    def _init_scaler(self):
        if str(self.train_config.Strategy.precision) == '16':
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.use_amp = False
            self.scaler = None

    def _load_state(self):
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
            if 'scheduler' in state_dct and state_dct['scheduler'] is not None:
                if self.train_config.Strategy.re_warmup:
                    logger.info('Re-warmup scheduler...')
                    # state_dct['scheduler']['last_epoch'] = -1
                    self.scheduler = TopoDiffLRScheduler(self.optimizer, verbose = False,
                            warmup_no_steps=self.train_config.Strategy.warmup_no_steps,
                            max_lr=self.train_config.Strategy.lr,
                            start_decay_after_n_steps=self.train_config.Strategy.start_decay_after_n_steps,
                            decay_every_n_steps=self.train_config.Strategy.decay_every_n_steps,
                            last_epoch=-1,
                            )
                else:
                    logger.info('Loading scheduler state...')
                    self.scheduler.load_state_dict(state_dct['scheduler'])
            else:
                logger.info('No scheduler state found in checkpoint, using default scheduler state')
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

            if 'global_accum_step' in state_dct:
                self.global_accum_step = state_dct['global_accum_step']
                logger.info('Loading global_accum_step, starting from global_accum_step {}'.format(self.global_accum_step + 1))
            
        else:
            logger.info('No training state found, using default training state')

    def run(self):
        if self.rank == 0:
            self._save_checkpoint(save_state = True)
        dist.barrier()

        self.sampling_iter()
        if self.train_config.Base.embedding_save_per_epoch is not None:
            self.validation_epoch()

        while self.current_epoch < self.train_config.Base.max_epochs:
            self.train_epoch()

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
        if self.global_step % self.grad_accumulation_step == 1 or self.grad_accumulation_step == 1:
            self.global_accum_step += 1

    def _set_seed(self, sd):
        torch.manual_seed(sd)
        np.random.seed(sd % 2**32)

    #, setup dataset and dataloader at the beginning of each epoch
    def setup_dataset_and_dataloader(self):
        self.train_set = TopoDiffTopologyDataset(
                data_dir = self.train_config.Data.train_data_dir,
                config = self.config.Data,
                topo_data_cache_path = self.train_config.Data.train_data_cache_path,
                mode = 'train',
                rank = self.rank,
                epoch = self.train_config.Base.base_seed + self.current_epoch,
                pin_memory = True,
                hierarchical_dir = True,
            )

        worker_init_fn_partial = partial(worker_init_fn, constant = self.current_epoch + self.train_config.Base.base_seed, rank = self.rank)

        sampler = torch.utils.data.distributed.DistributedSampler(self.train_set,
                                                                  shuffle=True,
                                                                  seed = self.train_config.Base.base_seed,
                                                                  num_replicas = self.world_size,
                                                                  rank = self.rank,)
        sampler.set_epoch(self.current_epoch)

        self.train_loader = TopoDiffDataLoader(
            self.train_set, config=self.config.Data, stage="train", 
            batch_size = self.train_config.Data.batch_size,
            num_workers = self.train_config.Data.num_workers,
            worker_init_fn = worker_init_fn_partial,
            collate_fn = self.collator,
            sampler = sampler,
            seed = self.train_config.Base.base_seed + self.current_epoch,
        )
        self.total_step_in_epoch = len(self.train_loader)

    def train_epoch(self):
        self._start_new_epoch()

        self.setup_dataset_and_dataloader()

        self.model.train()

        self._set_seed(self.train_config.Base.base_seed * 1234 + self.current_epoch)

        n_accum_step = len(self.train_loader) // self.grad_accumulation_step
        max_batch_idx = n_accum_step * self.grad_accumulation_step
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc = "Training epoch %d" % (self.current_epoch), disable = self.rank != 0)):
            if batch_idx >= max_batch_idx:
                # logger.warning("In training_epoch: batch index larger than max_batch_idx, skipping.. step %d, epoch %d, rank %d, batch index %d" % (self.global_step, self.current_epoch, self.rank, batch_idx))
                continue
            self._start_new_step()
            start = time.time()

            batch = tensor_tree_map(lambda x: x.to(self.rank), batch)
            is_grad_accumulation_step = self.step_in_epoch % self.grad_accumulation_step == 0
            batch_part = self.train_step(batch, batch_idx, is_grad_accumulation_step)
            end = time.time()
            batch_part['loss_dict']['consumed_time'] = np.ones_like(batch_part['sample_idx'], dtype = np.float32) * (end - start)

            self.train_step_end(batch_part)
        
        self.optimizer.zero_grad(set_to_none=True)
        self.train_epoch_end()
        
        if self.train_config.Sampling.sampling_per_epoch is not None and self.current_epoch % self.train_config.Sampling.sampling_per_epoch == 0:
            self.sampling_iter()

        if self.train_config.Base.embedding_save_per_epoch is not None and self.current_epoch % self.train_config.Base.embedding_save_per_epoch == 0:
            self.validation_epoch()

        
    def train_step(self, batch, batch_idx, is_grad_accumulation_step):
        sample_idx = batch['batch_idx'].cpu().numpy()
        init_timestep = batch['timestep'].cpu().numpy()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        # no_sync only when not grad_accumulation_step and ddp mode is on
        my_context = self.model.no_sync if self.use_ddp and (not is_grad_accumulation_step) else nullcontext

        #, only sync grad when grad_accumulation_step is reached
        with my_context():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):

                prediction = self.model(batch)
                loss_cum, loss_dict = self.loss(prediction, batch, _return_breakdown=True, epoch = self.current_epoch)  #, average over grad_accumulation_step
                if self.step_in_epoch == 1 and self.use_encoder and self.rank == 0:
                    logger.info('In training epoch %d, step %d, current KL loss weight %f' % (self.current_epoch, self.step_in_epoch, self.loss.kl_weight_scheduler(self.current_epoch)))
                
                loss_cum = loss_cum / self.grad_accumulation_step 
                loss_cum = torch.mean(loss_cum)

                torch.cuda.empty_cache()

                n_sample = len(sample_idx)
                loss_dict['step_in_epoch'] = np.ones(n_sample, dtype = np.int32) * self.step_in_epoch
                loss_dict['gobal_step'] = np.ones(n_sample, dtype = np.int32) * self.global_step
                loss_dict['sample_idx'] = sample_idx
                loss_dict['init_timestep'] = init_timestep
                loss_dict['self_condition_step'] = batch['self_condition_step'].cpu().numpy()
                loss_dict['global_accum_step'] = np.ones(n_sample, dtype = np.int32) * self.global_accum_step
                loss_dict['current_epoch'] = np.ones(n_sample, dtype = np.int32) * self.current_epoch

            if self.use_amp:
                self.scaler.scale(loss_cum).backward()
            else:
                loss_cum.backward()
            
            if is_grad_accumulation_step:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_config.Strategy.gradient_clip_val)

                if self.use_amp:
                    self.scaler.step(self.optimizer)  #, call optimizer.step() inside
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            
            return {'loss': loss_cum, 'loss_dict': loss_dict, 'batch_idx' : batch_idx, 'sample_idx' : sample_idx}


    def train_epoch_end(self):
        if self.current_epoch % self.train_config.Base.ckpt_save_interval == 0:
            save_state_tag = self.current_epoch % self.train_config.Base.ckpt_state_save_interval == 0
            self._save_checkpoint(save_state=save_state_tag)

        output_dicts_list = merge(self.train_track_list)

        if self.rank == 0:
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

    def setup_validation_dataset_and_dataloader(self):
        self.val_set = TopoDiffTopologyDataset(
                data_dir = self.train_config.Data.train_data_dir,
                config = self.config.Data,
                topo_data_cache_path = self.train_config.Data.train_data_cache_path,
                mode = 'train',
                rank = self.rank,
                epoch = self.train_config.Base.base_seed + self.current_epoch,
                pin_memory = True,
                hierarchical_dir = True,
                extra_config = {'encoder_no_noise': True},
            )

        worker_init_fn_partial = partial(worker_init_fn, constant = self.current_epoch + self.train_config.Base.base_seed, rank = self.rank)

        sampler = torch.utils.data.distributed.DistributedSampler(self.val_set,
                                                                  shuffle=False,
                                                                  seed = self.train_config.Base.base_seed,
                                                                  num_replicas = self.world_size,
                                                                  rank = self.rank,)
        sampler.set_epoch(self.current_epoch)

        self.val_loader = TopoDiffDataLoader(
            self.val_set, config=self.config.Data, stage="train", 
            batch_size = self.train_config.Data.batch_size,
            num_workers = self.train_config.Data.num_workers,
            worker_init_fn = worker_init_fn_partial,
            collate_fn = self.collator,
            sampler = sampler,
            seed = self.train_config.Base.base_seed + self.current_epoch,
        )

        self.val_track_list = []
        return

    def validation_epoch(self):
        if self.rank == 0:
            logger.info("In validation_epoch, epoch %d, rank %d" % (self.current_epoch, self.rank))

        self.setup_validation_dataset_and_dataloader()

        # if torch == 2.0 and not pad_in_collator, this will cause error
        self.model.eval()

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc = "Validation epoch %d" % (self.current_epoch), disable = self.rank != 0)):

            start = time.time()

            batch = tensor_tree_map(lambda x: x.to(self.rank), batch)

            batch_part = self.val_step(batch, batch_idx)
            end = time.time()
            batch_part['result_dict']['consumed_time'] = np.ones_like(batch_part['sample_idx'], dtype = np.float32) * (end - start)

            self.val_step_end(batch_part)

        self.val_epoch_end()

    def val_step(self, batch, batch_idx):
        sample_idx = batch['batch_idx'].cpu().numpy()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                
                if self.use_ddp:
                    prediction = self.model.module.encode_topology(batch)
                else:
                    prediction = self.model.encode_topology(batch)
                #. latent_mu [batch_size, latent_dim]
                #. latent_logvar [batch_size, latent_dim]
                #. latent_z [batch_size, latent_dim]
                for k, v in prediction.items():
                    prediction[k] = v.detach().cpu().numpy()

                prediction['sample_idx'] = sample_idx
        
        return {'result_dict': prediction, 'batch_idx' : batch_idx, 'sample_idx' : sample_idx}
    
    def val_step_end(self, batch_part):
        result_dict = batch_part['result_dict']
        self.val_track_list.append(result_dict)

    def val_epoch_end(self):
        output_dicts_list = merge(self.val_track_list)

        if self.rank == 0:
            self._save_embedding(output_dicts_list)

        del self.val_track_list

        if self.use_ddp:
            dist.barrier()

        return

    def setup_sampling_dataset_and_dataloader(self, idx):
        self.length_cur_epoch = self.train_config.Sampling.N_length_list[idx]
        self.n_sample_cur_epoch = self.train_config.Sampling.N_sample_list[idx]

        self.sampling_dataset = UnconditionalSamplingDummyDataset(
                n_samples=self.n_sample_cur_epoch,
                length_list=self.length_cur_epoch,
            )
        
        sampler = torch.utils.data.distributed.DistributedSampler(self.sampling_dataset, shuffle=False,
                                                                  num_replicas = self.world_size,
                                                                  rank = self.rank,)

        self.sampling_loader = torch.utils.data.DataLoader(
            self.sampling_dataset,
            batch_size=self.train_config.Sampling.batch_size,
            num_workers=self.train_config.Sampling.num_workers,
            collate_fn=self.sampling_collator,
            sampler=sampler,
        )

    def sampling_iter(self, label = None):

        for idx in range(len(self.train_config.Sampling.N_length_list)):
            self.setup_sampling_dataset_and_dataloader(idx)

            save_dir = os.path.join(self.train_config.Base.sampling_save_dir, 'epoch_%d' % self.current_epoch if label is None else 'epoch_%d_%s' % (self.current_epoch, label), 'length_%d' % self.length_cur_epoch)    
            if not os.path.exists(save_dir) and self.rank == 0:
                os.makedirs(save_dir, exist_ok = True)
            if self.use_ddp:
                dist.barrier()

            for batch_idx, batch in enumerate(tqdm(self.sampling_loader, desc = "Sampling epoch %d" % (self.current_epoch), disable = self.rank != 0)):
                start = time.time()
                
                sampling_result = self.sampling_step(batch, batch_idx)

                end = time.time()

                sampling_result['prediction']['consumed_time'] = end - start

                self.sampling_step_end(sampling_result['prediction'], label = label)

        self.sampling_epoch_end(label = label)
        return

    def sampling_step(self, batch, batch_idx):
        sample_idx, length = batch
        sd = sample_idx + self.train_config.Sampling.Sampling_base_seed + self.current_epoch * 10 + length * 5
        self._set_seed(sd)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            with torch.no_grad():
                if self.use_ddp:
                    prediction = self.model.module.sample_unconditional(return_traj = True,
                                                        return_frame = False,
                                                        return_position = True,
                                                        reconstruct_position = True,
                                                        num_res = length,
                                                        timestep = self.train_config.Sampling.timesteps,
                                                        init_latent=True,
                                                        rotation_reverse_strategy_override = self.config.Model.Diffuser.SO3.sampling_reverse_strategy if self.config.Model.Diffuser.SO3.reverse_strategy == 'score_and_noise' else None,
                                                        rotation_reverse_noise_scale_override = self.config.Model.Diffuser.SO3.reverse_parm.score_and_noise.sampling_noise_scale if self.config.Model.Diffuser.SO3.reverse_strategy == 'score_and_noise' else None,
                                                        rotation_reverse_score_scale_override = self.config.Model.Diffuser.SO3.reverse_parm.score_and_noise.sampling_score_scale if self.config.Model.Diffuser.SO3.reverse_strategy == 'score_and_noise' else None,
                                                        )
                else:
                    prediction = self.model.sample_unconditional(return_traj = True,
                                                        return_frame = False,
                                                        return_position = True,
                                                        reconstruct_position = True,
                                                        num_res = length,
                                                        timestep = self.train_config.Sampling.timesteps,
                                                        init_latent=True,
                                                        rotation_reverse_strategy_override = self.config.Model.Diffuser.SO3.sampling_reverse_strategy if self.config.Model.Diffuser.SO3.reverse_strategy == 'score_and_noise' else None,
                                                        rotation_reverse_noise_scale_override = self.config.Model.Diffuser.SO3.reverse_parm.score_and_noise.sampling_noise_scale if self.config.Model.Diffuser.SO3.reverse_strategy == 'score_and_noise' else None,
                                                        rotation_reverse_score_scale_override = self.config.Model.Diffuser.SO3.reverse_parm.score_and_noise.sampling_score_scale if self.config.Model.Diffuser.SO3.reverse_strategy == 'score_and_noise' else None,
                                                        )
                    
                prediction['sample_idx'] = sample_idx
                prediction['seed'] = sd
        
            torch.cuda.empty_cache()
        
        return {'prediction': prediction, 'batch_idx' : batch_idx, 'sample_idx' : sample_idx}


    def sampling_epoch_end(self, label = None):
        if self.use_ddp:
            dist.barrier()
        return
    
    def sampling_step_end(self, prediction_dict, label = None):
        sample_idx = prediction_dict['sample_idx']

        save_dir = os.path.join(self.train_config.Base.sampling_save_dir, 'epoch_%d' % self.current_epoch if label is None else 'epoch_%d_%s' % (self.current_epoch, label), 'length_%d' % self.length_cur_epoch)

        pdb_save_path = os.path.join(save_dir, 'sample_%d.pdb' % (sample_idx))
            
        coord37_record, coord37_mask = self.sb.coord14_to_coord37(prediction_dict['coord_hat'], trunc=True)
        prot_traj = self.sb.get_coord_traj(coord37_record[None],
                                    aa_mask=coord37_mask,
                                    label_override='sample_%d, seed %d' % (sample_idx, prediction_dict['seed']),
                                    default_res='G'
                                    )
        with open(pdb_save_path, 'w') as f:
            f.write(protein.to_pdb(prot_traj[0]))

    def _log_to_main(self, loss_dict_list):
        log_list = []
        for key in self.train_config.Base.log_to_main_metric_list:
            if key in loss_dict_list[0]:
                cur_loss = np.concatenate([loss_dict[key] for loss_dict in loss_dict_list], axis = 0)
                log_list.append('%s: %.4f' % (key, np.mean(cur_loss)))
        log_str = ', '.join(log_list)
        logger.info("Epoch %d, step %d: %s" % (self.current_epoch, self.step_in_epoch, log_str))

    def _save_embedding(self, output_dicts_list):
        n_rank = len(output_dicts_list)
        n_step = len(output_dicts_list[0])

        result_dict_flatten = [dcts[i] for i in range(n_step) for dcts in output_dicts_list]
        embedding_dict = {}

        for key in result_dict_flatten[0].keys():
            embedding_dict[key] = np.concatenate([d[key] for d in result_dict_flatten], axis = 0)

        # reorder result by sample_idx
        order_dict = {}
        for i, sample_idx in enumerate(embedding_dict['sample_idx']):
            if sample_idx not in order_dict:
                order_dict[sample_idx] = i
        new_order = [order_dict[sample_idx] for sample_idx in range(len(order_dict))]
        for k, v in embedding_dict.items():
            embedding_dict[k] = v[new_order]

        embedding_save_path = os.path.join(self.train_config.Base.embedding_save_dir, "epoch_%d.pkl" % self.current_epoch)
        with open(embedding_save_path, 'wb') as f:
            pickle.dump(embedding_dict, f)
        
    def _save_record(self, output_dicts_list, epoch_end = False):
        n_rank = len(output_dicts_list)
        n_step = len(output_dicts_list[0])

        train_loss_dict_flatten = [dcts[i] for i in range(n_step) for dcts in output_dicts_list]
        train_record_dict = {}

        for key in train_loss_dict_flatten[0].keys():
            train_record_dict['train/' + key] = np.concatenate([d[key] for d in train_loss_dict_flatten])

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
                state_dict['global_accum_step'] = self.global_accum_step
                state_dict_path = os.path.join(self.train_config.Base.ckpt_save_dir, "%s_state.pkl" % label)
                torch.save(state_dict, state_dict_path)