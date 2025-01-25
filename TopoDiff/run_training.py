import os
import sys
import argparse
import logging
from tqdm import tqdm
import pickle
from datetime import timedelta
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from TopoDiff.experiment.trainer import MyTrainer
from TopoDiff.experiment.latent_trainer import MyLatentTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Run training experiment with TopoDiff model')

    # exp dir
    parser.add_argument('-o', '--outdir', type=str, default=None, help='The output directory')
    parser.add_argument('--model', type=str, default=None, help='The model to train (choose from structure or latent)')

    # structure diffusion training
    parser.add_argument('--stage', type=int, default=None, help='The stage to train (1, 2, 3)')
    parser.add_argument('--batch_size', type=int, default=8, help='The batch size')
    parser.add_argument('--n_epoch', type=int, default=1000, help='The number of epochs')
    parser.add_argument('--init_ckpt', type=str, default=None, help='The initial checkpoint')

    # latent diffusion training
    parser.add_argument('--latent_epoch', type=int, default=550, help='Selected epoch for latent diffusion training')

    # common
    parser.add_argument('-s', '--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--gpu', type=str, default=None, help='The GPU devices to use')

    return parser.parse_args()

if __name__ == '__main__':

    cur_path = os.path.dirname(os.path.realpath(__file__))

    ############################# parse args #############################
    args = parse_args()
    if args.outdir is None:
        raise ValueError('Output directory is not specified')

    if args.model is None or args.model not in ['structure', 'latent']:
        raise ValueError('Invalid model to train, should be structure or latent')

    if args.model == 'structure':
        if args.stage is None or args.stage not in [1, 2, 3]:
            raise ValueError('Invalid training stage, should be 1, 2, or 3')

        if args.stage in [2, 3]:
            if args.init_ckpt is None:
                raise ValueError('Please specify the initial checkpoint for stage 2 or 3, or make sure you know what you are doing (then you can ignore this)')
            elif not os.path.exists(args.init_ckpt):
                raise ValueError('Initial checkpoint does not exist')

        if args.gpu is None:
            raise ValueError('Please specify the GPU devices to use')
        else:
            gpu_list = list(map(int, args.gpu.split(',')))
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            if 'LOCAL_RANK' in os.environ:
                local_rank = int(os.environ['LOCAL_RANK'])
                world_size = int(os.environ['WORLD_SIZE'])
            else:
                local_rank = 0
                world_size = 1

            assert len(gpu_list) == world_size, 'Number of GPUs does not match the world size'

    elif args.model == 'latent':
        if args.latent_epoch is None:
            raise ValueError('Please specify the selected epoch for latent diffusion training')
        
        train_data_path = os.path.join(args.outdir, 'save', 'embedding', 'epoch_%d.pkl' % args.latent_epoch)
        if not os.path.exists(train_data_path):
            raise ValueError('Training data does not exist, please first compute the embeddings for the selected epoch')

        if args.gpu is None:
            raise ValueError('Please specify the GPU devices to use')
        else:
            gpu_list = list(map(int, args.gpu.split(',')))
            n_gpu = len(gpu_list)
            if n_gpu > 1:
                raise ValueError('Latent diffusion training does not support multi-GPU training')
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            rank = 0
            world_size = 1

        logger.info("Start training with args: %s" % args)

    ############################# trainer #############################

    if args.model == 'structure':
        # print('setup, rank', local_rank)
        if world_size == 1:
            logger.info('World size is 1, disable ddp')
            pass
        else:
            logger.info('World size is %d, enable ddp' % world_size)
            print(os.environ['CUDA_VISIBLE_DEVICES'], local_rank)
            print(os.environ['MASTER_ADDR'], local_rank)
            print(os.environ['MASTER_PORT'], local_rank)
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend = "nccl")

        trainer = MyTrainer(args, local_rank, world_size)
        trainer.run()

    elif args.model == 'latent':
        trainer = MyLatentTrainer(args, rank, world_size)
        trainer.run()

