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

#. sampler
from TopoDiff.experiment.sampler import Sampler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Run sampling experiment with TopoDiff model')

    # outdir
    parser.add_argument('-o', '--outdir', type=str, default=None, help='The output directory')

    # model version
    parser.add_argument('-v', '--version', type=str, default='v1_1_2', help='The version of the model, default: v1_1_2 (recommended)')

    # sampling mode
    parser.add_argument('-m', '--mode', type=str, default=None, help='The mode of sampling (model variants with different sampling preference), default: None. Available options [base, designability, novelty, all_round]. Note that set this to a valid option will orverride the pred_* options.')

    # length
    parser.add_argument('-s', '--start', type=int, default=100, help='The start length of sampling, must be larger than 50, default: 100')
    parser.add_argument('-e', '--end', type=int, default=100, help='The end length of sampling (inclusive), must be smaller than 250, default: 100')
    parser.add_argument('-i', '--interval', type=int, default=10, help='The interval of sampling length, default: 10')
    parser.add_argument('-n', '--num_samples', type=int, default=5, help='The number of samples to generate for each length, default: 5')

    # designability cutoff
    parser.add_argument('--pred_sc', default=False, action='store_true', help='Whether to predict designability score, default: False')
    parser.add_argument('--min_sc', type=str, default='0.', help='The minimum predicted designability score of the latent, default: 0.0')
    parser.add_argument('--max_sc', type=str, default='1.', help='The maximum predicted designability score of the latent, default: 1.0')

    # novelty cutoff
    parser.add_argument('--pred_novelty', default=False, action='store_true', help='Whether to predict novelty score, default: False')
    parser.add_argument('--min_novelty', type=str, default='0.', help='The minimum predicted novelty score of the latent, default: 0.0')
    parser.add_argument('--max_novelty', type=str, default='1.', help='The maximum predicted novelty score of the latent, default: 1.0')

    # alpha ratio
    parser.add_argument('--pred_alpha', default=False, action='store_true', help='Whether to predict alpha ratio, default: False')
    parser.add_argument('--min_alpha', type=str, default='0.0', help='The minimum predicted alpha ratio of the latent, default: 0.0')
    parser.add_argument('--max_alpha', type=str, default='1.0', help='The maximum predicted alpha ratio of the latent, default: 1.0')

    # beta ratio
    parser.add_argument('--pred_beta', default=False, action='store_true', help='Whether to predict beta ratio, default: False')
    parser.add_argument('--min_beta', type=str, default='0.0', help='The minimum predicted beta ratio of the latent, default: 0.0')
    parser.add_argument('--max_beta', type=str, default='1.0', help='The maximum predicted beta ratio of the latent, default: 1.0')

    # coil ratio
    parser.add_argument('--pred_coil', default=False, action='store_true', help='Whether to predict coil ratio, default: False')
    parser.add_argument('--min_coil', type=str, default='0.0', help='The minimum predicted coil ratio of the latent, default: 0.0')
    parser.add_argument('--max_coil', type=str, default='1.0', help='The maximum predicted coil ratio of the latent, default: 1.0')

    # soft prob
    parser.add_argument('--soft_prob', default=0.1, type=float, help='The probability for accepting latent codes failed to pass all classifiers, default: 0.1')

    # seed
    parser.add_argument('--seed', type=int, default=42, help='The random seed for sampling, default: 42')

    # gpu
    parser.add_argument('--gpu', type=str, default=None, help='The gpu id for sampling, default: None')

    # length prediction
    parser.add_argument('--num_k', type=int, default=1, help='The number of k to decide the expected length of the latent, default: 1')
    parser.add_argument('--epsilon', type=float, default=0.2, help='The range of variation of the expected length of the latent, default: 0.2')

    return parser.parse_args()

if __name__ == '__main__':

    cur_path = os.path.dirname(os.path.realpath(__file__))

    args = parse_args()

    # check validity of args
    if args.outdir is None:
        raise ValueError('Output directory is not specified')
    
    if args.version not in ['v1_1_2', 'v1_1_1']:
        raise ValueError('The model version is not supported')
    
    if args.start < 50:
        raise ValueError('The start length must be larger than 50')
    
    if args.end > 250:
        raise ValueError('The end length must be smaller than 250')
    
    if args.start > args.end:
        raise ValueError('The start length must be smaller than the end length')
    
    if args.num_samples < 1:
        raise ValueError('The number of samples must be larger than 0')

    if args.mode is not None:
        assert args.mode in ['base', 'designability', 'novelty', 'all_round'], 'The mode must be one of [base, designability, novelty, all_round]'
        assert args.version == 'v1_1_2', 'The mode is only supported for model version v1_1_2'
    
    sample_goal_list = [(l, args.num_samples) for l in range(args.start, args.end+1, args.interval)]
    
    if args.pred_sc:
        if ',' in args.min_sc:
            min_sc = [float(x) for x in args.min_sc.split(',')]
            assert len(min_sc) == len(sample_goal_list), 'The number of specified min_sc must be a single value or the same as the number of lengths'
        else:
            min_sc = [float(args.min_sc)] * len(sample_goal_list)
        if ',' in args.max_sc:
            max_sc = [float(x) for x in args.max_sc.split(',')]
            assert len(max_sc) == len(sample_goal_list), 'The number of specified max_sc must be a single value or the same as the number of lengths'
    else:
        min_sc = None
        max_sc = None

    if args.pred_novelty:
        if ',' in args.min_novelty:
            min_novelty = [float(x) for x in args.min_novelty.split(',')]
            assert len(min_novelty) == len(sample_goal_list), 'The number of specified min_novelty must be a single value or the same as the number of lengths'
        else:
            min_novelty = [float(args.min_novelty)] * len(sample_goal_list)
        if ',' in args.max_novelty:
            max_novelty = [float(x) for x in args.max_novelty.split(',')]
            assert len(max_novelty) == len(sample_goal_list), 'The number of specified max_novelty must be a single value or the same as the number of lengths'
    else:
        min_novelty = None
        max_novelty = None

    if args.pred_alpha:
        if ',' in args.min_alpha:
            min_alpha = [float(x) for x in args.min_alpha.split(',')]
            assert len(min_alpha) == len(sample_goal_list), 'The number of specified min_alpha must be a single value or the same as the number of lengths'
        else:
            min_alpha = [float(args.min_alpha)] * len(sample_goal_list)
        if ',' in args.max_alpha:
            max_alpha = [float(x) for x in args.max_alpha.split(',')]
            assert len(max_alpha) == len(sample_goal_list), 'The number of specified max_alpha must be a single value or the same as the number of lengths'
    else:
        min_alpha = None
        max_alpha = None

    if args.pred_beta:
        if ',' in args.min_beta:
            min_beta = [float(x) for x in args.min_beta.split(',')]
            assert len(min_beta) == len(sample_goal_list), 'The number of specified min_beta must be a single value or the same as the number of lengths'
        else:
            min_beta = [float(args.min_beta)] * len(sample_goal_list)
        if ',' in args.max_beta:
            max_beta = [float(x) for x in args.max_beta.split(',')]
            assert len(max_beta) == len(sample_goal_list), 'The number of specified max_beta must be a single value or the same as the number of lengths'
    else:
        min_beta = None
        max_beta = None

    if args.pred_coil:
        if ',' in args.min_coil:
            min_coil = [float(x) for x in args.min_coil.split(',')]
            assert len(min_coil) == len(sample_goal_list), 'The number of specified min_coil must be a single value or the same as the number of lengths'
        else:
            min_coil = [float(args.min_coil)] * len(sample_goal_list)
        if ',' in args.max_coil:
            max_coil = [float(x) for x in args.max_coil.split(',')]
            assert len(max_coil) == len(sample_goal_list), 'The number of specified max_coil must be a single value or the same as the number of lengths'
    else:
        min_coil = None
        max_coil = None

    if args.gpu is not None:
        gpu_list = [int(x) for x in args.gpu.split(',')]
    else:
        gpu_list = [0]
    
    if args.num_k < 1:
        raise ValueError('The number of k for length prediction must be larger than 0')

    if args.epsilon < 0:
        raise ValueError('The epsilon for length prediction must be larger than 0')

    # ourdir to absolute path
    args.outdir = os.path.abspath(args.outdir)

    # set up config
    sampling_config = {
        # timestamp
        'label': time.strftime('%Y%m%d_%H%M%S'),

        # outdir
        'output_dir': args.outdir,

        # length
        'sample_goal_list': sample_goal_list,

        # seed
        'latent_seed': args.seed,
        'structure_seed': args.seed,

        # preset mode for rejection sampling
        'mode_preset': args.mode,
        
        # soft prob
        'soft_prob': args.soft_prob,

        # designability cutoff
        'pred_sc': args.pred_sc,
        'min_sc': min_sc,
        'max_sc': max_sc,

        # novelty cutoff
        'pred_novelty': args.pred_novelty,
        'min_novelty': min_novelty,
        'max_novelty': max_novelty,

        # alpha ratio
        'pred_alpha': args.pred_alpha,
        'min_alpha': min_alpha,
        'max_alpha': max_alpha,

        # beta ratio
        'pred_beta': args.pred_beta,
        'min_beta': min_beta,
        'max_beta': max_beta,

        # coil ratio
        'pred_coil': args.pred_coil,
        'min_coil': min_coil,
        'max_coil': max_coil,

        # length prediction
        'length_k': args.num_k,
        'length_epsilon': args.epsilon,

        'latent_batch_size': min(10000, 50 * sum([tup[1] for tup in sample_goal_list])),
        'structure_batch_size': 1, # NOTE currently fixed
    }

    ############################# env #############################
    print('gpu_list', gpu_list)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        local_rank = 0
        world_size = 1
    logger.info("local rank %d, world size %d" % (local_rank, world_size))

    ############################# ddp #############################

    print('setup, rank', local_rank)
    if world_size == 1:
        logger.info('World size is 1, disable ddp')
        pass
    else:
        logger.info('World size is %d, enable ddp' % world_size)
        print(os.environ['CUDA_VISIBLE_DEVICES'], local_rank)
        print(os.environ['MASTER_ADDR'], local_rank)
        print(os.environ['MASTER_PORT'], local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend = "nccl", timeout=timedelta(seconds=24 * 60 * 60))

    # print('sampling_config', sampling_config)

    ############################# experiment #############################
    
    # set up sampler
    sampler = Sampler(
        output_dir=sampling_config['output_dir'],

        rank=local_rank,
        world_size=world_size,

        sample_config=sampling_config,
        model_version=args.version,

        extra = None,
    )

    # run sampling
    sampler.run()


