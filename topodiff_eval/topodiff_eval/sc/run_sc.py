import os
import sys
import time
from tqdm import tqdm
import numpy as np
import subprocess
import logging
import pandas as pd
from biotite.sequence.io import fasta
from typing import Optional
import glob
import dataclasses
import traceback

import torch
import multiprocessing as mp
from multiprocessing import Manager

import esm
import logging

from topodiff_eval.sc.utils import calc_aligned_rmsd_corrected, calc_tm_score

sc_module_dir = os.path.dirname(os.path.abspath(__file__))
topodiff_dir = os.path.join(sc_module_dir, '..', '..', '..', 'TopoDiff')
sys.path.insert(0, os.path.abspath(topodiff_dir))

import myopenfold.np.residue_constants as rc
from myopenfold.np import protein

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(name)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger('self_consistency_test')
logger.setLevel(logging.DEBUG)

pmpnn_dir = os.path.join(sc_module_dir, '..', '..', 'ProteinMPNN')

restypes_with_x = np.array(rc.restypes_with_x)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run self consistency test using multiprocessing')
    parser.add_argument('--gpu_list', type=str, default='0,1,2,3,4,5,6,7',
                        help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--sample_root', type=str, default=None, required=True,
                        help='Root directory containing source structure samples')
    parser.add_argument('--sc_test_root', type=str, default=None, required=True,
                        help='Root directory to store self-consistency test results')

    parser.add_argument('--length_list', type=str, default='50',
                        help='Comma-separated list of sequence lengths to test')
    parser.add_argument('--n_sample', type=int, default=100,
                        help='Number of samples per length')
    parser.add_argument('--seq_per_sample', type=int, default=8,
                        help='Number of sequences to generate per structure with ProteinMPNN')
    parser.add_argument('--run_phase_1', action='store_true',
                        help='Run phase 1: structure to sequence with ProteinMPNN')
    parser.add_argument('--run_phase_2', action='store_true',
                        help='Run phase 2: sequence to structure with ESMFold')

    parser.add_argument('--ca_only', action='store_true',
                        help='Use CA-only mode for ProteinMPNN')
    parser.add_argument('--compute_plddt', action='store_true',
                        help='Compute pLDDT scores for ESMFold outputs')
    parser.add_argument('--update_filename', type=str, default='sc_results.csv',
                        help='Filename for storing the SC test results')
    return parser.parse_args()

class SelfConsistencyTester:

    def __init__(
            self,
            rank,
            task_queue,
            result_queue,
            seq_per_sample,
            run_phase_1,
            run_phase_2,
            ca_only,
            compute_plddt,
            update_filename,
        ):
        """Initialize sampler.

        Args:
            rank: GPU rank
            task_queue: Queue of tasks to process
            result_queue: Queue to store results
        """
        self._log = logging.getLogger('%s-%d' % (__name__, rank))

        self.rank = rank

        self.task_queue = task_queue
        self.result_queue = result_queue

        self.gpu_id = GPU_LIST[rank]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        self._rng = np.random.default_rng(42)
        self.device = torch.device(f'cuda:0')
        self.seq_per_sample = seq_per_sample
        self.run_phase_1 = run_phase_1
        self.run_phase_2 = run_phase_2
        self.ca_only = ca_only
        self.compute_plddt = compute_plddt
        self.update_filename = update_filename

        # Set model hub directory for ESMFold.
        # torch.hub.set_dir(pt_hub_dir)

        self._pmpnn_dir = pmpnn_dir

        # Load models and experiment
        if self.run_phase_2:
            self._folding_model = esm.pretrained.esmfold_v1().eval()
            self._folding_model = self._folding_model.to(self.device)

    def run_all(self):
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get()

                logger.info(f'rank {self.rank}, task: {task["sample_dest_path"]}')
                
                sample_src_path = task['sample_src_path']
                sample_dest_path = task['sample_dest_path']
                sample_result_path = os.path.join(os.path.dirname(sample_dest_path), self.update_filename)

                assert os.path.exists(sample_src_path), "sample_src_path not found: %s" % sample_src_path

                if not os.path.exists(os.path.dirname(sample_dest_path)):
                    os.makedirs(os.path.dirname(sample_dest_path), exist_ok=True)

                if os.path.exists(sample_result_path):
                    logger.info(f'{sample_result_path} exists, skip.')
                    self.result_queue.put((task, 'success'))
                    continue

                if self.run_phase_1:
                    if os.path.exists(sample_dest_path):
                        logger.info(f'{sample_dest_path} exists, skip symlinking.')
                    else:
                        os.symlink(sample_src_path, sample_dest_path)

                    if not self.run_phase_2:
                        sample_dir = os.path.dirname(sample_dest_path)
                        mpnn_fasta_path = os.path.join(
                            sample_dir,
                            'seqs',
                            os.path.basename(sample_dest_path).replace('.pdb', '.fa')
                        )
                        if os.path.exists(mpnn_fasta_path):
                            try:
                                fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
                                if len(fasta_seqs) >= self.seq_per_sample:
                                    logger.info(f'{mpnn_fasta_path} exists, skip phase 1.')
                                    self.result_queue.put((task, 'success'))
                                    continue
                            except Exception as e:
                                logger.error(f'Error reading existing fasta file: {mpnn_fasta_path}, re-running phase 1.')
                                os.remove(mpnn_fasta_path)

                self.run_self_consistency(
                    decoy_pdb_dir=os.path.dirname(sample_dest_path),
                    reference_pdb_path=sample_dest_path,
                )

            except Exception as e:
                logger.error(f'Error in task: {task}, {e}: {traceback.format_exc()}')
                self.result_queue.put((task, 'failed'))
            else:
                self.result_queue.put((task, 'success'))

    def run_self_consistency(
            self,
            decoy_pdb_dir: str,
            reference_pdb_path: str):
        """Run self-consistency on design proteins against reference protein.
        
        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
            Writes results in decoy_pdb_dir/sc_results.csv
        """

        # Run PorteinMPNN
        torch.cuda.empty_cache()

        if self.run_phase_1:
            output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
            process = subprocess.Popen([
                'python',
                f'{self._pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
                f'--input_path={decoy_pdb_dir}',
                f'--output_path={output_path}',
            ])
            _ = process.wait()
            num_tries = 0
            ret = -1
            pmpnn_args = [
                'python',
                f'{self._pmpnn_dir}/protein_mpnn_run.py',
                '--out_folder',
                decoy_pdb_dir,
                '--jsonl_path',
                output_path,
                '--num_seq_per_target',
                str(self.seq_per_sample),
                '--sampling_temp',
                '0.1',
                '--seed',
                '38',
                '--batch_size',
                '1',
            ]
            pmpnn_args.append('--device')
            pmpnn_args.append('0')  # Use device 0 as we've set CUDA_VISIBLE_DEVICES
            if self.ca_only:
                pmpnn_args.append('--ca_only')
            while ret < 0:
                try:
                    process = subprocess.Popen(
                        pmpnn_args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    out, err = process.communicate()
                    ret = process.returncode
                    if ret != 0:
                        print('ret', ret)
                        print('stdout', out)
                        print('stderr', err)
                    
                except Exception as e:
                    num_tries += 1
                    self._log.info('Executing command failed: ' + ' '.join(pmpnn_args))
                    self._log.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                    # log error stacktrace
                    self._log.error(e)
                    torch.cuda.empty_cache()
                    if num_tries > 4:
                        raise e

        if self.run_phase_2:
            mpnn_fasta_path = os.path.join(
                decoy_pdb_dir,
                'seqs',
                os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
            )

            # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
            mpnn_results = {
                'tm_score': [],
                'sample_path': [],
                'header': [],
                'sequence': [],
                'rmsd': [],
            }
            if self.compute_plddt:
                mpnn_results['esm_plddt_ca'] = []
                mpnn_results['esm_plddt_backbone'] = []

            esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
            os.makedirs(esmf_dir, exist_ok=True)
            fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
            sample_feats = dataclasses.asdict(protein.from_pdb_string(open(reference_pdb_path, 'r').read()))
            for i, (header, string) in enumerate(fasta_seqs.items()):

                # Run ESMFold
                esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
                if i == 0:
                    continue
                else:
                    _ = self.run_folding(string, esmf_sample_path)

                esmf_feats = dataclasses.asdict(protein.from_pdb_string(open(esmf_sample_path, 'r').read()))
                sample_seq = ''.join(restypes_with_x[sample_feats['aatype']])

                # Calculate scTM of ESMFold outputs with reference protein
                _, tm_score = calc_tm_score(
                    sample_feats['atom_positions'][:, 1], esmf_feats['atom_positions'][:, 1],
                    sample_seq, sample_seq)
                rmsd = calc_aligned_rmsd_corrected(
                    sample_feats['atom_positions'][:, 1], esmf_feats['atom_positions'][:, 1])

                mpnn_results['tm_score'].append(tm_score)
                mpnn_results['sample_path'].append(esmf_sample_path)
                mpnn_results['header'].append(header)
                mpnn_results['sequence'].append(string)
                mpnn_results['rmsd'].append(rmsd)

                if self.compute_plddt:
                    # Calculate plddt of ESMFold outputs
                    plddt_ca = esmf_feats['b_factors'][:, 1].reshape(-1)[esmf_feats['atom_mask'][:, 1].reshape(-1) == 1].mean()
                    plddt_backbone = esmf_feats['b_factors'][:, :5].reshape(-1)[esmf_feats['atom_mask'][:, :5].reshape(-1) == 1].mean()
                    mpnn_results['esm_plddt_ca'].append(plddt_ca)
                    mpnn_results['esm_plddt_backbone'].append(plddt_backbone)

            # Save results to CSV
            csv_path = os.path.join(decoy_pdb_dir, self.update_filename)
            mpnn_results = pd.DataFrame(mpnn_results)
            mpnn_results.to_csv(csv_path)

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output

def worker_process(rank, task_queue, result_queue, args):
    tester = SelfConsistencyTester(
        rank=rank, 
        task_queue=task_queue, 
        result_queue=result_queue,
        seq_per_sample=args.seq_per_sample,
        run_phase_1=args.run_phase_1,
        run_phase_2=args.run_phase_2,
        ca_only=args.ca_only,
        compute_plddt=args.compute_plddt,
        update_filename=args.update_filename,
    )
    tester.run_all()

if __name__ == "__main__":
    args = parse_args()
    GPU_LIST = [int(x) for x in args.gpu_list.split(',')]
    LENGTH_LIST = [int(x) for x in args.length_list.split(',')]
    REP_LIST = list(range(args.n_sample))

    ############################# init #############################
    logger.info('start self consistency test with multiprocessing..')
    logger.info(f'GPU list: {GPU_LIST}')

    ############################# get tasks #############################
    task_list = []

    for j, length in enumerate(LENGTH_LIST):
        for k, rep_idx in enumerate(REP_LIST):
            sample_src_path = os.path.join(args.sample_root, 'length_%d' % length, 'sample_%d.pdb' % rep_idx)
            sample_dest_dir = os.path.join(args.sc_test_root, 'length_%d' % length, 'sample_%d' % (rep_idx))
            sample_dest_path = os.path.join(sample_dest_dir, 'sample.pdb')

            task_list.append({
                'sample_src_path': sample_src_path,
                'sample_dest_path': sample_dest_path,
            })

    ############################# multi processing #############################
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    for task in task_list:
        task_queue.put(task)
    
    ############################# run tasks #############################
    processes = []
    for i in range(len(GPU_LIST)):
        p = mp.Process(target=worker_process, args=(i, task_queue, result_queue, args))
        logger.info(f'Starting process: rank {i}')
        p.start()
        processes.append(p)

    with tqdm(total=len(task_list)) as pbar:
        n_completed = 0
        n_success = 0
        n_failed = 0
        while n_completed < len(task_list):
            try:
                result = result_queue.get()
                n_completed += 1
                if result[1] == 'success':
                    n_success += 1
                elif result[1] == 'failed':
                    n_failed += 1
                pbar.set_postfix_str(f'success: {n_success}, failed: {n_failed}')
                pbar.update(1)
            except Exception as e:
                logger.error(f"Error tracking progress: {e}")
                pass

    for p in processes:
        p.join()

    ############################# finish #############################
    logger.info('Self consistency test finished.')