import os
from tqdm import tqdm
import logging
import pickle

import torch
import numpy as np

from progres import progres
import TopoDiff
from TopoDiff.evaluation.diversity import compute_tm_matrix

project_dir = os.path.join(TopoDiff.__path__[0], '../..')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TopoDiff.evaluation.diversity")

def compute_coverage(path_list,
                    metric = 'Progres',  # ['Progres', 'TM']
                    n_workers = 40,
                    length = 125,
                    scope = 25,
                    K = 100,
                    verbose = False,
    ):
    """
    Compute the coverage of a list of pdbs sampled at a certain length.

    Args:
        path_list:
            list of pdb file paths. (N)
        metric [Progres, TM]:
            metric for computing the coverage. 
        n_workers:
            number of workers for parallel computing.
        length:
            length of the sampled protein.
        scope:
            scope for the coverage calculation. reference proteins within [length-scope, length+scope] will be used.
        K:
            KNN parameter to define the receptive field of each reference protein.
    """
    if metric == 'Progres':
        # load CATH embeddings
        if verbose:
            logger.info('Loading precomputed CATH embeddings')
        cath_emb_filtered = filter_cath_embeddings(length, scope)
        # 1 - cosine similarity
        ref_dis_mat = 1 - np.matmul(cath_emb_filtered, cath_emb_filtered.T)

        # compute sample embeddings
        if verbose:
            logger.info('Computing embeddings for the sampled proteins')
        emb_sample = compute_progres_embedding(path_list)
        ref_sample_dis_mat = 1 - np.matmul(cath_emb_filtered, emb_sample.T)

        # compute coverage
        if verbose:
            logger.info('Computing coverage')
        # ref_dis_sorted = np.sort(ref_dis_mat, axis=1)[:, 1:]
        # ref_closest_dis = np.sort(ref_sample_dis_mat, axis=1)[:, 0]
        coverage = compute_coverage_with_precomputed_dist(ref_dis_mat, ref_sample_dis_mat, K = K)

    elif metric == 'TM':
        assert length in range(50, 251, 25) and scope == 25, "TM only supports length in [50, 75, 100, 125, 150, 175, 200, 225, 250] and scope = 25."
        coverage_data_dir = os.path.join(project_dir, 'data', 'evaluation', 'coverage')
        cath_tm_mat_dir = os.path.join(coverage_data_dir, 'cath_tm')
        cath_pdb_dir = os.path.join(coverage_data_dir, 'cath_pdb')

        # load TM score matrix
        if verbose:
            logger.info('Loading precomputed reference TM score matrix')
        # ref_tm_path = os.path.join(cath_tm_mat_dir, 'length_%d.pkl' % length)
        # with open(ref_tm_path, 'rb') as f:
        #     ref_tm_dict = pickle.load(f)
        # ref_dis_mat = 1 - (ref_tm_dict['tm_mat_norm_chain'] + ref_tm_dict['tm_mat_norm_chain'].T) / 2
        ref_dis_mat = load_cath_ref_tm_matrix(length, scope)

        # compute TM score matrix for the sampled proteins
        if verbose:
            logger.info('Computing reference v.s. sample TM score matrix')
        cath_pdb_path_list = [os.path.join(cath_pdb_dir, key) for key in ref_tm_dict['key_list']]
        res = compute_tm_matrix(path_list,
                        path_list_2 = cath_pdb_path_list,
                        n_workers = n_workers,
        )
        ref_sample_dis_mat = 1 - res['tm_matrix'].T

        # compute coverage
        if verbose:
            logger.info('Computing coverage')
        coverage = compute_coverage_with_precomputed_dist(ref_dis_mat, ref_sample_dis_mat, K = K)

    return coverage

def load_cath_ref_tm_matrix(length, scope):
    """
    Load precomputed TM score matrix for CATH reference proteins.
    Args:
        length:
            length of the sampled protein.
        scope:
            scope for the coverage calculation. reference proteins within [length-scope, length+scope] will be used.
    Returns:
        ref_dis_mat:
            distance matrix of reference proteins. (N_ref, N_ref)
    """
    assert length in range(50, 251, 25) and scope == 25, "TM only supports length in [50, 75, 100, 125, 150, 175, 200, 225, 250] and scope = 25."
    coverage_data_dir = os.path.join(project_dir, 'data', 'evaluation', 'coverage')
    cath_tm_mat_dir = os.path.join(coverage_data_dir, 'cath_tm')

    ref_tm_path = os.path.join(cath_tm_mat_dir, 'length_%d.pkl' % length)
    with open(ref_tm_path, 'rb') as f:
        ref_tm_dict = pickle.load(f)
    ref_dis_mat = 1 - (ref_tm_dict['tm_mat_norm_chain'] + ref_tm_dict['tm_mat_norm_chain'].T) / 2

    return ref_dis_mat

def filter_cath_embeddings(length, scope):
    """
    Filter CATH embeddings based on length and scope.
    Args:
        length:
            length of the sampled protein.
        scope:
            scope for the coverage calculation. reference proteins within [length-scope, length+scope] will be used.
    Returns:
        cath_emb_filtered:
            filtered CATH embeddings. (N, 128)
    """
    cath_emb_path = os.path.join(progres.__path__[0], 'databases', 'v_0_2_0', 'cath40.pt')
    cath_emb_dict = torch.load(cath_emb_path)
    cath_ref_length = torch.tensor(cath_emb_dict['nres'] )
    filter_mask = (cath_ref_length >= length - scope) & (cath_ref_length <= length + scope)
    cath_emb_filtered = cath_emb_dict['embeddings'][filter_mask].numpy()

    return cath_emb_filtered


def compute_coverage_with_precomputed_dist(ref_dis_mat,
                                            ref_sample_dis_mat,
                                            K = 100,
                                            ):
    """
    Compute the coverage with precomputed distance matrices.
    Args:
        ref_dis_mat:
            distance matrix of reference proteins. (N_ref, N_ref)
        ref_sample_dis_mat:
            distance matrix of reference proteins and sampled proteins. (N_ref, N_sample)
        K:
            KNN parameter to define the receptive field of each reference protein.
        n_sample:
            number of sampled proteins. If None, will use the 2nd dimension of ref_sample_dis_mat.
    """
    ref_dis_sorted = np.sort(ref_dis_mat, axis=1)[:, 1:]
    ref_closest_dis = np.sort(ref_sample_dis_mat, axis=1)[:, 0]
    coverage = np.sum(ref_closest_dis <= ref_dis_sorted[:, K-1]) / ref_closest_dis.shape[0]

    return coverage


def compute_progres_embedding(path_list,
                              n_workers = 40,
                              batch_size = 8,
                              save_path = None,
                              ):
    """
    Compute Progres embedding for list of pdbs.
    Args:
        path_list:
            list of pdb file paths. (N)
        n_workers:
            number of workers for parallel computing.
        save_path:
            save path for the computed embeddings. (Optional)
    Returns:
        embeddings (N, 128)
    """
    n_sample = len(path_list)   
    model = progres.load_trained_model()

    dataset = progres.StructureDataset(path_list,
                        fileformat = 'guess',
                        model = model,
                        device = 'cpu')

    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = n_workers,
                                                )

    emb_total = torch.zeros((n_sample, 128), dtype=torch.float32)
    with torch.no_grad():
        for i, (embs, nres) in enumerate(tqdm(dataloader)):
            start_idx = i * batch_size
            end_idx = start_idx + embs.size(0)
            emb_total[start_idx : end_idx] = embs

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            logger.warning('Embedding file already exists')
            raise FileExistsError

        torch.save(emb_total, save_path)

    return emb_total




