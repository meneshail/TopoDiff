import os
from tqdm import tqdm
import logging
import pickle

import torch
import numpy as np

import mdtraj as md
from tmtools import tm_align

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import ward, fcluster

from TopoDiff.data.experiment import ReadCoordDataset, TMAlignDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TopoDiff.evaluation.diversity")

def compute_tm_matrix(path_list_1,
                      key_list_1 = None,
                      path_list_2 = None,
                      key_list_2 = None,
                      n_workers = 40,
                      save_path = None):
    """
    Compute pairwise TM-score matrix for list of pdbs.
    if path_list_2 is None, will compute the inner TM-score matrix for path_list_1.
    if path_list_2 is not None, will compute the cross TM-score matrix for path_list_1 and path_list_2.
    Args:
        path_list_1:
            list of pdb file paths. (N1)
        key_list_1:
            name list for path_list_1. (Optional)
        path_list_2:
            list of pdb file paths. (N2)
        key_list_2:
            name list for path_list_2. (Optional)
        n_workers:
            number of workers for parallel computing.
        save_path:
            save path for the output matrix. (Optional)
    """

    ########## read coordinates 1 ##########
    ca_coord_dict_1 = {}
    if key_list_1 is not None:
        assert len(key_list_1) == len(path_list_1), "key_list_1 should have the same length as path_list_1."
    else:
        key_list_1 = list(range(len(path_list_1)))
    task_list = [{'pdb_path': pdb_path, 'key': key, 'idx': i} for i, (pdb_path, key) in enumerate(zip(path_list_1, key_list_1))]
    dataset = ReadCoordDataset(task_list)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = n_workers,
        collate_fn = lambda x: x[0],
    )
    for res_dict in tqdm(loader):
        ca_coord_dict_1[res_dict['key']] = res_dict['ca_coord']

    ########## read coordinates 2 ##########
    if path_list_2 is not None:
        ca_coord_dict_2 = {}
        if key_list_2 is not None:
            assert len(key_list_2) == len(path_list_2), "key_list_2 should have the same length as path_list_2."
        else:
            key_list_2 = list(range(len(path_list_2)))
        task_list = [{'pdb_path': pdb_path, 'key': key, 'idx': i} for i, (pdb_path, key) in enumerate(zip(path_list_2, key_list_2))]
        dataset = ReadCoordDataset(task_list)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = n_workers,
            collate_fn = lambda x: x[0],
        )
        for res_dict in tqdm(loader):
            ca_coord_dict_2[res_dict['key']] = res_dict['ca_coord']
    else:
        key_list_2 = None
        ca_coord_dict_2 = None

    ########## compute TM-score matrix ##########
    tm_dataset = TMAlignDataset(
        coord_dict_1=ca_coord_dict_1,
        coord_dict_2=ca_coord_dict_2,
        key_1=key_list_1,
        key_2=key_list_2,
    )
    tm_loader = torch.utils.data.DataLoader(
        tm_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = n_workers,
        collate_fn = lambda x: x[0],
    )

    if ca_coord_dict_2 is None:
        tm_matrix = np.zeros((len(path_list_1), len(path_list_1)))
    else:
        tm_matrix = np.zeros((len(path_list_1), len(path_list_2)))

    for res_dict in tqdm(tm_loader):
        tm_matrix[res_dict['i'], res_dict['j']] =  (res_dict['tm_norm_chain1'] + res_dict['tm_norm_chain1']) / 2

    if ca_coord_dict_2 is None:
        tm_matrix = tm_matrix + tm_matrix.T - np.diag(np.diag(tm_matrix))

    res_dict = {
        'tm_matrix': tm_matrix,
        'key_list_1': key_list_1,
        'key_list_2': key_list_2,
    }

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            logger.info(f"Create directory for {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            logger.warning(f"Overwrite existing file {save_path}")
            raise ValueError(f"File {save_path} already exists.")


        with open(save_path, 'wb') as f:
            pickle.dump(res_dict, f)

    return res_dict

def compute_unique_cluster(dis_mat,
                      cutoff = 0.4,
                      ):
    """
    Compute number of unique clusters based on distance matrix and cutoff.
    Args:
        dis_mat:
            inner distance matrix. (N, N)
        cutoff:
            cutoff for clustering.
    Returns:
        n_cluster:
            number of unique clusters.
    """
    condensed_dst = squareform(dis_mat)
    linkage = ward(condensed_dst)
    cluster = fcluster(linkage, cutoff, criterion='distance')
    n_cluster = len(np.unique(cluster))
    return n_cluster





