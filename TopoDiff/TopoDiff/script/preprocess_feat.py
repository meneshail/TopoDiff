# TopoDiff/script/preprecess_feat.py

import os
import argparse
import glob
import json
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

import torch

# Package imports - requires TopoDiff to be installed as a package
from TopoDiff.config.config import model_config
from TopoDiff.data import feature_pipeline
from myopenfold.data import data_pipeline


def process_single_file(item, feat_dir, dp, fp):
    """Process a single PDB file and extract features."""
    key, dct = item
    pdb_path = dct['pdb_path']
    feat_path = os.path.join(feat_dir, dct['path'] + '.pt')
    
    # Create subdirectories as needed
    feat_dir_for_file = os.path.dirname(feat_path)
    os.makedirs(feat_dir_for_file, exist_ok=True)

    try:
        dat = dp.process_pdb(
            pdb_path=pdb_path,
            is_distillation=False,
            alignment_dir=None,
            chain_id='A',
            parse_msa=False,
        )
        dat_processed = fp.preprocess_features(dat, 'eval')

        trimmed_length = dat_processed['seq_idx'].shape[0]
        
        torch.save(dat_processed, feat_path)
        
        result_dict = {
            'trimmed_length': trimmed_length,
            'feat_path': os.path.abspath(feat_path),
            'pdb_path': os.path.abspath(pdb_path),
        }
        return key, True, result_dict
    except Exception as e:
        return key, False, str(e)


def main():
    """Main function for processing PDB files to features."""
    parser = argparse.ArgumentParser(description='Process PDB files to features using TopoDiff pipeline')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Input directory containing PDB files')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory for features and metadata')
    parser.add_argument('--n_worker', type=int, default=35, help='Number of worker processes')
    args = parser.parse_args()
    
    # Create output directories
    feat_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(feat_dir, exist_ok=True)
    
    # Find all PDB files
    print("Searching for PDB files...")
    pdb_files = glob.glob(os.path.join(args.input_dir, '**', '*.pdb'), recursive=True)
    print(f"Found {len(pdb_files)} PDB files")
    
    # Create metadata dictionary
    info_dict = {}
    for pdb_path in pdb_files:
        rel_path = os.path.relpath(pdb_path, args.input_dir)
        key = os.path.splitext(rel_path)[0]
        info_dict[key] = {
            'path': key,
            'pdb_path': pdb_path,
        }

    # Initialize processing pipelines
    print("Initializing feature processing pipelines...")
    cfg = model_config('train_stage_3', low_prec=False, extra=['preprocess_structure'])
    dp = data_pipeline.DataPipeline()
    fp = feature_pipeline.PreprocessedFeaturePipeline(cfg.Data)

    # Process files with multiprocessing
    print("Processing PDB files...")
    worker_fn = partial(process_single_file, feat_dir=feat_dir, dp=dp, fp=fp)

    results = []
    with Pool(args.n_worker) as p:
        results = list(tqdm(p.imap(worker_fn, info_dict.items()), total=len(info_dict)))
    
    # Collect results
    successful_results = [r for r in results if r[1]]
    failed_results = [r for r in results if not r[1]]
    
    print(f"\nSuccessfully processed: {len(successful_results)}/{len(info_dict)} files")
    if failed_results:
        print(f"Failed to process {len(failed_results)} files:")
        for key, _, error in failed_results:
            print(f"  - {key}: {error}")
    
    # Save metadata
    meta_info = {key: result_dict for key, _, result_dict in successful_results}
    
    info_json_path = os.path.join(args.output_dir, 'info.json')
    with open(info_json_path, 'w') as f:
        json.dump(meta_info, f, indent=2, sort_keys=True)

    print(f"\nFeature processing complete. Metadata saved to: {info_json_path}")


if __name__ == '__main__':
    main()