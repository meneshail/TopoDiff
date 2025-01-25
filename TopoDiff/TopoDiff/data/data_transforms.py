from functools import reduce, wraps
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from myopenfold.np import residue_constants as rc
from myopenfold.utils.rigid_utils import Rotation, Rigid, rot_vec_mul
from myopenfold.utils.tensor_utils import tree_map, tensor_tree_map, batched_gather

from TopoDiff.config.config import NUM_RES, NUM_EXTRA_SEQ, NUM_MSA_SEQ
from TopoDiff.data.encoder_transform import coords_to_dict, add_noise

def make_one_hot(x, num_classes):
    """Make one-hot encoding from indices.
    """
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot

def curry1(f):
    """Supply all arguments but the first."""

    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc

#################### sequence ####################
@curry1
def make_seq_feat(protein, config_mask):
    """Create sequence features."""
    has_break = torch.zeros_like(protein["seq_type"])
    if config_mask.enabled:
        aatype_1hot = make_one_hot(torch.ones_like(protein["seq_type"], dtype = torch.long) * 20, 21)
    else:
        aatype_1hot = make_one_hot(protein["seq_type"], 21)

    target_feat = [
        torch.unsqueeze(has_break, dim=-1),
        aatype_1hot,
    ]

    #. [N_res, 21 + 1]
    protein["target_feat"] = torch.cat(target_feat, dim=-1)

    return protein

@curry1
def update_seq_idx(protein):
    """
    """
    if 'seq_idx' in protein:
        min_idx = protein['seq_idx'][0].item()
        protein['seq_idx'] = protein['seq_idx'] - min_idx
    else:
        length = protein['seq_mask'].shape[0]
        protein['seq_idx'] = torch.arange(length, dtype=torch.int32)
    return protein

#################### structure ####################
def get_fake_atom14_from_atom37(protein):
    protein['atom14_gt_positions'] = protein['coord_gt'][:, [0, 1, 2, 4, 3]]
    protein['atom14_gt_exists'] = protein['coord_gt_mask'][:, [0, 1, 2, 4, 3]]
    return protein

@curry1
def get_chi_angles(protein):
    if 'torsion_gt' not in protein:
        return protein
    
    dtype = protein['coord_gt_mask'].dtype

    #. [N_res, 4, 2]
    protein["chi_angles_sin_cos"] = (
        protein["torsion_gt"][..., 3:, :]
    ).to(dtype)

    #. [N_res, 4]
    protein["chi_mask"] = protein["torsion_gt_mask"][..., 3:].to(dtype)

    return protein

@curry1
def center_backbone_frames(protein, config):
    #. [N_res, 3]
    translation = protein["frame_gt"][..., :3, -1]
    mask = protein["frame_gt_mask"].bool()

    if config.random_rotation:
        rot = torch.from_numpy(R.random().as_matrix().T.astype(np.float32))
        protein["frame_gt"][mask, :3] = rot @ protein["frame_gt"][mask, :3]

        if config.oper_on_coord14:
            protein['atom14_gt_positions'] = rot_vec_mul(rot, protein['atom14_gt_positions'])

    if config.centered:
        deviation_vec = torch.rand([3])
        deviation_vec /= torch.norm(deviation_vec)
        radius = torch.rand(1) * config.shift_radius
        deviation = deviation_vec * radius

        translation_shift = deviation - translation[mask].mean(axis=0)
        translation_centered = translation + translation_shift[None]

        protein["frame_gt"][mask, :3, -1] = translation_centered[mask]

        if config.oper_on_coord14:
            protein['atom14_gt_positions'] += translation_shift[None, None]
    else:
        pass

    return protein

@curry1
def get_coord(protein, config = None):
    if config is None or config.coord_type is None:
        pass
    elif config.coord_type == 'backbone':
        protein['coord_gt'] = protein['atom14_gt_positions'][:, :4]
        protein['coord_gt_mask'] = protein['atom14_gt_exists'][:, :4]
    else:
        raise ValueError("Unknown coord type: {}".format(config.coord_type))
    return protein

#################### diffusion ####################
@curry1
def add_timestep(protein, config):
    protein["timestep"] = torch.randint(1, config.T + 1, ())
    return protein

@curry1
def add_frame_mask(protein, config_partial_fixed):
    n_res = protein['frame_gt_mask'].shape[0]
    protein['seq_mask'] = protein['frame_gt_mask']

    if config_partial_fixed.enabled:
        rd = torch.rand(())
        if rd < config_partial_fixed.fixed_prob:
            ratio = config_partial_fixed.fixed_ratio[0] + torch.rand(()) * (config_partial_fixed.fixed_ratio[1] - config_partial_fixed.fixed_ratio[0])
            continue_mask = torch.rand(()) < config_partial_fixed.continuous_prob
            protein['fixed'] = torch.tensor(True)
            protein['fixed_ratio'] = ratio
            protein['fixed_continue'] = continue_mask
            
            if continue_mask:
                n_masked = int(n_res * ratio)
                sample_end = n_res - n_masked
                start = torch.randint(0, sample_end+1, ())
                mask = torch.ones(n_res, dtype=torch.bool)
                mask[start:start+n_masked+1] = False
            else:
                mask = torch.rand(n_res) > ratio
        else:
            protein['fixed'] = torch.tensor(False)
            protein['fixed_ratio'] = torch.tensor(0)
            protein['fixed_continue'] = torch.tensor(False)
            mask = torch.ones(n_res, dtype=torch.bool)
    else:
        mask = torch.ones(n_res, dtype=torch.bool)
    
    protein['frame_mask'] = mask & protein['frame_gt_mask'].bool()
    protein['motif_mask'] = (~mask) & protein['frame_gt_mask'].bool()
    return protein

#################### topo conditioning ####################
@curry1
def add_topo_conditiona(protein, config_topo):
    if config_topo.continuous.mask_prob is not None:
        rd = torch.rand(())
        if rd <= config_topo.continuous.mask_prob:
            protein['latent_mask'] = torch.tensor(0, dtype=torch.int64)
        else:
            protein['latent_mask'] = torch.tensor(1, dtype=torch.int64)
    return protein

@curry1
def process_encoder_feature(protein, config_encoder_feat, extra_config = None):
    """
    Process features for topology encoder.

    Args:
        coord_gt: [N_res, 4/5]
        coord_gt_mask: [N_res, 4/5]
    """
    return process_encoder_feature_no_wrap(protein, config_encoder_feat, extra_config)

def process_encoder_feature_no_wrap(protein, config_encoder_feat, extra_config = None):
    """
    Process features for topology encoder.

    Args:
        coord_gt: [N_res, 4/5]
        coord_gt_mask: [N_res, 4/5]
        
    Returns:
        encoder_feats: [N_res, N_c]
            preprocessed features for encoder model
        encoder_coords: [N_res, 3]
            coordinates for encoder model
        encoder_mask: [N_res]
            mask for encoder model
        encoder_adj_mat [N_res, N_res]
            adjacency matrix for encoder model
    """
    if config_encoder_feat.layer_type == 'egnn':
        coords = protein['coord_gt'][:, 1]
        coords_gt_mask = protein['coord_gt_mask'][:, 1]
        coords = coords[coords_gt_mask]

        # add noise to the coordinates
        if (config_encoder_feat.add_noise.enabled and 
            not (extra_config is not None and extra_config['encoder_no_noise'])):
            coords = add_noise(coords, config_encoder_feat.add_noise)
        
        # get graph features
        encoder_feat = coords_to_dict(coords)
        protein.update(encoder_feat)

        # add mask
        protein['encoder_mask'] = torch.ones(encoder_feat['encoder_feats'].size(0), dtype = torch.bool)
    
    elif config_encoder_feat.layer_type == 'ipa':
        coords = protein['coord_gt'][:, 1]
        coords_gt_mask = protein['frame_gt_mask']
        coords = coords[coords_gt_mask]

        # add noise to the coordinates
        # TODO: add small perturbation to the rigid transformations

        # get graph features
        seq_idx = protein['seq_idx'][coords_gt_mask]
        encoder_feat = coords_to_dict(coords, seq_idx = seq_idx)
        protein.update(encoder_feat)

        # add rigid transformation
        if config_encoder_feat.frame_type == '4x4':
            protein['encoder_frame_gt'] = protein['frame_gt'][coords_gt_mask]
        else:
            protein['encoder_frame_gt'] = Rigid.from_tensor_4x4(protein['frame_gt'][coords_gt_mask]).to_tensor_7()
        
        protein['encoder_mask'] = torch.ones(encoder_feat['encoder_feats'].size(0), dtype = torch.bool)

    else:
        raise ValueError("Unknown layer type: {}".format(config_encoder_feat.layer_type))

    return protein


#################### feature selection ####################

@curry1
def random_crop_to_size(
    protein,
    crop_size,
    shape_schema,
):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    g = None

    seq_length = protein["seq_length"]

    num_res_crop_size = min(int(seq_length), crop_size)

    def _randint(lower, upper):
        return int(torch.randint(
                lower,
                upper + 1,
                (1,),
                device=protein["seq_length"].device,
                generator=g,
        )[0])

    n = seq_length - num_res_crop_size
    right_anchor = n

    num_res_crop_start = _randint(0, right_anchor)

    for k, v in protein.items():
        if k not in shape_schema or (NUM_RES not in shape_schema[k]):
            continue

        slices = []
        for i, (dim_size, dim) in enumerate(zip(shape_schema[k], v.shape)):
            is_num_res = dim_size == NUM_RES
            crop_start = num_res_crop_start if is_num_res else 0
            crop_size = num_res_crop_size if is_num_res else dim
            slices.append(slice(crop_start, crop_start + crop_size))
        protein[k] = v[slices]

    protein["seq_length"] = protein["seq_length"].new_tensor(num_res_crop_size)
    
    return protein

@curry1
def select_feat(protein, feature_list, rename_dict={}, dtype_dict={}):
    selected = {}
    for k, v in protein.items():
        if k in rename_dict:
            k = rename_dict[k]
            
        if k in feature_list:
            if k in dtype_dict:
                v = v.to(eval(dtype_dict[k]))
            selected[k] = v
    return selected

@curry1
def make_fixed_size_simple(
    protein,
    shape_schema,
    num_res,
):
    """
        pad all features to their max size.
    """
    pad_size_map = {
        NUM_RES: num_res,
    }

    for k, v in protein.items():
        if k not in shape_schema:
            continue
        shape = list(v.shape)
        schema = shape_schema[k]
        msg = "Rank mismatch between shape and shape schema for"
        assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"
        pad_size = [
            pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
        ]

        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(itertools.chain(*padding))
        if padding:
            protein[k] = torch.nn.functional.pad(v, padding)
            protein[k] = torch.reshape(protein[k], pad_size)
    
    return protein
