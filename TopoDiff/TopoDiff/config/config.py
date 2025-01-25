import logging
import copy
import ml_collections as mlc
import torch
import numpy as np

logger = logging.getLogger("TopoDiff.config.config")

def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf

def model_config(
    name, 
    low_prec=False,
    extra=None
):
    c = copy.deepcopy(config)

    if name == 'v1_1_2':
        """    
        Usable version. Default settings for model v1_1_2.
        """

        c.Model.Global.Embedder = 'Embedder_v2'
        c.Model.Global.Backbone = 'Backbone_v2'
        
        c.Model.Backbone_v2.reconstruct_CB = False

        c.Loss.translation.weight = 0.25 # default
        c.Loss.rotation.weight = 0.5  # default
        c.Loss.distogram.weight = 0.5
        c.Loss.fape.main_frame.weight = 0.
        c.Loss.fape.backbone.weight = 1.

        # NOTE topological conditioning (continuous)
        # NOTE data
        # topo_conditioning
        c.Data.common.topo_conditioning.enabled = True
        c.Data.common.topo_conditioning.type = 'continuous'

        # trim_end oper
        c.Data.common.trim_end = True

        # encoder_feat
        c.Data.common.encoder_feat.enabled = True
        c.Data.common.encoder_feat.add_noise.enabled = True

        # padding
        c.Data.common.feat.encoder_feats = [NUM_RES, None]
        c.Data.common.feat.encoder_coords = [NUM_RES, None]
        c.Data.common.feat.encoder_mask = [NUM_RES]
        c.Data.common.feat.encoder_adj_mat = [NUM_RES, NUM_RES]
        c.Data.common.feat.length_gt = []
        c.Data.common.feat.ss = [NUM_RES]
        c.Data.common.feat.ss_gt_mask = [NUM_RES]

        # type
        c.Data.common.feat_type.encoder_feats = 'torch.float32'
        c.Data.common.feat_type.encoder_coords = 'torch.float32'
        c.Data.common.feat_type.encoder_mask = 'torch.bool'
        c.Data.common.feat_type.encoder_adj_mat = 'torch.bool'
        c.Data.common.feat_type.length_gt = 'torch.float32'
        c.Data.common.feat_type.ss = 'torch.long'
        c.Data.common.feat_type.ss_gt_mask = 'torch.long'

        # NOTE model
        c.Model.Global.Encoder = 'Encoder_v1'

        # Embedder_v2
        c.Model.Embedder_v2.topo_embedder.enabled = True
        c.Model.Embedder_v2.topo_embedder.type = 'continuous_v2'

        # NOTE loss
        c.Loss.kl_regularization.weight = 1e-3

        # NOTE new for latent mask
        c.Data.common.topo_conditioning.continuous.mask_prob = 0.25
        c.Model.Embedder_v2.topo_embedder.embed_mask = True
        c.Data.common.feat.latent_mask = []
        c.Data.common.feat_type.latent_mask = 'torch.float32'

        c.Model.Diffuser.SO3.sampling_reverse_strategy = 'score_and_noise'
        c.Model.Diffuser.SO3.reverse_parm.score_and_noise.sampling_noise_scale = 0.
        c.Model.Diffuser.SO3.reverse_parm.score_and_noise.sampling_score_scale = 2.

    elif name == 'v1_1_1':
        """    
        Usable version. Default settings for model v1_1_1.
        """

        c.Model.Global.Embedder = 'Embedder_v2'
        c.Model.Global.Backbone = 'Backbone_v2'
        
        c.Model.Backbone_v2.reconstruct_CB = False

        c.Loss.translation.weight = 0.25 # default
        c.Loss.rotation.weight = 0.5  # default
        c.Loss.distogram.weight = 0.5
        c.Loss.fape.main_frame.weight = 0.
        c.Loss.fape.backbone.weight = 1.

        # NOTE topological conditioning (continuous)
        # NOTE data
        # topo_conditioning
        c.Data.common.topo_conditioning.enabled = True
        c.Data.common.topo_conditioning.type = 'continuous'

        # trim_end oper
        c.Data.common.trim_end = True

        # encoder_feat
        c.Data.common.encoder_feat.enabled = True
        c.Data.common.encoder_feat.add_noise.enabled = True

        # padding
        c.Data.common.feat.encoder_feats = [NUM_RES, None]
        c.Data.common.feat.encoder_coords = [NUM_RES, None]
        c.Data.common.feat.encoder_mask = [NUM_RES]
        c.Data.common.feat.encoder_adj_mat = [NUM_RES, NUM_RES]
        c.Data.common.feat.length_gt = []
        c.Data.common.feat.ss = [NUM_RES]
        c.Data.common.feat.ss_gt_mask = [NUM_RES]

        # type
        c.Data.common.feat_type.encoder_feats = 'torch.float32'
        c.Data.common.feat_type.encoder_coords = 'torch.float32'
        c.Data.common.feat_type.encoder_mask = 'torch.bool'
        c.Data.common.feat_type.encoder_adj_mat = 'torch.bool'
        c.Data.common.feat_type.length_gt = 'torch.float32'
        c.Data.common.feat_type.ss = 'torch.long'
        c.Data.common.feat_type.ss_gt_mask = 'torch.long'

        # NOTE model
        c.Model.Global.Encoder = 'Encoder_v1'

        # Embedder_v2
        c.Model.Embedder_v2.topo_embedder.enabled = True
        c.Model.Embedder_v2.topo_embedder.type = 'continuous'

        # NOTE loss
        c.Loss.kl_regularization.weight = 1e-4

        # NOTE new for latent mask
        c.Data.common.topo_conditioning.continuous.mask_prob = 0.25
        c.Model.Embedder_v2.topo_embedder.embed_mask = True
        c.Data.common.feat.latent_mask = []
        c.Data.common.feat_type.latent_mask = 'torch.float32'

        c.Model.Diffuser.SO3.sampling_reverse_strategy = 'score_and_noise'
        c.Model.Diffuser.SO3.reverse_parm.score_and_noise.sampling_noise_scale = 0.
        c.Model.Diffuser.SO3.reverse_parm.score_and_noise.sampling_score_scale = 3.

    elif name == 'train_stage_1':
        """    
        Training settings for stage 1.
        """
        # default for now
        # c.Model.Global.Embedder = 'Embedder_v2'
        # c.Model.Global.Backbone = 'Backbone_v2'

        # default for now
        # c.Loss.translation.weight = 0.25
        # c.Loss.rotation.weight = 0.5
        # c.Loss.distogram.weight = 0.5
        # c.Loss.fape.main_frame.weight = 0.
        # c.Loss.fape.backbone.weight = 1.

        c.Model.Diffuser.SO3.reverse_strategy = 'hat_and_noise'
        c.Model.Diffuser.SO3.sampling_reverse_strategy  = 'hat_and_noise'
        c.Model.Diffuser.SO3.hat_and_noise.noise_scale = (0, 5)

    elif name == 'train_stage_2':
        # default for now
        # c.Model.Global.Embedder = 'Embedder_v2'
        # c.Model.Global.Backbone = 'Backbone_v2'

        # default for now
        # c.Loss.translation.weight = 0.25
        # c.Loss.rotation.weight = 0.5
        # c.Loss.distogram.weight = 0.5
        # c.Loss.fape.main_frame.weight = 0.
        # c.Loss.fape.backbone.weight = 1.

        # default for now 
        c.Model.Diffuser.SO3.reverse_strategy = 'score_and_noise'
        c.Model.Diffuser.SO3.sampling_reverse_strategy = 'score_and_noise'
        c.Model.Diffuser.SO3.reverse_parm.score_and_noise.sampling_noise_scale = 0.
        c.Model.Diffuser.SO3.reverse_parm.score_and_noise.sampling_score_scale = 2.

    elif name == 'train_stage_3':
        # default for now
        # c.Model.Global.Embedder = 'Embedder_v2'
        # c.Model.Global.Backbone = 'Backbone_v2'

        # default for now
        # c.Loss.translation.weight = 0.25
        # c.Loss.rotation.weight = 0.5
        # c.Loss.distogram.weight = 0.5
        # c.Loss.fape.main_frame.weight = 0.
        # c.Loss.fape.backbone.weight = 1.

        # KL regularization schedule
        c.Loss.kl_regularization.weight = 0.
        c.Loss.kl_regularization.force_compute = True
        c.Loss.kl_regularization.schedule = mlc.ConfigDict(
            {
                'mode' : 'cold start',
                'cold_start_epoch' : 50,
                'warm_up_epoch' : 550,
                'weight_min' : 0,
                'weight_max' : 1e-3,
            }
        )

        # stop fixing too many residues
        c.Data.common.partial_fixed.fixed_prob = 0.3

        # NOTE topological conditioning (continuous)
        # NOTE data
        # topo_conditioning
        c.Data.common.topo_conditioning.enabled = True
        c.Data.common.topo_conditioning.type = 'continuous'

        # trim_end oper
        c.Data.common.trim_end = True

        # encoder_feat
        c.Data.common.encoder_feat.enabled = True
        c.Data.common.encoder_feat.add_noise.enabled = True

        # padding
        c.Data.common.feat.encoder_feats = [NUM_RES, None]
        c.Data.common.feat.encoder_coords = [NUM_RES, None]
        c.Data.common.feat.encoder_mask = [NUM_RES]
        c.Data.common.feat.encoder_adj_mat = [NUM_RES, NUM_RES]

        # type
        c.Data.common.feat_type.encoder_feats = 'torch.float32'
        c.Data.common.feat_type.encoder_coords = 'torch.float32'
        c.Data.common.feat_type.encoder_mask = 'torch.bool'
        c.Data.common.feat_type.encoder_adj_mat = 'torch.bool'

        # NOTE model
        c.Model.Global.Encoder = 'Encoder_v1'

        # Embedder_v2
        c.Model.Embedder_v2.topo_embedder.enabled = True
        c.Model.Embedder_v2.topo_embedder.type = 'continuous_v2'

        # NOTE new for latent mask
        c.Data.common.topo_conditioning.continuous.mask_prob = 0.25
        c.Model.Embedder_v2.topo_embedder.embed_mask = True
        c.Data.common.feat.latent_mask = []
        c.Data.common.feat_type.latent_mask = 'torch.float32'

        c.Model.Diffuser.SO3.sampling_reverse_strategy = 'score_and_noise'
        c.Model.Diffuser.SO3.reverse_parm.score_and_noise.sampling_noise_scale = 0.
        c.Model.Diffuser.SO3.reverse_parm.score_and_noise.sampling_score_scale = 2.


    else:
        raise ValueError(f'Unknown model name {name}')

    if extra is not None:
        if type(extra) == str:
            extra = [extra]

        if 'encoder_layer_ipa' in extra:
            c.Data.common.encoder_feat.enabled = True
            c.Data.common.encoder_feat.layer_type = 'ipa'

    if low_prec:
        logger.info('Using low precision training, setting eps to 1e-4')
        c.Global.eps = 1e-4
        set_inf(c, 1e4)

    return c

T = mlc.FieldReference(200, field_type=int)
c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(256, field_type=int)
c_s = mlc.FieldReference(256, field_type=int)
eps = mlc.FieldReference(1e-8, field_type=float)

encoder_layer_type = mlc.FieldReference('egnn', field_type=str)

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"

config = mlc.ConfigDict(
    {
        'Data': {
            "common": {
                "feat": {
                    "seq_type": [NUM_RES],
                    "seq_idx": [NUM_RES],
                    "seq_feat": [NUM_RES, None],
                    "seq_mask": [NUM_RES],
                    "seq_gt_mask": [NUM_RES],

                    "frame_gt": [NUM_RES, None, None],
                    "frame_gt_mask": [NUM_RES],
                    "frame_mask": [NUM_RES],

                    "contact_gt": [NUM_RES, None],
                    "contact_gt_mask": [NUM_RES],

                    'coord_gt': [NUM_RES, None, None],
                    'coord_gt_mask': [NUM_RES, None],

                    "seq_gt": [NUM_RES],
                    "fixed": [],
                    "fixed_ratio": [],
                    "fixed_continue": [],
                    
                    "resolution": [],
                    "seq_length": [],
                    "use_clamped_fape": [],
                    "timestep": [],
                },
                "feat_type" : {
                    "seq_type": 'torch.long',
                    "seq_idx": 'torch.long',
                    "seq_feat": 'torch.float32',
                    "seq_mask": 'torch.bool',

                    "frame_gt": 'torch.float32',
                    "frame_gt_mask": 'torch.bool',
                    "frame_mask": 'torch.bool',

                    "contact_gt": 'torch.float32',
                    "contact_gt_mask": 'torch.bool',

                    "chi_gt":'torch.float32',
                    'chi_mask': 'torch.bool',

                    'torsion_gt': 'torch.float32',
                    'torsion_gt_mask': 'torch.bool',

                    'coord_gt': 'torch.float32',
                    'coord_gt_mask': 'torch.bool',

                    'contact_gt': 'torch.float32',
                    'contact_gt_mask': 'torch.bool',

                    'timestep': 'torch.long',
                },
                'feat_rename': {
                    'aatype': 'seq_type',
                    'residue_index': 'seq_idx',
                    'target_feat': 'seq_feat',

                    'backbone_rigid_tensor': 'frame_gt',
                    'backbone_rigid_mask': 'frame_gt_mask',

                    "chi_angles_sin_cos": 'chi_gt',
                    "chi_mask": "chi_gt_mask"
                },
                "coord": {
                    "coord_type": "backbone",
                },
                "contact": {
                    "contact_type": "CA",
                },
                "partial_fixed":{
                    'enabled': True,
                    "fixed_prob": 0.6,
                    "fixed_ratio": (0.2, 0.8),
                    "continuous_prob": 0.5,
                },
                "masked_sequence": {
                    "enabled": True,
                },
                "coordinate_shift": {
                    'centered': True,
                    'shift_radius': 1.0,
                    'random_rotation': True,
                    'oper_on_coord14': True,
                },
                'timestep':{
                    'T': T,
                },
                "max_self_condition_step": 3,
                'structure_preprocessed': True,
                'trim_end': False, 
                'topo_conditioning': {
                    'enabled': False,
                    'type': 'category',
                    'category': {
                        'num_class': None,
                        'mask_prob': None,
                    },
                    'continuous': {
                        'mask_prob': None,
                    }
                },
                'encoder_feat': {
                    'enabled': False,
                    'add_noise': {
                        'enabled': False,
                        'type': 'gaussian',
                        'std': 1,
                    },
                    'layer_type': encoder_layer_type, #  'egnn', 'ipa'
                    'frame_type': '4x4',  # ['4x4', '7']
                },
                "pad_in_collator": {
                    "enabled" : True,
                },
                "unsupervised_features": [
                    "aatype",
                    "residue_index",
                    "msa",
                    "num_alignments",
                    "seq_length",
                    "between_segment_residues",
                    "deletion_matrix",
                    "no_recycling_iters",
                ],
            },
            "train": {
                    "crop": True,
                    "crop_size": 256,
            },
        },
        'Global': {
            'T' : T,
            "blocks_per_ckpt": None,
            "c_z": c_z,
            "c_m": c_m,
            "c_s": c_s,
            "eps": eps,
        },
        'Model':{
            'Global': {
                'infer_no_recyc': False,
                'Embedder': 'Embedder_v2',
                'Backbone': 'Backbone_v2',
                'Diffuser': 'Diffuser_v1',
                'Encoder' : None,
            },
            'Encoder_v1': {
                'feature_dim': 68,
                'hidden_dim': 128,
                'dropout': 0,
                'dropout_final': 0,
                'eps': eps,

                'n_layers': 6,
                'm_dim': 16,

                'layer_type': encoder_layer_type, # "egnn", "ipa"

                'hidden_egnn_dim': 64,
                'hidden_edge_dim': 256,

                'ipa':{
                    'trans_scale_factor': 10,
                    'c_hidden': 32,
                    'c_s': 128,
                    'c_z': 64,
                    'no_heads': 4,
                    'no_qk_points': 4,
                    'no_v_points': 6,
                    'inf': 1e5,
                    'eps': eps,
                },

                'embedding_size': 64,
                'latent_dim': 32,

                'normalize': False,
                'final_init': True,
                'reduce': 'sum',

                'transformer': {
                    'enable': True,
                    'version': 2,
                    'n_heads': 4,
                    'n_layers': 2,
                    'dropout': 0,
                },

                'trainable': True,
                'eps': eps,
                'temperature': 1.,
            },
            'Embedder_v2': {
                'c_s': c_s,
                'c_z': c_z,

                'tf_dim': 22,
                'pos_emb_dim': 32,
                'time_emb_dim': 32,
                'embed_fixed': True,

                'time_emb_max_positions': 10000,
                'pos_emb_max_positions': 2056,

                'recyc_struct': True,
                'recyc_struct_min_bin': 3.25,
                'recyc_struct_max_bin': 20.75,
                'recyc_struct_no_bin': 15,

                'eps': eps,
                'inf': 1e5,

                'topo_embedder': {
                    'enabled': False,
                    'type': 'continuous',
                    'embed_dim': 32,
                    'num_class': None,
                    'embed_mask': True,
                },

            },
            'Backbone_v2': {
                'c_s': c_s,
                'c_z': c_z,
                'c_skip': 64,
                'no_blocks': 4,
                'seq_tfmr_num_heads': 4,
                'seq_tfmr_num_layers': 2,

                'no_seq_transition_layers': 1,
                'seq_transition_dropout_rate': 0.1,

                'predict_torsion': True,
                'angle_c_resnet': 128,
                'angle_no_resnet_blocks': 2,
                'no_angles': 1,
                'reconstruct_backbone_atoms': True,
                'reconstruct_CB': False,
                'torsion_index': 0,

                'trans_scale_factor': 10,

                'epsilon': eps,
                'inf': 1e5,

                'ipa': {
                    'c_s': c_s,
                    'c_z': c_z,
                    'c_hidden': 256,
                    'no_heads': 8,
                    'no_qk_points': 8,
                    'no_v_points': 12,
                    'inf': 1e5,
                    'eps': eps,
                },

            },
            'Aux_head': {
                'distogram': {
                    'c_z': c_z,
                    "no_bins": 64,
                },
            },
            'Diffuser': {
                'Global':{
                    'T': T,
                    'trans_scale_factor': 4.,
                },
                'Cartesian': {
                    'alpha_1' : 0.99,
                    'alpha_T' : 0.93,
                    'T' : T,
                    'reverse_strategy': 'DDPM',
                },
                'SO3': {
                    'cache_dir': '/home/zhangyuy/workspace/dl/TopoDiff/submit_240923/TopoDiff/cache/',
                    'suffix': '_log',
                    'schedule': 'linear',
                    'sigma_1' : 0.1 * np.sqrt(2),
                    'sigma_T' : 1.5 * np.sqrt(2),
                    'reverse_strategy': 'score_and_noise',
                    'hat_and_noise':{
                        'noise_scale': (0, 5),
                        'noise_scale_schedule': 'linear',
                    },
                    'T' : T,
                    'reverse_parm':{
                        'hat_and_noise_2':{
                            'noise_scale': (1, 1),
                            'noise_scale_schedule': 'linear',
                        },
                        'score_and_noise':{
                            'noise_scale': 1.,
                            'score_scale': 1.,
                            'eps': 1e-8,
                            'Log_type': 2,
                        },
                    },
                }
            },
        },
        'Loss':{
            'length_scaling': {
                'enabled': True,
            },
            "distogram": {
                "min_bin": 2.3125,
                "max_bin": 21.6875,
                "no_bins": 64,
                "eps": eps,  # 1e-6,
                "weight": 0.5,
                "force_compute": False,
            },
            "fape": {
                "clamp_distance": 10.0,
                "loss_unit_distance": 10.0,
                "clamp_fape_ratio": 0.9,
                # "weight": 0.,
                "eps": 1e-4,
                "main_frame":{
                    "weight": 0.,
                    "force_compute": False,
                    "compute_for_all": True,
                },
                "backbone":{
                    "weight": 1.,
                    "force_compute": False,
                }
            },
            "translation": {
                "weight": 0.25,
                "eps": eps,
                "force_compute": True,
            },
            "rotation": {
                "weight": 0.5,
                "eps": eps,
                "force_compute": True,
            },
            "eps": eps,
            "cum_loss_scale": 1.0,
            "kl_regularization": {
                'weight': 0.,
                'eps': eps,
                'force_compute': False,
                'schedule': None,
            },
        },
    }
)