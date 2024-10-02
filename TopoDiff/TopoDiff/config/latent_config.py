import logging
import copy
import ml_collections as mlc
import torch

logger = logging.getLogger("TopoDiff.config.latent_config")

def latent_model_config(
    name,
    extra = None,
):
    c = copy.deepcopy(latent_config)
    
    if name == 'model_1':
        c.Model.MLPSkipNet.num_layers = 10
        c.Model.MLPSkipNet.skip_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    else:
        raise NotImplementedError

    if extra is not None:
        if type(extra) == str:
            extra = [extra]
        print(f'extra: {extra}')

    return c

latent_T = mlc.FieldReference(200, field_type=int)
latent_dim = mlc.FieldReference(32, field_type=int)
pred_type = mlc.FieldReference('eps', field_type=str)  # ['eps', 'x_0']
eps = mlc.FieldReference(1e-8, field_type=float)

latent_config = mlc.ConfigDict(
    {
        'Data': {
            'common': {
                'normalize': {
                    'enabled': True,
                    'mu': None,
                    'sigma': None,
                },
                'timestep':{
                    'T': latent_T,
                },
                'add_noise': {
                    'enabled': False,
                    'scale': 0.1,
                },
            },
        },
        'Global': {
            'eps': eps,
            'T': latent_T,
            'latent_dim': latent_dim,
            'pred_type': pred_type,
        },
        'Model': {
            'Global': {
                'Backbone': 'MLPSkipNet',
            },
            'MLPSkipNet': {
                'num_channels': latent_dim,
                'num_hid_channels': 128,
                'num_layers': None,
                'skip_layers': None,
                'skip_type': 0,  # 0 for input, 1 for previous layer

                'num_time_emb_channels': 64,
                'num_time_layers': 2,
                
                'activation': 'silu',
                'use_norm': True,
                'condition_bias': 1,
                'dropout': 0,
            },
            'Diffuser': {
                'T': latent_T,
                'beta_1': 0.005,
                'beta_T': 0.04,
                'schedule': 'linear',
                'reverse_type': 'eps-1',
                'pred_type': pred_type,
                'eps': eps,
            }
        },
        'Loss': {
            'recon': {
                'weight': 1,
                'type': 'l1',
                'pred_type': pred_type,
            },
            'cum_loss_scale': 1,
        }
    }
)