import logging
import copy
import ml_collections as mlc
import torch

logger = logging.getLogger("TopoDiff.config.head_config")

def pred_head_model_config(
    name,
    extra = None,
):
    c = copy.deepcopy(pred_head_config)

    if name == 'base':
        pass
    elif name == 'candidate_1':
        pass
    else:
        raise NotImplementedError

    if extra is not None:
        if type(extra) == str:
            extra = [extra]
        print(f'extra: {extra}')
    else:
        extra = []
    
    if 'pred_sc' in extra:
        c.Data.common.sc.scale_factor = 0.2

    return c

pred_head_config = mlc.ConfigDict(
    {
        'Model': {
            'latent_dim':32,
            'ff_dim': 32,
            'c_out': 1,
            'dropout': 0.1,
            'n_layers': 5,
        },
        'Data': {
            'common': {
                'add_noise': {
                    'enabled': True,
                    'scale': 0.1,
                    'sigma': None,
                },
                'sc': {
                    'scale_factor': 1.,
                }
            },
        },
        'Loss': {
            'sc': {
                'loss_type': 1,
                'weight': 1.0,
            }
        }
    }
)