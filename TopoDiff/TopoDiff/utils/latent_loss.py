import logging
import ml_collections
import numpy as np
import torch
import torch.nn as nn

from typing import Optional, Dict

from TopoDiff.utils import debug

logger = logging.getLogger("TopoDiff.utils.latent_loss")

####################################################################### recon loss #######################################################################

def recon_loss(
    truth: torch.Tensor,
    pred: torch.Tensor,
    loss_type: str,
):
    """
    Compute prediction loss

    Args:
        truth (torch.Tensor): [*, latent_dim]
            The ground truth
        pred (torch.Tensor): [*, latent_dim]
            The prediction
        loss_type (str): ['l1', 'l2']
    """

    if loss_type == 'l1':
        loss = torch.mean(torch.abs(truth - pred), dim = -1)
    elif loss_type == 'l2':
        loss = torch.mean((truth - pred) ** 2, dim = -1)
    else:
        raise NotImplementedError
    
    return loss


####################################################################### Main CLass #######################################################################

class LatentLoss(nn.Module):
    """Main class to compute all loss terms
    """
    def __init__(self, config, depth = 0, log = False, log_parm = False):
        super(LatentLoss, self).__init__()

        self.depth = depth
        self.log = log

        self.config = config

        if log_parm:
            for key in self.config.keys():
                logger.info(f"Loss config: {key} = \n{self.config[key]}")

    
    def forward(self, feat, result, _return_breakdown=False):
        """
        Args:
            feat (dict): 
                latent_gt: [*, latent_dim]
                    The ground truth latent
                time_step: [*,]

            result (dict):
                latent_noised: [*, latent_dim]
                    The noised latent
                latent_eps: [*, latent_dim]
                    The noise
                pred: [*, latent_dim]
                    The prediction
                latent_gt: [*, latent_dim]
                    The ground truth latent (normalized)
        """

        loss_fns = {}

        # compute recon loss
        if self.config.recon.pred_type == 'eps':
            loss_fns['recon'] = (lambda: recon_loss(
                truth = result['latent_eps'],
                pred = result['pred'],
                loss_type = self.config.recon.type,
                ),
                self.config.recon.weight,
            )
        elif self.config.recon.pred_type == 'x_0':
            loss_fns['recon'] = (lambda: recon_loss(
                truth = result['latent_gt'],
                pred = result['pred'],
                loss_type = self.config.recon.type,
                ),
                self.config.recon.weight,
            )
        else:
            raise NotImplementedError

        cum_loss = 0.
        losses = {}
        for loss_name, (loss_fn, loss_weight) in loss_fns.items():
            loss = loss_fn()
            losses[loss_name] = loss.detach().clone().cpu().numpy()

            cum_loss += loss * loss_weight

        losses["total_loss"] = cum_loss.detach().clone().cpu().numpy()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses
    
        
