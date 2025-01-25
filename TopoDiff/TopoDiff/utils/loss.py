import logging
import ml_collections
import numpy as np
import torch
import torch.nn as nn
import random

from typing import Optional, Dict

from myopenfold.utils.rigid_utils import Rotation, Rigid
from myopenfold.np import residue_constants

from TopoDiff.utils.loss_schedulers import LossScheduler

logger = logging.getLogger("TopoDiff.utils.loss")

####################################################################### utils #######################################################################

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

####################################################################### regularization loss #######################################################################

def KL_loss(latent_mu, latent_logvar, latent_mask = None, eps=1e-6, **kwargs):
    """
    Computes KL loss for VAE autoencoder

    Args:
        latent_mu: [*, latent_dim]
            infered latent mean
        latent_logvar: [*, latent_dim]
            infered latent log variance (log(std))
        latent_mask: [*]
    Returns:
        KL loss 【*】
    """
    if latent_mask is not None:
        return 0.5 * torch.sum(latent_mu ** 2 + latent_logvar.exp() ** 2 - 2 * latent_logvar - 1, dim = -1) * latent_mask
    return 0.5 * torch.sum(latent_mu ** 2 + latent_logvar.exp() ** 2 - 2 * latent_logvar - 1, dim = -1)

####################################################################### distogram loss #######################################################################

def distogram_loss(
    logits,
    coord,
    coord_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    **kwargs,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2

    dists = torch.sum((coord[..., None, :] - coord[..., None, :, :]) ** 2, dim=-1, keepdims=True)

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    square_mask = coord_mask[..., None] * coord_mask[..., None, :]

    scale = 0.25
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))

    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)
    mean = mean * scale

    return mean

####################################################################### translation loss #######################################################################

def translation_loss(
        trans_pred: torch.Tensor,
        trans_gt: torch.Tensor,
        mask: torch.Tensor,
        length_scale = 10.0,
        eps = 1e-6,
        **kwargs,
):
    errors = torch.sum(((trans_pred - trans_gt) / length_scale)**2, dim = -1) * mask

    denom = eps + torch.sum(mask, dim = -1)

    errors = errors / denom[..., None]
    errors = torch.sum(errors, dim = -1)

    return errors
    

####################################################################### rotation loss #######################################################################

def rotation_loss(
        rot_pred: torch.Tensor,
        rot_gt: torch.Tensor,
        mask: torch.Tensor,
        eps = 1e-6,
        **kwargs,
):
    rot_rev = torch.matmul(rot_gt.transpose(-1, -2), rot_pred)

    rot_rev_diff = rot_rev - torch.eye(3, device = rot_rev.device)[None, None, :, :]

    errors = torch.sum(rot_rev_diff**2, dim = (-1, -2)) * mask

    denom = eps + torch.sum(mask, dim = -1)

    errors = errors / denom[..., None]
    errors = torch.sum(errors, dim = -1)

    return errors

####################################################################### fape loss #######################################################################

def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
) -> torch.Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :], 
    )

    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale

    normed_error = normed_error * frames_mask[..., None]

    normed_error = normed_error * positions_mask[..., None, :]

    normed_error = torch.sum(normed_error, dim=-1)

    normed_error = (
        normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )

    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def main_frame_fape_loss(
    frame_hat,
    frame_gt,
    frame_mask,
    coord_gt_mask,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    clamp_fape_ratio = None,
    **kwargs):
    """
    Computes the FAPE loss between the predicted and ground truth frames.

    Args:
        frame_hat: [N_sm, *, N_res]
        frame_gt: [1, *, N_res]
        frame_mask: [1, *, N_res]
            Mask for fixed frames or frames with no ground truth
        gt_mask: [1, *, N_res]
            Mask for frames that have no ground truth
    """
    fape_loss = compute_fape(
        frame_hat,
        frame_gt,
        frame_mask,
        frame_hat.get_trans(),
        frame_gt.get_trans(),
        coord_gt_mask,
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps)
    
    if clamp_fape_ratio is not None:
        unclamped_fape_loss = compute_fape(
            frame_hat,
            frame_gt,
            frame_mask,
            frame_hat.get_trans(),
            frame_gt.get_trans(),
            coord_gt_mask,
            l1_clamp_distance = None,
            length_scale = loss_unit_distance,
            eps = eps)
        fape_loss = fape_loss * clamp_fape_ratio + unclamped_fape_loss * (1 - clamp_fape_ratio)

    if fape_loss.ndim == 2:
        fape_loss = fape_loss.mean(dim = 0)
    
    return fape_loss


def backbone_fape_loss(
        frame_hat,
        frame_gt,
        coord_hat,
        coord_gt,
        frame_mask,
        coord_gt_mask,
        clamp_distance: float = 10.0,
        loss_unit_distance: float = 10.0,
        eps: float = 1e-4,
        clamp_fape_ratio = None,
        **kwargs):
    """
    Computes the FAPE loss between the predicted and ground truth frames on all backbone coordinates.

    Args:
        frame_hat: [*, N_res]
        frame_gt: [*, N_res]
        coord_hat: [*, N_res, 3]
        coord_gt: [*, N_res, 3]
        frame_diff_mask: [*, N_res]
            Mask for fixed frames or frames with no ground truth
        frame_gt_mask: [*, N_res]
            Mask for frames that have no ground truth
    """
    fape_loss = compute_fape(
        frame_hat,
        frame_gt,
        frame_mask,
        coord_hat,
        coord_gt,
        coord_gt_mask,
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps)

    if clamp_fape_ratio is not None:
        unclamped_fape_loss = compute_fape(
            frame_hat,
            frame_gt,
            frame_mask,
            coord_hat,
            coord_gt,
            coord_gt_mask,
            l1_clamp_distance = None,
            length_scale = loss_unit_distance,
            eps = eps)
        fape_loss = fape_loss * clamp_fape_ratio + unclamped_fape_loss * (1 - clamp_fape_ratio)

    if fape_loss.ndim == 2:
        fape_loss = fape_loss.mean(dim = 0)

    return fape_loss
    


####################################################################### Main class #######################################################################

class TopoDiffLoss(nn.Module):
    """Main class to compute all loss terms
    """
    def __init__(self, config, depth = 0, log = False):
        super(TopoDiffLoss, self).__init__()

        self.depth = depth
        self.log = log

        self.config = config

        self.length_scaling = config.length_scaling
        if not self.length_scaling.enabled:
            logger.info('Length scaling is disabled, please double-check the loss cum scaling is within the range')
        
        if self.config.kl_regularization.force_compute or \
            self.config.kl_regularization.schedule is not None or \
            self.config.kl_regularization.weight > 0 :
            self.compute_kl = True
            if self.config.kl_regularization.schedule is not None:
                print('KL loss weight config: ', self.config.kl_regularization.schedule)
            self.kl_weight_scheduler = LossScheduler(self.config.kl_regularization)
        else:
            self.compute_kl = False

        self.counter = 0

    
    def forward(self, result, feat, _return_breakdown=False, epoch = 0):
        self.counter += 1

        frame_hat_rigid = Rigid.from_tensor_7(result['frame_hat'])
        frame_hat_rigid = Rigid(
                Rotation(rot_mats=frame_hat_rigid.get_rots().get_rot_mats(), quats=None),
                frame_hat_rigid.get_trans())
        frame_gt_rigid = Rigid.from_tensor_4x4(feat['frame_gt'])

        # only compute loss on the frames that have ground truth and not fixed
        coord_mask = feat['frame_gt_mask'] * feat['frame_mask']

        loss_fns = {}
        if self.config.translation.weight > 0 or self.config.translation.force_compute:

            loss_fns['translation'] = (lambda: translation_loss(
                trans_pred = frame_hat_rigid.get_trans(),
                trans_gt = frame_gt_rigid.get_trans(),
                mask = coord_mask,  # compute loss only on the frames that have ground truth and not fixed
                **self.config.translation,),
                self.config.translation.weight > 0 or self.config.translation.force_compute,
                self.config.translation.weight > 0,
                self.config.translation.weight
            )

        if self.config.rotation.weight > 0 or self.config.rotation.force_compute:
            loss_fns['rotation'] = (lambda: rotation_loss(
                rot_pred = frame_hat_rigid.get_rots().get_rot_mats(),
                rot_gt = frame_gt_rigid.get_rots().get_rot_mats(),
                mask = coord_mask,  # compute loss only on the frames that have ground truth and not fixed
                **self.config.rotation),
                self.config.rotation.weight > 0 or self.config.rotation.force_compute,
                self.config.rotation.weight > 0,
                self.config.rotation.weight
            )

        if self.config.distogram.weight > 0 or self.config.distogram.force_compute:
            loss_fns['distogram'] = (lambda: distogram_loss(
                logits = result['distogram_logits'],
                coord = feat['contact_gt'],
                coord_mask = feat['contact_gt_mask'],
                **self.config.distogram),
                self.config.distogram.weight > 0 or self.config.distogram.force_compute,
                self.config.distogram.weight > 0,
                self.config.distogram.weight 
            )

        if self.config.fape.main_frame.weight > 0 or self.config.fape.main_frame.force_compute:
            if self.config.fape.main_frame.compute_for_all:
                cur_frame_hat = Rigid.from_tensor_7(result['sm_result']['frames'])
                cur_frame_hat = Rigid(
                                Rotation(rot_mats=cur_frame_hat.get_rots().get_rot_mats(), quats=None),
                                cur_frame_hat.get_trans())
                cur_frame_gt = frame_gt_rigid[None]
            else:
                cur_frame_hat = frame_hat_rigid
                cur_frame_gt = frame_gt_rigid

            loss_fns['fape_main'] = (lambda: main_frame_fape_loss(
                frame_hat = cur_frame_hat,
                frame_gt = cur_frame_gt,
                frame_mask = coord_mask,
                coord_gt_mask = feat['frame_gt_mask'],
                **self.config.fape),
                True,
                self.config.fape.main_frame.weight > 0,
                self.config.fape.main_frame.weight
            )

        if self.config.fape.backbone.weight > 0 or self.config.fape.backbone.force_compute:
            batch_shape = result['backbone_positions'].shape[:-3]
            loss_fns['fape_backbone'] = (lambda: backbone_fape_loss(
                frame_hat = frame_hat_rigid,
                frame_gt = frame_gt_rigid,
                coord_hat = result['backbone_positions'].reshape(batch_shape + (-1, 3)),
                coord_gt = feat['coord_gt'].reshape(batch_shape + (-1, 3)),
                frame_mask = coord_mask,
                coord_gt_mask = feat['coord_gt_mask'].reshape(batch_shape + (-1,)),
                **self.config.fape),
                True,
                self.config.fape.backbone.weight > 0,
                self.config.fape.backbone.weight
            )

        if self.compute_kl:
            loss_fns['kl_regularization'] = (lambda: KL_loss(
                latent_mu = result['latent_mu'],
                latent_logvar = result['latent_logvar'],
                latent_mask = feat['latent_mask'] if 'latent_mask' in feat else None,
                **self.config.kl_regularization),
                True,
                self.kl_weight_scheduler(epoch) > 0,
                self.kl_weight_scheduler(epoch),
            )

        cum_loss = 0.
        losses = {}
        for loss_name, (loss_fn, loss_compute, loss_grad, loss_weight) in loss_fns.items():
            if loss_compute:
                with torch.set_grad_enabled(loss_grad):
                    loss = loss_fn()
                    losses[loss_name] = loss.detach().clone().cpu().numpy()
                if loss_grad:
                    cum_loss = cum_loss + loss_weight * loss

        losses["unscaled_loss"] = cum_loss.detach().clone().cpu().numpy()

        if cum_loss.ndim == 0:
            cum_loss = cum_loss.unsqueeze(0)
        n_sample = cum_loss.shape[0]

        if self.length_scaling.enabled:
            crop_len = feat['seq_type'].shape[-1]
            seq_len = torch.where(feat['seq_length'].float() < crop_len, feat['seq_length'].float(), torch.Tensor([crop_len]).float().to(feat["seq_length"].device).expand(n_sample))

            cum_loss = cum_loss * torch.sqrt(seq_len) * self.config.cum_loss_scale
        else:
            cum_loss = cum_loss * self.config.cum_loss_scale

        losses['scaled_loss'] = cum_loss.detach().clone().cpu().numpy()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses