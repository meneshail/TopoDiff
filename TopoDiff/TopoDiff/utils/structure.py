import torch

import myopenfold.utils.rigid_utils as ru

def backbone_torsion_OF_to_custom(torsion, torsion_mask = None):
    '''Converts OpenFold torsion angles to custom torsion angles.

    Args:
        torsion: [..., 3, 2]
            In the order of Omega_{i-1}, Phi_i, Psi_i
        torsion_mask: [..., 3]

    Returns:
        torsion_custom: [..., 3, 2]
            In the order of Phi_i, Psi_i, Omega_i
        torsion_custom_mask: [..., 3]
    '''
    omega = torch.nn.functional.pad(torsion[..., 1:, 0, :], (0, 0, 0, 1))  # pad for Omega-(-1)
    phi = torsion[..., 1, :]
    psi = torsion[..., 2, :]
    torsion_custom = torch.stack([phi, psi, omega], dim = -2)

    if torsion_mask is not None:
        omega_mask = torch.nn.functional.pad(torsion_mask[..., 1:, 0], (0, 1))
        torsion_custom_mask = torch.stack([torsion_mask[..., 1], torsion_mask[..., 2], omega_mask], dim = -1)
        return torsion_custom, torsion_custom_mask

    return torsion_custom


def backbone_torsion_custom_to_OF(torsion, torsion_mask = None):
    '''Converts custom torsion angles to OpenFold torsion angles.

    Input:
        torsion: [..., 3, 2]
            In the order of Phi_i, Psi_i, Omega_i
        torsion_mask: [..., 3]

    Returns:
        torsion_OF: [..., 3, 2]
            In the order of Omega_{i-1}, Phi_i, Psi_i
        torsion_OF_mask: [..., 3]
    '''
    omega = torch.nn.functional.pad(torsion[..., :-1, 2, :], (0, 0, 1, 0))  # pad for Omega-(-1)
    phi = torsion[..., 0, :]
    psi = torsion[..., 1, :]
    torsion_OF = torch.stack([omega, phi, psi], dim = -2)

    if torsion_mask is not None:
        omega_mask = torch.nn.functional.pad(torsion_mask[..., :-1, 2], (1, 0))
        torsion_OF_mask = torch.stack([omega_mask, torsion_mask[..., 0], torsion_mask[..., 1]], dim = -1)
        return torsion_OF, torsion_OF_mask
    
    return torsion_OF

def _normalize(tensor, dim=-1):
    '''Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def backbone_coord_to_torsion(coord, coord_mask = None, eps = 1e-8, return_sin_cos = False):
    """Get backbone torsion angles from backbone coordinates. (In canonical order)

    Args:
        coord: [..., N, 3, 3]
        coord_mask: [..., N, 3]
    
    Returns:
        torsion: [..., N, 3] or [..., N, 3, 2]
            In the order of Phi_i, Psi_i, Omega_i
        torsion_mask: [..., N, 3]
    """
    batch_shape = coord.shape[:-3]
    coord = coord.reshape(batch_shape + (-1, 3))

    if coord_mask is None:
        coord_mask = torch.ones(coord.shape[:-1], dtype = torch.bool, device = coord.device)
    else:
        coord_mask = coord_mask.reshape(batch_shape + (-1,))

    coord_backbone_diff = _normalize(coord[..., 1:, :] - coord[..., :-1, :], dim = -1)
    coord_backbone_diff_mask = coord_mask[..., 1:] * coord_mask[..., :-1]

    #. compute direction vector
    unit_2 = coord_backbone_diff[..., :-2, :]  # N0->CA0, CA0->C0, C0->N1...
    unit_1 = coord_backbone_diff[..., 1:-1, :]  # CA0->C0, C0->N1, N1->CA1...
    unit_0 = coord_backbone_diff[..., 2:, :]  # C0->N1, N1->CA1, CA1->C1...

    unit_2_mask = coord_backbone_diff_mask[..., :-2]  # N0->CA0, CA0->C0, C0->N1...
    unit_1_mask = coord_backbone_diff_mask[..., 1:-1]  # CA0->C0, C0->N1, N1->CA1...
    unit_0_mask = coord_backbone_diff_mask[..., 2:]  # C0->N1, N1->CA1, CA1->C1...

    #. compute normal vector through cross product
    normal_2 = _normalize(torch.cross(unit_2, unit_1), dim = -1)  # N0->CA0 x CA0->C0, CA0->C0 x C0->N1, C0->N1 x N1->CA1...
    normal_1 = _normalize(torch.cross(unit_1, unit_0), dim = -1)  # CA0->C0 x C0->N1, C0->N1 x N1->CA1, N1->CA1 x CA1->C1...

    normal_2_mask = unit_2_mask * unit_1_mask  # N0->CA0 x CA0->C0, CA0->C0 x C0->N1, C0->N1 x N1->CA1...
    normal_1_mask = unit_1_mask * unit_0_mask  # CA0->C0 x C0->N1, C0->N1 x N1->CA1, N1->CA1 x CA1->C1...

    #. compute torsion angle through dot product
    cos_D = torch.sum(normal_2 * normal_1, dim = -1)  # N0->CA0->C0->N1 (Psi-0), CA0->C0->N1->CA1 (Omega-0), C0->N1->CA1->C1 (Phi-1)...
    cos_D = torch.clamp(cos_D, -1 + eps, 1 - eps)
    D = torch.acos(cos_D) * torch.sign(torch.sum(unit_2 * normal_1, dim = -1))

    D_mask = normal_2_mask * normal_1_mask

    #. pad
    D = torch.nn.functional.pad(D, (1,2)) # pad for Phi-0, Psi-(-1), Omega-(-1)

    D = D.reshape(batch_shape + (-1, 3))  # [N, 3], [Phi, Psi, Omega]

    D_mask = torch.nn.functional.pad(D_mask, (1,2)) # pad for Phi-0, Psi-(-1), Omega-(-1)
    D_mask = D_mask.reshape(batch_shape + (-1, 3))  # [N, 3], [Phi, Psi, Omega]

    if return_sin_cos:
        D_sin_cos = torch.stack([torch.sin(D), torch.cos(D)], dim = -1)
        return D_sin_cos, D_mask

    return D, D_mask


def reconstruct_backbone_position(restype_backbone_rigid_group_default_frame, restype_backbone_rigid_group_positions, 
                                  frame_pred, torsion_sin_cos, torsion_index = 0, seq_type = None,
                                  include_CB = False):
    """
    Args:
        restype_rigid_group_default_frame [21, 4, 4]
            The default Psi frame of each residue type.
        restype_backbone_rigid_group_positions [21, 4, 3]
            The default backbone positions of each residue type.
        frame_pred (Rigid): [..., N_res]
            The predicted backbone rigid transformations of each residues.
        torsion_sin_cos (Tensor): [..., N_res, N_torsion, 2]
            The torsion angles of each residues.
        torsion_index (int)
            The index of torsion angles to be used for reconstruction. (Psi)
        seq_type (Tensor): [..., N_res]
            The sequence type of each residues.
    """
    device = frame_pred.device
    batch_shape = frame_pred.shape  # torsion_sin_cos.shape[:-2]
    if seq_type is None:
        seq_type = torch.ones(batch_shape, dtype = torch.long, device = device) * 20

    #. [..., N_res]
    psi_frames_per_res = ru.Rigid.from_tensor_4x4(restype_backbone_rigid_group_default_frame[seq_type])

    #. [..., N_res, 2]
    psi_torsion = torsion_sin_cos[..., torsion_index, :]

    #. [..., N_res]
    rots = torch.zeros(psi_frames_per_res.get_rots().get_rot_mats().shape, device = device)
    rots[..., 0, 0] = 1
    rots[..., 1, 1] = psi_torsion[..., 1]
    rots[..., 1, 2] = -psi_torsion[..., 0]
    rots[..., 2, 1:] = psi_torsion

    rots_rigid = ru.Rigid(ru.Rotation(rot_mats=rots), None)

    #. [..., N_res]
    psi_frames_to_bb = psi_frames_per_res.compose(rots_rigid)

    #. [..., N_res]
    psi_frames_to_global = frame_pred.compose(psi_frames_to_bb)

    #. [..., N_res, 4, 3]
    lit_pos_per_res = restype_backbone_rigid_group_positions[seq_type]

    if not include_CB:
        #. [..., N_res, 4, 3]
        coord_recon = torsion_sin_cos.new_zeros(batch_shape + (4, 3))

        coord_recon[..., :3, :] = frame_pred[..., None].apply(lit_pos_per_res[..., :3, :])
        coord_recon[..., 3, :] = psi_frames_to_global.apply(lit_pos_per_res[..., 3, :])
    else:
        #. [..., N_res, 5, 3]
        coord_recon = torsion_sin_cos.new_zeros(batch_shape + (5, 3))

        #. [..., N_res, 4, 3]
        coord_recon[..., [0, 1, 2, 4], :] = frame_pred[..., None].apply(lit_pos_per_res[..., [0, 1, 2, 4], :])
        coord_recon[..., 3, :] = psi_frames_to_global.apply(lit_pos_per_res[..., 3, :])

    return coord_recon

def reconstruct_backbone_position_without_torsion(restype_backbone_rigid_group_default_frame, restype_backbone_rigid_group_positions,
                                                  frame_pred, seq_type = None):
    """Reconstruct backbone + oxygen position from predicted rigid transformation. 
    The Psi torsion angle is deduced from the consecutive backbone.

    Args:
        restype_rigid_group_default_frame [21, 4, 4]
            The default Psi frame of each residue type.
        restype_backbone_rigid_group_positions [21, 4, 3]
            The default backbone positions of each residue type.
        frame_pred (Rigid): [..., N_res]
            The predicted backbone rigid transformations of each residues.
        seq_type (Tensor): [..., N_res]
            The sequence type of each residues.

    """
    float_type = frame_pred._trans.dtype
    device = frame_pred.device
    batch_shape = frame_pred.shape
    if seq_type is None:
        seq_type = torch.ones(batch_shape, dtype = torch.long, device = device) * 20

    #. [..., N_res, 4, 3]
    lit_pos_per_res = restype_backbone_rigid_group_positions[seq_type]
    
    #. [..., N_res, 4, 3]
    coord_recon = torch.zeros(batch_shape + (4, 3), device = device, dtype=float_type)

    coord_recon[..., :3, :] = frame_pred[..., None].apply(lit_pos_per_res[..., :3, :])

    #. [..., N_res, 3, 2]
    backbone_torsion_sin_cos, backbone_torsion_mask = backbone_coord_to_torsion(coord_recon[..., :3, :], return_sin_cos=True)

    #. [..., N_res]
    psi_frames_per_res = ru.Rigid.from_tensor_4x4(restype_backbone_rigid_group_default_frame[seq_type])

    #. [..., N_res, 2]
    psi_torsion = backbone_torsion_sin_cos[..., 1, :]

    #. [..., N_res]
    rots = torch.zeros(psi_frames_per_res.get_rots().get_rot_mats().shape, device = device, dtype = float_type)
    rots[..., 0, 0] = 1
    rots[..., 1, 1] = psi_torsion[..., 1]
    rots[..., 1, 2] = -psi_torsion[..., 0]
    rots[..., 2, 1:] = psi_torsion

    rots_rigid = ru.Rigid(ru.Rotation(rot_mats=rots), None)

    #. [..., N_res]
    psi_frames_to_bb = psi_frames_per_res.compose(rots_rigid)

    #. [..., N_res]
    psi_frames_to_global = frame_pred.compose(psi_frames_to_bb)

    #. [..., N_res, 4, 3]
    coord_recon[..., 3, :] = psi_frames_to_global.apply(lit_pos_per_res[..., 3, :])

    return coord_recon


def reconstruct_backbone_position_2(restype_backbone_rigid_group_default_frame, restype_backbone_rigid_group_positions, 
                                  frame_pred, torsion_sin_cos, torsion_index = 0, seq_type = None):
    """
    Args:
        restype_rigid_group_default_frame [21, 4, 4]
            The default Psi frame of each residue type.
        restype_backbone_rigid_group_positions [21, 4, 3]
            The default backbone positions of each residue type.
        frame_pred (Rigid): [..., N_res]
            The predicted backbone rigid transformations of each residues.
        torsion_sin_cos (Tensor): [..., N_res, N_torsion, 2]
            The torsion angles of each residues.
        torsion_index (int)
            The index of torsion angles to be used for reconstruction. (Psi)
        seq_type (Tensor): [..., N_res]
            The sequence type of each residues.
    """
    batch_shape = torsion_sin_cos.shape[:-2]
    if seq_type is None:
        seq_type = torsion_sin_cos.new_ones(batch_shape, dtype = torch.long) * 20

    #. [..., N_res]
    psi_frames_per_res = ru.Rigid.from_tensor_4x4(restype_backbone_rigid_group_default_frame[seq_type])

    #. [..., N_res, 2]
    psi_torsion = torsion_sin_cos[..., torsion_index, :]

    #. [..., N_res]
    rots = psi_torsion.new_zeros(psi_frames_per_res.get_rots().get_rot_mats().shape)
    rots[..., 0, 0] = 1
    rots[..., 1, 1] = psi_torsion[..., 1]
    rots[..., 1, 2] = -psi_torsion[..., 0]
    rots[..., 2, 1:] = psi_torsion

    rots_rigid = ru.Rigid(ru.Rotation(rot_mats=rots), None)

    #. [..., N_res]
    psi_frames_to_bb = psi_frames_per_res.compose(rots_rigid)

    #. [..., N_res]
    psi_frames_to_global = frame_pred.compose(psi_frames_to_bb)

    #. [..., N_res, 2]
    all_frames_to_global = ru.Rigid.cat([frame_pred[..., None], psi_frames_to_global[..., None]], dim = -1)

    #. [..., N_res, 4, 3]
    lit_pos_per_res = restype_backbone_rigid_group_positions[seq_type]

    #. [..., 1, 4, 2]
    frame_one_hot = torch.nn.functional.one_hot(torch.tensor([0, 0, 0, 1])[(None,) * len(batch_shape)], num_classes = 2)

    #. [..., N_res, 4, 2] -> [..., N_res, 4]
    frame_per_res = all_frames_to_global[..., None, :] * frame_one_hot
    frame_per_res = frame_per_res.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    #. [..., N_res, 4, 3]
    coord_recon = frame_per_res.apply(lit_pos_per_res)

    return coord_recon