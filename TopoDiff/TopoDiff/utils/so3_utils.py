import numpy as np

import torch


# compute logarithm of rotation matrix
def so3_log(R, eps = 1e-8):
    """(torch ver.) log map of SO(3) -> skew-symmetric matrix

    Args:
        R: [*, 3, 3]

    Returns:
        skew-symmetric matrix: [*, 3, 3]
    """
    # gradient of acos(1) is inf, may need special treatment
    #. [..., 1, 1]
    # theta = np.arccos((np.trace(R) - 1) / 2)
    # add clamp to avoid nan
    theta = torch.acos((torch.einsum('...ii->...', R) - 1).clamp(-2, 2) / 2)[..., None, None]
    
    #. [..., 3, 3]
    R_skew = theta / (2 * torch.sin(theta + eps)) * (R - R.swapaxes(-1, -2))
    
    return R_skew


# compute logarithm of rotation matrix
def so3_Log(R, eps = 1e-8, type = 1):
    """(torch ver.) log map of SO(3) -> skew-symmetric matrix

    Args:
        R: [*, 3, 3]

    Returns:
        skew-symmetric matrix: [*, 3, 3]
    """
    if type == 1:
        R_skew = so3_log(R, eps)

        #. [..., 3]
        rot_vec = skew_to_vec(R_skew)
    elif type == 2:
        # another way to compute log map, skip error propagation from theta to R_skew
        theta = torch.acos((torch.einsum('...ii->...', R) - 1).clamp(-2, 2) / 2)

        rot_vec = skew_to_vec(R - R.swapaxes(-1, -2))
        rot_vec = rot_vec / torch.linalg.norm(rot_vec, dim=-1, keepdim=True) * theta[..., None]
    
    return rot_vec
    

def so3_log_decompose(R, eps=1e-8):
    """(torch ver.) log map of SO(3) -> skew-symmetric matrix (unit length) and theta

    Args:
        R: [*, 3, 3]

    Returns:
        skew-symmetric matrix: [*, 3, 3]
        theta: [*,]
    """
    #. [...,]
    # add clamp to avoid nan
    theta = torch.acos((torch.einsum('...ii->...', R) - 1).clamp(-2, 2) / 2)

    #. [..., 3, 3]
    R_skew = 1 / (2 * torch.sin(theta[..., None, None] + eps)) * (R - R.swapaxes(-1, -2))

    return R_skew, theta


def so3_Log_decompose(R, eps=1e-8, type = 1):
    """(torch ver.) Log map of SO(3) -> rotation vector (unit length) and theta

    Args:
        R: [*, 3, 3]

    Returns:
        skew-symmetric matrix: [*, 3, 3]
        theta: [*,]
    """
    if type == 1:
        R_skew, theta = so3_log_decompose(R, eps)

        #. [..., 3]
        rot_vec = skew_to_vec(R_skew)
    elif type == 2:
        # another way to compute log map, skip error propagation from theta to R_skew
        theta = torch.acos((torch.einsum('...ii->...', R) - 1).clamp(-2, 2) / 2)

        rot_vec = skew_to_vec(R - R.swapaxes(-1, -2))
        rot_vec = rot_vec / (eps + torch.linalg.norm(rot_vec, dim=-1, keepdim=True))

    return rot_vec, theta


def so3_Exp(w, eps = 1e-8):
    """(torch ver.) Exp map of rotation vector to SO(3) matrix (Remember to disable autocast or increase the eps when using float16)

    Args:
        w: [*, 3]
    
    Returns:
        SO(3) matrix: [*, 3, 3]
    """
    # print(w.device)
    #. [..., 3, 3]
    w_hat = vec_to_skew(w)

    #. [..., 1, 1]
    theta = torch.linalg.norm(w, dim=-1)[..., None, None]
    
    #. [..., 3, 3]
    w_hat_2 = w_hat @ w_hat

    # np.eye(3) + np.sin(theta) / theta * w_hat + (1 - np.cos(theta)) / (theta ** 2) * w_hat @ w_hat
    return torch.eye(3, device=w.device) + torch.sin(theta) / (theta + eps) * w_hat + (1 - torch.cos(theta)) / (theta ** 2 + eps) * w_hat_2


def so3_exp(w_hat, eps = 1e-8):
    """(torch ver.) exp map of skew matrix to SO(3) matrix (Remember to disable autocast or increase the eps when using float16)

    Args:
        w_hat: [*, 3, 3]
    
    Returns:
        SO(3) matrix: [*, 3, 3]
    """
    #. [..., 3]
    w = skew_to_vec(w_hat)

    #. [..., 1, 1]
    theta = torch.linalg.norm(w, dim=-1)[..., None, None]

    #. [..., 3, 3]
    w_hat_2 = w_hat @ w_hat

    return torch.eye(3, device=w_hat.device) + torch.sin(theta) / (theta + eps) * w_hat + (1 - torch.cos(theta)) / (theta ** 2 + eps) * w_hat_2
    

def vec_to_skew(w):
    """(torch ver.) rotation vector to so(3) skew-symmetric matrix

    Args:
        w: [*, 3]

    Returns:
        skew-symmetric matrix: [*, 3, 3]
    """
    #. [...,]
    x, y, z = w[..., 0], w[..., 1], w[..., 2]

    #. [...,]
    o = torch.zeros_like(x, device=w.device)

    #. [..., 3, 3]
    # return np.array([[0, -w[..., 2], w[..., 1]], [w[..., 2], 0, -w[..., 0]], [-w[..., 1], w[..., 0], 0]])
    return torch.stack([torch.stack([o, -z, y], dim=-1), torch.stack([z, o, -x], dim=-1), torch.stack([-y, x, o], dim=-1)], dim=-2)


def skew_to_vec(w_hat):
    """(torch ver.) so(3) skew-symmetric matrix to rotation vector

    Args:
        w_hat: [*, 3, 3]

    Returns:
        rotation vector: [*, 3]
    """
    return torch.stack([w_hat[..., 2, 1], w_hat[..., 0, 2], w_hat[..., 1, 0]], dim=-1)

######################################## same as above, but with numpy ########################################
# compute logarithm of rotation matrix
def so3_log_np(R, eps = 1e-8):
    """(numpy ver.) log map of SO(3) -> skew-symmetric matrix

    Args:
        R: [*, 3, 3]

    Returns:
        skew-symmetric matrix: [*, 3, 3]
    """
    # NOTE gradient of acos(1) is inf, may need special treatment
    #. [..., 1, 1]
    theta = np.arccos((np.einsum('...ii->...', R) - 1) / 2)[..., None, None]
    
    #. [..., 3, 3]
    R_skew = theta / (2 * np.sin(theta + eps)) * (R - R.swapaxes(-1, -2))
    
    return R_skew


def so3_Log_np(R, eps = 1e-8):
    """(numpy ver.) log map of SO(3) -> skew-symmetric matrix

    Args:
        R: [*, 3, 3]

    Returns:
        skew-symmetric matrix: [*, 3, 3]
    """
    R_skew = so3_log_np(R, eps)

    #. [..., 3]
    rot_vec = skew_to_vec_np(R_skew)
    
    return rot_vec


def so3_log_decompose_np(R, eps=1e-8):
    """(numpy ver.) log map of SO(3) -> skew-symmetric matrix (unit length) and theta

    Args:
        R: [*, 3, 3]

    Returns:
        skew-symmetric matrix: [*, 3, 3]
        theta: [*,]
    """
    #. [...,]
    theta = np.arccos((np.einsum('...ii->...', R) - 1) / 2)

    #. [..., 3, 3]
    R_skew = 1 / (2 * np.sin(theta[..., None, None] + eps)) * (R - R.swapaxes(-1, -2))

    return R_skew, theta


def so3_Log_decompose_np(R, eps=1e-8):
    """(numpy ver.) log map of SO(3) -> skew-symmetric matrix (unit length) and theta

    Args:
        R: [*, 3, 3]

    Returns:
        skew-symmetric matrix: [*, 3, 3]
        theta: [*,]
    """
    R_skew, theta = so3_log_decompose_np(R, eps)

    #. [..., 3]
    rot_vec = skew_to_vec_np(R_skew)
    
    return rot_vec, theta


def so3_Exp_np(w, eps = 1e-8):
    """(numpy ver.) Exp map of rotation vector to SO(3) matrix (Remember to disable autocast or increase the eps when using float16)

    Args:
        w: [*, 3]
    
    Returns:
        SO(3) matrix: [*, 3, 3]
    """
    #. [..., 3, 3]
    w_hat = vec_to_skew_np(w)

    #. [..., 1, 1]
    theta = np.linalg.norm(w, axis=-1)[..., None, None]
    
    #. [..., 3, 3]
    w_hat_2 = w_hat @ w_hat

    # np.eye(3) + np.sin(theta) / theta * w_hat + (1 - np.cos(theta)) / (theta ** 2) * w_hat @ w_hat
    return np.eye(3) + np.sin(theta) / (theta + eps) * w_hat + (1 - np.cos(theta)) / (theta ** 2 + eps) * w_hat_2


def so3_exp_np(w_hat, eps = 1e-8):
    """(numpy ver.) exp map of skew matrix to SO(3) matrix (Remember to disable autocast or increase the eps when using float16)

    Args:
        w: [*, 3]
    
    Returns:
        SO(3) matrix: [*, 3, 3]
    """
    #. [..., 3]
    w = skew_to_vec_np(w_hat)

    #. [..., 1, 1]
    theta = np.linalg.norm(w, axis=-1)[..., None, None]
    
    #. [..., 3, 3]
    w_hat_2 = w_hat @ w_hat

    # np.eye(3) + np.sin(theta) / theta * w_hat + (1 - np.cos(theta)) / (theta ** 2) * w_hat @ w_hat
    return np.eye(3) + np.sin(theta) / (theta + eps) * w_hat + (1 - np.cos(theta)) / (theta ** 2 + eps) * w_hat_2
    

def vec_to_skew_np(w):
    """(numpy ver.) rotation vector to so(3) skew-symmetric matrix

    Args:
        w: [*, 3]

    Returns:
        skew-symmetric matrix: [*, 3, 3]
    """
    #. [...,]
    x, y, z = w[..., 0], w[..., 1], w[..., 2]

    #. [...,]
    o = np.zeros_like(x)

    #. [..., 3, 3]
    # return np.array([[0, -w[..., 2], w[..., 1]], [w[..., 2], 0, -w[..., 0]], [-w[..., 1], w[..., 0], 0]])
    return np.stack([np.stack([o, -z, y], axis=-1), np.stack([z, o, -x], axis=-1), np.stack([-y, x, o], axis=-1)], axis=-2)


def skew_to_vec_np(w_hat):
    """(numpy ver.) so(3) skew-symmetric matrix to rotation vector

    Args:
        w_hat: [*, 3, 3]

    Returns:
        rotation vector: [*, 3]
    """
    return np.stack([w_hat[..., 2, 1], w_hat[..., 0, 2], w_hat[..., 1, 0]], axis=-1)

# deprecated, test for speed
def log_R_np2(R):
    """log map of SO(3) -> skew-symmetric matrix

    Args:
        R: [*, 3, 3]

    Returns:
        skew-symmetric matrix: [*, 3, 3]
    """
    #. [..., 1, 1]
    # theta = np.arccos((np.trace(R) - 1) / 2)
    theta = np.arccos((np.einsum('...ii->...', R) - 1) / 2)[..., None, None]
    
    #. [..., 3, 3]
    theta_val = np.zeros_like(theta)
    theta_mask = theta != 0
    theta_val[theta_mask] = theta[theta_mask] / (2 * np.sin(theta[theta_mask]))
    R_skew = theta_val * (R - R.swapaxes(-1, -2))
    
    return R_skew

def log_R_np3(R):
    """log map of SO(3) -> skew-symmetric matrix

    Args:
        R: [*, 3, 3]

    Returns:
        skew-symmetric matrix: [*, 3, 3]
    """
    #. [..., 1, 1]
    # theta = np.arccos((np.trace(R) - 1) / 2)
    theta = np.arccos((np.einsum('...ii->...', R) - 1) / 2)[..., None, None]
    
    #. [..., 3, 3]
    theta_val = theta / (2 * np.sin(theta))
    theta_val[theta == 0] = 0
    R_skew = theta_val * (R - R.swapaxes(-1, -2))
    
    return R_skew