import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

def make_one_hot(x, num_classes):
    """Make one-hot encoding from indices.
    """
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot