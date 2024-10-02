from typing import Tuple, List, Callable, Any, Dict, Sequence, Optional

import numpy as np

import torch
import torch.nn as nn

def batched_index_select(values, indices, dim=1):
    """
    Modified from progres
    Batched index select
    Args:
        values (torch.Tensor): 
            Tensor to select values from. Shape: ((*batch_dims), index_dim, *(values_dims))
        indices (torch.Tensor):
            Indices to select. Shape: ((*batch_dims), (*extra_dims), index_dim)
        dim (int):
            Dimension to select from. Default: 1
    Returns:
        torch.Tensor:
            Selected values. Shape: ((*batch_dims), (*extra_dims), index_dim, *(values_dims))

    Example:
        >>> values = torch.arange(24).reshape(2, 3, 4)
        >>> indices = torch.randint(0, 3, (2, 5, 3))
        >>> batched_index_select(values, indices) -> torch.Size([2, 5, 3, 4])
    """
    # shape of dims after index dim
    value_dims = values.shape[(dim + 1):]
    # shape of `values` and `indices`
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))

    # add value_dims to indices, which will be the final shape of returned tensor
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)

    # add extra dims to values, such that values and indices have the same shape
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]
    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    # the final indexed dim is original dim + value_expand_len
    dim += value_expand_len
    return values.gather(dim, indices)
