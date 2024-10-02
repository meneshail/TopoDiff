from functools import partial
import importlib
import math
from typing import Optional, Callable, List, Tuple, Sequence
import numpy as np
import logging

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if(deepspeed_is_installed):
    import deepspeed

# fa_is_installed = importlib.util.find_spec("flash_attn") is not None
# if(fa_is_installed):
#     from flash_attn.bert_padding import unpad_input, pad_input
#     from flash_attn.flash_attention import FlashAttention
#     from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func

import torch
import torch.nn as nn
from scipy.stats import truncnorm

# from myopenfold.utils.checkpointing import get_checkpoint_fn

#  Skip load attention_core when the dependency is not installed.
if importlib.util.find_spec("attn_core_inplace_cuda") is not None:
    from myopenfold.utils.kernel.attention_core import attention_core
else:
    logging.warning("attn_core_inplace_cuda not found. Skipping load of attention_core")
from myopenfold.utils.precision_utils import is_fp16_enabled
from myopenfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


DEFAULT_LMA_Q_CHUNK_SIZE=1024
DEFAULT_LMA_KV_CHUNK_SIZE=4096


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")


class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()
        
        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x): 
        d = x.dtype
        # print(d)
        deepspeed_is_initialized = (
            deepspeed_is_installed and 
            deepspeed.utils.is_initialized()
        )
        if(d is torch.bfloat16 and not deepspeed_is_initialized):
            with torch.cuda.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(
                    x, 
                    self.c_in, 
                    self.weight.to(dtype=d), 
                    self.bias.to(dtype=d), 
                    self.eps
                )
        else:
            # print('x', torch.isnan(x).any(), torch.sum(torch.isnan(x)), x)
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )
        # print('out', torch.isnan(out).any(), torch.sum(torch.isnan(out)))

        return out


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    d = t.dtype
    deepspeed_is_initialized = (
        deepspeed_is_installed and 
        deepspeed.utils.is_initialized()
    )
    if(d is torch.bfloat16 and not deepspeed_is_initialized):
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


#@torch.jit.script
def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    #. query [*, H, Q, C_hidden]
    #. key [*, H, K, C_hidden]
    #. value [*, H, K, C_hidden]
    
    # key [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    #. batched matmul, the [:-2] dims are using as batched
    # a [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b
    
    #. softmax over the key dimension, which means each row over the query dimension sums to 1
    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


# @torch.jit.ignore
# def _attention_chunked_trainable(
#     query, key, value, biases, chunk_size, chunk_dim, checkpoint, 
# ):
#     if(checkpoint and len(biases) > 2):
#         raise ValueError(
#             "Checkpointed version permits only permits two bias terms"
#         )

#     def _checkpointable_attention(q, k, v, b1, b2):
#         bs = [b for b in [b1, b2] if b is not None]
#         a = _attention(q, k, v, bs)
#         return a

#     o_chunks = []
#     checkpoint_fn = get_checkpoint_fn()
#     count = query.shape[chunk_dim]
#     for start in range(0, count, chunk_size):
#         end = start + chunk_size
#         idx = [slice(None)] * len(query.shape)
#         idx[chunk_dim] = slice(start, end)
#         idx_tup = tuple(idx)
#         q_chunk = query[idx_tup]
#         k_chunk = key[idx_tup]
#         v_chunk = value[idx_tup]

#         def _slice_bias(b):
#             idx[chunk_dim] = (
#                 slice(start, end) if b.shape[chunk_dim] != 1 else slice(None)
#             )
#             return b[tuple(idx)]

#         if(checkpoint):
#             bias_1_chunk, bias_2_chunk = [
#                 _slice_bias(b) if b is not None else None
#                 for b in (biases + [None, None])[:2]
#             ]

#             o_chunk = checkpoint_fn(_checkpointable_attention,
#                 q_chunk, k_chunk, v_chunk, bias_1_chunk, bias_2_chunk
#             )
#         else:
#             bias_chunks = [
#                 _slice_bias(b) for b in biases
#             ]

#             o_chunk = _attention(q_chunk, k_chunk, v_chunk, bias_chunks)
            
#         o_chunk = o_chunk.transpose(-2, -3)
#         o_chunks.append(o_chunk)

#     o = torch.cat(o_chunks, dim=chunk_dim)
#     return o


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        depth = 0,
        ind = 0,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            #. gating is used by default
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.depth = depth
        self.ind = ind

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        #. in the current version, c_hidden is de facto per-head channel dimension
        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )

        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, init="final"
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, init="gating"
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
        q_x: torch.Tensor, 
        kv_x: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """ compute q, k, v from q_x and kv_x

        Args:
            q_x: [*, Q, C_q] query data
            kv_x: [*, K, C_k] key data

        Returns:
            q: [*, H, Q, C_hidden] query data
            k: [*, H, K, C_hidden] key data
            v: [*, H, K, C_hidden] value data

        """
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self,
        o: torch.Tensor, 
        q_x: torch.Tensor
    ) -> torch.Tensor:
        if(self.linear_g is not None):
            #. [*, Q, H * C_hidden]
            g = self.sigmoid(self.linear_g(q_x))
        
            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_memory_efficient_kernel: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_memory_efficient_kernel:
                Whether to use a custom memory-efficient attention kernel.
                This should be the default choice for most. If none of the
                "use_<...>" flags are True, a stock PyTorch implementation
                is used instead
            lma_q_chunk_size:
                Query chunk size (for LMA)
            lma_kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """

        attn_options = [use_memory_efficient_kernel]
        if(sum(attn_options) > 1):
            raise ValueError(
                "Choose at most one alternative attention algorithm"
            )

        if(biases is None):
            biases = []
        
        #. 1. compute q, k, v
        # [*, H, Q/K, C_hidden]
        q, k, v = self._prep_qkv(q_x, kv_x)

        # [*, Q, H, C_hidden]
        #. torch.get_autocast_gpu_dtype() could be [torch.float16, torch.half]
        if is_fp16_enabled():
            use_memory_efficient_kernel = False
        
        #. 2. compute attention
        if(use_memory_efficient_kernel):
            if(len(biases) > 2):
                raise ValueError(
                    "If use_memory_efficient_kernel is True, you may only "
                    "provide up to two bias terms"
                )
            # print("Warning: Using memory_efficient_kernel for attention computation, which may cause error on RTX Titan GPUs.")  # DEBUG
            o = attention_core(q, k, v, *((biases + [None] * 2)[:2]))
            o = o.transpose(-2, -3)
        else:
            o = _attention(q, k, v, biases)
            o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o


class GlobalAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, inf, eps):
        super(GlobalAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.linear_q = Linear(
            c_in, c_hidden * no_heads, bias=False, init="glorot"
        )

        self.linear_k = Linear(
            c_in, c_hidden, bias=False, init="glorot",
        )
        self.linear_v = Linear(
            c_in, c_hidden, bias=False, init="glorot",
        )
        self.linear_g = Linear(c_in, c_hidden * no_heads, init="gating")
        self.linear_o = Linear(c_hidden * no_heads, c_in, init="final")

        self.sigmoid = nn.Sigmoid()

    def forward(self, 
        m: torch.Tensor, 
        mask: torch.Tensor,
        use_lma: bool = False,
    ) -> torch.Tensor:
        # [*, N_res, C_in]
        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (
            torch.sum(mask, dim=-1)[..., None] + self.eps
        )

        # [*, N_res, H * C_hidden]
        q = self.linear_q(q)
        q *= (self.c_hidden ** (-0.5))

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, C_hidden]
        k = self.linear_k(m)
        v = self.linear_v(m)

        bias = (self.inf * (mask - 1))[..., :, None, :]
        if(not use_lma):
            # [*, N_res, H, N_seq]
            a = torch.matmul(
                q,
                k.transpose(-1, -2),  # [*, N_res, C_hidden, N_seq]
            )
            a += bias
            a = softmax_no_cast(a)

            # [*, N_res, H, C_hidden]
            o = torch.matmul(
                a,
                v,
            )
        else:
            o = _lma(
                q, 
                k, 
                v, 
                [bias], 
                DEFAULT_LMA_Q_CHUNK_SIZE, 
                DEFAULT_LMA_KV_CHUNK_SIZE
            )

        # [*, N_res, N_seq, C_hidden]
        g = self.sigmoid(self.linear_g(m))

        # [*, N_res, N_seq, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, H, C_hidden]
        o = o.unsqueeze(-3) * g

        # [*, N_res, N_seq, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))

        # [*, N_res, N_seq, C_in]
        m = self.linear_o(o)

        return m