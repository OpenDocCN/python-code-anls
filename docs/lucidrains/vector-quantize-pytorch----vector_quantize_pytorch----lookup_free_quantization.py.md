# `.\lucidrains\vector-quantize-pytorch\vector_quantize_pytorch\lookup_free_quantization.py`

```py
"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""

from math import log2, ceil
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module
from torch.cuda.amp import autocast

from einops import rearrange, reduce, pack, unpack

# constants

Return = namedtuple('Return', ['quantized', 'indices', 'entropy_aux_loss'])

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# entropy

def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

# class

class LFQ(Module):
    def __init__(
        self,
        *,
        dim = None,
        codebook_size = None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 0.25,
        diversity_gamma = 1.,
        straight_through_activation = nn.Identity(),
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.,            # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy = 1.    # make less than 1. to only use a random fraction of the probs for per sample entropy
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'

        codebook_size = default(codebook_size, lambda: 2 ** dim)
        codebook_dim = int(log2(codebook_size))

        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.project_in = nn.Linear(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # straight through activation

        self.activation = straight_through_activation

        # entropy aux loss related weights

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # for no auxiliary loss, during inference

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # codes

        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.register_buffer('codebook', codebook, persistent = False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    # 返回当前对象的数据类型
    def dtype(self):
        return self.codebook.dtype

    # 将索引转换为代码
    def indices_to_codes(
        self,
        indices,
        project_out = True
    ):
        # 判断输入的索引是否为图像或视频数据
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        # 如果不保留代码簿的数量维度，则重新排列索引
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... -> ... 1')

        # 将索引转换为代码，代码为-1或1的位
        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        # 将位转换为代码
        codes = self.bits_to_codes(bits)

        # 重新排列代码的维度
        codes = rearrange(codes, '... c d -> ... (c d)')

        # 是否将代码投影到原始维度
        # 如果输入特征维度不是log2(代码簿大小)
        if project_out:
            codes = self.project_out(codes)

        # 将代码重新排列回原始形状
        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    # 前向传播函数
    @autocast(enabled = False)
    def forward(
        self,
        x,
        inv_temperature = 100.,
        return_loss_breakdown = False,
        mask = None,
```