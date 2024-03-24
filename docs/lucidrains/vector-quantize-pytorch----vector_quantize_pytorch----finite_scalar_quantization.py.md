# `.\lucidrains\vector-quantize-pytorch\vector_quantize_pytorch\finite_scalar_quantization.py`

```py
"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.cuda.amp import autocast

from einops import rearrange, pack, unpack

# helper functions

# 检查变量是否存在
def exists(v):
    return v is not None

# 返回第一个存在的参数
def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

# 将单个张量按照指定模式打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 将单个张量按照指定模式解包
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

# 使用直通梯度进行四舍五入
def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

# main class

class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64)
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out = False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

        self.allowed_dtypes = allowed_dtypes

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width
    
    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)
    
    def indices_to_codes(
        self,
        indices: Tensor,
        project_out = True
    def codes_to_indices(self, indices: Tensor) -> Tensor:
        """Inverse of `codes_to_indices`."""
        
        # 检查输入张量的维度是否大于等于3（图片或视频）
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        # 将输入张量的维度调整为 '... -> ... 1'
        indices = rearrange(indices, '... -> ... 1')
        
        # 计算非中心化的编码
        codes_non_centered = (indices // self._basis) % self._levels
        # 对编码进行缩放和偏移
        codes = self._scale_and_shift_inverse(codes_non_centered)

        # 如果需要保留编码簇维度
        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        # 如果需要进行投影
        if project_out:
            codes = self.project_out(codes)

        # 如果是图片或视频
        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        # 返回编码
        return codes

    @autocast(enabled = False)
    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        # 保存原始数据类型
        orig_dtype = z.dtype
        # 检查输入张量的维度是否大于等于4（图片或视频）
        is_img_or_video = z.ndim >= 4

        # 确保输入张量的数据类型在允许的范围内
        if z.dtype not in self.allowed_dtypes:
            z = z.float()

        # 标准化图片或视频数据为 (batch, seq, dimension) 的形式
        if is_img_or_video:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        # 断言输入张量的最后一个维度是否与指定的维度相匹配
        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        # 对输入张量进行投影
        z = self.project_in(z)

        # 调整输入张量的维度为 'b n (c d)'
        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # 对输入张量进行量化
        codes = self.quantize(z)
        # 将编码转换为索引
        indices = self.codes_to_indices(codes)

        # 调整编码的维度为 'b n (c d)'
        codes = rearrange(codes, 'b n c d -> b n (c d)')

        # 对输出进行投影
        out = self.project_out(codes)

        # 恢复图片或视频的维度
        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        # 如果不需要保留编码簇维度
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        # 将输出转换回原始数据类型
        if out.dtype != orig_dtype:
            out = out.type(orig_dtype)

        # 返回量化输出和索引
        return out, indices
```