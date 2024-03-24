# `.\lucidrains\vector-quantize-pytorch\vector_quantize_pytorch\latent_quantization.py`

```
"""
Disentanglement via Latent Quantization
 - https://arxiv.org/abs/2305.18378
Code adapted from Jax version in https://github.com/kylehkhsu/latent_quantization
"""

# 导入所需的库
from typing import List, Optional, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor, int32
from torch.optim import Optimizer
from einops import rearrange, pack, unpack

# 辅助函数

# 检查变量是否存在
def exists(v):
    return v is not None

# 返回第一个非空参数
def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

# 将单个张量按指定模式打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 将单个张量按指定模式解包
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 主类

class LatentQuantize(Module):
    # 计算量化损失
    def quantization_loss(self, z: Tensor, zhat: Tensor, reduce="mean") -> Tensor:
        """Computes the quantization loss."""
        return F.mse_loss(zhat.detach(), z, reduction=reduce)

    # 计算约束损失
    def commitment_loss(self, z: Tensor, zhat: Tensor, reduce="mean") -> Tensor:
        """Computes the commitment loss."""
        return F.mse_loss(z.detach(), zhat, reduction=reduce)    

    # 对 z 进行量化
    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z.
        The quantization is done by measuring the distance between the input and the codebook values per latent dimension
        and returning the index of the closest codebook value.
        """
        def distance(x, y):
            return torch.abs(x - y)
        
        if self._equal_levels:
            index = torch.argmin(distance(z[..., None], self.values_per_latent), dim=-1)
            quantize = self.values_per_latent[torch.arange(self.dim), index]
        else:
            index = torch.stack([torch.argmin(distance(z[..., i, None], self.values_per_latent[i]), dim=-1) for i in range(self.codebook_dim)], dim=-1)
            quantize = torch.stack([self.values_per_latent[i][index[..., i]] for i in range(self.codebook_dim)], dim=-1)

        quantize = z + (quantize - z).detach()
        #half_width = self._levels // 2 / 2  # Renormalize to [-0.5, 0.5].
        return quantize #/ half_width
    
    # 缩放和移位 zhat 从 [-0.5, 0.5] 到 [0, level_per_dim]
    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        """ scale and shift zhat from [-0.5, 0.5] to [0, level_per_dim]"""
        half_width = self._levels // 2
        return (zhat_normalized * 2 * half_width) + half_width
    
    # 将 zhat 反向缩放和移位为 [-0.5, 0.5]
    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        """normalize zhat to [-0.5, 0.5]"""
        half_width = self._levels // 2
        return (zhat - half_width) / half_width / 2
    
    # 将编码转换为索引
    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` which contains the number per latent to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)
    
    # 将索引转换为编码
    def indices_to_codes(
        self,
        indices: Tensor,
        project_out = True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes
    # 对输入张量进行量化和投影操作
    def quantize_and_project(self, z: Tensor, is_img_or_video, ps) -> Tensor:
        # 对输入张量进行量化操作
        codes = self.quantize(z)
        # 将量化后的结果转换为索引
        indices = self.codes_to_indices(codes)

        # 重排列张量维度
        codes = rearrange(codes, 'b n c d -> b n (c d)')

        # 对量化后的结果进行投影操作
        out = self.project_out(codes)

        # 重新构建图像或视频的维度

        if is_img_or_video:
            # 解包张量
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')
        return codes, out, indices

    # 前向传播函数
    def forward(self,
                 z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension 
        c - number of codebook dim
        """

        # 判断输入张量是否为图像或视频
        is_img_or_video = z.ndim >= 4
        original_input = z
        # 标准化图像或视频为 (batch, seq, dimension) 格式
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        if is_img_or_video:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        # 投影输入张量
        z = self.project_in(z)
        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # 对输入张量进行量化操作
        codes = self.quantize(z)
        # 将量化后的结果转换为索引
        indices = self.codes_to_indices(codes)

        # 重排列张量维度
        codes = rearrange(codes, 'b n c d -> b n (c d)')

        # 对量化后的结果进行投影操作
        out = self.project_out(codes)

        # 重新构建图像或视频的维度
        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')
            
        if should_inplace_optimize and self.training and not self.optimize_values:
            # 更新码���
            loss = self.commitment_loss(z, out) if self.commitment_loss_weight!=0  else torch.tensor(0.)
            loss+= self.quantization_loss(z, out) if self.quantization_loss_weight!=0 else torch.tensor(0.)
            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()
            # 再次对输入张量进行量化
            codes = self.quantize(z)
            indices = self.codes_to_indices(codes)
            codes = rearrange(codes, 'b n c d -> b n (c d)')
            out = self.project_out(codes)
            
            if is_img_or_video:
                out = unpack_one(out, ps, 'b * d')
                out = rearrange(out, 'b ... d -> b d ...')

                indices = unpack_one(indices, ps, 'b * c')

            if not self.keep_num_codebooks_dim:
                indices = rearrange(indices, '... 1 -> ...')


        # 计算损失
        commitment_loss = self.commitment_loss(original_input, out) if self.training and self.commitment_loss_weight!=0  else torch.tensor(0.)
        quantization_loss = self.quantization_loss(original_input, out) if self.training and self.quantization_loss_weight!=0 else torch.tensor(0.)


        loss = self.commitment_loss_weight * commitment_loss + self.quantization_loss_weight * quantization_loss 

        return out, indices, loss
```