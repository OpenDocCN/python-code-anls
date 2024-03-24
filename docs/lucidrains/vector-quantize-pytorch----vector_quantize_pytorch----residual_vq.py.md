# `.\lucidrains\vector-quantize-pytorch\vector_quantize_pytorch\residual_vq.py`

```py
# 导入必要的库
import random
from math import ceil
from functools import partial
from itertools import zip_longest

import torch
from torch import nn
import torch.nn.functional as F
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize

from einops import rearrange, repeat, reduce, pack, unpack

from einx import get_at

# 辅助函数

# 检查变量是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 向上取整到最接近的倍数
def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# 主类

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        dim,
        num_quantizers,
        codebook_dim = None,
        shared_codebook = False,
        heads = 1,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        accept_image_fmap = False,
        **kwargs
    ):
        super().__init__()
        assert heads == 1, 'residual vq is not compatible with multi-headed codes'
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        # 如果需要投影，则创建投影层
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers

        self.accept_image_fmap = accept_image_fmap
        # 创建多个 VectorQuantize 层
        self.layers = nn.ModuleList([VectorQuantize(dim = codebook_dim, codebook_dim = codebook_dim, accept_image_fmap = accept_image_fmap, **kwargs) for _ in range(num_quantizers)])

        assert all([not vq.has_projections for vq in self.layers])

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

        if not shared_codebook:
            return

        # 如果共享码书，则将所有 VectorQuantize 层的码书设置为第一个 VectorQuantize 层的码书
        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    @property
    def codebooks(self):
        # 获取所有码书的嵌入向量
        codebooks = [layer._codebook.embed for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        codebooks = rearrange(codebooks, 'q 1 c d -> q c d')
        return codebooks

    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # 可能接收到形状为 'b h w q' 的索引（如果 accept_image_fmap 为 True）

        indices, ps = pack([indices], 'b * q')

        # 由于量化丢失，可能传入粗糙的索引，网络应该能够重建

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0., 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # 处理量化器丢失

        mask = indices == -1.
        indices = indices.masked_fill(mask, 0) # 用一个虚拟码填充，稍后会被屏蔽

        all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)

        # 屏蔽任何被丢弃的码

        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)

        # 如果（accept_image_fmap = True），则返回形状为（量化器，批次，高度，宽度，维度）

        all_codes, = unpack(all_codes, ps, 'q b * d')

        return all_codes
    # 根据给定的索引获取输出
    def get_output_from_indices(self, indices):
        # 从索引获取编码
        codes = self.get_codes_from_indices(indices)
        # 对编码进行求和
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        # 投影编码
        return self.project_out(codes_summed)

    # 前向传播函数
    def forward(
        self,
        x,
        mask = None,
        indices = None,
        return_all_codes = False,
        sample_codebook_temp = None,
        freeze_codebook = False,
        rand_quantize_dropout_fixed_seed = None
    ):
        # 获取一些参数
        num_quant, quant_dropout_multiple_of, return_loss, device = self.num_quantizers, self.quantize_dropout_multiple_of, exists(indices), x.device

        # 对输入进行投影
        x = self.project_in(x)

        # 断言不接受图像特征图和存在索引
        assert not (self.accept_image_fmap and exists(indices))

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []

        if return_loss:
            assert not torch.any(indices == -1), 'some of the residual vq indices were dropped out. please use indices derived when the module is in eval mode to derive cross entropy loss'
            ce_losses = []

        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss

        # 采样一个层索引来进一步丢弃残差量化
        # 同时准备空索引和损失
        if should_quantize_dropout:
            rand = random.Random(rand_quantize_dropout_fixed_seed) if exists(rand_quantize_dropout_fixed_seed) else random

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1

            null_indices_shape = (x.shape[0], *x.shape[-2:]) if self.accept_image_fmap else tuple(x.shape[:2])
            null_indices = torch.full(null_indices_shape, -1., device = device, dtype = torch.long)
            null_loss = torch.full((1,), 0., device = device, dtype = x.dtype)

        # 遍历所有层
        for quantizer_index, layer in enumerate(self.layers):

            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            quantized, *rest = layer(
                residual,
                mask = mask,
                indices = layer_indices,
                sample_codebook_temp = sample_codebook_temp,
                freeze_codebook = freeze_codebook
            )

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            if return_loss:
                ce_loss = rest[0]
                ce_losses.append(ce_loss)
                continue

            embed_indices, loss = rest

            all_indices.append(embed_indices)
            all_losses.append(loss)

        # 投影输出，如果需要的话
        quantized_out = self.project_out(quantized_out)

        # 是否提前返回交叉熵损失
        if return_loss:
            return quantized_out, sum(ce_losses)

        # 堆叠所有损失和索引
        all_losses, all_indices = map(partial(torch.stack, dim = -1), (all_losses, all_indices))

        ret = (quantized_out, all_indices, all_losses)

        if return_all_codes:
            # 是否返回所有层中所有码书的所有编码
            all_codes = self.get_codes_from_indices(all_indices)

            # 将返回所有编码的形状设置为(量化器，批次，序列长度，码书维度)
            ret = (*ret, all_codes)

        return ret
# 定义一个名为 GroupedResidualVQ 的类，继承自 nn.Module
class GroupedResidualVQ(nn.Module):
    # 初始化函数，接受参数 dim、groups、accept_image_fmap 和 kwargs
    def __init__(
        self,
        *,
        dim,
        groups = 1,
        accept_image_fmap = False,
        **kwargs
    ):
        super().__init__()
        # 初始化类的属性
        self.dim = dim
        self.groups = groups
        assert (dim % groups) == 0
        dim_per_group = dim // groups

        self.accept_image_fmap = accept_image_fmap

        self.rvqs = nn.ModuleList([])

        # 根据 groups 的数量创建 ResidualVQ 对象列表
        for _ in range(groups):
            self.rvqs.append(ResidualVQ(
                dim = dim_per_group,
                accept_image_fmap = accept_image_fmap,
                **kwargs
            ))

    # 返回所有 rvq 对象的 codebooks
    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    # 返回分割维度的值
    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    # 根据索引获取代码
    def get_codes_from_indices(self, indices):
        codes = tuple(rvq.get_codes_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.stack(codes)

    # 根据索引获取输出
    def get_output_from_indices(self, indices):
        outputs = tuple(rvq.get_output_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.cat(outputs, dim = self.split_dim)

    # 前向传播函数
    def forward(
        self,
        x,
        indices = None,
        return_all_codes = False,
        sample_codebook_temp = None,
        freeze_codebook = False,
        mask = None,
    ):
        shape, split_dim = x.shape, self.split_dim
        assert shape[split_dim] == self.dim

        # 将特征维度分成多个组

        x = x.chunk(self.groups, dim = split_dim)

        indices = default(indices, tuple())
        return_ce_loss = len(indices) > 0
        assert len(indices) == 0 or len(indices) == self.groups

        forward_kwargs = dict(
            return_all_codes = return_all_codes,
            sample_codebook_temp = sample_codebook_temp,
            mask = mask,
            freeze_codebook = freeze_codebook,
            rand_quantize_dropout_fixed_seed = random.randint(0, 1e7)
        )

        # 对每个组调用 ResidualVQ

        out = tuple(rvq(chunk, indices = chunk_indices, **forward_kwargs) for rvq, chunk, chunk_indices in zip_longest(self.rvqs, x, indices))
        out = tuple(zip(*out))

        # 如果���回交叉熵损失到 rvq codebooks

        if return_ce_loss:
            quantized, ce_losses = out
            return torch.cat(quantized, dim = split_dim), sum(ce_losses)

        # 否则，获取所有输出并组合它们

        quantized, all_indices, commit_losses, *maybe_all_codes = out

        quantized = torch.cat(quantized, dim = split_dim)
        all_indices = torch.stack(all_indices)
        commit_losses = torch.stack(commit_losses)

        ret = (quantized, all_indices, commit_losses, *maybe_all_codes)
        return ret
```