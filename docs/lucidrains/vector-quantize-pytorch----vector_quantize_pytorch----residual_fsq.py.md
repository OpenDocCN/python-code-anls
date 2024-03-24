# `.\lucidrains\vector-quantize-pytorch\vector_quantize_pytorch\residual_fsq.py`

```py
import random
from math import log2
from functools import partial

from typing import List

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast

from vector_quantize_pytorch.finite_scalar_quantization import FSQ

from einops import rearrange, repeat, reduce, pack, unpack

from einx import get_at

# helper functions

# 检查值是否存在
def exists(val):
    return val is not None

# 返回列表的第一个元素
def first(l):
    return l[0]

# 如果值存在则返回值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将数字向上取整到最接近的倍数
def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# main class

class ResidualFSQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        dim,
        levels: List[int],
        num_quantizers,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        **kwargs
    ):
        super().__init__()
        codebook_dim = len(levels)

        requires_projection = codebook_dim != dim
        # 如果需要投影，则创建输入和输出的线性层
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers

        self.levels = levels
        self.layers = nn.ModuleList([])

        levels_tensor = torch.Tensor(levels)

        scales = []

        for ind in range(num_quantizers):
            scales.append((levels_tensor - 1) ** -ind)

            fsq = FSQ(
                levels = levels,
                dim = codebook_dim,
                **kwargs
            )

            self.layers.append(fsq)

        assert all([not fsq.has_projections for fsq in self.layers])

        self.codebook_size = self.layers[0].codebook_size

        # 将尺度存储为缓冲区
        self.register_buffer('scales', torch.stack(scales), persistent = False)

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

    @property
    def codebooks(self):
        # 获取所有量化器的隐式码书
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks

    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # 可能会接收到形状为 'b h w q' 的索引（accept_image_fmap）

        indices, ps = pack([indices], 'b * q')

        # 由于量化丢失，可能会传入粗糙的索引，网络应该能够重建

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0., 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # 处理量化器丢失

        mask = indices == -1
        indices = indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)

        # 屏蔽任何被丢弃的代码

        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)

        # 缩放代码

        scales = rearrange(self.scales, 'q d -> q 1 1 d')
        all_codes = all_codes * scales

        # 如果（accept_image_fmap = True），则返回形状（量化，批量，高度，宽度，维度）

        all_codes, = unpack(all_codes, ps, 'q b * d')

        return all_codes
    # 从给定的索引中获取输出
    def get_output_from_indices(self, indices):
        # 从索引中获取编码
        codes = self.get_codes_from_indices(indices)
        # 对编码进行求和
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        # 对求和后的编码进行投影
        return self.project_out(codes_summed)

    # 前向传播函数
    def forward(
        self,
        x,
        return_all_codes = False,
        rand_quantize_dropout_fixed_seed = None
    ):
        # 获取量化器数量、量化丢弃倍数、设备信息
        num_quant, quant_dropout_multiple_of, device = self.num_quantizers, self.quantize_dropout_multiple_of, x.device

        # 对输入进行投影
        x = self.project_in(x)

        quantized_out = 0.
        residual = first(self.layers).bound(x)

        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout

        # 从中随机选择一个层索引，用于进一步丢弃残差量化
        # 同时准备空索引
        if should_quantize_dropout:
            rand = random.Random(rand_quantize_dropout_fixed_seed) if exists(rand_quantize_dropout_fixed_seed) else random

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1

            null_indices = torch.full(x.shape[:2], -1., device = device, dtype = torch.long)

        # 遍历所有层
        with autocast(enabled = False):
            for quantizer_index, (layer, scale) in enumerate(zip(self.layers, self.scales)):

                if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                    all_indices.append(null_indices)
                    continue

                quantized, indices = layer(residual / scale)
                quantized = quantized * scale

                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_indices.append(indices)

        # 如果需要，进行投影
        quantized_out = self.project_out(quantized_out)

        # 将所有索引堆叠在一起
        all_indices = torch.stack(all_indices, dim = -1)

        ret = (quantized_out, all_indices)

        if not return_all_codes:
            return ret

        # 是否返回所有层中所有码书的所有编码
        all_codes = self.get_codes_from_indices(all_indices)

        # 返回所有编码的形状为 (量化器，批次，序列长度，码书维度)
        return (*ret, all_codes)
# 定义一个名为 GroupedResidualFSQ 的类，继承自 Module 类
class GroupedResidualFSQ(Module):
    # 初始化函数，接收参数 dim、groups、accept_image_fmap 和 kwargs
    def __init__(
        self,
        *,
        dim,
        groups = 1,
        accept_image_fmap = False,
        **kwargs
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化类的属性 dim 和 groups
        self.dim = dim
        self.groups = groups
        # 断言 dim 能够被 groups 整除
        assert (dim % groups) == 0
        # 计算每个组的维度
        dim_per_group = dim // groups

        # 初始化类的属性 accept_image_fmap
        self.accept_image_fmap = accept_image_fmap

        # 初始化一个空的 ModuleList 对象 rvqs
        self.rvqs = nn.ModuleList([])

        # 循环创建 groups 个 ResidualFSQ 对象并添加到 rvqs 中
        for _ in range(groups):
            self.rvqs.append(ResidualFSQ(
                dim = dim_per_group,
                **kwargs
            ))

        # 获取第一个 ResidualFSQ 对象的 codebook_size 属性作为类的 codebook_size 属性
        self.codebook_size = self.rvqs[0].codebook_size

    # 定义 codebooks 属性，返回所有 rvqs 中的 codebooks 组成的张量
    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    # 定义 split_dim 属性，根据 accept_image_fmap 的值返回不同的维度
    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    # 定义 get_codes_from_indices 方法，根据 indices 获取对应的 codes
    def get_codes_from_indices(self, indices):
        codes = tuple(rvq.get_codes_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.stack(codes)

    # 定义 get_output_from_indices 方法，根据 indices 获取对应的 outputs
    def get_output_from_indices(self, indices):
        outputs = tuple(rvq.get_output_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.cat(outputs, dim = self.split_dim)

    # 定义前向传播函数 forward，接收参数 x 和 return_all_codes
    def forward(
        self,
        x,
        return_all_codes = False
    ):
        # 获取输入 x 的形状和 split_dim
        shape, split_dim = x.shape, self.split_dim
        # 断言输入 x 在 split_dim 维度上的大小等于 dim

        assert shape[split_dim] == self.dim

        # 将特征维度分成 groups 组

        x = x.chunk(self.groups, dim = split_dim)

        forward_kwargs = dict(
            return_all_codes = return_all_codes,
            rand_quantize_dropout_fixed_seed = random.randint(0, 1e7)
        )

        # 对每个组分别调用对应的 ResidualFSQ 对象进行前向传播

        out = tuple(rvq(chunk, **forward_kwargs) for rvq, chunk in zip(self.rvqs, x))
        out = tuple(zip(*out))

        # 否则，获取所有的 zipped 输出并将它们组合起来

        quantized, all_indices, *maybe_all_codes = out

        quantized = torch.cat(quantized, dim = split_dim)
        all_indices = torch.stack(all_indices)

        ret = (quantized, all_indices, *maybe_all_codes)
        return ret
```