# `.\lucidrains\vector-quantize-pytorch\vector_quantize_pytorch\residual_lfq.py`

```
# 导入所需的库
import random
from math import log2
from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast

# 导入自定义的 LFQ 模块
from vector_quantize_pytorch.lookup_free_quantization import LFQ

# 导入 einops 库中的函数
from einops import rearrange, repeat, reduce, pack, unpack

# 导入自定义的 get_at 函数
from einx import get_at

# 辅助函数

# 检查变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将数字向上取整到最接近的倍数
def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# 主类

class ResidualLFQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        dim,
        num_quantizers,
        codebook_size,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        **kwargs
    ):
        super().__init__()
        codebook_dim = int(log2(codebook_size))

        requires_projection = codebook_dim != dim
        # 如果 codebook_dim 不等于 dim，则需要进行投影
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers

        self.layers = nn.ModuleList([])

        # 创建 num_quantizers 个 LFQ 层
        for ind in range(num_quantizers):
            codebook_scale = 2 ** -ind

            lfq = LFQ(
                dim = codebook_dim,
                codebook_scale = codebook_scale,
                **kwargs
            )

            self.layers.append(lfq)

        # 断言所有 LFQ 层都没有投影
        assert all([not lfq.has_projections for lfq in self.layers])

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        # 断言 quantize_dropout_cutoff_index 大于等于 0
        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # 编码论文提出结构化的 dropout，这里设置为 4

    @property
    def codebooks(self):
        # 获取所有 LFQ 层的 codebook，并按维度 0 进行堆叠
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks

    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # 可能接收到 'b h w q' 形状的 indices（accept_image_fmap）

        indices, ps = pack([indices], 'b * q')

        # 由于 quantize dropout，可能传入粗糙的 indices，网络应该能够重构

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0., '如果希望从较少的精细量化信号重构，则 quantize dropout 必须大于 0'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # 处理量化器 dropout

        mask = indices == -1.
        indices = indices.masked_fill(mask, 0)  # 有一个虚拟代码被掩盖

        all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)

        # 掩盖任何被 dropout 的代码

        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)

        # 如果（accept_image_fmap = True），则返回形状为（quantize，batch，height，width，dimension）

        all_codes, = unpack(all_codes, ps, 'q b * d')

        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def forward(
        self,
        x,
        mask = None,
        return_all_codes = False,
        rand_quantize_dropout_fixed_seed = None
        ):
            # 获取量化器数量、量化丢弃的倍数、设备信息
            num_quant, quant_dropout_multiple_of, device = self.num_quantizers, self.quantize_dropout_multiple_of, x.device

            # 对输入进行投影
            x = self.project_in(x)

            # 初始化量化输出和残差
            quantized_out = 0.
            residual = x

            # 初始化损失列表和索引列表
            all_losses = []
            all_indices = []

            # 是否需要进行量化丢弃
            should_quantize_dropout = self.training and self.quantize_dropout

            # 随机选择一个层索引，用于进一步丢弃残差量化
            # 同时准备空索引和损失
            if should_quantize_dropout:
                rand = random.Random(rand_quantize_dropout_fixed_seed) if exists(rand_quantize_dropout_fixed_seed) else random

                rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

                if quant_dropout_multiple_of != 1:
                    rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1

                null_indices = torch.full(x.shape[:2], -1., device=device, dtype=torch.long)
                null_loss = torch.tensor(0., device=device, dtype=x.dtype)

            # 遍历所有层
            with autocast(enabled=False):
                for quantizer_index, layer in enumerate(self.layers):

                    # 如果需要进行量化丢弃且当前层索引大于随机选择的丢弃索引
                    if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                        all_indices.append(null_indices)
                        all_losses.append(null_loss)
                        continue

                    # 进行量化操作，获取量化结果、索引和损失
                    quantized, indices, loss = layer(residual, mask=mask)

                    # 更新残差和量化输出
                    residual = residual - quantized.detach()
                    quantized_out = quantized_out + quantized

                    # 添加索引和损失到列表中
                    all_indices.append(indices)
                    all_losses.append(loss)

            # 对输出进行投影
            quantized_out = self.project_out(quantized_out)

            # 合并所有损失和索引
            all_losses, all_indices = map(partial(torch.stack, dim=-1), (all_losses, all_indices))

            # 返回结果
            ret = (quantized_out, all_indices, all_losses)

            # 如果不需要返回所有编码，则直接返回结果
            if not return_all_codes:
                return ret

            # 是否返回所有层中所有码书的所有编码
            all_codes = self.get_codes_from_indices(all_indices)

            # 返回所有编码的形状为(量化器，批次，序列长度，码书维度)
            return (*ret, all_codes)
# 定义一个名为 GroupedResidualLFQ 的类，继承自 Module 类
class GroupedResidualLFQ(Module):
    # 初始化函数，接受一些参数
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
        # 初始化类的属性
        self.dim = dim
        self.groups = groups
        # 确保 dim 能够被 groups 整除
        assert (dim % groups) == 0
        dim_per_group = dim // groups

        self.accept_image_fmap = accept_image_fmap

        # 创建一个空的 ModuleList 对象
        self.rvqs = nn.ModuleList([])

        # 根据 groups 的数量循环创建 ResidualLFQ 对象并添加到 rvqs 中
        for _ in range(groups):
            self.rvqs.append(ResidualLFQ(
                dim = dim_per_group,
                **kwargs
            ))

    # 定义 codebooks 属性，返回所有 rvq 对象的 codebooks 组成的张量
    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    # 定义 split_dim 属性，根据 accept_image_fmap 的值返回不同的维度
    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    # 根据 indices 获取每个 rvq 对象的 codes，并返回组合后的张量
    def get_codes_from_indices(self, indices):
        codes = tuple(rvq.get_codes_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.stack(codes)

    # 根据 indices 获取每个 rvq 对象的 output，并返回组合后的张量
    def get_output_from_indices(self, indices):
        outputs = tuple(rvq.get_output_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.cat(outputs, dim = self.split_dim)

    # 前向传播函数，接受输入 x 和一些参数
    def forward(
        self,
        x,
        mask = None,
        return_all_codes = False
    ):
        shape, split_dim = x.shape, self.split_dim
        assert shape[split_dim] == self.dim

        # 将特征维度按 split_dim 分成 groups 组

        x = x.chunk(self.groups, dim = split_dim)

        forward_kwargs = dict(
            mask = mask,
            return_all_codes = return_all_codes,
            rand_quantize_dropout_fixed_seed = random.randint(0, 1e7)
        )

        # 对每个 group 调用 residual vq

        out = tuple(rvq(chunk, **forward_kwargs) for rvq, chunk in zip(self.rvqs, x))
        out = tuple(zip(*out))

        # 否则，获取所有的 zipped 输出并组合它们

        quantized, all_indices, commit_losses, *maybe_all_codes = out

        quantized = torch.cat(quantized, dim = split_dim)
        all_indices = torch.stack(all_indices)
        commit_losses = torch.stack(commit_losses)

        ret = (quantized, all_indices, commit_losses, *maybe_all_codes)
        return ret
```