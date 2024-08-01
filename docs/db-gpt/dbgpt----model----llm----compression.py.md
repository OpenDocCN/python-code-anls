# `.\DB-GPT-src\dbgpt\model\llm\compression.py`

```py
# 指定 Python 解释器
#!/usr/bin/env python3
# 指定编码格式为 UTF-8
# -*- coding: utf-8 -*-

# 导入 dataclasses 模块，用于支持数据类
import dataclasses

# 导入 PyTorch 库
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

# 定义数据类 CompressionConfig，用于存储压缩相关的配置信息
@dataclasses.dataclass
class CompressionConfig:
    """Group-wise quantization."""

    num_bits: int             # 每个数值的位数
    group_size: int           # 每组的大小
    group_dim: int            # 组维度
    symmetric: bool           # 是否对称量化
    enabled: bool = True      # 是否启用量化，默认启用

# 默认的压缩配置
default_compression_config = CompressionConfig(
    num_bits=8,                # 每个数值的位数为 8 位
    group_size=256,            # 每组包含 256 个元素
    group_dim=1,               # 组的维度为 1
    symmetric=True,            # 使用对称量化
    enabled=True               # 启用压缩
)

# 定义 CLinear 类，用于压缩的线性层
class CLinear(nn.Module):
    """Compressed Linear Layer."""

    def __init__(self, weight, bias, device):
        super().__init__()
        
        # 压缩权重数据并移动到指定设备
        self.weight = compress(weight.data.to(device), default_compression_config)
        self.bias = bias  # 不进行压缩的偏置项

    def forward(self, input: Tensor) -> Tensor:
        # 解压缩权重数据
        weight = decompress(self.weight, default_compression_config)
        # 使用解压缩后的权重进行线性变换
        return F.linear(input, weight, self.bias)

# 压缩模块的函数，遍历模块中的所有属性，并对线性层进行压缩
def compress_module(module, target_device):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            setattr(
                module,
                attr_str,
                CLinear(target_attr.weight, target_attr.bias, target_device),
            )
    for name, child in module.named_children():
        compress_module(child, target_device)

# 压缩函数，模拟组内量化过程
def compress(tensor, config):
    """Simulate group-wise quantization."""
    if not config.enabled:
        return tensor

    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )
    assert num_bits <= 8

    original_shape = tensor.shape
    # 计算组的数量
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = (
        original_shape[:group_dim]
        + (num_groups, group_size)
        + original_shape[group_dim + 1 :]
    )

    # 填充
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = (
            original_shape[:group_dim] + (pad_len,) + original_shape[group_dim + 1 :]
        )
        tensor = torch.cat(
            [tensor, torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim,
        )
    data = tensor.view(new_shape)

    # 量化过程
    if symmetric:
        B = 2 ** (num_bits - 1) - 1
        scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
        data = data * scale
        data = data.clamp_(-B, B).round_().to(torch.int8)
        return data, scale, original_shape
    else:
        B = 2**num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        return data, mn, scale, original_shape

# 解压缩函数的声明
def decompress(packed_data, config):
    """Simulate group-wise dequantization."""
    # 如果未启用配置中的功能，则直接返回原始打包数据
    if not config.enabled:
        return packed_data

    # 从配置中获取组大小、位数、组维度和对称性设置
    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )

    # 对数据进行反量化操作
    # 如果设置为对称模式
    if symmetric:
        data, scale, original_shape = packed_data
        # 数据除以缩放比例，实现反量化
        data = data / scale
    else:
        data, mn, scale, original_shape = packed_data
        # 数据除以缩放比例，然后加上均值，实现反量化
        data = data / scale
        data.add_(mn)

    # 解除填充操作
    # 计算需要填充的长度，确保数据尺寸可以被组大小整除
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len:
        # 创建一个包含填充后维度的形状
        padded_original_shape = (
            original_shape[:group_dim]
            + (original_shape[group_dim] + pad_len,)
            + original_shape[group_dim + 1 :]
        )
        # 调整数据形状，以应用填充
        data = data.reshape(padded_original_shape)
        # 生成索引切片，以裁剪填充后的数据到原始形状
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        # 如果不需要填充，则直接返回数据，视图化为原始形状
        return data.view(original_shape)
```