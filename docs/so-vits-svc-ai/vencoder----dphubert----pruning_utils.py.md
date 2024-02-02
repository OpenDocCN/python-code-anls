# `so-vits-svc\vencoder\dphubert\pruning_utils.py`

```py
# 用于修剪的实用函数

from typing import Union

import torch
import torch.nn as nn


def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: str):
    "在原地修剪线性层。"
    # 注意：权重：（输出特征数，输入特征数），偏置：（输出特征数，）
    if dim == "input":
        dim = 1
        layer.in_features = len(index)
    elif dim == "output":
        dim = 0
        layer.out_features = len(index)
    else:
        raise ValueError

    layer.weight = nn.Parameter(layer.weight.index_select(dim, index).clone().detach())
    if layer.bias is not None and dim == 0:
        layer.bias = nn.Parameter(layer.bias.index_select(0, index).clone().detach())


def prune_conv1d_layer(layer: nn.Conv1d, index: torch.LongTensor, dim: str):
    """在原地修剪一维卷积层。"""
    # 注意：权重：（输出通道数，输入通道数，卷积核大小），偏置：（输出通道数，）
    if dim == "input":
        dim = 1
        layer.in_channels = len(index)
    elif dim == "output":
        dim = 0
        layer.out_channels = len(index)
    else:
        raise ValueError
    
    layer.weight = nn.Parameter(layer.weight.index_select(dim, index).clone().detach())
    if layer.bias is not None and dim == 0:
        layer.bias = nn.Parameter(layer.bias.index_select(0, index).clone().detach())


def prune_layer_norm(layernorm: Union[nn.LayerNorm, nn.GroupNorm], index: torch.LongTensor):
    """在原地修剪层归一化或分组归一化。"""
    layernorm.weight = nn.Parameter(layernorm.weight.index_select(0, index).clone().detach())
    layernorm.bias = nn.Parameter(layernorm.bias.index_select(0, index).clone().detach())
    if isinstance(layernorm, nn.LayerNorm):
        layernorm.normalized_shape = (len(index),)
    elif isinstance(layernorm, nn.GroupNorm):
        layernorm.num_groups = len(index)
        layernorm.num_channels = len(index)
```