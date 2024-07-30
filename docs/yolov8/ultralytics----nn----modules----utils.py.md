# `.\yolov8\ultralytics\nn\modules\utils.py`

```py
# 导入必要的库和模块
import copy  # 导入copy模块，用于深拷贝对象
import math  # 导入math模块，提供数学函数

import numpy as np  # 导入NumPy库，用于科学计算
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数库
from torch.nn.init import uniform_  # 从PyTorch的初始化模块中导入uniform_函数

__all__ = "multi_scale_deformable_attn_pytorch", "inverse_sigmoid"  # 定义模块的公开接口

def _get_clones(module, n):
    """Create a list of cloned modules from the given module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value."""
    return float(-np.log((1 - prior_prob) / prior_prob))  # 返回根据先验概率初始化的偏置值

def linear_init(module):
    """Initialize the weights and biases of a linear module."""
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)  # 初始化线性模块的权重
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)  # 如果模块具有偏置项，则初始化偏置项

def inverse_sigmoid(x, eps=1e-5):
    """Calculate the inverse sigmoid function for a tensor."""
    x = x.clamp(min=0, max=1)  # 将输入张量限制在区间 [0, 1]
    x1 = x.clamp(min=eps)  # 将输入张量在最小值eps处截断
    x2 = (1 - x).clamp(min=eps)  # 将 1-x 在最小值eps处截断
    return torch.log(x1 / x2)  # 返回对数的差值，计算逆sigmoid函数的值

def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Multiscale deformable attention.

    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape  # 获取输入张量value的形状信息
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape  # 获取采样位置张量sampling_locations的形状信息
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)  # 根据value_spatial_shapes切分value张量
    sampling_grids = 2 * sampling_locations - 1  # 计算采样网格的位置
    sampling_value_list = []  # 初始化采样值列表
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # 将value_list[level]展平并转置，然后重塑为(bs*num_heads, embed_dims, H_, W_)
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # 将sampling_grids[:, :, :, level]转置并展平，得到(bs*num_heads, num_queries, num_points, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # 使用双线性插值对value_l_进行采样，得到采样值sampling_value_l_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)  # 将采样值添加到列表中
    # 将attention_weights转置和重塑，得到形状为(bs*num_heads, 1, num_queries, num_levels*num_points)的张量
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    # 计算加权平均后的输出张量
    output = (
        # 将采样值列表按照指定维度堆叠成张量，并展开至倒数第二维
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        # 沿着最后一维度求和，得到加权平均值
        .sum(-1)
        # 将结果重新调整形状为(bs, num_heads * embed_dims, num_queries)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    # 调换第一和第二维度，并使得张量的存储顺序连续化
    return output.transpose(1, 2).contiguous()
```