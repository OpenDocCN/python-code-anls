# `.\pytorch\torch\nn\utils\fusion.py`

```
# 导入未来版本中的注解支持
from __future__ import annotations

# 导入必要的库
import copy
from typing import Optional, Tuple, TypeVar

# 导入 PyTorch 库
import torch

# 定义公开的符号列表
__all__ = [
    "fuse_conv_bn_eval",
    "fuse_conv_bn_weights",
    "fuse_linear_bn_eval",
    "fuse_linear_bn_weights",
]

# 定义类型变量
ConvT = TypeVar("ConvT", bound="torch.nn.modules.conv._ConvNd")
LinearT = TypeVar("LinearT", bound="torch.nn.Linear")

# 将卷积和 BatchNorm 融合为一个评估模式下的卷积模块
def fuse_conv_bn_eval(
    conv: ConvT,
    bn: torch.nn.modules.batchnorm._BatchNorm,
    transpose: bool = False,
) -> ConvT:
    r"""Fuse a convolutional module and a BatchNorm module into a single, new convolutional module.

    Args:
        conv (torch.nn.modules.conv._ConvNd): A convolutional module.
        bn (torch.nn.modules.batchnorm._BatchNorm): A BatchNorm module.
        transpose (bool, optional): If True, transpose the convolutional weight. Defaults to False.

    Returns:
        torch.nn.modules.conv._ConvNd: The fused convolutional module.

    .. note::
        Both ``conv`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.
    """
    # 断言确保 conv 和 bn 都处于评估模式
    assert not (conv.training or bn.training), "Fusion only for eval!"
    # 深拷贝卷积模块，以避免原始模块被修改
    fused_conv = copy.deepcopy(conv)

    # 断言确保 BatchNorm 的运行均值和方差已经计算
    assert bn.running_mean is not None and bn.running_var is not None
    # 调用函数将卷积和 BatchNorm 的权重融合
    fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
        fused_conv.weight,
        fused_conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
        transpose,
    )

    # 返回融合后的卷积模块
    return fused_conv


# 将卷积和 BatchNorm 的权重融合为新的卷积权重和偏置
def fuse_conv_bn_weights(
    conv_w: torch.Tensor,
    conv_b: Optional[torch.Tensor],
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    bn_eps: float,
    bn_w: Optional[torch.Tensor],
    bn_b: Optional[torch.Tensor],
    transpose: bool = False,
) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
    r"""Fuse convolutional module parameters and BatchNorm module parameters into new convolutional module parameters.

    Args:
        conv_w (torch.Tensor): Convolutional weight.
        conv_b (Optional[torch.Tensor]): Convolutional bias.
        bn_rm (torch.Tensor): BatchNorm running mean.
        bn_rv (torch.Tensor): BatchNorm running variance.
        bn_eps (float): BatchNorm epsilon.
        bn_w (Optional[torch.Tensor]): BatchNorm weight.
        bn_b (Optional[torch.Tensor]): BatchNorm bias.
        transpose (bool, optional): If True, transpose the conv weight. Defaults to False.

    Returns:
        Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused convolutional weight and bias.
    """
    # 获取卷积权重和偏置的数据类型
    conv_weight_dtype = conv_w.dtype
    conv_bias_dtype = conv_b.dtype if conv_b is not None else conv_weight_dtype
    # 如果没有给定卷积偏置，创建一个与 BatchNorm 运行均值相同大小的零张量
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    # 如果没有给定 BatchNorm 的权重，创建一个与运行均值相同大小的全一张量
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    # 如果没有给定 BatchNorm 的偏置，创建一个与运行均值相同大小的零张量
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    # 计算 BatchNorm 方差的平方根倒数
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    # 如果需要转置卷积权重
    if transpose:
        # 设置形状，保持第一个维度为 1，其他维度不变
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    # 如果不是第一种情况，计算形状以便广播
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

    # 使用融合后的权重计算融合后的卷积权重
    fused_conv_w = (conv_w * (bn_w * bn_var_rsqrt).reshape(shape)).to(
        dtype=conv_weight_dtype
    )
    # 使用融合后的偏置计算融合后的卷积偏置
    fused_conv_b = ((conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b).to(
        dtype=conv_bias_dtype
    )

    # 返回作为参数的融合后的卷积权重和偏置
    return (
        torch.nn.Parameter(fused_conv_w, conv_w.requires_grad),
        torch.nn.Parameter(fused_conv_b, conv_b.requires_grad),
    )
def fuse_linear_bn_eval(
    linear: LinearT,
    bn: torch.nn.modules.batchnorm._BatchNorm,
) -> LinearT:
    r"""Fuse a linear module and a BatchNorm module into a single, new linear module.

    Args:
        linear (torch.nn.Linear): A Linear module.
        bn (torch.nn.modules.batchnorm._BatchNorm): A BatchNorm module.

    Returns:
        torch.nn.Linear: The fused linear module.

    .. note::
        Both ``linear`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.
    """
    assert not (linear.training or bn.training), "Fusion only for eval!"
    # 创建深层副本以进行融合
    fused_linear = copy.deepcopy(linear)

    """
    Linear-BN需要在保持线性权重/偏置形状的同时进行融合。
    为了保持线性权重/偏置的形状，需要确保 BatchNorm 的通道维度可广播到线性层的最后一个维度上，
    因为 BatchNorm 操作的是通道维度 (N, C_in, H, W)，而线性层操作的是最后一个维度 (*, H_in)。
    要使其可广播，BatchNorm 的特征数和线性层的输出特征数必须满足以下条件：
    1. 它们相等，或者
    2. BatchNorm 的特征数为1
    否则，跳过融合路径
    """
    assert (
        linear.out_features == bn.num_features or bn.num_features == 1
    ), "To fuse, linear.out_features == bn.num_features or bn.num_features == 1"

    assert bn.running_mean is not None and bn.running_var is not None
    # 融合线性层和 BatchNorm 的权重和偏置
    fused_linear.weight, fused_linear.bias = fuse_linear_bn_weights(
        fused_linear.weight,
        fused_linear.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )

    return fused_linear


def fuse_linear_bn_weights(
    linear_w: torch.Tensor,
    linear_b: Optional[torch.Tensor],
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    bn_eps: float,
    bn_w: torch.Tensor,
    bn_b: torch.Tensor,
) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
    r"""Fuse linear module parameters and BatchNorm module parameters into new linear module parameters.

    Args:
        linear_w (torch.Tensor): Linear weight.
        linear_b (Optional[torch.Tensor]): Linear bias.
        bn_rm (torch.Tensor): BatchNorm running mean.
        bn_rv (torch.Tensor): BatchNorm running variance.
        bn_eps (float): BatchNorm epsilon.
        bn_w (torch.Tensor): BatchNorm weight.
        bn_b (torch.Tensor): BatchNorm bias.
        transpose (bool, optional): If True, transpose the conv weight. Defaults to False.

    Returns:
        Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused linear weight and bias.
    """
    # 如果线性层没有偏置，则创建一个与 BatchNorm running mean 形状相同的零张量
    if linear_b is None:
        linear_b = torch.zeros_like(bn_rm)
    # 计算 BatchNorm 的缩放因子
    bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)

    # 计算融合后的权重和偏置
    fused_w = linear_w * bn_scale.unsqueeze(-1)
    fused_b = (linear_b - bn_rm) * bn_scale + bn_b

    return torch.nn.Parameter(fused_w, linear_w.requires_grad), torch.nn.Parameter(
        fused_b, linear_b.requires_grad
    )
```