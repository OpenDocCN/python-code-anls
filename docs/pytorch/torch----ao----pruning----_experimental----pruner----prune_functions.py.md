# `.\pytorch\torch\ao\pruning\_experimental\pruner\prune_functions.py`

```py
"""
Collection of conversion functions for linear / conv2d structured pruning
Also contains utilities for bias propagation
"""
# 导入所需的类型注解和函数
from typing import cast, List, Optional, Callable, Tuple

# 导入 PyTorch 相关模块和类
import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
# 导入自定义模块
from .parametrization import FakeStructuredSparsity, BiasHook

# BIAS PROPAGATION
# 从模块中移除所有与偏置相关的处理函数
def _remove_bias_handles(module: nn.Module) -> None:
    # 如果模块具有 _forward_hooks 属性
    if hasattr(module, "_forward_hooks"):
        # 初始化存储偏置钩子的列表
        bias_hooks: List[int] = []
        # 遍历模块的前向钩子
        for key, hook in module._forward_hooks.items():
            # 如果钩子是 BiasHook 的实例，则添加到列表中
            if isinstance(hook, BiasHook):
                bias_hooks.append(key)

        # 删除所有偏置钩子
        for key in bias_hooks:
            del module._forward_hooks[key]

# 获取调整后的下一层的偏置参数
def _get_adjusted_next_layer_bias(
    next_layer: nn.Module, pruned_biases: Tensor, mask: Tensor
) -> nn.Parameter:
    r"""Returns new adjusted bias for the second supported module"""
    # 如果下一层是参数化的
    if parametrize.is_parametrized(next_layer):
        # 访问原始权重参数
        parametrization_dict = cast(nn.ModuleDict, next_layer.parametrizations)
        weight_parameterizations = cast(
            ParametrizationList, parametrization_dict.weight
        )
        next_weight = weight_parameterizations.original
    else:
        next_weight = cast(Tensor, next_layer.weight)

    # 计算缩放后的权重
    scaling_weight = next_weight[:, ~mask]

    # 如果下一层是卷积层
    if isinstance(next_layer, nn.Conv2d):
        # 计算缩放后的偏置
        scaling_product = torch.matmul(
            pruned_biases.reshape(1, -1), torch.transpose(scaling_weight, 1, 2)
        )
        sum_range = list(range(len(scaling_product.shape)))[
            1:
        ]  # all but the first dimension
        scaled_biases = torch.sum(scaling_product, sum_range)
    # 如果下一层是全连接层
    elif isinstance(next_layer, nn.Linear):
        # 计算缩放后的偏置
        scaled_biases = torch.matmul(
            pruned_biases, torch.transpose(scaling_weight, 0, 1)
        )
    else:
        # 抛出未实现的错误
        raise NotImplementedError(f"Type {type(next_layer)} not supported yet.")

    # 如果下一层是参数化的并且具有原始偏置 ._bias
    if (
        parametrize.is_parametrized(next_layer)
        and getattr(next_layer, "_bias", None) is not None
    ):
        # 创建调整后的偏置参数
        adjusted_bias = nn.Parameter(scaled_biases + next_layer._bias)
    # 如果下一层不是参数化的并且具有 .bias
    elif (
        not parametrize.is_parametrized(next_layer) and next_layer.bias is not None
    ):
        # 创建调整后的偏置参数
        adjusted_bias = nn.Parameter(scaled_biases + next_layer.bias)
    else:  # 如果 next_layer 没有偏置项
        adjusted_bias = nn.Parameter(scaled_biases)
    return adjusted_bias
def _prune_module_bias(module: nn.Module, mask: Tensor) -> None:
    r"""Applies mask to given modules bias"""
    # 获取原始偏置值，如果存在则使用，否则使用模块自带的偏置
    original_bias = cast(Tensor, getattr(module, "_bias", module.bias))
    if original_bias is not None:
        # 根据掩码剪枝偏置值
        module.bias = nn.Parameter(original_bias[mask])

    # 删除 _bias 属性
    if hasattr(module, "_bias"):
        delattr(module, "_bias")


def _propagate_module_bias(module: nn.Module, mask: Tensor) -> Optional[Tensor]:
    r"""
    In the case that we need to propagate biases, this function will return the biases we need
    """
    # 设置当前模块的偏置值
    if module.bias is not None:
        module.bias = nn.Parameter(cast(Tensor, module.bias)[mask])
    elif getattr(module, "_bias", None) is not None:
        module.bias = nn.Parameter(cast(Tensor, module._bias)[mask])

    # 获取剪枝后需要传播到下一层的偏置值
    if getattr(module, "_bias", None) is not None:
        pruned_biases = cast(Tensor, module._bias)[~mask]
    else:
        pruned_biases = None

    # 删除 _bias 属性
    if hasattr(module, "_bias"):
        delattr(module, "_bias")

    return pruned_biases


# LINEAR
def _prune_linear_helper(linear: nn.Linear) -> Tensor:
    # expects linear to be a parameterized linear module
    parametrization_dict = cast(nn.ModuleDict, linear.parametrizations)
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    for p in weight_parameterizations:
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)

    with torch.no_grad():
        # 移除权重的参数化设置，保留参数化后的权重值
        parametrize.remove_parametrizations(linear, "weight", leave_parametrized=True)
        linear.weight = nn.Parameter(linear.weight[mask])  # type: ignore[possibly-undefined]
    linear.out_features = linear.weight.shape[0]
    _remove_bias_handles(linear)

    return mask


def prune_linear(linear: nn.Linear) -> None:
    # 获得剪枝掩码并应用到线性层的权重和偏置
    mask = _prune_linear_helper(linear)
    if getattr(linear, "prune_bias", False):
        _prune_module_bias(linear, mask)


def prune_linear_linear(linear1: nn.Linear, linear2: nn.Linear) -> None:
    # 对第一个线性层进行剪枝，并将剪枝后的偏置传播到第二个线性层
    prune_linear_activation_linear(linear1, None, linear2)


def prune_linear_activation_linear(
    linear1: nn.Linear,
    activation: Optional[Callable[[Tensor], Tensor]],
    linear2: nn.Linear,
):
    # 获得剪枝掩码并应用到第一个线性层的权重和偏置
    mask = _prune_linear_helper(linear1)
    if getattr(linear1, "prune_bias", False):
        _prune_module_bias(linear1, mask)
    else:
        # 获取剪枝后需要传播到下一层的偏置，并根据激活函数调整这些偏置
        pruned_biases = _propagate_module_bias(linear1, mask)
        if pruned_biases is not None:
            if activation:
                pruned_biases = activation(pruned_biases)
            linear2.bias = _get_adjusted_next_layer_bias(linear2, pruned_biases, mask)
    # 使用 `torch.no_grad()` 上下文管理器，确保在此范围内不计算梯度
    with torch.no_grad():
        # 检查 `linear2` 是否被参数化
        if parametrize.is_parametrized(linear2):
            # 将 `linear2` 的参数化字典转换为 `nn.ModuleDict` 类型
            parametrization_dict = cast(nn.ModuleDict, linear2.parametrizations)
            # 获取权重参数化列表，并将其转换为 `ParametrizationList` 类型
            weight_parameterizations = cast(
                ParametrizationList, parametrization_dict.weight
            )
    
            # 使用 `mask` 切片原始权重参数化，将结果设为新的参数
            weight_parameterizations.original = nn.Parameter(
                weight_parameterizations.original[:, mask]
            )
            # 更新 `linear2` 的输入特征数为新权重的形状的第二个维度大小
            linear2.in_features = weight_parameterizations.original.shape[1]
        else:
            # 如果 `linear2` 没有被参数化，直接使用 `mask` 切片权重，设为新的参数
            linear2.weight = nn.Parameter(linear2.weight[:, mask])
            # 更新 `linear2` 的输入特征数为新权重的第二个维度大小
            linear2.in_features = linear2.weight.shape[1]
# CONV2D
def _prune_conv2d_helper(conv2d: nn.Conv2d) -> Tensor:
    # 获取卷积层的参数化字典
    parametrization_dict = cast(nn.ModuleDict, conv2d.parametrizations)
    # 获取权重参数化列表
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    # 遍历权重参数化列表
    for p in weight_parameterizations:
        # 如果是 FakeStructuredSparsity 参数化类型，则获取掩码
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)

    # 使用 torch.no_grad() 禁用梯度计算上下文
    with torch.no_grad():
        # 移除卷积层的权重参数化，保持参数化状态
        parametrize.remove_parametrizations(conv2d, "weight", leave_parametrized=True)
        # 根据掩码选择部分权重形成新的权重参数
        conv2d.weight = nn.Parameter(conv2d.weight[mask])  # type: ignore[possibly-undefined]
    # 更新卷积层的输出通道数为新权重的形状
    conv2d.out_channels = conv2d.weight.shape[0]

    # 移除偏置处理
    _remove_bias_handles(conv2d)
    # 返回掩码，用于进一步操作
    return mask


def prune_conv2d_padded(conv2d_1: nn.Conv2d) -> None:
    # 获取卷积层的参数化字典
    parametrization_dict = cast(nn.ModuleDict, conv2d_1.parametrizations)
    # 获取权重参数化列表
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    # 遍历权重参数化列表
    for p in weight_parameterizations:
        # 如果是 FakeStructuredSparsity 参数化类型，则获取掩码
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)

    # 使用 torch.no_grad() 禁用梯度计算上下文
    with torch.no_grad():
        # 移除卷积层的权重参数化，保持参数化状态
        parametrize.remove_parametrizations(conv2d_1, "weight", leave_parametrized=True)

    # 如果卷积层具有原始偏置
    if getattr(conv2d_1, "_bias", None) is not None:
        # 如果卷积层有偏置且原始偏置不为空
        if conv2d_1.bias is not None:
            # 创建一个新的偏置张量，其中保留掩码选择的部分偏置
            new_bias = torch.zeros(conv2d_1.bias.shape)
            new_bias[mask] = conv2d_1.bias[mask]  # type: ignore[possibly-undefined]
            # 在新的偏置中保留通过前一层传播的偏置
            new_bias[~mask] = cast(Tensor, conv2d_1._bias)[~mask]
            # 将新偏置设置为卷积层的参数
            conv2d_1.bias = nn.Parameter(new_bias)
        else:
            # 如果卷积层只有原始偏置，则将其设置为卷积层的参数
            conv2d_1.bias = nn.Parameter(cast(Tensor, conv2d_1._bias))
    else:
        # 如果没有原始偏置，但有通过前一层传播的偏置
        if conv2d_1.bias is not None:
            # 将通过前一层传播的偏置的非掩码部分设为零
            conv2d_1.bias.data[~mask] = 0  # type: ignore[possibly-undefined]

    # 如果卷积层具有 "_bias" 属性，则删除该属性
    if hasattr(conv2d_1, "_bias"):
        delattr(conv2d_1, "_bias")


def prune_conv2d(conv2d: nn.Conv2d) -> None:
    # 调用辅助函数获取掩码
    mask = _prune_conv2d_helper(conv2d)
    # 如果卷积层设置了 prune_bias 标志，则调整偏置
    if getattr(conv2d, "prune_bias", False):
        _prune_module_bias(conv2d, mask)


def prune_conv2d_conv2d(conv2d_1: nn.Conv2d, conv2d_2: nn.Conv2d) -> None:
    # 调用函数处理卷积层和激活函数之间的剪枝模式
    prune_conv2d_activation_conv2d(conv2d_1, None, conv2d_2)


def prune_conv2d_activation_conv2d(
    conv2d_1: nn.Conv2d,
    activation: Optional[Callable[[Tensor], Tensor]],
    conv2d_2: nn.Conv2d,
):
    r"""
    Fusion Pattern for conv2d -> some activation module / function -> conv2d layers
    """
    # 获取卷积层的参数化字典
    parametrization_dict = cast(nn.ModuleDict, conv2d_1.parametrizations)
    # 获取权重参数化列表
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    # 遍历权重参数化列表
    for p in weight_parameterizations:
        # 如果是 FakeStructuredSparsity 参数化类型，则获取掩码
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)
    # 获取 conv2d_1 的 prune_bias 属性，如果没有则默认为 False
    prune_bias = getattr(conv2d_1, "prune_bias", False)
    
    # 检查 conv2d_2 是否具有 padding 属性，并且 padding 大于 (0, 0)，以及 conv2d_1 是否具有偏置或者 _bias 属性
    if (
        hasattr(conv2d_2, "padding")
        and cast(Tuple[int], conv2d_2.padding) > (0, 0)
        and (conv2d_1.bias is not None or getattr(conv2d_1, "_bias", None) is not None)
    ):
        # 如果满足条件，对 conv2d_1 进行带填充的裁剪
        prune_conv2d_padded(conv2d_1)
    else:
        # 否则，使用 _prune_conv2d_helper 对 conv2d_1 进行裁剪，并获取掩码 mask
        mask = _prune_conv2d_helper(conv2d_1)
        
        # 如果 prune_bias 为 True，则对 conv2d_1 的偏置进行裁剪
        if prune_bias:
            _prune_module_bias(conv2d_1, mask)
        else:
            # 否则，根据 mask 传播裁剪后的模块偏置，并可能应用激活函数 activation
            pruned_biases = _propagate_module_bias(conv2d_1, mask)
            if pruned_biases is not None:
                if activation:
                    pruned_biases = activation(pruned_biases)
                # 调整 conv2d_2 的偏置，以适应裁剪后的偏置
                conv2d_2.bias = _get_adjusted_next_layer_bias(
                    conv2d_2, pruned_biases, mask
                )

        # 如果 conv2d_2 没有 padding 属性或者 padding 小于等于 (0, 0)，或者 conv2d_1 没有偏置
        if (
            not (
                hasattr(conv2d_2, "padding")
                and cast(Tuple[int], conv2d_2.padding) > (0, 0)
            )
            or conv2d_1.bias is None
        ):
            # 使用 torch.no_grad() 上下文
            with torch.no_grad():
                # 如果 conv2d_2 是参数化的模块，则调整其权重参数化
                if parametrize.is_parametrized(conv2d_2):
                    parametrization_dict = cast(
                        nn.ModuleDict, conv2d_2.parametrizations
                    )
                    weight_parameterizations = cast(
                        ParametrizationList, parametrization_dict.weight
                    )
                    # 裁剪后的权重参数化
                    weight_parameterizations.original = nn.Parameter(
                        weight_parameterizations.original[:, mask]
                    )
                    # 更新 conv2d_2 的输入通道数
                    conv2d_2.in_channels = weight_parameterizations.original.shape[1]
                else:
                    # 否则，直接裁剪 conv2d_2 的权重，并更新其输入通道数
                    conv2d_2.weight = nn.Parameter(conv2d_2.weight[:, mask])
                    conv2d_2.in_channels = conv2d_2.weight.shape[1]
# 对两个 Conv2d 层和一个池化层之间的激活函数进行剪枝
def prune_conv2d_pool_activation_conv2d(
    c1: nn.Conv2d,
    pool: nn.Module,
    activation: Optional[Callable[[Tensor], Tensor]],
    c2: nn.Conv2d,
) -> None:
    # 调用剪枝函数，对第一个 Conv2d 层和第二个 Conv2d 层之间的激活函数进行剪枝
    prune_conv2d_activation_conv2d(c1, activation, c2)


# 对一个 Conv2d 层、一个激活函数和一个池化层之间的 Conv2d 层进行剪枝
def prune_conv2d_activation_pool_conv2d(
    c1: nn.Conv2d,
    activation: Optional[Callable[[Tensor], Tensor]],
    pool: nn.Module,
    c2: nn.Conv2d,
) -> None:
    # 调用剪枝函数，对第一个 Conv2d 层、激活函数和池化层之间的第二个 Conv2d 层进行剪枝
    prune_conv2d_activation_conv2d(c1, activation, c2)


# 对一个 Conv2d 层、一个池化层、一个展平函数和一个线性层之间的 Conv2d 层进行剪枝
def prune_conv2d_pool_flatten_linear(
    conv2d: nn.Conv2d,
    pool: nn.Module,
    flatten: Optional[Callable[[Tensor], Tensor]],
    linear: nn.Linear,
) -> None:
    # 获取 Conv2d 层的剪枝掩码
    mask = _prune_conv2d_helper(conv2d)

    # 将 Conv2d 层的剪枝索引映射到展平层后的线性层的索引
    # 确定展平比例 (h * w)，重新调整 `first_pruned_indices`（每个索引映射到范围 idx * h * w 到 (idx+1) * h * w）、
    # `first_valid_indices` 和 `pruned_biases`（每个偏置重复 h * w 次）
    if parametrize.is_parametrized(linear):
        parametrization_dict = cast(nn.ModuleDict, linear.parametrizations)
        weight_parameterizations = cast(
            ParametrizationList, parametrization_dict.weight
        )
        linear_ic = weight_parameterizations.original.shape[1]
    else:
        linear_ic = linear.weight.shape[1]

    conv2d_oc = len(mask)
    assert (
        linear_ic % conv2d_oc == 0
    ), f"Flattening from dimensions {conv2d_oc} to {linear_ic} not supported"

    flatten_scale = linear_ic // conv2d_oc
    # 创建展平后的掩码张量，用于线性层
    flattened_mask = torch.tensor(
        [[val] * flatten_scale for val in mask], dtype=torch.bool, device=mask.device
    ).flatten()

    if getattr(conv2d, "prune_bias", False):
        # 如果 Conv2d 层需要剪枝偏置，则调用对应的剪枝函数
        _prune_module_bias(conv2d, mask)
    else:
        # 否则，推广模块偏置并调整到下一层的偏置
        pruned_biases = cast(Tensor, _propagate_module_bias(conv2d, mask))
        flattened_pruned_biases = torch.tensor(
            [[bias] * flatten_scale for bias in pruned_biases], device=mask.device
        ).flatten()
        linear.bias = _get_adjusted_next_layer_bias(
            linear, flattened_pruned_biases, flattened_mask
        )

    # 使用 torch.no_grad() 上下文管理器更新参数化后的线性层权重
    with torch.no_grad():
        if parametrize.is_parametrized(linear):
            parametrization_dict = cast(nn.ModuleDict, linear.parametrizations)
            weight_parameterizations = cast(
                ParametrizationList, parametrization_dict.weight
            )
            weight_parameterizations.original = nn.Parameter(
                weight_parameterizations.original[:, flattened_mask]
            )
            linear.in_features = weight_parameterizations.original.shape[1]
        else:
            linear.weight = nn.Parameter(linear.weight[:, flattened_mask])
            linear.in_features = linear.weight.shape[1]


# 对一个 LSTM 层、一个获取函数和一个线性层之间的 LSTM 输出进行剪枝
def prune_lstm_output_linear(
    lstm: nn.LSTM, getitem: Callable, linear: nn.Linear
) -> None:
    # 调用剪枝函数，对 LSTM 层输出和线性层之间的连接进行剪枝
    prune_lstm_output_layernorm_linear(lstm, getitem, None, linear)


# 对一个 LSTM 层、一个获取函数、一个归一化层和一个线性层之间的 LSTM 输出进行剪枝
def prune_lstm_output_layernorm_linear(
    lstm: nn.LSTM,
    getitem: Callable,
    # getitem 是一个类型为 Callable 的变量，用于表示一个可调用对象（函数、方法等）
    layernorm: Optional[nn.LayerNorm],
    # layernorm 是一个类型为 Optional[nn.LayerNorm] 的变量，表示一个可选的 nn.LayerNorm 类型对象
    linear: nn.Linear,
    # linear 是一个类型为 nn.Linear 的变量，表示一个 nn.Linear 对象，通常用于神经网络中的线性层
) -> None:
```