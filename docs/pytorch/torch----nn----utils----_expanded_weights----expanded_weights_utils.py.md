# `.\pytorch\torch\nn\utils\_expanded_weights\expanded_weights_utils.py`

```py
# mypy: allow-untyped-defs
# 导入类型提示模块
from typing import Optional

# 导入 PyTorch 库
import torch

# 导入自定义模块中的 ExpandedWeight 类
from .expanded_weights_impl import ExpandedWeight


def is_batch_first(expanded_args_and_kwargs):
    # 初始化 batch_first 变量为 None
    batch_first = None
    # 遍历扩展参数和关键字参数列表
    for arg in expanded_args_and_kwargs:
        # 如果参数不是 ExpandedWeight 类型，则跳过当前循环
        if not isinstance(arg, ExpandedWeight):
            continue

        # 如果 batch_first 仍为 None，则设置为当前参数的 batch_first 属性
        if not batch_first:
            batch_first = arg.batch_first
        # 否则，如果当前参数的 batch_first 属性与 batch_first 变量不一致，则抛出异常
        elif arg.batch_first != batch_first:
            raise RuntimeError(
                "Got conflicting batch_first arguments in the same layer"
            )
    # 返回最终确定的 batch_first 值（或者 None）
    return batch_first


def standard_kwargs(kwarg_names, expanded_args):
    r"""Separate args and kwargs from `__torch_function__`s that standardize kwargs.

    Most `__torch_function__`s standardize the kwargs that they give, so this will separate
    the args and kwargs they pass. Functions that don't are linear and convND.
    """
    # 提取关键字参数的值列表
    kwarg_values = expanded_args[len(expanded_args) - len(kwarg_names) :]
    # 提取不包含关键字参数的扩展参数列表
    expanded_args_without_kwargs = expanded_args[
        : len(expanded_args) - len(kwarg_names)
    ]
    # 创建关键字参数和其对应值的字典
    expanded_kwargs = dict(zip(kwarg_names, kwarg_values))
    # 返回不包含关键字参数的扩展参数列表和关键字参数字典
    return expanded_args_without_kwargs, expanded_kwargs


def forward_helper(func, expanded_args, expanded_kwargs):
    r"""Compute the forward pass for a function that has expanded weight(s) passed to it.

    It will run the forward pass where all ExpandedWeights are their original
    weight. It runs checks on the given arguments and detaches the outputs.

    .. note:: First argument in :attr:`expanded_args` must be the input with the batch
    dimension as the first element of the shape

    .. note:: :attr:`func` must return a Tensor or tuple of Tensors

    Args:
        func: The function to be called
        expanded_args: Arguments to be passed to :attr:`func`. Will include arguments
          that need to be unpacked because they are ExpandedWeights
        expanded_kwargs: Keyword arguments to be passed to :attr:`func`.
          Similar to :attr:`expanded_args`.
    """
    # 检查和还原扩展参数和关键字参数
    unexpanded_args, unexpanded_kwargs = _check_and_unexpand_args(
        func, expanded_args, expanded_kwargs
    )
    # 调用指定函数并返回其结果
    return func(*unexpanded_args, **unexpanded_kwargs)


def _check_and_unexpand_args(func, expanded_args, expanded_kwargs):
    # 获取第一个传入参数
    input = expanded_args[0]
    # 如果第一个参数是 ExpandedWeight 类型，则抛出异常
    if isinstance(input, ExpandedWeight):
        raise RuntimeError(
            "Expanded Weights do not support inputs that are also ExpandedWeights. "
            f"Input must be a Tensor, got {type(input).__name__} in function {func.__name__}"
        )
    # 如果第一个参数不是 torch.Tensor 类型，则抛出异常
    if not isinstance(input, torch.Tensor):
        raise RuntimeError(
            "Expanded Weights requires a Tensor as the first input to get the batch dimension, "
            f"got {type(input).__name__} in function {func.__name__}"
        )
    # 检查输入张量是否具有维度
    if len(input.shape) == 0:
        # 如果没有批次维度，则抛出异常
        raise RuntimeError(
            f"Expanded Weights requires a batch dimension but got an input of size 0 in function {func.__name__}"
        )
    
    # 检查输入张量的第一维是否为0
    if input.shape[0] == 0:
        # 如果批次大小为0，则抛出异常
        raise RuntimeError(
            "0 is not a valid batch size for Expanded Weights but got input tensor of "
            f"{input} in function {func.__name__}"
        )
    
    # 遍历扩展参数列表和扩展关键字参数的值
    for arg in expanded_args + tuple(expanded_kwargs.values()):
        # 如果参数不是 ExpandedWeight 类型，则继续下一个参数
        if not isinstance(arg, ExpandedWeight):
            continue
        
        # 根据 ExpandedWeight 的设置确定批次大小
        batch_size = input.shape[0] if arg.batch_first else input.shape[1]
        
        # 检查批次大小是否符合预期
        if (arg.allow_smaller_batches and batch_size > arg.batch_size) or (
            not arg.allow_smaller_batches and arg.batch_size != batch_size
        ):
            # 如果不符合预期，则抛出异常
            raise RuntimeError(
                "Expected ExpandedWeights to have batch size matching input but got "
                f"input batch size of {batch_size} with ExpandedWeight of batch size {arg.batch_size}"
            )

    # 初始化损失函数减少方式为 None
    loss_reduction: Optional[str] = None
    
    # 检查扩展参数列表和扩展关键字参数的值，获取损失函数减少方式
    for arg in expanded_args + tuple(expanded_kwargs.values()):
        if isinstance(arg, ExpandedWeight):
            # 如果是 ExpandedWeight，则获取其损失函数减少方式
            if loss_reduction is None:
                loss_reduction = arg.loss_reduction
            elif loss_reduction != arg.loss_reduction:
                # 如果存在不一致的损失函数减少方式，则抛出异常
                raise RuntimeError(
                    "Expected ExpandedWeights to all have the same loss_reduction argument but got one"
                    f"with {loss_reduction} and one with {arg.loss_reduction}"
                )

    # 生成未扩展参数列表，将 ExpandedWeight 替换为其原始权重
    unexpanded_args = tuple(
        arg.orig_weight if isinstance(arg, ExpandedWeight) else arg
        for arg in expanded_args
    )
    
    # 生成未扩展关键字参数字典，将 ExpandedWeight 替换为其原始权重
    unexpanded_kwargs = {
        name: arg.orig_weight if isinstance(arg, ExpandedWeight) else arg
        for (name, arg) in expanded_kwargs.items()
    }
    
    # 返回未扩展的参数和关键字参数
    return unexpanded_args, unexpanded_kwargs
# 如果扩展权重的损失减少方式为 "mean"，则按批次大小缩放梯度样本
def maybe_scale_by_batch_size(grad_sample, expanded_weight):
    if expanded_weight.loss_reduction == "mean":
        return grad_sample * expanded_weight.batch_size
    else:
        return grad_sample

# 如果存在梯度样本函数，则设置梯度样本
def set_grad_sample_if_exists(maybe_expanded_weight, per_sample_grad_fn):
    # 解压扩展权重或张量
    unpacked = unpack_expanded_weight_or_tensor(maybe_expanded_weight)
    if isinstance(maybe_expanded_weight, ExpandedWeight):
        # 计算梯度样本的贡献，可能会按批次大小缩放
        grad_sample_contribution = maybe_scale_by_batch_size(
            per_sample_grad_fn(unpacked), maybe_expanded_weight
        )

        if maybe_expanded_weight.batch_size > grad_sample_contribution.shape[0]:
            # 仅当参数允许更小的批次大小时才通过其他检查
            # 创建与扩展梯度样本贡献相同形状的零张量
            intermediate = torch.zeros(
                maybe_expanded_weight.batch_size,
                *grad_sample_contribution.shape[1:],
                dtype=grad_sample_contribution.dtype,
                device=grad_sample_contribution.device,
            )
            # 将梯度样本贡献复制到零张量中
            intermediate[: grad_sample_contribution.shape[0]] = grad_sample_contribution
            grad_sample_contribution = intermediate

        # 如果已经存在梯度样本属性并且不为 None，则累加梯度样本贡献
        if hasattr(unpacked, "grad_sample") and unpacked.grad_sample is not None:
            unpacked.grad_sample = unpacked.grad_sample + grad_sample_contribution
        else:
            unpacked.grad_sample = grad_sample_contribution

# 解压扩展权重或张量，如果是扩展权重则返回其原始权重，如果是张量则直接返回
def unpack_expanded_weight_or_tensor(maybe_expanded_weight, func=lambda x: x):
    if isinstance(maybe_expanded_weight, ExpandedWeight):
        orig_weight = maybe_expanded_weight.orig_weight
        return func(orig_weight)
    elif (
        isinstance(maybe_expanded_weight, torch.Tensor)
        and not maybe_expanded_weight.requires_grad
    ):
        return func(maybe_expanded_weight)
    elif isinstance(maybe_expanded_weight, torch.Tensor):
        raise RuntimeError(
            "ExpandedWeights currently does not support a mixture of ExpandedWeight parameters "
            "and normal Parameters. Please file and issue with pytorch/pytorch"
        )

# 对输入张量进行求和，忽略第一个维度（批次维度）和最后 n_dims 个维度
def sum_over_all_but_batch_and_last_n(
    tensor: torch.Tensor,
    n_dims: int,
) -> torch.Tensor:
    r"""
    Calculate the sum over all dimensions, except the first (batch dimension), and excluding the last n_dims.

    This function will ignore the first dimension and it will
    not aggregate over the last n_dims dimensions.
    Args:
        tensor: An input tensor of shape ``(B, ..., X[n_dims-1])``.
        n_dims: Number of dimensions to keep.
    Example:
        >>> tensor = torch.ones(1, 2, 3, 4, 5)
        >>> sum_over_all_but_batch_and_last_n(tensor, n_dims=2).shape
        torch.Size([1, 4, 5])
    Returns:
        A tensor of shape ``(B, ..., X[n_dims-1])``
    """
    # 如果张量的维度等于 n_dims+1，则直接返回张量本身
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        # 计算需要进行求和的维度列表
        dims = list(range(1, tensor.dim() - n_dims))
        # 沿指定维度求和
        return tensor.sum(dim=dims)
```