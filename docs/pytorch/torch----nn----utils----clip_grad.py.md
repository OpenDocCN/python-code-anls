# `.\pytorch\torch\nn\utils\clip_grad.py`

```
# 标记该文件允许未类型化的定义
# 导入 functools 库，用于函数式编程工具
import functools
# 从 typing 模块导入类型相关的工具，如字典、迭代器、列表、可选值、元组和联合类型
from typing import cast, Dict, Iterable, List, Optional, Tuple, Union
# 导入 deprecated 扩展，用于标记已弃用的功能
from typing_extensions import deprecated

# 导入 PyTorch 库
import torch
# 从 torch 模块中导入 Tensor 类型
from torch import Tensor
# 从 torch.utils._foreach_utils 模块中导入一些私有函数
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

# 模块内公开的函数列表
__all__ = ["clip_grad_norm_", "clip_grad_norm", "clip_grad_value_"]

# 定义联合类型 _tensor_or_tensors，可以是单个 Tensor 或 Tensor 的迭代器
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

# 装饰器函数 _no_grad，用于禁用梯度计算的上下文管理器包装器
def _no_grad(func):
    """
    This wrapper is needed to avoid a circular import when using @torch.no_grad on the exposed functions
    clip_grad_norm_ and clip_grad_value_ themselves.
    """
    # 实际的包装器函数，应用 torch.no_grad 上下文管理器
    def _no_grad_wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    # 更新包装器函数的元数据，使其与原始函数 func 兼容
    functools.update_wrapper(_no_grad_wrapper, func)
    return _no_grad_wrapper

# 用 @_no_grad 装饰的函数 clip_grad_norm_
@_no_grad
def clip_grad_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    r"""Clip the gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    # 如果 parameters 是单个 Tensor，则转换为列表
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    # 收集所有非空梯度的参数列表
    grads = [p.grad for p in parameters if p.grad is not None]
    # 将 max_norm 和 norm_type 转换为浮点数
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    # 如果没有梯度可用，返回 0.0 的 Tensor
    if len(grads) == 0:
        return torch.tensor(0.0)
    # 获取第一个梯度的设备信息
    first_device = grads[0].device
    # 根据设备和数据类型分组梯度张量，返回一个字典
    grouped_grads: Dict[
        Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]
    ] = _group_tensors_by_device_and_dtype(
        [grads]
    )  # type: ignore[assignment]

    # 初始化空列表 norms，用于存储每组梯度的范数
    norms: List[Tensor] = []
    # 遍历 grouped_grads 字典中的每个项，每个项的结构为 ((device, _), ([device_grads], _))
    # type: ignore[assignment] 表示忽略类型检查中的赋值问题
    for (device, _), ([device_grads], _) in grouped_grads.items():
        # 检查是否应该使用 foreach API 处理梯度，根据条件决定是否扩展 norms 列表
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_grads, norm_type))
        # 如果 foreach=True 但当前设备不支持 foreach API，则抛出运行时错误
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        # 否则，对每个设备梯度计算其向量范数，并加入 norms 列表
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    # 计算 norms 列表中所有元素的向量范数，转换为首个设备上的张量
    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    # 如果启用了 error_if_nonfinite 并且 total_norm 是 NaN 或 Inf，则抛出运行时错误
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    
    # 计算剪切系数 clip_coef，用于梯度剪裁，避免除零错误并保证 clip_coef 不小于 1e-6
    clip_coef = max_norm / (total_norm + 1e-6)
    
    # 注释：当 clip_coef 被夹在 1.0 时，乘以被夹住的 coef 是多余的，但这样做避免了 `if clip_coef < 1:` 的条件判断，
    # 这种判断可能需要在梯度不驻留在 CPU 内存时进行 CPU <=> 设备的同步。
    # 对 clip_coef 进行夹紧处理，确保其不超过 1.0
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    # 再次遍历 grouped_grads 字典中的每个项，对每个设备的梯度乘以 clip_coef_clamped
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        # 检查是否应该使用 foreach API 处理梯度，根据条件决定是否扩展 norms 列表
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            # 使用 foreach API 对设备的梯度乘以 clip_coef_clamped，并转换到当前设备
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        # 如果 foreach=True 但当前设备不支持 foreach API，则抛出运行时错误
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        # 否则，逐个计算每个设备梯度并乘以 clip_coef_clamped
        else:
            # 将 clip_coef_clamped 转换到当前设备
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            # 对每个设备的梯度执行原位乘法操作
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)

    # 返回总体梯度范数 total_norm
    return total_norm
# 标记函数为已弃用，并提供替代的函数信息
@deprecated(
    "`torch.nn.utils.clip_grad_norm` is now deprecated "
    "in favor of `torch.nn.utils.clip_grad_norm_`.",
    category=FutureWarning,
)
# 定义函数：对参数列表中的梯度进行裁剪，限制梯度范数不超过指定的最大范数
def clip_grad_norm(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    r"""Clip the gradient norm of an iterable of parameters.

    .. warning::
        This method is now deprecated in favor of
        :func:`torch.nn.utils.clip_grad_norm_`.
    """
    # 调用替代函数 clip_grad_norm_，并返回其结果
    return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)


# 标记函数为无梯度计算上下文，即不需要计算梯度的操作
@_no_grad
# 定义函数：对参数列表中的梯度进行裁剪，限制梯度值不超过指定的裁剪值
def clip_grad_value_(
    parameters: _tensor_or_tensors,
    clip_value: float,
    foreach: Optional[bool] = None,
) -> None:
    r"""Clip the gradients of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        foreach (bool): use the faster foreach-based implementation
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and
            silently fall back to the slow implementation for other device types.
            Default: ``None``
    """
    # 如果 parameters 是单个 Tensor，则转换为列表形式
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    # 将 clip_value 转换为浮点数
    clip_value = float(clip_value)

    # 收集所有参数的梯度，仅保留非空梯度
    grads = [p.grad for p in parameters if p.grad is not None]
    # 根据设备和数据类型对梯度进行分组
    grouped_grads = _group_tensors_by_device_and_dtype([grads])

    # 遍历分组后的梯度
    for (device, _), ([grads], _) in grouped_grads.items():  # type: ignore[assignment]
        # 如果 foreach 参数为 None，并且当前设备支持 foreach API，则使用 foreach 实现快速裁剪
        if (
            foreach is None
            and _has_foreach_support(cast(List[Tensor], grads), device=device)
        ) or (foreach and _device_has_foreach_support(device)):
            # 使用 foreach API 对梯度进行裁剪，限制在 [-clip_value, clip_value] 范围内
            torch._foreach_clamp_min_(cast(List[Tensor], grads), -clip_value)
            torch._foreach_clamp_max_(cast(List[Tensor], grads), clip_value)
        # 如果 foreach 参数为 True，但当前设备不支持 foreach API，则抛出运行时错误
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        # 如果 foreach 参数为 False，则使用普通的循环实现裁剪
        else:
            for grad in grads:
                cast(Tensor, grad).clamp_(min=-clip_value, max=clip_value)
```