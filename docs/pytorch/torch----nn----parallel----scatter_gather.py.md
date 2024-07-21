# `.\pytorch\torch\nn\parallel\scatter_gather.py`

```py
# mypy: allow-untyped-defs
# 导入必要的类型和函数
from typing import Any, Dict, List, Optional, overload, Sequence, Tuple, TypeVar, Union
from typing_extensions import deprecated

# 导入PyTorch库
import torch

# 导入本地模块中的函数
from ._functions import Gather, Scatter

# 指定模块中公开的符号列表
__all__ = ["scatter", "scatter_kwargs", "gather"]

# 函数装饰器，标记函数已被弃用
@deprecated(
    "`is_namedtuple` is deprecated, please use the python checks instead",
    category=FutureWarning,
)
def is_namedtuple(obj: Any) -> bool:
    # 检查对象是否是由collections.namedtuple或typing.NamedTuple创建的
    return _is_namedtuple(obj)

# 内部函数，检查对象是否是由collections.namedtuple或typing.NamedTuple创建的
def _is_namedtuple(obj: Any) -> bool:
    # 检查对象是否是元组类型并且具有"_asdict"和"_fields"属性
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )

# 定义一个类型变量T，可以是dict、list或tuple类型的其中一种
T = TypeVar("T", dict, list, tuple)

# 函数重载，根据输入类型的不同返回不同类型的结果
@overload
def scatter(
    inputs: torch.Tensor,
    target_gpus: Sequence[Union[int, torch.device]],
    dim: int = ...,
) -> Tuple[torch.Tensor, ...]:
    ...

@overload
def scatter(
    inputs: T,
    target_gpus: Sequence[Union[int, torch.device]],
    dim: int = ...,
) -> List[T]:
    ...

# 函数定义，将输入数据切片成近似相等的块，并分发到给定的GPU上
def scatter(inputs, target_gpus, dim=0):
    r"""Slice tensors into approximately equal chunks and distributes them across given GPUs.

    Duplicates references to objects that are not tensors.
    """

    # 内部函数，递归地对输入对象进行分发
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for _ in target_gpus]

    # 调用内部函数进行分发操作
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None  # type: ignore[assignment]
    
    return res

# 函数定义，支持关键字参数的分发操作
def scatter_kwargs(
    inputs: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    target_gpus: Sequence[Union[int, torch.device]],
    dim: int = 0,
) -> Tuple[Tuple[Any, ...], Tuple[Dict[str, Any], ...]]:
    r"""Scatter with support for kwargs dictionary."""
    
    # 使用scatter函数分发输入和关键字参数
    scattered_inputs = scatter(inputs, target_gpus, dim) if inputs else []
    scattered_kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    # 如果散列输入列表的长度小于散列关键字参数列表的长度，则执行以下操作
    if len(scattered_inputs) < len(scattered_kwargs):
        # 使用生成器表达式扩展散列输入列表，使其长度与散列关键字参数列表相等
        scattered_inputs.extend(
            () for _ in range(len(scattered_kwargs) - len(scattered_inputs))
        )
    # 否则，如果散列关键字参数列表的长度小于输入参数列表的长度，则执行以下操作
    elif len(scattered_kwargs) < len(inputs):
        # 使用生成器表达式扩展散列关键字参数列表，使其长度与散列输入列表相等
        scattered_kwargs.extend(
            {} for _ in range(len(scattered_inputs) - len(scattered_kwargs))
        )
    # 返回扩展后的散列输入列表和散列关键字参数列表，均转换为元组形式
    return tuple(scattered_inputs), tuple(scattered_kwargs)
# 定义函数 gather，用于从不同的 GPU 收集张量到指定设备上
def gather(outputs: Any, target_device: Union[int, torch.device], dim: int = 0) -> Any:
    r"""Gather tensors from different GPUs on a specified device.

    This function is useful for gathering the results of a distributed computation.
    It takes a sequence of objects, one for each GPU, and returns a single object
    on the specified device.

    Args:
        outputs (Any): A sequence of objects (potentially tensors) to gather.
        target_device (Union[int, torch.device]): The device to gather the tensors to.
            Use 'cpu' for CPU to avoid a deprecation warning.
        dim (int, optional): The dimension along which to gather. Default: 0.

    Returns:
        Any: A gathered object (potentially tensor) on the specified device.
    """

    # 定义内部递归函数 gather_map，用于递归收集并组合输出结果
    def gather_map(outputs):
        # 取第一个输出对象作为参考对象
        out = outputs[0]
        # 如果参考对象是 torch.Tensor 类型，则调用 Gather.apply 收集张量
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        # 如果参考对象为 None，则直接返回 None
        if out is None:
            return None
        # 如果参考对象是 dict 类型
        if isinstance(out, dict):
            # 检查所有 dict 是否具有相同数量的键
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError("All dicts must have the same number of keys")
            # 递归地按键收集 dict 类型的数据
            return type(out)((k, gather_map([d[k] for d in outputs])) for k in out)
        # 如果参考对象是命名元组
        if _is_namedtuple(out):
            # 使用 map 函数递归地处理命名元组
            return type(out)._make(map(gather_map, zip(*outputs)))
        # 对于其他类型的对象，使用 map 函数递归处理
        return type(out)(map(gather_map, zip(*outputs)))

    # 尝试调用 gather_map 函数收集输出结果
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None  # type: ignore[assignment]
    
    # 返回收集后的结果对象
    return res
```