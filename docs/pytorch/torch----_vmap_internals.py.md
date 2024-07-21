# `.\pytorch\torch\_vmap_internals.py`

```py
# mypy: allow-untyped-defs
# 引入 functools 库，用于处理函数相关的工具函数
import functools
# 引入 typing 库中的类型定义
from typing import Any, Callable, List, Optional, Tuple, Union
# 引入 typing_extensions 库中的 deprecated 类型
from typing_extensions import deprecated

# 引入 PyTorch 库
import torch
# 从 torch 库中引入 Tensor 类型
from torch import Tensor
# 从 torch.utils._pytree 中引入相关函数
from torch.utils._pytree import _broadcast_to_and_flatten, tree_flatten, tree_unflatten

# 定义类型别名，用于输入维度的表示
in_dims_t = Union[int, Tuple]
# 定义类型别名，用于输出维度的表示
out_dims_t = Union[int, Tuple[int, ...]]


# 检查所有待批处理的参数是否具有相同的批次维度大小
def _validate_and_get_batch_size(
    flat_in_dims: List[Optional[int]],
    flat_args: List,
) -> int:
    # 获取每个参数在指定维度上的大小
    batch_sizes = [
        arg.size(in_dim)
        for in_dim, arg in zip(flat_in_dims, flat_args)
        if in_dim is not None
    ]
    # 如果存在不同的大小，则抛出 ValueError 异常
    if batch_sizes and any(size != batch_sizes[0] for size in batch_sizes):
        raise ValueError(
            f"vmap: Expected all tensors to have the same size in the mapped "
            f"dimension, got sizes {batch_sizes} for the mapped dimension"
        )
    # 返回批次大小
    return batch_sizes[0]


# 获取批量输出的数量
def _num_outputs(batched_outputs: Union[Tensor, Tuple[Tensor, ...]]) -> int:
    # 如果输出是一个元组，则返回其长度
    if isinstance(batched_outputs, tuple):
        return len(batched_outputs)
    # 否则返回 1
    return 1


# 将 value 转换为元组，如果 value 是元组且长度与 num_elements 不符则抛出异常
# 如果 value 不是元组，则重复 value num_elements 次作为元组返回
def _as_tuple(
    value: Any,
    num_elements: int,
    error_message_lambda: Callable[[], str],
) -> Tuple:
    if not isinstance(value, tuple):
        return (value,) * num_elements
    if len(value) != num_elements:
        raise ValueError(error_message_lambda())
    return value


# 为每个需要批处理的 Tensor 创建批处理张量
# 返回可能已经批处理的参数及其批次大小
def _create_batched_inputs(
    in_dims: in_dims_t,
    args: Tuple,
    vmap_level: int,
    func: Callable,
) -> Tuple[Tuple, int]:
    # 检查 in_dims 是否为 int 或 tuple 类型，否则抛出异常
    if not isinstance(in_dims, int) and not isinstance(in_dims, tuple):
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            f"expected `in_dims` to be int or a (potentially nested) tuple "
            f"matching the structure of inputs, got: {type(in_dims)}."
        )
    # 检查参数是否为空，如果为空则抛出异常
    if len(args) == 0:
        raise ValueError(
            f"vmap({_get_name(func)})(<inputs>): got no inputs. Maybe you forgot to add "
            f"inputs, or you are trying to vmap over a function with no inputs. "
            f"The latter is unsupported."
        )

    # 将输入参数扁平化并获取其结构描述
    flat_args, args_spec = tree_flatten(args)
    # 使用输入维度信息对参数进行广播并扁平化处理
    flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
    # 如果扁平化后的维度信息为 None，则抛出异常
    if flat_in_dims is None:
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            f"in_dims is not compatible with the structure of `inputs`. "
            f"in_dims has structure {tree_flatten(in_dims)[1]} but inputs "
            f"has structure {args_spec}."
        )
    # 遍历输入参数和对应的输入维度
    for arg, in_dim in zip(flat_args, flat_in_dims):
        # 如果输入维度不是整数且不为None，抛出数值错误异常
        if not isinstance(in_dim, int) and in_dim is not None:
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input but in_dim must be either "
                f"an integer dimension or None."
            )
        # 如果输入维度是整数但输入参数不是Tensor类型，抛出数值错误异常
        if isinstance(in_dim, int) and not isinstance(arg, Tensor):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input but the input is of type "
                f"{type(arg)}. We cannot vmap over non-Tensor arguments, "
                f"please use None as the respective in_dim"
            )
        # 如果输入维度不为None且不满足0 <= in_dim < arg.dim()，抛出数值错误异常
        if in_dim is not None and (in_dim < 0 or in_dim >= arg.dim()):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for some input, but that input is a Tensor "
                f"of dimensionality {arg.dim()} so expected in_dim to satisfy "
                f"0 <= in_dim < {arg.dim()}."
            )

    # 验证并获取批处理大小
    batch_size = _validate_and_get_batch_size(flat_in_dims, flat_args)
    # 查看注释 [Ignored _remove_batch_dim, _add_batch_dim]
    # 根据输入维度情况，对输入参数进行批处理转换
    batched_inputs = [
        arg if in_dim is None else torch._add_batch_dim(arg, in_dim, vmap_level)
        for in_dim, arg in zip(flat_in_dims, flat_args)
    ]
    # 使用树解构函数将批处理后的输入参数解构为原始输入参数结构，同时返回批处理大小
    return tree_unflatten(batched_inputs, args_spec), batch_size
# 取消与 `vmap_level` 相关的分批操作（以及任何批次维度）。
# 函数签名：
# _unwrap_batched(
#     batched_outputs: Union[Tensor, Tuple[Tensor, ...]],
#     out_dims: out_dims_t,
#     vmap_level: int,
#     batch_size: int,
#     func: Callable,
#     allow_none_pass_through: bool = False,
# ) -> Tuple
def _unwrap_batched(
    batched_outputs: Union[Tensor, Tuple[Tensor, ...]],  # 输入参数：批次输出，可以是单个张量或张量元组
    out_dims: out_dims_t,  # 输入参数：输出维度描述
    vmap_level: int,  # 输入参数：vmap 层级
    batch_size: int,  # 输入参数：批次大小
    func: Callable,  # 输入参数：函数对象
    allow_none_pass_through: bool = False,  # 输入参数：是否允许 None 通过
) -> Tuple:
    num_outputs = _num_outputs(batched_outputs)  # 计算批次输出的数量
    out_dims_as_tuple = _as_tuple(
        out_dims,
        num_outputs,
        lambda: f"vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must "
        f"have one dim per output (got {num_outputs} outputs) of {_get_name(func)}.",
    )  # 转换输出维度描述为元组形式

    # NOTE [Ignored _remove_batch_dim, _add_batch_dim]
    # There is something wrong with our type bindings for functions that begin
    # with '_', see #40397.
    if isinstance(batched_outputs, Tensor):  # 如果批次输出是张量
        out_dim = out_dims_as_tuple[0]  # 获取第一个输出维度描述
        return torch._remove_batch_dim(batched_outputs, vmap_level, batch_size, out_dim)  # 移除批次维度并返回结果张量  # type: ignore[return-value]
    if allow_none_pass_through:  # 如果允许 None 通过
        return tuple(
            (
                torch._remove_batch_dim(out, vmap_level, batch_size, out_dim)
                if out is not None
                else None
            )
            for out, out_dim in zip(batched_outputs, out_dims_as_tuple)
        )  # 对于每个批次输出，移除批次维度并返回结果元组
    else:
        return tuple(
            torch._remove_batch_dim(out, vmap_level, batch_size, out_dim)
            for out, out_dim in zip(batched_outputs, out_dims_as_tuple)
        )  # 对于每个批次输出，移除批次维度并返回结果元组


# 检查 `fn` 返回的是一个或多个张量，而不是其他类型。
# 注意：一个返回多个参数的 Python 函数会返回一个单独的元组，
# 因此我们实际上在检查 `outputs` 是否是单个张量或张量元组。
# 函数签名：
# _validate_outputs(outputs: Any, func: Callable) -> None
def _validate_outputs(outputs: Any, func: Callable) -> None:
    if isinstance(outputs, Tensor):  # 如果输出是张量，则通过检查
        return
    if not isinstance(outputs, tuple):  # 如果输出不是元组
        raise ValueError(
            f"vmap({_get_name(func)}, ...): `{_get_name(func)}` must only return "
            f"Tensors, got type {type(outputs)} as the return."
        )  # 抛出值错误，指出函数返回类型错误
    for idx, output in enumerate(outputs):  # 遍历输出元组中的每个元素
        if isinstance(output, Tensor):  # 如果输出是张量，则继续检查下一个
            continue
        raise ValueError(
            f"vmap({_get_name(func)}, ...): `{_get_name(func)}` must only return "
            f"Tensors, got type {type(output)} for return {idx}."
        )  # 抛出值错误，指出函数返回类型错误


# 检查 `out_dims` 是否为整数或整数元组。
# 函数签名：
# _check_out_dims_is_int_or_int_tuple(out_dims: out_dims_t, func: Callable) -> None
def _check_out_dims_is_int_or_int_tuple(out_dims: out_dims_t, func: Callable) -> None:
    if isinstance(out_dims, int):  # 如果输出维度描述是整数，则通过检查
        return
    if not isinstance(out_dims, tuple) or not all(
        isinstance(out_dim, int) for out_dim in out_dims
    ):  # 如果输出维度描述不是元组或者元组中有非整数元素
        raise ValueError(
            f"vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must be "
            f"an int or a tuple of int representing where in the outputs the "
            f"vmapped dimension should appear."
        )  # 抛出值错误，指出输出维度描述类型错误


# 获取函数的名称。
# 函数签名：
# _get_name(func: Callable)
def _get_name(func: Callable):
    if hasattr(func, "__name__"):  # 如果函数对象有 '__name__' 属性
        return func.__name__  # 返回函数名称
    # 返回给定函数对象的可打印字符串表示形式
    # repr() 函数返回一个对象的字符串表示形式，该字符串可以通过 eval() 函数重新得到该对象
    return repr(func)
# 使用 @deprecated 装饰器标记函数，提示用户该函数已过时，请使用 torch.vmap 替代。
# 参数 "category" 表示警告的类别为 FutureWarning。
@deprecated(
    "Please use `torch.vmap` instead of `torch._vmap_internals.vmap`.",
    category=FutureWarning,
)
# 定义函数 vmap，接受一个可调用对象 func 和两个可选的参数 in_dims 和 out_dims，
# 返回一个新的可调用对象。
def vmap(func: Callable, in_dims: in_dims_t = 0, out_dims: out_dims_t = 0) -> Callable:
    """
    Please use torch.vmap instead of this API.
    """
    # 调用 _vmap 函数，返回其结果。
    return _vmap(func, in_dims, out_dims)


# 定义函数 _vmap，接受一个可调用对象 func 和多个可选的参数，
# 返回一个新的可调用对象。
def _vmap(
    func: Callable,
    in_dims: in_dims_t = 0,
    out_dims: out_dims_t = 0,
    allow_none_pass_through: bool = False,
) -> Callable:
    # 参数 allow_none_pass_through 是一个临时解决方案，用于处理在 autograd 引擎中，
    # 如果输入的任何一个未使用，则可能返回 None 的情况。
    # 参见讨论此问题的 GitHub issue: https://github.com/facebookresearch/functorch/issues/159
    @functools.wraps(func)
    def wrapped(*args):
        # 检查 out_dims 是否为整数或整数元组，用于确定批处理的输出维度。
        _check_out_dims_is_int_or_int_tuple(out_dims, func)
        # 增加 torch._C._vmapmode 的嵌套层级计数。
        vmap_level = torch._C._vmapmode_increment_nesting()
        try:
            # 创建批处理后的输入数据和批处理大小。
            batched_inputs, batch_size = _create_batched_inputs(
                in_dims, args, vmap_level, func
            )
            # 调用原始函数 func 处理批处理后的输入数据，得到批处理后的输出数据。
            batched_outputs = func(*batched_inputs)
            # 如果不允许 None 传递，则验证输出结果的有效性。
            if not allow_none_pass_through:
                _validate_outputs(batched_outputs, func)
            # 解包批处理后的输出数据，并返回结果。
            return _unwrap_batched(
                batched_outputs,
                out_dims,
                vmap_level,
                batch_size,
                func,
                allow_none_pass_through=allow_none_pass_through,
            )
        finally:
            # 减少 torch._C._vmapmode 的嵌套层级计数。
            torch._C._vmapmode_decrement_nesting()

    return wrapped
```