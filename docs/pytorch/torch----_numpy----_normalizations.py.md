# `.\pytorch\torch\_numpy\_normalizations.py`

```
# mypy: ignore-errors
# 忽略 mypy 类型检查时可能出现的错误

""" "Normalize" arguments: convert array_likes to tensors, dtypes to torch dtypes and so on.
"""
# 引入未来的注解模块，以便支持 Type annotations
from __future__ import annotations

# 引入 functools 模块，用于高阶函数操作
import functools
# 引入 inspect 模块，用于对象内省（introspection）
import inspect
# 引入 operator 模块，提供了大量的操作符函数
import operator
# 引入 typing 模块，支持类型提示
import typing

# 引入 PyTorch 库
import torch

# 从当前包中导入 _dtypes、_dtypes_impl 和 _util 模块
from . import _dtypes, _dtypes_impl, _util

# 定义一个泛型类型 ArrayLike，表示类数组结构
ArrayLike = typing.TypeVar("ArrayLike")
# 定义一个标量类型 Scalar，可以是 int、float、complex、bool 中的一种
Scalar = typing.Union[int, float, complex, bool]
# 定义一个类型变量，可以是 ArrayLike 或 Scalar
ArrayLikeOrScalar = typing.Union[ArrayLike, Scalar]

# 定义一个泛型类型 DTypeLike，表示数据类型
DTypeLike = typing.TypeVar("DTypeLike")
# 定义一个泛型类型 AxisLike，表示轴向标识
AxisLike = typing.TypeVar("AxisLike")
# 定义一个泛型类型 NDArray，表示多维数组
NDArray = typing.TypeVar("NDArray")
# 定义一个泛型类型 CastingModes，表示类型转换模式
CastingModes = typing.TypeVar("CastingModes")
# 定义一个泛型类型 KeepDims，表示是否保留维度信息
KeepDims = typing.TypeVar("KeepDims")

# 定义一个特殊的泛型类型 OutArray，用于标注 out= 数组参数
#
# 这个类型在多个方面都比较特殊：
# 首先，它需要是一个 NDArray，并且我们需要保持 `result is out` 的语义。
# 因此，我们不能仅仅从 out 数组中提取张量。
# 所以我们不会将 out 数组传递给实现函数，而是在下面的 `normalizer` 中处理它。
# 其次，out= 参数既可以是关键字参数，也可以是位置参数，并且作为位置参数时可以出现在签名的任何位置。
# 为了处理这一切，我们定义了一个特殊的 `OutArray` 标注并在其上进行调度。
#
OutArray = typing.TypeVar("OutArray")

# 尝试导入 NotImplementedType 类型，如果导入失败则定义为 NotImplementedType 泛型
try:
    from typing import NotImplementedType
except ImportError:
    NotImplementedType = typing.TypeVar("NotImplementedType")


# 函数：将类数组结构转换为张量，并返回
def normalize_array_like(x, parm=None):
    # 从 _ndarray 模块中导入 asarray 函数
    from ._ndarray import asarray

    # 调用 asarray 函数将 x 转换为张量，并返回
    return asarray(x).tensor


# 函数：将类数组结构或标量转换为张量或直接返回
def normalize_array_like_or_scalar(x, parm=None):
    # 如果 x 是标量或符号型，则直接返回 x
    if _dtypes_impl.is_scalar_or_symbolic(x):
        return x
    # 否则，调用 normalize_array_like 函数将 x 转换为张量并返回
    return normalize_array_like(x, parm)


# 函数：将可选的类数组结构或标量转换为张量或直接返回
def normalize_optional_array_like_or_scalar(x, parm=None):
    # 如果 x 是 None，则直接返回 None
    if x is None:
        return None
    # 否则，调用 normalize_array_like_or_scalar 函数将 x 转换为张量或标量，并返回
    return normalize_array_like_or_scalar(x, parm)


# 函数：将可选的类数组结构转换为张量或直接返回 None
def normalize_optional_array_like(x, parm=None):
    # 如果 x 是 None，则直接返回 None
    if x is None:
        return None
    # 否则，调用 normalize_array_like 函数将 x 转换为张量，并返回
    return normalize_array_like(x, parm)


# 函数：将序列中的类数组结构转换为张量的元组
def normalize_seq_array_like(x, parm=None):
    # 遍历序列 x 中的每个值，调用 normalize_array_like 函数将其转换为张量，并生成元组
    return tuple(normalize_array_like(value) for value in x)


# 函数：将数据类型转换为 Torch 支持的数据类型
def normalize_dtype(dtype, parm=None):
    # 根据 _decorators.dtype_to_torch 的实现，将 dtype 转换为 Torch 的数据类型
    torch_dtype = None
    if dtype is not None:
        dtype = _dtypes.dtype(dtype)
        torch_dtype = dtype.torch_dtype
    return torch_dtype


# 函数：处理不支持的参数，抛出 NotImplementedError
def normalize_not_implemented(arg, parm):
    # 如果参数 arg 不等于 parm 的默认值，则抛出 NotImplementedError 异常
    if arg != parm.default:
        raise NotImplementedError(f"'{parm.name}' parameter is not supported.")


# 函数：将类似轴标识转换为标量索引
def normalize_axis_like(arg, parm=None):
    # 从 _ndarray 模块中导入 ndarray 类
    from ._ndarray import ndarray

    # 如果 arg 是 ndarray 类型，则将其转换为标量索引
    if isinstance(arg, ndarray):
        arg = operator.index(arg)
    return arg


# 函数：将 NDArray 类型参数提取其张量属性
def normalize_ndarray(arg, parm=None):
    # 如果 arg 是 None，则直接返回 None
    if arg is None:
        return arg

    # 从 _ndarray 模块中导入 ndarray 类
    from ._ndarray import ndarray

    # 如果 arg 不是 ndarray 类型，则抛出 TypeError 异常
    if not isinstance(arg, ndarray):
        raise TypeError(f"'{parm.name}' must be an array")
    # 返回 arg 的张量属性
    return arg.tensor
# 将参数标准化为 ndarray，不返回其张量
def normalize_outarray(arg, parm=None):
    if arg is None:
        return arg
    from ._ndarray import ndarray  # 导入自定义的 ndarray 类

    # 如果参数是 torch.Tensor 类型，则转换为 ndarray
    if isinstance(arg, torch.Tensor):
        arg = ndarray(arg)

    # 如果参数不是 ndarray 类型，则抛出类型错误异常
    if not isinstance(arg, ndarray):
        raise TypeError(f"'{parm.name}' must be an array")
    return arg


# 校验参数是否为预定义的 casting 值之一
def normalize_casting(arg, parm=None):
    if arg not in ["no", "equiv", "safe", "same_kind", "unsafe"]:
        raise ValueError(
            f"casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe' (got '{arg}')"
        )
    return arg


# 预定义的参数标准化函数映射表
normalizers = {
    "ArrayLike": normalize_array_like,
    "ArrayLikeOrScalar": normalize_array_like_or_scalar,
    "Optional[ArrayLike]": normalize_optional_array_like,
    "Sequence[ArrayLike]": normalize_seq_array_like,
    "Optional[ArrayLikeOrScalar]": normalize_optional_array_like_or_scalar,
    "Optional[NDArray]": normalize_ndarray,
    "Optional[OutArray]": normalize_outarray,  # 使用 normalize_outarray 函数
    "NDArray": normalize_ndarray,
    "Optional[DTypeLike]": normalize_dtype,
    "AxisLike": normalize_axis_like,
    "NotImplementedType": normalize_not_implemented,
    "Optional[CastingModes]": normalize_casting,  # 使用 normalize_casting 函数
}


# 根据参数类型选择相应的标准化函数来标准化参数
def maybe_normalize(arg, parm):
    """Normalize arg if a normalizer is registered."""
    normalizer = normalizers.get(parm.annotation, None)  # 获取参数类型对应的标准化函数
    return normalizer(arg, parm) if normalizer else arg


# ### Return value helpers ###


# 将计算结果复制到输出参数 out 中
def maybe_copy_to(out, result, promote_scalar_result=False):
    # 注意：这里的 out 可能是 ndarray，也可能是 None
    if out is None:
        return result
    elif isinstance(result, torch.Tensor):
        # 如果 result 是 torch.Tensor 类型，则处理形状匹配并复制数据到 out 的 tensor 中
        if result.shape != out.shape:
            can_fit = result.numel() == 1 and out.ndim == 0
            if promote_scalar_result and can_fit:
                result = result.squeeze()
            else:
                raise ValueError(
                    f"Bad size of the out array: out.shape = {out.shape}"
                    f" while result.shape = {result.shape}."
                )
        out.tensor.copy_(result)
        return out
    elif isinstance(result, (tuple, list)):
        # 如果 result 是 tuple 或 list 类型，则逐个处理其中的元素并递归调用 maybe_copy_to 函数
        return type(result)(
            maybe_copy_to(o, r, promote_scalar_result) for o, r in zip(out, result)
        )
    else:
        raise AssertionError  # 不应该到达这条路径，表示出现意料之外的情况


# 包装结果为 ndarray，如果是 torch.Tensor 则先转换为 ndarray
def wrap_tensors(result):
    from ._ndarray import ndarray  # 导入自定义的 ndarray 类

    if isinstance(result, torch.Tensor):
        return ndarray(result)
    elif isinstance(result, (tuple, list)):
        # 如果 result 是 tuple 或 list 类型，则逐个处理其中的元素并递归调用 wrap_tensors 函数
        result = type(result)(wrap_tensors(x) for x in result)
    return result


# 将数值或数组包装为 ndarray 类型
def array_or_scalar(values, py_type=float, return_scalar=False):
    if return_scalar:
        return py_type(values.item())  # 返回标量值
    else:
        from ._ndarray import ndarray  # 导入自定义的 ndarray 类

        return ndarray(values)  # 返回 ndarray 对象


# ### The main decorator to normalize arguments / postprocess the output ###
# 定义一个装饰器函数，可以接受一个函数作为参数，也可以接受一个关键字参数promote_scalar_result
def normalizer(_func=None, *, promote_scalar_result=False):
    # 内部函数，接受一个函数作为参数
    def normalizer_inner(func):
        # 使用functools库中的wraps装饰器，保留被装饰函数的元数据
        @functools.wraps(func)
        # 包装函数，接受任意数量的位置参数和关键字参数
        def wrapped(*args, **kwds):
            # 获取函数的参数签名
            sig = inspect.signature(func)
            params = sig.parameters
            # 获取第一个参数
            first_param = next(iter(params.values()))

            # 如果第一个参数是可变位置参数，对所有位置参数进行归一化处理
            if first_param.kind == inspect.Parameter.VAR_POSITIONAL:
                args = [maybe_normalize(arg, first_param) for arg in args]
            else:
                # 否则，对所有参数进行归一化处理
                args = (
                    tuple(
                        maybe_normalize(arg, parm)
                        for arg, parm in zip(args, params.values())
                    )
                    + args[len(params.values()) :]
                )

            # 对所有关键字参数进行归一化处理
            kwds = {
                name: maybe_normalize(arg, params[name]) if name in params else arg
                for name, arg in kwds.items()
            }

            # 调用原始函数，传入归一化后的参数
            result = func(*args, **kwds)

            # 处理参数中的keepdims
            bound_args = None
            if "keepdims" in params and params["keepdims"].annotation == "KeepDims":
                bound_args = sig.bind(*args, **kwds).arguments
                if bound_args.get("keepdims", False):
                    tensor = args[0]
                    axis = bound_args.get("axis")
                    # 应用_keepdims函数处理结果
                    result = _util.apply_keepdims(result, axis, tensor.ndim)

            # 处理参数中的out
            if "out" in params:
                if bound_args is None:
                    bound_args = sig.bind(*args, **kwds).arguments
                out = bound_args.get("out")
                # 复制结果到指定的out参数中
                result = maybe_copy_to(out, result, promote_scalar_result)

            # 封装结果为张量
            result = wrap_tensors(result)

            return result

        return wrapped

    # 如果_func为None，返回内部函数normalizer_inner
    if _func is None:
        return normalizer_inner
    else:
        # 否则，直接调用内部函数normalizer_inner，并传入_func作为参数
        return normalizer_inner(_func)
```