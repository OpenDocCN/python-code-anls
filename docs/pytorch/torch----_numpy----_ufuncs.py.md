# `.\pytorch\torch\_numpy\_ufuncs.py`

```
# 忽略 mypy 的类型检查错误
# 导入未来版本的注解特性，以支持更高级的类型注解
from __future__ import annotations

# 导入可选类型
from typing import Optional

# 导入 PyTorch 库
import torch

# 导入一些内部实现模块和工具函数
from . import _binary_ufuncs_impl, _dtypes_impl, _unary_ufuncs_impl, _util

# 导入正常化函数和相关类型
from ._normalizations import (
    ArrayLike,
    ArrayLikeOrScalar,
    CastingModes,
    DTypeLike,
    normalizer,
    NotImplementedType,
    OutArray,
)

# 定义一个函数用于二元通用函数的后处理
def _ufunc_postprocess(result, out, casting):
    # 如果指定了输出张量 out，则进行类型转换，将结果广播到与 out 相同的形状
    if out is not None:
        result = _util.typecast_tensor(result, out.dtype.torch_dtype, casting)
        result = torch.broadcast_to(result, out.shape)
    return result


# ############# Binary ufuncs ######################

# 列出 _binary_ufuncs_impl 模块中的所有非下划线开头的函数名，排除 "torch", "matmul", "divmod", "ldexp"
_binary = [
    name
    for name in dir(_binary_ufuncs_impl)
    if not name.startswith("_") and name not in ["torch", "matmul", "divmod", "ldexp"]
]

# 定义 NEP50_FUNCS 常量，包含支持 NEP50 提案的函数名称
NEP50_FUNCS = (
    "add", "subtract", "multiply", "floor_divide", "true_divide", "divide", "remainder",
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_left_shift", "bitwise_right_shift",
    "hypot", "arctan2", "logaddexp", "logaddexp2", "heaviside", "copysign", "fmax", "minimum",
    "fmin", "maximum", "fmod", "gcd", "lcm", "pow"
)


# 装饰器函数，用于装饰二元通用函数
def deco_binary_ufunc(torch_func):
    """Common infra for binary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """

    # 嵌套函数 wrapped，用于对二元通用函数进行类型规范化等处理
    @normalizer
    def wrapped(
        x1: ArrayLikeOrScalar,
        x2: ArrayLikeOrScalar,
        /,
        out: Optional[OutArray] = None,
        *,
        where: NotImplementedType = True,
        casting: Optional[CastingModes] = "same_kind",
        order: NotImplementedType = "K",
        dtype: Optional[DTypeLike] = None,
        subok: NotImplementedType = False,
        signature: NotImplementedType = None,
        extobj: NotImplementedType = None,
    ):
        # 如果指定了 dtype，则进行类型转换
        if dtype is not None:

            # 辅助函数 cast，用于将输入 x 转换为指定的 dtype 类型
            def cast(x, dtype):
                if isinstance(x, torch.Tensor):
                    return _util.typecast_tensor(x, dtype, casting)
                else:
                    return torch.as_tensor(x, dtype=dtype)

            # 分别对 x1 和 x2 进行 dtype 类型转换
            x1 = cast(x1, dtype)
            x2 = cast(x2, dtype)
        
        # 如果 x1 和 x2 都是 torch.Tensor 类型，则确定其结果类型，并进行类型转换
        elif isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            dtype = _dtypes_impl.result_type_impl(x1, x2)
            x1, x2 = _util.typecast_tensors((x1, x2), dtype, casting)
        
        # 否则，根据 NEP50 提案将 x1 和 x2 转换为张量
        else:
            x1, x2 = _dtypes_impl.nep50_to_tensors(
                x1, x2, torch_func.__name__ in NEP50_FUNCS, torch_func.__name__
            )

        # 调用 PyTorch 函数进行计算
        result = torch_func(x1, x2)

        # 对计算结果进行后处理并返回
        return _ufunc_postprocess(result, out, casting)

    # 设置 wrapped 函数的 __qualname__ 和 __name__ 属性
    wrapped.__qualname__ = torch_func.__name__
    wrapped.__name__ = torch_func.__name__

    return wrapped


# matmul 函数的签名与其他 ufuncs 稍有不同：
# - 没有 where=...
# - 额外有 axis=..., axes=...
# - 输入输出中不包含 NEP50 标量
@normalizer
def matmul(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    # 第一个位置参数 `/` 的作用是强制其后的所有参数必须以关键字参数的形式传入，不能使用位置参数
    out: Optional[OutArray] = None,
    # `out` 参数用于指定输出结果的数组，类型为 `Optional[OutArray]`，默认为 `None`
    *,
    # `*` 表示此处之后的参数必须使用关键字参数传入，不能使用位置参数
    casting: Optional[CastingModes] = "same_kind",
    # `casting` 参数用于指定数据类型转换方式，类型为 `Optional[CastingModes]`，默认为 `"same_kind"`
    order: NotImplementedType = "K",
    # `order` 参数指定数组在内存中的存储顺序，类型为 `NotImplementedType`，默认为 `"K"`
    dtype: Optional[DTypeLike] = None,
    # `dtype` 参数用于指定数组元素的数据类型，类型为 `Optional[DTypeLike]`，默认为 `None`
    subok: NotImplementedType = False,
    # `subok` 参数指示是否允许子类对象传递给返回的数组，类型为 `NotImplementedType`，默认为 `False`
    signature: NotImplementedType = None,
    # `signature` 参数用于指定函数的签名，类型为 `NotImplementedType`，默认为 `None`
    extobj: NotImplementedType = None,
    # `extobj` 参数用于指定额外的对象，类型为 `NotImplementedType`，默认为 `None`
    axes: NotImplementedType = None,
    # `axes` 参数用于指定操作的轴列表，类型为 `NotImplementedType`，默认为 `None`
    axis: NotImplementedType = None,
    # `axis` 参数用于指定操作的轴，类型为 `NotImplementedType`，默认为 `None`
# ldexp 函数的修饰器，用于标准化处理函数参数和返回值
@normalizer
# ldexp 函数定义，计算 x1 * (2 ** x2)
def ldexp(
    x1: ArrayLikeOrScalar,  # 第一个参数可以是数组或标量
    x2: ArrayLikeOrScalar,  # 第二个参数可以是数组或标量
    /,  # 表示后续参数必须按位置传递
    out: Optional[OutArray] = None,  # 输出数组，可选
    *,  # 表示后续参数必须按关键字传递
    where: NotImplementedType = True,  # 条件判断，未实现类型默认为 True
    casting: Optional[CastingModes] = "same_kind",  # 类型转换模式，可选，默认为 "same_kind"
    order: NotImplementedType = "K",  # 顺序，未实现类型默认为 "K"
    dtype: Optional[DTypeLike] = None,  # 数据类型，可选
    subok: NotImplementedType = False,  # 子类型，未实现类型默认为 False
    signature: NotImplementedType = None,  # 签名，未实现类型默认为 None
    extobj: NotImplementedType = None,  # 扩展对象，未实现类型默认为 None
):
    if dtype is not None:
        if isinstance(x1, torch.Tensor):
            x1 = _util.typecast_tensor(x1, dtype, casting)
        else:
            x1 = torch.as_tensor(x1, dtype=dtype)
    else:
        if not isinstance(x1, torch.Tensor):
            x1 = torch.as_tensor(x1)
            x1 = _util.cast_int_to_float(x1)

    x2 = torch.as_tensor(x2)
    # 第二个参数必须是整数
    if _dtypes_impl._category(x2.dtype) != 1:
        raise ValueError("ldexp 2nd arg must be integer")

    # 调用内部实现的 ldexp 函数计算结果
    result = _binary_ufuncs_impl.ldexp(x1, x2)

    # 如果 x1 的数据类型是 torch.float16，则将结果转换为 torch.float16 类型
    if x1.dtype == torch.float16:
        result = result.to(torch.float16)

    # 返回经过处理的结果
    return _ufunc_postprocess(result, out, casting)


# divmod 函数的修饰器，用于标准化处理函数参数和返回值
@normalizer
# divmod 函数定义，计算 x1 除以 x2 的商和余数
def divmod(
    x1: ArrayLike,  # 第一个参数可以是数组
    x2: ArrayLike,  # 第二个参数可以是数组
    out1: Optional[OutArray] = None,  # 第一个输出数组，可选
    out2: Optional[OutArray] = None,  # 第二个输出数组，可选
    /,  # 表示后续参数必须按位置传递
    out: tuple[Optional[OutArray], Optional[OutArray]] = (None, None),  # 输出元组，包含两个可选的输出数组
    *,  # 表示后续参数必须按关键字传递
    where: NotImplementedType = True,  # 条件判断，未实现类型默认为 True
    casting: Optional[CastingModes] = "same_kind",  # 类型转换模式，可选，默认为 "same_kind"
    order: NotImplementedType = "K",  # 顺序，未实现类型默认为 "K"
    dtype: Optional[DTypeLike] = None,  # 数据类型，可选
    subok: NotImplementedType = False,  # 子类型，未实现类型默认为 False
    signature: NotImplementedType = None,  # 签名，未实现类型默认为 None
    extobj: NotImplementedType = None,  # 扩展对象，未实现类型默认为 None
):
    # 确保没有同时提供 out1 和 out2，或者同时提供 out 作为位置参数和关键字参数
    num_outs = sum(x is not None for x in [out1, out2])
    if num_outs == 1:
        raise ValueError("both out1 and out2 need to be provided")
    elif num_outs == 2:
        o1, o2 = out
        if o1 is not None or o2 is not None:
            raise TypeError(
                "cannot specify 'out' as both a positional and keyword argument"
            )
    else:
        out1, out2 = out

    if dtype is None:
        dtype = _dtypes_impl.result_type_impl(x1, x2)
    x1, x2 = _util.typecast_tensors((x1, x2), dtype, casting)

    # 调用内部实现的 divmod 函数计算商和余数
    quot, rem = _binary_ufuncs_impl.divmod(x1, x2)

    # 对商和余数进行后处理
    quot = _ufunc_postprocess(quot, out1, casting)
    rem = _ufunc_postprocess(rem, out2, casting)
    return quot, rem


#
# 将 ufuncs 附加到此模块，以供在 __init__.py 中进一步导出到公共命名空间
#
for name in _binary:
    ufunc = getattr(_binary_ufuncs_impl, name)
    # 将给定的函数名作为变量名，使用装饰器`deco_binary_ufunc`来修饰`ufunc`函数，并将结果赋值给该变量名
    vars()[name] = deco_binary_ufunc(ufunc)
# 定义了一个函数 `modf`，用于将浮点数 x 拆分为整数部分和小数部分
def modf(x, /, *args, **kwds):
    # 使用 divmod 函数将 x 拆分为商和余数，其中 *args 和 **kwds 是 divmod 函数的额外参数
    quot, rem = divmod(x, 1, *args, **kwds)
    # 返回拆分后的结果，余数 rem 和 商 quot
    return rem, quot


# 将一些字符串添加到 `_binary` 列表中，这些字符串代表了数学操作函数的名称
_binary = _binary + ["divmod", "modf", "matmul", "ldexp"]


# ############# Unary ufuncs ######################


# 使用列表推导式从 `_unary_ufuncs_impl` 模块中选择以非下划线开头且不为 "torch" 的名称
_unary = [
    name
    for name in dir(_unary_ufuncs_impl)
    if not name.startswith("_") and name != "torch"
]


# 这些是一元通用函数，接受整数参数并返回浮点数结果
_fp_unary = [
    "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh", 
    "cbrt", "cos", "cosh", "deg2rad", "degrees", "exp", "exp2", 
    "expm1", "log", "log10", "log1p", "log2", "rad2deg", "radians", 
    "reciprocal", "sin", "sinh", "sqrt", "square", "tan", "tanh", 
    "trunc",
]


# 定义了一个装饰器函数 `deco_unary_ufunc`，用于处理一元通用函数的共同基础结构
def deco_unary_ufunc(torch_func):
    """Common infra for unary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """
    
    # 内部定义了一个被装饰的函数 `wrapped`
    @normalizer
    def wrapped(
        x: ArrayLike,
        /,
        out: Optional[OutArray] = None,
        *,
        where=True,
        casting: Optional[CastingModes] = "same_kind",
        order="K",
        dtype: Optional[DTypeLike] = None,
        subok: NotImplementedType = False,
        signature=None,
        extobj=None,
    ):
        # 如果指定了 dtype，则使用 `_util.typecast_tensor` 将输入 x 转换为指定类型 dtype
        if dtype is not None:
            x = _util.typecast_tensor(x, dtype, casting)
        
        # 如果 torch_func 的名称在 `_fp_unary` 列表中，则使用 `_util.cast_int_to_float` 将整数转换为浮点数
        if torch_func.__name__ in _fp_unary:
            x = _util.cast_int_to_float(x)

        # 调用给定的 torch_func 函数来处理 x
        result = torch_func(x)
        # 对结果进行后处理，使用 `_ufunc_postprocess` 函数处理输出
        result = _ufunc_postprocess(result, out, casting)
        # 返回处理后的结果
        return result

    # 设置函数 `wrapped` 的特殊属性，使其名称与 torch_func 相同
    wrapped.__qualname__ = torch_func.__name__
    wrapped.__name__ = torch_func.__name__

    # 返回装饰后的函数 `wrapped`
    return wrapped


#
# Attach ufuncs to this module, for a further export to the public namespace in __init__.py
#

# 将 `_unary` 列表中的每个名称绑定到 `_unary_ufuncs_impl` 模块中相应的 ufunc 函数
for name in _unary:
    ufunc = getattr(_unary_ufuncs_impl, name)
    vars()[name] = deco_unary_ufunc(ufunc)


# 设置 `__all__` 列表，以便在 `__init__.py` 中将 `_binary` 和 `_unary` 中的所有函数导出到公共命名空间
__all__ = _binary + _unary  # noqa: PLE0605
```