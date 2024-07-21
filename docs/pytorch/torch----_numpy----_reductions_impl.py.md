# `.\pytorch\torch\_numpy\_reductions_impl.py`

```py
# 忽略 mypy 类型检查错误
# 实现降维操作，可以被数组、数据类型等在“公共”层面包装。

""" Implementation of reduction operations, to be wrapped into arrays, dtypes etc
in the 'public' layer.

Anything here only deals with torch objects, e.g. "dtype" is a torch.dtype instance etc
"""

# 导入必要的库和模块
from __future__ import annotations

import functools
from typing import Optional, TYPE_CHECKING

import torch

from . import _dtypes_impl, _util

# 如果在类型检查模式下
if TYPE_CHECKING:
    from ._normalizations import (
        ArrayLike,
        AxisLike,
        DTypeLike,
        KeepDims,
        NotImplementedType,
        OutArray,
    )

# 装饰器函数，处理轴参数在降维操作中的通用情况
def _deco_axis_expand(func):
    """
    Generically handle axis arguments in reductions.
    axis is *always* the 2nd arg in the function so no need to have a look at its signature
    """
    @functools.wraps(func)
    def wrapped(a, axis=None, *args, **kwds):
        # 如果 axis 参数不为空，则将其标准化为元组形式
        if axis is not None:
            axis = _util.normalize_axis_tuple(axis, a.ndim)

        # 如果 axis 是空元组
        if axis == ():
            # 插入一个长度为一的轴并沿此轴执行降维操作
            # 不能直接返回 a.clone()，因为这会绕过函数内部的检查
            newshape = _util.expand_shape(a.shape, axis=0)
            a = a.reshape(newshape)
            axis = (0,)

        # 调用原始的函数并返回结果
        return func(a, axis, *args, **kwds)

    return wrapped


# 返回一个实数或复数浮点类型的数据类型
def _atleast_float(dtype, other_dtype):
    """Return a dtype that is real or complex floating-point.

    For inputs that are boolean or integer dtypes, this returns the default
    float dtype; inputs that are complex get converted to the default complex
    dtype; real floating-point dtypes (`float*`) get passed through unchanged
    """
    if dtype is None:
        dtype = other_dtype
    # 如果 dtype 不是浮点类型或复数类型，则返回默认的浮点数数据类型
    if not (dtype.is_floating_point or dtype.is_complex):
        return _dtypes_impl.default_dtypes().float_dtype
    return dtype


# 装饰器函数，处理轴参数在降维操作中的通用情况
@_deco_axis_expand
def count_nonzero(a: ArrayLike, axis: AxisLike = None, *, keepdims: KeepDims = False):
    return a.count_nonzero(axis)


# 装饰器函数，处理轴参数在降维操作中的通用情况
@_deco_axis_expand
def argmax(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    *,
    keepdims: KeepDims = False,
):
    # 如果输入数组的类型是复数，则抛出未实现异常
    if a.is_complex():
        raise NotImplementedError(f"argmax with dtype={a.dtype}.")

    # 规范化 axis 参数，使其只能是单个轴
    axis = _util.allow_only_single_axis(axis)

    # 如果输入数组的数据类型是布尔型
    if a.dtype == torch.bool:
        # 把布尔类型转换为无符号整型
        a = a.to(torch.uint8)

    # 调用 PyTorch 提供的 argmax 函数
    return torch.argmax(a, axis)


# 装饰器函数，处理轴参数在降维操作中的通用情况
@_deco_axis_expand
def argmin(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    *,
    keepdims: KeepDims = False,
):
    # 如果输入数组的类型是复数，则抛出未实现异常
    if a.is_complex():
        raise NotImplementedError(f"argmin with dtype={a.dtype}.")

    # 规范化 axis 参数，使其只能是单个轴
    axis = _util.allow_only_single_axis(axis)

    # 如果输入数组的数据类型是布尔型
    if a.dtype == torch.bool:
        # 把布尔类型转换为无符号整型
        a = a.to(torch.uint8)

    # 调用 PyTorch 提供的 argmin 函数
    return torch.argmin(a, axis)


# 装饰器函数，处理轴参数在降维操作中的通用情况
@_deco_axis_expand
def any(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    keepdims: KeepDims = False,
    # 定义一个带有默认参数的函数
    *,
    # "*" 表示这是一个仅带有关键字参数的参数列表开始
    where: NotImplementedType = None,
    # 定义了一个关键字参数 "where"，默认为 None，类型为 NotImplementedType
@_deco_axis_expand
def any(
    a: ArrayLike,
    axis: AxisLike = None,
    *,
    where: NotImplementedType = None,
):
    # 确保 axis 参数是有效的单轴或者为 None
    axis = _util.allow_only_single_axis(axis)
    # 如果 axis 是 None，则使用空字典作为关键字参数 axis_kw
    axis_kw = {} if axis is None else {"dim": axis}
    # 调用 torch.any 函数，返回是否在指定轴上的任意元素为真的布尔值
    return torch.any(a, **axis_kw)


@_deco_axis_expand
def all(
    a: ArrayLike,
    axis: AxisLike = None,
    *,
    where: NotImplementedType = None,
):
    # 确保 axis 参数是有效的单轴或者为 None
    axis = _util.allow_only_single_axis(axis)
    # 如果 axis 是 None，则使用空字典作为关键字参数 axis_kw
    axis_kw = {} if axis is None else {"dim": axis}
    # 调用 torch.all 函数，返回是否在指定轴上的所有元素为真的布尔值
    return torch.all(a, **axis_kw)


@_deco_axis_expand
def amax(
    a: ArrayLike,
    axis: AxisLike = None,
    *,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
):
    # 如果数组 a 是复数类型，抛出未实现的错误
    if a.is_complex():
        raise NotImplementedError(f"amax with dtype={a.dtype}")

    # 返回数组 a 在指定轴上的最大值
    return a.amax(axis)


max = amax  # 将 amax 函数别名为 max


@_deco_axis_expand
def amin(
    a: ArrayLike,
    axis: AxisLike = None,
    *,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
):
    # 如果数组 a 是复数类型，抛出未实现的错误
    if a.is_complex():
        raise NotImplementedError(f"amin with dtype={a.dtype}")

    # 返回数组 a 在指定轴上的最小值
    return a.amin(axis)


min = amin  # 将 amin 函数别名为 min


@_deco_axis_expand
def ptp(
    a: ArrayLike,
    axis: AxisLike = None,
    *,
    keepdims: KeepDims = False,
):
    # 返回数组 a 在指定轴上的峰峰值（最大值减最小值）
    return a.amax(axis) - a.amin(axis)


@_deco_axis_expand
def sum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    *,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
):
    # 断言 dtype 为空或者是 torch 的数据类型
    assert dtype is None or isinstance(dtype, torch.dtype)

    # 如果 dtype 是布尔类型，则将其转换为默认整数类型
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype

    # 如果 axis 是 None，则使用空字典作为关键字参数 axis_kw
    axis_kw = {} if axis is None else {"dim": axis}
    # 返回数组 a 在指定轴上的和
    return a.sum(dtype=dtype, **axis_kw)


@_deco_axis_expand
def prod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    *,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
):
    # 确保 axis 参数是有效的单轴
    axis = _util.allow_only_single_axis(axis)

    # 如果 dtype 是布尔类型，则将其转换为默认整数类型
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype

    # 如果 axis 是 None，则使用空字典作为关键字参数 axis_kw
    axis_kw = {} if axis is None else {"dim": axis}
    # 返回数组 a 在指定轴上的乘积
    return a.prod(dtype=dtype, **axis_kw)


product = prod  # 将 prod 函数别名为 product


@_deco_axis_expand
def mean(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    *,
    where: NotImplementedType = None,
):
    # 将 dtype 至少转换为浮点数类型
    dtype = _atleast_float(dtype, a.dtype)

    # 如果 axis 是 None，则使用空字典作为关键字参数 axis_kw
    axis_kw = {} if axis is None else {"dim": axis}
    # 返回数组 a 在指定轴上的均值
    result = a.mean(dtype=dtype, **axis_kw)

    return result


@_deco_axis_expand
def std(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    ddof=0,
    *,
    where: NotImplementedType = None,
    keepdims: KeepDims = False,
):
    # 以下代码未提供完整，但需要添加注释
    in_dtype = dtype
    # 将输入的数据类型存储在变量in_dtype中，备份用于返回结果时恢复原始数据类型
    
    dtype = _atleast_float(dtype, a.dtype)
    # 调用函数_atleast_float，确保dtype至少是浮点数类型，将结果存储在dtype中
    
    tensor = _util.cast_if_needed(a, dtype)
    # 调用_util.cast_if_needed函数，将输入张量a转换为指定的dtype类型，并将结果存储在tensor中
    
    result = tensor.std(dim=axis, correction=ddof)
    # 对tensor张量计算指定维度axis上的标准差，使用自由度校正因子ddof，将结果存储在result中
    
    return _util.cast_if_needed(result, in_dtype)
    # 将计算得到的标准差result转换回原始的数据类型in_dtype，并返回结果
````
@_deco_axis_expand
# 定义计算方差的函数，接受数组 a、指定轴 axis、数据类型 dtype、输出 out、自由度 ddof、是否保持维度 keepdims 和一个关键字参数 where
def var(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    out: Optional[OutArray] = None,
    ddof=0,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
):
    in_dtype = dtype  # 保存输入数据类型
    # 将输入数据转换为浮点类型，如果 dtype 为 None，则使用 a 的数据类型
    dtype = _atleast_float(dtype, a.dtype)
    # 将输入数据 a 转换为指定的数据类型 dtype
    tensor = _util.cast_if_needed(a, dtype)
    # 计算 tensor 在指定轴上的方差，ddof 为自由度调整参数
    result = tensor.var(dim=axis, correction=ddof)
    # 将结果转换为输入数据类型 in_dtype
    return _util.cast_if_needed(result, in_dtype)


# cumsum / cumprod 函数是几乎是归约操作：
#   1. 不保持维度 keepdims
#   2. axis=None 时会将数组展平

def cumsum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    out: Optional[OutArray] = None,
):
    # 如果数据类型 dtype 为 torch.bool，则转换为整数类型
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype
    # 如果 dtype 为 None，则使用 a 的数据类型
    if dtype is None:
        dtype = a.dtype

    # 将 a 和 axis 进行处理，返回元组 (a,) 和 axis
    (a,), axis = _util.axis_none_flatten(a, axis=axis)
    # 将 axis 转换为规范化的轴索引
    axis = _util.normalize_axis_index(axis, a.ndim)

    # 返回数组 a 在指定轴上的累加和
    return a.cumsum(axis=axis, dtype=dtype)


def cumprod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    out: Optional[OutArray] = None,
):
    # 如果数据类型 dtype 为 torch.bool，则转换为整数类型
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype
    # 如果 dtype 为 None，则使用 a 的数据类型
    if dtype is None:
        dtype = a.dtype

    # 将 a 和 axis 进行处理，返回元组 (a,) 和 axis
    (a,), axis = _util.axis_none_flatten(a, axis=axis)
    # 将 axis 转换为规范化的轴索引
    axis = _util.normalize_axis_index(axis, a.ndim)

    # 返回数组 a 在指定轴上的累积乘积
    return a.cumprod(axis=axis, dtype=dtype)


cumproduct = cumprod  # 将 cumprod 函数赋值给 cumproduct


def average(
    a: ArrayLike,
    axis=None,
    weights: ArrayLike = None,
    returned=False,
    *,
    keepdims=False,
):
    # 如果权重 weights 为 None，则计算 a 的均值
    if weights is None:
        result = mean(a, axis=axis)
        # 计算所有元素个数的倒数作为加权和的权重
        wsum = torch.as_tensor(a.numel() / result.numel(), dtype=result.dtype)
    else:
        # 如果 a 的数据类型不是浮点类型，则转换为 double 类型
        if not a.dtype.is_floating_point:
            a = a.double()

        # 检查权重 weights 是否与数组 a 的形状相同
        if a.shape != weights.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights differ."
                )
            if weights.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ."
                )
            if weights.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis."
                )

            # 将权重 weights 广播到数组 a 的指定轴上
            weights = torch.broadcast_to(weights, (a.ndim - 1) * (1,) + weights.shape)
            weights = weights.swapaxes(-1, axis)

        # 计算加权平均的结果
        result_dtype = _dtypes_impl.result_type_impl(a, weights)
        numerator = sum(a * weights, axis, dtype=result_dtype)
        wsum = sum(weights, axis, dtype=result_dtype)
        result = numerator / wsum

    # 如果 keepdims 为 True，则应用保持维度操作
    if keepdims:
        result = _util.apply_keepdims(result, axis, a.ndim)
    # 检查是否有返回值
    if returned:
        # 如果权重总和的形状与结果的形状不匹配
        if wsum.shape != result.shape:
            # 使用广播将权重总和广播到与结果相同的形状，并进行克隆
            wsum = torch.broadcast_to(wsum, result.shape).clone()
        # 返回结果和权重总和
        return result, wsum
    else:
        # 如果没有返回值，直接返回结果
        return result
# Not using deco_axis_expand as it assumes that axis is the second arg
def quantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    overwrite_input=False,
    method="linear",
    keepdims: KeepDims = False,
    *,
    interpolation: NotImplementedType = None,
):
    if overwrite_input:
        # raise NotImplementedError("overwrite_input in quantile not implemented.")
        # NumPy documents that `overwrite_input` MAY modify inputs:
        # https://numpy.org/doc/stable/reference/generated/numpy.percentile.html#numpy-percentile
        # Here we choose to work out-of-place because why not.
        # 如果设置了 overwrite_input 标志，根据 NumPy 文档，`overwrite_input` 可能会修改输入数组。
        # 这里选择采用非原地计算方式，以避免修改输入数组。

        pass

    if not a.dtype.is_floating_point:
        # 如果数组 `a` 的数据类型不是浮点型，将其转换为默认的浮点数数据类型
        dtype = _dtypes_impl.default_dtypes().float_dtype
        a = a.to(dtype)

    # edge case: torch.quantile only supports float32 and float64
    # 特殊情况处理：torch.quantile 仅支持 float32 和 float64 类型
    if a.dtype == torch.float16:
        a = a.to(torch.float32)

    if axis is None:
        # 如果未指定轴向 `axis`，将数组 `a` 和分位数 `q` 展平处理
        a = a.flatten()
        q = q.flatten()
        axis = (0,)
    else:
        # 否则，将轴向 `axis` 规范化为元组形式，适应数组 `a` 的维度
        axis = _util.normalize_axis_tuple(axis, a.ndim)

    # FIXME(Mario) Doesn't np.quantile accept a tuple?
    # torch.quantile does accept a number. If we don't want to implement the tuple behaviour
    # (it's deffo low prio) change `normalize_axis_tuple` into a normalize_axis index above.
    # FIXME(Mario) np.quantile 是否接受元组作为参数？
    # torch.quantile 接受单个数字作为参数。如果不打算实现元组行为（这不是高优先级任务），
    # 将 `normalize_axis_tuple` 更改为上面的 normalize_axis 索引。

    axis = _util.allow_only_single_axis(axis)

    q = _util.cast_if_needed(q, a.dtype)

    # 调用 torch.quantile 函数计算分位数
    return torch.quantile(a, q, axis=axis, interpolation=method)


def percentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    overwrite_input=False,
    method="linear",
    keepdims: KeepDims = False,
    *,
    interpolation: NotImplementedType = None,
):
    # np.percentile(float_tensor, 30) : q.dtype is int64 => q / 100.0 is float32
    # 如果 q 的数据类型为整数型，将其转换为默认的浮点数数据类型
    if _dtypes_impl.python_type_for_torch(q.dtype) == int:
        q = q.to(_dtypes_impl.default_dtypes().float_dtype)
    # 将 q 转换为百分数表示
    qq = q / 100.0

    # 调用 quantile 函数计算百分位数
    return quantile(
        a,
        qq,
        axis=axis,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        interpolation=interpolation,
    )


def median(
    a: ArrayLike,
    axis=None,
    out: Optional[OutArray] = None,
    overwrite_input=False,
    keepdims: KeepDims = False,
):
    # 调用 quantile 函数计算中位数，中位数对应于分位数为 0.5 的情况
    return quantile(
        a,
        torch.as_tensor(0.5),
        axis=axis,
        overwrite_input=overwrite_input,
        out=out,
        keepdims=keepdims,
    )
```