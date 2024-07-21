# `.\pytorch\torch\_numpy\_funcs_impl.py`

```
# mypy: ignore-errors

"""A thin pytorch / numpy compat layer.

Things imported from here have numpy-compatible signatures but operate on
pytorch tensors.
"""
# Contents of this module ends up in the main namespace via _funcs.py
# where type annotations are used in conjunction with the @normalizer decorator.
from __future__ import annotations

import builtins
import itertools
import operator
from typing import Optional, Sequence, TYPE_CHECKING

import torch

from . import _dtypes_impl, _util

if TYPE_CHECKING:
    from ._normalizations import (
        ArrayLike,
        ArrayLikeOrScalar,
        CastingModes,
        DTypeLike,
        NDArray,
        NotImplementedType,
        OutArray,
    )


def copy(
    a: ArrayLike, order: NotImplementedType = "K", subok: NotImplementedType = False
):
    return a.clone()


def copyto(
    dst: NDArray,
    src: ArrayLike,
    casting: Optional[CastingModes] = "same_kind",
    where: NotImplementedType = None,
):
    (src,) = _util.typecast_tensors((src,), dst.dtype, casting=casting)
    dst.copy_(src)


def atleast_1d(*arys: ArrayLike):
    res = torch.atleast_1d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def atleast_2d(*arys: ArrayLike):
    res = torch.atleast_2d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def atleast_3d(*arys: ArrayLike):
    res = torch.atleast_3d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def _concat_check(tup, dtype, out):
    if tup == ():
        raise ValueError("need at least one array to concatenate")

    """Check inputs in concatenate et al."""
    if out is not None and dtype is not None:
        # mimic numpy
        raise TypeError(
            "concatenate() only takes `out` or `dtype` as an "
            "argument, but both were provided."
        )


def _concat_cast_helper(tensors, out=None, dtype=None, casting="same_kind"):
    """Figure out dtypes, cast if necessary."""

    if out is not None or dtype is not None:
        # figure out the type of the inputs and outputs
        out_dtype = out.dtype.torch_dtype if dtype is None else dtype
    else:
        out_dtype = _dtypes_impl.result_type_impl(*tensors)

    # cast input arrays if necessary; do not broadcast them agains `out`
    tensors = _util.typecast_tensors(tensors, out_dtype, casting)

    return tensors


def _concatenate(
    tensors, axis=0, out=None, dtype=None, casting: Optional[CastingModes] = "same_kind"
):
    # pure torch implementation, used below and in cov/corrcoef below
    tensors, axis = _util.axis_none_flatten(*tensors, axis=axis)
    tensors = _concat_cast_helper(tensors, out, dtype, casting)
    return torch.cat(tensors, axis)


def concatenate(
    ar_tuple: Sequence[ArrayLike],
    axis=0,
    out: Optional[OutArray] = None,
    dtype: Optional[DTypeLike] = None,
    casting: Optional[CastingModes] = "same_kind",
):
    """Concatenate arrays along a specified axis.

    Parameters:
    - ar_tuple: Sequence of arrays to concatenate.
    - axis: The axis along which the arrays will be joined. Default is 0.
    - out: Optional. If provided, the result will be placed into this array.
    - dtype: Optional. Desired data-type for the arrays. If not given,
      inferred from the input arrays.
    - casting: Optional. Controls what kind of data casting may occur.

    Returns:
    - Concatenated tensor.

    Notes:
    - Mimics the behavior of numpy's concatenate function.
    """
    # 调用函数 _concat_check，检查并准备要拼接的数组元组和数据类型
    _concat_check(ar_tuple, dtype, out=out)
    # 调用函数 _concatenate，进行数组拼接操作
    result = _concatenate(ar_tuple, axis=axis, out=out, dtype=dtype, casting=casting)
    # 返回拼接后的结果
    return result
# 根据给定的序列或可变长度参数进行垂直堆叠，返回堆叠后的张量
def vstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: Optional[DTypeLike] = None,
    casting: Optional[CastingModes] = "same_kind",
):
    # 检查堆叠操作的有效性，不输出到特定的张量（out=None）
    _concat_check(tup, dtype, out=None)
    # 将输入序列进行类型转换后进行堆叠操作
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    # 调用 Torch 库的垂直堆叠方法，返回结果
    return torch.vstack(tensors)


# 将 vstack 函数定义赋值给 row_stack，使其成为 vstack 的别名
row_stack = vstack


# 根据给定的序列或可变长度参数进行水平堆叠，返回堆叠后的张量
def hstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: Optional[DTypeLike] = None,
    casting: Optional[CastingModes] = "same_kind",
):
    # 检查堆叠操作的有效性，不输出到特定的张量（out=None）
    _concat_check(tup, dtype, out=None)
    # 将输入序列进行类型转换后进行堆叠操作
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    # 调用 Torch 库的水平堆叠方法，返回结果
    return torch.hstack(tensors)


# 根据给定的序列或可变长度参数进行深度堆叠，返回堆叠后的张量
def dstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: Optional[DTypeLike] = None,
    casting: Optional[CastingModes] = "same_kind",
):
    # XXX: 在 numpy 1.24 中，dstack 方法不包含 dtype 和 casting 关键字，
    # 但 hstack 和 vstack 包含。因此，为了一致性在此添加这些关键字。
    # 检查堆叠操作的有效性，不输出到特定的张量（out=None）
    _concat_check(tup, dtype, out=None)
    # 将输入序列进行类型转换后进行堆叠操作
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    # 调用 Torch 库的深度堆叠方法，返回结果
    return torch.dstack(tensors)


# 根据给定的序列或可变长度参数进行列堆叠，返回堆叠后的张量
def column_stack(
    tup: Sequence[ArrayLike],
    *,
    dtype: Optional[DTypeLike] = None,
    casting: Optional[CastingModes] = "same_kind",
):
    # XXX: 在 numpy 1.24 中，column_stack 方法不包含 dtype 和 casting 关键字，
    # 但 row_stack 包含（因为 row_stack 实际上是 vstack 的别名）。因此，为了一致性在此添加这些关键字。
    # 检查堆叠操作的有效性，不输出到特定的张量（out=None）
    _concat_check(tup, dtype, out=None)
    # 将输入序列进行类型转换后进行堆叠操作
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    # 调用 Torch 库的列堆叠方法，返回结果
    return torch.column_stack(tensors)


# 根据给定的序列或可变长度参数进行沿指定轴堆叠，返回堆叠后的张量
def stack(
    arrays: Sequence[ArrayLike],
    axis=0,
    out: Optional[OutArray] = None,
    *,
    dtype: Optional[DTypeLike] = None,
    casting: Optional[CastingModes] = "same_kind",
):
    # 检查堆叠操作的有效性，将结果输出到指定的张量（out=out）
    _concat_check(arrays, dtype, out=out)
    # 将输入序列进行类型转换后进行堆叠操作
    tensors = _concat_cast_helper(arrays, dtype=dtype, casting=casting)
    # 确定堆叠后的张量维度
    result_ndim = tensors[0].ndim + 1
    # 标准化轴索引以适应结果张量的维度
    axis = _util.normalize_axis_index(axis, result_ndim)
    # 调用 Torch 库的沿指定轴堆叠方法，返回结果
    return torch.stack(tensors, axis=axis)


# 将给定数组或值沿指定轴附加到数组的末尾
def append(arr: ArrayLike, values: ArrayLike, axis=None):
    if axis is None:
        # 如果未指定轴，则将数组扁平化（一维化）
        if arr.ndim != 1:
            arr = arr.flatten()
        # 将要附加的值也扁平化（一维化）
        values = values.flatten()
        # 设置默认轴为数组的最后一个维度
        axis = arr.ndim - 1
    # 调用内部的连接函数 _concatenate，沿指定轴附加数组或值，并返回结果
    return _concatenate((arr, values), axis=axis)


# 内部辅助函数：根据索引或分段数对张量进行分割，沿指定轴进行操作，严格模式下处理不同的情况
def _split_helper(tensor, indices_or_sections, axis, strict=False):
    if isinstance(indices_or_sections, int):
        # 如果 indices_or_sections 是整数，调用对应的整数分割函数
        return _split_helper_int(tensor, indices_or_sections, axis, strict)
    elif isinstance(indices_or_sections, (list, tuple)):
        # 如果 indices_or_sections 是列表或元组，调用对应的列表分割函数
        # 注意：忽略 split=...，它仅适用于 split_helper_int
        return _split_helper_list(tensor, list(indices_or_sections), axis)
    else:
        # 抛出类型错误异常，指示不支持的 indices_or_sections 类型
        raise TypeError("split_helper: ", type(indices_or_sections))


# 内部辅助函数：根据整数分割张量，沿指定轴进行操作，处理严格模式下的不同情况
def _split_helper_int(tensor, indices_or_sections, axis, strict=False):
    if not isinstance(indices_or_sections, int):
        # 如果 indices_or_sections 不是整数，则抛出未实现的异常，指示不支持的分割情况
        raise NotImplementedError("split: indices_or_sections")
    # 标准化轴索引以适应张量的维度
    axis = _util.normalize_axis_index(axis, tensor.ndim)
    # 获取张量在给定轴上的尺寸和切分段数
    l, n = tensor.shape[axis], indices_or_sections
    
    # 检查切分段数是否小于等于零，如果是则引发数值错误异常
    if n <= 0:
        raise ValueError
    
    # 如果张量长度能够整除切分段数，则每段的数量为 l//n，共有 n 段
    if l % n == 0:
        num, sz = n, l // n
        lst = [sz] * num
    else:
        # 如果不能整除且严格模式开启，则引发数值错误异常
        if strict:
            raise ValueError("array split does not result in an equal division")
    
        # 计算出不整除时的段数和每段的大小
        num, sz = l % n, l // n + 1
        lst = [sz] * num
    
    # 将剩余的段补全为每段大小 sz-1，以保证总段数为 n
    lst += [sz - 1] * (n - num)
    
    # 使用计算得到的段列表 lst 对张量进行分割，按照指定的轴进行分割操作
    return torch.split(tensor, lst, axis)
def _split_helper_list(tensor, indices_or_sections, axis):
    if not isinstance(indices_or_sections, list):
        # 如果 indices_or_sections 不是 list 类型，则抛出 NotImplementedError 异常
        raise NotImplementedError("split: indices_or_sections: list")

    # numpy 需要索引，而 torch 需要各段的长度
    # 另外，numpy 会为超出 shape[axis] 的索引附加大小为零的数组
    # 这里 lst 是一个过滤后的 indices_or_sections 列表
    lst = [x for x in indices_or_sections if x <= tensor.shape[axis]]
    num_extra = len(indices_or_sections) - len(lst)

    # 将 tensor.shape[axis] 加入列表末尾
    lst.append(tensor.shape[axis])

    # 计算各段的长度，并添加到 lst 中
    lst = [
        lst[0],
    ] + [a - b for a, b in zip(lst[1:], lst[:-1])]

    # 补充长度为零的数组，使其长度与原 indices_or_sections 一致
    lst += [0] * num_extra

    # 使用 torch.split 将 tensor 按 lst 切分，并沿指定 axis 返回结果
    return torch.split(tensor, lst, axis)


def array_split(ary: ArrayLike, indices_or_sections, axis=0):
    # 对 array_split 的调用会委托给 _split_helper 函数
    return _split_helper(ary, indices_or_sections, axis)


def split(ary: ArrayLike, indices_or_sections, axis=0):
    # 对 split 的调用会委托给 _split_helper 函数，且启用 strict 模式
    return _split_helper(ary, indices_or_sections, axis, strict=True)


def hsplit(ary: ArrayLike, indices_or_sections):
    if ary.ndim == 0:
        # 如果数组维度为 0，则抛出 ValueError 异常
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")

    # 根据数组维度选择切分轴
    axis = 1 if ary.ndim > 1 else 0

    # 对 hsplit 的调用会委托给 _split_helper 函数，且启用 strict 模式
    return _split_helper(ary, indices_or_sections, axis, strict=True)


def vsplit(ary: ArrayLike, indices_or_sections):
    if ary.ndim < 2:
        # 如果数组维度小于 2，则抛出 ValueError 异常
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")

    # 对 vsplit 的调用会委托给 _split_helper 函数，且严格要求切分轴为 0
    return _split_helper(ary, indices_or_sections, 0, strict=True)


def dsplit(ary: ArrayLike, indices_or_sections):
    if ary.ndim < 3:
        # 如果数组维度小于 3，则抛出 ValueError 异常
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")

    # 对 dsplit 的调用会委托给 _split_helper 函数，且严格要求切分轴为 2
    return _split_helper(ary, indices_or_sections, 2, strict=True)


def kron(a: ArrayLike, b: ArrayLike):
    # 返回 a 和 b 的 Kronecker 乘积
    return torch.kron(a, b)


def vander(x: ArrayLike, N=None, increasing=False):
    # 返回 Vandermonde 矩阵
    return torch.vander(x, N, increasing)


# ### linspace, geomspace, logspace and arange ###


def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num=50,
    endpoint=True,
    retstep=False,
    dtype: Optional[DTypeLike] = None,
    axis=0,
):
    if axis != 0 or retstep or not endpoint:
        # 如果 axis 不为 0，或者 retstep 不为 False，或者 endpoint 不为 True，则抛出 NotImplementedError 异常
        raise NotImplementedError

    if dtype is None:
        # 如果 dtype 为 None，则设置为默认的浮点数类型
        dtype = _dtypes_impl.default_dtypes().float_dtype

    # 注意：如果 start 或 stop 不是标量，则会引发 TypeError
    # 返回从 start 到 stop 等间隔生成的数组
    return torch.linspace(start, stop, num, dtype=dtype)


def geomspace(
    start: ArrayLike,
    stop: ArrayLike,
    num=50,
    endpoint=True,
    dtype: Optional[DTypeLike] = None,
    axis=0,
):
    if axis != 0 or not endpoint:
        # 如果 axis 不为 0，或者 endpoint 不为 True，则抛出 NotImplementedError 异常
        raise NotImplementedError

    # 计算等比数列的基数
    base = torch.pow(stop / start, 1.0 / (num - 1))
    logbase = torch.log(base)

    # 返回以对数刻度分布的数组
    return torch.logspace(
        torch.log(start) / logbase,
        torch.log(stop) / logbase,
        num,
        base=base,
    )


def logspace(
    start,
    stop,
    num=50,
    endpoint=True,
    base=10.0,
    dtype: Optional[DTypeLike] = None,
    axis=0,
):
    if axis != 0 or not endpoint:
        # 如果 axis 不为 0，或者 endpoint 不为 True，则抛出 NotImplementedError 异常
        raise NotImplementedError

    # 返回以对数刻度分布的数组
    return torch.logspace(start, stop, num, base=base, dtype=dtype)


def arange(
    start: Optional[ArrayLikeOrScalar] = None,
    # 停止值（结束值），可以是数组或标量，默认为 None
    stop: Optional[ArrayLikeOrScalar] = None,
    # 步长值，可以是数组或标量，默认为 1
    step: Optional[ArrayLikeOrScalar] = 1,
    # 数据类型，可以是指定的数据类型或 None
    dtype: Optional[DTypeLike] = None,
    # 使用 NotImplementedType，但本例中不可用，必须以关键字方式使用
    *,
    # like 参数，用于指定形状或其他类似特征，但在这里未定义使用方式
    like: NotImplementedType = None,
# 如果步长为零，则抛出 ZeroDivisionError 异常
# 如果 stop 和 start 都为 None，则抛出 TypeError 异常
# 如果 stop 为 None，则将 start 赋值给 stop，同时 start 置为 0
if stop is None:
    # XXX: 如果作为关键字参数传递了 start，则会出错：
    # arange(start=4) 应该会抛出异常（没有 stop），但实际上并没有
    start, stop = 0, start
# 如果 start 为 None，则将其设置为 0
if start is None:
    start = 0

# 确定结果的数据类型
if dtype is None:
    # 如果 start、stop 或 step 中有任何一个是浮点数或浮点数张量，则结果为 float 类型
    dtype = (
        _dtypes_impl.default_dtypes().float_dtype
        if any(_dtypes_impl.is_float_or_fp_tensor(x) for x in (start, stop, step))
        else _dtypes_impl.default_dtypes().int_dtype
    )
# 根据是否复数类型，确定工作数据类型
work_dtype = torch.float64 if dtype.is_complex else dtype

# 如果 start、stop 或 step 中有任何一个是复数或复数张量，则抛出 NotImplementedError 异常
if any(_dtypes_impl.is_complex_or_complex_tensor(x) for x in (start, stop, step)):
    raise NotImplementedError

# 如果范围为空，则返回一个空的张量
if (step > 0 and start > stop) or (step < 0 and start < stop):
    # 空范围
    return torch.empty(0, dtype=dtype)

# 使用给定的 start、stop 和 step 创建一个张量并返回
result = torch.arange(start, stop, step, dtype=work_dtype)
# 根据需要将结果转换为指定的 dtype 类型
result = _util.cast_if_needed(result, dtype)
return result


# ### zeros/ones/empty/full ###


# 创建一个指定形状的空张量
def empty(
    shape,
    dtype: Optional[DTypeLike] = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    # 如果未指定 dtype，则使用默认的 float 类型
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.empty(shape, dtype=dtype)


# NB: *_like 函数故意与 numpy 不同：numpy 的默认值为 subok=True；
# 我们设置 subok=False，并在其他情况下抛出异常。


# 根据给定的原型创建一个空张量
def empty_like(
    prototype: ArrayLike,
    dtype: Optional[DTypeLike] = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape=None,
):
    result = torch.empty_like(prototype, dtype=dtype)
    # 如果指定了 shape，则重新调整结果的形状
    if shape is not None:
        result = result.reshape(shape)
    return result


# 根据指定的值创建一个填充的张量
def full(
    shape,
    fill_value: ArrayLike,
    dtype: Optional[DTypeLike] = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    # 如果 shape 是整数，则转为元组形式
    if isinstance(shape, int):
        shape = (shape,)
    # 如果未指定 dtype，则使用 fill_value 的数据类型
    if dtype is None:
        dtype = fill_value.dtype
    # 如果 shape 不是元组或列表，则转为元组形式
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    return torch.full(shape, fill_value, dtype=dtype)


# 根据给定的张量 a 创建一个填充的张量
def full_like(
    a: ArrayLike,
    fill_value,
    dtype: Optional[DTypeLike] = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape=None,
):
    # XXX: fill_value 将进行广播
    result = torch.full_like(a, fill_value, dtype=dtype)
    # 如果指定了 shape，则重新调整结果的形状
    if shape is not None:
        result = result.reshape(shape)
    return result


# 创建一个指定形状的全 1 张量
def ones(
    shape,
    dtype: Optional[DTypeLike] = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    # 如果未指定 dtype，则使用默认的 float 类型
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.ones(shape, dtype=dtype)


# 根据给定的张量 a 创建一个全 1 张量
def ones_like(
    a: ArrayLike,
    # dtype 参数，用于指定数组的数据类型，如果未指定则为 None
    dtype: Optional[DTypeLike] = None,
    # order 参数，用于指定数组元素在内存中的存储顺序，默认为 "K" (表示元素以尽可能紧凑的顺序存储)
    order: NotImplementedType = "K",
    # subok 参数，指定是否允许子类继承数组的子类，默认为 False (表示不允许)
    subok: NotImplementedType = False,
    # shape 参数，用于指定数组的形状，如果未指定则为 None (表示形状未知或灵活)
    shape=None,
def _xy_helper_corrcoef(x_tensor, y_tensor=None, rowvar=True):
    """Prepare inputs for cov and corrcoef."""

    # 如果提供了y_tensor，则将x_tensor和y_tensor至少视为二维数组
    if y_tensor is not None:
        ndim_extra = 2 - x_tensor.ndim
        # 如果x_tensor维度不够，则添加额外的维度，使其至少为二维数组
        if ndim_extra > 0:
            x_tensor = x_tensor.view((1,) * ndim_extra + x_tensor.shape)
        # 如果不是按行来计算相关系数，并且x_tensor的第一个维度不是1，则转置x_tensor
        if not rowvar and x_tensor.shape[0] != 1:
            x_tensor = x_tensor.mT  # 疑似应为 x_tensor.t()
        x_tensor = x_tensor.clone()

        ndim_extra = 2 - y_tensor.ndim
        # 如果y_tensor维度不够，则添加额外的维度，使其至少为二维数组
        if ndim_extra > 0:
            y_tensor = y_tensor.view((1,) * ndim_extra + y_tensor.shape)
        # 如果不是按行来计算相关系数，并且y_tensor的第一个维度不是1，则转置y_tensor
        if not rowvar and y_tensor.shape[0] != 1:
            y_tensor = y_tensor.mT  # 疑似应为 y_tensor.t()
        y_tensor = y_tensor.clone()

        # 将x_tensor和y_tensor在指定轴上连接起来
        x_tensor = _concatenate((x_tensor, y_tensor), axis=0)

    return x_tensor


def corrcoef(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    rowvar=True,
    bias=None,
    ddof=None,
    *,
    dtype: Optional[DTypeLike] = None,
):
    # 如果提供了bias或ddof参数，则抛出NotImplementedError
    if bias is not None or ddof is not None:
        raise NotImplementedError

    # 准备输入以计算相关系数，返回连接后的tensor
    xy_tensor = _xy_helper_corrcoef(x, y, rowvar)

    is_half = (xy_tensor.dtype == torch.float16) and xy_tensor.is_cpu
    if is_half:
        # 由于torch的"addmm_impl_cpu_"方法不支持'Half'类型，因此需要使用float32类型进行计算
        dtype = torch.float32

    # 将tensor转换为指定dtype类型（如果有的话）
    xy_tensor = _util.cast_if_needed(xy_tensor, dtype)
    # 计算相关系数
    result = torch.corrcoef(xy_tensor)

    if is_half:
        # 如果之前使用了float32类型进行计算，则将结果转换回float16类型
        result = result.to(torch.float16)

    return result


def cov(
    m: ArrayLike,
    y: Optional[ArrayLike] = None,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights: Optional[ArrayLike] = None,
    aweights: Optional[ArrayLike] = None,
    *,
    dtype: Optional[DTypeLike] = None,
):
    # 准备输入以计算协方差，返回连接后的tensor
    m = _xy_helper_corrcoef(m, y, rowvar)

    # 如果未提供ddof参数，则根据bias值来确定
    if ddof is None:
        ddof = 1 if bias == 0 else 0

    is_half = (m.dtype == torch.float16) and m.is_cpu
    if is_half:
        # 由于torch的"addmm_impl_cpu_"方法不支持'Half'类型，因此需要使用float32类型进行计算
        dtype = torch.float32

    # 将tensor转换为指定dtype类型（如果有的话）
    m = _util.cast_if_needed(m, dtype)
    # 计算协方差
    result = torch.cov(m, correction=ddof, aweights=aweights, fweights=fweights)

    if is_half:
        # 如果之前使用了float32类型进行计算，则将结果转换回float16类型
        result = result.to(torch.float16)

    return result
    返回函数的结果变量result
def _conv_corr_impl(a, v, mode):
    # 确定输入数组 a 和 v 的数据类型
    dt = _dtypes_impl.result_type_impl(a, v)
    # 如果需要，将数组 a 和 v 转换为指定的数据类型 dt
    a = _util.cast_if_needed(a, dt)
    v = _util.cast_if_needed(v, dt)

    # 根据模式 mode 确定填充值的数量
    padding = v.shape[0] - 1 if mode == "full" else mode

    if padding == "same" and v.shape[0] % 2 == 0:
        # 如果使用 'same' 模式且卷积核长度为偶数，抛出未实现错误
        raise NotImplementedError("mode='same' and even-length weights")

    # 将输入数组 aa 转换为 2D 输入形式
    aa = a[None, :]
    # 将卷积核数组 vv 转换为 3D 卷积核形式
    vv = v[None, None, :]

    # 使用 PyTorch 执行 1D 卷积操作，返回结果
    result = torch.nn.functional.conv1d(aa, vv, padding=padding)

    # 返回结果中的第一个元素，因为 PyTorch 返回一个 2D 结果而 NumPy 返回 1D 数组
    return result[0, :]


def convolve(a: ArrayLike, v: ArrayLike, mode="full"):
    # 如果 v 的长度比 a 长，交换数组 a 和 v
    if a.shape[0] < v.shape[0]:
        a, v = v, a

    # 翻转卷积核 v，因为 NumPy 翻转而 PyTorch 不翻转
    v = torch.flip(v, (0,))

    # 调用 _conv_corr_impl 执行卷积操作，并返回结果
    return _conv_corr_impl(a, v, mode)


def correlate(a: ArrayLike, v: ArrayLike, mode="valid"):
    # 对卷积核 v 进行共轭操作（仅适用于物理值），然后调用 _conv_corr_impl 执行相关操作
    v = torch.conj_physical(v)
    return _conv_corr_impl(a, v, mode)


# ### logic & element selection ###


def bincount(x: ArrayLike, /, weights: Optional[ArrayLike] = None, minlength=0):
    if x.numel() == 0:
        # 处理特殊情况：当 x 中没有元素时，创建一个空的数组
        x = x.new_empty(0, dtype=int)

    # 将数组 x 的数据类型转换为整数类型
    int_dtype = _dtypes_impl.default_dtypes().int_dtype
    (x,) = _util.typecast_tensors((x,), int_dtype, casting="safe")

    # 使用 PyTorch 的 bincount 函数计算 x 中各元素出现的次数，返回结果
    return torch.bincount(x, weights, minlength)


def where(
    condition: ArrayLike,
    x: Optional[ArrayLikeOrScalar] = None,
    y: Optional[ArrayLikeOrScalar] = None,
    /,
):
    # 如果只给定 x 或 y 中的一个而不是两者同时给定，则抛出值错误
    if (x is None) != (y is None):
        raise ValueError("either both or neither of x and y should be given")

    # 如果条件数组的数据类型不是布尔类型，则转换为布尔类型
    if condition.dtype != torch.bool:
        condition = condition.to(torch.bool)

    if x is None and y is None:
        # 如果未给定 x 和 y，则使用条件数组进行 where 操作，返回结果
        result = torch.where(condition)
    else:
        # 否则，使用条件数组、x 和 y 进行 where 操作，返回结果
        result = torch.where(condition, x, y)
    return result


# ###### module-level queries of object properties


def ndim(a: ArrayLike):
    # 返回数组 a 的维度数量
    return a.ndim


def shape(a: ArrayLike):
    # 返回数组 a 的形状（尺寸）
    return tuple(a.shape)


def size(a: ArrayLike, axis=None):
    if axis is None:
        # 如果未指定轴，则返回数组 a 中元素的总数
        return a.numel()
    else:
        # 否则，返回指定轴上的尺寸大小
        return a.shape[axis]


# ###### shape manipulations and indexing


def expand_dims(a: ArrayLike, axis):
    # 根据指定的轴扩展数组 a 的维度
    shape = _util.expand_shape(a.shape, axis)
    return a.view(shape)  # 永远不复制数据，只是改变视图


def flip(m: ArrayLike, axis=None):
    # XXX: 语义上的区别：np.flip 返回一个视图，torch.flip 复制数据
    if axis is None:
        axis = tuple(range(m.ndim))
    else:
        axis = _util.normalize_axis_tuple(axis, m.ndim)
    return torch.flip(m, axis)


def flipud(m: ArrayLike):
    # 上下翻转数组 m
    return torch.flipud(m)


def fliplr(m: ArrayLike):
    # 左右翻转数组 m
    return torch.fliplr(m)
# 定义一个函数 `rot90`，旋转数组 `m` 90 度。
def rot90(m: ArrayLike, k=1, axes=(0, 1)):
    # 规范化轴元组，确保在数组维度范围内
    axes = _util.normalize_axis_tuple(axes, m.ndim)
    # 使用 torch 库中的 `rot90` 函数进行数组旋转
    return torch.rot90(m, k, axes)


# ### broadcasting and indices ###


# 将输入的数组 `array` 广播到指定的 `shape` 大小
def broadcast_to(array: ArrayLike, shape, subok: NotImplementedType = False):
    return torch.broadcast_to(array, size=shape)


# 导入 torch 库中的 `broadcast_shapes` 函数，用于广播数组
from torch import broadcast_shapes


# 广播输入的多个数组 `args`，返回广播后的张量元组
def broadcast_arrays(*args: ArrayLike, subok: NotImplementedType = False):
    return torch.broadcast_tensors(*args)


# 根据输入的数组 `xi` 创建网格
def meshgrid(*xi: ArrayLike, copy=True, sparse=False, indexing="xy"):
    ndim = len(xi)

    # 检查 `indexing` 参数是否有效
    if indexing not in ["xy", "ij"]:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1 :]) for i, x in enumerate(xi)]

    # 如果 `indexing` 为 'xy' 并且维度大于 1，则交换第一和第二个轴
    if indexing == "xy" and ndim > 1:
        output[0] = output[0].reshape((1, -1) + s0[2:])
        output[1] = output[1].reshape((-1, 1) + s0[2:])

    # 如果 `sparse` 为 False，则返回完整的 N-D 矩阵
    if not sparse:
        output = torch.broadcast_tensors(*output)

    # 如果 `copy` 为 True，则对输出的每个数组进行克隆操作
    if copy:
        output = [x.clone() for x in output]

    return list(output)  # 返回与 numpy 匹配的列表


# 根据给定的 `dimensions` 创建索引数组
def indices(dimensions, dtype: Optional[DTypeLike] = int, sparse=False):
    # 将 `dimensions` 转换为元组
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N

    # 如果 `sparse` 为 True，则返回空元组；否则创建指定形状的空张量
    if sparse:
        res = tuple()
    else:
        res = torch.empty((N,) + dimensions, dtype=dtype)

    # 遍历 `dimensions`，创建每个维度的索引数组并填充到结果 `res` 中
    for i, dim in enumerate(dimensions):
        idx = torch.arange(dim, dtype=dtype).reshape(
            shape[:i] + (dim,) + shape[i + 1 :]
        )
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx

    return res


# ### tri*-something ###


# 返回矩阵 `m` 的下三角部分
def tril(m: ArrayLike, k=0):
    return torch.tril(m, k)


# 返回矩阵 `m` 的上三角部分
def triu(m: ArrayLike, k=0):
    return torch.triu(m, k)


# 返回下三角矩阵的索引
def tril_indices(n, k=0, m=None):
    if m is None:
        m = n
    return torch.tril_indices(n, m, offset=k)


# 返回上三角矩阵的索引
def triu_indices(n, k=0, m=None):
    if m is None:
        m = n
    return torch.triu_indices(n, m, offset=k)


# 返回输入数组 `arr` 的下三角部分的索引
def tril_indices_from(arr: ArrayLike, k=0):
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    # 返回张量而不是元组，以避免图断裂
    return torch.tril_indices(arr.shape[0], arr.shape[1], offset=k)


# 返回输入数组 `arr` 的上三角部分的索引
def triu_indices_from(arr: ArrayLike, k=0):
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    # 返回张量而不是元组，以避免图断裂
    return torch.triu_indices(arr.shape[0], arr.shape[1], offset=k)


# 返回一个 N x M 的三角矩阵，其对角线偏移量为 `k`
def tri(
    N,
    M=None,
    k=0,
    dtype: Optional[DTypeLike] = None,
    *,
    like: NotImplementedType = None,
):
    if M is None:
        M = N
    tensor = torch.ones((N, M), dtype=dtype)
    return torch.tril(tensor, diagonal=k)
# ### equality, equivalence, allclose ###

# 判断两个数组是否在误差范围内相等
def isclose(a: ArrayLike, b: ArrayLike, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    # 确定数组的数据类型
    dtype = _dtypes_impl.result_type_impl(a, b)
    # 根据需要将数组转换为指定的数据类型
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    # 使用 torch.isclose 判断数组是否在指定的相对和绝对误差范围内相等
    return torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


# 判断两个数组是否在误差范围内全相等
def allclose(a: ArrayLike, b: ArrayLike, rtol=1e-05, atol=1e-08, equal_nan=False):
    # 确定数组的数据类型
    dtype = _dtypes_impl.result_type_impl(a, b)
    # 根据需要将数组转换为指定的数据类型
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    # 使用 torch.allclose 判断数组是否在指定的相对和绝对误差范围内全相等
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


# 比较两个张量是否元素级别相等（支持处理 NaN）
def _tensor_equal(a1, a2, equal_nan=False):
    # 如果两个张量的形状不同，则它们不相等
    if a1.shape != a2.shape:
        return False
    # 比较两个张量是否元素级别相等，如果指定了 equal_nan 为 True，则处理 NaN 值
    cond = a1 == a2
    if equal_nan:
        cond = cond | (torch.isnan(a1) & torch.isnan(a2))
    return cond.all().item()


# 比较两个数组是否在误差范围内元素级别相等（支持处理 NaN）
def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan=False):
    return _tensor_equal(a1, a2, equal_nan=equal_nan)


# 比较两个数组是否在误差范围内元素级别等效（支持广播）
def array_equiv(a1: ArrayLike, a2: ArrayLike):
    # 尝试广播两个数组，如果无法广播，则它们不等效
    try:
        a1_t, a2_t = torch.broadcast_tensors(a1, a2)
    except RuntimeError:
        # 广播失败 => 不等效
        return False
    # 比较广播后的两个张量是否元素级别相等
    return _tensor_equal(a1_t, a2_t)


# 将数组中的 NaN 替换为指定的数值（支持复数）
def nan_to_num(
    x: ArrayLike, copy: NotImplementedType = True, nan=0.0, posinf=None, neginf=None
):
    # 如果数组是复数类型，分别处理实部和虚部的 NaN
    if x.is_complex():
        re = torch.nan_to_num(x.real, nan=nan, posinf=posinf, neginf=neginf)
        im = torch.nan_to_num(x.imag, nan=nan, posinf=posinf, neginf=neginf)
        return re + 1j * im
    else:
        # 否则直接使用 torch.nan_to_num 处理数组中的 NaN
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


# ### put/take_along_axis ###

# 在指定轴上获取数组的元素
def take(
    a: ArrayLike,
    indices: ArrayLike,
    axis=None,
    out: Optional[OutArray] = None,
    mode: NotImplementedType = "raise",
):
    # 将数组展平，同时处理轴参数
    (a,), axis = _util.axis_none_flatten(a, axis=axis)
    # 规范化轴索引
    axis = _util.normalize_axis_index(axis, a.ndim)
    # 构造索引元组并获取相应元素
    idx = (slice(None),) * axis + (indices, ...)
    result = a[idx]
    return result


# 在指定轴上获取数组的元素（支持 torch 的 take_along_dim 方法）
def take_along_axis(arr: ArrayLike, indices: ArrayLike, axis):
    # 将数组展平，同时处理轴参数
    (arr,), axis = _util.axis_none_flatten(arr, axis=axis)
    # 规范化轴索引
    axis = _util.normalize_axis_index(axis, arr.ndim)
    # 使用 torch.take_along_dim 在指定轴上获取数组的元素
    return torch.take_along_dim(arr, indices, axis)


# 将指定的值放置到数组的指定位置
def put(
    a: NDArray,
    indices: ArrayLike,
    values: ArrayLike,
    mode: NotImplementedType = "raise",
):
    # 将 values 转换为与 a 相同的数据类型
    v = values.type(a.dtype)
    # 如果 indices 的元素数量大于 values 的元素数量，则扩展 values 以匹配 indices
    if indices.numel() > v.numel():
        ratio = (indices.numel() + v.numel() - 1) // v.numel()
        v = v.unsqueeze(0).expand((ratio,) + v.shape)
    # 返回放置好的数组 a
    # 注意：任何多余的尾部元素将被截断，这与 np.put() 的默认行为相同
    # 如果 indices 的元素数量小于 v 的元素数量，则进入条件判断
    if indices.numel() < v.numel():
        # 将 v 展平为一维数组
        v = v.flatten()
        # 仅保留 v 中与 indices 元素数量相同的部分
        v = v[: indices.numel()]
    
    # 将 v 中的值按照 indices 指定的位置放入张量 a 中
    a.put_(indices, v)
    
    # 返回 None
    return None
def put_along_axis(arr: ArrayLike, indices: ArrayLike, values: ArrayLike, axis):
    # 将 arr 扁平化并标准化轴索引
    (arr,), axis = _util.axis_none_flatten(arr, axis=axis)
    axis = _util.normalize_axis_index(axis, arr.ndim)

    # 广播 indices 和 values
    indices, values = torch.broadcast_tensors(indices, values)
    # 将 values 转换为 arr 的数据类型
    values = _util.cast_if_needed(values, arr.dtype)
    # 在 arr 上执行 scatter 操作，将 values 根据 indices 分散到 axis 轴上
    result = torch.scatter(arr, axis, indices, values)
    # 将结果重新形状为 arr 的形状并拷贝回 arr
    arr.copy_(result.reshape(arr.shape))
    # 返回空值
    return None


def choose(
    a: ArrayLike,
    choices: Sequence[ArrayLike],
    out: Optional[OutArray] = None,
    mode: NotImplementedType = "raise",
):
    # 首先，广播 choices 的元素
    choices = torch.stack(torch.broadcast_tensors(*choices))

    # 使用类似 gather(choices, 0, a) 的方法，对 a 和 choices 进行广播：
    # （参考自 https://github.com/pytorch/pytorch/issues/9407#issuecomment-1427907939）
    idx_list = [
        torch.arange(dim).view((1,) * i + (dim,) + (1,) * (choices.ndim - i - 1))
        for i, dim in enumerate(choices.shape)
    ]

    idx_list[0] = a
    # 按照 idx_list 的索引获取 choices 的值并压缩维度
    return choices[idx_list].squeeze(0)


# ### unique et al. ###


def unique(
    ar: ArrayLike,
    return_index: NotImplementedType = False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    *,
    equal_nan: NotImplementedType = True,
):
    # 将 ar 扁平化并标准化轴索引
    (ar,), axis = _util.axis_none_flatten(ar, axis=axis)
    axis = _util.normalize_axis_index(axis, ar.ndim)

    # 使用 torch.unique 函数获取唯一值
    result = torch.unique(
        ar, return_inverse=return_inverse, return_counts=return_counts, dim=axis
    )

    return result


def nonzero(a: ArrayLike):
    # 返回非零元素的索引，作为元组形式返回
    return torch.nonzero(a, as_tuple=True)


def argwhere(a: ArrayLike):
    # 返回非零元素的索引
    return torch.argwhere(a)


def flatnonzero(a: ArrayLike):
    # 返回扁平化后的非零元素索引，作为元组形式返回
    return torch.flatten(a).nonzero(as_tuple=True)[0]


def clip(
    a: ArrayLike,
    min: Optional[ArrayLike] = None,
    max: Optional[ArrayLike] = None,
    out: Optional[OutArray] = None,
):
    # 使用 torch.clamp 对 a 进行限制在 [min, max] 范围内的操作
    return torch.clamp(a, min, max)


def repeat(a: ArrayLike, repeats: ArrayLikeOrScalar, axis=None):
    # 使用 torch.repeat_interleave 对 a 进行重复值操作
    return torch.repeat_interleave(a, repeats, axis)


def tile(A: ArrayLike, reps):
    if isinstance(reps, int):
        reps = (reps,)
    # 使用 torch.tile 对 A 进行多次重复操作
    return torch.tile(A, reps)


def resize(a: ArrayLike, new_shape=None):
    # 实现来自 numpy 的 resize 方法，将 a 重新调整为新的形状
    if new_shape is None:
        return a

    if isinstance(new_shape, int):
        new_shape = (new_shape,)

    a = a.flatten()

    new_size = 1
    for dim_length in new_shape:
        new_size *= dim_length
        if dim_length < 0:
            raise ValueError("all elements of `new_shape` must be non-negative")

    if a.numel() == 0 or new_size == 0:
        # 如果 a 为空或者 new_size 为零，则返回相应形状的零张量
        return torch.zeros(new_shape, dtype=a.dtype)

    # 计算需要的重复次数并扩展 a
    repeats = -(-new_size // a.numel())  # ceil division
    a = concatenate((a,) * repeats)[:new_size]

    return reshape(a, new_shape)


# ### diag et al. ###
# 返回数组的对角线元素，可以通过指定偏移量和轴来控制对角线的位置
def diagonal(a: ArrayLike, offset=0, axis1=0, axis2=1):
    # 规范化轴的索引，确保在数组维度范围内
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    # 调用 torch 的 diagonal 函数返回对角线元素
    return torch.diagonal(a, offset, axis1, axis2)


# 计算数组的迹（对角线元素之和）
def trace(
    a: ArrayLike,
    offset=0,
    axis1=0,
    axis2=1,
    dtype: Optional[DTypeLike] = None,
    out: Optional[OutArray] = None,
):
    # 调用 torch 的 diagonal 函数获取对角线元素，然后求和
    result = torch.diagonal(a, offset, dim1=axis1, dim2=axis2).sum(-1, dtype=dtype)
    return result


# 创建一个单位矩阵
def eye(
    N,
    M=None,
    k=0,
    dtype: Optional[DTypeLike] = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    # 如果未指定数据类型，使用默认的浮点类型
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    # 如果 M 未指定，使用 N 的值
    if M is None:
        M = N
    # 创建一个 N x M 的零矩阵，并设置对角线上的元素为 1
    z = torch.zeros(N, M, dtype=dtype)
    z.diagonal(k).fill_(1)
    return z


# 创建一个 n x n 的单位矩阵
def identity(n, dtype: Optional[DTypeLike] = None, *, like: NotImplementedType = None):
    return torch.eye(n, dtype=dtype)


# 提取数组的对角线元素
def diag(v: ArrayLike, k=0):
    return torch.diag(v, k)


# 创建一个从数组中提取出的对角线数组
def diagflat(v: ArrayLike, k=0):
    return torch.diagflat(v, k)


# 返回给定形状的对角线索引
def diag_indices(n, ndim=2):
    idx = torch.arange(n)
    return (idx,) * ndim


# 返回给定数组的对角线索引
def diag_indices_from(arr: ArrayLike):
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # 检查数组各维度是否相等
    s = arr.shape
    if s[1:] != s[:-1]:
        raise ValueError("All dimensions of input must be of equal length")
    # 返回数组的对角线索引
    return diag_indices(s[0], arr.ndim)


# 填充数组的对角线
def fill_diagonal(a: ArrayLike, val: ArrayLike, wrap=False):
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    if val.numel() == 0 and not wrap:
        # 使用给定值填充数组的对角线
        a.fill_diagonal_(val)
        return a

    if val.ndim == 0:
        val = val.unsqueeze(0)

    # 如果数组是二维的，根据 wrap 参数填充对角线
    if a.ndim == 2:
        tall = a.shape[0] > a.shape[1]
        if not wrap or not tall:
            # 填充正常的对角线
            diag = a.diagonal()
            diag.copy_(val[: diag.numel()])
        else:
            # 使用 wrap 选项填充高宽比较大的情况
            max_, min_ = a.shape
            idx = torch.arange(max_ - max_ // (min_ + 1))
            mod = idx % min_
            div = idx // min_
            a[(div * (min_ + 1) + mod, mod)] = val[: idx.numel()]
    else:
        # 对于更高维度的数组，填充其对角线
        idx = diag_indices_from(a)
        a[idx] = val[: a.shape[0]]

    return a


# 计算两个数组的向量内积
def vdot(a: ArrayLike, b: ArrayLike, /):
    # 调用 torch 的 atleast_1d 函数确保输入是 1 维数组
    t_a, t_b = torch.atleast_1d(a, b)
    # 如果数组超过 1 维，将其展平
    if t_a.ndim > 1:
        t_a = t_a.flatten()
    if t_b.ndim > 1:
        t_b = t_b.flatten()
    # 确定返回的数据类型
    dtype = _dtypes_impl.result_type_impl(t_a, t_b)
    # 检查数据类型是否为 torch.float16 并且至少一个张量在 CPU 上
    is_half = dtype == torch.float16 and (t_a.is_cpu or t_b.is_cpu)
    
    # 检查数据类型是否为 torch.bool
    is_bool = dtype == torch.bool
    
    # 为了解决 torch 中对 'Half' 和 'Bool' 类型未实现 "dot" 操作的问题
    if is_half:
        # 将数据类型调整为 torch.float32
        dtype = torch.float32
    elif is_bool:
        # 将数据类型调整为 torch.uint8
        dtype = torch.uint8
    
    # 如果需要，将输入张量 t_a 转换为指定的数据类型 dtype
    t_a = _util.cast_if_needed(t_a, dtype)
    
    # 如果需要，将输入张量 t_b 转换为指定的数据类型 dtype
    t_b = _util.cast_if_needed(t_b, dtype)
    
    # 使用 torch.vdot 计算张量 t_a 和 t_b 的点积
    result = torch.vdot(t_a, t_b)
    
    # 根据之前的数据类型判断，如果是半精度（torch.float16），则将结果转换回 torch.float16
    if is_half:
        result = result.to(torch.float16)
    # 如果是布尔类型（torch.bool），则将结果转换回 torch.bool
    elif is_bool:
        result = result.to(torch.bool)
    
    # 返回计算的结果
    return result
# 定义函数 `tensordot`，计算张量点积
def tensordot(a: ArrayLike, b: ArrayLike, axes=2):
    # 如果 axes 是 list 或 tuple，则对每个元素检查是否是 int，不是则将其包装成 list
    if isinstance(axes, (list, tuple)):
        axes = [[ax] if isinstance(ax, int) else ax for ax in axes]

    # 确定目标数据类型
    target_dtype = _dtypes_impl.result_type_impl(a, b)
    # 根据目标数据类型转换 a 和 b
    a = _util.cast_if_needed(a, target_dtype)
    b = _util.cast_if_needed(b, target_dtype)

    # 使用 torch 的 tensordot 函数计算张量点积
    return torch.tensordot(a, b, dims=axes)


# 定义函数 `dot`，计算向量或矩阵的点积
def dot(a: ArrayLike, b: ArrayLike, out: Optional[OutArray] = None):
    # 确定结果数据类型
    dtype = _dtypes_impl.result_type_impl(a, b)
    # 检查是否为布尔型数据
    is_bool = dtype == torch.bool
    # 如果是布尔型，则转换为 uint8 类型
    if is_bool:
        dtype = torch.uint8

    # 根据数据类型转换 a 和 b
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    # 根据 a 和 b 的维度进行不同的计算
    if a.ndim == 0 or b.ndim == 0:
        result = a * b  # 对标量进行乘法运算
    else:
        result = torch.matmul(a, b)  # 对向量或矩阵进行矩阵乘法运算

    # 如果输入数据是布尔型，则结果转换为布尔型
    if is_bool:
        result = result.to(torch.bool)

    return result


# 定义函数 `inner`，计算向量的内积
def inner(a: ArrayLike, b: ArrayLike, /):
    # 确定结果数据类型
    dtype = _dtypes_impl.result_type_impl(a, b)
    # 检查是否是半精度浮点数且在 CPU 上运算
    is_half = dtype == torch.float16 and (a.is_cpu or b.is_cpu)
    # 检查是否为布尔型数据
    is_bool = dtype == torch.bool

    # 如果是半精度浮点数，则转换为 float32 类型处理
    if is_half:
        dtype = torch.float32
    elif is_bool:
        dtype = torch.uint8

    # 根据数据类型转换 a 和 b
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    # 使用 torch 的 inner 函数计算向量的内积
    result = torch.inner(a, b)

    # 如果输入数据是半精度浮点数，则结果转换为 float16
    if is_half:
        result = result.to(torch.float16)
    elif is_bool:
        result = result.to(torch.bool)
    return result


# 定义函数 `outer`，计算向量的外积
def outer(a: ArrayLike, b: ArrayLike, out: Optional[OutArray] = None):
    # 使用 torch 的 outer 函数计算向量的外积
    return torch.outer(a, b)


# 定义函数 `cross`，计算向量的叉积
def cross(a: ArrayLike, b: ArrayLike, axisa=-1, axisb=-1, axisc=-1, axis=None):
    # 从 https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1486-L1685 导入的实现方式
    # 如果 axis 参数不为 None，则将 axisa、axisb、axisc 都设置为 axis
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3

    # 将 axisa 和 axisb 规范化为合法的索引
    axisa = _util.normalize_axis_index(axisa, a.ndim)
    axisb = _util.normalize_axis_index(axisb, b.ndim)

    # 将工作轴移动到数组形状的末尾
    a = torch.moveaxis(a, axisa, -1)
    b = torch.moveaxis(b, axisb, -1)

    # 检查叉积计算所需的维度是否符合要求
    msg = "incompatible dimensions for cross product\n(dimension must be 2 or 3)"
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        raise ValueError(msg)

    # 创建输出数组
    shape = broadcast_shapes(a[..., 0].shape, b[..., 0].shape)
    if a.shape[-1] == 3 or b.shape[-1] == 3:
        shape += (3,)
        # 将 axisc 规范化为合法的索引
        axisc = _util.normalize_axis_index(axisc, len(shape))
    dtype = _dtypes_impl.result_type_impl(a, b)
    cp = torch.empty(shape, dtype=dtype)

    # 根据数据类型转换 a 和 b
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    # 为提升可读性创建局部别名
    a0 = a[..., 0]
    a1 = a[..., 1]
    if a.shape[-1] == 3:
        a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    if b.shape[-1] == 3:
        b2 = b[..., 2]
    # 检查 cp 的维度不为零且最后一个维度为 3
    if cp.ndim != 0 and cp.shape[-1] == 3:
        # 取出 cp 的第一个通道数据
        cp0 = cp[..., 0]
        # 取出 cp 的第二个通道数据
        cp1 = cp[..., 1]
        # 取出 cp 的第三个通道数据
        cp2 = cp[..., 2]

    # 检查 a 的最后一个维度是否为 2
    if a.shape[-1] == 2:
        # 如果 b 的最后一个维度也为 2
        if b.shape[-1] == 2:
            # 计算叉乘结果 a0 * b1 - a1 * b0，结果存入 cp 中
            cp[...] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
            # 返回计算结果 cp
            return cp
        else:
            # 否则，确保 b 的最后一个维度为 3
            assert b.shape[-1] == 3
            # 计算叉乘的部分结果并存入 cp0, cp1, cp2 中
            cp0[...] = a[..., 1] * b[..., 2]
            cp1[...] = -a[..., 0] * b[..., 2]
            cp2[...] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    else:
        # 否则，确保 a 的最后一个维度为 3
        assert a.shape[-1] == 3
        # 确保 b 的最后一个维度也为 3
        if b.shape[-1] == 3:
            # 计算叉乘的部分结果并存入 cp0, cp1, cp2 中
            cp0[...] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
            cp1[...] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
            cp2[...] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        else:
            # 否则，确保 b 的最后一个维度为 2
            assert b.shape[-1] == 2
            # 计算叉乘的部分结果并存入 cp0, cp1, cp2 中
            cp0[...] = -a[..., 2] * b[..., 1]
            cp1[...] = a[..., 2] * b[..., 0]
            cp2[...] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    # 返回 cp 在最后一个维度上移动到 axisc 位置的结果
    return torch.moveaxis(cp, -1, axisc)
# 根据传入的参数定义一个 einsum 函数，支持多个操作数以及各种选项
def einsum(*operands, out=None, dtype=None, order="K", casting="safe", optimize=False):
    # 从本地模块中导入所需的函数和类，以避免污染全局空间，这些将在 funcs.py 中导出
    from ._ndarray import ndarray
    from ._normalizations import (
        maybe_copy_to,
        normalize_array_like,
        normalize_casting,
        normalize_dtype,
        wrap_tensors,
    )

    # 规范化 dtype 参数，确保其符合要求
    dtype = normalize_dtype(dtype)
    # 规范化 casting 参数，确保其符合要求
    casting = normalize_casting(casting)
    # 如果指定了 out 参数，确保其为 ndarray 类型，否则抛出类型错误异常
    if out is not None and not isinstance(out, ndarray):
        raise TypeError("'out' must be an array")
    # 目前不支持除 "K" 外的 order 参数，抛出未实现异常
    if order != "K":
        raise NotImplementedError("'order' parameter is not supported.")

    # 解析操作数并进行规范化
    sublist_format = not isinstance(operands[0], str)
    if sublist_format:
        # 对于 op, str, op, str ... [sublistout] 格式，规范化每个奇数位置的操作数
        # 如果没有给出 sublistout，则操作数的长度是偶数，我们选择奇数位置的元素（数组）。
        # 如果给出了 sublistout，则操作数的长度是奇数，我们取出最后一个元素，并选择奇数位置的元素（数组）。
        # 没有 [:-1]，我们将会选择 sublistout。
        array_operands = operands[:-1][::2]
    else:
        # 对于 ("ij->", arrays) 格式，第一个元素是子脚本，其余是数组
        subscripts, array_operands = operands[0], operands[1:]

    # 规范化数组操作数，确保它们符合要求
    tensors = [normalize_array_like(op) for op in array_operands]
    # 如果未指定 dtype，则根据操作数推断目标 dtype
    target_dtype = _dtypes_impl.result_type_impl(*tensors) if dtype is None else dtype

    # 对于目标 dtype 是 torch.float16 且所有张量都在 CPU 上的情况，将目标 dtype 更改为 torch.float32
    is_half = target_dtype == torch.float16 and all(t.is_cpu for t in tensors)
    if is_half:
        target_dtype = torch.float32

    # 如果目标 dtype 是 [torch.uint8, torch.int8, torch.int16, torch.int32] 中的一种，将目标 dtype 更改为 torch.int64
    is_short_int = target_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32]
    if is_short_int:
        target_dtype = torch.int64

    # 将张量转换为目标 dtype，使用指定的 casting 规则
    tensors = _util.typecast_tensors(tensors, target_dtype, casting)

    # 从 torch.backends 中导入 opt_einsum 函数，进行优化的 einsum 计算
    from torch.backends import opt_einsum
    # 尝试执行以下代码块，用于设置处理 optimize=... 参数的全局状态，并在退出时恢复原状态
    if opt_einsum.is_available():
        # 保存当前的 torch.backends.opt_einsum.strategy 和 torch.backends.opt_einsum.enabled 状态
        old_strategy = torch.backends.opt_einsum.strategy
        old_enabled = torch.backends.opt_einsum.enabled

        # 处理 optimize 参数为 True 或 False 的情况
        if optimize is True:
            optimize = "auto"
        elif optimize is False:
            # 如果 optimize 参数为 False，则禁用 torch.backends.opt_einsum
            torch.backends.opt_einsum.enabled = False

        # 设置新的优化策略
        torch.backends.opt_einsum.strategy = optimize

    if sublist_format:
        # 重新组合操作数，从操作数列表中获取子列表
        sublists = operands[1::2]
        # 检查操作数列表长度是否为奇数，确定是否有 sublistout
        has_sublistout = len(operands) % 2 == 1
        if has_sublistout:
            sublistout = operands[-1]

        # 将 tensors 和 sublists 组合成新的操作数列表
        operands = list(itertools.chain.from_iterable(zip(tensors, sublists)))
        if has_sublistout:
            operands.append(sublistout)

        # 使用 torch.einsum 执行张量乘法操作
        result = torch.einsum(*operands)
    else:
        # 使用给定的 subscripts 和 tensors 执行 torch.einsum 操作
        result = torch.einsum(subscripts, *tensors)

    # 在 finally 语句块中，恢复旧的 torch.backends.opt_einsum 策略和状态
    finally:
        if opt_einsum.is_available():
            torch.backends.opt_einsum.strategy = old_strategy
            torch.backends.opt_einsum.enabled = old_enabled

    # 将结果复制到输出张量 out，然后返回结果
    result = maybe_copy_to(out, result)
    return wrap_tensors(result)
# ### sort and partition ###

# `_sort_helper`函数用于辅助排序操作，检查复杂数据类型并标准化轴索引
def _sort_helper(tensor, axis, kind, order):
    # 如果数据类型为复数，则不支持排序，抛出异常
    if tensor.dtype.is_complex:
        raise NotImplementedError(f"sorting {tensor.dtype} is not supported")
    # 调用_util模块的axis_none_flatten函数，用于处理轴参数
    (tensor,), axis = _util.axis_none_flatten(tensor, axis=axis)
    # 标准化轴索引
    axis = _util.normalize_axis_index(axis, tensor.ndim)

    # 是否稳定排序
    stable = kind == "stable"

    return tensor, axis, stable


# `sort`函数用于对数组进行排序
def sort(a: ArrayLike, axis=-1, kind=None, order: NotImplementedType = None):
    # 排序辅助函数，返回排序后的结果
    a, axis, stable = _sort_helper(a, axis, kind, order)
    # 调用torch库的排序函数，按指定轴和稳定性进行排序
    result = torch.sort(a, dim=axis, stable=stable)
    # 返回排序后的数值部分
    return result.values


# `argsort`函数用于返回数组排序后的索引
def argsort(a: ArrayLike, axis=-1, kind=None, order: NotImplementedType = None):
    # 排序辅助函数，返回排序后的结果
    a, axis, stable = _sort_helper(a, axis, kind, order)
    # 调用torch库的argsort函数，按指定轴和稳定性进行排序，返回索引
    return torch.argsort(a, dim=axis, stable=stable)


# `searchsorted`函数用于在有序数组中查找元素应插入的位置
def searchsorted(
    a: ArrayLike, v: ArrayLike, side="left", sorter: Optional[ArrayLike] = None
):
    # 如果数组数据类型为复数，则不支持searchsorted操作，抛出异常
    if a.dtype.is_complex:
        raise NotImplementedError(f"searchsorted with dtype={a.dtype}")

    # 调用torch库的searchsorted函数，在数组a中查找v的插入位置
    return torch.searchsorted(a, v, side=side, sorter=sorter)


# ### swap/move/roll axis ###

# `moveaxis`函数用于移动数组的轴
def moveaxis(a: ArrayLike, source, destination):
    # 标准化源轴的索引
    source = _util.normalize_axis_tuple(source, a.ndim, "source")
    # 标准化目标轴的索引
    destination = _util.normalize_axis_tuple(destination, a.ndim, "destination")
    # 调用torch库的moveaxis函数，实现轴的移动操作
    return torch.moveaxis(a, source, destination)


# `swapaxes`函数用于交换数组的两个轴
def swapaxes(a: ArrayLike, axis1, axis2):
    # 标准化轴1的索引
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    # 标准化轴2的索引
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    # 调用torch库的swapaxes函数，实现轴的交换操作
    return torch.swapaxes(a, axis1, axis2)


# `rollaxis`函数用于滚动（循环移动）数组的轴
def rollaxis(a: ArrayLike, axis, start=0):
    # 数组的维度数
    n = a.ndim
    # 标准化轴的索引
    axis = _util.normalize_axis_index(axis, n)
    # 如果起始位置为负数，则转换成非负数
    if start < 0:
        start += n
    # 检查起始位置是否在有效范围内，否则抛出异常
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise _util.AxisError(msg % ("start", -n, "start", n + 1, start))
    # 如果轴索引小于起始位置，则调整起始位置
    if axis < start:
        start -= 1
    # 如果轴索引等于起始位置，则返回数组本身
    if axis == start:
        # numpy返回一个视图，这里尝试返回张量本身
        return a
    # 构造新的轴顺序列表，实现轴的滚动操作
    axes = list(range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.view(axes)


# `roll`函数用于在指定轴上对数组进行滚动（循环移动）
def roll(a: ArrayLike, shift, axis=None):
    # 如果指定了轴参数，则标准化轴索引
    if axis is not None:
        axis = _util.normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        # 如果shift不是元组，则转换成元组形式
        if not isinstance(shift, tuple):
            shift = (shift,) * len(axis)
    # 调用torch库的roll函数，在指定轴上对数组进行滚动操作
    return torch.roll(a, shift, axis)


# ### shape manipulations ###

# `squeeze`函数用于移除数组中的单维度条目
def squeeze(a: ArrayLike, axis=None):
    # 如果axis参数为()空元组，则结果直接为数组本身
    if axis == ():
        result = a
    # 如果 axis 参数为 None，则使用 squeeze() 函数压缩数组 a 的维度
    elif axis is None:
        result = a.squeeze()
    # 如果 axis 参数不为 None
    else:
        # 如果 axis 是一个元组
        if isinstance(axis, tuple):
            result = a
            # 遍历元组中的每个轴，并依次对数组 a 使用 squeeze() 函数压缩该轴的维度
            for ax in axis:
                result = a.squeeze(ax)
        # 如果 axis 不是一个元组，即单个轴
        else:
            # 对数组 a 使用 squeeze() 函数压缩指定的轴的维度
            result = a.squeeze(axis)
    # 返回压缩后的结果数组
    return result
# 定义函数 reshape，用于改变数组或张量的形状
def reshape(a: ArrayLike, newshape, order: NotImplementedType = "C"):
    # 如果 newshape 是长度为 1 的元组，将其转换为整数
    newshape = newshape[0] if len(newshape) == 1 else newshape
    # 调用数组或张量的 reshape 方法，改变其形状为 newshape
    return a.reshape(newshape)


# NB: 不能使用 torch.reshape(a, newshape)，因为
# (Pdb) torch.reshape(torch.as_tensor([1]), 1)
# *** TypeError: reshape(): argument 'shape' (position 2) must be tuple of SymInts, not int


# 定义函数 transpose，用于交换数组或张量的轴
def transpose(a: ArrayLike, axes=None):
    # 如果 axes 是空元组、None 或者 (None,)，则使用默认的轴顺序
    if axes in [(), None, (None,)]:
        axes = tuple(reversed(range(a.ndim)))
    # 如果 axes 是长度为 1 的列表，将其转换为整数
    elif len(axes) == 1:
        axes = axes[0]
    # 调用数组或张量的 permute 方法，交换指定的轴顺序
    return a.permute(axes)


# 定义函数 ravel，用于展平数组或张量
def ravel(a: ArrayLike, order: NotImplementedType = "C"):
    # 调用 torch 的 flatten 方法，将数组或张量展平为一维
    return torch.flatten(a)


# 定义函数 diff，用于计算数组或张量的差分
def diff(
    a: ArrayLike,
    n=1,
    axis=-1,
    prepend: Optional[ArrayLike] = None,
    append: Optional[ArrayLike] = None,
):
    # 标准化轴索引，确保 axis 在合法范围内
    axis = _util.normalize_axis_index(axis, a.ndim)

    if n < 0:
        # 如果 n 小于 0，抛出 ValueError 异常
        raise ValueError(f"order must be non-negative but got {n}")

    if n == 0:
        # 如果 n 等于 0，返回原始输入数组或张量
        return a

    if prepend is not None:
        # 如果有 prepend 参数，调整其形状并广播到合适的维度
        shape = list(a.shape)
        shape[axis] = prepend.shape[axis] if prepend.ndim > 0 else 1
        prepend = torch.broadcast_to(prepend, shape)

    if append is not None:
        # 如果有 append 参数，调整其形状并广播到合适的维度
        shape = list(a.shape)
        shape[axis] = append.shape[axis] if append.ndim > 0 else 1
        append = torch.broadcast_to(append, shape)

    # 调用 torch 的 diff 方法，计算数组或张量的差分
    return torch.diff(a, n, axis=axis, prepend=prepend, append=append)


# ### math functions ###


# 定义函数 angle，用于计算数组或张量中元素的角度
def angle(z: ArrayLike, deg=False):
    result = torch.angle(z)
    if deg:
        # 如果 deg 为 True，则将弧度转换为角度
        result = result * (180 / torch.pi)
    return result


# 定义函数 sinc，用于计算数组或张量中元素的 sinc 函数值
def sinc(x: ArrayLike):
    return torch.sinc(x)


# NB: 必须手动对 *varargs 进行规范化
# 定义函数 gradient，用于计算数组或张量的梯度
def gradient(f: ArrayLike, *varargs, axis=None, edge_order=1):
    # 获取数组或张量的维度数
    N = f.ndim

    # 将 varargs 中的所有 ndarray 转换为 tensor
    varargs = _util.ndarrays_to_tensors(varargs)

    if axis is None:
        # 如果 axis 为 None，则对所有轴进行梯度计算
        axes = tuple(range(N))
    else:
        # 否则，标准化轴元组，确保 axis 在合法范围内
        axes = _util.normalize_axis_tuple(axis, N)

    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        # 如果没有间距参数，所有轴的间距设为 1
        dx = [1.0] * len_axes
    elif n == 1 and (_dtypes_impl.is_scalar(varargs[0]) or varargs[0].ndim == 0):
        # 如果只有一个标量参数或者 0 维张量参数，将其复制到所有轴
        dx = varargs * len_axes
    elif n == len_axes:
        # 如果 n 等于 len_axes，说明每个轴上都有标量或者一维数组
        dx = list(varargs)
        # 遍历 dx 列表中的元素
        for i, distances in enumerate(dx):
            # 将 distances 转换为 PyTorch 的张量
            distances = torch.as_tensor(distances)
            # 如果 distances 的维度为 0，则跳过当前循环
            if distances.ndim == 0:
                continue
            # 如果 distances 的维度不为 1，则抛出 ValueError 异常
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            # 如果 distances 的长度与对应维度的长度不匹配，则抛出 ValueError 异常
            if len(distances) != f.shape[axes[i]]:
                raise ValueError(
                    "when 1d, distances must match "
                    "the length of the corresponding dimension"
                )
            # 如果 distances 的数据类型不是浮点数或复数类型，则将其转换为双精度浮点数类型
            if not (distances.dtype.is_floating_point or distances.dtype.is_complex):
                distances = distances.double()

            # 计算 distances 的差分
            diffx = torch.diff(distances)
            # 如果 distances 中所有元素的差分都相同，则将 diffx 简化为一个标量
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            # 更新 dx 中的第 i 个元素为 diffx
            dx[i] = diffx
    else:
        # 如果 n 不等于 len_axes，则抛出 TypeError 异常
        raise TypeError("invalid number of arguments")

    # 如果 edge_order 大于 2，则抛出 ValueError 异常
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # 使用中心差分计算内部点，使用单边差分计算端点。
    # 这样可以在整个定义域上保持二阶精度。
    outvals = []

    # 创建切片对象，初始都是 [:, :, ..., :]
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N

    # 获取 f 的数据类型
    otype = f.dtype
    # 如果 f 的数据类型是 torch 的整数类型或布尔类型
    if _dtypes_impl.python_type_for_torch(otype) in (int, bool):
        # 将 f 转换为双精度浮点数类型，避免在计算 f 的变化时出现模运算
        f = f.double()
        # 更新 otype 为 torch.float64

    # 如果 len_axes 等于 1，则返回 outvals 中的第一个元素
    if len_axes == 1:
        return outvals[0]
    else:
        # 否则返回 outvals
        return outvals
# ### Type/shape etc queries ###

# 定义一个函数 round，用于对输入的数组或类数组进行四舍五入操作
def round(a: ArrayLike, decimals=0, out: Optional[OutArray] = None):
    # 如果输入数组是浮点类型，则使用 torch.round 对其进行四舍五入
    if a.is_floating_point():
        result = torch.round(a, decimals=decimals)
    # 如果输入数组是复数类型，则分别对实部和虚部进行四舍五入，并组合成复数
    elif a.is_complex():
        # RuntimeError: "round_cpu" not implemented for 'ComplexFloat'
        result = torch.complex(
            torch.round(a.real, decimals=decimals),
            torch.round(a.imag, decimals=decimals),
        )
    else:
        # 如果输入数组是其他类型（如整数），直接返回原始值
        result = a
    return result


# 定义了两个别名函数 around 和 round_
around = round
round_ = round


# 定义一个函数 real_if_close，用于判断复数是否接近实数，若接近则返回实部
def real_if_close(a: ArrayLike, tol=100):
    # 如果输入数组不是复数类型，则直接返回原始值
    if not torch.is_complex(a):
        return a
    # 根据给定的容差 tol，判断是使用相对误差还是绝对误差
    if tol > 1:
        # Undocumented in numpy: if tol < 1, it's an absolute tolerance!
        # Otherwise, tol > 1 is relative tolerance, in units of the dtype epsilon
        # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/type_check.py#L577
        tol = tol * torch.finfo(a.dtype).eps

    # 使用绝对值比较判断虚部是否足够小，若是则返回实部；否则返回原始复数数组
    mask = torch.abs(a.imag) < tol
    return a.real if mask.all() else a


# 定义一个函数 real，返回复数数组的实部
def real(a: ArrayLike):
    return torch.real(a)


# 定义一个函数 imag，返回复数数组的虚部
def imag(a: ArrayLike):
    # 如果输入数组是复数类型，则返回其虚部；否则返回与输入数组形状相同的零数组
    if a.is_complex():
        return a.imag
    return torch.zeros_like(a)


# 定义一个函数 iscomplex，判断输入数组是否为复数类型，并且虚部不为零
def iscomplex(x: ArrayLike):
    if torch.is_complex(x):
        return x.imag != 0
    return torch.zeros_like(x, dtype=torch.bool)


# 定义一个函数 isreal，判断输入数组是否为实数类型（虚部为零）
def isreal(x: ArrayLike):
    if torch.is_complex(x):
        return x.imag == 0
    return torch.ones_like(x, dtype=torch.bool)


# 定义一个函数 iscomplexobj，判断输入是否为复数对象
def iscomplexobj(x: ArrayLike):
    return torch.is_complex(x)


# 定义一个函数 isrealobj，判断输入是否为实数对象
def isrealobj(x: ArrayLike):
    return not torch.is_complex(x)


# 定义一个函数 isneginf，判断输入数组中是否存在负无穷
def isneginf(x: ArrayLike, out: Optional[OutArray] = None):
    return torch.isneginf(x)


# 定义一个函数 isposinf，判断输入数组中是否存在正无穷
def isposinf(x: ArrayLike, out: Optional[OutArray] = None):
    return torch.isposinf(x)


# 定义一个函数 i0，计算输入数组的修正贝塞尔函数（第一类，零阶）
def i0(x: ArrayLike):
    return torch.special.i0(x)


# 定义一个函数 isscalar，用于判断输入是否是标量（单个元素的数组）
def isscalar(a):
    # 需要使用 normalize_array_like 函数，但不导出到 funcs.py 中
    from ._normalizations import normalize_array_like

    try:
        # 尝试将输入标准化成一个张量 t
        t = normalize_array_like(a)
        # 返回判断张量 t 是否是标量（元素个数为 1）
        return t.numel() == 1
    except Exception:
        # 发生异常则返回 False
        return False


# ### Filter windows ###

# 定义汉明窗口函数 hamming，返回指定长度 M 的汉明窗口
def hamming(M):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.hamming_window(M, periodic=False, dtype=dtype)


# 定义汉宁窗口函数 hanning，返回指定长度 M 的汉宁窗口
def hanning(M):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.hann_window(M, periodic=False, dtype=dtype)


# 定义凯泽窗口函数 kaiser，返回指定长度 M 和形状参数 beta 的凯泽窗口
def kaiser(M, beta):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.kaiser_window(M, beta=beta, periodic=False, dtype=dtype)


# 定义布莱克曼窗口函数 blackman，返回指定长度 M 的布莱克曼窗口
def blackman(M):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.blackman_window(M, periodic=False, dtype=dtype)


# 定义巴特利特窗口函数 bartlett，返回指定长度 M 的巴特利特窗口
def bartlett(M):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.bartlett_window(M, periodic=False, dtype=dtype)


# ### Dtype routines ###

# vendored from https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/type_check.py#L666

# 定义一个数组类型列表 array_type
array_type = [
    # 一个包含不同浮点数类型的列表，分别是半精度 (float16)、单精度 (float32) 和双精度 (float64)
    [torch.float16, torch.float32, torch.float64],
    
    # 一个包含不同复数类型的列表，分别是没有指定类型 (None)、复数类型为复数64 (complex64) 和复数类型为复数128 (complex128)
    [None, torch.complex64, torch.complex128],
# 定义一个字典，将不同的 Torch 张量数据类型映射到其精度级别
array_precision = {
    torch.float16: 0,          # torch.float16 对应精度级别 0
    torch.float32: 1,          # torch.float32 对应精度级别 1
    torch.float64: 2,          # torch.float64 对应精度级别 2
    torch.complex64: 1,        # torch.complex64 对应精度级别 1
    torch.complex128: 2,       # torch.complex128 对应精度级别 2
}


# 定义一个函数 common_type，用于确定多个张量的共同数据类型
def common_type(*tensors: ArrayLike):
    is_complex = False          # 是否存在复数类型的标志位
    precision = 0               # 数据类型的最高精度级别初始化为 0
    for a in tensors:
        t = a.dtype             # 获取当前张量的数据类型
        if iscomplexobj(a):     # 检查张量是否为复数类型
            is_complex = True   # 如果是复数类型，则设置 is_complex 标志为 True
        if not (t.is_floating_point or t.is_complex):
            p = 2               # 如果不是浮点数或复数，则设置 p 为 2（默认双精度）
        else:
            p = array_precision.get(t, None)  # 否则从 array_precision 字典中获取其精度级别
            if p is None:
                raise TypeError("can't get common type for non-numeric array")  # 如果找不到对应的精度级别，抛出类型错误异常
        precision = builtins.max(precision, p)  # 更新当前的最高精度级别
    if is_complex:
        return array_type[1][precision]  # 如果存在复数类型，则返回复数类型对应的精度级别
    else:
        return array_type[0][precision]  # 否则返回普通类型对应的精度级别


# ### histograms ###


# 定义直方图函数 histogram
def histogram(
    a: ArrayLike,
    bins: ArrayLike = 10,
    range=None,
    normed=None,
    weights: Optional[ArrayLike] = None,
    density=None,
):
    if normed is not None:
        raise ValueError("normed argument is deprecated, use density= instead")  # 如果使用了 normed 参数，则抛出值错误异常

    if weights is not None and weights.dtype.is_complex:
        raise NotImplementedError("complex weights histogram.")  # 如果权重是复数类型，则抛出未实现错误异常

    is_a_int = not (a.dtype.is_floating_point or a.dtype.is_complex)  # 检查输入数组是否为整数类型
    is_w_int = weights is None or not weights.dtype.is_floating_point  # 检查权重数组是否为整数类型或者未提供权重
    if is_a_int:
        a = a.double()  # 如果输入数组为整数类型，则转换为双精度浮点数

    if weights is not None:
        weights = _util.cast_if_needed(weights, a.dtype)  # 如果有权重数组，则根据输入数组的数据类型转换权重数组的数据类型

    if isinstance(bins, torch.Tensor):
        if bins.ndim == 0:
            bins = operator.index(bins)  # 如果 bins 是一个标量张量，则转换为索引
        else:
            bins = _util.cast_if_needed(bins, a.dtype)  # 否则根据输入数组的数据类型转换 bins 的数据类型

    if range is None:
        h, b = torch.histogram(a, bins, weight=weights, density=bool(density))  # 计算直方图
    else:
        h, b = torch.histogram(
            a, bins, range=range, weight=weights, density=bool(density)  # 根据指定范围计算直方图
        )

    if not density and is_w_int:
        h = h.long()  # 如果不是密度直方图且权重是整数类型，则将直方图类型转换为长整型
    if is_a_int:
        b = b.long()  # 如果输入数组是整数类型，则将 bin 边界转换为长整型

    return h, b  # 返回计算得到的直方图和 bin 边界


# 定义二维直方图函数 histogram2d
def histogram2d(
    x,
    y,
    bins=10,
    range: Optional[ArrayLike] = None,
    normed=None,
    weights: Optional[ArrayLike] = None,
    density=None,
):
    # vendored from https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/twodim_base.py#L655-L821
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")  # 如果 x 和 y 的长度不相等，则抛出值错误异常

    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1 and N != 2:
        bins = [bins, bins]  # 如果 bins 不是标量，则转换为包含两个元素的列表

    h, e = histogramdd((x, y), bins, range, normed, weights, density)  # 调用多维直方图函数计算二维直方图

    return h, e[0], e[1]  # 返回计算得到的二维直方图和两个维度的 bin 边界


# 定义多维直方图函数 histogramdd
def histogramdd(
    sample,
    bins=10,
    range: Optional[ArrayLike] = None,
    normed=None,
    weights: Optional[ArrayLike] = None,
    density=None,
):
    # have to normalize manually because `sample` interpretation differs
    # for a list of lists and a 2D array
    if normed is not None:
        raise ValueError("normed argument is deprecated, use density= instead")  # 如果使用了 normed 参数，则抛出值错误异常
    # 导入模块中的特定函数
    from ._normalizations import normalize_array_like, normalize_seq_array_like

    # 如果 sample 是列表或元组，则使用 normalize_array_like 函数对其进行归一化并转置
    if isinstance(sample, (list, tuple)):
        sample = normalize_array_like(sample).T
    else:
        # 否则，直接对 sample 进行归一化处理
        sample = normalize_array_like(sample)

    # 确保 sample 至少是二维的张量
    sample = torch.atleast_2d(sample)

    # 如果 sample 的数据类型不是浮点型或复数型，则将其转换为双精度浮点型
    if not (sample.dtype.is_floating_point or sample.dtype.is_complex):
        sample = sample.double()

    # 判断 bins 是否是数组类型（而不是单个整数或整数序列）
    bins_is_array = not (
        isinstance(bins, int) or builtins.all(isinstance(b, int) for b in bins)
    )
    if bins_is_array:
        # 对 bins 中的数组进行归一化处理
        bins = normalize_seq_array_like(bins)
        # 记录归一化后的每个数组的数据类型
        bins_dtypes = [b.dtype for b in bins]
        # 将 bins 中的数组元素转换为与 sample 相同的数据类型
        bins = [_util.cast_if_needed(b, sample.dtype) for b in bins]

    # 如果指定了 range 参数，则将其展平并转换为列表
    if range is not None:
        range = range.flatten().tolist()

    # 如果指定了 weights 参数，则计算 sample 中每个维度的最小值和最大值，并按需转换 weights 的数据类型
    if weights is not None:
        # 计算 sample 中每个维度的最小值和最大值
        mm = sample.aminmax(dim=0)
        # 将最小值和最大值交错排列，并展平为一维数组
        range = torch.cat(mm).reshape(2, -1).T.flatten()
        # 将 range 转换为元组形式
        range = tuple(range.tolist())
        # 将 weights 转换为与 sample 相同的数据类型
        weights = _util.cast_if_needed(weights, sample.dtype)
        # 构造用于传递给直方图函数的关键字参数字典，其中包含权重信息
        w_kwd = {"weight": weights}
    else:
        # 如果未指定 weights 参数，则初始化为空字典
        w_kwd = {}

    # 调用 torch 的直方图函数，计算多维直方图 h 和相应的 bin 边界 b
    h, b = torch.histogramdd(sample, bins, range, density=bool(density), **w_kwd)

    # 如果 bins 是数组类型，则将 b 中的每个 bin 边界转换为与对应 bins 中数组元素相同的数据类型
    if bins_is_array:
        b = [_util.cast_if_needed(bb, dtyp) for bb, dtyp in zip(b, bins_dtypes)]

    # 返回计算得到的直方图 h 和 bin 边界 b
    return h, b
# ### odds and ends

# 定义函数 min_scalar_type，接收一个参数 a，类型为 ArrayLike，只能通过位置参数传递
def min_scalar_type(a: ArrayLike, /):
    # 导入 DType 类
    from ._dtypes import DType

    # 如果 a 中的元素数量大于 1
    if a.numel() > 1:
        # 返回 a 的数据类型，不作修改
        return DType(a.dtype)

    # 如果 a 的数据类型为 torch.bool
    if a.dtype == torch.bool:
        # 数据类型为 torch.bool
        dtype = torch.bool

    # 如果 a 的数据类型是复数
    elif a.dtype.is_complex:
        # 获取 float32 的数值范围信息
        fi = torch.finfo(torch.float32)
        # 检查是否可以转换为 torch.complex64
        fits_in_single = a.dtype == torch.complex64 or (
            fi.min <= a.real <= fi.max and fi.min <= a.imag <= fi.max
        )
        # 如果可以，数据类型为 torch.complex64，否则为 torch.complex128
        dtype = torch.complex64 if fits_in_single else torch.complex128

    # 如果 a 的数据类型是浮点数
    elif a.dtype.is_floating_point:
        # 遍历可能的浮点数数据类型
        for dt in [torch.float16, torch.float32, torch.float64]:
            # 获取当前数据类型的数值范围信息
            fi = torch.finfo(dt)
            # 如果 a 的值在当前数据类型的范围内，则选择此数据类型
            if fi.min <= a <= fi.max:
                dtype = dt
                break

    # 如果 a 的数据类型不是浮点数，即为整数类型
    else:
        # 遍历可能的整数数据类型
        for dt in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            # 尽可能选择无符号整数数据类型，与 numpy 类似
            ii = torch.iinfo(dt)
            # 如果 a 的值在当前数据类型的范围内，则选择此数据类型
            if ii.min <= a <= ii.max:
                dtype = dt
                break

    # 返回选择的数据类型对象
    return DType(dtype)


# 定义函数 pad，接收三个参数：array（数组）、pad_width（填充宽度）、mode（填充模式，默认为 "constant"）、kwargs（其他关键字参数）
def pad(array: ArrayLike, pad_width: ArrayLike, mode="constant", **kwargs):
    # 如果填充模式不是 "constant"，则抛出 NotImplementedError
    if mode != "constant":
        raise NotImplementedError
    # 获取关键字参数中的 constant_values，如果没有则默认为 0
    value = kwargs.get("constant_values", 0)
    # 获取 array 的数据类型对应的 Python 标量类型
    typ = _dtypes_impl.python_type_for_torch(array.dtype)
    # 将 value 转换为 typ 类型的值
    value = typ(value)

    # 将 pad_width 广播为 (array.ndim, 2) 的形状
    pad_width = torch.broadcast_to(pad_width, (array.ndim, 2))
    # 翻转 pad_width 的顺序，并展平为一维数组
    pad_width = torch.flip(pad_width, (0,)).flatten()

    # 使用 torch.nn.functional.pad 函数对 array 进行填充，填充宽度为 tuple(pad_width)，填充值为 value
    return torch.nn.functional.pad(array, tuple(pad_width), value=value)
```