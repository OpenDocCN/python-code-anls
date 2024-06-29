# `.\numpy\numpy\_core\_methods.py`

```py
"""
Array methods which are called by both the C-code for the method
and the Python code for the NumPy-namespace function

"""
# 导入所需模块
import os  # 导入操作系统模块
import pickle  # 导入pickle模块，用于对象序列化和反序列化
import warnings  # 导入警告模块，用于处理警告信息
from contextlib import nullcontext  # 导入上下文管理器nullcontext

# 导入NumPy的核心模块
from numpy._core import multiarray as mu  # 导入NumPy的multiarray模块并重命名为mu
from numpy._core import umath as um  # 导入NumPy的umath模块并重命名为um
from numpy._core.multiarray import asanyarray  # 导入asanyarray函数
from numpy._core import numerictypes as nt  # 导入numerictypes模块并重命名为nt
from numpy._core import _exceptions  # 导入异常处理模块_exceptions
from numpy._core._ufunc_config import _no_nep50_warning  # 导入_ufunc_config模块中的_no_nep50_warning
from numpy._globals import _NoValue  # 导入_NoValue对象

# save those O(100) nanoseconds!
bool_dt = mu.dtype("bool")  # 创建布尔类型的数据类型对象bool_dt
umr_maximum = um.maximum.reduce  # 将um模块的maximum.reduce方法赋值给umr_maximum
umr_minimum = um.minimum.reduce  # 将um模块的minimum.reduce方法赋值给umr_minimum
umr_sum = um.add.reduce  # 将um模块的add.reduce方法赋值给umr_sum
umr_prod = um.multiply.reduce  # 将um模块的multiply.reduce方法赋值给umr_prod
umr_bitwise_count = um.bitwise_count  # 将um模块的bitwise_count方法赋值给umr_bitwise_count
umr_any = um.logical_or.reduce  # 将um模块的logical_or.reduce方法赋值给umr_any
umr_all = um.logical_and.reduce  # 将um模块的logical_and.reduce方法赋值给umr_all

# Complex types to -> (2,)float view for fast-path computation in _var()
_complex_to_float = {
    nt.dtype(nt.csingle) : nt.dtype(nt.single),  # 将csingle对应的数据类型映射为single
    nt.dtype(nt.cdouble) : nt.dtype(nt.double),  # 将cdouble对应的数据类型映射为double
}
# Special case for windows: ensure double takes precedence
if nt.dtype(nt.longdouble) != nt.dtype(nt.double):
    _complex_to_float.update({
        nt.dtype(nt.clongdouble) : nt.dtype(nt.longdouble),  # 将clongdouble对应的数据类型映射为longdouble
    })

# avoid keyword arguments to speed up parsing, saves about 15%-20% for very
# small reductions
# 定义_amax函数，实现对数组的最大值计算
def _amax(a, axis=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_maximum(a, axis, None, out, keepdims, initial, where)

# 定义_amin函数，实现对数组的最小值计算
def _amin(a, axis=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_minimum(a, axis, None, out, keepdims, initial, where)

# 定义_sum函数，实现对数组元素求和
def _sum(a, axis=None, dtype=None, out=None, keepdims=False,
         initial=_NoValue, where=True):
    return umr_sum(a, axis, dtype, out, keepdims, initial, where)

# 定义_prod函数，实现对数组元素求积
def _prod(a, axis=None, dtype=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_prod(a, axis, dtype, out, keepdims, initial, where)

# 定义_any函数，实现对数组中是否有True值的判断
def _any(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    # 默认情况下，对于_any和_all函数，返回布尔值
    if dtype is None:
        dtype = bool_dt
    # 目前解析关键字参数的速度相对较慢，因此暂时避免使用它
    if where is True:
        return umr_any(a, axis, dtype, out, keepdims)
    return umr_any(a, axis, dtype, out, keepdims, where=where)

# 定义_all函数，实现对数组中所有元素是否为True的判断
def _all(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    # 默认情况下，对于_any和_all函数，返回布尔值
    if dtype is None:
        dtype = bool_dt
    # 目前解析关键字参数的速度相对较慢，因此暂时避免使用它
    if where is True:
        return umr_all(a, axis, dtype, out, keepdims)
    return umr_all(a, axis, dtype, out, keepdims, where=where)

# 定义_count_reduce_items函数，用于快速处理默认情况的数组减少操作
def _count_reduce_items(arr, axis, keepdims=False, where=True):
    # fast-path for the default case
    # 默认情况的快速处理路径
    # 如果 where 参数为 True，则计算项目数量而不考虑布尔掩码
    if where is True:
        # 如果没有指定 axis 参数，则默认为数组的所有维度
        if axis is None:
            axis = tuple(range(arr.ndim))
        # 如果 axis 不是元组，则转换为元组
        elif not isinstance(axis, tuple):
            axis = (axis,)
        # 计算项目数量的初始值为 1
        items = 1
        # 遍历所有指定的 axis 维度，计算项目数量
        for ax in axis:
            items *= arr.shape[mu.normalize_axis_index(ax, arr.ndim)]
        # 将 items 转换为 numpy 的整数类型
        items = nt.intp(items)
    else:
        # TODO: 优化当 `where` 沿着非约简轴进行广播时的情况，
        # 并且完整求和超出所需的范围。

        # 导入 broadcast_to 函数来保护循环导入问题
        from numpy.lib._stride_tricks_impl import broadcast_to
        # 计算布尔掩码中 True 值的数量（可能进行了广播）
        items = umr_sum(broadcast_to(where, arr.shape), axis, nt.intp, None,
                        keepdims)
    # 返回计算得到的项目数量
    return items
# 定义一个函数，用于对数组进行剪裁操作，限制其数值在给定范围内
def _clip(a, min=None, max=None, out=None, **kwargs):
    # 如果未同时给定最小值和最大值，则抛出数值错误
    if min is None and max is None:
        raise ValueError("One of max or min must be given")

    # 如果只给定最大值，则调用 um.minimum 函数进行剪裁
    if min is None:
        return um.minimum(a, max, out=out, **kwargs)
    # 如果只给定最小值，则调用 um.maximum 函数进行剪裁
    elif max is None:
        return um.maximum(a, min, out=out, **kwargs)
    # 如果同时给定最小值和最大值，则调用 um.clip 函数进行剪裁
    else:
        return um.clip(a, min, max, out=out, **kwargs)

# 定义一个函数，计算数组的均值
def _mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    # 将输入数组转换为通用数组类型
    arr = asanyarray(a)

    # 初始化是否得到 float16 结果的标志
    is_float16_result = False

    # 计算在指定轴向上进行均值计算的有效元素数
    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
    # 如果计数为零（根据 where 参数），则发出警告
    if rcount == 0 if where is True else umr_any(rcount == 0, axis=None):
        warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)

    # 如果未指定输出数据类型，则根据输入数组的类型确定默认类型
    if dtype is None:
        if issubclass(arr.dtype.type, (nt.integer, nt.bool)):
            dtype = mu.dtype('f8')  # 整数和布尔型转换为 float64
        elif issubclass(arr.dtype.type, nt.float16):
            dtype = mu.dtype('f4')  # float16 转换为 float32
            is_float16_result = True

    # 计算数组在指定轴向上的总和
    ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
    # 如果返回结果为 ndarray 类型，则进行除法操作
    if isinstance(ret, mu.ndarray):
        with _no_nep50_warning():
            ret = um.true_divide(
                    ret, rcount, out=ret, casting='unsafe', subok=False)
        # 如果结果为 float16 类型且未指定输出，则转换为原数组的类型
        if is_float16_result and out is None:
            ret = arr.dtype.type(ret)
    # 如果返回结果具有 dtype 属性，则进行除法操作
    elif hasattr(ret, 'dtype'):
        if is_float16_result:
            ret = arr.dtype.type(ret / rcount)
        else:
            ret = ret.dtype.type(ret / rcount)
    # 否则直接进行除法操作
    else:
        ret = ret / rcount

    # 返回计算得到的均值结果
    return ret

# 定义一个函数，计算数组的方差
def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *,
         where=True, mean=None):
    # 将输入数组转换为通用数组类型
    arr = asanyarray(a)

    # 计算在指定轴向上进行计算的有效元素数
    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
    # 如果自由度小于等于零（根据 where 参数），则发出警告
    if ddof >= rcount if where is True else umr_any(ddof >= rcount, axis=None):
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning,
                      stacklevel=2)

    # 如果未指定输出数据类型且输入数组为整数或布尔类型，则将其转换为 float64 类型
    if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool)):
        dtype = mu.dtype('f8')

    # 如果给定均值参数，则使用该均值参数作为均值
    if mean is not None:
        arrmean = mean
    else:
        # 计算均值。
        # 需要注意的是，如果 dtype 不是浮点类型，则 arraymean 也不会是浮点类型。
        arrmean = umr_sum(arr, axis, dtype, keepdims=True, where=where)
        # rcount 的形状必须与 arrmean 匹配，以便在广播中不改变输出的形状。否则，不能将其存回到 arrmean。
        if rcount.ndim == 0:
            # 默认情况的快速路径，当 where 参数为 True 时
            div = rcount
        else:
            # 当 where 参数作为数组指定时，将 rcount 的形状匹配到 arrmean
            div = rcount.reshape(arrmean.shape)
        if isinstance(arrmean, mu.ndarray):
            with _no_nep50_warning():
                arrmean = um.true_divide(arrmean, div, out=arrmean,
                                         casting='unsafe', subok=False)
        elif hasattr(arrmean, "dtype"):
            arrmean = arrmean.dtype.type(arrmean / rcount)
        else:
            arrmean = arrmean / rcount

    # 计算相对于均值的平方偏差的总和
    # 需要注意 x 可能不是浮点数，并且我们需要它是一个数组，而不是一个标量。
    x = asanyarray(arr - arrmean)

    if issubclass(arr.dtype.type, (nt.floating, nt.integer)):
        x = um.multiply(x, x, out=x)
    # 内置复数类型的快速路径
    elif x.dtype in _complex_to_float:
        xv = x.view(dtype=(_complex_to_float[x.dtype], (2,)))
        um.multiply(xv, xv, out=xv)
        x = um.add(xv[..., 0], xv[..., 1], out=x.real).real
    # 最一般的情况；包括处理包含虚数和具有非本机字节顺序的复杂类型的对象数组
    else:
        x = um.multiply(x, um.conjugate(x), out=x).real

    ret = umr_sum(x, axis, dtype, out, keepdims=keepdims, where=where)

    # 计算自由度并确保其非负。
    rcount = um.maximum(rcount - ddof, 0)

    # 除以自由度
    if isinstance(ret, mu.ndarray):
        with _no_nep50_warning():
            ret = um.true_divide(
                    ret, rcount, out=ret, casting='unsafe', subok=False)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount

    return ret
# 计算标准差的函数
def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *,
         where=True, mean=None):
    # 调用 _var 函数计算方差，并返回结果
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
               keepdims=keepdims, where=where, mean=mean)

    # 如果 ret 是 mu.ndarray 类型的实例
    if isinstance(ret, mu.ndarray):
        # 对 ret 进行平方根运算，并将结果存储在 ret 中
        ret = um.sqrt(ret, out=ret)
    # 如果 ret 具有 'dtype' 属性
    elif hasattr(ret, 'dtype'):
        # 以 ret.dtype.type 类型对 ret 进行平方根运算，并将结果存储在 ret 中
        ret = ret.dtype.type(um.sqrt(ret))
    else:
        # 否则直接对 ret 进行平方根运算，并将结果存储在 ret 中
        ret = um.sqrt(ret)

    # 返回计算得到的标准差
    return ret

# 计算数组沿指定轴的峰值到峰值（peak-to-peak）值的函数
def _ptp(a, axis=None, out=None, keepdims=False):
    # 计算数组沿指定轴的最大值与最小值之差，并返回结果
    return um.subtract(
        umr_maximum(a, axis, None, out, keepdims),
        umr_minimum(a, axis, None, None, keepdims),
        out
    )

# 将对象序列化到文件中的函数
def _dump(self, file, protocol=2):
    # 如果 file 具有 'write' 方法，则使用 nullcontext 封装 file
    if hasattr(file, 'write'):
        ctx = nullcontext(file)
    else:
        # 否则打开 file 并以二进制写模式进行操作，并使用 nullcontext 封装打开的文件对象
        ctx = open(os.fspath(file), "wb")
    # 使用 nullcontext 上下文管理器打开文件，并使用 pickle.dump 将 self 对象序列化到文件中
    with ctx as f:
        pickle.dump(self, f, protocol=protocol)

# 将对象序列化到字节流中的函数
def _dumps(self, protocol=2):
    # 使用 pickle.dumps 将 self 对象序列化为字节流，并返回结果
    return pickle.dumps(self, protocol=protocol)

# 对数组进行位运算计数的函数
def _bitwise_count(a, out=None, *, where=True, casting='same_kind',
          order='K', dtype=None, subok=True):
    # 调用 umr_bitwise_count 函数对数组进行位运算计数，并返回结果
    return umr_bitwise_count(a, out, where=where, casting=casting,
            order=order, dtype=dtype, subok=subok)
```