# `D:\src\scipysrc\scipy\scipy\fft\_pocketfft\helper.py`

```
from numbers import Number  # 导入Number类，用于检查变量类型是否为数字
import operator  # 导入operator模块，用于操作符的函数形式
import os  # 导入os模块，提供了与操作系统交互的功能
import threading  # 导入threading模块，提供了多线程处理支持
import contextlib  # 导入contextlib模块，用于创建和管理上下文管理器的工具

import numpy as np  # 导入NumPy库，并用np作为别名

from scipy._lib._util import copy_if_needed  # 从SciPy库的私有模块中导入copy_if_needed函数

# 从当前目录的pypocketfft模块中导入good_size和prev_good_size，这些函数被公开使用
from .pypocketfft import good_size, prev_good_size

__all__ = ['good_size', 'prev_good_size', 'set_workers', 'get_workers']

_config = threading.local()  # 创建线程局部存储对象_config
_cpu_count = os.cpu_count()  # 获取系统CPU核心数量


def _iterable_of_int(x, name=None):
    """Convert ``x`` to an iterable sequence of int

    Parameters
    ----------
    x : value, or sequence of values, convertible to int
    name : str, optional
        Name of the argument being converted, only used in the error message

    Returns
    -------
    y : ``List[int]``
    """
    if isinstance(x, Number):  # 如果x是Number类型（数字）
        x = (x,)  # 将x转换为元组形式

    try:
        x = [operator.index(a) for a in x]  # 尝试将x中的每个元素转换为整数索引形式
    except TypeError as e:
        name = name or "value"  # 如果未提供name，则使用默认值"value"
        raise ValueError(f"{name} must be a scalar or iterable of integers") from e  # 抛出值错误异常

    return x  # 返回转换后的列表形式


def _init_nd_shape_and_axes(x, shape, axes):
    """Handles shape and axes arguments for nd transforms"""
    noshape = shape is None  # 判断是否未提供shape参数
    noaxes = axes is None  # 判断是否未提供axes参数

    if not noaxes:
        axes = _iterable_of_int(axes, 'axes')  # 将axes参数转换为整数列表
        axes = [a + x.ndim if a < 0 else a for a in axes]  # 处理负数索引，并转换为有效索引

        if any(a >= x.ndim or a < 0 for a in axes):  # 检查所有轴是否在有效范围内
            raise ValueError("axes exceeds dimensionality of input")  # 如果超出输入的维度数，则抛出值错误异常
        if len(set(axes)) != len(axes):  # 检查轴是否唯一
            raise ValueError("all axes must be unique")  # 如果轴不唯一，则抛出值错误异常

    if not noshape:
        shape = _iterable_of_int(shape, 'shape')  # 将shape参数转换为整数列表

        if axes and len(axes) != len(shape):  # 检查shape和axes参数的长度是否一致
            raise ValueError("when given, axes and shape arguments"
                             " have to be of the same length")  # 如果长度不一致，则抛出值错误异常
        if noaxes:
            if len(shape) > x.ndim:  # 如果shape参数长度大于输入数据的维度数
                raise ValueError("shape requires more axes than are present")  # 抛出值错误异常
            axes = range(x.ndim - len(shape), x.ndim)  # 根据shape参数补齐缺失的轴

        shape = [x.shape[a] if s == -1 else s for s, a in zip(shape, axes)]  # 根据轴和形状参数创建形状列表
    elif noaxes:
        shape = list(x.shape)  # 如果未提供shape和axes参数，则使用输入数据的形状和所有轴
        axes = range(x.ndim)
    else:
        shape = [x.shape[a] for a in axes]  # 如果未提供shape参数，则根据提供的轴创建形状列表

    if any(s < 1 for s in shape):  # 检查所有形状参数是否大于等于1
        raise ValueError(
            f"invalid number of data points ({shape}) specified")  # 如果形状参数无效，则抛出值错误异常

    return tuple(shape), list(axes)  # 返回形状元组和轴列表


def _asfarray(x):
    """
    Convert to array with floating or complex dtype.

    float16 values are also promoted to float32.
    """
    if not hasattr(x, "dtype"):  # 如果x没有dtype属性
        x = np.asarray(x)  # 将x转换为NumPy数组

    if x.dtype == np.float16:  # 如果x的数据类型是float16
        return np.asarray(x, np.float32)  # 将x转换为float32类型的NumPy数组
    elif x.dtype.kind not in 'fc':  # 如果x的数据类型不是浮点数或复数
        return np.asarray(x, np.float64)  # 将x转换为float64类型的NumPy数组

    # Require native byte order
    dtype = x.dtype.newbyteorder('=')  # 获取与本地字节顺序匹配的数据类型
    # Always align input
    copy = True if not x.flags['ALIGNED'] else copy_if_needed  # 根据输入是否对齐，确定是否需要复制数据
    return np.array(x, dtype=dtype, copy=copy)  # 返回转换后的NumPy数组


def _datacopied(arr, original):
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = asarray(original)
    """
    """
    检查给定的数组 `arr` 是否是原始数组 `original` 的视图（view）
    如果是，则返回 False，表示 `arr` 是原始数组的视图
    如果 `original` 不是 numpy 数组（np.ndarray），但是有 `__array__` 属性，则返回 False
    否则，检查 `arr` 的基础（base）是否为 None，如果是，则返回 True，表示 `arr` 是原始数组的拷贝（copy）
    """
    if arr is original:
        # 检查 `arr` 是否与 `original` 是同一个对象，如果是，说明 `arr` 是原始数组本身，不是视图
        return False
    if not isinstance(original, np.ndarray) and hasattr(original, '__array__'):
        # 如果 `original` 不是 numpy 数组，但是有 `__array__` 属性，则说明 `original` 可能是其他对象，不是 numpy 数组
        return False
    # 检查 `arr` 的基础（base）是否为 None，如果是，则表示 `arr` 是原始数组的拷贝
    return arr.base is None
def _fix_shape(x, shape, axes):
    """Internal auxiliary function for _raw_fft, _raw_fftnd."""
    # 标记是否需要复制数据
    must_copy = False

    # 构建一个 nd 切片，指定从 x 中读取的维度
    index = [slice(None)] * x.ndim
    for n, ax in zip(shape, axes):
        if x.shape[ax] >= n:
            index[ax] = slice(0, n)
        else:
            index[ax] = slice(0, x.shape[ax])
            must_copy = True

    index = tuple(index)

    # 如果不需要复制数据，则直接返回切片后的 x 和 False
    if not must_copy:
        return x[index], False

    # 如果需要复制数据，创建一个和 x 形状相同的全零数组 z
    s = list(x.shape)
    for n, axis in zip(shape, axes):
        s[axis] = n
    z = np.zeros(s, x.dtype)

    # 将 x 中切片后的数据复制到 z 中，并返回 z 和 True
    z[index] = x[index]
    return z, True


def _fix_shape_1d(x, n, axis):
    if n < 1:
        raise ValueError(
            f"invalid number of data points ({n}) specified")

    # 调用 _fix_shape 函数，指定维度为 (n,)，在轴 axis 上
    return _fix_shape(x, (n,), (axis,))


_NORM_MAP = {None: 0, 'backward': 0, 'ortho': 1, 'forward': 2}


def _normalization(norm, forward):
    """Returns the pypocketfft normalization mode from the norm argument"""
    try:
        # 根据 norm 参数返回相应的 pypocketfft 的标准化模式
        inorm = _NORM_MAP[norm]
        return inorm if forward else (2 - inorm)
    except KeyError:
        # 抛出异常，如果 norm 不在 _NORM_MAP 中定义
        raise ValueError(
            f'Invalid norm value {norm!r}, should '
            'be "backward", "ortho" or "forward"') from None


def _workers(workers):
    if workers is None:
        # 如果 workers 参数为 None，返回默认的工作线程数
        return getattr(_config, 'default_workers', 1)

    if workers < 0:
        # 如果 workers 参数小于 0，根据范围调整 workers 的值
        if workers >= -_cpu_count:
            workers += 1 + _cpu_count
        else:
            raise ValueError(f"workers value out of range; got {workers}, must not be"
                             f" less than {-_cpu_count}")
    elif workers == 0:
        # 如果 workers 参数为 0，抛出异常
        raise ValueError("workers must not be zero")

    # 返回 workers 参数本身
    return workers


@contextlib.contextmanager
def set_workers(workers):
    """Context manager for the default number of workers used in `scipy.fft`

    Parameters
    ----------
    workers : int
        The default number of workers to use

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fft, signal
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal((128, 64))
    >>> with fft.set_workers(4):
    ...     y = signal.fftconvolve(x, x)

    """
    # 保存旧的工作线程数，并设置新的工作线程数
    old_workers = get_workers()
    _config.default_workers = _workers(operator.index(workers))
    try:
        yield
    finally:
        # 恢复旧的工作线程数
        _config.default_workers = old_workers


def get_workers():
    """Returns the default number of workers within the current context

    Examples
    --------
    >>> from scipy import fft
    >>> fft.get_workers()
    1
    >>> with fft.set_workers(4):
    ...     fft.get_workers()
    4
    """
    # 返回当前上下文中的默认工作线程数
    return getattr(_config, 'default_workers', 1)
```