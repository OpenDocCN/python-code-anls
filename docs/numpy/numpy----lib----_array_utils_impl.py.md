# `.\numpy\numpy\lib\_array_utils_impl.py`

```
"""
Miscellaneous utils.
"""
# 从 numpy 核心模块中导入必要的函数和类
from numpy._core import asarray
from numpy._core.numeric import normalize_axis_tuple, normalize_axis_index
from numpy._utils import set_module

# 将该模块公开的函数和类列出，便于使用者了解可用功能
__all__ = ["byte_bounds", "normalize_axis_tuple", "normalize_axis_index"]


# 使用装饰器将下面的函数标记为属于 numpy.lib.array_utils 模块
@set_module("numpy.lib.array_utils")
def byte_bounds(a):
    """
    Returns pointers to the end-points of an array.

    Parameters
    ----------
    a : ndarray
        Input array. It must conform to the Python-side of the array
        interface.

    Returns
    -------
    (low, high) : tuple of 2 integers
        The first integer is the first byte of the array, the second
        integer is just past the last byte of the array.  If `a` is not
        contiguous it will not use every byte between the (`low`, `high`)
        values.

    Examples
    --------
    >>> I = np.eye(2, dtype='f'); I.dtype
    dtype('float32')
    >>> low, high = np.lib.array_utils.byte_bounds(I)
    >>> high - low == I.size*I.itemsize
    True
    >>> I = np.eye(2); I.dtype
    dtype('float64')
    >>> low, high = np.lib.array_utils.byte_bounds(I)
    >>> high - low == I.size*I.itemsize
    True

    """
    # 获取数组 a 的数组接口
    ai = a.__array_interface__
    # 获取数组数据的起始地址
    a_data = ai['data'][0]
    # 获取数组的步长信息
    astrides = ai['strides']
    # 获取数组的形状信息
    ashape = ai['shape']
    # 获取数组元素的字节大小
    bytes_a = asarray(a).dtype.itemsize

    # 初始化低地址和高地址为数组数据的起始地址
    a_low = a_high = a_data

    # 如果数组是连续存储的情况
    if astrides is None:
        # 计算数组的高地址
        a_high += a.size * bytes_a
    else:
        # 遍历数组的形状和步长信息，计算非连续存储情况下的高地址和低地址
        for shape, stride in zip(ashape, astrides):
            if stride < 0:
                a_low += (shape - 1) * stride
            else:
                a_high += (shape - 1) * stride
        # 加上数组元素的字节大小得到高地址
        a_high += bytes_a

    # 返回计算得到的低地址和高地址
    return a_low, a_high
```