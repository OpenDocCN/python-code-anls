# `D:\src\scipysrc\scipy\scipy\signal\_arraytools.py`

```
"""
Functions for acting on a axis of an array.
"""
import numpy as np

def axis_slice(a, start=None, stop=None, step=None, axis=-1):
    """Take a slice along axis 'axis' from 'a'.

    Parameters
    ----------
    a : numpy.ndarray
        The array to be sliced.
    start, stop, step : int or None
        The slice parameters.
    axis : int, optional
        The axis of `a` to be sliced.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal._arraytools import axis_slice
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> axis_slice(a, start=0, stop=1, axis=1)
    array([[1],
           [4],
           [7]])
    >>> axis_slice(a, start=1, axis=0)
    array([[4, 5, 6],
           [7, 8, 9]])

    Notes
    -----
    The keyword arguments start, stop and step are used by calling
    slice(start, stop, step). This implies axis_slice() does not
    handle its arguments the exactly the same as indexing. To select
    a single index k, for example, use
        axis_slice(a, start=k, stop=k+1)
    In this case, the length of the axis 'axis' in the result will
    be 1; the trivial dimension is not removed. (Use numpy.squeeze()
    to remove trivial axes.)
    """
    # Create a list of slice objects, one for each dimension of 'a'
    a_slice = [slice(None)] * a.ndim
    # Replace the slice object for 'axis' with the specified slice(start, stop, step)
    a_slice[axis] = slice(start, stop, step)
    # Apply the slice to array 'a' and return the resulting array
    b = a[tuple(a_slice)]
    return b


def axis_reverse(a, axis=-1):
    """Reverse the 1-D slices of `a` along axis `axis`.

    Returns axis_slice(a, step=-1, axis=axis).
    """
    # Call axis_slice() with step=-1 to reverse the slices along the specified axis
    return axis_slice(a, step=-1, axis=axis)


def odd_ext(x, n, axis=-1):
    """
    Odd extension at the boundaries of an array

    Generate a new ndarray by making an odd extension of `x` along an axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the axis.
    axis : int, optional
        The axis along which to extend `x`. Default is -1.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal._arraytools import odd_ext
    >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> odd_ext(a, 2)
    array([[-1,  0,  1,  2,  3,  4,  5,  6,  7],
           [-4, -1,  0,  1,  4,  9, 16, 23, 28]])

    Odd extension is a "180 degree rotation" at the endpoints of the original
    array:

    >>> t = np.linspace(0, 1.5, 100)
    >>> a = 0.9 * np.sin(2 * np.pi * t**2)
    >>> b = odd_ext(a, 40)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(np.arange(-40, 140), b, 'b', lw=1, label='odd extension')
    >>> plt.plot(np.arange(100), a, 'r', lw=2, label='original')
    >>> plt.legend(loc='best')
    >>> plt.show()
    """
    # Check if n is less than 1, return x as is
    if n < 1:
        return x
    # Check if n is greater than the length of axis 'axis' in array 'x' minus 1
    if n > x.shape[axis] - 1:
        raise ValueError(("The extension length n (%d) is too big. " +
                         "It must not exceed x.shape[axis]-1, which is %d.")
                         % (n, x.shape[axis] - 1))
    # Perform odd extension on the left end of axis 'axis' of array 'x'
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    # 使用自定义函数 axis_slice 对数组 x 进行切片操作，从索引 n 开始向索引 0 方向倒序切片，结果赋给 left_ext
    left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    # 使用自定义函数 axis_slice 对数组 x 进行切片操作，从最后一个元素向前取一个元素，结果赋给 right_end
    right_end = axis_slice(x, start=-1, axis=axis)
    # 使用自定义函数 axis_slice 对数组 x 进行切片操作，从倒数第二个元素向倒数第 n+2 个元素的位置倒序切片，结果赋给 right_ext
    right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    # 使用 NumPy 的 concatenate 函数沿指定轴(axis)连接左端外推、原始数组 x 和右端外推的结果，结果赋给 ext
    ext = np.concatenate((2 * left_end - left_ext,
                          x,
                          2 * right_end - right_ext),
                         axis=axis)
    # 返回连接后的数组 ext
    return ext
# 定义一个函数，用于沿指定轴对数组进行偶数扩展
def even_ext(x, n, axis=-1):
    # 如果扩展长度小于1，则返回原始数组
    if n < 1:
        return x
    # 如果扩展长度大于数组在指定轴上的长度减1，则抛出数值错误
    if n > x.shape[axis] - 1:
        raise ValueError(("The extension length n (%d) is too big. " +
                         "It must not exceed x.shape[axis]-1, which is %d.")
                         % (n, x.shape[axis] - 1))
    # 从数组的指定轴的起始位置向左（逆序）切片获取左侧扩展部分
    left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    # 从数组的指定轴的倒数第二个元素向左（逆序）切片获取右侧扩展部分
    right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    # 沿指定轴连接左侧扩展、原始数组和右侧扩展，形成偶数扩展后的数组
    ext = np.concatenate((left_ext,
                          x,
                          right_ext),
                         axis=axis)
    # 返回偶数扩展后的数组
    return ext


# 定义一个函数，用于沿指定轴对数组进行常数扩展
def const_ext(x, n, axis=-1):
    # 如果扩展长度小于1，则返回原始数组
    if n < 1:
        return x
    # 调用 axis_slice 函数，从 x 中取出指定轴上的起始元素，作为左端点
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    # 创建一个由 1 组成的列表，长度与 x 的维度相同
    ones_shape = [1] * x.ndim
    # 将 ones_shape 中索引为 axis 的元素设为 n，用于扩展数组
    ones_shape[axis] = n
    # 创建一个元素类型与 x 相同的数组，所有元素为 1，形状由 ones_shape 决定
    ones = np.ones(ones_shape, dtype=x.dtype)
    # 将 left_end 扩展成与 x 相同维度的数组，左端扩展
    left_ext = ones * left_end
    # 调用 axis_slice 函数，从 x 中取出指定轴上的末尾元素，作为右端点
    right_end = axis_slice(x, start=-1, axis=axis)
    # 将 right_end 扩展成与 x 相同维度的数组，右端扩展
    right_ext = ones * right_end
    # 在指定轴上连接 left_ext、x、right_ext，形成扩展后的数组
    ext = np.concatenate((left_ext,
                          x,
                          right_ext),
                         axis=axis)
    # 返回扩展后的数组
    return ext
# 按指定轴向在数组的边界进行零填充

# 生成一个新的 ndarray，其沿着指定轴向零填充扩展 `x`

# Parameters
# ----------
# x : ndarray
#     要扩展的数组。
# n : int
#     要在轴的每一端扩展 `x` 的元素数。
# axis : int, optional
#     要扩展 `x` 的轴。默认为 -1。

# Examples
# --------
# >>> import numpy as np
# >>> from scipy.signal._arraytools import zero_ext
# >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
# >>> zero_ext(a, 2)
# array([[ 0,  0,  1,  2,  3,  4,  5,  0,  0],
#        [ 0,  0,  0,  1,  4,  9, 16,  0,  0]])
def zero_ext(x, n, axis=-1):
    if n < 1:
        return x
    # 创建一个与 `x` 相同数据类型的零数组，形状沿着指定轴扩展 `n` 个元素
    zeros_shape = list(x.shape)
    zeros_shape[axis] = n
    zeros = np.zeros(zeros_shape, dtype=x.dtype)
    # 沿着指定轴连接零数组、`x` 和零数组，形成扩展后的数组
    ext = np.concatenate((zeros, x, zeros), axis=axis)
    return ext


def _validate_fs(fs, allow_none=True):
    """
    检查给定的采样频率是否为标量，否则引发异常。如果 allow_none 为 False，
    还会针对空采样率引发异常。返回作为浮点数的采样频率或者如果输入为 none，则返回 none。
    """
    if fs is None:
        if not allow_none:
            raise ValueError("采样频率不能为 None。")
    else:  # 应为浮点数
        if not np.isscalar(fs):
            raise ValueError("采样频率 fs 必须是单个标量值。")
        fs = float(fs)
    return fs
```