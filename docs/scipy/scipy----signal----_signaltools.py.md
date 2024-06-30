# `D:\src\scipysrc\scipy\scipy\signal\_signaltools.py`

```
# Author: Travis Oliphant
# 1999 -- 2002

# 导入未来版本的注解，为 Python 3.9 提供了类型联合运算符 `|`
from __future__ import annotations
# 导入运算符模块
import operator
# 导入数学模块
import math
# 从数学模块导入 `prod` 函数并重命名为 `_prod`
from math import prod as _prod
# 导入计时模块
import timeit
# 导入警告模块
import warnings
# 导入类型提示中的字面量类型
from typing import Literal

# 导入 cKDTree 类
from scipy.spatial import cKDTree
# 导入本地模块 _sigtools
from . import _sigtools
# 导入本地模块 _ltisys 中的 dlti 类
from ._ltisys import dlti
# 导入本地模块 _upfirdn 中的 upfirdn 函数和相关常量
from ._upfirdn import upfirdn, _output_len, _upfirdn_modes
# 导入线性代数模块和快速傅里叶变换模块
from scipy import linalg, fft as sp_fft
# 导入图像处理模块
from scipy import ndimage
# 导入 FFT 辅助函数
from scipy.fft._helper import _init_nd_shape_and_axes
# 导入 NumPy 库并重命名为 np
import numpy as np
# 导入 lambertw 函数
from scipy.special import lambertw
# 导入本地模块 windows 中的 get_window 函数
from .windows import get_window
# 导入本地模块 _arraytools 中的函数和类
from ._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext
# 导入本地模块 _filter_design 中的 cheby1 函数和相关辅助函数
from ._filter_design import cheby1, _validate_sos, zpk2sos
# 导入本地模块 _fir_filter_design 中的 firwin 函数
from ._fir_filter_design import firwin
# 导入本地模块 _sosfilt 中的 _sosfilt 函数
from ._sosfilt import _sosfilt

# 定义公开的接口
__all__ = ['correlate', 'correlation_lags', 'correlate2d',
           'convolve', 'convolve2d', 'fftconvolve', 'oaconvolve',
           'order_filter', 'medfilt', 'medfilt2d', 'wiener', 'lfilter',
           'lfiltic', 'sosfilt', 'deconvolve', 'hilbert', 'hilbert2',
           'unique_roots', 'invres', 'invresz', 'residue',
           'residuez', 'resample', 'resample_poly', 'detrend',
           'lfilter_zi', 'sosfilt_zi', 'sosfiltfilt', 'choose_conv_method',
           'filtfilt', 'decimate', 'vectorstrength']

# 定义模式字典，映射字符串到整数值
_modedict = {'valid': 0, 'same': 1, 'full': 2}

# 定义边界字典，映射字符串到整数值（位移后）
_boundarydict = {'fill': 0, 'pad': 0, 'wrap': 2, 'circular': 2, 'symm': 1,
                 'symmetric': 1, 'reflect': 4}

# 定义函数，根据模式返回相应整数值
def _valfrommode(mode):
    try:
        return _modedict[mode]
    except KeyError as e:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.") from e

# 定义函数，根据边界返回相应整数值
def _bvalfromboundary(boundary):
    try:
        return _boundarydict[boundary] << 2
    except KeyError as e:
        raise ValueError("Acceptable boundary flags are 'fill', 'circular' "
                         "(or 'wrap'), and 'symmetric' (or 'symm').") from e

# 定义函数，判断是否需要交换输入数组的顺序（用于 "valid" 模式）
def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    """Determine if inputs arrays need to be swapped in `"valid"` mode.

    If in `"valid"` mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every calculated dimension.

    This is important for some of the correlation and convolution
    implementations in this module, where the larger array input needs to come
    before the smaller array input when operating in this mode.

    Note that if the mode provided is not 'valid', False is immediately
    returned.

    """
    if mode != 'valid':
        return False

    if not shape1:
        return False

    if axes is None:
        axes = range(len(shape1))

    # 判断是否在所有指定轴上 shape1 都大于等于 shape2
    ok1 = all(shape1[i] >= shape2[i] for i in axes)
    # 判断是否在所有指定轴上 shape2 都大于等于 shape1
    ok2 = all(shape2[i] >= shape1[i] for i in axes)

    # 返回判断结果
    return not ok1 and ok2
    # 如果 ok1 和 ok2 都不为真（即都为假），则抛出值错误异常
    if not (ok1 or ok2):
        raise ValueError("For 'valid' mode, one must be at least "
                         "as large as the other in every dimension")

    # 返回 ok1 的逻辑非结果
    return not ok1
# 定义函数 correlate，用于计算两个 N 维数组的交叉相关
def correlate(in1, in2, mode='full', method='auto'):
    r"""
    Cross-correlate two N-dimensional arrays.

    Cross-correlate `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the correlation.

        ``direct``
           The correlation is determined directly from sums, the definition of
           correlation.
        ``fft``
           The Fast Fourier Transform is used to perform the correlation more
           quickly (only available for numerical arrays.)
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).  See `convolve` Notes for more detail.

           .. versionadded:: 0.19.0

    Returns
    -------
    correlate : array
        An N-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.

    See Also
    --------
    choose_conv_method : contains more documentation on `method`.
    correlation_lags : calculates the lag / displacement indices array for 1D
        cross-correlation.

    Notes
    -----
    The correlation z of two d-dimensional arrays x and y is defined as::

        z[...,k,...] = sum[..., i_l, ...] x[..., i_l,...] * conj(y[..., i_l - k,...])

    This way, if x and y are 1-D arrays and ``z = correlate(x, y, 'full')``
    then

    .. math::

          z[k] = (x * y)(k - N + 1)
               = \sum_{l=0}^{||x||-1}x_l y_{l-k+N-1}^{*}

    for :math:`k = 0, 1, ..., ||x|| + ||y|| - 2`

    where :math:`||x||` is the length of ``x``, :math:`N = \max(||x||,||y||)`,
    and :math:`y_m` is 0 when m is outside the range of y.

    ``method='fft'`` only works for numerical arrays as it relies on
    `fftconvolve`. In certain cases (i.e., arrays of objects or when
    rounding integers can lose precision), ``method='direct'`` is always used.

    When using "same" mode with even-length inputs, the outputs of `correlate`
    and `correlate2d` differ: There is a 1-index offset between them.

    Examples
    --------
    Implement a matched filter using cross-correlation, to recover a signal
    """
    """
    Compute the cross-correlation of two arrays in1 and in2.

    Parameters:
    - in1 : array_like
        First input array.
    - in2 : array_like
        Second input array.
    - mode : {'valid', 'same', 'full'}, optional
        Specifies the cross-correlation mode.
        'valid' returns output only where overlap exists.
        'same' returns output of the same shape as the largest input.
        'full' returns all possible cross-correlation values.
    - method : {'auto', 'fft', 'direct'}, optional
        Specifies the method used for computation.
        'auto' uses 'fft' for large inputs, otherwise 'direct'.
        'fft' computes using Fast Fourier Transform.
        'direct' computes using direct convolution.
    
    Returns:
    - ndarray
        Cross-correlation of in1 with in2.
    
    Raises:
    - ValueError
        If in1 and in2 have different dimensionalities or if mode is invalid.

    Notes:
    - This function uses convolution for mode 'fft' or 'auto', and direct convolution otherwise.
    - Valid modes are 'valid', 'same', or 'full'.
    - For 'fft' or 'auto' methods, it calls convolve() with transformed input.
    """
    in1 = np.asarray(in1)  # Convert in1 to numpy array
    in2 = np.asarray(in2)  # Convert in2 to numpy array

    if in1.ndim == in2.ndim == 0:  # Check if both inputs are scalars
        return in1 * in2.conj()  # Return the complex conjugate product
    elif in1.ndim != in2.ndim:  # Check if inputs have different dimensions
        raise ValueError("in1 and in2 should have the same dimensionality")  # Raise error for dimensionality mismatch

    # Lookup the mode in the _modedict dictionary
    try:
        val = _modedict[mode]
    except KeyError as e:
        raise ValueError("Acceptable mode flags are 'valid', 'same', or 'full'.") from e  # Raise error for invalid mode

    # Determine computation method based on 'method' parameter
    if method in ('fft', 'auto'):
        return convolve(in1, _reverse_and_conj(in2), mode, method)  # Use convolve with transformed input for 'fft' or 'auto'
    elif method == 'direct':
        # 如果方法选择为 'direct'
        # 快速路径，针对可能的一维输入，使用更快的 numpy.correlate
        if _np_conv_ok(in1, in2, mode):
            return np.correlate(in1, in2, mode)

        # 当 in2.size > in1.size 时，_correlateND 要慢得多，因此交换它们
        # 并在 mode == 'full' 时恢复效果。此外，在 'valid' 模式下，如果 in2 大于 in1，它也会失败，因此也要交换它们。
        # 对于 'same' 模式，不要交换输入，因为 in1 的形状很重要。
        swapped_inputs = ((mode == 'full') and (in2.size > in1.size) or
                          _inputs_swap_needed(mode, in1.shape, in2.shape))

        if swapped_inputs:
            # 如果需要交换，交换 in1 和 in2
            in1, in2 = in2, in1

        if mode == 'valid':
            # 计算输出数组的形状 ps
            ps = [i - j + 1 for i, j in zip(in1.shape, in2.shape)]
            # 创建一个空的输出数组 out
            out = np.empty(ps, in1.dtype)
            # 调用 _correlateND 函数计算相关性
            z = _sigtools._correlateND(in1, in2, out, val)

        else:
            # 计算输出数组的形状 ps
            ps = [i + j - 1 for i, j in zip(in1.shape, in2.shape)]
            
            # 对输入进行零填充
            in1zpadded = np.zeros(ps, in1.dtype)
            sc = tuple(slice(0, i) for i in in1.shape)
            in1zpadded[sc] = in1.copy()

            if mode == 'full':
                # 创建一个空的输出数组 out
                out = np.empty(ps, in1.dtype)
            elif mode == 'same':
                # 创建一个空的输出数组 out，形状与 in1 相同
                out = np.empty(in1.shape, in1.dtype)

            # 调用 _correlateND 函数计算相关性
            z = _sigtools._correlateND(in1zpadded, in2, out, val)

        if swapped_inputs:
            # 如果进行了输入交换，反转并共轭以撤消交换输入的效果
            z = _reverse_and_conj(z)

        # 返回计算得到的相关性结果 z
        return z

    else:
        # 如果 method 不是 'auto', 'direct', 或 'fft'，则抛出 ValueError 异常
        raise ValueError("Acceptable method flags are 'auto',"
                         " 'direct', or 'fft'.")
# 计算不同模式下的滞后范围
if mode == "full":
    # 输出为输入的完整离散线性卷积
    lags = np.arange(-in2_len + 1, in1_len)
elif mode == "same":
    # 输出与 `in1` 相同大小，以 'full' 输出为中心
    # 计算完整输出
    lags = np.arange(-in2_len + 1, in1_len)
    # 确定完整输出的中点
    mid = lags.size // 2
    # 根据中点确定用于 'midpoint' 的 lag_bound
    lag_bound = in1_len // 2
    # 计算偶数和奇数情况下的 lag 范围
    if in1_len % 2 == 0:
        lags = lags[(mid-lag_bound):(mid+lag_bound)]
    else:
        lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    # 如果模式为 "valid"
    elif mode == "valid":
        # 输出结果只包含不依赖于零填充的元素。在 'valid' 模式下，`in1` 或 `in2`
        # 必须在每个维度上至少与另一个维度大小相同。

        # 计算两个输入的长度差
        lag_bound = in1_len - in2_len
        # 如果 lag_bound 大于等于零，表示 in1 至少和 in2 相同大小或更大
        if lag_bound >= 0:
            # 创建一个从 0 到 lag_bound 的整数数组，包括 lag_bound
            lags = np.arange(lag_bound + 1)
        else:
            # 创建一个从 lag_bound 到 1 的整数数组，不包括 1
            lags = np.arange(lag_bound, 1)
    # 返回计算得到的 lags 数组
    return lags
def _centered(arr, newshape):
    # 返回数组的中心 newshape 部分。
    newshape = np.asarray(newshape)  # 将 newshape 转换为 NumPy 数组
    currshape = np.array(arr.shape)  # 获取 arr 的形状并转换为 NumPy 数组
    startind = (currshape - newshape) // 2  # 计算开始索引，使得取中心部分
    endind = startind + newshape  # 计算结束索引
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]  # 创建切片对象
    return arr[tuple(myslice)]  # 返回中心部分的数组


def _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False):
    """处理频域卷积的 axes 参数。

    返回标准形式的输入和 axes，消除多余的轴，必要时交换输入，并检查各种潜在错误。

    Parameters
    ----------
    in1 : array
        第一个输入。
    in2 : array
        第二个输入。
    mode : str {'full', 'valid', 'same'}, optional
        表示输出大小的字符串。
        更多信息请参见 `fftconvolve` 的文档。
    axes : list of ints
        要计算 FFT 的轴。
    sorted_axes : bool, optional
        如果为 `True`，则对轴进行排序。
        默认为 `False`，不排序。

    Returns
    -------
    in1 : array
        第一个输入，可能与第二个输入交换。
    in2 : array
        第二个输入，可能与第一个输入交换。
    axes : list of ints
        要计算 FFT 的轴。

    """
    s1 = in1.shape  # 获取输入 in1 的形状
    s2 = in2.shape  # 获取输入 in2 的形状
    noaxes = axes is None  # 检查是否没有指定 axes

    _, axes = _init_nd_shape_and_axes(in1, shape=None, axes=axes)  # 初始化形状和轴

    if not noaxes and not len(axes):
        raise ValueError("when provided, axes cannot be empty")  # 如果提供了 axes，但它为空，则引发 ValueError

    # 长度为 1 的轴可以依赖广播规则进行乘法，不需要 FFT。
    axes = [a for a in axes if s1[a] != 1 and s2[a] != 1]  # 过滤掉长度为 1 的轴

    if sorted_axes:
        axes.sort()  # 如果需要，对轴进行排序

    # 检查除了指定轴外的所有轴上的形状是否兼容
    if not all(s1[a] == s2[a] or s1[a] == 1 or s2[a] == 1
               for a in range(in1.ndim) if a not in axes):
        raise ValueError("incompatible shapes for in1 and in2:"
                         f" {s1} and {s2}")

    # 检查输入大小是否与 'valid' 模式兼容
    if _inputs_swap_needed(mode, s1, s2, axes=axes):
        # 卷积是可交换的；顺序对输出没有影响。
        in1, in2 = in2, in1  # 如果需要交换输入，则进行交换

    return in1, in2, axes  # 返回处理后的输入和轴


def _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=False):
    """在频域中对两个数组进行卷积。

    此函数仅实现基本的与 FFT 相关的操作。
    具体而言，它将信号转换为频域，将它们相乘，然后将它们转换回时域。
    关于轴、形状、卷积模式等的计算在更高级的函数中实现，
    例如 `fftconvolve` 和 `oaconvolve`。应该使用那些函数，而不是使用这个。

    Parameters
    ----------
    in1 : array_like
        第一个输入。
    in2 : array_like
        第二个输入。应该与 `in1` 具有相同的维度。

    axes : list of ints
        要进行 FFT 计算的轴。
    shape : tuple
        输出数组的形状。
    calc_fast_len : bool, optional
        如果为 `True`，计算快速长度。

    """
    # 如果没有指定计算FFT的轴，则直接返回输入数组的点积结果
    if not len(axes):
        return in1 * in2

    # 判断是否需要处理复数结果
    complex_result = (in1.dtype.kind == 'c' or in2.dtype.kind == 'c')

    # 如果需要计算最优的FFT长度
    if calc_fast_len:
        # 对每个指定的轴计算下一个最快的FFT长度
        fshape = [
            sp_fft.next_fast_len(shape[a], not complex_result) for a in axes]
    else:
        # 否则使用给定的形状参数
        fshape = shape

    # 根据是否需要复数结果选择对应的FFT和IFFT函数
    if not complex_result:
        fft, ifft = sp_fft.rfftn, sp_fft.irfftn
    else:
        fft, ifft = sp_fft.fftn, sp_fft.ifftn

    # 对输入数组进行FFT变换
    sp1 = fft(in1, fshape, axes=axes)
    sp2 = fft(in2, fshape, axes=axes)

    # 计算频域中的乘积
    ret = ifft(sp1 * sp2, fshape, axes=axes)

    # 如果需要计算最优FFT长度，则根据原始的形状参数进行切片操作
    if calc_fast_len:
        fslice = tuple([slice(sz) for sz in shape])
        ret = ret[fslice]

    # 返回线性卷积结果
    return ret
def _apply_conv_mode(ret, s1, s2, mode, axes):
    """Calculate the convolution result shape based on the `mode` argument.

    Returns the result sliced to the correct size for the given mode.

    Parameters
    ----------
    ret : array
        The result array, with the appropriate shape for the 'full' mode.
    s1 : list of int
        The shape of the first input.
    s2 : list of int
        The shape of the second input.
    mode : str {'full', 'valid', 'same'}
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the convolution.

    Returns
    -------
    ret : array
        A copy of `res`, sliced to the correct size for the given `mode`.

    """
    # 根据 `mode` 参数计算卷积结果的形状
    if mode == "full":
        # 如果模式是 'full'，直接返回结果的副本
        return ret.copy()
    elif mode == "same":
        # 如果模式是 'same'，返回结果在正确大小上进行居中处理后的副本
        return _centered(ret, s1).copy()
    elif mode == "valid":
        # 如果模式是 'valid'，计算有效输出的形状，并返回处理后的副本
        shape_valid = [ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
                       for a in range(ret.ndim)]
        return _centered(ret, shape_valid).copy()
    else:
        # 如果模式不是 'valid'、'same' 或 'full'，抛出值错误的异常
        raise ValueError("acceptable mode flags are 'valid',"
                         " 'same', or 'full'")


def fftconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using FFT.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).

    As of v0.19, `convolve` automatically chooses this method or the direct
    method based on an estimation of which is faster.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    """
    """
    in1 = np.asarray(in1)
    将输入in1转换为NumPy数组，以确保其为NumPy数组类型
    in2 = np.asarray(in2)
    将输入in2转换为NumPy数组，以确保其为NumPy数组类型

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        如果in1和in2都是0维的（标量输入）
        return in1 * in2
        直接返回它们的乘积
    elif in1.ndim != in2.ndim:
        如果in1和in2的维度不相等
        raise ValueError("in1 and in2 should have the same dimensionality")
        抛出值错误，提示in1和in2应具有相同的维度

    elif in1.size == 0 or in2.size == 0:  # empty arrays
        如果in1或in2是空数组（大小为0）
        return np.array([])
        返回一个空的NumPy数组

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes,
                                          sorted_axes=False)
    调用_init_freq_conv_axes函数初始化频域卷积的输入参数in1、in2和轴axes

    s1 = in1.shape
    获取in1的形状保存到s1
    s2 = in2.shape
    获取in2的形状保存到s2

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
             for i in range(in1.ndim)]
    根据in1和in2的形状以及轴axes的定义，计算卷积结果的形状shape

    ret = _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True)
    调用_freq_domain_conv函数进行频域卷积，计算卷积结果ret

    return _apply_conv_mode(ret, s1, s2, mode, axes)
    调用_apply_conv_mode函数，根据给定的模式（mode）和轴（axes），对卷积结果进行处理并返回
    """
    # 设置传统FFT方法的参数
    fallback = (s1+s2-1, None, s1, s2)

    # 如果两个数组的大小相同，或者其中一个数组的大小为1，则使用传统FFT方法
    if s1 == s2 or s1 == 1 or s2 == 1:
        return fallback

    # 确保 s1 >= s2，方便后续处理
    if s2 > s1:
        s1, s2 = s2, s1
        swapped = True
    else:
        swapped = False

    # 如果 s2 大于等于 s1 的一半，无法找到有用的块大小，返回传统FFT方法的参数
    if s2 >= s1/2:
        return fallback

    # 计算重叠的长度
    overlap = s2 - 1

    # 计算最优块大小
    opt_size = -overlap * lambertw(-1/(2*math.e*overlap), k=-1).real
    block_size = sp_fft.next_fast_len(math.ceil(opt_size))

    # 如果块大小大于等于 s1，返回传统FFT方法的参数
    if block_size >= s1:
        return fallback

    # 根据是否交换了 s1 和 s2 来确定每个数组的步长
    if not swapped:
        in1_step = block_size - s2 + 1
        in2_step = s2
    else:
        in1_step = s2
        in2_step = block_size - s2 + 1
    # 返回四个变量：block_size, overlap, in1_step, in2_step
    return block_size, overlap, in1_step, in2_step
# 使用重叠-加法方法对两个N维数组进行卷积计算。
# 输出大小由 `mode` 参数确定，支持 'full', 'valid', 'same' 三种模式。

def oaconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using the overlap-add method.

    Convolve `in1` and `in2` using the overlap-add method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    and generally much faster than `fftconvolve` when one array is much
    larger than the other, but can be slower when only a few output values are
    needed or when the arrays are very similar in shape, and can only
    output float arrays (int or object array inputs will be cast to float).

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    convolve : Uses the direct convolution or FFT convolution algorithm
               depending on which is faster.
    fftconvolve : An implementation of convolution using FFT.

    Notes
    -----
    .. versionadded:: 1.4.0

    References
    ----------
    .. [1] Wikipedia, "Overlap-add_method".
           https://en.wikipedia.org/wiki/Overlap-add_method
    .. [2] Richard G. Lyons. Understanding Digital Signal Processing,
           Third Edition, 2011. Chapter 13.10.
           ISBN 13: 978-0137-02741-5

    Examples
    --------
    Convolve a 100,000 sample signal with a 512-sample filter.

    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> sig = rng.standard_normal(100000)
    >>> filt = signal.firwin(512, 0.01)
    >>> fsig = signal.oaconvolve(sig, filt)

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('White noise')
    >>> ax_mag.plot(fsig)
    >>> ax_mag.set_title('Filtered noise')
    >>> fig.tight_layout()
    >>> fig.show()

    """
    # 将输入转换为数组
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    # 如果输入是标量，则直接返回乘积结果
    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    # 如果输入数组的维度不同，则抛出值错误异常
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    # 如果其中一个输入数组为空数组，则返回一个空的 NumPy 数组
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])
    # 如果两个输入数组的形状相同，则调用 fftconvolve 函数进行卷积计算
    elif in1.shape == in2.shape:  # Equivalent to fftconvolve
        return fftconvolve(in1, in2, mode=mode, axes=axes)

    # 初始化频域卷积的参数，并根据需要对轴进行排序
    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes,
                                          sorted_axes=True)

    s1 = in1.shape  # 获取输入数组 in1 的形状
    s2 = in2.shape  # 获取输入数组 in2 的形状

    if not axes:
        ret = in1 * in2  # 对应位置相乘
        return _apply_conv_mode(ret, s1, s2, mode, axes)

    # 计算最终输出的形状，考虑到频域卷积的特殊情况
    shape_final = [None if i not in axes else
                   s1[i] + s2[i] - 1 for i in range(in1.ndim)]

    # 计算输出的块大小、重叠、输入数组的步长等参数
    optimal_sizes = ((-1, -1, s1[i], s2[i]) if i not in axes else
                     _calc_oa_lens(s1[i], s2[i]) for i in range(in1.ndim))
    block_size, overlaps, \
        in1_step, in2_step = zip(*optimal_sizes)

    # 如果每个维度中只有一个块，则回退到 fftconvolve 函数处理
    if in1_step == s1 and in2_step == s2:
        return fftconvolve(in1, in2, mode=mode, axes=axes)

    # 计算步数和填充大小
    nsteps1 = []
    nsteps2 = []
    pad_size1 = []
    pad_size2 = []
    for i in range(in1.ndim):
        if i not in axes:
            pad_size1 += [(0, 0)]
            pad_size2 += [(0, 0)]
            continue

        if s1[i] > in1_step[i]:
            curnstep1 = math.ceil((s1[i]+1)/in1_step[i])
            if (block_size[i] - overlaps[i])*curnstep1 < shape_final[i]:
                curnstep1 += 1

            curpad1 = curnstep1*in1_step[i] - s1[i]
        else:
            curnstep1 = 1
            curpad1 = 0

        if s2[i] > in2_step[i]:
            curnstep2 = math.ceil((s2[i]+1)/in2_step[i])
            if (block_size[i] - overlaps[i])*curnstep2 < shape_final[i]:
                curnstep2 += 1

            curpad2 = curnstep2*in2_step[i] - s2[i]
        else:
            curnstep2 = 1
            curpad2 = 0

        nsteps1 += [curnstep1]
        nsteps2 += [curnstep2]
        pad_size1 += [(0, curpad1)]
        pad_size2 += [(0, curpad2)]

    # 根据计算得到的填充大小，对输入数组进行常数填充
    if not all(curpad == (0, 0) for curpad in pad_size1):
        in1 = np.pad(in1, pad_size1, mode='constant', constant_values=0)

    if not all(curpad == (0, 0) for curpad in pad_size2):
        in2 = np.pad(in2, pad_size2, mode='constant', constant_values=0)

    # 将重叠-添加方法的部分重塑为输入块大小的形状
    split_axes = [iax+i for i, iax in enumerate(axes)]
    fft_axes = [iax+1 for iax in split_axes]
    # 将每个新的维度插入到对应的重塑维度之前，以确保最终数据的正确布局。
    reshape_size1 = list(in1_step)  # 复制输入1的步长作为重塑大小的初始列表
    reshape_size2 = list(in2_step)  # 复制输入2的步长作为重塑大小的初始列表
    for i, iax in enumerate(split_axes):  # 遍历分割轴及其索引
        reshape_size1.insert(iax, nsteps1[i])  # 在输入1的重塑大小列表中插入新的维度
        reshape_size2.insert(iax, nsteps2[i])  # 在输入2的重塑大小列表中插入新的维度

    in1 = in1.reshape(*reshape_size1)  # 使用重塑大小列表重塑输入1
    in2 = in2.reshape(*reshape_size2)  # 使用重塑大小列表重塑输入2

    # 进行卷积操作。
    fft_shape = [block_size[i] for i in axes]  # 创建 FFT 的形状
    ret = _freq_domain_conv(in1, in2, fft_axes, fft_shape, calc_fast_len=False)  # 执行频域卷积操作

    # 执行重叠-添加算法。
    for ax, ax_fft, ax_split in zip(axes, fft_axes, split_axes):  # 遍历轴、FFT 轴和分割轴
        overlap = overlaps[ax]  # 获取当前轴的重叠部分大小
        if overlap is None:  # 如果没有重叠部分，跳过当前轴
            continue

        # 按照重叠部分对结果进行分割
        ret, overpart = np.split(ret, [-overlap], ax_fft)
        overpart = np.split(overpart, [-1], ax_split)[0]

        # 更新分割后的部分
        ret_overpart = np.split(ret, [overlap], ax_fft)[0]
        ret_overpart = np.split(ret_overpart, [1], ax_split)[1]
        ret_overpart += overpart  # 将重叠部分添加回去

    # 将结果重塑回正确的维度。
    shape_ret = [ret.shape[i] if i not in fft_axes else
                 ret.shape[i]*ret.shape[i-1]
                 for i in range(ret.ndim) if i not in split_axes]
    ret = ret.reshape(*shape_ret)  # 使用重塑的形状对结果进行重塑

    # 对结果进行切片以得到最终的大小。
    slice_final = tuple([slice(islice) for islice in shape_final])
    ret = ret[slice_final]

    # 应用卷积模式并返回最终结果。
    return _apply_conv_mode(ret, s1, s2, mode, axes)
def _numeric_arrays(arrays, kinds='buifc'):
    """
    See if a list of arrays are all numeric.

    Parameters
    ----------
    arrays : array or list of arrays
        arrays to check if numeric.
    kinds : string-like
        The dtypes of the arrays to be checked. If the dtype.kind of
        the ndarrays are not in this string the function returns False and
        otherwise returns True.
    """
    # 如果输入是单个 NumPy 数组，则检查其 dtype 是否在指定的 kinds 中
    if type(arrays) == np.ndarray:
        return arrays.dtype.kind in kinds
    # 否则，遍历数组列表中的每个数组
    for array_ in arrays:
        # 如果数组的 dtype.kind 不在 kinds 中，则返回 False
        if array_.dtype.kind not in kinds:
            return False
    # 如果所有数组的 dtype.kind 都在 kinds 中，则返回 True
    return True


def _conv_ops(x_shape, h_shape, mode):
    """
    Find the number of operations required for direct/fft methods of
    convolution. The direct operations were recorded by making a dummy class to
    record the number of operations by overriding ``__mul__`` and ``__add__``.
    The FFT operations rely on the (well-known) computational complexity of the
    FFT (and the implementation of ``_freq_domain_conv``).

    """
    # 根据模式计算输出形状
    if mode == "full":
        out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    elif mode == "valid":
        out_shape = [abs(n - k) + 1 for n, k in zip(x_shape, h_shape)]
    elif mode == "same":
        out_shape = x_shape
    else:
        # 如果 mode 不是 'valid', 'same', 'full' 中的一种，则引发 ValueError
        raise ValueError("Acceptable mode flags are 'valid',"
                         f" 'same', or 'full', not mode={mode}")

    s1, s2 = x_shape, h_shape
    # 如果输入是一维数组
    if len(x_shape) == 1:
        s1, s2 = s1[0], s2[0]
        # 根据模式计算直接方法所需的操作数
        if mode == "full":
            direct_ops = s1 * s2
        elif mode == "valid":
            direct_ops = (s2 - s1 + 1) * s1 if s2 >= s1 else (s1 - s2 + 1) * s2
        elif mode == "same":
            direct_ops = (s1 * s2 if s1 < s2 else
                          s1 * s2 - (s2 // 2) * ((s2 + 1) // 2))
    else:
        # 如果输入不是一维数组，则计算直接方法所需的操作数
        if mode == "full":
            direct_ops = min(_prod(s1), _prod(s2)) * _prod(out_shape)
        elif mode == "valid":
            direct_ops = min(_prod(s1), _prod(s2)) * _prod(out_shape)
        elif mode == "same":
            direct_ops = _prod(s1) * _prod(s2)

    # 计算 FFT 方法所需的操作数，假设有三次大小为 full_out_shape 的 FFT 操作
    full_out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    N = _prod(full_out_shape)
    fft_ops = 3 * N * np.log(N)  # 3 separate FFTs of size full_out_shape
    return fft_ops, direct_ops


def _fftconv_faster(x, h, mode):
    """
    See if using fftconvolve or convolve is faster.

    Parameters
    ----------
    x : np.ndarray
        Signal
    h : np.ndarray
        Kernel
    mode : str
        Mode passed to convolve

    Returns
    -------
    fft_faster : bool

    Notes
    -----
    See docstring of `choose_conv_method` for details on tuning hardware.

    See pull request 11031 for more detail:
    https://github.com/scipy/scipy/pull/11031.

    """
    # 调用 _conv_ops 函数获取 FFT 和直接方法的操作数
    fft_ops, direct_ops = _conv_ops(x.shape, h.shape, mode)
    # 根据信号 x 的维度选择偏移量
    offset = -1e-3 if x.ndim == 1 else -1e-4
    # 定义常量字典，根据输入数组维度不同选择不同的值
    constants = {
        # 如果输入数组是一维的，则选择以下常量值
        "valid": (1.89095737e-9, 2.1364985e-10, offset),
        "full": (1.7649070e-9, 2.1414831e-10, offset),
        "same": (3.2646654e-9, 2.8478277e-10, offset)
        if h.size <= x.size  # 如果 h 的大小小于等于 x 的大小，则选择 same 对应的常量
        else  # 否则选择另一组常量
        (3.21635404e-9, 1.1773253e-8, -1e-5),
    } if x.ndim == 1  # 如果 x 的维度为 1，则使用第一组常量字典
    else {  # 否则使用第二组常量字典
        "valid": (1.85927e-9, 2.11242e-8, offset),
        "full": (1.99817e-9, 1.66174e-8, offset),
        "same": (2.04735e-9, 1.55367e-8, offset),
    }
    # 从常量字典中根据 mode 选择对应的 FFT、直接方法和偏移量值
    O_fft, O_direct, O_offset = constants[mode]
    # 返回比较结果，判断是否使用 FFT 方法执行的条件
    return O_fft * fft_ops < O_direct * direct_ops + O_offset
# 定义一个函数，用于将数组 `x` 在所有维度上进行反转并计算复共轭
def _reverse_and_conj(x):
    reverse = (slice(None, None, -1),) * x.ndim  # 创建用于反转的切片对象，反转所有维度
    return x[reverse].conj()  # 返回反转后数组的复共轭


# 检查 numpy 是否支持对 `volume` 和 `kernel` 的卷积操作（即两者均为1D ndarray且形状适当）
def _np_conv_ok(volume, kernel, mode):
    if volume.ndim == kernel.ndim == 1:  # 检查 `volume` 和 `kernel` 是否都是1维数组
        if mode in ('full', 'valid'):  # 如果模式是 'full' 或 'valid'
            return True  # 返回 True 表示支持
        elif mode == 'same':  # 如果模式是 'same'
            return volume.size >= kernel.size  # 返回比较 `volume` 和 `kernel` 大小是否符合条件
    else:
        return False  # 其它情况下返回 False，表示不支持


# 测量语句或函数执行时间，返回以秒为单位的时间
def _timeit_fast(stmt="pass", setup="pass", repeat=3):
    timer = timeit.Timer(stmt, setup)  # 创建计时器对象

    # 确定每次重复调用的次数，使得每次重复的总时间 >= 5 毫秒
    x = 0
    for p in range(0, 10):
        number = 10**p
        x = timer.timeit(number)  # 计算执行 `stmt` `number` 次所需的时间（秒）
        if x >= 5e-3 / 10:  # 如果达到 5 毫秒的要求（第一次循环 1/10 倍）
            break
    if x > 1:  # 如果时间较长（大于1秒）
        best = x  # 直接选取该时间作为最佳时间
    else:
        number *= 10
        r = timer.repeat(repeat, number)  # 多次重复计时，取最小时间作为最佳时间
        best = min(r)

    sec = best / number  # 计算每次操作的平均时间
    return sec  # 返回平均时间（秒）


# 选择最快的卷积/相关方法
def choose_conv_method(in1, in2, mode='full', measure=False):
    """
    Find the fastest convolution/correlation method.

    This primarily exists to be called during the ``method='auto'`` option in
    `convolve` and `correlate`. It can also be used to determine the value of
    ``method`` for many different convolutions of the same dtype/shape.
    In addition, it supports timing the convolution to adapt the value of
    ``method`` to a particular set of inputs and/or hardware.
    """
    mode : str {'full', 'valid', 'same'}, optional
        # 选择输出的大小：
        ``full``
           输出是输入的完整离散线性卷积。
           (默认)
        ``valid``
           输出仅包含不依赖于零填充的元素。
        ``same``
           输出与 `in1` 相同大小，相对于 'full' 输出进行了居中处理。

    measure : bool, optional
        # 是否进行测量：
        如果为 True，将使用两种方法运行和计时 `in1` 和 `in2` 的卷积，并返回最快的方法。
        如果为 False（默认），则使用预先计算的值预测最快的方法。

    Returns
    -------
    method : str
        # 返回一个字符串，指示哪种卷积方法最快，可以是 'direct' 或 'fft'
    times : dict, optional
        # 一个包含每种方法所需时间（以秒为单位）的字典。
        仅当 ``measure=True`` 时返回该值。

    See Also
    --------
    convolve
    correlate

    Notes
    -----
    # 一般来说，此方法对于随机选择的输入大小的 2D 信号精度达到 99%，对于 1D 信号达到 85%。
    对于更高精度，请使用 ``measure=True`` 来通过计时卷积来找到最快的方法。
    这可以用来避免后续查找最快 ``method`` 的最小开销，或者根据特定的输入集调整 ``method`` 的值。

    在 Amazon EC2 r5a.2xlarge 机器上运行实验来测试此函数。这些实验测量了使用 ``method='auto'`` 时所需时间与最快方法所需时间之间的比率
    （即 ``ratio = time_auto / min(time_fft, time_direct)``）。在这些实验中，我们发现：

    * 对于 1D 信号，这个比率小于 1.5 的概率为 95%，对于 2D 信号小于 2.5 的概率为 99%。
    * 对于 1D/2D 信号，这个比率总是小于 2.5/5。
    * 对于使用 ``method='direct'`` 时 1 到 10 毫秒的 1D 卷积，这个函数的精度最低。在我们的实验中，一个好的近似是 ``1e6 <= in1.size * in2.size <= 1e7``。

    2D 结果几乎肯定可以推广到 3D/4D 等，因为实现是相同的（1D 实现不同）。

    所有上述数字都是针对 EC2 机器的。但是，我们发现这个函数在硬件上具有相当良好的泛化能力。速度测试的质量与用于调整此函数数字的机器上的相同测试（一台配备 16GB RAM 和 2.5GHz Intel i7 处理器的 2014 年中期 15 英寸 MacBook Pro）相似，甚至稍好一些。

    在 `fftconvolve` 支持输入但此函数返回 `direct` 的情况下（例如，为了防止浮点数整数精度）。

    .. versionadded:: 0.19

    Examples
    --------
    Estimate the fastest method for a given input:

    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> img = rng.random((32, 32))
    >>> filter = rng.random((8, 8))
    >>> method = signal.choose_conv_method(img, filter, mode='same')
    >>> method
    'fft'

    This can then be applied to other arrays of the same dtype and shape:

    >>> img2 = rng.random((32, 32))
    >>> filter2 = rng.random((8, 8))
    >>> corr2 = signal.correlate(img2, filter2, mode='same', method=method)
    >>> conv2 = signal.convolve(img2, filter2, mode='same', method=method)

    The output of this function (``method``) works with `correlate` and
    `convolve`.

    """
    # 将输入转换为 NumPy 数组
    volume = np.asarray(in1)
    kernel = np.asarray(in2)

    # 如果需要测量执行时间
    if measure:
        # 初始化时间记录字典
        times = {}
        # 遍历不同的卷积方法
        for method in ['fft', 'direct']:
            # 计算每种方法的执行时间并记录
            times[method] = _timeit_fast(lambda: convolve(volume, kernel,
                                         mode=mode, method=method))

        # 选择最快的方法
        chosen_method = 'fft' if times['fft'] < times['direct'] else 'direct'
        return chosen_method, times

    # 对于整数输入，当需要比浮点数提供更高的精度时进行处理
    if any([_numeric_arrays([x], kinds='ui') for x in [volume, kernel]]):
        # 计算最大值，用于评估是否需要直接方法
        max_value = int(np.abs(volume).max()) * int(np.abs(kernel).max())
        max_value *= int(min(volume.size, kernel.size))
        # 如果超过浮点数的精度范围，则选择直接方法
        if max_value > 2**np.finfo('float').nmant - 1:
            return 'direct'

    # 对于布尔类型的输入数组，选择直接方法
    if _numeric_arrays([volume, kernel], kinds='b'):
        return 'direct'

    # 对于一般的数值输入数组，根据 `_fftconv_faster` 函数的结果选择方法
    if _numeric_arrays([volume, kernel]):
        if _fftconv_faster(volume, kernel, mode):
            return 'fft'

    # 默认选择直接方法
    return 'direct'
def convolve(in1, in2, mode='full', method='auto'):
    """
    Convolve two N-dimensional arrays.

    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.

        ``direct``
           The convolution is determined directly from sums, the definition of
           convolution.
        ``fft``
           The Fourier Transform is used to perform the convolution by calling
           `fftconvolve`.
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).  See Notes for more detail.

           .. versionadded:: 0.19.0

    Returns
    -------
    convolve : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    Warns
    -----
    RuntimeWarning
        Use of the FFT convolution on input containing NAN or INF will lead
        to the entire output being NAN or INF. Use method='direct' when your
        input contains NAN or INF values.

    See Also
    --------
    numpy.polymul : performs polynomial multiplication (same operation, but
                    also accepts poly1d objects)
    choose_conv_method : chooses the fastest appropriate convolution method
    fftconvolve : Always uses the FFT method.
    oaconvolve : Uses the overlap-add method to do convolution, which is
                 generally faster when the input arrays are large and
                 significantly different in size.

    Notes
    -----
    By default, `convolve` and `correlate` use ``method='auto'``, which calls
    `choose_conv_method` to choose the fastest method using pre-computed
    values (`choose_conv_method` can also measure real-world timing with a
    keyword argument). Because `fftconvolve` relies on floating point numbers,
    there are certain constraints that may force ``method='direct'`` (more detail
    in `choose_conv_method` docstring).

    Examples
    --------
    Smooth a square pulse using a Hann window:

    >>> import numpy as np
    >>> from scipy import signal
    >>> sig = np.repeat([0., 1., 0.], 100)
    """

    # 根据传入的参数调用选择最快的卷积方法
    convolve_func = choose_conv_method(in1, in2, mode, method)
    # 调用选定的卷积方法计算结果并返回
    return convolve_func(in1, in2, mode)
    >>> win = signal.windows.hann(50)
    # 创建一个长度为50的汉宁窗口（Hann window）

    >>> filtered = signal.convolve(sig, win, mode='same') / sum(win)
    # 使用汉宁窗口对信号 sig 进行卷积，并对卷积结果进行归一化处理

    >>> import matplotlib.pyplot as plt
    # 导入 matplotlib 库中的 pyplot 模块，用于绘图

    >>> fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
    # 创建一个包含三个子图的图形窗口，共享 x 轴

    >>> ax_orig.plot(sig)
    # 在第一个子图上绘制原始信号 sig
    >>> ax_orig.set_title('Original pulse')
    # 设置第一个子图的标题为 'Original pulse'
    >>> ax_orig.margins(0, 0.1)
    # 设置第一个子图的边距

    >>> ax_win.plot(win)
    # 在第二个子图上绘制汉宁窗口 win
    >>> ax_win.set_title('Filter impulse response')
    # 设置第二个子图的标题为 'Filter impulse response'
    >>> ax_win.margins(0, 0.1)
    # 设置第二个子图的边距

    >>> ax_filt.plot(filtered)
    # 在第三个子图上绘制经过滤波后的信号 filtered
    >>> ax_filt.set_title('Filtered signal')
    # 设置第三个子图的标题为 'Filtered signal'
    >>> ax_filt.margins(0, 0.1)
    # 设置第三个子图的边距

    >>> fig.tight_layout()
    # 调整图形窗口的布局，使得子图之间的间距合适
    >>> fig.show()
    # 显示图形窗口，展示绘制好的图形

    """
    volume = np.asarray(in1)
    # 将输入参数 in1 转换为 NumPy 数组 volume

    kernel = np.asarray(in2)
    # 将输入参数 in2 转换为 NumPy 数组 kernel

    if volume.ndim == kernel.ndim == 0:
        return volume * kernel
    # 如果 volume 和 kernel 都是零维数组，则返回它们的乘积
    elif volume.ndim != kernel.ndim:
        raise ValueError("volume and kernel should have the same "
                         "dimensionality")
    # 如果 volume 和 kernel 的维度不相同，则引发 ValueError 异常

    if _inputs_swap_needed(mode, volume.shape, kernel.shape):
        # 如果需要交换输入参数的顺序，根据卷积的交换律，顺序不影响输出
        volume, kernel = kernel, volume

    if method == 'auto':
        method = choose_conv_method(volume, kernel, mode=mode)
        # 如果方法选择为 'auto'，则根据输入的 volume 和 kernel 选择合适的卷积方法

    if method == 'fft':
        out = fftconvolve(volume, kernel, mode=mode)
        # 如果方法选择为 'fft'，则使用 FFT 进行卷积计算
        result_type = np.result_type(volume, kernel)
        # 确定卷积结果的数据类型

        if result_type.kind in {'u', 'i'}:
            out = np.around(out)
            # 如果结果类型是无符号整数或整数，则将输出四舍五入为整数

        if np.isnan(out.flat[0]) or np.isinf(out.flat[0]):
            warnings.warn("Use of fft convolution on input with NAN or inf"
                          " results in NAN or inf output. Consider using"
                          " method='direct' instead.",
                          category=RuntimeWarning, stacklevel=2)
            # 如果 FFT 卷积的输入包含 NaN 或 inf，会导致输出也包含 NaN 或 inf，发出警告

        return out.astype(result_type)
        # 返回经过类型转换后的卷积结果

    elif method == 'direct':
        # 如果方法选择为 'direct'
        if _np_conv_ok(volume, kernel, mode):
            return np.convolve(volume, kernel, mode)
            # 如果可以使用 NumPy 的快速卷积函数，则使用之

        return correlate(volume, _reverse_and_conj(kernel), mode, 'direct')
        # 否则使用相关运算计算卷积

    else:
        raise ValueError("Acceptable method flags are 'auto',"
                         " 'direct', or 'fft'.")
    # 如果方法选择不在可接受的选项中，引发 ValueError 异常
# 定义一个函数，用于对 N 维数组执行顺序滤波操作
def order_filter(a, domain, rank):
    """
    Perform an order filter on an N-D array.

    Perform an order filter on the array in. The domain argument acts as a
    mask centered over each pixel. The non-zero elements of domain are
    used to select elements surrounding each input pixel which are placed
    in a list. The list is sorted, and the output for that pixel is the
    element corresponding to rank in the sorted list.

    Parameters
    ----------
    a : ndarray
        The N-dimensional input array.
    domain : array_like
        A mask array with the same number of dimensions as `a`.
        Each dimension should have an odd number of elements.
    rank : int
        A non-negative integer which selects the element from the
        sorted list (0 corresponds to the smallest element, 1 is the
        next smallest element, etc.).

    Returns
    -------
    out : ndarray
        The results of the order filter in an array with the same
        shape as `a`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> x = np.arange(25).reshape(5, 5)
    >>> domain = np.identity(3)
    >>> x
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])
    >>> signal.order_filter(x, domain, 0)
    array([[  0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   1.,   2.,   0.],
           [  0.,   5.,   6.,   7.,   0.],
           [  0.,  10.,  11.,  12.,   0.],
           [  0.,   0.,   0.,   0.,   0.]])
    >>> signal.order_filter(x, domain, 2)
    array([[  6.,   7.,   8.,   9.,   4.],
           [ 11.,  12.,  13.,  14.,   9.],
           [ 16.,  17.,  18.,  19.,  14.],
           [ 21.,  22.,  23.,  24.,  19.],
           [ 20.,  21.,  22.,  23.,  24.]])

    """
    # 将 domain 转换为 NumPy 数组
    domain = np.asarray(domain)
    # 检查 domain 的每个维度是否含有奇数个元素，否则抛出异常
    for dimsize in domain.shape:
        if (dimsize % 2) != 1:
            raise ValueError("Each dimension of domain argument "
                             "should have an odd number of elements.")

    # 将输入数组 a 转换为 NumPy 数组
    a = np.asarray(a)
    # 检查输入数组的数据类型，必须是整数或者浮点数（32位或64位），否则抛出异常
    if not (np.issubdtype(a.dtype, np.integer) 
            or a.dtype in [np.float32, np.float64]):
        raise ValueError(f"dtype={a.dtype} is not supported by order_filter")

    # 使用 ndimage 库的 rank_filter 函数进行顺序滤波操作
    result = ndimage.rank_filter(a, rank, footprint=domain, mode='constant')
    # 返回滤波后的结果数组
    return result
    # kernel_size 是一个数组或可选的参数，指定中值滤波器在每个维度上的窗口大小。每个元素都应为奇数。
    # 如果 kernel_size 是一个标量，则在每个维度上使用相同大小。默认大小为每个维度上的 3。

    # 返回值
    # -------
    # out : ndarray
    #     与输入相同大小的数组，包含中值滤波的结果。

    # 警告
    # -----
    # UserWarning
    #     如果数组大小在任何维度上小于 kernel size，则会发出警告。

    # 参见
    # --------
    # scipy.ndimage.median_filter
    # scipy.signal.medfilt2d

    # 注释
    # -----
    # 更通用的函数 `scipy.ndimage.median_filter` 实现了更高效的中值滤波器，因此运行速度更快。

    # 对于二维图像，使用 `uint8`, `float32` 或 `float64` 数据类型时，
    # 专用函数 `scipy.signal.medfilt2d` 可能更快。

    """
    volume = np.atleast_1d(volume)
    # 将输入的 volume 至少视为一维数组

    if not (np.issubdtype(volume.dtype, np.integer) 
            or volume.dtype in [np.float32, np.float64]):
        # 如果 volume 的数据类型不是整数类型，也不是 float32 或 float64 类型，则抛出错误
        raise ValueError(f"dtype={volume.dtype} is not supported by medfilt")

    if kernel_size is None:
        # 如果 kernel_size 为 None，则默认使用每个维度上大小为 3 的窗口
        kernel_size = [3] * volume.ndim
    kernel_size = np.asarray(kernel_size)
    # 将 kernel_size 转换为 NumPy 数组
    if kernel_size.shape == ():
        # 如果 kernel_size 是标量，则在每个维度上重复使用相同大小
        kernel_size = np.repeat(kernel_size.item(), volume.ndim)

    for k in range(volume.ndim):
        # 检查每个维度上的 kernel_size 是否为奇数，如果不是，则抛出错误
        if (kernel_size[k] % 2) != 1:
            raise ValueError("Each element of kernel_size should be odd.")
    if any(k > s for k, s in zip(kernel_size, volume.shape)):
        # 如果任何一个 kernel_size 大于 volume 在对应维度上的大小，则发出警告
        warnings.warn('kernel_size exceeds volume extent: the volume will be '
                      'zero-padded.',
                      stacklevel=2)

    size = math.prod(kernel_size)
    # 计算 kernel_size 的乘积，即窗口大小
    result = ndimage.rank_filter(volume, size // 2, size=kernel_size,
                                 mode='constant')
    # 使用常数模式进行中值滤波，并返回结果

    return result
    # 返回中值滤波后的结果数组
    ```
# 二维 Wiener 滤波函数，用于对 N 维数组 `im` 执行 Wiener 滤波处理
def wiener(im, mysize=None, noise=None):
    # 将输入的数组 `im` 转换为 NumPy 数组
    im = np.asarray(im)
    
    # 如果未指定 `mysize`，则默认为每个维度上都为 3 的窗口大小
    if mysize is None:
        mysize = [3] * im.ndim
    mysize = np.asarray(mysize)
    if mysize.shape == ():
        mysize = np.repeat(mysize.item(), im.ndim)

    # 估计局部均值
    size = math.prod(mysize)
    lMean = correlate(im, np.ones(mysize), 'same') / size

    # 估计局部方差
    lVar = (correlate(im ** 2, np.ones(mysize), 'same') / size - lMean ** 2)

    # 如果需要估计噪声功率，则计算
    if noise is None:
        noise = np.mean(np.ravel(lVar), axis=0)

    # 计算 Wiener 滤波结果
    res = (im - lMean)
    res *= (1 - noise / lVar)
    res += lMean
    out = np.where(lVar < noise, lMean, res)

    return out


# 二维卷积函数，用于计算两个二维数组 `in1` 和 `in2` 的卷积
def convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
    """
    Convolve two 2-dimensional arrays.

    Convolve `in1` and `in2` with output size determined by `mode`, and
    boundary conditions determined by `boundary` and `fillvalue`.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : {'full', 'valid', 'same'}, optional
        Determines the shape of the output array. Defaults to 'full'.
    boundary : {'fill', 'wrap', 'reflect', 'symm', 'zeros'}, optional
        Determines how the input is extended beyond its boundaries. Defaults to 'fill'.
    fillvalue : scalar, optional
        Value to fill past edges of input if `boundary='fill'`. Defaults to 0.

    Returns
    -------
    out : ndarray
        Result of the convolution.
    """
    # 将输入数组转换为 NumPy 数组
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    # 检查输入数组的维度是否为二维，如果不是则引发 ValueError 异常
    if not in1.ndim == in2.ndim == 2:
        raise ValueError('convolve2d inputs must both be 2-D arrays')

    # 根据指定的模式检查是否需要交换输入数组的顺序
    if _inputs_swap_needed(mode, in1.shape, in2.shape):
        in1, in2 = in2, in1

    # 根据模式字符串获取对应的值
    val = _valfrommode(mode)
    # 根据边界处理方式字符串获取对应的值
    bval = _bvalfromboundary(boundary)
    # 调用内部函数进行二维卷积操作，返回结果数组
    out = _sigtools._convolve2d(in1, in2, 1, val, bval, fillvalue)
    # 返回计算得到的卷积结果数组
    return out
# 定义函数，用于二维数组的交叉相关操作
def correlate2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
    """
    Cross-correlate two 2-dimensional arrays.

    Cross correlate `in1` and `in2` with output size determined by `mode`, and
    boundary conditions determined by `boundary` and `fillvalue`.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:

        ``fill``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm``
           symmetrical boundary conditions.

    fillvalue : scalar, optional
        Value to fill pad input arrays with. Default is 0.

    Returns
    -------
    correlate2d : ndarray
        A 2-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.

    Notes
    -----
    When using "same" mode with even-length inputs, the outputs of `correlate`
    and `correlate2d` differ: There is a 1-index offset between them.

    Examples
    --------
    Use 2D cross-correlation to find the location of a template in a noisy
    image:

    >>> import numpy as np
    >>> from scipy import signal, datasets, ndimage
    >>> rng = np.random.default_rng()
    >>> face = datasets.face(gray=True) - datasets.face(gray=True).mean()
    >>> face = ndimage.zoom(face[30:500, 400:950], 0.5)  # extract the face
    >>> template = np.copy(face[135:165, 140:175])  # right eye
    >>> template -= template.mean()
    >>> face = face + rng.standard_normal(face.shape) * 50  # add noise
    >>> corr = signal.correlate2d(face, template, boundary='symm', mode='same')
    >>> y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1,
    ...                                                     figsize=(6, 15))
    >>> ax_orig.imshow(face, cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_template.imshow(template, cmap='gray')
    >>> ax_template.set_title('Template')
    >>> ax_template.set_axis_off()
    >>> ax_corr.imshow(corr, cmap='gray')
    >>> ax_corr.set_title('Cross-correlation')
    >>> ax_corr.set_axis_off()
    """
    # 调用 SciPy 中的 cross-correlate 函数，执行二维数组的交叉相关计算
    return signal.correlate2d(in1, in2, mode=mode, boundary=boundary, fillvalue=fillvalue)
    >>> ax_orig.plot(x, y, 'ro')
    >>> fig.show()

    """
    将数据绘制为红色圆点并添加到 ax_orig 对象中
    使用 fig 对象显示图形

    in1 = np.asarray(in1)
    将输入参数 in1 转换为 NumPy 数组
    in2 = np.asarray(in2)
    将输入参数 in2 转换为 NumPy 数组

    if not in1.ndim == in2.ndim == 2:
        如果 in1 和 in2 不都是二维数组，则抛出数值错误异常
        raise ValueError('correlate2d inputs must both be 2-D arrays')

    swapped_inputs = _inputs_swap_needed(mode, in1.shape, in2.shape)
    检查是否需要交换输入参数顺序，并返回布尔值 swapped_inputs

    if swapped_inputs:
        如果需要交换输入参数顺序，则交换 in1 和 in2 的值
        in1, in2 = in2, in1

    val = _valfrommode(mode)
    根据给定的 mode 获取相应的值 val
    bval = _bvalfromboundary(boundary)
    根据给定的 boundary 获取相应的值 bval
    out = _sigtools._convolve2d(in1, in2.conj(), 0, val, bval, fillvalue)
    调用 _sigtools 模块中的 _convolve2d 函数，对输入数组 in1 和 in2 进行二维卷积运算，并存储结果在 out 中

    if swapped_inputs:
        如果之前交换过输入参数顺序，则将 out 反转
        out = out[::-1, ::-1]

    return out
    返回卷积运算的结果数组 out
    ```
# 导入必要的库
import numpy as np

# 定义一个函数，用于对二维数组进行中值滤波
def medfilt2d(input, kernel_size=3):
    """
    Median filter a 2-dimensional array.

    Apply a median filter to the `input` array using a local window-size
    given by `kernel_size` (must be odd). The array is zero-padded
    automatically.

    Parameters
    ----------
    input : array_like
        A 2-dimensional input array.
    kernel_size : array_like, optional
        A scalar or a list of length 2, giving the size of the
        median filter window in each dimension.  Elements of
        `kernel_size` should be odd.  If `kernel_size` is a scalar,
        then this scalar is used as the size in each dimension.
        Default is a kernel of size (3, 3).

    Returns
    -------
    out : ndarray
        An array the same size as input containing the median filtered
        result.

    See Also
    --------
    scipy.ndimage.median_filter

    Notes
    -----
    This is faster than `medfilt` when the input dtype is ``uint8``,
    ``float32``, or ``float64``; for other types, this falls back to
    `medfilt`. In some situations, `scipy.ndimage.median_filter` may be
    faster than this function.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> x = np.arange(25).reshape(5, 5)
    >>> x
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])

    # Replaces i,j with the median out of 5*5 window

    >>> signal.medfilt2d(x, kernel_size=5)
    array([[ 0,  0,  2,  0,  0],
           [ 0,  3,  7,  4,  0],
           [ 2,  8, 12,  9,  4],
           [ 0,  8, 12,  9,  0],
           [ 0,  0, 12,  0,  0]])

    # Replaces i,j with the median out of default 3*3 window

    >>> signal.medfilt2d(x)
    array([[ 0,  1,  2,  3,  0],
           [ 1,  6,  7,  8,  4],
           [ 6, 11, 12, 13,  9],
           [11, 16, 17, 18, 14],
           [ 0, 16, 17, 18,  0]])

    # Replaces i,j with the median out of default 5*3 window

    >>> signal.medfilt2d(x, kernel_size=[5,3])
    array([[ 0,  1,  2,  3,  0],
           [ 0,  6,  7,  8,  3],
           [ 5, 11, 12, 13,  8],
           [ 5, 11, 12, 13,  8],
           [ 0, 11, 12, 13,  0]])

    # Replaces i,j with the median out of default 3*5 window

    >>> signal.medfilt2d(x, kernel_size=[3,5])
    array([[ 0,  0,  2,  1,  0],
           [ 1,  5,  7,  6,  3],
           [ 6, 10, 12, 11,  8],
           [11, 15, 17, 16, 13],
           [ 0, 15, 17, 16,  0]])

    # As seen in the examples,
    # kernel numbers must be odd and not exceed original array dim

    """
    # 将输入转换为 numpy 数组
    image = np.asarray(input)

    # 检查数组元素的数据类型，根据不同的类型选择不同的中值滤波方法
    if image.dtype.type not in (np.ubyte, np.float32, np.float64):
        return medfilt(image, kernel_size)

    # 如果未指定 kernel_size，则使用默认值 (3, 3)
    if kernel_size is None:
        kernel_size = [3] * 2
    kernel_size = np.asarray(kernel_size)
    # 检查 kernel_size 是否为单个数值，如果是，则转换为长度为 2 的重复数组
    if kernel_size.shape == ():
        kernel_size = np.repeat(kernel_size.item(), 2)

    # 遍历 kernel_size 中的每个元素，确保每个元素都是奇数
    for size in kernel_size:
        if (size % 2) != 1:
            # 如果发现有偶数元素，抛出数值错误异常
            raise ValueError("Each element of kernel_size should be odd.")

    # 使用 _sigtools._medfilt2d 函数对图像 image 进行二维中值滤波处理，使用指定的 kernel_size
    return _sigtools._medfilt2d(image, kernel_size)
def lfilter(b, a, x, axis=-1, zi=None):
    """
    Filter data along one-dimension with an IIR or FIR filter.

    Filter a data sequence, `x`, using a digital filter.  This works for many
    fundamental data types (including Object type).  The filter is a direct
    form II transposed implementation of the standard difference equation
    (see Notes).

    The function `sosfilt` (and filter design using ``output='sos'``) should be
    preferred over `lfilter` for most filtering tasks, as second-order sections
    have fewer numerical problems.

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    a : array_like
        The denominator coefficient vector in a 1-D sequence.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the filter delays.  It is a vector
        (or array of vectors for an N-dimensional input) of length
        ``max(len(a), len(b)) - 1``.  If `zi` is None or is not given then
        initial rest is assumed.  See `lfiltic` for more information.

    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.

    See Also
    --------
    lfiltic : Construct initial conditions for `lfilter`.
    lfilter_zi : Compute initial state (steady state of step response) for
                 `lfilter`.
    filtfilt : A forward-backward filter, to obtain a filter with zero phase.
    savgol_filter : A Savitzky-Golay filter.
    sosfilt: Filter data using cascaded second-order sections.
    sosfiltfilt: A forward-backward filter using second-order sections.

    Notes
    -----
    The filter function is implemented as a direct II transposed structure.
    This means that the filter implements::

       a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                             - a[1]*y[n-1] - ... - a[N]*y[n-N]

    where `M` is the degree of the numerator, `N` is the degree of the
    denominator, and `n` is the sample number.  It is implemented using
    the following difference equations (assuming M = N)::

         a[0]*y[n] = b[0] * x[n]               + d[0][n-1]
           d[0][n] = b[1] * x[n] - a[1] * y[n] + d[1][n-1]
           d[1][n] = b[2] * x[n] - a[2] * y[n] + d[2][n-1]
         ...
         d[N-2][n] = b[N-1]*x[n] - a[N-1]*y[n] + d[N-1][n-1]
         d[N-1][n] = b[N] * x[n] - a[N] * y[n]

    where `d` are the state variables.

    The rational transfer function describing this filter in the
    """
    # 保证系数向量 `b` 和 `a` 是一维序列，并进行归一化处理
    b = np.asarray(b)
    a = np.asarray(a)
    # 获取输入数组 `x` 的形状信息
    sh = list(x.shape)
    # 若 `axis` 为负数，则计算其绝对索引
    axis = axis % len(sh)
    # 获取输入数组 `x` 在 `axis` 维度上的长度
    Nx = sh[axis]
    # 初始化滤波结果的数组
    y = np.zeros_like(x, dtype=np.float64)
    # 若初始条件 `zi` 未提供，则使用默认初始条件
    if zi is None:
        zi = np.zeros(max(len(a), len(b)) - 1, dtype=np.float64)
    # 初始化状态变量 `d`，长度为 `N`（分母系数的长度）
    d = np.zeros((len(a) - 1, Nx), dtype=np.float64)
    # 遍历输入数组 `x` 的每个子数组，并应用线性滤波
    for i in range(Nx):
        # 获取当前的输入数据 `x` 的子数组
        xx = x.take(i, axis=axis)
        # 应用直接 II 转置结构的差分方程来进行滤波计算
        y[..., i], d[..., i] = _lfilter(b, a, xx, axis, zi, d[..., i])
    # 如果存在输出状态变量 `d`，则计算最终状态
    if zi is not None:
        zf = d[..., -1].copy()
        return y, zf
    else:
        return y
    # 将 b 系数转换为至少为一维的 NumPy 数组
    b = np.atleast_1d(b)
    # 将 a 系数转换为至少为一维的 NumPy 数组
    a = np.atleast_1d(a)
    if len(a) == 1:
        # 当 a 的长度为 1 时执行以下代码块，此路径仅支持 fdgFDGO 类型以匹配下方的 _linear_filter。
        # b, a, x, 或 zi 中的任何一个可以设置 dtype，但不会默认转换其他类型；而是会引发 NotImplementedError。
        
        b = np.asarray(b)  # 将 b 转换为 NumPy 数组
        a = np.asarray(a)  # 将 a 转换为 NumPy 数组
        
        if b.ndim != 1 and a.ndim != 1:
            raise ValueError('object of too small depth for desired array')
            # 如果 b 或 a 的维度不为 1，则引发 ValueError
        
        x = _validate_x(x)  # 调用 _validate_x 函数验证 x
        
        inputs = [b, a, x]  # 创建输入列表包含 b, a, x
        
        if zi is not None:
            # 如果 zi 不为 None，则执行以下代码块
            # _linear_filter 不广播 zi，但会展开单维度。
            
            zi = np.asarray(zi)  # 将 zi 转换为 NumPy 数组
            
            if zi.ndim != x.ndim:
                raise ValueError('object of too small depth for desired array')
                # 如果 zi 的维度与 x 不匹配，则引发 ValueError
            
            expected_shape = list(x.shape)
            expected_shape[axis] = b.shape[0] - 1
            expected_shape = tuple(expected_shape)
            
            # 检查 zi 是否为正确的形状
            if zi.shape != expected_shape:
                strides = zi.ndim * [None]
                if axis < 0:
                    axis += zi.ndim
                for k in range(zi.ndim):
                    if k == axis and zi.shape[k] == expected_shape[k]:
                        strides[k] = zi.strides[k]
                    elif k != axis and zi.shape[k] == expected_shape[k]:
                        strides[k] = zi.strides[k]
                    elif k != axis and zi.shape[k] == 1:
                        strides[k] = 0
                    else:
                        raise ValueError('Unexpected shape for zi: expected '
                                         f'{expected_shape}, found {zi.shape}.')
                zi = np.lib.stride_tricks.as_strided(zi, expected_shape,
                                                     strides)
            
            inputs.append(zi)  # 将 zi 加入输入列表
        
        dtype = np.result_type(*inputs)  # 根据 inputs 的类型推断出 dtype

        if dtype.char not in 'fdgFDGO':
            raise NotImplementedError("input type '%s' not supported" % dtype)
            # 如果 dtype 的类型不在 'fdgFDGO' 中，则引发 NotImplementedError
        
        b = np.array(b, dtype=dtype)  # 使用指定的 dtype 将 b 转换为 NumPy 数组
        a = np.asarray(a, dtype=dtype)  # 使用指定的 dtype 将 a 转换为 NumPy 数组
        
        b /= a[0]  # 将 b 数组除以 a 的第一个元素
        x = np.asarray(x, dtype=dtype)  # 使用指定的 dtype 将 x 转换为 NumPy 数组
        
        # 在 axis 轴上应用函数，使用 b 对 x 进行卷积
        out_full = np.apply_along_axis(lambda y: np.convolve(b, y), axis, x)
        
        ind = out_full.ndim * [slice(None)]
        
        if zi is not None:
            ind[axis] = slice(zi.shape[axis])
            out_full[tuple(ind)] += zi
        
        ind[axis] = slice(out_full.shape[axis] - len(b) + 1)
        out = out_full[tuple(ind)]  # 截取 out_full 的部分作为输出
        
        if zi is None:
            return out  # 如果 zi 为 None，则直接返回 out
        else:
            ind[axis] = slice(out_full.shape[axis] - len(b) + 1, None)
            zf = out_full[tuple(ind)]  # 截取 out_full 的另一部分作为 zf
            return out, zf  # 返回 out 和 zf
    else:
        if zi is None:
            return _sigtools._linear_filter(b, a, x, axis)
            # 如果 a 的长度不为 1，且 zi 为 None，则调用 _sigtools._linear_filter 函数
        else:
            return _sigtools._linear_filter(b, a, x, axis, zi)
            # 如果 a 的长度不为 1，且 zi 不为 None，则调用 _sigtools._linear_filter 函数
def lfiltic(b, a, y, x=None):
    """
    Construct initial conditions for lfilter given input and output vectors.

    Given a linear filter (b, a) and initial conditions on the output `y`
    and the input `x`, return the initial conditions on the state vector zi
    which is used by `lfilter` to generate the output given the input.

    Parameters
    ----------
    b : array_like
        Linear filter term.
    a : array_like
        Linear filter term.
    y : array_like
        Initial conditions.

        If ``N = len(a) - 1``, then ``y = {y[-1], y[-2], ..., y[-N]}``.

        If `y` is too short, it is padded with zeros.
    x : array_like, optional
        Initial conditions.

        If ``M = len(b) - 1``, then ``x = {x[-1], x[-2], ..., x[-M]}``.

        If `x` is not given, its initial conditions are assumed zero.

        If `x` is too short, it is padded with zeros.

    Returns
    -------
    zi : ndarray
        The state vector ``zi = {z_0[-1], z_1[-1], ..., z_K-1[-1]}``,
        where ``K = max(M, N)``.

    See Also
    --------
    lfilter, lfilter_zi

    """
    # 计算系数向量 a 和 b 的长度
    N = np.size(a) - 1
    M = np.size(b) - 1
    # 确定状态向量 zi 的长度为 M 和 N 中的较大值
    K = max(M, N)
    # 将 y 转换为 ndarray 类型
    y = np.asarray(y)

    # 如果 x 未提供，则根据 b 和 a 的数据类型创建零填充的初始条件
    if x is None:
        result_type = np.result_type(np.asarray(b), np.asarray(a), y)
        if result_type.kind in 'bui':
            result_type = np.float64
        x = np.zeros(M, dtype=result_type)
    else:
        # 如果 x 已提供，则根据 b、a、y 和 x 的数据类型创建零填充的初始条件
        x = np.asarray(x)
        result_type = np.result_type(np.asarray(b), np.asarray(a), y, x)
        if result_type.kind in 'bui':
            result_type = np.float64
        x = x.astype(result_type)

        # 如果 x 的长度小于 M，则用零填充它
        L = np.size(x)
        if L < M:
            x = np.r_[x, np.zeros(M - L)]

    # 将 y 转换为与结果类型相同的类型
    y = y.astype(result_type)
    # 创建长度为 K 的零填充状态向量 zi
    zi = np.zeros(K, result_type)

    # 如果 y 的长度小于 N，则用零填充它
    L = np.size(y)
    if L < N:
        y = np.r_[y, np.zeros(N - L)]

    # 计算状态向量 zi 的初始值
    for m in range(M):
        zi[m] = np.sum(b[m + 1:] * x[:M - m], axis=0)

    for m in range(N):
        zi[m] -= np.sum(a[m + 1:] * y[:N - m], axis=0)

    # 返回计算得到的状态向量 zi
    return zi
    >>> recorded = signal.convolve(impulse_response, original)
    # 使用信号处理库中的convolve函数，将脉冲响应和原始信号进行卷积运算，得到recorded

    >>> recorded
    array([0, 2, 1, 0, 2, 3, 1, 0, 0])
    # 打印输出变量recorded，显示其值为一个NumPy数组

    >>> recovered, remainder = signal.deconvolve(recorded, impulse_response)
    # 使用信号处理库中的deconvolve函数，对recorded进行反卷积操作，使用脉冲响应impulse_response，
    # 返回反卷积后的结果recovered和余数remainder

    >>> recovered
    array([ 0.,  1.,  0.,  0.,  1.,  1.,  0.,  0.])
    # 打印输出变量recovered，显示其值为一个NumPy数组

    """
    num = np.atleast_1d(signal)
    # 将输入的signal转换为至少是1维的NumPy数组num

    den = np.atleast_1d(divisor)
    # 将输入的divisor转换为至少是1维的NumPy数组den

    if num.ndim > 1:
        raise ValueError("signal must be 1-D.")
    # 如果数组num的维度大于1，则抛出数值错误，信号必须是1维的。

    if den.ndim > 1:
        raise ValueError("divisor must be 1-D.")
    # 如果数组den的维度大于1，则抛出数值错误，除数必须是1维的。

    N = len(num)
    # 计算num数组的长度N

    D = len(den)
    # 计算den数组的长度D

    if D > N:
        quot = []
        rem = num
    else:
        input = np.zeros(N - D + 1, float)
        input[0] = 1
        quot = lfilter(num, den, input)
        rem = num - convolve(den, quot, mode='full')
    # 如果D大于N，则设置quot为空列表，余数rem为num；
    # 否则，创建一个长度为N-D+1的全零数组input，将第一个元素设为1，
    # 使用lfilter函数对num和den进行滤波操作得到quot，
    # rem为num与使用convolve函数对den和quot进行卷积（完整模式）后的差值。

    return quot, rem
    # 返回quot和rem作为结果
# 将输入信号转换为NumPy数组，以确保可以进行后续的数学运算和处理
x = np.asarray(x)
    # 检查输入数组 x 是否包含复数，如果是则抛出 ValueError 异常
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    
    # 如果 N 为 None，则将 N 设置为数组 x 在指定轴上的长度
    if N is None:
        N = x.shape[axis]
    
    # 检查 N 是否为正数，若不是则抛出 ValueError 异常
    if N <= 0:
        raise ValueError("N must be positive.")
    
    # 对输入数组 x 进行快速傅里叶变换，得到频域表示 Xf
    Xf = sp_fft.fft(x, N, axis=axis)
    
    # 创建一个与 Xf 相同类型的零数组 h
    h = np.zeros(N, dtype=Xf.dtype)
    
    # 根据 N 的奇偶性设置 h 的值
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    
    # 如果输入数组 x 的维度大于 1，则调整 h 的维度以匹配 x
    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim  # 创建一个与 x 维度相同的索引列表
        ind[axis] = slice(None)      # 在指定轴上使用 slice(None) 表示所有元素
        h = h[tuple(ind)]            # 根据索引列表调整 h 的维度
    
    # 对频域表示 Xf 应用带通滤波器 h，然后进行逆傅里叶变换，得到滤波后的数组 x
    x = sp_fft.ifft(Xf * h, axis=axis)
    
    # 返回滤波后的数组 x
    return x
def hilbert2(x, N=None):
    """
    Compute the '2-D' analytic signal of `x`

    Parameters
    ----------
    x : array_like
        2-D signal data.
    N : int or tuple of two ints, optional
        Number of Fourier components. Default is ``x.shape``

    Returns
    -------
    xa : ndarray
        Analytic signal of `x` taken along axes (0,1).

    References
    ----------
    .. [1] Wikipedia, "Analytic signal",
        https://en.wikipedia.org/wiki/Analytic_signal

    """
    # 将输入数据至少视为二维数组
    x = np.atleast_2d(x)
    # 如果输入数据维度大于2，则引发错误
    if x.ndim > 2:
        raise ValueError("x must be 2-D.")
    # 如果输入数据是复数类型，则引发错误，要求输入为实数
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    # 如果未指定 N，则默认为 x 的形状
    if N is None:
        N = x.shape
    # 如果 N 是整数，则将其转换为 (N, N) 形式的元组
    elif isinstance(N, int):
        if N <= 0:
            raise ValueError("N must be positive.")
        N = (N, N)
    # 如果 N 是长度为2的元组，并且元组中的每个值都大于0
    elif len(N) != 2 or np.any(np.asarray(N) <= 0):
        raise ValueError("When given as a tuple, N must hold exactly "
                         "two positive integers")

    # 对输入数据进行二维傅里叶变换
    Xf = sp_fft.fft2(x, N, axes=(0, 1))
    # 初始化两个全零数组，用于构造二维 Hilbert 滤波器
    h1 = np.zeros(N[0], dtype=Xf.dtype)
    h2 = np.zeros(N[1], dtype=Xf.dtype)
    # 为每个滤波器数组 h1 和 h2 赋值，使其成为 Hilbert 滤波器
    for h in (h1, h2):
        N1 = h.shape[0]
        if N1 % 2 == 0:
            h[0] = h[N1 // 2] = 1
            h[1:N1 // 2] = 2
        else:
            h[0] = 1
            h[1:(N1 + 1) // 2] = 2

    # 构造二维 Hilbert 滤波器
    h = h1[:, np.newaxis] * h2[np.newaxis, :]
    k = x.ndim
    # 将 Hilbert 滤波器 h 扩展到与输入数据 x 相同的维度
    while k > 2:
        h = h[:, np.newaxis]
        k -= 1
    # 对变换后的频域数据进行逆傅里叶变换，得到解析信号
    x = sp_fft.ifft2(Xf * h, axes=(0, 1))
    return x


def _cmplx_sort(p):
    """Sort roots based on magnitude.

    Parameters
    ----------
    p : array_like
        The roots to sort, as a 1-D array.

    Returns
    -------
    p_sorted : ndarray
        Sorted roots.
    indx : ndarray
        Array of indices needed to sort the input `p`.

    Examples
    --------
    >>> from scipy import signal
    >>> vals = [1, 4, 1+1.j, 3]
    >>> p_sorted, indx = signal.cmplx_sort(vals)
    >>> p_sorted
    array([1.+0.j, 1.+1.j, 3.+0.j, 4.+0.j])
    >>> indx
    array([0, 2, 3, 1])
    """
    # 将输入的根据大小排序，并返回排序后的根及其对应的索引
    p = np.asarray(p)
    indx = np.argsort(abs(p))
    return np.take(p, indx, 0), indx


def unique_roots(p, tol=1e-3, rtype='min'):
    """Determine unique roots and their multiplicities from a list of roots.

    Parameters
    ----------
    p : array_like
        The list of roots.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. Refer to Notes about
        the details on roots grouping.
    # 定义函数的可选参数 `rtype`，表示如何确定多个根在 `tol` 范围内时返回的根
    # 支持的选项包括：'max'、'maximum'、'min'、'minimum'、'avg'、'mean'
    # - 'max', 'maximum': 返回这些根中的最大值
    # - 'min', 'minimum': 返回这些根中的最小值
    # - 'avg', 'mean': 返回这些根的平均值
    # 复数根的最小或最大值比较时，先比较实部，再比较虚部

    if rtype in ['max', 'maximum']:
        # 如果 `rtype` 是 'max' 或 'maximum'，则选择最大值函数
        reduce = np.max
    elif rtype in ['min', 'minimum']:
        # 如果 `rtype` 是 'min' 或 'minimum'，则选择最小值函数
        reduce = np.min
    elif rtype in ['avg', 'mean']:
        # 如果 `rtype` 是 'avg' 或 'mean'，则选择平均值函数
        reduce = np.mean
    else:
        # 如果 `rtype` 不是预定义的值，则抛出 ValueError 异常
        raise ValueError("`rtype` must be one of "
                         "{'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}")

    # 将输入数组 `p` 转换为 numpy 数组
    p = np.asarray(p)

    # 创建一个二维数组 `points`，用来存储 `p` 中各复数根的实部和虚部
    points = np.empty((len(p), 2))
    points[:, 0] = np.real(p)  # 将实部填充到数组的第一列
    points[:, 1] = np.imag(p)  # 将虚部填充到数组的第二列

    # 基于 `points` 构建一个 cKDTree，用于快速查询附近的根
    tree = cKDTree(points)

    # 存储独特根和它们的重复次数的列表
    p_unique = []
    p_multiplicity = []

    # 创建一个布尔数组 `used`，用于跟踪已处理的根
    used = np.zeros(len(p), dtype=bool)

    # 遍历输入数组 `p` 中的每个根
    for i in range(len(p)):
        # 如果根已经被处理过，则继续下一个根
        if used[i]:
            continue

        # 查找与当前根 `points[i]` 距离在 `tol` 范围内的所有根的索引
        group = tree.query_ball_point(points[i], tol)
        # 筛选出尚未处理的根索引
        group = [x for x in group if not used[x]]

        # 将选定组中根的减少（最大、最小或平均）添加到 `p_unique`
        p_unique.append(reduce(p[group]))
        # 记录当前组中根的数量添加到 `p_multiplicity`
        p_multiplicity.append(len(group))

        # 标记已使用的根
        used[group] = True

    # 返回独特根和它们的重复次数的 numpy 数组
    return np.asarray(p_unique), np.asarray(p_multiplicity)
# 计算分数部分展开的分子多项式 b(s) 和分母多项式 a(s)
def invres(r, p, k, tol=1e-3, rtype='avg'):
    """Compute b(s) and a(s) from partial fraction expansion.

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

              b(s)     b[0] s**(M) + b[1] s**(M-1) + ... + b[M]
      H(s) = ------ = ------------------------------------------
              a(s)     a[0] s**(N) + a[1] s**(N-1) + ... + a[N]

    then the partial-fraction expansion H(s) is defined as::

               r[0]       r[1]             r[-1]
           = -------- + -------- + ... + --------- + k(s)
             (s-p[0])   (s-p[1])         (s-p[-1])

    If there are any repeated roots (closer together than `tol`), then H(s)
    has terms like::

          r[i]      r[i+1]              r[i+n-1]
        -------- + ----------- + ... + -----------
        (s-p[i])  (s-p[i])**2          (s-p[i])**n

    This function is used for polynomials in positive powers of s or z,
    such as analog filters or digital filters in controls engineering.  For
    negative powers of z (typical for digital filters in DSP), use `invresz`.

    Parameters
    ----------
    r : array_like
        Residues corresponding to the poles. For repeated poles, the residues
        must be ordered to correspond to ascending by power fractions.
    p : array_like
        Poles. Equal poles must be adjacent.
    k : array_like
        Coefficients of the direct polynomial term.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    See Also
    --------
    residue, invresz, unique_roots

    """
    # 将输入参数 r, p, k 转换为至少是一维的 numpy 数组
    r = np.atleast_1d(r)
    p = np.atleast_1d(p)
    k = np.trim_zeros(np.atleast_1d(k), 'f')

    # 对极点进行分组，获取唯一极点和它们的重数
    unique_poles, multiplicity = _group_poles(p, tol, rtype)

    # 计算每个唯一极点的因子多项式及总的分母多项式
    factors, denominator = _compute_factors(unique_poles, multiplicity,
                                            include_powers=True)

    # 如果直接项系数 k 的长度为 0，则分子多项式为常数 0
    if len(k) == 0:
        numerator = 0
    else:
        # 否则，分子多项式为 k 与分母多项式的乘积
        numerator = np.polymul(k, denominator)

    # 对于每一个残留和对应的因子，将其加到分子多项式中
    for residue, factor in zip(r, factors):
        numerator = np.polyadd(numerator, residue * factor)

    # 返回分子和分母多项式
    return numerator, denominator


def _compute_factors(roots, multiplicity, include_powers=False):
    """Compute the total polynomial divided by factors for each root."""
    current = np.array([1])
    suffixes = [current]
    # 对每个极点及其重数计算因子多项式
    for pole, mult in zip(roots[-1:0:-1], multiplicity[-1:0:-1]):
        monomial = np.array([1, -pole])
        # 多次乘以单项式构成每个因子
        for _ in range(mult):
            current = np.polymul(current, monomial)
        suffixes.append(current)
    # 返回所有因子的列表及总的分母多项式
    suffixes = suffixes[::-1]
    return suffixes, current
    # 初始化一个空列表，用于存放多项式的因子
    factors = []
    # 初始化当前多项式为 [1]
    current = np.array([1])

    # 遍历给定的根、重数和后缀列表，分别为 pole、mult 和 suffix
    for pole, mult, suffix in zip(roots, multiplicity, suffixes):
        # 创建一个单项式数组 [1, -pole]
        monomial = np.array([1, -pole])
        # 初始化一个空列表，用于存放当前多项式乘以后缀的结果
        block = []

        # 对于每个重数 mult，循环多次（最多 mult 次）
        for i in range(mult):
            # 如果是第一次循环或者 include_powers 为 True，则将当前多项式与后缀相乘并加入 block 列表
            if i == 0 or include_powers:
                block.append(np.polymul(current, suffix))
            # 更新当前多项式为当前多项式与单项式的乘积
            current = np.polymul(current, monomial)

        # 将 block 中的结果反转并加入到 factors 列表中
        factors.extend(reversed(block))

    # 返回因子列表 factors 和最终的当前多项式 current
    return factors, current
def _compute_residues(poles, multiplicity, numerator):
    # 计算分母的因子和未使用的结果
    denominator_factors, _ = _compute_factors(poles, multiplicity)
    # 将分子转换为与极点相同的数据类型
    numerator = numerator.astype(poles.dtype)

    residues = []
    for pole, mult, factor in zip(poles, multiplicity,
                                  denominator_factors):
        if mult == 1:
            # 对于单重极点，计算分数残留
            residues.append(np.polyval(numerator, pole) /
                            np.polyval(factor, pole))
        else:
            numer = numerator.copy()
            monomial = np.array([1, -pole])
            factor, d = np.polydiv(factor, monomial)

            block = []
            for _ in range(mult):
                numer, n = np.polydiv(numer, monomial)
                r = n[0] / d[0]
                numer = np.polysub(numer, r * factor)
                block.append(r)

            # 将结果逆序添加到残留列表中
            residues.extend(reversed(block))

    return np.asarray(residues)


def residue(b, a, tol=1e-3, rtype='avg'):
    """Compute partial-fraction expansion of b(s) / a(s).

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

              b(s)     b[0] s**(M) + b[1] s**(M-1) + ... + b[M]
      H(s) = ------ = ------------------------------------------
              a(s)     a[0] s**(N) + a[1] s**(N-1) + ... + a[N]

    then the partial-fraction expansion H(s) is defined as::

               r[0]       r[1]             r[-1]
           = -------- + -------- + ... + --------- + k(s)
             (s-p[0])   (s-p[1])         (s-p[-1])

    If there are any repeated roots (closer together than `tol`), then H(s)
    has terms like::

          r[i]      r[i+1]              r[i+n-1]
        -------- + ----------- + ... + -----------
        (s-p[i])  (s-p[i])**2          (s-p[i])**n

    This function is used for polynomials in positive powers of s or z,
    such as analog filters or digital filters in controls engineering.  For
    negative powers of z (typical for digital filters in DSP), use `residuez`.

    See Notes for details about the algorithm.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    r : ndarray
        Residues corresponding to the poles. For repeated poles, the residues
        are ordered to correspond to ascending by power fractions.
    p : ndarray
        Poles ordered by magnitude in ascending order.
    k : ndarray
        Coefficients of the direct polynomial term.

    See Also
    --------
    invres, residuez, numpy.poly, unique_roots

    Notes
    -----
    This function uses `_compute_residues` internally to compute residues for
    each pole based on their multiplicity and numerator polynomial.
    """
    # 获取分子多项式的阶数和分母多项式的阶数
    M = len(b) - 1
    N = len(a) - 1

    # 计算分子多项式的阶数和分母多项式的阶数
    poles, residues, k = unique_roots(b, a, tol=tol, rtype=rtype)

    # 使用 `_compute_residues` 函数计算残留值
    residues = _compute_residues(poles, residues, b)

    return residues, poles, k
    """
    The "deflation through subtraction" algorithm is used for
    computations --- method 6 in [1]_.

    The form of partial fraction expansion depends on poles multiplicity in
    the exact mathematical sense. However there is no way to exactly
    determine multiplicity of roots of a polynomial in numerical computing.
    Thus you should think of the result of `residue` with given `tol` as
    partial fraction expansion computed for the denominator composed of the
    computed poles with empirically determined multiplicity. The choice of
    `tol` can drastically change the result if there are close poles.

    References
    ----------
    .. [1] J. F. Mahoney, B. D. Sivazlian, "Partial fractions expansion: a
           review of computational methodology and efficiency", Journal of
           Computational and Applied Mathematics, Vol. 9, 1983.
    """
    # 将输入向量 b 和 a 转换为 NumPy 数组
    b = np.asarray(b)
    a = np.asarray(a)
    
    # 如果 b 或者 a 中包含复数，则将它们转换为复数类型，否则转换为浮点类型
    if (np.issubdtype(b.dtype, np.complexfloating)
            or np.issubdtype(a.dtype, np.complexfloating)):
        b = b.astype(complex)
        a = a.astype(complex)
    else:
        b = b.astype(float)
        a = a.astype(float)
    
    # 移除 b 和 a 中末尾的零元素，确保它们是一维数组
    b = np.trim_zeros(np.atleast_1d(b), 'f')
    a = np.trim_zeros(np.atleast_1d(a), 'f')
    
    # 如果分母数组 a 的大小为零，则抛出 ValueError 异常
    if a.size == 0:
        raise ValueError("Denominator `a` is zero.")
    
    # 计算分母多项式 a 的根（极点）
    poles = np.roots(a)
    
    # 如果分子数组 b 的大小为零，则返回适当形状的零数组和排序后的极点
    if b.size == 0:
        return np.zeros(poles.shape), _cmplx_sort(poles)[0], np.array([])
    
    # 如果分子 b 的长度小于分母 a 的长度，则直接将商 k 设置为空数组
    if len(b) < len(a):
        k = np.empty(0)
    else:
        # 否则，使用多项式除法计算商 k 和余数 b
        k, b = np.polydiv(b, a)
    
    # 找到唯一的极点和它们的重数，基于给定的容差 tol 和类型 rtype
    unique_poles, multiplicity = unique_roots(poles, tol=tol, rtype=rtype)
    # 对唯一极点进行排序，并获取排序后的索引顺序
    unique_poles, order = _cmplx_sort(unique_poles)
    multiplicity = multiplicity[order]
    
    # 计算每个唯一极点对应的残差（根据其重数和分子多项式 b）
    residues = _compute_residues(unique_poles, multiplicity, b)
    
    # 将计算出的极点按照它们的重数插入到原始极点数组中
    index = 0
    for pole, mult in zip(unique_poles, multiplicity):
        poles[index:index + mult] = pole
        index += mult
    
    # 返回最终的结果，包括残差除以分母多项式的首项、排序后的极点和商 k
    return residues / a[0], poles, k
    # 定义函数 residuez，用于计算 b(z) / a(z) 的部分分式展开
    """Compute partial-fraction expansion of b(z) / a(z).

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

                b(z)     b[0] + b[1] z**(-1) + ... + b[M] z**(-M)
        H(z) = ------ = ------------------------------------------
                a(z)     a[0] + a[1] z**(-1) + ... + a[N] z**(-N)

    then the partial-fraction expansion H(z) is defined as::

                 r[0]                   r[-1]
         = --------------- + ... + ---------------- + k[0] + k[1]z**(-1) ...
           (1-p[0]z**(-1))         (1-p[-1]z**(-1))

    If there are any repeated roots (closer than `tol`), then the partial
    fraction expansion has terms like::

             r[i]              r[i+1]                    r[i+n-1]
        -------------- + ------------------ + ... + ------------------
        (1-p[i]z**(-1))  (1-p[i]z**(-1))**2         (1-p[i]z**(-1))**n

    This function is used for polynomials in negative powers of z,
    such as digital filters in DSP.  For positive powers, use `residue`.

    See Notes of `residue` for details about the algorithm.
    """

    # 将输入的 b 和 a 转换为 numpy 数组
    b = np.asarray(b)
    a = np.asarray(a)

    # 如果 b 或者 a 中有复数类型的元素，则将它们转换为复数类型；否则转换为浮点数类型
    if (np.issubdtype(b.dtype, np.complexfloating)
            or np.issubdtype(a.dtype, np.complexfloating)):
        b = b.astype(complex)
        a = a.astype(complex)
    else:
        b = b.astype(float)
        a = a.astype(float)

    # 去除 b 和 a 数组末尾的零元素
    b = np.trim_zeros(np.atleast_1d(b), 'b')
    a = np.trim_zeros(np.atleast_1d(a), 'b')

    # 如果分母数组 a 为空，则抛出 ValueError 异常
    if a.size == 0:
        raise ValueError("Denominator `a` is zero.")
    # 如果分母数组 a 的第一个系数为零，则抛出 ValueError 异常
    elif a[0] == 0:
        raise ValueError("First coefficient of denominator `a` must be non-zero.")

    # 计算分母多项式 a 的根（极点）
    poles = np.roots(a)

    # 如果分子数组 b 为空，则返回与极点形状相同的零数组及排好序的极点和空数组
    if b.size == 0:
        return np.zeros(poles.shape), _cmplx_sort(poles)[0], np.array([])

    # 反转 b 和 a 数组，以便进行多项式除法
    b_rev = b[::-1]
    a_rev = a[::-1]

    # 如果反转后的 b 数组长度小于反转后的 a 数组长度，则直接返回空数组作为直接项系数
    if len(b_rev) < len(a_rev):
        k_rev = np.empty(0)
    else:
        # 否则进行多项式除法，得到直接项的系数 k_rev 和余数 b_rev
        k_rev, b_rev = np.polydiv(b_rev, a_rev)
    # 调用函数 unique_roots 处理 poles，并返回唯一的极点和它们的重数
    unique_poles, multiplicity = unique_roots(poles, tol=tol, rtype=rtype)
    
    # 调用函数 _cmplx_sort 对唯一的极点进行排序，并返回排序后的极点和排序后的索引顺序
    unique_poles, order = _cmplx_sort(unique_poles)
    
    # 根据排序后的索引顺序重新排列重数数组
    multiplicity = multiplicity[order]

    # 计算残留
    residues = _compute_residues(1 / unique_poles, multiplicity, b_rev)

    # 初始化索引为 0，为后续的极点和幂次计算做准备
    index = 0
    
    # 创建一个空数组，用于存储幂次
    powers = np.empty(len(residues), dtype=int)
    
    # 遍历唯一的极点和它们的重数，将极点和对应的幂次存入 poles 和 powers 数组
    for pole, mult in zip(unique_poles, multiplicity):
        poles[index:index + mult] = pole
        powers[index:index + mult] = 1 + np.arange(mult)
        index += mult

    # 计算最终的残留值，使用极点、幂次和反向系数 a_rev[0]
    residues *= (-poles) ** powers / a_rev[0]

    # 返回计算得到的残留、极点和反向系数的逆序
    return residues, poles, k_rev[::-1]
def _group_poles(poles, tol, rtype):
    # 根据 rtype 的值选择合适的归约函数
    if rtype in ['max', 'maximum']:
        reduce = np.max
    elif rtype in ['min', 'minimum']:
        reduce = np.min
    elif rtype in ['avg', 'mean']:
        reduce = np.mean
    else:
        # 如果 rtype 不在预定的范围内，抛出数值错误异常
        raise ValueError("`rtype` must be one of "
                         "{'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}")

    # 初始化存储唯一根和多重度的列表
    unique = []
    multiplicity = []

    # 初始化处理的第一个极点
    pole = poles[0]
    block = [pole]

    # 遍历所有极点，根据公差 tol 分组
    for i in range(1, len(poles)):
        if abs(poles[i] - pole) <= tol:
            # 如果当前极点与前一个极点之差小于等于公差 tol，则属于同一组
            block.append(pole)
        else:
            # 否则，计算当前组的归约值并存储
            unique.append(reduce(block))
            multiplicity.append(len(block))
            pole = poles[i]
            block = [pole]

    # 存储最后一组的归约值和多重度
    unique.append(reduce(block))
    multiplicity.append(len(block))

    # 将结果转换为 NumPy 数组并返回
    return np.asarray(unique), np.asarray(multiplicity)


def invresz(r, p, k, tol=1e-3, rtype='avg'):
    """Compute b(z) and a(z) from partial fraction expansion.

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

                b(z)     b[0] + b[1] z**(-1) + ... + b[M] z**(-M)
        H(z) = ------ = ------------------------------------------
                a(z)     a[0] + a[1] z**(-1) + ... + a[N] z**(-N)

    then the partial-fraction expansion H(z) is defined as::

                 r[0]                   r[-1]
         = --------------- + ... + ---------------- + k[0] + k[1]z**(-1) ...
           (1-p[0]z**(-1))         (1-p[-1]z**(-1))

    If there are any repeated roots (closer than `tol`), then the partial
    fraction expansion has terms like::

             r[i]              r[i+1]                    r[i+n-1]
        -------------- + ------------------ + ... + ------------------
        (1-p[i]z**(-1))  (1-p[i]z**(-1))**2         (1-p[i]z**(-1))**n

    This function is used for polynomials in negative powers of z,
    such as digital filters in DSP.  For positive powers, use `invres`.

    Parameters
    ----------
    r : array_like
        Residues corresponding to the poles. For repeated poles, the residues
        must be ordered to correspond to ascending by power fractions.
    p : array_like
        Poles. Equal poles must be adjacent.
    k : array_like
        Coefficients of the direct polynomial term.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    See Also
    --------
    residuez, unique_roots, invres

    """
    r = np.atleast_1d(r)  # 确保 r 是至少一维的 NumPy 数组
    p = np.atleast_1d(p)  # 确保 p 是至少一维的 NumPy 数组
    k = np.trim_zeros(np.atleast_1d(k), 'b')  # 移除 k 数组末尾的零元素
    # 使用 _group_poles 函数对极点 p 进行分组，返回唯一的极点和它们的重数
    unique_poles, multiplicity = _group_poles(p, tol, rtype)
    
    # 使用 _compute_factors 函数计算分子的因子和分母
    factors, denominator = _compute_factors(unique_poles, multiplicity,
                                            include_powers=True)
    
    # 如果 k 的长度为 0，则分子为 0
    if len(k) == 0:
        numerator = 0
    else:
        # 否则，将 k 和 denominator 的反转多项式相乘，得到分子
        numerator = np.polymul(k[::-1], denominator[::-1])

    # 遍历 r 和 factors 的组合，计算分子的每一项
    for residue, factor in zip(r, factors):
        numerator = np.polyadd(numerator, residue * factor[::-1])

    # 返回完整计算结果的分子和分母，分别反转顺序
    return numerator[::-1], denominator
def resample(x, num, t=None, axis=0, window=None, domain='time'):
    """
    Resample `x` to `num` samples using Fourier method along the given axis.

    The resampled signal starts at the same value as `x` but is sampled
    with a spacing of ``len(x) / num * (spacing of x)``.  Because a
    Fourier method is used, the signal is assumed to be periodic.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    num : int
        The number of samples in the resampled signal.
    t : array_like, optional
        If `t` is given, it is assumed to be the equally spaced sample
        positions associated with the signal data in `x`.
    axis : int, optional
        The axis of `x` that is resampled.  Default is 0.
    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.  See below for details.
    domain : string, optional
        A string indicating the domain of the input `x`:
        ``time`` Consider the input `x` as time-domain (Default),
        ``freq`` Consider the input `x` as frequency-domain.

    Returns
    -------
    resampled_x or (resampled_x, resampled_t)
        Either the resampled array, or, if `t` was given, a tuple
        containing the resampled array and the corresponding resampled
        positions.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample_poly : Resample using polyphase filtering and an FIR filter.

    Notes
    -----
    The argument `window` controls a Fourier-domain window that tapers
    the Fourier spectrum before zero-padding to alleviate ringing in
    the resampled values for sampled signals you didn't intend to be
    interpreted as band-limited.

    If `window` is a function, then it is called with a vector of inputs
    indicating the frequency bins (i.e. fftfreq(x.shape[axis]) ).

    If `window` is an array of the same length as `x.shape[axis]` it is
    assumed to be the window to be applied directly in the Fourier
    domain (with dc and low-frequency first).

    For any other type of `window`, the function `scipy.signal.get_window`
    is called to generate the window.

    The first sample of the returned vector is the same as the first
    sample of the input vector.  The spacing between samples is changed
    from ``dx`` to ``dx * len(x) / num``.

    If `t` is not None, then it is used solely to calculate the resampled
    positions `resampled_t`

    As noted, `resample` uses FFT transformations, which can be very
    slow if the number of input or output samples is large and prime;
    see `scipy.fft.fft`.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle:

    >>> import numpy as np
    >>> from scipy import signal

    >>> x = np.linspace(0, 10, 20, endpoint=False)
    >>> y = np.cos(-x**2/6.0)
    """
    if domain not in ('time', 'freq'):
        # 如果传入的域不是'time'或'freq'，则抛出数值错误异常
        raise ValueError("Acceptable domain flags are 'time' or"
                         f" 'freq', not domain={domain}")

    x = np.asarray(x)
    Nx = x.shape[axis]

    # Check if we can use faster real FFT
    real_input = np.isrealobj(x)

    if domain == 'time':
        # 如果域是时间域：
        if real_input:
            # 如果输入是实数对象，使用快速实数FFT
            X = sp_fft.rfft(x, axis=axis)
        else:
            # 否则使用完整复数FFT
            X = sp_fft.fft(x, axis=axis)
    else:  # domain == 'freq'
        # 如果域是频率域，直接将X赋值为输入x
        X = x

    # Apply window to spectrum
    # 如果有窗口函数要应用在频谱上：
    if window is not None:
        if callable(window):
            # 如果窗口是可调用对象，使用窗口函数对频率轴进行窗口化
            W = window(sp_fft.fftfreq(Nx))
        elif isinstance(window, np.ndarray):
            # 如果窗口是numpy数组，检查其长度是否与数据一致
            if window.shape != (Nx,):
                raise ValueError('window must have the same length as data')
            W = window
        else:
            # 否则，通过sp_fft.ifftshift和get_window函数获取窗口
            W = sp_fft.ifftshift(get_window(window, Nx))

        newshape_W = [1] * x.ndim
        newshape_W[axis] = X.shape[axis]
        if real_input:
            # 如果输入是实数，折叠窗口以模拟复数行为
            W_real = W.copy()
            W_real[1:] += W_real[-1:0:-1]
            W_real[1:] *= 0.5
            X *= W_real[:newshape_W[axis]].reshape(newshape_W)
        else:
            # 否则，直接将窗口乘以X
            X *= W.reshape(newshape_W)

    # Copy each half of the original spectrum to the output spectrum, either
    # truncating high frequencies (downsampling) or zero-padding them
    # (upsampling)

    # Placeholder array for output spectrum
    newshape = list(x.shape)
    if real_input:
        # 如果输入是实数，输出的spectrum的轴的长度为num // 2 + 1
        newshape[axis] = num // 2 + 1
    else:
        # 否则，输出的spectrum的轴的长度为num
        newshape[axis] = num
    Y = np.zeros(newshape, X.dtype)

    # Copy positive frequency components (and Nyquist, if present)
    # 复制正频率分量（和奈奎斯特频率，如果存在）
    N = min(num, Nx)
    nyq = N // 2 + 1  # Slice index that includes Nyquist if present
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]
    if not real_input:
        # 如果输入不是实数，复制负频率分量
        if N > 2:  # (slice expression doesn't collapse to empty array)
            sl[axis] = slice(nyq - N, None)
            Y[tuple(sl)] = X[tuple(sl)]

    # Split/join Nyquist component(s) if present
    # 如果存在奈奎斯特分量，则拆分/合并它们
    # 到此为止，我们设置了Y[+N/2]=X[+N/2]
    # 如果 N 是偶数
    if N % 2 == 0:
        # 如果 num 小于 Nx，表示下采样
        if num < Nx:  # downsampling
            # 如果是实数输入
            if real_input:
                # 在轴上创建一个切片，选取频率为 +N/2 的成分，乘以2
                sl[axis] = slice(N//2, N//2 + 1)
                Y[tuple(sl)] *= 2.
            else:
                # 选择频率为 +N/2 的 Y 成分，并加上频率为 -N/2 的 X 成分
                sl[axis] = slice(-N//2, -N//2 + 1)
                Y[tuple(sl)] += X[tuple(sl)]
        # 如果 Nx 小于 num，表示上采样
        elif Nx < num:  # upsampling
            # 选择频率为 +N/2 的成分并除以2
            sl[axis] = slice(N//2, N//2 + 1)
            Y[tuple(sl)] *= 0.5
            # 如果不是实数输入
            if not real_input:
                # 保存频率为 +N/2 的 Y 成分
                temp = Y[tuple(sl)]
                # 将频率为 -N/2 的 Y 成分设为频率为 +N/2 的 Y 成分值
                sl[axis] = slice(num-N//2, num-N//2 + 1)
                Y[tuple(sl)] = temp

    # 反向变换
    if real_input:
        # 使用逆实数快速傅里叶变换进行逆变换
        y = sp_fft.irfft(Y, num, axis=axis)
    else:
        # 使用逆快速傅里叶变换进行逆变换，同时覆盖输入数组 X
        y = sp_fft.ifft(Y, axis=axis, overwrite_x=True)

    # 调整反变换结果的幅度
    y *= (float(num) / float(Nx))

    # 如果没有给定时间数组 t，则返回反变换结果 y
    if t is None:
        return y
    else:
        # 根据采样数和时间步长创建新的时间数组 new_t
        new_t = np.arange(0, num) * (t[1] - t[0]) * Nx / float(num) + t[0]
        return y, new_t
# 使用多相滤波器对信号 `x` 沿指定轴进行重采样

def resample_poly(x, up, down, axis=0, window=('kaiser', 5.0),
                  padtype='constant', cval=None):
    """
    Resample `x` along the given axis using polyphase filtering.

    The signal `x` is upsampled by the factor `up`, a zero-phase low-pass
    FIR filter is applied, and then it is downsampled by the factor `down`.
    The resulting sample rate is ``up / down`` times the original sample
    rate. By default, values beyond the boundary of the signal are assumed
    to be zero during the filtering step.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    axis : int, optional
        The axis of `x` that is resampled. Default is 0.
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR filter
        coefficients to employ. See below for details.
    padtype : string, optional
        `constant`, `line`, `mean`, `median`, `maximum`, `minimum` or any of
        the other signal extension modes supported by `scipy.signal.upfirdn`.
        Changes assumptions on values beyond the boundary. If `constant`,
        assumed to be `cval` (default zero). If `line` assumed to continue a
        linear trend defined by the first and last points. `mean`, `median`,
        `maximum` and `minimum` work as in `np.pad` and assume that the values
        beyond the boundary are the mean, median, maximum or minimum
        respectively of the array along the axis.

        .. versionadded:: 1.4.0
    cval : float, optional
        Value to use if `padtype='constant'`. Default is zero.

        .. versionadded:: 1.4.0

    Returns
    -------
    resampled_x : array
        The resampled array.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample : Resample up or down using the FFT method.

    Notes
    -----
    This polyphase method will likely be faster than the Fourier method
    in `scipy.signal.resample` when the number of samples is large and
    prime, or when the number of samples is large and `up` and `down`
    share a large greatest common denominator. The length of the FIR
    filter used will depend on ``max(up, down) // gcd(up, down)``, and
    the number of operations during polyphase filtering will depend on
    the filter length and `down` (see `scipy.signal.upfirdn` for details).

    The argument `window` specifies the FIR low-pass filter design.

    If `window` is an array_like it is assumed to be the FIR filter
    coefficients. Note that the FIR filter is applied after the upsampling
    step, so it should be designed to operate on a signal at a sampling
    frequency higher than the original by a factor of `up//gcd(up, down)`.
    This function's output will be centered with respect to this array, so it
    """
    # 执行多相滤波器重采样过程

    # 导入必要的库函数
    from scipy.signal import upfirdn

    # 计算 `up` 和 `down` 的最大公约数
    from math import gcd

    # 确定滤波器系数 `h` 的获取方式
    if isinstance(window, str):
        # 若 `window` 是字符串，则根据字符串名创建滤波器系数
        h = get_window(window, up)
    elif isinstance(window, tuple):
        # 若 `window` 是元组，则直接使用给定的滤波器系数
        h = window[1]
    else:
        # 否则，假定 `window` 是一个数组，直接使用它作为滤波器系数
        h = window

    # 计算滤波器的长度
    filter_length = max(up, down) // gcd(up, down)

    # 在信号 `x` 的指定轴上进行多相滤波重采样
    resampled_x = upfirdn(h, x, up, down, axis=axis, padtype=padtype, cval=cval)

    # 返回重采样后的信号
    return resampled_x
    # 将输入参数 x 转换为 NumPy 数组
    x = np.asarray(x)
    
    # 检查 up 是否为整数，若不是则抛出数值错误异常
    if up != int(up):
        raise ValueError("up must be an integer")
    
    # 检查 down 是否为整数，若不是则抛出数值错误异常
    if down != int(down):
        raise ValueError("down must be an integer")
    
    # 将 up 和 down 强制转换为整数类型
    up = int(up)
    down = int(down)
    
    # 检查 up 和 down 是否均大于等于 1，若不是则抛出数值错误异常
    if up < 1 or down < 1:
        raise ValueError('up and down must be >= 1')
    
    # 如果 cval 不为 None 且 padtype 不是 'constant'，则抛出数值错误异常
    if cval is not None and padtype != 'constant':
        raise ValueError('cval has no effect when padtype is ', padtype)
    
    # 计算最大公约数以确定 up 和 down 的最简分数形式
    g_ = math.gcd(up, down)
    up //= g_
    down //= g_
    
    # 如果 up 和 down 均为 1，则直接返回输入数组的拷贝
    if up == down == 1:
        return x.copy()
    
    # 计算输入数组在指定轴上的长度
    n_in = x.shape[axis]
    
    # 计算输出数组的长度，根据 up 和 down 的比例关系计算
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)
    # 如果 window 是 list 或者 numpy 数组，则将其转换为 numpy 数组，以便进行修改
    if isinstance(window, (list, np.ndarray)):
        window = np.array(window)  # 使用 array 强制复制一份副本（因为我们会修改它）
        # 如果 window 的维度大于 1，则抛出数值错误异常
        if window.ndim > 1:
            raise ValueError('window must be 1-D')
        # 计算窗口的半长度
        half_len = (window.size - 1) // 2
        h = window
    else:
        # 设计一个线性相位的低通 FIR 滤波器
        max_rate = max(up, down)
        # FIR 滤波器的截止频率（相对于奈奎斯特频率）
        f_c = 1. / max_rate
        # 合理的截止长度，适用于 sinc 函数
        half_len = 10 * max_rate
        # 使用 firwin 函数生成 FIR 滤波器系数，使其类型与 x 的类型匹配
        h = firwin(2 * half_len + 1, f_c,
                   window=window).astype(x.dtype)
    # 将 h 乘以 up，以调整其频率响应
    h *= up

    # 将滤波器零填充，使输出样本位于中心
    n_pre_pad = (down - half_len % down)
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    # 在 _output_len 函数返回的输出长度不足以满足 n_out + n_pre_remove 的要求时，追加零填充
    while _output_len(len(h) + n_pre_pad + n_post_pad, n_in,
                      up, down) < n_out + n_pre_remove:
        n_post_pad += 1
    # 将 h 扩展为包含零填充的新数组
    h = np.concatenate((np.zeros(n_pre_pad, dtype=h.dtype), h,
                        np.zeros(n_post_pad, dtype=h.dtype)))
    # 计算需要移除的前置零填充的末尾位置
    n_pre_remove_end = n_pre_remove + n_out

    # 根据 padtype 选项移除背景
    funcs = {'mean': np.mean, 'median': np.median,
             'minimum': np.amin, 'maximum': np.amax}
    upfirdn_kwargs = {'mode': 'constant', 'cval': 0}
    if padtype in funcs:
        # 计算 x 中沿指定轴的背景值，方法由 padtype 指定
        background_values = funcs[padtype](x, axis=axis, keepdims=True)
    elif padtype in _upfirdn_modes:
        # 根据 padtype 设置 upfirdn 的参数
        upfirdn_kwargs = {'mode': padtype}
        if padtype == 'constant':
            if cval is None:
                cval = 0
            upfirdn_kwargs['cval'] = cval
    else:
        # 如果 padtype 不在预定义的模式列表中，则抛出数值错误异常
        raise ValueError(
            'padtype must be one of: maximum, mean, median, minimum, ' +
            ', '.join(_upfirdn_modes))

    # 如果 padtype 在 funcs 中，则从 x 中减去背景值
    if padtype in funcs:
        x = x - background_values

    # 对 x 应用滤波器 h，然后移除多余的部分
    y = upfirdn(h, x, up, down, axis=axis, **upfirdn_kwargs)
    # 创建一个保留有效数据的切片对象
    keep = [slice(None), ]*x.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)
    # 提取保留区域的数据
    y_keep = y[tuple(keep)]

    # 如果 padtype 在 funcs 中，则将背景值重新加回 y_keep 中
    if padtype in funcs:
        y_keep += background_values

    # 返回最终处理后的数据 y_keep
    return y_keep
def vectorstrength(events, period):
    '''
    Determine the vector strength of the events corresponding to the given
    period.

    The vector strength is a measure of phase synchrony, how well the
    timing of the events is synchronized to a single period of a periodic
    signal.

    If multiple periods are used, calculate the vector strength of each.
    This is called the "resonating vector strength".

    Parameters
    ----------
    events : 1D array_like
        An array of time points containing the timing of the events.
    period : float or array_like
        The period of the signal that the events should synchronize to.
        The period is in the same units as `events`.  It can also be an array
        of periods, in which case the outputs are arrays of the same length.

    Returns
    -------
    strength : float or 1D array
        The strength of the synchronization.  1.0 is perfect synchronization
        and 0.0 is no synchronization.  If `period` is an array, this is also
        an array with each element containing the vector strength at the
        corresponding period.
    phase : float or array
        The phase that the events are most strongly synchronized to in radians.
        If `period` is an array, this is also an array with each element
        containing the phase for the corresponding period.

    References
    ----------
    van Hemmen, JL, Longtin, A, and Vollmayr, AN. Testing resonating vector
        strength: Auditory system, electric fish, and noise.
        Chaos 21, 047508 (2011);
        :doi:`10.1063/1.3670512`.
    van Hemmen, JL.  Vector strength after Goldberg, Brown, and von Mises:
        biological and mathematical perspectives.  Biol Cybern.
        2013 Aug;107(4):385-96. :doi:`10.1007/s00422-013-0561-7`.
    van Hemmen, JL and Vollmayr, AN.  Resonating vector strength: what happens
        when we vary the "probing" frequency while keeping the spike times
        fixed.  Biol Cybern. 2013 Aug;107(4):491-94.
        :doi:`10.1007/s00422-013-0560-8`.
    '''
    events = np.asarray(events)  # 将事件转换为NumPy数组
    period = np.asarray(period)  # 将周期转换为NumPy数组

    if events.ndim > 1:  # 如果事件数组的维度大于1，抛出值错误
        raise ValueError('events cannot have dimensions more than 1')
    if period.ndim > 1:  # 如果周期数组的维度大于1，抛出值错误
        raise ValueError('period cannot have dimensions more than 1')

    scalarperiod = not period.ndim  # 记录period是否原本是标量

    events = np.atleast_2d(events)  # 确保events至少是2维的数组
    period = np.atleast_2d(period)  # 确保period至少是2维的数组

    if (period <= 0).any():  # 如果周期中有非正数值，抛出值错误
        raise ValueError('periods must be positive')

    vectors = np.exp(np.dot(2j*np.pi/period.T, events))  # 将事件转换为复数向量表示

    vectormean = np.mean(vectors, axis=1)  # 计算向量的均值
    strength = abs(vectormean)  # 计算向量均值的幅度，即同步强度
    phase = np.angle(vectormean)  # 计算向量均值的角度，即同步的相位

    if scalarperiod:  # 如果原始period是标量，返回标量结果
        return strength.item(), phase.item()
    else:  # 否则返回数组结果
        return strength, phase
    # 如果 scalarperiod 存在（即非空），则执行以下操作
    if scalarperiod:
        # 将 strength 变量限定为其第一个元素
        strength = strength[0]
        # 将 phase 变量限定为其第一个元素
        phase = phase[0]
    # 返回经过可能修改后的 strength 和 phase 变量
    return strength, phase
# 定义一个函数用于去除数据沿指定轴向的线性或常数趋势
def detrend(data: np.ndarray, axis: int = -1,
            type: Literal['linear', 'constant'] = 'linear',
            bp: ArrayLike | int = 0, overwrite_data: bool = False) -> np.ndarray:
    r"""Remove linear or constant trend along axis from data.

    Parameters
    ----------
    data : array_like
        输入数据。
    axis : int, optional
        要去除趋势的轴。默认为最后一个轴 (-1)。
    type : {'linear', 'constant'}, optional
        去趋势的类型。如果 ``type == 'linear'``（默认），
        则从 `data` 中减去线性最小二乘拟合的结果。
        如果 ``type == 'constant'``，则只减去 `data` 的均值。
    bp : array_like of ints, optional
        中断点序列。如果给定，则对 `data` 中每个中断点之间的部分执行单独的线性拟合。
        中断点被指定为 `data` 的索引。当 ``type == 'linear'`` 时，此参数有效。
    overwrite_data : bool, optional
        如果为 True，执行就地去趋势操作并避免复制。默认为 False。

    Returns
    -------
    ret : ndarray
        去趋势后的输入数据。

    Notes
    -----
    去趋势可以解释为减去一个最小二乘拟合的多项式：
    将参数 `type` 设置为 'constant' 相当于拟合零次多项式，
    'linear' 相当于拟合一次多项式。参考下面的示例。

    See Also
    --------
    numpy.polynomial.polynomial.Polynomial.fit: 创建最小二乘拟合多项式。

    Examples
    --------
    以下示例去趋势函数 :math:`x(t) = \sin(\pi t) + 1/4`：

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from scipy.signal import detrend
    ...
    >>> t = np.linspace(-0.5, 0.5, 21)
    >>> x = np.sin(np.pi*t) + 1/4
    ...
    >>> x_d_const = detrend(x, type='constant')
    >>> x_d_linear = detrend(x, type='linear')
    ...
    >>> fig1, ax1 = plt.subplots()
    >>> ax1.set_title(r"Detrending $x(t)=\sin(\pi t) + 1/4$")
    >>> ax1.set(xlabel="t", ylabel="$x(t)$", xlim=(t[0], t[-1]))
    >>> ax1.axhline(y=0, color='black', linewidth=.5)
    >>> ax1.axvline(x=0, color='black', linewidth=.5)
    >>> ax1.plot(t, x, 'C0.-',  label="No detrending")
    >>> ax1.plot(t, x_d_const, 'C1x-', label="type='constant'")
    >>> ax1.plot(t, x_d_linear, 'C2+-', label="type='linear'")
    >>> ax1.legend()
    >>> plt.show()

    另外，也可以使用 NumPy 的 `~numpy.polynomial.polynomial.Polynomial` 进行去趋势：

    >>> pp0 = np.polynomial.Polynomial.fit(t, x, deg=0)  # 拟合零次多项式
    >>> np.allclose(x_d_const, x - pp0(t))  # 与常数去趋势进行比较
    True
    >>> pp1 = np.polynomial.Polynomial.fit(t, x, deg=1)  # 拟合一次多项式
    >>> np.allclose(x_d_linear, x - pp1(t))  # 与线性去趋势进行比较
    True
    Note that `~numpy.polynomial.polynomial.Polynomial` also allows fitting higher
    degree polynomials. Consult its documentation on how to extract the polynomial
    coefficients.
    """
    # 如果 type 不是 ['linear', 'l', 'constant', 'c'] 中的一种，抛出数值错误异常
    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    # 将数据转换为 NumPy 数组
    data = np.asarray(data)
    # 获取数据类型的字符表示
    dtype = data.dtype.char
    # 如果数据类型不在 'dfDF' 中，则默认使用 'd' 类型
    if dtype not in 'dfDF':
        dtype = 'd'
    # 如果 type 是 ['constant', 'c'] 中的一种，则计算数据与均值的差值并返回
    if type in ['constant', 'c']:
        ret = data - np.mean(data, axis, keepdims=True)
        return ret
    else:
        # 获取数据的形状
        dshape = data.shape
        # 获取指定轴的长度
        N = dshape[axis]
        # 对断点进行排序并去重，确保都在数据长度 N 内
        bp = np.sort(np.unique(np.concatenate(np.atleast_1d(0, bp, N))))
        if np.any(bp > N):
            raise ValueError("Breakpoints must be less than length "
                             "of data along given axis.")

        # 将数据重组，使指定的轴成为第一维，其它维度被折叠为第二维
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdata = np.moveaxis(data, axis, 0)
        newdata_shape = newdata.shape
        newdata = newdata.reshape(N, -1)

        # 如果不覆盖原数据，确保有一份副本
        if not overwrite_data:
            newdata = newdata.copy()  # 确保我们有一个副本
        # 如果新数据的数据类型不在 'dfDF' 中，则转换为指定的数据类型
        if newdata.dtype.char not in 'dfDF':
            newdata = newdata.astype(dtype)
# Perform least squares fitting and remove trend for each segment defined by breakpoints
for m in range(len(bp) - 1):
    # Calculate the number of points in the current segment
    Npts = bp[m + 1] - bp[m]
    # Create a matrix A of shape (Npts, 2) with the first column as normalized time indices
    A = np.ones((Npts, 2), dtype)
    A[:, 0] = np.arange(1, Npts + 1, dtype=dtype) / Npts
    # Define the slice for the current segment in the data
    sl = slice(bp[m], bp[m + 1])
    # Perform least squares fit using matrix A and data in the current slice
    coef, resids, rank, s = linalg.lstsq(A, newdata[sl])
    # Remove the trend from the current segment of data
    newdata[sl] = newdata[sl] - A @ coef

# Reshape the modified data back to its original shape
newdata = newdata.reshape(newdata_shape)
# Move the axis of the data array to the specified position 'axis'
ret = np.moveaxis(newdata, 0, axis)
# Return the reshaped and axis-moved data
return ret
    array([ 0.5       ,  0.5       ,  0.5       ,  0.49836039,  0.48610528,
        0.44399389,  0.35505241])

    Note that the `zi` argument to `lfilter` was computed using
    `lfilter_zi` and scaled by ``x[0]``.  Then the output `y` has no
    transient until the input drops from 0.5 to 0.0.

    """

    # FIXME: Can this function be replaced with an appropriate
    # use of lfiltic?  For example, when b,a = butter(N,Wn),
    #    lfiltic(b, a, y=numpy.ones_like(a), x=numpy.ones_like(b)).
    #

    # We could use scipy.signal.normalize, but it uses warnings in
    # cases where a ValueError is more appropriate, and it allows
    # b to be 2D.

    # 将b数组至少转换为一维数组
    b = np.atleast_1d(b)
    if b.ndim != 1:
        raise ValueError("Numerator b must be 1-D.")

    # 将a数组至少转换为一维数组
    a = np.atleast_1d(a)
    if a.ndim != 1:
        raise ValueError("Denominator a must be 1-D.")

    # 去除a开头的零元素
    while len(a) > 1 and a[0] == 0.0:
        a = a[1:]
    if a.size < 1:
        raise ValueError("There must be at least one nonzero `a` coefficient.")

    # 如果a的首个系数不为1.0，则将b和a数组归一化
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]

    # 确定系数数组的最大长度
    n = max(len(a), len(b))

    # 使用零填充a或b数组，使它们具有相同的长度
    if len(a) < n:
        a = np.r_[a, np.zeros(n - len(a), dtype=a.dtype)]
    elif len(b) < n:
        b = np.r_[b, np.zeros(n - len(b), dtype=b.dtype)]

    # 构建方程 I - A.T * zi = B，解出zi
    IminusA = np.eye(n - 1, dtype=np.result_type(a, b)) - linalg.companion(a).T
    B = b[1:] - a[1:] * b[0]
    zi = np.linalg.solve(IminusA, B)

    # 供将来参考：还可以使用以下显式公式来解线性系统：
    #
    # zi = np.zeros(n - 1)
    # zi[0] = B.sum() / IminusA[:,0].sum()
    # asum = 1.0
    # csum = 0.0
    # for k in range(1,n-1):
    #     asum += a[k]
    #     csum += b[k] - a[k]*b[0]
    #     zi[k] = asum*zi[0] - csum

    return zi
# 构造用于 sosfilt 函数的初始条件，以实现步响应稳态。

# 计算一个与步响应稳态对应的初始状态 `zi`，用于 `sosfilt` 函数。

def sosfilt_zi(sos):
    """
    Construct initial conditions for sosfilt for step response steady-state.

    Compute an initial state `zi` for the `sosfilt` function that corresponds
    to the steady state of the step response.

    A typical use of this function is to set the initial state so that the
    output of the filter starts at the same value as the first element of
    the signal to be filtered.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    Returns
    -------
    zi : ndarray
        Initial conditions suitable for use with ``sosfilt``, shape
        ``(n_sections, 2)``.

    See Also
    --------
    sosfilt, zpk2sos

    Notes
    -----
    .. versionadded:: 0.16.0

    Examples
    --------
    Filter a rectangular pulse that begins at time 0, with and without
    the use of the `zi` argument of `scipy.signal.sosfilt`.

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> sos = signal.butter(9, 0.125, output='sos')
    >>> zi = signal.sosfilt_zi(sos)
    >>> x = (np.arange(250) < 100).astype(int)
    >>> f1 = signal.sosfilt(sos, x)
    >>> f2, zo = signal.sosfilt(sos, x, zi=zi)

    >>> plt.plot(x, 'k--', label='x')
    >>> plt.plot(f1, 'b', alpha=0.5, linewidth=2, label='filtered')
    >>> plt.plot(f2, 'g', alpha=0.25, linewidth=4, label='filtered with zi')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    # 将 sos 转换为 numpy 数组
    sos = np.asarray(sos)
    # 检查 sos 的维度和形状是否正确
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError('sos must be shape (n_sections, 6)')

    # 如果 sos 的数据类型是整数类型，则转换为浮点数类型
    if sos.dtype.kind in 'bui':
        sos = sos.astype(np.float64)

    # 获取 sos 的段数
    n_sections = sos.shape[0]
    # 创建一个空数组，用于存储初始条件 zi，其形状为 (n_sections, 2)
    zi = np.empty((n_sections, 2), dtype=sos.dtype)
    # 初始比例为 1.0
    scale = 1.0
    # 遍历每个段
    for section in range(n_sections):
        # 提取当前段的分子和分母系数
        b = sos[section, :3]
        a = sos[section, 3:]
        # 计算当前段的初始条件 zi
        zi[section] = scale * lfilter_zi(b, a)
        # 更新比例，scale *= b.sum() / a.sum() 是当前段在 omega=0 处的增益
        scale *= b.sum() / a.sum()

    # 返回计算得到的初始条件 zi
    return zi


def _filtfilt_gust(b, a, x, axis=-1, irlen=None):
    """Forward-backward IIR filter that uses Gustafsson's method.

    Apply the IIR filter defined by ``(b,a)`` to `x` twice, first forward
    then backward, using Gustafsson's initial conditions [1]_.

    Let ``y_fb`` be the result of filtering first forward and then backward,
    and let ``y_bf`` be the result of filtering first backward then forward.
    Gustafsson's method is to compute initial conditions for the forward
    pass and the backward pass such that ``y_fb == y_bf``.

    Parameters
    ----------
    b : scalar or 1-D ndarray
        Numerator coefficients of the filter.
    
    # 将输入参数 `a` 转换为至少是一维的 numpy 数组
    b = np.atleast_1d(b)
    # 将输入参数 `b` 转换为至少是一维的 numpy 数组
    a = np.atleast_1d(a)

    # 计算滤波器的阶数，即 `b` 和 `a` 中较大的长度减去 1
    order = max(len(b), len(a)) - 1
    
    # 如果滤波器阶数为 0，即只有标量乘法且没有状态
    if order == 0:
        # 计算标量乘法的比例
        scale = (b[0] / a[0])**2
        # 返回乘以比例的滤波后数据 `y`，以及空的初始条件数组 `x0` 和 `x1`
        y = scale * x
        return y, np.array([]), np.array([])

    # 如果指定了 `axis` 并且不是 `-1` 或 `x` 的最后一个维度
    if axis != -1 or axis != x.ndim - 1:
        # 将包含数据的轴移动到最后
        x = np.swapaxes(x, axis, x.ndim - 1)

    # `n` 是待滤波数据中的样本数
    n = x.shape[-1]

    # 如果 `irlen` 为 None 或者信号长度小于等于 `2 * irlen`
    if irlen is None or n <= 2 * irlen:
        m = n
    else:
        m = irlen

    # 创建观测矩阵 Obs（在论文中称为 O）
    Obs = np.zeros((m, order))
    zi = np.zeros(order)
    zi[0] = 1
    # 填充观测矩阵的第一列，使用 lfilter 函数进行前向滤波
    Obs[:, 0] = lfilter(b, a, np.zeros(m), zi=zi)[0]

    # 使用循环填充观测矩阵的其余列
    for k in range(1, order):
        Obs[k:, k] = Obs[:-k, 0]

    # 构建观测矩阵的逆序矩阵 Obsr
    Obsr = Obs[::-1]

    # 创建 S 矩阵，它将滤波应用于反向传播的初始条件
    S = lfilter(b, a, Obs[::-1], axis=0)

    # 构建 S 的逆序矩阵 Sr
    Sr = S[::-1]

    # 如果 m 等于 n，则构建 M 矩阵，包含 [(S^R - O), (O^R - S)]
    if m == n:
        M = np.hstack((Sr - Obs, Obsr - S))
    else:
        # 根据[1]节描述的矩阵。
        M = np.zeros((2*m, 2*order))
        M[:m, :order] = Sr - Obs
        M[m:, order:] = Obsr - S

    # Naive forward-backward and backward-forward filters.
    # These have large transients because the filters use zero initial
    # conditions.
    # 使用前向-后向和后向-前向滤波器。
    # 这些滤波器因使用零初始条件而具有较大的瞬态响应。
    y_f = lfilter(b, a, x)
    y_fb = lfilter(b, a, y_f[..., ::-1])[..., ::-1]

    y_b = lfilter(b, a, x[..., ::-1])[..., ::-1]
    y_bf = lfilter(b, a, y_b)

    delta_y_bf_fb = y_bf - y_fb
    if m == n:
        delta = delta_y_bf_fb
    else:
        start_m = delta_y_bf_fb[..., :m]
        end_m = delta_y_bf_fb[..., -m:]
        delta = np.concatenate((start_m, end_m), axis=-1)

    # ic_opt holds the "optimal" initial conditions.
    # The following code computes the result shown in the formula
    # of the paper between equations (6) and (7).
    # ic_opt保存了“最优”初始条件。
    # 以下代码计算了在论文中6和7之间公式中显示的结果。
    if delta.ndim == 1:
        ic_opt = linalg.lstsq(M, delta)[0]
    else:
        # Reshape delta so it can be used as an array of multiple
        # right-hand-sides in linalg.lstsq.
        # 重新整形delta，使其可以作为linalg.lstsq中多个右侧矩阵的数组使用。
        delta2d = delta.reshape(-1, delta.shape[-1]).T
        ic_opt0 = linalg.lstsq(M, delta2d)[0].T
        ic_opt = ic_opt0.reshape(delta.shape[:-1] + (M.shape[-1],))

    # Now compute the filtered signal using equation (7) of [1].
    # First, form [S^R, O^R] and call it W.
    # 现在使用[1]中的第7个方程计算过滤信号。
    # 首先，形成[S^R, O^R]，称为W。
    if m == n:
        W = np.hstack((Sr, Obsr))
    else:
        W = np.zeros((2*m, 2*order))
        W[:m, :order] = Sr
        W[m:, order:] = Obsr

    # Equation (7) of [1] says
    #     Y_fb^opt = Y_fb^0 + W * [x_0^opt; x_{N-1}^opt]
    # `wic` is (almost) the product on the right.
    # W has shape (m, 2*order), and ic_opt has shape (..., 2*order),
    # so we can't use W.dot(ic_opt).  Instead, we dot ic_opt with W.T,
    # so wic has shape (..., m).
    # [1]中的第7个方程式说
    #     Y_fb^opt = Y_fb^0 + W * [x_0^opt; x_{N-1}^opt]
    # `wic` (几乎)是右侧的乘积。
    # W的形状为(m, 2*order)，ic_opt的形状为(..., 2*order)，
    # 因此我们不能使用W.dot(ic_opt)。相反，我们用ic_opt与W.T点乘，
    # 所以wic的形状为(..., m)。
    wic = ic_opt.dot(W.T)

    # `wic` is "almost" the product of W and the optimal ICs in equation
    # (7)--if we're using a truncated impulse response (m < n), `wic`
    # contains only the adjustments required for the ends of the signal.
    # Here we form y_opt, taking this into account if necessary.
    # `wic` (几乎)是方程(7)中W和最优初始条件的乘积，
    # 如果我们使用截断冲激响应(m < n)，`wic`仅包含信号末端所需的调整。
    # 在这里，我们形成y_opt，必要时考虑这一点。
    y_opt = y_fb
    if m == n:
        y_opt += wic
    else:
        y_opt[..., :m] += wic[..., :m]
        y_opt[..., -m:] += wic[..., -m:]

    x0 = ic_opt[..., :order]
    x1 = ic_opt[..., -order:]
    if axis != -1 or axis != x.ndim - 1:
        # Restore the data axis to its original position.
        # 将数据轴恢复到其原始位置。
        x0 = np.swapaxes(x0, axis, x.ndim - 1)
        x1 = np.swapaxes(x1, axis, x.ndim - 1)
        y_opt = np.swapaxes(y_opt, axis, x.ndim - 1)

    return y_opt, x0, x1
# 定义一个函数，对信号应用前向和后向的数字滤波器，以实现零相位滤波效果
def filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad',
             irlen=None):
    """
    Apply a digital filter forward and backward to a signal.

    This function applies a linear digital filter twice, once forward and
    once backwards.  The combined filter has zero phase and a filter order
    twice that of the original.

    The function provides options for handling the edges of the signal.

    The function `sosfiltfilt` (and filter design using ``output='sos'``)
    should be preferred over `filtfilt` for most filtering tasks, as
    second-order sections have fewer numerical problems.

    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    a : (N,) array_like
        The denominator coefficient vector of the filter.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is ``3 * max(len(a), len(b))``.
    method : str, optional
        Determines the method for handling the edges of the signal, either
        "pad" or "gust".  When `method` is "pad", the signal is padded; the
        type of padding is determined by `padtype` and `padlen`, and `irlen`
        is ignored.  When `method` is "gust", Gustafsson's method is used,
        and `padtype` and `padlen` are ignored.
    irlen : int or None, optional
        When `method` is "gust", `irlen` specifies the length of the
        impulse response of the filter.  If `irlen` is None, no part
        of the impulse response is ignored.  For a long signal, specifying
        `irlen` can significantly improve the performance of the filter.

    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `x`.

    See Also
    --------
    sosfiltfilt, lfilter_zi, lfilter, lfiltic, savgol_filter, sosfilt

    Notes
    -----
    When `method` is "pad", the function pads the data along the given axis
    in one of three ways: odd, even or constant.  The odd and even extensions
    have the corresponding symmetry about the end point of the data.  The
    constant extension extends the data with the values at the end points. On
    both the forward and backward passes, the initial condition of the
    """
    filter is found by using `lfilter_zi` and scaling it by the end point of
    the extended data.



    When `method` is "gust", Gustafsson's method [1]_ is used.  Initial
    conditions are chosen for the forward and backward passes so that the
    forward-backward filter gives the same result as the backward-forward
    filter.



    The option to use Gustaffson's method was added in scipy version 0.16.0.



    References
    ----------
    .. [1] F. Gustaffson, "Determining the initial states in forward-backward
           filtering", Transactions on Signal Processing, Vol. 46, pp. 988-992,
           1996.



    Examples
    --------
    The examples will use several functions from `scipy.signal`.



    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt



    First we create a one second signal that is the sum of two pure sine
    waves, with frequencies 5 Hz and 250 Hz, sampled at 2000 Hz.



    >>> t = np.linspace(0, 1.0, 2001)
    >>> xlow = np.sin(2 * np.pi * 5 * t)
    >>> xhigh = np.sin(2 * np.pi * 250 * t)
    >>> x = xlow + xhigh



    Now create a lowpass Butterworth filter with a cutoff of 0.125 times
    the Nyquist frequency, or 125 Hz, and apply it to ``x`` with `filtfilt`.
    The result should be approximately ``xlow``, with no phase shift.



    >>> b, a = signal.butter(8, 0.125)
    >>> y = signal.filtfilt(b, a, x, padlen=150)
    >>> np.abs(y - xlow).max()
    9.1086182074789912e-06



    We get a fairly clean result for this artificial example because
    the odd extension is exact, and with the moderately long padding,
    the filter's transients have dissipated by the time the actual data
    is reached.  In general, transient effects at the edges are
    unavoidable.



    The following example demonstrates the option ``method="gust"``.



    First, create a filter.



    >>> b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied.



    `sig` is a random input signal to be filtered.



    >>> rng = np.random.default_rng()
    >>> n = 60
    >>> sig = rng.standard_normal(n)**3 + 3*rng.standard_normal(n).cumsum()



    Apply `filtfilt` to `sig`, once using the Gustafsson method, and
    once using padding, and plot the results for comparison.



    >>> fgust = signal.filtfilt(b, a, sig, method="gust")
    >>> fpad = signal.filtfilt(b, a, sig, padlen=50)
    >>> plt.plot(sig, 'k-', label='input')
    >>> plt.plot(fgust, 'b-', linewidth=4, label='gust')
    >>> plt.plot(fpad, 'c-', linewidth=1.5, label='pad')
    >>> plt.legend(loc='best')
    >>> plt.show()



    The `irlen` argument can be used to improve the performance
    of Gustafsson's method.



    Estimate the impulse response length of the filter.



    >>> z, p, k = signal.tf2zpk(b, a)
    >>> eps = 1e-9
    >>> r = np.max(np.abs(p))
    >>> approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
    >>> approx_impulse_len
    137



    Apply the filter to a longer signal, with and without the `irlen`
    """
    Apply a zero-phase digital filter to a signal using the filtfilt function.

    >>> x = rng.standard_normal(4000)
    >>> y1 = signal.filtfilt(b, a, x, method='gust')
    >>> y2 = signal.filtfilt(b, a, x, method='gust', irlen=approx_impulse_len)
    >>> print(np.max(np.abs(y1 - y2)))
    2.875334415008979e-10

    """
    # Ensure b and a are arrays of at least one dimension.
    b = np.atleast_1d(b)
    a = np.atleast_1d(a)
    x = np.asarray(x)

    # Validate the method parameter.
    if method not in ["pad", "gust"]:
        raise ValueError("method must be 'pad' or 'gust'.")

    # Apply the Gustafsson method for zero-phase filtering.
    if method == "gust":
        y, z1, z2 = _filtfilt_gust(b, a, x, axis=axis, irlen=irlen)
        return y

    # Handle the 'pad' method.
    edge, ext = _validate_pad(padtype, padlen, x, axis,
                              ntaps=max(len(a), len(b)))

    # Compute initial conditions for the filter.
    zi = lfilter_zi(b, a)

    # Reshape zi and create x0 to match the 'zi' keyword argument shape.
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = np.reshape(zi, zi_shape)
    x0 = axis_slice(ext, stop=1, axis=axis)

    # Forward filtering.
    (y, zf) = lfilter(b, a, ext, axis=axis, zi=zi * x0)

    # Backward filtering.
    y0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = lfilter(b, a, axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

    # Reverse y to obtain zero-phase filtering result.
    y = axis_reverse(y, axis=axis)

    # Slice the actual signal from the extended signal if edge padding was applied.
    if edge > 0:
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y
# 辅助函数，用于验证 filtfit 的填充方式
def _validate_pad(padtype, padlen, x, axis, ntaps):
    """Helper to validate padding for filtfilt"""
    # 检查填充类型是否合法，必须是 'even', 'odd', 'constant' 或 None
    if padtype not in ['even', 'odd', 'constant', None]:
        raise ValueError(("Unknown value '%s' given to padtype.  padtype "
                          "must be 'even', 'odd', 'constant', or None.") %
                         padtype)

    if padtype is None:
        # 如果填充类型为 None，则填充长度设为 0
        padlen = 0

    if padlen is None:
        # 对于旧的填充方式，保留以保证向后兼容性。
        edge = ntaps * 3
    else:
        edge = padlen

    # x 在 'axis' 维度上的长度必须大于 edge
    if x.shape[axis] <= edge:
        raise ValueError("The length of the input vector x must be greater "
                         "than padlen, which is %d." % edge)

    if padtype is not None and edge > 0:
        # 在输入数组的每一端扩展长度为 `edge`
        if padtype == 'even':
            ext = even_ext(x, edge, axis=axis)
        elif padtype == 'odd':
            ext = odd_ext(x, edge, axis=axis)
        else:
            ext = const_ext(x, edge, axis=axis)
    else:
        ext = x
    return edge, ext


def _validate_x(x):
    # 将输入 x 转换为 NumPy 数组
    x = np.asarray(x)
    if x.ndim == 0:
        # 如果 x 的维度为 0，抛出异常
        raise ValueError('x must be at least 1-D')
    return x


def sosfilt(sos, x, axis=-1, zi=None):
    """
    Filter data along one dimension using cascaded second-order sections.

    Filter a data sequence, `x`, using a digital IIR filter defined by
    `sos`.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the cascaded filter delays.  It is a (at
        least 2D) vector of shape ``(n_sections, ..., 2, ...)``, where
        ``..., 2, ...`` denotes the shape of `x`, but with ``x.shape[axis]``
        replaced by 2.  If `zi` is None or is not given then initial rest
        (i.e. all zeros) is assumed.
        Note that these initial conditions are *not* the same as the initial
        conditions given by `lfiltic` or `lfilter_zi`.

    Returns
    -------
    y : ndarray
        The output of the digital filter.
    zf : ndarray, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.

    See Also
    --------
    zpk2sos, sos2zpk, sosfilt_zi, sosfiltfilt, sosfreqz

    Notes
    -----
    The filter function is implemented as a series of second-order filters
    """
    """
    Perform digital filtering using second-order sections (SOS).
    
    This function applies a digital filter to a signal `x` using the given SOS
    (filtering sections) `sos`. It supports various filter designs and is 
    designed to handle high-order filters with reduced numerical precision errors.
    
    Parameters
    ----------
    x : array_like
        Input signal to be filtered.
    sos : array_like
        Second-order sections of the filter. This can be obtained from filter 
        design functions like `signal.ellip`.
    zi : array_like, optional
        Initial conditions for the filter states.
    axis : int, optional
        Axis along which to apply the filter. Default is -1.
        
    Returns
    -------
    out : ndarray or tuple of ndarray
        Filtered signal or tuple of filtered signal and updated initial conditions
        (`zi`) depending on whether `zi` is provided.
    
    Raises
    ------
    NotImplementedError
        If the input data type is not supported (must be 'float64', 'float32', 
        'complex128', 'complex64', 'float', 'double', 'long double', or 'g' in 
        any of these cases).
    ValueError
        If the initial conditions `zi` do not have a compatible shape with the input
        signal `x` and the number of filtering sections.
    
    Notes
    -----
    This function operates in-place on the input signal `x` and the initial 
    conditions `zi` when provided.
    
    Examples
    --------
    Filtering an input signal `x` using a 13th-order elliptic filter, comparing
    the results of `lfilter` and `sosfilt` to visualize stability issues arising
    from high-order filters:
    
    >>> import matplotlib.pyplot as plt
    >>> from scipy import signal
    >>> b, a = signal.ellip(13, 0.009, 80, 0.05, output='ba')
    >>> sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
    >>> x = signal.unit_impulse(700)
    >>> y_tf = signal.lfilter(b, a, x)
    >>> y_sos = signal.sosfilt(sos, x)
    >>> plt.plot(y_tf, 'r', label='TF')
    >>> plt.plot(y_sos, 'k', label='SOS')
    >>> plt.legend(loc='best')
    >>> plt.show()
    
    Version Information
    -------------------
    0.16.0: Added support for SOS filtering to handle high-order filters.
    
    """
    x = _validate_x(x)
    # Validate and return the input signal `x`.
    sos, n_sections = _validate_sos(sos)
    # Validate the SOS (second-order sections) `sos` and return it along with
    # the number of sections (`n_sections`).
    x_zi_shape = list(x.shape)
    x_zi_shape[axis] = 2
    x_zi_shape = tuple([n_sections] + x_zi_shape)
    # Define the shape of initial conditions `zi` based on `x` and `axis`.
    
    inputs = [sos, x]
    # Create a list of inputs for determining the dtype.
    if zi is not None:
        inputs.append(np.asarray(zi))
    # Include `zi` in inputs if it's provided.
    
    dtype = np.result_type(*inputs)
    # Determine the common dtype for the inputs.
    if dtype.char not in 'fdgFDGO':
        raise NotImplementedError("input type '%s' not supported" % dtype)
    # Raise an error if the determined dtype is not supported.
    
    if zi is not None:
        zi = np.array(zi, dtype)  # make a copy so that we can operate in place
        if zi.shape != x_zi_shape:
            raise ValueError('Invalid zi shape. With axis=%r, an input with '
                             'shape %r, and an sos array with %d sections, zi '
                             'must have shape %r, got %r.' %
                             (axis, x.shape, n_sections, x_zi_shape, zi.shape))
        return_zi = True
    else:
        zi = np.zeros(x_zi_shape, dtype=dtype)
        return_zi = False
    # Set up initial conditions `zi` based on whether it's provided or not.
    
    axis = axis % x.ndim  # make positive
    # Ensure `axis` is within the valid range.
    
    x = np.moveaxis(x, axis, -1)
    zi = np.moveaxis(zi, [0, axis + 1], [-2, -1])
    # Move axes to prepare `x` and `zi` for filtering.
    
    x_shape, zi_shape = x.shape, zi.shape
    # Store the original shapes of `x` and `zi`.
    
    x = np.reshape(x, (-1, x.shape[-1]))
    x = np.array(x, dtype, order='C')  # make a copy, can modify in place
    # Reshape and copy `x` to prepare for filtering.
    
    zi = np.ascontiguousarray(np.reshape(zi, (-1, n_sections, 2)))
    sos = sos.astype(dtype, copy=False)
    # Prepare `zi` and `sos` for filtering.
    
    _sosfilt(sos, x, zi)
    # Perform the filtering operation.
    
    x.shape = x_shape
    x = np.moveaxis(x, -1, axis)
    # Restore the shape of `x` and its axis.
    
    if return_zi:
        zi.shape = zi_shape
        zi = np.moveaxis(zi, [-2, -1], [0, axis + 1])
        out = (x, zi)
    else:
        out = x
    # Prepare the output depending on whether `zi` needs to be returned.
    
    return out
    # Return the filtered signal or tuple of filtered signal and `zi`.
# 定义一个函数，使用级联的二阶段数字滤波器进行前向-后向滤波操作
def sosfiltfilt(sos, x, axis=-1, padtype='odd', padlen=None):
    """
    A forward-backward digital filter using cascaded second-order sections.

    See `filtfilt` for more complete information about this method.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is::

            3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                        (sos[:, 5] == 0).sum()))

        The extra subtraction at the end attempts to compensate for poles
        and zeros at the origin (e.g. for odd-order filters) to yield
        equivalent estimates of `padlen` to those of `filtfilt` for
        second-order section filters built with `scipy.signal` functions.

    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `x`.

    See Also
    --------
    filtfilt, sosfilt, sosfilt_zi, sosfreqz

    Notes
    -----
    .. versionadded:: 0.18.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import sosfiltfilt, butter
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()

    Create an interesting signal to filter.

    >>> n = 201
    >>> t = np.linspace(0, 1, n)
    >>> x = 1 + (t < 0.5) - 0.25*t**2 + 0.05*rng.standard_normal(n)

    Create a lowpass Butterworth filter, and use it to filter `x`.

    >>> sos = butter(4, 0.125, output='sos')
    >>> y = sosfiltfilt(sos, x)

    For comparison, apply an 8th order filter using `sosfilt`.  The filter
    is initialized using the mean of the first four values of `x`.

    >>> from scipy.signal import sosfilt, sosfilt_zi
    >>> sos8 = butter(8, 0.125, output='sos')
    >>> zi = x[:4].mean() * sosfilt_zi(sos8)
    >>> y2, zo = sosfilt(sos8, x, zi=zi)

    Plot the results.  Note that the phase of `y` matches the input, while
    `y2` has a significant phase delay.

    >>> plt.plot(t, x, alpha=0.5, label='x(t)')
    >>> plt.plot(t, y, label='y(t)')
    >>> plt.plot(t, y2, label='y2(t)')
    """
    # 添加图例，设置透明度为1，启用阴影效果
    >>> plt.legend(framealpha=1, shadow=True)
    # 添加网格线，设置透明度为0.25
    >>> plt.grid(alpha=0.25)
    # 设置 x 轴标签为 't'
    >>> plt.xlabel('t')
    # 显示绘图结果
    >>> plt.show()

    """
    # 验证并返回修正后的 sos 和 n_sections
    sos, n_sections = _validate_sos(sos)
    # 验证并返回修正后的输入信号 x
    x = _validate_x(x)

    # 如果 `padtype` 是 "pad"，计算滤波器的 taps 数量
    ntaps = 2 * n_sections + 1
    # 减去不参与计算的 sos 段数，以优化滤波效果
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    # 验证并返回填充后的输入信号及扩展部分
    edge, ext = _validate_pad(padtype, padlen, x, axis, ntaps=ntaps)

    # 使用 sosfilt_zi 函数计算初始状态 zi
    zi = sosfilt_zi(sos)  # 形状为 (n_sections, 2) --> (n_sections, ..., 2, ...)
    # 调整 zi 的形状以匹配输入信号的维度
    zi_shape = [1] * x.ndim
    zi_shape[axis] = 2
    zi.shape = [n_sections] + zi_shape
    # 提取输入信号的前一部分进行初步滤波
    x_0 = axis_slice(ext, stop=1, axis=axis)
    # 执行初步滤波，返回滤波后的信号 y 和最终状态 zf
    (y, zf) = sosfilt(sos, ext, axis=axis, zi=zi * x_0)
    # 提取滤波后信号的最后一部分进行反向滤波
    y_0 = axis_slice(y, start=-1, axis=axis)
    # 执行反向滤波，返回反向滤波后的信号 y 和最终状态 zf
    (y, zf) = sosfilt(sos, axis_reverse(y, axis=axis), axis=axis, zi=zi * y_0)
    # 恢复信号的正向顺序
    y = axis_reverse(y, axis=axis)
    # 如果进行了填充，移除填充的边缘部分
    if edge > 0:
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)
    # 返回最终处理完成的信号 y
    return y
    # 将输入信号转换为 NumPy 数组，确保处理为多维数组
    x = np.asarray(x)
    # 将 downsampling factor 转换为整数类型，以便后续使用
    q = operator.index(q)

    # 如果指定了滤波器的阶数，将其转换为整数类型
    if n is not None:
        n = operator.index(n)

    # 确定输出信号的数据类型，默认为输入信号的数据类型，但不支持精确类型和 float16
    result_type = x.dtype
    if not np.issubdtype(result_type, np.inexact) \
       or result_type.type == np.float16:
        # 对于整数和 float16 类型的数据，将其转换为 float64 类型
        result_type = np.float64
    # 如果滤波类型为 'fir'
    if ftype == 'fir':
        # 如果未指定滤波器长度 n，则设定为 10 * q 的长度，用于类似 sinc 函数的截止
        if n is None:
            half_len = 10 * q  # 合理的截止长度，适用于我们类似 sinc 函数的滤波器
            n = 2 * half_len
        # 使用 hamming 窗口生成 FIR 滤波器系数 b 和 a 设置为 1
        b, a = firwin(n+1, 1. / q, window='hamming'), 1.
        # 将 b 转换为指定结果类型的 numpy 数组
        b = np.asarray(b, dtype=result_type)
        # 将 a 转换为指定结果类型的 numpy 数组
        a = np.asarray(a, dtype=result_type)
    
    # 如果滤波类型为 'iir'
    elif ftype == 'iir':
        # 设定使用 sos 形式的 IIR 滤波器
        iir_use_sos = True
        # 如果未指定滤波器长度 n，则设定为 8
        if n is None:
            n = 8
        # 使用 cheby1 方法生成指定参数的 IIR 滤波器的二阶节序列
        sos = cheby1(n, 0.05, 0.8 / q, output='sos')
        # 将 sos 转换为指定结果类型的 numpy 数组
        sos = np.asarray(sos, dtype=result_type)
    
    # 如果滤波类型是 dlti 类型的实例
    elif isinstance(ftype, dlti):
        # 将 dlti 类型转换为零极点增益形式的系统表示
        system = ftype._as_zpk()
        # 如果系统的极点数量为 0，判断为 FIR 类型
        if system.poles.shape[0] == 0:
            # 转换为传递函数形式的系统表示
            system = ftype._as_tf()
            # 获取传递函数的分子和分母系数
            b, a = system.num, system.den
            # 将滤波类型设定为 'fir'
            ftype = 'fir'
        # 如果极点包含复数或者增益包含复数，判断为 IIR 类型，但不适合 sosfilt & sosfiltfilt
        elif (any(np.iscomplex(system.poles))
              or any(np.iscomplex(system.poles))
              or np.iscomplex(system.gain)):
            # 不使用 sos 形式滤波器
            iir_use_sos = False
            # 转换为传递函数形式的系统表示
            system = ftype._as_tf()
            # 获取传递函数的分子和分母系数
            b, a = system.num, system.den
        else:
            # 使用 sos 形式表示系统的零极点增益
            iir_use_sos = True
            sos = zpk2sos(system.zeros, system.poles, system.gain)
            # 将 sos 转换为指定结果类型的 numpy 数组
            sos = np.asarray(sos, dtype=result_type)
    
    else:
        # 如果滤波类型未定义，引发值错误异常
        raise ValueError('invalid ftype')

    # 创建一个与输入信号 x 维度相同的切片列表
    sl = [slice(None)] * x.ndim

    # 如果滤波类型为 'fir'
    if ftype == 'fir':
        # 对 b 和 a 进行归一化处理
        b = b / a
        # 如果需要零相移
        if zero_phase:
            # 使用 resample_poly 函数进行多项式重采样，使用 b 作为窗口函数
            y = resample_poly(x, 1, q, axis=axis, window=b)
        else:
            # 使用 upfirdn 函数进行多项式重采样
            # 通常比 lfilter 快 q 倍，因为它只计算所需的输出
            n_out = x.shape[axis] // q + bool(x.shape[axis] % q)
            y = upfirdn(b, x, up=1, down=q, axis=axis)
            # 更新切片列表，以输出所需范围的数据
            sl[axis] = slice(None, n_out, None)

    else:  # IIR 滤波器情况
        # 如果需要零相移
        if zero_phase:
            # 如果使用 sos 形式的 IIR 滤波器
            if iir_use_sos:
                # 使用 sosfiltfilt 函数进行零相移滤波
                y = sosfiltfilt(sos, x, axis=axis)
            else:
                # 使用 filtfilt 函数进行零相移滤波
                y = filtfilt(b, a, x, axis=axis)
        else:
            # 如果使用 sos 形式的 IIR 滤波器
            if iir_use_sos:
                # 使用 sosfilt 函数进行滤波
                y = sosfilt(sos, x, axis=axis)
            else:
                # 使用 lfilter 函数进行滤波
                y = lfilter(b, a, x, axis=axis)

        # 更新切片列表，以输出所需步长的数据
        sl[axis] = slice(None, None, q)

    # 返回根据更新切片列表取得的输出信号 y
    return y[tuple(sl)]
```