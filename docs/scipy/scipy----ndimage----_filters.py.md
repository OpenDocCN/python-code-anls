# `D:\src\scipysrc\scipy\scipy\ndimage\_filters.py`

```
# 导入必要的模块和库
from collections.abc import Iterable  # 导入集合模块中的Iterable类
import numbers  # 导入处理数值的模块
import warnings  # 导入警告模块
import numpy as np  # 导入NumPy库，并使用别名np
import operator  # 导入操作符模块

# 导入Scipy图像处理模块中的子模块和文档字符串模块
from scipy._lib._util import normalize_axis_index
from . import _ni_support
from . import _nd_image
from . import _ni_docstrings

# 定义本模块中可以导出的函数和类列表
__all__ = ['correlate1d', 'convolve1d', 'gaussian_filter1d', 'gaussian_filter',
           'prewitt', 'sobel', 'generic_laplace', 'laplace',
           'gaussian_laplace', 'generic_gradient_magnitude',
           'gaussian_gradient_magnitude', 'correlate', 'convolve',
           'uniform_filter1d', 'uniform_filter', 'minimum_filter1d',
           'maximum_filter1d', 'minimum_filter', 'maximum_filter',
           'rank_filter', 'median_filter', 'percentile_filter',
           'generic_filter1d', 'generic_filter']

# 定义一个函数，用于检查卷积操作的起始点是否有效
def _invalid_origin(origin, lenw):
    return (origin < -(lenw // 2)) or (origin > (lenw - 1) // 2)

# 定义一个函数，用于实现通过实部分量进行复杂卷积的功能
def _complex_via_real_components(func, input, weights, output, cval, **kwargs):
    """Complex convolution via a linear combination of real convolutions."""
    # 检查输入和权重是否是复数类型
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    # 如果输入和权重都是复数
    if complex_input and complex_weights:
        # 处理输出的实部
        func(input.real, weights.real, output=output.real,
             cval=np.real(cval), **kwargs)
        # 从输出的实部中减去处理输入的虚部
        output.real -= func(input.imag, weights.imag, output=None,
                            cval=np.imag(cval), **kwargs)
        
        # 处理输出的虚部
        func(input.real, weights.imag, output=output.imag,
             cval=np.real(cval), **kwargs)
        # 在输出的虚部中加上处理输入的实部
        output.imag += func(input.imag, weights.real, output=None,
                            cval=np.imag(cval), **kwargs)
    
    # 如果输入是复数但权重是实数
    elif complex_input:
        # 处理输出的实部
        func(input.real, weights, output=output.real, cval=np.real(cval),
             **kwargs)
        # 处理输出的虚部
        func(input.imag, weights, output=output.imag, cval=np.imag(cval),
             **kwargs)
    
    # 如果输入是实数
    else:
        # 如果 cval 是复数则抛出错误
        if np.iscomplexobj(cval):
            raise ValueError("Cannot provide a complex-valued cval when the "
                             "input is real.")
        # 处理输出的实部
        func(input, weights.real, output=output.real, cval=cval, **kwargs)
        # 处理输出的虚部
        func(input, weights.imag, output=output.imag, cval=cval, **kwargs)
    
    # 返回处理后的输出
    return output
# 使用装饰器 `_ni_docstrings.docfiller` 来填充文档字符串中的参数描述
@_ni_docstrings.docfiller
# 定义函数 `correlate1d`，用于计算沿给定轴的一维相关性
def correlate1d(input, weights, axis=-1, output=None, mode="reflect",
                cval=0.0, origin=0):
    """Calculate a 1-D correlation along the given axis.

    The lines of the array along the given axis are correlated with the
    given weights.

    Parameters
    ----------
    %(input)s
        ndarray or sequence
        The input array or sequence to be correlated.
    weights : array
        1-D sequence of numbers.
        The weights to be used for correlation.
    %(axis)s
        int, optional
        The axis along which to compute the correlation.
    %(output)s
        ndarray, optional
        The output array in which to place the result.
    %(mode_reflect)s
        {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the input array is extended when the filter overlaps a border.
    %(cval)s
        float, optional
        Value to fill past edges of input if mode is 'constant'.
    %(origin)s
        int, optional
        The origin parameter controls the placement of the filter.

    Returns
    -------
    result : ndarray
        Correlation result. Has the same shape as `input`.

    Examples
    --------
    >>> from scipy.ndimage import correlate1d
    >>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([ 8, 26,  8, 12,  7, 28, 36,  9])
    """
    # 将输入转换为 ndarray 类型
    input = np.asarray(input)
    # 将权重转换为 ndarray 类型
    weights = np.asarray(weights)
    # 检查输入是否为复数类型
    complex_input = input.dtype.kind == 'c'
    # 检查权重是否为复数类型
    complex_weights = weights.dtype.kind == 'c'
    # 如果输入或权重是复数类型，则处理复数相关性
    if complex_input or complex_weights:
        # 如果权重是复数类型，则对其进行共轭操作并转换为 np.complex128 类型
        if complex_weights:
            weights = weights.conj()
            weights = weights.astype(np.complex128, copy=False)
        # 设置关键字参数字典
        kwargs = dict(axis=axis, mode=mode, origin=origin)
        # 获取输出数组，支持复数输出
        output = _ni_support._get_output(output, input, complex_output=True)
        # 使用实部组成复数相关性结果
        return _complex_via_real_components(correlate1d, input, weights,
                                            output, cval, **kwargs)

    # 获取输出数组，不支持复数输出
    output = _ni_support._get_output(output, input)
    # 将权重数组转换为 np.float64 类型
    weights = np.asarray(weights, dtype=np.float64)
    # 检查权重数组是否为一维且长度大于等于1
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise RuntimeError('no filter weights given')
    # 如果权重数组不是连续存储的，则进行复制以确保连续性
    if not weights.flags.contiguous:
        weights = weights.copy()
    # 规范化轴索引
    axis = normalize_axis_index(axis, input.ndim)
    # 检查起始点是否合法
    if _invalid_origin(origin, len(weights)):
        raise ValueError('Invalid origin; origin must satisfy '
                         '-(len(weights) // 2) <= origin <= '
                         '(len(weights)-1) // 2')
    # 将模式扩展为对应的代码
    mode = _ni_support._extend_mode_to_code(mode)
    # 调用 C 扩展的函数进行一维相关性计算
    _nd_image.correlate1d(input, weights, axis, output, mode, cval,
                          origin)
    # 返回相关性结果数组
    return output


# 使用装饰器 `_ni_docstrings.docfiller` 来填充文档字符串中的参数描述
@_ni_docstrings.docfiller
# 定义函数 `convolve1d`，用于计算沿给定轴的一维卷积
def convolve1d(input, weights, axis=-1, output=None, mode="reflect",
               cval=0.0, origin=0):
    """Calculate a 1-D convolution along the given axis.

    The lines of the array along the given axis are convolved with the
    given weights.

    Parameters
    ----------
    %(input)s
        ndarray or sequence
        The input array or sequence to be convolved.
    weights : ndarray
        1-D sequence of numbers.
        The weights to be used for convolution.
    %(axis)s
        int, optional
        The axis along which to compute the convolution.
    %(output)s
        ndarray, optional
        The output array in which to place the result.
    %(mode_reflect)s
        {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the input array is extended when the filter overlaps a border.
    %(cval)s
        float, optional
        Value to fill past edges of input if mode is 'constant'.
    %(origin)s
        int, optional
        The origin parameter controls the placement of the filter.

    Returns
    -------
    convolve1d : ndarray
        Convolved array with same shape as input

    Examples
    --------
    >>> from scipy.ndimage import convolve1d
    >>> convolve1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([14, 24,  4, 13, 12, 36, 27,  0])
    """
    # 将权重数组翻转
    weights = weights[::-1]
    # 将起始点取反
    origin = -origin
    # 如果权重数组长度为偶数，则调整起始点
    if not len(weights) & 1:
        origin -= 1
    # 将权重数组转换为 ndarray 类型
    weights = np.asarray(weights)
    # 检查权重数组的数据类型是否为复数（complex）
    if weights.dtype.kind == 'c':
        # 如果是复数，进行共轭操作，以抵消 correlate1d 中的共轭操作影响
        weights = weights.conj()
    # 调用 correlate1d 函数进行一维相关操作，并返回结果
    return correlate1d(input, weights, axis, output, mode, cval, origin)
# 定义一个函数用于计算一维高斯卷积核
def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    计算一维高斯卷积核。

    Parameters
    ----------
    sigma : scalar
        Gaussian核的标准差
    order : int
        卷积的阶数，0表示与高斯核卷积，正数表示与高斯核的导数卷积
    radius : int
        半径，用于定义核的大小

    Returns
    -------
    ndarray
        返回计算出的高斯卷积核

    Notes
    -----
    实现了高斯卷积核的计算，支持不同阶数和半径。

    """
    # 如果阶数小于0，抛出值错误异常
    if order < 0:
        raise ValueError('order must be non-negative')
    
    # 创建指数范围，长度为阶数加1
    exponent_range = np.arange(order + 1)
    
    # 计算 sigma 的平方
    sigma2 = sigma * sigma
    
    # 创建以 [-radius, radius] 范围内的数组 x
    x = np.arange(-radius, radius+1)
    
    # 计算高斯函数的值 phi(x)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()  # 归一化
    
    # 如果阶数为0，直接返回 phi_x
    if order == 0:
        return phi_x
    else:
        # 计算 q(x) 的系数矩阵运算
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        # 计算最终的卷积核 q(x) * phi(x)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


# 定义一个装饰器函数，用于填充文档字符串
@_ni_docstrings.docfiller
def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0, *, radius=None):
    """1-D Gaussian filter.

    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.
    %(output)s
    %(mode_reflect)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    radius : None or int, optional
        Radius of the Gaussian kernel. If specified, the size of
        the kernel will be ``2*radius + 1``, and `truncate` is ignored.
        Default is None.

    Returns
    -------
    gaussian_filter1d : ndarray
        返回一维高斯滤波后的数组

    Notes
    -----
    高斯核的大小为 ``2*radius + 1``。如果 `radius` 是 None，将使用默认的 ``radius = round(truncate * sigma)``。

    Examples
    --------
    提供了几个使用示例，展示了如何对数据进行一维高斯滤波。

    """
    sd = float(sigma)
    # 将滤波器的半径设置为截断标准偏差的整数值
    lw = int(truncate * sd + 0.5)
    # 如果指定了半径参数，则使用指定的半径值
    if radius is not None:
        lw = radius
    # 检查半径值是否为非负整数，若不是则抛出数值错误异常
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError('Radius must be a nonnegative integer.')
    # 由于调用的是相关函数 correlate1d 而不是卷积函数 convolve1d，因此需要将内核反转
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    # 返回对输入数据进行一维相关运算的结果
    return correlate1d(input, weights, axis, output, mode, cval, 0)
# 使用装饰器 @_ni_docstrings.docfiller 注册此函数以填充文档字符串
@_ni_docstrings.docfiller
# 定义多维高斯滤波器函数，接受多个参数
def gaussian_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0, *, radius=None,
                    axes=None):
    """Multidimensional Gaussian filter.

    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    order : int or sequence of ints, optional
        The order of the filter along each axis is given as a sequence
        of integers, or as a single number. An order of 0 corresponds
        to convolution with a Gaussian kernel. A positive order
        corresponds to convolution with that derivative of a Gaussian.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    radius : None or int or sequence of ints, optional
        Radius of the Gaussian kernel. The radius are given for each axis
        as a sequence, or as a single number, in which case it is equal
        for all axes. If specified, the size of the kernel along each axis
        will be ``2*radius + 1``, and `truncate` is ignored.
        Default is None.
    axes : tuple of int or None, optional
        If None, `input` is filtered along all axes. Otherwise,
        `input` is filtered along the specified axes. When `axes` is
        specified, any tuples used for `sigma`, `order`, `mode` and/or `radius`
        must match the length of `axes`. The ith entry in any of these tuples
        corresponds to the ith entry in `axes`.

    Returns
    -------
    gaussian_filter : ndarray
        Returned array of same shape as `input`.

    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    1-D convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.

    The Gaussian kernel will have size ``2*radius + 1`` along each axis. If
    `radius` is None, the default ``radius = round(truncate * sigma)`` will be
    used.

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter
    >>> import numpy as np
    >>> a = np.arange(50, step=2).reshape((5,5))
    >>> a
    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18],
           [20, 22, 24, 26, 28],
           [30, 32, 34, 36, 38],
           [40, 42, 44, 46, 48]])
    >>> gaussian_filter(a, sigma=1)
    array([[ 4,  6,  8,  9, 11],
           [10, 12, 14, 15, 17],
           [20, 22, 24, 25, 27],
           [29, 31, 33, 34, 36],
           [35, 37, 39, 40, 42]])

    >>> from scipy import datasets
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)

    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    orders = _ni_support._normalize_sequence(order, num_axes)
    sigmas = _ni_support._normalize_sequence(sigma, num_axes)
    modes = _ni_support._normalize_sequence(mode, num_axes)
    radiuses = _ni_support._normalize_sequence(radius, num_axes)
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii], radiuses[ii])
            for ii in range(num_axes) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        # 对每个轴应用高斯滤波器
        for axis, sigma, order, mode, radius in axes:
            # 调用一维高斯滤波器函数来处理输入数据
            gaussian_filter1d(input, sigma, axis, order, output,
                              mode, cval, truncate, radius=radius)
            # 将输出赋给输入，以便下一个轴使用
            input = output
    else:
        # 如果没有轴需要处理，则直接将输入复制到输出
        output[...] = input[...]
    # 返回经过高斯滤波器处理后的输出数据
    return output
@_ni_docstrings.docfiller
def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Calculate a Prewitt filter.

    Parameters
    ----------
    %(input)s
        Input array to filter.
    %(axis)s
        Axis along which to apply the filter.
    %(output)s
        Array to store the output. If None, a new array is created.
    %(mode_multiple)s
        Determines how the input array is extended beyond its boundaries.
    %(cval)s
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    prewitt : ndarray
        Filtered array. Has the same shape as `input`.

    See Also
    --------
    sobel: Sobel filter

    Notes
    -----
    This function computes the one-dimensional Prewitt filter.
    Horizontal edges are emphasised with the horizontal transform (axis=0),
    vertical edges with the vertical transform (axis=1), and so on for higher
    dimensions. These can be combined to give the magnitude.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> ascent = datasets.ascent()
    >>> prewitt_h = ndimage.prewitt(ascent, axis=0)
    >>> prewitt_v = ndimage.prewitt(ascent, axis=1)
    >>> magnitude = np.sqrt(prewitt_h ** 2 + prewitt_v ** 2)
    >>> magnitude *= 255 / np.max(magnitude)  # Normalization
    >>> fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    >>> plt.gray()
    >>> axes[0, 0].imshow(ascent)
    >>> axes[0, 1].imshow(prewitt_h)
    >>> axes[1, 0].imshow(prewitt_v)
    >>> axes[1, 1].imshow(magnitude)
    >>> titles = ["original", "horizontal", "vertical", "magnitude"]
    >>> for i, ax in enumerate(axes.ravel()):
    ...     ax.set_title(titles[i])
    ...     ax.axis("off")
    >>> plt.show()

    """
    input = np.asarray(input)  # 将输入转换为 NumPy 数组
    axis = normalize_axis_index(axis, input.ndim)  # 根据输入数组的维度规范化轴索引
    output = _ni_support._get_output(output, input)  # 获取输出数组或者创建一个新数组
    modes = _ni_support._normalize_sequence(mode, input.ndim)  # 规范化 mode 参数为与输入数组维度一致的序列
    correlate1d(input, [-1, 0, 1], axis, output, modes[axis], cval, 0)  # 对输入数组应用一维 Prewitt 滤波器
    axes = [ii for ii in range(input.ndim) if ii != axis]  # 获取除了指定轴外的其他轴
    for ii in axes:
        correlate1d(output, [1, 1, 1], ii, output, modes[ii], cval, 0)  # 对输出数组应用一维 Prewitt 滤波器
    return output


@_ni_docstrings.docfiller
def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Calculate a Sobel filter.

    Parameters
    ----------
    %(input)s
        Input array to filter.
    %(axis)s
        Axis along which to apply the filter.
    %(output)s
        Array to store the output. If None, a new array is created.
    %(mode_multiple)s
        Determines how the input array is extended beyond its boundaries.
    %(cval)s
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    sobel : ndarray
        Filtered array. Has the same shape as `input`.

    Notes
    -----
    This function computes the axis-specific Sobel gradient.
    The horizontal edges can be emphasised with the horizontal transform (axis=0),
    the vertical edges with the vertical transform (axis=1) and so on for higher
    dimensions. These can be combined to give the magnitude.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> ascent = datasets.ascent().astype('int32')
    >>> sobel_h = ndimage.sobel(ascent, 0)  # horizontal gradient
    >>> sobel_v = ndimage.sobel(ascent, 1)  # vertical gradient
    >>> magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    """
    # 将 magnitude 数组进行归一化，使其数值范围映射到 [0, 255]，便于显示
    magnitude *= 255.0 / np.max(magnitude)  # normalization
    
    # 创建一个 2x2 的图像布局，并设置图像的大小为 8x8 英寸
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    
    # 将图像显示模式设置为灰度
    plt.gray()  # show the filtered result in grayscale
    
    # 在第一个子图中显示原始图像 `ascent`
    axs[0, 0].imshow(ascent)
    
    # 在第二个子图中显示水平Sobel滤波器处理后的图像 `sobel_h`
    axs[0, 1].imshow(sobel_h)
    
    # 在第三个子图中显示垂直Sobel滤波器处理后的图像 `sobel_v`
    axs[1, 0].imshow(sobel_v)
    
    # 在第四个子图中显示合成后的梯度幅值图像 `magnitude`
    axs[1, 1].imshow(magnitude)
    
    # 每个子图对应的标题
    titles = ["original", "horizontal", "vertical", "magnitude"]
    
    # 遍历所有子图，并设置标题和关闭坐标轴
    for i, ax in enumerate(axs.ravel()):
        ax.set_title(titles[i])
        ax.axis("off")
    
    # 显示整个图像布局
    plt.show()
# 使用注释填充文档字符串
@_ni_docstrings.docfiller
def generic_laplace(input, derivative2, output=None, mode="reflect",
                    cval=0.0,
                    extra_arguments=(),
                    extra_keywords=None):
    """
    N-D Laplace filter using a provided second derivative function.

    Parameters
    ----------
    %(input)s
        Input array to be filtered.
    derivative2 : callable
        Callable with the following signature::

            derivative2(input, axis, output, mode, cval,
                        *extra_arguments, **extra_keywords)

        Function computing the second derivative along a given axis.
    %(output)s
        Optional. Output array to store the result.
    %(mode_multiple)s
        {{"reflect", "constant", "nearest", "mirror", "wrap"}}
        How to handle the boundaries. Default is 'reflect'.
    %(cval)s
        Value used for constant boundary mode.
    %(extra_keywords)s
        Additional keyword arguments passed to `derivative2`.
    %(extra_arguments)s
        Additional positional arguments passed to `derivative2`.

    Returns
    -------
    generic_laplace : ndarray
        Filtered array. Has the same shape as `input`.
    """
    if extra_keywords is None:
        extra_keywords = {}
    # 将输入转换为 NumPy 数组
    input = np.asarray(input)
    # 获取输出数组或创建一个新的输出数组
    output = _ni_support._get_output(output, input)
    # 获取输入数组的维度列表
    axes = list(range(input.ndim))
    # 如果维度大于 0
    if len(axes) > 0:
        # 标准化处理模式参数
        modes = _ni_support._normalize_sequence(mode, len(axes))
        # 对第一个轴进行二阶导数计算
        derivative2(input, axes[0], output, modes[0], cval,
                    *extra_arguments, **extra_keywords)
        # 对于其他轴，累加计算二阶导数
        for ii in range(1, len(axes)):
            tmp = derivative2(input, axes[ii], output.dtype, modes[ii], cval,
                              *extra_arguments, **extra_keywords)
            output += tmp
    else:
        # 如果输入数组没有维度，则直接复制输入到输出
        output[...] = input[...]
    # 返回过滤后的输出数组
    return output


# 使用注释填充文档字符串
@_ni_docstrings.docfiller
def laplace(input, output=None, mode="reflect", cval=0.0):
    """N-D Laplace filter based on approximate second derivatives.

    Parameters
    ----------
    %(input)s
        Input array to be filtered.
    %(output)s
        Optional. Output array to store the result.
    %(mode_multiple)s
        {{"reflect", "constant", "nearest", "mirror", "wrap"}}
        How to handle the boundaries. Default is 'reflect'.
    %(cval)s
        Value used for constant boundary mode.

    Returns
    -------
    laplace : ndarray
        Filtered array. Has the same shape as `input`.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = ndimage.laplace(ascent)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    # 定义二阶导数函数，使用 correlate1d 进行卷积计算
    def derivative2(input, axis, output, mode, cval):
        return correlate1d(input, [1, -2, 1], axis, output, mode, cval, 0)
    # 调用通用 Laplace 滤波函数
    return generic_laplace(input, derivative2, output, mode, cval)


# 使用注释填充文档字符串
@_ni_docstrings.docfiller
def gaussian_laplace(input, sigma, output=None, mode="reflect",
                     cval=0.0, **kwargs):
    """Multidimensional Laplace filter using Gaussian second derivatives.

    Parameters
    ----------
    %(input)s
        Input array to be filtered.
    sigma : scalar or sequence of scalars
        The standard deviations of the Gaussian filter are given for
        each axis as a sequence, or as a single number, in which case
        it is equal for all axes.
    %(output)s
        Optional. Output array to store the result.
    %(mode_multiple)s
        {{"reflect", "constant", "nearest", "mirror", "wrap"}}
        How to handle the boundaries. Default is 'reflect'.
    %(cval)s
        Value used for constant boundary mode.

    """
    # 将输入转换为 NumPy 数组，确保 input 是 ndarray 类型
    input = np.asarray(input)

    # 定义 derivative2 函数，用于计算二阶导数
    def derivative2(input, axis, output, mode, cval, sigma, **kwargs):
        # 设置导数的阶数，对于给定的轴设置为二阶导数，其它轴为零阶导数
        order = [0] * input.ndim
        order[axis] = 2
        # 调用 gaussian_filter 函数对输入进行高斯滤波，并返回结果
        return gaussian_filter(input, sigma, order, output, mode, cval,
                               **kwargs)

    # 返回通过 generic_laplace 函数处理后的结果
    return generic_laplace(input,            # 输入数组
                           derivative2,      # 二阶导数函数
                           output,           # 输出数组
                           mode,             # 边界处理模式
                           cval,             # 边界外常数值
                           extra_arguments=(sigma,),  # 额外参数，包括 sigma
                           extra_keywords=kwargs)      # 额外的关键字参数
@_ni_docstrings.docfiller
def generic_gradient_magnitude(input, derivative, output=None,
                               mode="reflect", cval=0.0,
                               extra_arguments=(), extra_keywords=None):
    """Gradient magnitude using a provided gradient function.

    Parameters
    ----------
    %(input)s
        Input array.
    derivative : callable
        Callable with the following signature::

            derivative(input, axis, output, mode, cval,
                       *extra_arguments, **extra_keywords)

        See `extra_arguments`, `extra_keywords` below.
        `derivative` can assume that `input` and `output` are ndarrays.
        Note that the output from `derivative` is modified inplace;
        be careful to copy important inputs before returning them.
    %(output)s
        Output array, optional.
    %(mode_multiple)s
        Padding mode for input. Multiple modes can be specified.
    %(cval)s
        Value to fill past edges of input, if `mode` is "constant".
    %(extra_keywords)s
        Additional keyword arguments passed to `derivative`.
    %(extra_arguments)s
        Additional positional arguments passed to `derivative`.

    Returns
    -------
    generic_gradient_magnitude : ndarray
        Filtered array. Has the same shape as `input`.
    """
    if extra_keywords is None:
        extra_keywords = {}  # 如果额外关键字为空，设置为一个空字典
    input = np.asarray(input)  # 将输入转换为NumPy数组
    output = _ni_support._get_output(output, input)  # 获取输出数组，根据输入的形状
    axes = list(range(input.ndim))  # 获取输入数组的维度列表
    if len(axes) > 0:
        modes = _ni_support._normalize_sequence(mode, len(axes))  # 根据轴数规范化填充模式
        derivative(input, axes[0], output, modes[0], cval,
                   *extra_arguments, **extra_keywords)  # 对第一个轴应用导数函数
        np.multiply(output, output, output)  # 对输出数组进行平方操作
        for ii in range(1, len(axes)):
            tmp = derivative(input, axes[ii], output.dtype, modes[ii], cval,
                             *extra_arguments, **extra_keywords)  # 对每个轴应用导数函数
            np.multiply(tmp, tmp, tmp)  # 对临时数组进行平方操作
            output += tmp  # 将临时数组加到输出数组上
        # This allows the sqrt to work with a different default casting
        np.sqrt(output, output, casting='unsafe')  # 对输出数组进行开方操作，不安全地进行类型转换
    else:
        output[...] = input[...]  # 如果轴数为零，直接将输入数组复制给输出数组
    return output


@_ni_docstrings.docfiller
def gaussian_gradient_magnitude(input, sigma, output=None,
                                mode="reflect", cval=0.0, **kwargs):
    """Multidimensional gradient magnitude using Gaussian derivatives.

    Parameters
    ----------
    %(input)s
        Input array.
    sigma : scalar or sequence of scalars
        The standard deviations of the Gaussian filter are given for
        each axis as a sequence, or as a single number, in which case
        it is equal for all axes.
    %(output)s
        Output array, optional.
    %(mode_multiple)s
        Padding mode for input. Multiple modes can be specified.
    %(cval)s
        Value to fill past edges of input, if `mode` is "constant".
    Extra keyword arguments will be passed to gaussian_filter().

    Returns
    -------
    gaussian_gradient_magnitude : ndarray
        Filtered array. Has the same shape as `input`.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = ndimage.gaussian_gradient_magnitude(ascent, sigma=5)

    """
    # 将输入转换为 NumPy 数组（如果尚未是数组）
    input = np.asarray(input)

    # 定义一个函数 derivative，用于计算输入数组的指定轴向的导数
    def derivative(input, axis, output, mode, cval, sigma, **kwargs):
        # 根据指定轴设置导数的阶数
        order = [0] * input.ndim
        order[axis] = 1
        # 调用 gaussian_filter 函数，计算输入数组的高斯平滑后的导数
        return gaussian_filter(input, sigma, order, output, mode,
                               cval, **kwargs)

    # 返回输入数组的梯度幅值，使用 derivative 函数计算导数
    return generic_gradient_magnitude(input, derivative, output, mode,
                                      cval, extra_arguments=(sigma,),
                                      extra_keywords=kwargs)
def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution):
    # 将输入转换为 NumPy 数组
    input = np.asarray(input)
    # 将权重转换为 NumPy 数组
    weights = np.asarray(weights)
    # 检查输入是否为复数类型
    complex_input = input.dtype.kind == 'c'
    # 检查权重是否为复数类型
    complex_weights = weights.dtype.kind == 'c'
    
    # 如果输入或权重其中之一为复数类型
    if complex_input or complex_weights:
        # 如果权重为复数类型且不是卷积操作，将权重取共轭
        if complex_weights and not convolution:
            # 对于 np.correlate，应该对权重取共轭而不是输入
            weights = weights.conj()
        
        # 构建参数字典
        kwargs = dict(
            mode=mode, origin=origin, convolution=convolution
        )
        # 获取输出数组，支持复数输出
        output = _ni_support._get_output(output, input, complex_output=True)

        # 调用复数数据的相关或卷积函数
        return _complex_via_real_components(_correlate_or_convolve, input,
                                            weights, output, cval, **kwargs)

    # 标准化 origin 到输入数组的维度
    origins = _ni_support._normalize_sequence(origin, input.ndim)
    # 将权重数组转换为 float64 类型
    weights = np.asarray(weights, dtype=np.float64)
    # 检查权重数组的形状是否正确
    wshape = [ii for ii in weights.shape if ii > 0]
    if len(wshape) != input.ndim:
        raise RuntimeError('filter weights array has incorrect shape.')
    
    # 如果是卷积操作，反转权重数组
    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        for ii in range(len(origins)):
            origins[ii] = -origins[ii]
            if not weights.shape[ii] & 1:
                origins[ii] -= 1
    
    # 检查每个 origin 是否有效
    for origin, lenw in zip(origins, wshape):
        if _invalid_origin(origin, lenw):
            raise ValueError('Invalid origin; origin must satisfy '
                             '-(weights.shape[k] // 2) <= origin[k] <= '
                             '(weights.shape[k]-1) // 2')

    # 如果权重数组不是连续的，进行复制
    if not weights.flags.contiguous:
        weights = weights.copy()
    
    # 获取输出数组，确保它与输入数组不共享内存
    output = _ni_support._get_output(output, input)
    temp_needed = np.may_share_memory(input, output)
    
    # 如果需要临时数组来存储输出，分配临时数组
    if temp_needed:
        # 输入和输出数组不能共享内存
        temp = output
        output = _ni_support._get_output(output.dtype, input)
    
    # 检查 mode 参数是否是字符串或可迭代对象
    if not isinstance(mode, str) and isinstance(mode, Iterable):
        raise RuntimeError("A sequence of modes is not supported")
    
    # 将 mode 扩展为相应的代码
    mode = _ni_support._extend_mode_to_code(mode)
    
    # 调用 C 函数执行相关操作
    _nd_image.correlate(input, weights, output, mode, cval, origins)
    
    # 如果需要临时数组来存储输出，将结果复制回临时数组，并将输出重新指向临时数组
    if temp_needed:
        temp[...] = output
        output = temp
    
    # 返回输出数组
    return output
    as kernel over the image and computing the sum of products at each location.

    >>> from scipy.ndimage import correlate
    >>> import numpy as np
    >>> input_img = np.arange(25).reshape(5,5)
    >>> print(input_img)
    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]]

    Define a kernel (weights) for correlation. In this example, it is for sum of
    center and up, down, left and right next elements.

    >>> weights = [[0, 1, 0],
    ...            [1, 1, 1],
    ...            [0, 1, 0]]

    We can calculate a correlation result:
    For example, element ``[2,2]`` is ``7 + 11 + 12 + 13 + 17 = 60``.

    >>> correlate(input_img, weights)
    array([[  6,  10,  15,  20,  24],
           [ 26,  30,  35,  40,  44],
           [ 51,  55,  60,  65,  69],
           [ 76,  80,  85,  90,  94],
           [ 96, 100, 105, 110, 114]])

    """
    # 调用 _correlate_or_convolve 函数执行相关操作，返回结果
    return _correlate_or_convolve(input, weights, output, mode, cval,
                                  origin, False)
# 定义一个装饰器函数，用于自动填充文档字符串中的参数
@_ni_docstrings.docfiller
# 定义多维卷积函数
def convolve(input, weights, output=None, mode='reflect', cval=0.0,
             origin=0):
    """
    Multidimensional convolution.

    The array is convolved with the given kernel.

    Parameters
    ----------
    %(input)s
        Input array to be convolved.
    weights : array_like
        Array of weights, same number of dimensions as input.
    %(output)s
        Optional output array to store the result of convolution.
    %(mode_reflect)s
        Controls how the input is extended beyond its boundaries.

    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : int or sequence, optional
        Controls the placement of the filter on the input array's pixels.
        A value of 0 (the default) centers the filter over the pixel, with
        positive values shifting the filter to the right, and negative ones
        to the left. By passing a sequence of origins with length equal to
        the number of dimensions of the input array, different shifts can
        be specified along each axis.

    Returns
    -------
    result : ndarray
        The result of convolution of `input` with `weights`.

    See Also
    --------
    correlate : Correlate an image with a kernel.

    Notes
    -----
    Each value in result is :math:`C_i = \\sum_j{I_{i+k-j} W_j}`, where
    W is the `weights` kernel,
    j is the N-D spatial index over :math:`W`,
    I is the `input` and k is the coordinate of the center of
    W, specified by `origin` in the input parameters.

    Examples
    --------
    Example 1: Using `mode='constant', cval=0.0`
    >>> import numpy as np
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> k = np.array([[1,1,1],[1,1,0],[1,0,0]])
    >>> from scipy import ndimage
    >>> ndimage.convolve(a, k, mode='constant', cval=0.0)
    array([[11, 10,  7,  4],
           [10,  3, 11, 11],
           [15, 12, 14,  7],
           [12,  3,  7,  0]])

    Example 2: Setting `cval=1.0`
    >>> ndimage.convolve(a, k, mode='constant', cval=1.0)
    array([[13, 11,  8,  7],
           [11,  3, 11, 14],
           [16, 12, 14, 10],
           [15,  6, 10,  5]])

    Example 3: Using `mode='reflect'`
    >>> b = np.array([[2, 0, 0],
    ...               [1, 0, 0],
    ...               [0, 0, 0]])
    >>> k = np.array([[0,1,0], [0,1,0], [0,1,0]])
    >>> ndimage.convolve(b, k, mode='reflect')
    array([[5, 0, 0],
           [3, 0, 0],
           [1, 0, 0]])

    Example 4: Diagonal convolution with `mode='reflect'`
    >>> k = np.array([[1,0,0],[0,1,0],[0,0,1]])
    >>> ndimage.convolve(b, k)
    """
    # 输入数组，表示要进行卷积或相关操作的原始数据
    array([[4, 2, 0],
           [3, 2, 0],
           [1, 1, 0]])
    
    # 使用 'nearest' 模式时，将输入数据边缘最接近的单个值重复使用，直到与重叠的权重匹配
    With ``mode='nearest'``, the single nearest value in to an edge in
    `input` is repeated as many times as needed to match the overlapping
    `weights`.
    
    # 示例：定义一个二维数组 c
    >>> c = np.array([[2, 0, 1],
    ...               [1, 0, 0],
    ...               [0, 0, 0]])
    # 示例：定义一个二维卷积核 k
    >>> k = np.array([[0, 1, 0],
    ...               [0, 1, 0],
    ...               [0, 1, 0],
    ...               [0, 1, 0],
    ...               [0, 1, 0]])
    # 对数组 c 使用 ndimage 库中的 convolve 函数进行 'nearest' 模式的卷积操作
    >>> ndimage.convolve(c, k, mode='nearest')
    array([[7, 0, 3],
           [5, 0, 2],
           [3, 0, 1]])
    
    # 返回 _correlate_or_convolve 函数的调用结果，执行卷积或相关操作
    """
        return _correlate_or_convolve(input, weights, output, mode, cval,
                                      origin, True)
# 使用文档字符串填充装饰器来为函数添加文档
@_ni_docstrings.docfiller
# 定义一个一维均匀滤波器函数
def uniform_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    """Calculate a 1-D uniform filter along the given axis.

    The lines of the array along the given axis are filtered with a
    uniform filter of given size.

    Parameters
    ----------
    %(input)s
        Input array to be filtered.
    size : int
        Length of the uniform filter.
    %(axis)s
        Axis along which to apply the filter.
    %(output)s
        Output array where the filtered result is placed.
    %(mode_reflect)s
        How edges of input are treated.
    %(cval)s
        Value to fill past edges of input.
    %(origin)s
        Displacement from the origin in the filter.

    Returns
    -------
    result : ndarray
        Filtered array. Has the same shape as `input`.

    Examples
    --------
    >>> from scipy.ndimage import uniform_filter1d
    >>> uniform_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
    array([4, 3, 4, 1, 4, 6, 6, 3])
    """
    # 将输入转换为 ndarray 类型
    input = np.asarray(input)
    # 标准化轴的索引
    axis = normalize_axis_index(axis, input.ndim)
    # 如果滤波器大小小于 1，则引发运行时错误
    if size < 1:
        raise RuntimeError('incorrect filter size')
    # 检查是否是复数输出
    complex_output = input.dtype.kind == 'c'
    # 获取输出数组
    output = _ni_support._get_output(output, input,
                                     complex_output=complex_output)
    # 检查原点位置的有效性
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError('invalid origin')
    # 将模式扩展为代码
    mode = _ni_support._extend_mode_to_code(mode)
    # 如果不是复数输出，则调用一维均匀滤波器函数
    if not complex_output:
        _nd_image.uniform_filter1d(input, size, axis, output, mode, cval,
                                   origin)
    else:
        # 如果是复数输出，则分别对实部和虚部调用一维均匀滤波器函数
        _nd_image.uniform_filter1d(input.real, size, axis, output.real, mode,
                                   np.real(cval), origin)
        _nd_image.uniform_filter1d(input.imag, size, axis, output.imag, mode,
                                   np.imag(cval), origin)
    # 返回输出数组
    return output


# 使用文档字符串填充装饰器来为函数添加文档
@_ni_docstrings.docfiller
# 定义一个多维均匀滤波器函数
def uniform_filter(input, size=3, output=None, mode="reflect",
                   cval=0.0, origin=0, *, axes=None):
    """Multidimensional uniform filter.

    Parameters
    ----------
    %(input)s
        Input array to be filtered.
    size : int or sequence of ints, optional
        The sizes of the uniform filter are given for each axis as a
        sequence, or as a single number, in which case the size is
        equal for all axes.
    %(output)s
        Output array where the filtered result is placed.
    %(mode_multiple)s
        How edges of input are treated for each axis.
    %(cval)s
        Value to fill past edges of input.
    %(origin_multiple)s
        Displacement from the origin in the filter for each axis.
    axes : tuple of int or None, optional
        Axes along which to apply the filter. If None, filter is applied
        along all axes. Otherwise, filter is applied along specified axes.
        Length of `axes` must match lengths of `size`, `origin`, and `mode`
        if they are sequences.

    Returns
    -------
    uniform_filter : ndarray
        Filtered array. Has the same shape as `input`.

    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    1-D uniform filters. The intermediate arrays are stored
    in the same data type as the output. Therefore, for output types
    with a limited precision, the results may be imprecise because
    of accumulation of rounding errors in intermediate steps.
    """
    """
    将输入数据转换为 NumPy 数组
    """
    input = np.asarray(input)
    
    """
    根据输入的数据类型是否为复数，获取输出数组
    """
    output = _ni_support._get_output(output, input,
                                     complex_output=input.dtype.kind == 'c')
    
    """
    检查并标准化轴参数，以匹配输入数据的维度
    """
    axes = _ni_support._check_axes(axes, input.ndim)
    
    """
    确定需要应用滤波的轴的数量
    """
    num_axes = len(axes)
    
    """
    将滤波器大小标准化为序列
    """
    sizes = _ni_support._normalize_sequence(size, num_axes)
    
    """
    将起始点标准化为序列
    """
    origins = _ni_support._normalize_sequence(origin, num_axes)
    
    """
    将模式标准化为序列
    """
    modes = _ni_support._normalize_sequence(mode, num_axes)
    
    """
    创建包含轴、大小、起始点和模式的列表
    """
    axes = [(axes[ii], sizes[ii], origins[ii], modes[ii])
            for ii in range(num_axes) if sizes[ii] > 1]
    
    """
    如果存在需要应用滤波的轴，则依次对其应用一维均匀滤波器
    """
    if len(axes) > 0:
        for axis, size, origin, mode in axes:
            uniform_filter1d(input, int(size), axis, output, mode,
                             cval, origin)
            input = output
    
    """
    如果没有需要应用滤波的轴，则直接将输出数组赋值为输入数组
    """
    else:
        output[...] = input[...]
    
    """
    返回滤波后的输出数组
    """
    return output
# 使用装饰器填充文档字符串，为 1-D 最小值滤波器函数添加文档说明
@_ni_docstrings.docfiller
def minimum_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    """Calculate a 1-D minimum filter along the given axis.

    The lines of the array along the given axis are filtered with a
    minimum filter of given size.

    Parameters
    ----------
    %(input)s
        Input array to filter.
    size : int
        Length along which to calculate 1D minimum.
    %(axis)s
        Axis along which to apply the filter.
    %(output)s
        Optional output array.
    %(mode_reflect)s
        How edges are handled.
    %(cval)s
        Value to fill past edges of input.
    %(origin)s
        Displacement from the center element.

    Returns
    -------
    result : ndarray
        Filtered image. Has the same shape as `input`.

    Notes
    -----
    This function implements the MINLIST algorithm [1]_, as described by
    Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
    the `input` length, regardless of filter size.

    References
    ----------
    .. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
    .. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html

    Examples
    --------
    >>> from scipy.ndimage import minimum_filter1d
    >>> minimum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
    array([2, 0, 0, 0, 1, 1, 0, 0])
    """
    # 将输入转换为 ndarray
    input = np.asarray(input)
    # 检查输入数组是否为复数类型，如果是则抛出异常
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    # 规范化轴索引，确保在有效范围内
    axis = normalize_axis_index(axis, input.ndim)
    # 检查滤波器尺寸是否合理，如果小于1则抛出运行时错误
    if size < 1:
        raise RuntimeError('incorrect filter size')
    # 获取输出数组，如果未提供则创建一个新数组
    output = _ni_support._get_output(output, input)
    # 检查原点位置是否有效，如果不在合理范围内则抛出值错误
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError('invalid origin')
    # 将模式扩展为对应的代码
    mode = _ni_support._extend_mode_to_code(mode)
    # 调用底层的 min_or_max_filter1d 函数执行最小值滤波
    _nd_image.min_or_max_filter1d(input, size, axis, output, mode, cval,
                                  origin, 1)
    # 返回滤波后的结果数组
    return output


# 使用装饰器填充文档字符串，为 1-D 最大值滤波器函数添加文档说明
@_ni_docstrings.docfiller
def maximum_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    """Calculate a 1-D maximum filter along the given axis.

    The lines of the array along the given axis are filtered with a
    maximum filter of given size.

    Parameters
    ----------
    %(input)s
        Input array to filter.
    size : int
        Length along which to calculate the 1-D maximum.
    %(axis)s
        Axis along which to apply the filter.
    %(output)s
        Optional output array.
    %(mode_reflect)s
        How edges are handled.
    %(cval)s
        Value to fill past edges of input.
    %(origin)s
        Displacement from the center element.

    Returns
    -------
    maximum1d : ndarray, None
        Maximum-filtered array with same shape as input.
        None if `output` is not None

    Notes
    -----
    This function implements the MAXLIST algorithm [1]_, as described by
    Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
    the `input` length, regardless of filter size.

    References
    ----------
    .. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
    .. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html

    Examples
    --------
    >>> from scipy.ndimage import maximum_filter1d
    >>> maximum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
    array([8, 8, 8, 4, 9, 9, 9, 9])
    """
    # 将输入数据转换为 NumPy 数组
    input = np.asarray(input)
    # 检查输入数据是否包含复数类型，如果是则抛出类型错误异常
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    # 根据给定的轴参数，规范化轴的索引值，确保其在输入数据维度范围内
    axis = normalize_axis_index(axis, input.ndim)
    # 检查滤波器大小是否小于1，如果是则抛出运行时错误异常
    if size < 1:
        raise RuntimeError('incorrect filter size')
    # 获取输出数组，确保其与输入数据的形状和类型兼容
    output = _ni_support._get_output(output, input)
    # 检查滤波器原点的位置是否在有效范围内，如果不在则抛出值错误异常
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError('invalid origin')
    # 将扩展模式转换为对应的代码
    mode = _ni_support._extend_mode_to_code(mode)
    # 执行一维最小值或最大值滤波操作，并将结果存储在输出数组中
    _nd_image.min_or_max_filter1d(input, size, axis, output, mode, cval,
                                  origin, 0)
    # 返回滤波处理后的输出数组
    return output
# 定义一个函数，用于执行最小或最大值滤波操作
def _min_or_max_filter(input, size, footprint, structure, output, mode,
                       cval, origin, minimum, axes=None):
    # 如果同时设置了 size 和 footprint，则发出警告并忽略 size 参数
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=3)
    
    # 如果未提供 structure 参数，则根据 footprint 或 size 参数判断滤波方式
    if structure is None:
        # 如果 footprint 也未提供，则检查 size 是否为空，如果为空则抛出运行时错误
        if footprint is None:
            if size is None:
                raise RuntimeError("no footprint provided")
            # 如果提供了 size 参数但未提供 footprint 参数，则默认使用可分离滤波
            separable = True
        else:
            # 将 footprint 转换为布尔类型的 NumPy 数组
            footprint = np.asarray(footprint, dtype=bool)
            # 如果 footprint 全为零，则抛出值错误
            if not footprint.any():
                raise ValueError("All-zero footprint is not supported.")
            # 如果 footprint 全为一，则使用其形状作为 size，同时将 footprint 设置为 None
            if footprint.all():
                size = footprint.shape
                footprint = None
                separable = True
            else:
                separable = False
    else:
        # 将 structure 转换为双精度浮点类型的 NumPy 数组
        structure = np.asarray(structure, dtype=np.float64)
        separable = False
        # 如果未提供 footprint 参数，则使用结构的形状创建全一的 footprint
        if footprint is None:
            footprint = np.ones(structure.shape, bool)
        else:
            # 否则将 footprint 转换为布尔类型的 NumPy 数组
            footprint = np.asarray(footprint, dtype=bool)
    
    # 将输入数据转换为 NumPy 数组
    input = np.asarray(input)
    # 如果输入数据是复数类型，则抛出类型错误
    if np.iscomplexobj(input):
        raise TypeError("Complex type not supported")
    
    # 确定输出数组，并确保不与输入数组共享内存
    output = _ni_support._get_output(output, input)
    temp_needed = np.may_share_memory(input, output)
    if temp_needed:
        # 如果输入和输出数组共享内存，则创建临时输出数组
        temp = output
        output = _ni_support._get_output(output.dtype, input)
    
    # 检查滤波操作的轴参数，确保其合法性
    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    
    # 如果使用可分离滤波，则规范化 origin、size 和 mode 参数
    if separable:
        origins = _ni_support._normalize_sequence(origin, num_axes)
        sizes = _ni_support._normalize_sequence(size, num_axes)
        modes = _ni_support._normalize_sequence(mode, num_axes)
        
        # 创建包含轴、大小、起点和模式的元组列表
        axes = [(axes[ii], sizes[ii], origins[ii], modes[ii])
                for ii in range(len(axes)) if sizes[ii] > 1]
        
        # 根据 minimum 参数选择最小值或最大值滤波函数
        if minimum:
            filter_ = minimum_filter1d
        else:
            filter_ = maximum_filter1d
        
        # 如果存在要处理的轴，则逐一进行滤波操作
        if len(axes) > 0:
            for axis, size, origin, mode in axes:
                filter_(input, int(size), axis, output, mode, cval, origin)
                input = output
        else:
            # 如果没有要处理的轴，则直接复制输入到输出
            output[...] = input[...]
    else:
        # 将 origin 规范化为与输入数组维数相同的序列
        origins = _ni_support._normalize_sequence(origin, num_axes)
        # 如果过滤的轴数少于输入数组的维数
        if num_axes < input.ndim:
            # 如果脚印数组的维数不等于过滤后的轴数，引发运行时错误
            if footprint.ndim != num_axes:
                raise RuntimeError("footprint array has incorrect shape")
            # 在非过滤的轴上扩展脚印数组，设置未过滤轴的 origin 为 0
            footprint = np.expand_dims(
                footprint,
                tuple(ax for ax in range(input.ndim) if ax not in axes)
            )
            # 临时存储未过滤轴的 origin 值，初始化为全为 0
            origins_temp = [0,] * input.ndim
            # 将实际 origin 值替换到对应的轴上
            for o, ax in zip(origins, axes):
                origins_temp[ax] = o
            origins = origins_temp

        # 获取脚印数组的有效形状
        fshape = [ii for ii in footprint.shape if ii > 0]
        # 如果脚印数组的有效形状与输入数组的维数不一致，引发运行时错误
        if len(fshape) != input.ndim:
            raise RuntimeError('footprint array has incorrect shape.')
        # 检查每个轴的 origin 是否在有效范围内
        for origin, lenf in zip(origins, fshape):
            if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
                raise ValueError("invalid origin")
        # 如果脚印数组不是连续的，复制一份以确保连续性
        if not footprint.flags.contiguous:
            footprint = footprint.copy()
        # 如果提供了结构数组
        if structure is not None:
            # 如果结构数组的维数与输入数组的维数不同，引发运行时错误
            if len(structure.shape) != input.ndim:
                raise RuntimeError("structure array has incorrect shape")
            # 如果结构数组的维数与过滤后的轴数不同，扩展结构数组
            if num_axes != structure.ndim:
                structure = np.expand_dims(
                    structure,
                    tuple(ax for ax in range(structure.ndim) if ax not in axes)
                )
            # 如果结构数组不是连续的，复制一份以确保连续性
            if not structure.flags.contiguous:
                structure = structure.copy()
        # 如果 mode 不是字符串且是可迭代对象，引发运行时错误
        if not isinstance(mode, str) and isinstance(mode, Iterable):
            raise RuntimeError(
                "A sequence of modes is not supported for non-separable "
                "footprints")
        # 将 mode 扩展为相应的代码
        mode = _ni_support._extend_mode_to_code(mode)
        # 调用 _nd_image.min_or_max_filter 函数执行最小值或最大值过滤操作
        _nd_image.min_or_max_filter(input, footprint, structure, output,
                                    mode, cval, origins, minimum)
    # 如果需要临时存储
    if temp_needed:
        # 将 output 的内容复制到临时数组 temp 中
        temp[...] = output
        # 将 temp 赋值给 output
        output = temp
    # 返回输出结果
    return output
# 定义一个用于计算多维最小值滤波的函数
@_ni_docstrings.docfiller
def minimum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0, *, axes=None):
    """Calculate a multidimensional minimum filter.

    Parameters
    ----------
    %(input)s
        输入数组，需要进行滤波操作的多维数组。
    %(size_foot)s
        滤波器的大小或足印（footprint），控制滤波器的形状和尺寸。
    %(output)s
        可选参数，用于指定输出数组。
    %(mode_multiple)s
        字符串，用于指定处理边界的模式。
    %(cval)s
        浮点数，边界模式为常数时使用的常数值。
    %(origin_multiple)s
        整数，定义滤波器的原点位置。
    axes : tuple of int or None, optional
        如果为 None，则沿着所有轴进行滤波。否则，只沿着指定的轴进行滤波。
        当指定了 axes 参数时，用于 size、origin 和/或 mode 的元组必须与 axes 的长度相匹配。
        这些元组中的第 i 个条目对应于 axes 中的第 i 个条目。

    Returns
    -------
    minimum_filter : ndarray
        过滤后的数组，与输入数组具有相同的形状。

    Notes
    -----
    当足印（footprint）是可分离的时，才支持轴的模式序列（每个轴一个模式）。否则，必须提供单个模式字符串。

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # 以灰度显示滤波结果
    >>> ax1 = fig.add_subplot(121)  # 左侧
    >>> ax2 = fig.add_subplot(122)  # 右侧
    >>> ascent = datasets.ascent()
    >>> result = ndimage.minimum_filter(ascent, size=20)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    # 调用 _min_or_max_filter 函数，执行最小值滤波操作
    return _min_or_max_filter(input, size, footprint, None, output, mode,
                              cval, origin, 1, axes)


# 定义一个用于计算多维最大值滤波的函数
@_ni_docstrings.docfiller
def maximum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0, *, axes=None):
    """Calculate a multidimensional maximum filter.

    Parameters
    ----------
    %(input)s
        输入数组，需要进行滤波操作的多维数组。
    %(size_foot)s
        滤波器的大小或足印（footprint），控制滤波器的形状和尺寸。
    %(output)s
        可选参数，用于指定输出数组。
    %(mode_multiple)s
        字符串，用于指定处理边界的模式。
    %(cval)s
        浮点数，边界模式为常数时使用的常数值。
    %(origin_multiple)s
        整数，定义滤波器的原点位置。
    axes : tuple of int or None, optional
        如果为 None，则沿着所有轴进行滤波。否则，只沿着指定的轴进行滤波。
        当指定了 axes 参数时，用于 size、origin 和/或 mode 的元组必须与 axes 的长度相匹配。
        这些元组中的第 i 个条目对应于 axes 中的第 i 个条目。

    Returns
    -------
    maximum_filter : ndarray
        过滤后的数组，与输入数组具有相同的形状。

    Notes
    -----
    当足印（footprint）是可分离的时，才支持轴的模式序列（每个轴一个模式）。否则，必须提供单个模式字符串。

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # 以灰度显示滤波结果
    >>> ax1 = fig.add_subplot(121)  # 左侧
    >>> ax2 = fig.add_subplot(122)  # 右侧
    >>> ascent = datasets.ascent()
    >>> result = ndimage.maximum_filter(ascent, size=20)
    """
    # 在 ax1 上显示名为 ascent 的图像
    ax1.imshow(ascent)
    # 在 ax2 上显示名为 result 的图像
    ax2.imshow(result)
    # 显示图形界面，展示 ax1 和 ax2 上的图像
    plt.show()
    """
    调用 _min_or_max_filter 函数，并返回其结果。
    该函数用于执行最小或最大滤波操作，根据传入的参数进行处理。
    参数解释：
    - input: 输入数据，进行滤波操作的对象
    - size: 滤波器的大小
    - footprint: 滤波器的形状
    - output: 输出数组，可选参数
    - mode: 边界处理模式
    - cval: 填充值
    - origin: 滤波器的原点
    - 0: 操作类型，0表示最小滤波，1表示最大滤波
    - axes: 操作轴
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode,
                              cval, origin, 0, axes)
@_ni_docstrings.docfiller
# 使用特定的文档字符串填充函数装饰器，可能用于生成文档或注释
def _rank_filter(input, rank, size=None, footprint=None, output=None,
                 mode="reflect", cval=0.0, origin=0, operation='rank',
                 axes=None):
    # 如果同时设置了 size 和 footprint，则发出警告并忽略 size 参数
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=3)
    # 将输入数据转换为 NumPy 数组
    input = np.asarray(input)
    # 如果输入数据是复杂类型，则抛出类型错误异常
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    # 检查并规范化 axes 参数，确保其与输入数据维度相匹配
    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    # 根据输入的 origin 参数，规范化 origins 数组
    origins = _ni_support._normalize_sequence(origin, num_axes)
    # 如果未提供 footprint，则根据 size 创建一个全为 True 的布尔数组作为 footprint
    if footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _ni_support._normalize_sequence(size, num_axes)
        footprint = np.ones(sizes, dtype=bool)
    else:
        # 将 footprint 转换为布尔数组
        footprint = np.asarray(footprint, dtype=bool)
    
    # 如果未被过滤的轴数小于输入数据的总轴数，则进行以下处理
    if num_axes < input.ndim:
        # 对于未被过滤的轴，将 origin 设为 0
        origins_temp = [0,] * input.ndim
        for o, ax in zip(origins, axes):
            origins_temp[ax] = o
        origins = origins_temp
        
        # 如果 mode 不是字符串而是可迭代对象，则设置未被过滤的轴的 mode 为 'constant'
        if not isinstance(mode, str) and isinstance(mode, Iterable):
            modes = _ni_support._normalize_sequence(mode, num_axes)
            modes_temp = ['constant'] * input.ndim
            for m, ax in zip(modes, axes):
                modes_temp[ax] = m
            mode = modes_temp
        
        # 在未被过滤的轴上插入单例维度
        if footprint.ndim != num_axes:
            raise RuntimeError("footprint array has incorrect shape")
        footprint = np.expand_dims(
            footprint,
            tuple(ax for ax in range(input.ndim) if ax not in axes)
        )
    
    # 检查 footprint 的形状是否正确
    fshape = [ii for ii in footprint.shape if ii > 0]
    if len(fshape) != input.ndim:
        raise RuntimeError('footprint array has incorrect shape.')
    
    # 检查 origin 参数的有效性
    for origin, lenf in zip(origins, fshape):
        if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
            raise ValueError('invalid origin')
    
    # 如果 footprint 不是连续的，则复制一份连续的 footprint 数组
    if not footprint.flags.contiguous:
        footprint = footprint.copy()
    
    # 计算 filter_size，即 footprint 中为 True 的元素个数
    filter_size = np.where(footprint, 1, 0).sum()
    
    # 根据操作类型调整 rank 的值
    if operation == 'median':
        rank = filter_size // 2
    elif operation == 'percentile':
        percentile = rank
        if percentile < 0.0:
            percentile += 100.0
        if percentile < 0 or percentile > 100:
            raise RuntimeError('invalid percentile')
        if percentile == 100.0:
            rank = filter_size - 1
        else:
            rank = int(float(filter_size) * percentile / 100.0)
    
    # 确保 rank 的值在有效范围内
    if rank < 0:
        rank += filter_size
    if rank < 0 or rank >= filter_size:
        raise RuntimeError('rank not within filter footprint size')
    
    # 如果 rank 为 0，则调用 minimum_filter 函数进行最小值滤波
    if rank == 0:
        return minimum_filter(input, None, footprint, output, mode, cval,
                              origins, axes=None)
    # 如果 rank 等于 filter_size - 1，则调用 maximum_filter 函数
    elif rank == filter_size - 1:
        return maximum_filter(input, None, footprint, output, mode, cval,
                              origins, axes=None)
    # 否则执行以下操作
    else:
        # 根据输入数组和输出数组获取输出数组
        output = _ni_support._get_output(output, input)
        # 检查输入数组和输出数组是否共享内存
        temp_needed = np.may_share_memory(input, output)
        if temp_needed:
            # 如果共享内存，则将 output 赋值给 temp
            temp = output
            # 根据输入数组的数据类型获取新的输出数组
            output = _ni_support._get_output(output.dtype, input)
        # 如果 mode 不是字符串且是可迭代对象，则抛出 RuntimeError
        if not isinstance(mode, str) and isinstance(mode, Iterable):
            raise RuntimeError(
                "A sequence of modes is not supported by non-separable rank "
                "filters")
        # 将 mode 扩展为对应的模式代码
        mode = _ni_support._extend_mode_to_code(mode)
        # 执行 _nd_image.rank_filter 函数，对输入数组进行秩为 rank 的过滤
        _nd_image.rank_filter(input, rank, footprint, output, mode, cval,
                              origins)
        if temp_needed:
            # 如果之前共享内存，将计算结果复制回 temp
            temp[...] = output
            # 将输出数组重新赋值为 temp
            output = temp
        # 返回处理后的输出数组
        return output
# 填充文档字符串的装饰器，用于填充函数签名和参数说明
@_ni_docstrings.docfiller
# 计算多维排名滤波器
def rank_filter(input, rank, size=None, footprint=None, output=None,
                mode="reflect", cval=0.0, origin=0, *, axes=None):
    """Calculate a multidimensional rank filter.

    Parameters
    ----------
    %(input)s
        输入数组。
    rank : int
        排名参数，可以是负数，例如，rank = -1 表示最大的元素。
    %(size_foot)s
        滤波器的尺寸或足迹。
    %(output)s
        输出数组，用于保存结果。
    %(mode_reflect)s
        用于处理边界的模式。
    %(cval)s
        当模式需要常数值填充时使用的常数值。
    %(origin_multiple)s
        滤波器操作的起点。
    axes : tuple of int or None, optional
        如果为 None，则沿着所有轴进行滤波。否则，沿指定的轴进行滤波。

    Returns
    -------
    rank_filter : ndarray
        滤波后的数组，与输入数组具有相同的形状。

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # 在灰度中显示滤波结果
    >>> ax1 = fig.add_subplot(121)  # 左侧
    >>> ax2 = fig.add_subplot(122)  # 右侧
    >>> ascent = datasets.ascent()
    >>> result = ndimage.rank_filter(ascent, rank=42, size=20)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    # 将 rank 参数转换为整数索引
    rank = operator.index(rank)
    # 调用内部的 _rank_filter 函数执行滤波操作
    return _rank_filter(input, rank, size, footprint, output, mode, cval,
                        origin, 'rank', axes=axes)


# 填充文档字符串的装饰器，用于填充函数签名和参数说明
@_ni_docstrings.docfiller
# 计算多维中值滤波器
def median_filter(input, size=None, footprint=None, output=None,
                  mode="reflect", cval=0.0, origin=0, *, axes=None):
    """
    Calculate a multidimensional median filter.

    Parameters
    ----------
    %(input)s
        输入数组。
    %(size_foot)s
        滤波器的尺寸或足迹。
    %(output)s
        输出数组，用于保存结果。
    %(mode_reflect)s
        用于处理边界的模式。
    %(cval)s
        当模式需要常数值填充时使用的常数值。
    %(origin_multiple)s
        滤波器操作的起点。
    axes : tuple of int or None, optional
        如果为 None，则沿着所有轴进行滤波。否则，沿指定的轴进行滤波。

    Returns
    -------
    median_filter : ndarray
        滤波后的数组，与输入数组具有相同的形状。

    See Also
    --------
    scipy.signal.medfilt2d
        相关的二维中值滤波器函数。

    Notes
    -----
    对于二维图像，当数据类型为 `uint8`、`float32` 或 `float64` 时，
    使用专门的 `scipy.signal.medfilt2d` 函数可能会更快。然而，它只支持
    常数模式且 `cval=0`。

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # 在灰度中显示滤波结果
    >>> ax1 = fig.add_subplot(121)  # 左侧
    >>> ax2 = fig.add_subplot(122)  # 右侧
    >>> ascent = datasets.ascent()
    >>> result = ndimage.median_filter(ascent, size=20)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    # 调用内部的 _rank_filter 函数执行中值滤波操作，rank 参数为 0
    return _rank_filter(input, 0, size, footprint, output, mode, cval,
                        origin, 'median', axes=axes)
# 定义一个多维百分位数滤波器函数
def percentile_filter(input, percentile, size=None, footprint=None,
                      output=None, mode="reflect", cval=0.0, origin=0, *,
                      axes=None):
    """Calculate a multidimensional percentile filter.

    Parameters
    ----------
    %(input)s
        输入数组，可以是任意维度的。
    percentile : scalar
        百分位数参数，可以小于零，例如，percentile=-20 相当于 percentile=80。
    %(size_foot)s
        可选参数，指定滤波器的大小或足迹（footprint）。
    %(output)s
        可选参数，指定输出数组的位置。
    %(mode_reflect)s
        可选参数，指定在滤波过程中边界处理的模式。
    %(cval)s
        可选参数，用于边界填充时的常数值。
    %(origin_multiple)s
        可选参数，指定原点在滤波器中的位置。
    axes : tuple of int or None, optional
        如果为 None，则沿着所有轴进行滤波。否则，沿着指定的轴进行滤波。

    Returns
    -------
    percentile_filter : ndarray
        过滤后的数组，与输入数组 `input` 具有相同的形状。

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = ndimage.percentile_filter(ascent, percentile=20, size=20)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    # 调用内部函数 `_rank_filter` 执行百分位数滤波
    return _rank_filter(input, percentile, size, footprint, output, mode,
                        cval, origin, 'percentile', axes=axes)


@_ni_docstrings.docfiller
def generic_filter1d(input, function, filter_size, axis=-1,
                     output=None, mode="reflect", cval=0.0, origin=0,
                     extra_arguments=(), extra_keywords=None):
    """Calculate a 1-D filter along the given axis.

    `generic_filter1d` iterates over the lines of the array, calling the
    given function at each line. The arguments of the line are the
    input line, and the output line. The input and output lines are 1-D
    double arrays. The input line is extended appropriately according
    to the filter size and origin. The output line must be modified
    in-place with the result.

    Parameters
    ----------
    %(input)s
        输入数组，可以是任意维度的。
    function : {callable, scipy.LowLevelCallable}
        要在给定轴上应用的函数。
    filter_size : scalar
        滤波器的长度。
    %(axis)s
        可选参数，指定应用滤波器的轴。
    %(output)s
        可选参数，指定输出数组的位置。
    %(mode_reflect)s
        可选参数，指定在滤波过程中边界处理的模式。
    %(cval)s
        可选参数，用于边界填充时的常数值。
    %(origin)s
        可选参数，指定原点在滤波器中的位置。
    %(extra_arguments)s
        可选参数，额外传递给函数的参数元组。
    %(extra_keywords)s
        可选参数，额外传递给函数的关键字参数。

    Returns
    -------
    generic_filter1d : ndarray
        过滤后的数组，与输入数组 `input` 具有相同的形状。

    Notes
    -----
    该函数还接受具有以下签名之一的低级回调函数，并封装在 `scipy.LowLevelCallable` 中：

    .. code:: c

       int function(double *input_line, npy_intp input_length,
                    double *output_line, npy_intp output_length,
                    void *user_data)
       int function(double *input_line, intptr_t input_length,
                    double *output_line, intptr_t output_length,
                    void *user_data)
    """
    pass  # 这里是函数体，因为是函数定义，所以暂时不需要添加具体注释
    """
    如果 `extra_keywords` 为 None，则设为一个空字典
    """
    if extra_keywords is None:
        extra_keywords = {}
    
    """
    将 `input` 转换为 NumPy 数组
    """
    input = np.asarray(input)
    
    """
    如果 `input` 是复数类型，则抛出类型错误异常
    """
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    
    """
    根据 `input` 数组的形状获取输出数组 `output`
    """
    output = _ni_support._get_output(output, input)
    
    """
    如果 `filter_size` 小于 1，则抛出运行时错误异常
    """
    if filter_size < 1:
        raise RuntimeError('invalid filter size')
    
    """
    根据 `axis` 参数对输入的轴进行标准化处理
    """
    axis = normalize_axis_index(axis, input.ndim)
    
    """
    检查滤波器原点 `origin` 是否在有效范围内
    """
    if (filter_size // 2 + origin < 0) or (filter_size // 2 + origin >= filter_size):
        raise ValueError('invalid origin')
    
    """
    将扩展模式 `mode` 转换为对应的代码
    """
    mode = _ni_support._extend_mode_to_code(mode)
    
    """
    调用 `_nd_image.generic_filter1d` 函数进行一维通用滤波处理
    """
    _nd_image.generic_filter1d(input, function, filter_size, axis, output,
                               mode, cval, origin, extra_arguments,
                               extra_keywords)
    
    """
    返回经过滤波处理后的输出数组 `output`
    """
    return output
# 使用文档字符串填充函数的装饰器，这个装饰器通常用于添加文档字符串到函数中
@_ni_docstrings.docfiller
# 定义一个通用的多维过滤器函数，可以使用给定的函数计算过滤后的结果
def generic_filter(input, function, size=None, footprint=None,
                   output=None, mode="reflect", cval=0.0, origin=0,
                   extra_arguments=(), extra_keywords=None):
    """Calculate a multidimensional filter using the given function.

    At each element the provided function is called. The input values
    within the filter footprint at that element are passed to the function
    as a 1-D array of double values.

    Parameters
    ----------
    %(input)s
        输入数组，作为过滤器的输入。
    function : {callable, scipy.LowLevelCallable}
        在每个元素上调用的函数。
    %(size_foot)s
        指定过滤器的大小或脚印。如果指定了大小，则使用大小参数；如果指定了脚印，则使用脚印参数。
    %(output)s
        可选参数，用于存储过滤后的结果的数组。
    %(mode_reflect)s
        控制输入数组在边界外的行为。默认为"reflect"。
    %(cval)s
        当使用"constant"模式时，用于填充边界外的常数值。
    %(origin_multiple)s
        控制过滤器起始位置的整数或整数元组。
    %(extra_arguments)s
        用于传递给函数的额外参数元组。
    %(extra_keywords)s
        用于传递给函数的额外关键字参数。

    Returns
    -------
    generic_filter : ndarray
        过滤后的数组，与输入数组具有相同的形状。

    Notes
    -----
    此函数还接受具有以下签名之一的低级回调函数，并包装在`scipy.LowLevelCallable`中：

    .. code:: c

       int callback(double *buffer, npy_intp filter_size,
                    double *return_value, void *user_data)
       int callback(double *buffer, intptr_t filter_size,
                    double *return_value, void *user_data)

    调用函数遍历输入和输出数组的元素，在每个元素处调用回调函数。当前元素的过滤器脚印中的元素通过`buffer`参数传递，
    过滤器脚印中的元素数量通过`filter_size`传递。计算得到的值存储在`return_value`中。
    `user_data`是作为`scipy.LowLevelCallable`提供的数据指针。

    回调函数必须返回整数错误状态，如果出现问题则为零，否则为一。如果发生错误，通常在返回之前应设置Python错误状态并提供信息性消息，
    否则调用函数将设置默认错误消息。

    此外，还接受一些其他低级函数指针规范，但这些仅用于向后兼容，不应在新代码中使用。

    Examples
    --------
    导入必要的模块并加载用于过滤的示例图像。

    >>> import numpy as np
    >>> from scipy import datasets
    >>> from scipy.ndimage import zoom, generic_filter
    >>> import matplotlib.pyplot as plt
    >>> ascent = zoom(datasets.ascent(), 0.5)

    使用简单的NumPy聚合函数作为`function`参数计算内核大小为5的最大过滤器。

    >>> maximum_filter_result = generic_filter(ascent, np.amax, [5, 5])

    虽然也可以直接使用`maximum_filter`获得最大过滤器结果，但`generic_filter`允许使用通用的Python函数或`scipy.LowLevelCallable`作为过滤器。
    在这里，我们计算的是
    """
    # 如果同时提供了 size 和 footprint 参数，则发出警告并忽略 size 参数，因为 footprint 参数已设置
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=2)
    
    # 如果 extra_keywords 参数为 None，则将其设为空字典
    if extra_keywords is None:
        extra_keywords = {}
    
    # 将输入数据转换为 NumPy 数组
    input = np.asarray(input)
    
    # 如果输入数据是复数类型，则抛出 TypeError
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    
    # 根据输入数据的维度，将 origin 参数规范化成序列
    origins = _ni_support._normalize_sequence(origin, input.ndim)
    
    # 如果 footprint 参数为 None，则根据 size 参数创建默认的 footprint
    if footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _ni_support._normalize_sequence(size, input.ndim)
        footprint = np.ones(sizes, dtype=bool)
    else:
        footprint = np.asarray(footprint, dtype=bool)
    
    # 确保 footprint 是一个连续的数组
    if not footprint.flags.contiguous:
        footprint = footprint.copy()
    
    # 根据输入数据的维度，获取输出数组
    output = _ni_support._get_output(output, input)
    
    # 将 mode 参数转换为对应的数值代码
    mode = _ni_support._extend_mode_to_code(mode)
    
    # 调用底层的 generic_filter 函数进行通用滤波操作
    _nd_image.generic_filter(input, function, footprint, output, mode,
                             cval, origins, extra_arguments, extra_keywords)
    
    # 返回滤波后的输出数组
    return output
```