# `D:\src\scipysrc\scipy\scipy\ndimage\_fourier.py`

```
# 导入NumPy库，用于科学计算
import numpy as np
# 导入SciPy库中的normalize_axis_index函数，用于规范化轴索引
from scipy._lib._util import normalize_axis_index
# 导入当前包中的_ni_support和_nd_image模块
from . import _ni_support
from . import _nd_image

# 定义公开接口，包括以下四个函数
__all__ = ['fourier_gaussian', 'fourier_uniform', 'fourier_ellipsoid',
           'fourier_shift']

# 定义一个内部函数，用于获取Fourier变换输出数组
def _get_output_fourier(output, input):
    # 如果输出数组未提供，则根据输入数组的数据类型创建相应类型的零数组
    if output is None:
        if input.dtype.type in [np.complex64, np.complex128, np.float32]:
            output = np.zeros(input.shape, dtype=input.dtype)
        else:
            output = np.zeros(input.shape, dtype=np.float64)
    # 如果输出类型为类型对象，则根据指定的类型创建相应类型的零数组
    elif type(output) is type:
        if output not in [np.complex64, np.complex128,
                          np.float32, np.float64]:
            raise RuntimeError("output type not supported")
        output = np.zeros(input.shape, dtype=output)
    # 如果输出数组的形状与输入数组不匹配，则抛出运行时异常
    elif output.shape != input.shape:
        raise RuntimeError("output shape not correct")
    return output

# 定义一个内部函数，用于获取复数域的Fourier变换输出数组
def _get_output_fourier_complex(output, input):
    # 如果输出数组未提供，则根据输入数组的复数类型创建相应类型的零数组
    if output is None:
        if input.dtype.type in [np.complex64, np.complex128]:
            output = np.zeros(input.shape, dtype=input.dtype)
        else:
            output = np.zeros(input.shape, dtype=np.complex128)
    # 如果输出类型为类型对象，则根据指定的复数类型创建相应类型的零数组
    elif type(output) is type:
        if output not in [np.complex64, np.complex128]:
            raise RuntimeError("output type not supported")
        output = np.zeros(input.shape, dtype=output)
    # 如果输出数组的形状与输入数组不匹配，则抛出运行时异常
    elif output.shape != input.shape:
        raise RuntimeError("output shape not correct")
    return output

# 定义Fourier高斯函数，用于对输入数组进行Fourier变换
def fourier_gaussian(input, sigma, n=-1, axis=-1, output=None):
    """
    # 将输入数组转换为 NumPy 数组
    input = np.asarray(input)
    
    # 根据输入数组和输出数组获取四维傅里叶变换的输出数组
    output = _get_output_fourier(output, input)
    
    # 根据给定的轴索引，规范化轴的值
    axis = normalize_axis_index(axis, input.ndim)
    
    # 根据输入数组的维度数，规范化 sigma 参数，使其成为一个标准化的序列
    sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
    
    # 将 sigmas 转换为 NumPy 数组，并指定数据类型为 np.float64
    sigmas = np.asarray(sigmas, dtype=np.float64)
    
    # 如果 sigmas 不是连续的数组，则进行复制以确保连续性
    if not sigmas.flags.contiguous:
        sigmas = sigmas.copy()

    # 使用 _nd_image 模块中的 fourier_filter 函数对输入数组进行 Fourier 滤波处理
    _nd_image.fourier_filter(input, sigmas, n, axis, output, 0)
    
    # 返回经过 Fourier 变换滤波处理后的输出数组
    return output
# 定义一个多维均匀傅里叶滤波器函数
def fourier_uniform(input, size, n=-1, axis=-1, output=None):
    """
    Multidimensional uniform fourier filter.

    The array is multiplied with the Fourier transform of a box of given
    size.

    Parameters
    ----------
    input : array_like
        The input array.
    size : float or sequence
        The size of the box used for filtering.
        If a float, `size` is the same for all axes. If a sequence, `size` has
        to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : ndarray, optional
        If given, the result of filtering the input is placed in this array.

    Returns
    -------
    fourier_uniform : ndarray
        The filtered input.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import numpy.fft
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = datasets.ascent()
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_uniform(input_, size=20)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """
    # 将输入转换为 ndarray 类型
    input = np.asarray(input)
    # 获取用于傅里叶变换结果的输出数组
    output = _get_output_fourier(output, input)
    # 规范化实数变换的轴索引
    axis = normalize_axis_index(axis, input.ndim)
    # 标准化给定的滤波器尺寸序列
    sizes = _ni_support._normalize_sequence(size, input.ndim)
    sizes = np.asarray(sizes, dtype=np.float64)
    # 如果尺寸数组不是连续的，复制一份
    if not sizes.flags.contiguous:
        sizes = sizes.copy()
    # 应用多维傅里叶滤波器
    _nd_image.fourier_filter(input, sizes, n, axis, output, 1)
    # 返回滤波后的输出数组
    return output


# 定义一个多维椭球傅里叶滤波器函数
def fourier_ellipsoid(input, size, n=-1, axis=-1, output=None):
    """
    Multidimensional ellipsoid Fourier filter.

    The array is multiplied with the fourier transform of an ellipsoid of
    given sizes.

    Parameters
    ----------
    input : array_like
        The input array.
    size : float or sequence
        The size of the box used for filtering.
        If a float, `size` is the same for all axes. If a sequence, `size` has
        to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : ndarray, optional
        If given, the result of filtering the input is placed in this array.
    """
    # 将输入转换为 NumPy 数组，确保输入的维度不超过3
    input = np.asarray(input)
    if input.ndim > 3:
        # 如果输入的维度超过3，则抛出异常
        raise NotImplementedError("Only 1d, 2d and 3d inputs are supported")
    
    # 根据输入和输出的情况获取输出数组
    output = _get_output_fourier(output, input)
    
    # 如果输出数组的大小为0，则直接返回空数组，避免可能导致的段错误
    if output.size == 0:
        return output
    
    # 标准化轴的索引，确保在输入数据的维度范围内
    axis = normalize_axis_index(axis, input.ndim)
    
    # 将指定的大小参数标准化为 NumPy 数组，确保其为浮点类型
    sizes = _ni_support._normalize_sequence(size, input.ndim)
    sizes = np.asarray(sizes, dtype=np.float64)
    
    # 如果 sizes 数组不是连续的，则复制一份以确保连续性
    if not sizes.flags.contiguous:
        sizes = sizes.copy()
    
    # 调用 C 扩展的 Fourier 滤波函数，对输入数据进行处理
    _nd_image.fourier_filter(input, sizes, n, axis, output, 2)
    
    # 返回经过 Fourier 滤波处理后的输出数组
    return output
# 导入必要的库和模块
def fourier_shift(input, shift, n=-1, axis=-1, output=None):
    """
    多维傅里叶位移滤波器。

    将数组与位移操作的傅里叶变换相乘。

    Parameters
    ----------
    input : array_like
        输入数组。
    shift : float or sequence
        用于滤波的盒子大小。
        如果是 float，对所有轴使用相同的 `shift`。如果是序列，`shift`
        必须包含每个轴的一个值。
    n : int, optional
        如果 `n` 是负数（默认），则假设输入是复数FFT的结果。
        如果 `n` 大于或等于零，则假设输入是实FFT的结果，并且 `n` 给出变换前数组的长度
        沿着实变换方向。
    axis : int, optional
        实变换的轴。
    output : ndarray, optional
        如果给定，则将输入的位移结果放入此数组中。

    Returns
    -------
    fourier_shift : ndarray
        位移后的输入。

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> import numpy.fft
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # 在灰度中显示滤波结果
    >>> ascent = datasets.ascent()
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_shift(input_, shift=200)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # 虚部是一个伪影
    >>> plt.show()
    """
    # 将输入转换为 numpy 数组
    input = np.asarray(input)
    # 获取复杂傅里叶变换的输出数组
    output = _get_output_fourier_complex(output, input)
    # 规范化实变换的轴索引
    axis = normalize_axis_index(axis, input.ndim)
    # 规范化位移序列
    shifts = _ni_support._normalize_sequence(shift, input.ndim)
    # 将位移序列转换为浮点数数组
    shifts = np.asarray(shifts, dtype=np.float64)
    # 如果位移数组不是连续的，则复制一份
    if not shifts.flags.contiguous:
        shifts = shifts.copy()
    # 应用傅里叶位移滤波器
    _nd_image.fourier_shift(input, shifts, n, axis, output)
    # 返回结果数组
    return output
```