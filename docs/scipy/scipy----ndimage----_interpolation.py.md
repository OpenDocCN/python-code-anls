# `D:\src\scipysrc\scipy\scipy\ndimage\_interpolation.py`

```
# 导入必要的模块和函数
import itertools  # 提供用于迭代操作的工具函数
import warnings   # 用于处理警告信息

import numpy as np  # 导入NumPy库，用于数值计算
from scipy._lib._util import normalize_axis_index  # 导入辅助函数，用于规范化轴索引

from scipy import special  # 导入SciPy的特殊函数模块
from . import _ni_support  # 导入局部模块_ni_support
from . import _nd_image  # 导入局部模块_nd_image
from ._ni_docstrings import docfiller  # 导入局部模块_ni_docstrings中的docfiller函数

__all__ = ['spline_filter1d', 'spline_filter', 'geometric_transform',
           'map_coordinates', 'affine_transform', 'shift', 'zoom', 'rotate']

@docfiller
def spline_filter1d(input, order=3, axis=-1, output=np.float64,
                    mode='mirror'):
    """
    Calculate a 1-D spline filter along the given axis.

    The lines of the array along the given axis are filtered by a
    spline filter. The order of the spline must be >= 2 and <= 5.

    Parameters
    ----------
    %(input)s
        Input array to be filtered.
    order : int, optional
        The order of the spline, default is 3.
    axis : int, optional
        The axis along which the spline filter is applied. Default is the last
        axis.
    output : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array. Default is ``numpy.float64``.
    %(mode_interp_mirror)s
        Mode used to handle values outside the array borders.

    Returns
    -------
    spline_filter1d : ndarray
        The filtered input array.

    See Also
    --------
    spline_filter : Multidimensional spline filter.

    Notes
    -----
    All of the interpolation functions in `ndimage` do spline interpolation of
    the input image. If using B-splines of `order > 1`, the input image
    values have to be converted to B-spline coefficients first, which is
    handled by this function.
    """
    """
    Apply a 1-D spline filter to the input array `input`.
    
    Parameters
    ----------
    input : array_like
        The input array to filter.
    order : int
        The order of the spline filter. Must be between 0 and 5.
    axis : int
        The axis along which to apply the filter.
    output : ndarray, optional
        The array to store the output. If not provided, a new array is created.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode used to determine how to handle boundaries.
    
    Returns
    -------
    output : ndarray
        The filtered output array.
    
    Raises
    ------
    RuntimeError
        If spline order is not supported (order < 0 or order > 5).
    
    Notes
    -----
    The spline filter is applied sequentially along all axes of the input.
    Functions requiring B-spline coefficients will filter their inputs
    automatically, controlled by the `prefilter` keyword argument. The
    correctness of the result for functions with a `mode` parameter depends
    on matching the `mode` used during filtering.
    
    For complex-valued `input`, the function processes real and imaginary
    components independently.
    
    .. versionadded:: 1.6.0
        Support for complex-valued inputs.
    
    Examples
    --------
    Filtering an image using 1-D spline filters along specified axes:
    
    >>> from scipy.ndimage import spline_filter1d
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> orig_img = np.eye(20)  # create an image
    >>> orig_img[10, :] = 1.0
    >>> sp_filter_axis_0 = spline_filter1d(orig_img, axis=0)
    >>> sp_filter_axis_1 = spline_filter1d(orig_img, axis=1)
    >>> f, ax = plt.subplots(1, 3, sharex=True)
    >>> for ind, data in enumerate([[orig_img, "original image"],
    ...             [sp_filter_axis_0, "spline filter (axis=0)"],
    ...             [sp_filter_axis_1, "spline filter (axis=1)"]]):
    ...     ax[ind].imshow(data[0], cmap='gray_r')
    ...     ax[ind].set_title(data[1])
    >>> plt.tight_layout()
    >>> plt.show()
    
    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    
    # Convert input to numpy array
    input = np.asarray(input)
    
    # Check if input is complex-valued
    complex_output = np.iscomplexobj(input)
    
    # Determine output array
    output = _ni_support._get_output(output, input,
                                     complex_output=complex_output)
    
    # If input is complex, process real and imaginary parts separately
    if complex_output:
        spline_filter1d(input.real, order, axis, output.real, mode)
        spline_filter1d(input.imag, order, axis, output.imag, mode)
        return output
    
    # If order is 0 or 1, directly copy input to output
    if order in [0, 1]:
        output[...] = np.array(input)
    else:
        # Convert mode to its corresponding code
        mode = _ni_support._extend_mode_to_code(mode)
        # Normalize axis index
        axis = normalize_axis_index(axis, input.ndim)
        # Apply spline filter along the specified axis
        _nd_image.spline_filter1d(input, order, axis, output, mode)
    
    # Return the filtered output
    return output
@docfiller
def spline_filter(input, order=3, output=np.float64, mode='mirror'):
    """
    Multidimensional spline filter.

    Parameters
    ----------
    %(input)s
    order : int, optional
        The order of the spline, default is 3.
    output : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array. Default is ``numpy.float64``.
    %(mode_interp_mirror)s

    Returns
    -------
    spline_filter : ndarray
        Filtered array. Has the same shape as `input`.

    See Also
    --------
    spline_filter1d : Calculate a 1-D spline filter along the given axis.

    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    1-D spline filters. The intermediate arrays are stored
    in the same data type as the output. Therefore, for output types
    with a limited precision, the results may be imprecise because
    intermediate results may be stored with insufficient precision.

    For complex-valued `input`, this function processes the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    We can filter an image using multidimentional splines:

    >>> from scipy.ndimage import spline_filter
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> orig_img = np.eye(20)  # create an image
    >>> orig_img[10, :] = 1.0
    >>> sp_filter = spline_filter(orig_img, order=3)
    >>> f, ax = plt.subplots(1, 2, sharex=True)
    >>> for ind, data in enumerate([[orig_img, "original image"],
    ...                             [sp_filter, "spline filter"]]):
    ...     ax[ind].imshow(data[0], cmap='gray_r')
    ...     ax[ind].set_title(data[1])
    >>> plt.tight_layout()
    >>> plt.show()

    """
    # 如果阶数小于2或大于5，则抛出运行时错误
    if order < 2 or order > 5:
        raise RuntimeError('spline order not supported')
    # 将输入转换为 ndarray 格式
    input = np.asarray(input)
    # 检查输入是否为复数对象
    complex_output = np.iscomplexobj(input)
    # 根据输入类型获取输出数组或数据类型
    output = _ni_support._get_output(output, input,
                                     complex_output=complex_output)
    # 如果输入为复数，则分别处理实部和虚部
    if complex_output:
        spline_filter(input.real, order, output.real, mode)
        spline_filter(input.imag, order, output.imag, mode)
        return output
    # 如果阶数不在[0, 1]之间且输入维度大于0
    if order not in [0, 1] and input.ndim > 0:
        # 对每个轴进行 1-D 样条滤波
        for axis in range(input.ndim):
            spline_filter1d(input, order, axis, output=output, mode=mode)
            input = output
    else:
        # 直接将输出数组设为输入数组
        output[...] = input[...]
    return output


def _prepad_for_spline_filter(input, mode, cval):
    # 如果模式为 'nearest' 或 'grid-constant'
    if mode in ['nearest', 'grid-constant']:
        # 设定填充数量为 12
        npad = 12
        # 根据模式选择填充方式
        if mode == 'grid-constant':
            padded = np.pad(input, npad, mode='constant',
                               constant_values=cval)
        elif mode == 'nearest':
            padded = np.pad(input, npad, mode='edge')
    else:
        # 对于其他模式，已经实现了确切的边界条件，
        # 因此不需要预填充
        npad = 0
        # 如果不需要填充，则直接使用输入作为填充后的数据
        padded = input
    # 返回填充后的数据和填充的数量
    return padded, npad
# 定义函数 geometric_transform，实现任意几何变换操作
@docfiller
def geometric_transform(input, mapping, output_shape=None,
                        output=None, order=3,
                        mode='constant', cval=0.0, prefilter=True,
                        extra_arguments=(), extra_keywords={}):
    """
    Apply an arbitrary geometric transform.

    The given mapping function is used to find, for each point in the
    output, the corresponding coordinates in the input. The value of the
    input at those coordinates is determined by spline interpolation of
    the requested order.

    Parameters
    ----------
    %(input)s
        输入数组，可以是任意维度的输入数据。
    mapping : {callable, scipy.LowLevelCallable}
        一个可调用对象，接受一个长度等于输出数组秩的元组，并返回输入数组秩长度的对应输入坐标元组。
    output_shape : tuple of ints, optional
        输出数组的形状元组。
    %(output)s
        可选的输出数组。
    order : int, optional
        样条插值的阶数，默认为3。
        阶数必须在0到5之间。
    %(mode_interp_constant)s
        插值时使用的模式，默认为'constant'。
    %(cval)s
        用于填充超出边界的常数值，默认为0.0。
    %(prefilter)s
        是否对输入进行预过滤，默认为True。
    extra_arguments : tuple, optional
        传递给 `mapping` 的额外参数。
    extra_keywords : dict, optional
        传递给 `mapping` 的额外关键字参数。

    Returns
    -------
    output : ndarray
        经过变换后的输出数组。

    See Also
    --------
    map_coordinates, affine_transform, spline_filter1d


    Notes
    -----
    This function also accepts low-level callback functions with one
    the following signatures and wrapped in `scipy.LowLevelCallable`:

    .. code:: c

       int mapping(npy_intp *output_coordinates, double *input_coordinates,
                   int output_rank, int input_rank, void *user_data)
       int mapping(intptr_t *output_coordinates, double *input_coordinates,
                   int output_rank, int input_rank, void *user_data)

    The calling function iterates over the elements of the output array,
    calling the callback function at each element. The coordinates of the
    current output element are passed through ``output_coordinates``. The
    callback function must return the coordinates at which the input must
    be interpolated in ``input_coordinates``. The rank of the input and
    output arrays are given by ``input_rank`` and ``output_rank``
    respectively. ``user_data`` is the data pointer provided
    to `scipy.LowLevelCallable` as-is.

    The callback function must return an integer error status that is zero
    if something went wrong and one otherwise. If an error occurs, you should
    normally set the Python error status with an informative message
    before returning, otherwise a default error message is set by the
    calling function.

    In addition, some other low-level function pointer specifications
    are accepted, but these are for backward compatibility only and should
    not be used in new code.
    """
    """
    For complex-valued `input`, this function transforms the real and imaginary
    components independently.
    
    .. versionadded:: 1.6.0
        Complex-valued support added.
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.ndimage import geometric_transform
    >>> a = np.arange(12.).reshape((4, 3))
    >>> def shift_func(output_coords):
    ...     return (output_coords[0] - 0.5, output_coords[1] - 0.5)
    ...
    >>> geometric_transform(a, shift_func)
    array([[ 0.   ,  0.   ,  0.   ],
           [ 0.   ,  1.362,  2.738],
           [ 0.   ,  4.812,  6.187],
           [ 0.   ,  8.263,  9.637]])
    
    >>> b = [1, 2, 3, 4, 5]
    >>> def shift_func(output_coords):
    ...     return (output_coords[0] - 3,)
    ...
    >>> geometric_transform(b, shift_func, mode='constant')
    array([0, 0, 0, 1, 2])
    >>> geometric_transform(b, shift_func, mode='nearest')
    array([1, 1, 1, 1, 2])
    >>> geometric_transform(b, shift_func, mode='reflect')
    array([3, 2, 1, 1, 2])
    >>> geometric_transform(b, shift_func, mode='wrap')
    array([2, 3, 4, 1, 2])
    
    """
    # 如果次数(order)小于0或大于5，则抛出运行时错误
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    
    # 将输入转换为 NumPy 数组
    input = np.asarray(input)
    
    # 如果输出形状未指定，则设为与输入相同的形状
    if output_shape is None:
        output_shape = input.shape
    
    # 如果输入的维度小于1或输出形状的长度小于1，则抛出运行时错误
    if input.ndim < 1 or len(output_shape) < 1:
        raise RuntimeError('input and output rank must be > 0')
    
    # 检查输入是否为复数对象
    complex_output = np.iscomplexobj(input)
    
    # 获取输出数组，用于存储变换后的结果
    output = _ni_support._get_output(output, input, shape=output_shape,
                                     complex_output=complex_output)
    
    # 如果输入为复数，分别对实部和虚部进行几何变换
    if complex_output:
        kwargs = dict(order=order, mode=mode, prefilter=prefilter,
                      output_shape=output_shape,
                      extra_arguments=extra_arguments,
                      extra_keywords=extra_keywords)
        geometric_transform(input.real, mapping, output=output.real,
                            cval=np.real(cval), **kwargs)
        geometric_transform(input.imag, mapping, output=output.imag,
                            cval=np.imag(cval), **kwargs)
        return output
    
    # 如果需要预滤波并且次数大于1，则对输入进行样条滤波
    if prefilter and order > 1:
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        filtered = spline_filter(padded, order, output=np.float64,
                                 mode=mode)
    else:
        npad = 0
        filtered = input
    
    # 将模式转换为对应的代码
    mode = _ni_support._extend_mode_to_code(mode)
    
    # 使用几何变换函数对过滤后的数据进行几何变换，并存储到输出数组中
    _nd_image.geometric_transform(filtered, mapping, None, None, None, output,
                                  order, mode, cval, npad, extra_arguments,
                                  extra_keywords)
    
    # 返回几何变换后的输出数组
    return output
# 定义函数 map_coordinates，将输入数组映射到新的坐标上进行插值操作
@docfiller
def map_coordinates(input, coordinates, output=None, order=3,
                    mode='constant', cval=0.0, prefilter=True):
    """
    Map the input array to new coordinates by interpolation.

    The array of coordinates is used to find, for each point in the output,
    the corresponding coordinates in the input. The value of the input at
    those coordinates is determined by spline interpolation of the
    requested order.

    The shape of the output is derived from that of the coordinate
    array by dropping the first axis. The values of the array along
    the first axis are the coordinates in the input array at which the
    output value is found.

    Parameters
    ----------
    %(input)s
        输入的数组，用于插值操作。
    coordinates : array_like
        要评估 `input` 的坐标。
    %(output)s
        可选，用于存储结果的数组，形状从 `coordinates` 的形状推导而来。
    order : int, optional
        插值的阶数，默认为 3。
        阶数必须在 0-5 范围内。
    %(mode_interp_constant)s
        插值时的模式，如 'constant'、'nearest' 等。
    %(cval)s
        当 mode 为 'constant' 时，使用的常数值。
    %(prefilter)s
        是否对 `input` 进行预滤波处理。

    Returns
    -------
    map_coordinates : ndarray
        变换输入的结果。输出的形状由 `coordinates` 的形状推导而来，去除第一个轴。

    See Also
    --------
    spline_filter, geometric_transform, scipy.interpolate

    Notes
    -----
    For complex-valued `input`, this function maps the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.arange(12.).reshape((4, 3))
    >>> a
    array([[  0.,   1.,   2.],
           [  3.,   4.,   5.],
           [  6.,   7.,   8.],
           [  9.,  10.,  11.]])
    >>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=1)
    array([ 2.,  7.])

    Above, the interpolated value of a[0.5, 0.5] gives output[0], while
    a[2, 1] is output[1].

    >>> inds = np.array([[0.5, 2], [0.5, 4]])
    >>> ndimage.map_coordinates(a, inds, order=1, cval=-33.3)
    array([  2. , -33.3])
    >>> ndimage.map_coordinates(a, inds, order=1, mode='nearest')
    array([ 2.,  8.])
    >>> ndimage.map_coordinates(a, inds, order=1, cval=0, output=bool)
    array([ True, False], dtype=bool)

    """
    # 如果插值阶数不在支持范围内，抛出运行时错误
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    # 将输入转换为 NumPy 数组
    input = np.asarray(input)
    # 将坐标转换为 NumPy 数组
    coordinates = np.asarray(coordinates)
    # 如果坐标是复数类型的，则抛出类型错误
    if np.iscomplexobj(coordinates):
        raise TypeError('Complex type not supported')
    # 推导输出形状
    output_shape = coordinates.shape[1:]
    # 检查输入和输出的秩必须大于 0
    if input.ndim < 1 or len(output_shape) < 1:
        raise RuntimeError('input and output rank must be > 0')
    # 检查坐标数组的第一个维度必须与输入数组的秩相同
    if coordinates.shape[0] != input.ndim:
        raise RuntimeError('invalid shape for coordinate array')
    # 检查输入是否为复数对象
    complex_output = np.iscomplexobj(input)
    # 获取输出数组，根据输入和指定的形状以及是否复数对象来确定
    output = _ni_support._get_output(output, input, shape=output_shape,
                                     complex_output=complex_output)
    # 如果需要复杂输出，则准备关键字参数字典
    if complex_output:
        kwargs = dict(order=order, mode=mode, prefilter=prefilter)
        # 对实部进行坐标映射操作，将结果存储在输出的实部中
        map_coordinates(input.real, coordinates, output=output.real,
                        cval=np.real(cval), **kwargs)
        # 对虚部进行坐标映射操作，将结果存储在输出的虚部中
        map_coordinates(input.imag, coordinates, output=output.imag,
                        cval=np.imag(cval), **kwargs)
        # 返回复杂输出结果
        return output
    
    # 如果需要预过滤且阶数大于1，则进行填充和样条滤波操作
    if prefilter and order > 1:
        # 根据模式和指定的填充值预填充输入数据
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        # 对预填充后的数据进行样条滤波，并将结果存储在指定的输出中
        filtered = spline_filter(padded, order, output=np.float64, mode=mode)
    else:
        # 否则，不进行填充，直接使用原始输入数据进行处理
        npad = 0
        filtered = input
    
    # 将处理模式扩展为对应的代码
    mode = _ni_support._extend_mode_to_code(mode)
    # 执行几何变换，将处理后的数据存储在指定的输出中
    _nd_image.geometric_transform(filtered, None, coordinates, None, None,
                                  output, order, mode, cval, npad, None, None)
    # 返回处理后的输出数据
    return output
@docfiller
def affine_transform(input, matrix, offset=0.0, output_shape=None,
                     output=None, order=3,
                     mode='constant', cval=0.0, prefilter=True):
    """
    Apply an affine transformation.

    Given an output image pixel index vector ``o``, the pixel value
    is determined from the input image at position
    ``np.dot(matrix, o) + offset``.

    This does 'pull' (or 'backward') resampling, transforming the output space
    to the input to locate data. Affine transformations are often described in
    the 'push' (or 'forward') direction, transforming input to output. If you
    have a matrix for the 'push' transformation, use its inverse
    (:func:`numpy.linalg.inv`) in this function.

    Parameters
    ----------
    %(input)s
    matrix : ndarray
        The inverse coordinate transformation matrix, mapping output
        coordinates to input coordinates. If ``ndim`` is the number of
        dimensions of ``input``, the given matrix must have one of the
        following shapes:

            - ``(ndim, ndim)``: the linear transformation matrix for each
              output coordinate.
            - ``(ndim,)``: assume that the 2-D transformation matrix is
              diagonal, with the diagonal specified by the given value. A more
              efficient algorithm is then used that exploits the separability
              of the problem.
            - ``(ndim + 1, ndim + 1)``: assume that the transformation is
              specified using homogeneous coordinates [1]_. In this case, any
              value passed to ``offset`` is ignored.
            - ``(ndim, ndim + 1)``: as above, but the bottom row of a
              homogeneous transformation matrix is always ``[0, 0, ..., 1]``,
              and may be omitted.

    offset : float or sequence, optional
        The offset into the array where the transform is applied. If a float,
        `offset` is the same for each axis. If a sequence, `offset` should
        contain one value for each axis.
    output_shape : tuple of ints, optional
        Shape tuple.
    %(output)s
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
    %(cval)s
    %(prefilter)s

    Returns
    -------
    affine_transform : ndarray
        The transformed input.

    Notes
    -----
    The given matrix and offset are used to find for each point in the
    output the corresponding coordinates in the input by an affine
    transformation. The value of the input at those coordinates is
    determined by spline interpolation of the requested order. Points
    outside the boundaries of the input are filled according to the given
    mode.
    """
    """
    .. versionchanged:: 0.18.0
        Previously, the exact interpretation of the affine transformation
        depended on whether the matrix was supplied as a 1-D or a
        2-D array. If a 1-D array was supplied
        to the matrix parameter, the output pixel value at index ``o``
        was determined from the input image at position
        ``matrix * (o + offset)``.
    """

    # 如果指定的阶数不在0到5之间，抛出运行时错误
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')

    # 将输入转换为 NumPy 数组
    input = np.asarray(input)

    # 如果未指定输出形状，则根据输出参数的类型确定形状或者使用输入形状
    if output_shape is None:
        if isinstance(output, np.ndarray):
            output_shape = output.shape
        else:
            output_shape = input.shape

    # 如果输入的维数小于1或者输出形状的长度小于1，抛出运行时错误
    if input.ndim < 1 or len(output_shape) < 1:
        raise RuntimeError('input and output rank must be > 0')

    # 检查输入是否为复数对象
    complex_output = np.iscomplexobj(input)

    # 获取输出数组，根据输入和复数输出条件进行调整
    output = _ni_support._get_output(output, input, shape=output_shape,
                                     complex_output=complex_output)

    # 如果输入为复数，分别对实部和虚部进行仿射变换，并返回输出数组
    if complex_output:
        kwargs = dict(offset=offset, output_shape=output_shape, order=order,
                      mode=mode, prefilter=prefilter)
        affine_transform(input.real, matrix, output=output.real,
                         cval=np.real(cval), **kwargs)
        affine_transform(input.imag, matrix, output=output.imag,
                         cval=np.imag(cval), **kwargs)
        return output

    # 如果需要预过滤且阶数大于1，则对输入数据进行预填充并进行样条滤波
    if prefilter and order > 1:
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        filtered = spline_filter(padded, order, output=np.float64, mode=mode)
    else:
        npad = 0
        filtered = input

    # 将模式扩展为对应的代码
    mode = _ni_support._extend_mode_to_code(mode)

    # 将仿射矩阵转换为 NumPy 数组，确保其为浮点数类型
    matrix = np.asarray(matrix, dtype=np.float64)

    # 检查仿射矩阵的维度和形状是否符合要求
    if matrix.ndim not in [1, 2] or matrix.shape[0] < 1:
        raise RuntimeError('no proper affine matrix provided')

    # 如果仿射矩阵是二维的且形状符合输入维度加一的要求，则进行额外检查
    if (matrix.ndim == 2 and matrix.shape[1] == input.ndim + 1 and
            (matrix.shape[0] in [input.ndim, input.ndim + 1])):
        if matrix.shape[0] == input.ndim + 1:
            exptd = [0] * input.ndim + [1]
            if not np.all(matrix[input.ndim] == exptd):
                msg = (f'Expected homogeneous transformation matrix with '
                       f'shape {matrix.shape} for image shape {input.shape}, '
                       f'but bottom row was not equal to {exptd}')
                raise ValueError(msg)

        # 假设输入为齐次坐标变换矩阵，则获取偏移量和矩阵
        offset = matrix[:input.ndim, input.ndim]
        matrix = matrix[:input.ndim, :input.ndim]

    # 检查仿射矩阵的行数是否与输入的维度相匹配
    if matrix.shape[0] != input.ndim:
        raise RuntimeError('affine matrix has wrong number of rows')
    # 检查矩阵的维度是否为2，并且列数与输出的维度不匹配时，抛出运行时错误
    if matrix.ndim == 2 and matrix.shape[1] != output.ndim:
        raise RuntimeError('affine matrix has wrong number of columns')
    
    # 如果矩阵不是连续存储的，进行复制以确保连续性
    if not matrix.flags.contiguous:
        matrix = matrix.copy()
    
    # 规范化偏移量序列，使其与输入数据的维度相匹配
    offset = _ni_support._normalize_sequence(offset, input.ndim)
    
    # 将偏移量转换为numpy数组，并确保数据类型为np.float64
    offset = np.asarray(offset, dtype=np.float64)
    
    # 如果偏移量的维度不为1或者其长度小于1，则抛出运行时错误
    if offset.ndim != 1 or offset.shape[0] < 1:
        raise RuntimeError('no proper offset provided')
    
    # 如果偏移量不是连续存储的，进行复制以确保连续性
    if not offset.flags.contiguous:
        offset = offset.copy()
    
    # 如果矩阵的维度为1，发出警告提示用户affine_transform在SciPy 0.18.0中的行为已更改
    if matrix.ndim == 1:
        warnings.warn(
            "The behavior of affine_transform with a 1-D "
            "array supplied for the matrix parameter has changed in "
            "SciPy 0.18.0.",
            stacklevel=2
        )
        # 使用_nd_image.zoom_shift进行仿射变换，传入参数filtered, matrix, offset/matrix, output等
        _nd_image.zoom_shift(filtered, matrix, offset/matrix, output, order,
                             mode, cval, npad, False)
    else:
        # 使用_nd_image.geometric_transform进行几何变换，传入参数filtered, None, None, matrix, offset等
        _nd_image.geometric_transform(filtered, None, None, matrix, offset,
                                      output, order, mode, cval, npad, None,
                                      None)
    
    # 返回处理后的输出数据
    return output
@docfiller
def shift(input, shift, output=None, order=3, mode='constant', cval=0.0,
          prefilter=True):
    """
    Shift an array.

    The array is shifted using spline interpolation of the requested order.
    Points outside the boundaries of the input are filled according to the
    given mode.

    Parameters
    ----------
    %(input)s
        input : array_like
            The input array to be shifted.
    shift : float or sequence
        The shift along the axes. If a float, `shift` is the same for each
        axis. If a sequence, `shift` should contain one value for each axis.
    %(output)s
        output : ndarray, optional
            The array in which to place the output, or None if a new array
            should be allocated.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
        mode : str, optional
            Points outside the boundaries of the input are filled according
            to the given mode ('constant', 'nearest', 'reflect', or 'wrap').
    %(cval)s
        cval : scalar, optional
            Value to fill past edges of input if mode is 'constant'.
    %(prefilter)s
        prefilter : bool, optional
            Determines if the input array should be prefiltered with spline
            filter before interpolation. Default is True.

    Returns
    -------
    shift : ndarray
        The shifted input.

    See Also
    --------
    affine_transform : Affine transformations

    Notes
    -----
    For complex-valued `input`, this function shifts the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    Import the necessary modules and an exemplary image.

    >>> from scipy.ndimage import shift
    >>> import matplotlib.pyplot as plt
    >>> from scipy import datasets
    >>> image = datasets.ascent()

    Shift the image vertically by 20 pixels.

    >>> image_shifted_vertically = shift(image, (20, 0))

    Shift the image vertically by -200 pixels and horizontally by 100 pixels.

    >>> image_shifted_both_directions = shift(image, (-200, 100))

    Plot the original and the shifted images.

    >>> fig, axes = plt.subplots(3, 1, figsize=(4, 12))
    >>> plt.gray()  # show the filtered result in grayscale
    >>> top, middle, bottom = axes
    >>> for ax in axes:
    ...     ax.set_axis_off()  # remove coordinate system
    >>> top.imshow(image)
    >>> top.set_title("Original image")
    >>> middle.imshow(image_shifted_vertically)
    >>> middle.set_title("Vertically shifted image")
    >>> bottom.imshow(image_shifted_both_directions)
    >>> bottom.set_title("Image shifted in both directions")
    >>> fig.tight_layout()
    """
    # Check if the spline order is within the supported range
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    
    # Convert input to a NumPy array
    input = np.asarray(input)
    
    # Ensure input array has a valid rank
    if input.ndim < 1:
        raise RuntimeError('input and output rank must be > 0')
    
    # Check if input array is complex-valued
    complex_output = np.iscomplexobj(input)
    
    # Determine the output array shape and type
    output = _ni_support._get_output(output, input, complex_output=complex_output)
    
    # Handle complex-valued input separately
    if complex_output:
        # Import shift function under a different name to avoid conflict with parameter name
        from scipy.ndimage._interpolation import shift as _shift
        
        # Set keyword arguments for the shift function
        kwargs = dict(order=order, mode=mode, prefilter=prefilter)
        
        # Perform shift operation on real and imaginary parts separately
        _shift(input.real, shift, output=output.real, cval=np.real(cval), **kwargs)
        _shift(input.imag, shift, output=output.imag, cval=np.imag(cval), **kwargs)
        
        # Return the shifted complex-valued output
        return output
    # 如果预过滤器开启且插值次数大于1
    if prefilter and order > 1:
        # 对输入进行样条滤波前填充，并获取填充数量
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        # 对填充后的数据进行样条滤波
        filtered = spline_filter(padded, order, output=np.float64, mode=mode)
    else:
        # 否则，不进行填充
        npad = 0
        filtered = input
    # 将模式转换为扩展模式代码
    mode = _ni_support._extend_mode_to_code(mode)
    # 规范化平移序列，并将其转换为负数
    shift = _ni_support._normalize_sequence(shift, input.ndim)
    shift = [-ii for ii in shift]
    # 将平移序列转换为 np.float64 类型的数组
    shift = np.asarray(shift, dtype=np.float64)
    # 如果平移数组不是连续的，则复制一份
    if not shift.flags.contiguous:
        shift = shift.copy()
    # 执行图像的缩放和平移操作，将结果存入 output 中
    _nd_image.zoom_shift(filtered, None, shift, output, order, mode, cval,
                         npad, False)
    # 返回处理后的输出数据
    return output
# 定义一个函数用于对数组进行缩放处理，支持多种参数配置
@docfiller
def zoom(input, zoom, output=None, order=3, mode='constant', cval=0.0,
         prefilter=True, *, grid_mode=False):
    """
    Zoom an array.

    The array is zoomed using spline interpolation of the requested order.

    Parameters
    ----------
    %(input)s
        输入的数组或者序列。
    zoom : float or sequence
        沿各轴的缩放因子。如果是浮点数，各轴的缩放因子相同；如果是序列，则每个轴都可以设置一个缩放因子。
    %(output)s
        可选，输出数组。
    order : int, optional
        样条插值的阶数，默认为 3。
        阶数必须在 0 到 5 之间。
    %(mode_interp_constant)s
        插值模式，默认为 'constant'。
    %(cval)s
        在边界外的常数值，默认为 0.0。
    %(prefilter)s
        是否进行预滤波，默认为 True。
    grid_mode : bool, optional
        如果为 False，按照像素中心的距离进行缩放；否则，包括像素完整范围的距离。例如，长度为 5 的 1D 信号在 grid_mode 为 False 时被视为长度为 4，但在 grid_mode 为 True 时被视为长度为 5。参考以下示意图：

        .. code-block:: text

                | pixel 1 | pixel 2 | pixel 3 | pixel 4 | pixel 5 |
                     |<-------------------------------------->|
                                        vs.
                |<----------------------------------------------->|

        上图中箭头的起始点对应每种模式下的坐标位置 0。

    Returns
    -------
    zoom : ndarray
        缩放后的数组。

    Notes
    -----
    对于复数输入，此函数会独立缩放实部和虚部。

    .. versionadded:: 1.6.0
        添加了对复数的支持。

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt

    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)  # 左侧
    >>> ax2 = fig.add_subplot(122)  # 右侧
    >>> ascent = datasets.ascent()
    >>> result = ndimage.zoom(ascent, 3.0)
    >>> ax1.imshow(ascent, vmin=0, vmax=255)
    >>> ax2.imshow(result, vmin=0, vmax=255)
    >>> plt.show()

    >>> print(ascent.shape)
    (512, 512)

    >>> print(result.shape)
    (1536, 1536)
    """
    # 如果阶数不在支持的范围内，抛出运行时错误
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    # 将输入转换为 NumPy 数组
    input = np.asarray(input)
    # 如果输入数组的维度小于 1，则抛出运行时错误
    if input.ndim < 1:
        raise RuntimeError('input and output rank must be > 0')
    # 根据输入数组的维度，将缩放因子规范化为序列
    zoom = _ni_support._normalize_sequence(zoom, input.ndim)
    # 计算输出数组的形状，根据每个轴的大小和缩放因子
    output_shape = tuple(
            [int(round(ii * jj)) for ii, jj in zip(input.shape, zoom)])
    # 判断输入数组是否为复数类型
    complex_output = np.iscomplexobj(input)
    # 获取输出数组，根据输入数组和指定形状
    output = _ni_support._get_output(output, input, shape=output_shape,
                                     complex_output=complex_output)
    if complex_output:
        # 如果需要复杂输出，则导入 zoom 函数，避免与 zoom 参数混淆
        from scipy.ndimage._interpolation import zoom as _zoom

        # 准备参数字典，包括插值次数、模式和预过滤器
        kwargs = dict(order=order, mode=mode, prefilter=prefilter)
        # 对实部进行缩放操作，并将结果存入输出的实部，使用实部的实部值作为填充值
        _zoom(input.real, zoom, output=output.real, cval=np.real(cval), **kwargs)
        # 对虚部进行缩放操作，并将结果存入输出的虚部，使用虚部的虚部值作为填充值
        _zoom(input.imag, zoom, output=output.imag, cval=np.imag(cval), **kwargs)
        # 返回复数输出
        return output
    
    if prefilter and order > 1:
        # 如果需要预过滤且插值次数大于1，则进行预填充操作
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        # 对填充后的数据进行样条滤波，输出数据类型为 np.float64，使用指定的模式
        filtered = spline_filter(padded, order, output=np.float64, mode=mode)
    else:
        # 否则，不进行预填充，直接使用输入数据
        npad = 0
        filtered = input
    
    if grid_mode:
        # 如果启用了网格模式，警告某些可能具有意外行为的模式
        suggest_mode = None
        if mode == 'constant':
            suggest_mode = 'grid-constant'
        elif mode == 'wrap':
            suggest_mode = 'grid-wrap'
        # 如果存在建议的模式，则发出警告
        if suggest_mode is not None:
            warnings.warn(
                (f"It is recommended to use mode = {suggest_mode} instead of {mode} "
                 f"when grid_mode is True."),
                stacklevel=2
            )
    
    # 将模式转换为对应的代码
    mode = _ni_support._extend_mode_to_code(mode)

    # 计算缩放因子的分子和分母
    zoom_div = np.array(output_shape)
    zoom_nominator = np.array(input.shape)
    if not grid_mode:
        # 如果不是网格模式，则将分子和分母各减去1
        zoom_div -= 1
        zoom_nominator -= 1
    
    # 避免缩放到无限值，选择缩放因子为1
    zoom = np.divide(zoom_nominator, zoom_div,
                     out=np.ones_like(input.shape, dtype=np.float64),
                     where=zoom_div != 0)
    zoom = np.ascontiguousarray(zoom)
    # 调用 _nd_image.zoom_shift 函数进行缩放和平移操作
    _nd_image.zoom_shift(filtered, zoom, None, output, order, mode, cval, npad,
                         grid_mode)
    # 返回缩放后的输出
    return output
# 使用 @docfiller 装饰器填充函数文档字符串中的占位符
@docfiller
# 定义旋转函数，用于旋转数组
def rotate(input, angle, axes=(1, 0), reshape=True, output=None, order=3,
           mode='constant', cval=0.0, prefilter=True):
    """
    Rotate an array.

    The array is rotated in the plane defined by the two axes given by the
    `axes` parameter using spline interpolation of the requested order.

    Parameters
    ----------
    %(input)s
    angle : float
        The rotation angle in degrees.
    axes : tuple of 2 ints, optional
        The two axes that define the plane of rotation. Default is the first
        two axes.
    reshape : bool, optional
        If `reshape` is true, the output shape is adapted so that the input
        array is contained completely in the output. Default is True.
    %(output)s
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
    %(cval)s
    %(prefilter)s

    Returns
    -------
    rotate : ndarray
        The rotated input.

    Notes
    -----
    For complex-valued `input`, this function rotates the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure(figsize=(10, 3))
    >>> ax1, ax2, ax3 = fig.subplots(1, 3)
    >>> img = datasets.ascent()
    >>> img_45 = ndimage.rotate(img, 45, reshape=False)
    >>> full_img_45 = ndimage.rotate(img, 45, reshape=True)
    >>> ax1.imshow(img, cmap='gray')
    >>> ax1.set_axis_off()
    >>> ax2.imshow(img_45, cmap='gray')
    >>> ax2.set_axis_off()
    >>> ax3.imshow(full_img_45, cmap='gray')
    >>> ax3.set_axis_off()
    >>> fig.set_layout_engine('tight')
    >>> plt.show()
    >>> print(img.shape)
    (512, 512)
    >>> print(img_45.shape)
    (512, 512)
    >>> print(full_img_45.shape)
    (724, 724)

    """
    # 将输入转换为 NumPy 数组
    input_arr = np.asarray(input)
    # 获取输入数组的维度
    ndim = input_arr.ndim

    # 如果数组维度小于2，则抛出 ValueError 异常
    if ndim < 2:
        raise ValueError('input array should be at least 2D')

    # 将 axes 转换为列表
    axes = list(axes)

    # 如果 axes 长度不为2，则抛出 ValueError 异常
    if len(axes) != 2:
        raise ValueError('axes should contain exactly two values')

    # 检查 axes 是否包含整数值
    if not all([float(ax).is_integer() for ax in axes]):
        raise ValueError('axes should contain only integer values')

    # 处理负数索引，将其转换为非负索引
    if axes[0] < 0:
        axes[0] += ndim
    if axes[1] < 0:
        axes[1] += ndim

    # 如果转换后的 axes 仍有负数或者超出维度范围，则抛出 ValueError 异常
    if axes[0] < 0 or axes[1] < 0 or axes[0] >= ndim or axes[1] >= ndim:
        raise ValueError('invalid rotation plane specified')

    # 对 axes 进行排序
    axes.sort()

    # 计算旋转角度的余弦和正弦值
    c, s = special.cosdg(angle), special.sindg(angle)

    # 创建旋转矩阵
    rot_matrix = np.array([[c, s],
                           [-s, c]])

    # 获取输入数组的形状
    img_shape = np.asarray(input_arr.shape)
    # 获取输入数组在指定平面上的形状
    in_plane_shape = img_shape[axes]
    # 如果设置了 reshape 参数，则计算变换后的输入边界
    iy, ix = in_plane_shape  # 获取输入平面的形状（行数和列数）
    out_bounds = rot_matrix @ [[0, 0, iy, iy],  # 通过旋转矩阵计算变换后的边界框
                               [0, ix, 0, ix]]
    # 计算变换后输入平面的形状
    out_plane_shape = (np.ptp(out_bounds, axis=1) + 0.5).astype(int)

else:
    out_plane_shape = img_shape[axes]  # 如果未设置 reshape 参数，则直接使用给定轴上的图像形状

out_center = rot_matrix @ ((out_plane_shape - 1) / 2)  # 计算输出中心点的坐标
in_center = (in_plane_shape - 1) / 2  # 计算输入中心点的坐标
offset = in_center - out_center  # 计算中心点的偏移量

output_shape = img_shape  # 输出的图像形状与输入相同
output_shape[axes] = out_plane_shape  # 更新输出形状的指定轴上的值为变换后的平面形状
output_shape = tuple(output_shape)  # 转换成元组形式的输出形状

complex_output = np.iscomplexobj(input_arr)  # 检查输入数组是否为复数类型
output = _ni_support._get_output(output, input_arr, shape=output_shape,
                                 complex_output=complex_output)  # 获取输出数组，确保与输入类型和形状匹配

if ndim <= 2:
    # 如果数组的维度小于等于2，则直接应用仿射变换
    affine_transform(input_arr, rot_matrix, offset, output_shape, output,
                     order, mode, cval, prefilter)
else:
    # 如果数组的维度大于2，则在所有平面上并行应用旋转
    # 平行于指定轴的所有平面坐标组合
    planes_coord = itertools.product(
        *[[slice(None)] if ax in axes else range(img_shape[ax])
          for ax in range(ndim)])

    out_plane_shape = tuple(out_plane_shape)  # 转换成元组形式的输出平面形状

    for coordinates in planes_coord:
        ia = input_arr[coordinates]  # 获取输入数组的当前平面
        oa = output[coordinates]  # 获取输出数组的当前平面
        affine_transform(ia, rot_matrix, offset, out_plane_shape,
                         oa, order, mode, cval, prefilter)  # 应用仿射变换到当前平面

return output  # 返回应用仿射变换后的输出数组
```