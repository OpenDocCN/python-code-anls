# `D:\src\scipysrc\scipy\scipy\ndimage\_morphology.py`

```
# 警告：导入警告模块，用于处理可能出现的警告信息
import warnings
# 导入操作符模块，用于处理各种运算操作
import operator

# 导入 NumPy 库，用于科学计算
import numpy as np
# 导入局部模块，以下划线开头的模块，可能是指相同项目内的其他模块
from . import _ni_support
from . import _nd_image
from . import _filters

# 定义模块公开的函数和类名列表
__all__ = ['iterate_structure', 'generate_binary_structure', 'binary_erosion',
           'binary_dilation', 'binary_opening', 'binary_closing',
           'binary_hit_or_miss', 'binary_propagation', 'binary_fill_holes',
           'grey_erosion', 'grey_dilation', 'grey_opening', 'grey_closing',
           'morphological_gradient', 'morphological_laplace', 'white_tophat',
           'black_tophat', 'distance_transform_bf', 'distance_transform_cdt',
           'distance_transform_edt']

# 定义一个函数，用于判断结构元素中心是否为真值
def _center_is_true(structure, origin):
    # 将结构元素转换为 NumPy 数组
    structure = np.asarray(structure)
    # 计算结构元素中心坐标
    coor = tuple([oo + ss // 2 for ss, oo in zip(structure.shape,
                                                 origin)])
    # 返回结构元素中心坐标位置的布尔值
    return bool(structure[coor])

# 定义一个函数，用于迭代一个结构元素，通过将其与自身膨胀
def iterate_structure(structure, iterations, origin=None):
    """
    Iterate a structure by dilating it with itself.

    Parameters
    ----------
    structure : array_like
       Structuring element (an array of bools, for example), to be dilated with
       itself.
    iterations : int
       number of dilations performed on the structure with itself
    origin : optional
        If origin is None, only the iterated structure is returned. If
        not, a tuple of the iterated structure and the modified origin is
        returned.

    Returns
    -------
    """
    # 将输入的结构元素转换为 NumPy 数组
    structure = np.asarray(structure)
    
    # 如果迭代次数小于 2，直接返回结构元素的副本
    if iterations < 2:
        return structure.copy()
    
    # 计算迭代后输出数组的形状
    ni = iterations - 1
    shape = [ii + ni * (ii - 1) for ii in structure.shape]
    
    # 计算起始位置，使得结构元素位于输出数组中心
    pos = [ni * (structure.shape[ii] // 2) for ii in range(len(shape))]
    
    # 创建用于切片的元组，以将结构元素放置在输出数组的中心位置
    slc = tuple(slice(pos[ii], pos[ii] + structure.shape[ii], None)
                for ii in range(len(shape)))
    
    # 创建一个布尔类型的全零数组作为输出
    out = np.zeros(shape, bool)
    
    # 将结构元素的非零位置标记为真值，放置到输出数组的中心位置
    out[slc] = structure != 0
    
    # 对输出数组进行二值膨胀操作，使用给定的结构元素和迭代次数
    out = binary_dilation(out, structure, iterations=ni)
    
    # 如果未指定原点，直接返回输出数组
    if origin is None:
        return out
    else:
        # 标准化原点坐标序列，以与结构元素的维度相匹配
        origin = _ni_support._normalize_sequence(origin, structure.ndim)
        
        # 将原点坐标乘以迭代次数，以适应输出数组的尺寸
        origin = [iterations * o for o in origin]
        
        # 返回输出数组及其原点坐标
        return out, origin
# 定义生成二进制形态学操作结构的函数，用于形态学操作时的结构元素

def generate_binary_structure(rank, connectivity):
    """
    Generate a binary structure for binary morphological operations.

    Parameters
    ----------
    rank : int
         Number of dimensions of the array to which the structuring element
         will be applied, as returned by `np.ndim`.
    connectivity : int
         `connectivity` determines which elements of the output array belong
         to the structure, i.e., are considered as neighbors of the central
         element. Elements up to a squared distance of `connectivity` from
         the center are considered neighbors. `connectivity` may range from 1
         (no diagonal elements are neighbors) to `rank` (all elements are
         neighbors).

    Returns
    -------
    output : ndarray of bools
         Structuring element which may be used for binary morphological
         operations, with `rank` dimensions and all dimensions equal to 3.

    See Also
    --------
    iterate_structure, binary_dilation, binary_erosion

    Notes
    -----
    `generate_binary_structure` can only create structuring elements with
    dimensions equal to 3, i.e., minimal dimensions. For larger structuring
    elements, that are useful e.g., for eroding large objects, one may either
    use `iterate_structure`, or create directly custom arrays with
    numpy functions such as `numpy.ones`.

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> struct = ndimage.generate_binary_structure(2, 1)
    >>> struct
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> a = np.zeros((5,5))
    >>> a[2, 2] = 1
    >>> a
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> b = ndimage.binary_dilation(a, structure=struct).astype(a.dtype)
    >>> b
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(b, structure=struct).astype(a.dtype)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.]])
    >>> struct = ndimage.generate_binary_structure(2, 2)
    >>> struct
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    >>> struct = ndimage.generate_binary_structure(3, 1)
    >>> struct # no diagonal elements
    """
    # 创建一个3x3x3的布尔类型的多维数组，表示一个二维平面上的模式
    array([[[False, False, False],
            [False,  True, False],
            [False, False, False]],
           [[False,  True, False],
            [ True,  True,  True],
            [False,  True, False]],
           [[False, False, False],
            [False,  True, False],
            [False, False, False]]], dtype=bool)

    """
    # 如果连接数小于1，则将其设为1，确保连接数不为非正数
    if connectivity < 1:
        connectivity = 1
    # 如果秩小于1，则直接返回一个包含单个True值的布尔类型数组
    if rank < 1:
        return np.array(True, dtype=bool)
    # 创建一个以0为中心的以rank为维度的数组索引
    output = np.fabs(np.indices([3] * rank) - 1)
    # 沿着第一个轴（即索引轴）计算所有数组的和
    output = np.add.reduce(output, 0)
    # 返回一个布尔类型的数组，表示每个元素是否小于等于给定的连接数
    return output <= connectivity
# 将 iterations 参数转换为整数，如果无法转换则抛出类型错误
try:
    iterations = operator.index(iterations)
except TypeError as e:
    raise TypeError('iterations parameter should be an integer') from e

# 将输入数据转换为 NumPy 数组
input = np.asarray(input)

# 如果输入数据类型为复数类型，则抛出类型错误
if np.iscomplexobj(input):
    raise TypeError('Complex type not supported')

# 如果未提供结构参数，则使用输入数据的维度生成默认的二进制结构
if structure is None:
    structure = generate_binary_structure(input.ndim, 1)
else:
    # 否则，将结构参数转换为布尔型 NumPy 数组
    structure = np.asarray(structure, dtype=bool)

# 检查结构参数与输入数据维度是否一致，若不一致则引发运行时错误
if structure.ndim != input.ndim:
    raise RuntimeError('structure and input must have same dimensionality')

# 如果结构数组不是连续存储的，则复制一份
if not structure.flags.contiguous:
    structure = structure.copy()

# 如果结构数组大小小于 1，则引发运行时错误
if structure.size < 1:
    raise RuntimeError('structure must not be empty')

# 如果提供了掩码参数，则将其转换为 NumPy 数组，并确保与输入数据具有相同的形状
if mask is not None:
    mask = np.asarray(mask)
    if mask.shape != input.shape:
        raise RuntimeError('mask and input must have equal sizes')

# 规范化原点参数，确保其与输入数据的维度一致
origin = _ni_support._normalize_sequence(origin, input.ndim)

# 检查结构的中心是否为 True
cit = _center_is_true(structure, origin)

# 如果输出参数是 NumPy 数组，则检查其数据类型是否为复数类型，若是则引发类型错误
if isinstance(output, np.ndarray):
    if np.iscomplexobj(output):
        raise TypeError('Complex output type not supported')
else:
    # 否则，将输出类型设置为布尔型
    output = bool

# 获取输出数组，确保其与输入数据具有相同的形状
output = _ni_support._get_output(output, input)

# 检查是否需要临时数组以防止输入和输出数组共享内存
temp_needed = np.may_share_memory(input, output)
if temp_needed:
    # 如果需要临时数组，则将输出数组复制到临时变量中
    temp = output
    output = _ni_support._get_output(output.dtype, input)

# 根据迭代次数调用二进制侵蚀函数
if iterations == 1:
    _nd_image.binary_erosion(input, structure, mask, output,
                             border_value, origin, invert, cit, 0)
elif cit and not brute_force:
    # 如果使用中心为 True 的结构并且不使用暴力方法，则调用改进的二进制侵蚀函数
    changed, coordinate_list = _nd_image.binary_erosion(
        input, structure, mask, output,
        border_value, origin, invert, cit, 1)
    
    # 反转结构数组的每个维度
    structure = structure[tuple([slice(None, None, -1)] * structure.ndim)]
    
    # 调整原点参数的符号
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure.shape[ii] & 1:
            origin[ii] -= 1
    
    # 如果提供了掩码参数，则将其转换为 int8 类型的 NumPy 数组
    if mask is not None:
        mask = np.asarray(mask, dtype=np.int8)
    
    # 如果结构数组不是连续存储的，则复制一份
    if not structure.flags.contiguous:
        structure = structure.copy()
    
    # 调用改进的二进制侵蚀函数来执行剩余的迭代
    _nd_image.binary_erosion2(output, structure, mask, iterations - 1,
                              origin, invert, coordinate_list)
    else:
        # 创建一个与输入数组 input 类型和形状相同的空布尔数组
        tmp_in = np.empty_like(input, dtype=bool)
        # 将输出数组 output 赋值给 tmp_out
        tmp_out = output
        # 如果迭代次数大于等于 1 并且迭代次数为偶数，则交换 tmp_in 和 tmp_out
        if iterations >= 1 and not iterations & 1:
            tmp_in, tmp_out = tmp_out, tmp_in
        # 进行第一次二进制侵蚀操作，结果存储在 tmp_out 中
        changed = _nd_image.binary_erosion(
            input, structure, mask, tmp_out,
            border_value, origin, invert, cit, 0)
        # 初始化迭代计数器为 1
        ii = 1
        # 当迭代次数 ii 小于指定的迭代次数 iterations 或者（当 iterations 小于 1 且 changed 为真时）继续循环
        while ii < iterations or (iterations < 1 and changed):
            # 交换 tmp_in 和 tmp_out
            tmp_in, tmp_out = tmp_out, tmp_in
            # 执行二进制侵蚀操作，将结果存储在 tmp_out 中，并更新 changed 变量
            changed = _nd_image.binary_erosion(
                tmp_in, structure, mask, tmp_out,
                border_value, origin, invert, cit, 0)
            # 迭代计数器 ii 自增 1
            ii += 1
    # 如果需要保存临时输出结果
    if temp_needed:
        # 将临时输出结果复制给输出数组 output
        temp[...] = output
        output = temp
    # 返回最终的输出数组 output
    return output
# 多维二进制腐蚀，使用给定的结构元素。

# 参数说明：
# input : array_like
#     要腐蚀的二进制图像。非零（True）元素形成待腐蚀的子集。
# structure : array_like, optional
#     用于腐蚀的结构元素。非零元素被视为True。如果未提供结构元素，则生成一个方形连接性为1的元素。
# iterations : int, optional
#     腐蚀操作重复执行的次数（默认为1）。如果迭代次数小于1，则一直重复腐蚀直到结果不再改变。
# mask : array_like, optional
#     如果提供了掩码，只有在相应掩码元素处为True时才会在每次迭代中修改这些元素。
# output : ndarray, optional
#     输出的形状与输入相同的数组。默认情况下，会创建一个新数组。
# border_value : int (cast to 0 or 1), optional
#     输出数组边界的值。
# origin : int or tuple of ints, optional
#     过滤器的放置位置，默认为0。
# brute_force : boolean, optional
#     内存条件：如果为False，则只跟踪上一次迭代中值发生变化的像素作为当前迭代中要更新（腐蚀）的候选像素；
#     如果为True，则不管上一次迭代发生了什么，所有像素都被视为腐蚀的候选像素。默认为False。

# 返回值：
# binary_erosion : ndarray of bools
#     通过结构元素对输入进行的腐蚀操作的结果。

# 参考资料：
# Erosion 是一种数学形态学操作，通过一个结构元素收缩图像中的形状。图像的二进制腐蚀是结构元素在点上居中时，
# 其重叠完全包含在图像的非零元素集合中的点的轨迹。

# 引用：
# [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
# [2] https://en.wikipedia.org/wiki/Mathematical_morphology
def binary_erosion(input, structure=None, iterations=1, mask=None, output=None,
                   border_value=0, origin=0, brute_force=False):
    """
    Multidimensional binary erosion with a given structuring element.

    Binary erosion is a mathematical morphology operation used for image
    processing.

    Parameters
    ----------
    input : array_like
        Binary image to be eroded. Non-zero (True) elements form
        the subset to be eroded.
    structure : array_like, optional
        Structuring element used for the erosion. Non-zero elements are
        considered True. If no structuring element is provided, an element
        is generated with a square connectivity equal to one.
    iterations : int, optional
        The erosion is repeated `iterations` times (one, by default).
        If iterations is less than 1, the erosion is repeated until the
        result does not change anymore.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated (eroded) in
        the current iteration; if True all pixels are considered as candidates
        for erosion, regardless of what happened in the previous iteration.
        False by default.

    Returns
    -------
    binary_erosion : ndarray of bools
        Erosion of the input by the structuring element.

    See Also
    --------
    grey_erosion, binary_dilation, binary_closing, binary_opening,
    generate_binary_structure

    Notes
    -----
    Erosion [1]_ is a mathematical morphology operation [2]_ that uses a
    structuring element for shrinking the shapes in an image. The binary
    erosion of an image by a structuring element is the locus of the points
    where a superimposition of the structuring element centered on the point
    is entirely contained in the set of non-zero elements of the image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.zeros((7,7), dtype=int)
    >>> a[1:6, 2:5] = 1
    >>> a
    """
    # 创建一个二维数组，表示一个图像或二值化后的像素矩阵
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.binary_erosion(a).astype(a.dtype)
    # 对输入的二进制图像进行腐蚀操作，去除小于结构元素的对象
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> #Erosion removes objects smaller than the structure
    # 腐蚀操作可以移除小于结构元素的对象，这是腐蚀操作的常见效果描述
    >>> ndimage.binary_erosion(a, structure=np.ones((5,5))).astype(a.dtype)
    # 使用一个5x5的全1结构元素对二进制图像进行腐蚀操作
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    
    """
    # 返回二进制腐蚀操作的结果
    return _binary_erosion(input, structure, iterations, mask,
                           output, border_value, origin, 0, brute_force)
# 定义一个多维二进制膨胀函数，使用给定的结构元素进行操作
def binary_dilation(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0, origin=0,
                    brute_force=False):
    """
    Multidimensional binary dilation with the given structuring element.

    Parameters
    ----------
    input : array_like
        Binary array_like to be dilated. Non-zero (True) elements form
        the subset to be dilated.
    structure : array_like, optional
        Structuring element used for the dilation. Non-zero elements are
        considered True. If no structuring element is provided an element
        is generated with a square connectivity equal to one.
    iterations : int, optional
        The dilation is repeated `iterations` times (one, by default).
        If iterations is less than 1, the dilation is repeated until the
        result does not change anymore. Only an integer of iterations is
        accepted.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated (dilated)
        in the current iteration; if True all pixels are considered as
        candidates for dilation, regardless of what happened in the previous
        iteration. False by default.

    Returns
    -------
    binary_dilation : ndarray of bools
        Dilation of the input by the structuring element.

    See Also
    --------
    grey_dilation, binary_erosion, binary_closing, binary_opening,
    generate_binary_structure

    Notes
    -----
    Dilation [1]_ is a mathematical morphology operation [2]_ that uses a
    structuring element for expanding the shapes in an image. The binary
    dilation of an image by a structuring element is the locus of the points
    covered by the structuring element, when its center lies within the
    non-zero points of the image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.zeros((5, 5))
    >>> a[2, 2] = 1
    >>> a
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(a)
    """
    # 实现多维二进制膨胀操作，根据输入的参数进行不同的处理
    pass
    input = np.asarray(input)
    # 将输入转换为 NumPy 数组
    
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    # 如果结构参数为 None，则使用默认的 3x3 连通性为 1 的二进制结构
    
    origin = _ni_support._normalize_sequence(origin, input.ndim)
    # 标准化 origin 参数，以确保其与输入的维度相匹配
    
    structure = np.asarray(structure)
    # 将结构参数转换为 NumPy 数组
    
    structure = structure[tuple([slice(None, None, -1)] *
                                structure.ndim)]
    # 对结构数组进行逆序操作，以便进行二进制侵蚀操作
    
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        # 对 origin 中的每个元素取负值，用于定义侵蚀的原点
    
        if not structure.shape[ii] & 1:
            origin[ii] -= 1
        # 如果结构数组的形状的第 ii 维度不是奇数，则将 origin 的对应维度再减去 1
    
    return _binary_erosion(input, structure, iterations, mask,
                           output, border_value, origin, 1, brute_force)
    # 调用二进制侵蚀函数 _binary_erosion，使用指定的参数进行操作并返回结果
# 定义一个函数，执行给定结构元素的多维二值开运算
def binary_opening(input, structure=None, iterations=1, output=None,
                   origin=0, mask=None, border_value=0, brute_force=False):
    """
    Multidimensional binary opening with the given structuring element.

    The *opening* of an input image by a structuring element is the
    *dilation* of the *erosion* of the image by the structuring element.

    Parameters
    ----------
    input : array_like
        Binary array_like to be opened. Non-zero (True) elements form
        the subset to be opened.
    structure : array_like, optional
        Structuring element used for the opening. Non-zero elements are
        considered True. If no structuring element is provided an element
        is generated with a square connectivity equal to one (i.e., only
        nearest neighbors are connected to the center, diagonally-connected
        elements are not considered neighbors).
    iterations : int, optional
        The erosion step of the opening, then the dilation step are each
        repeated `iterations` times (one, by default). If `iterations` is
        less than 1, each operation is repeated until the result does
        not change anymore. Only an integer of iterations is accepted.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.

        .. versionadded:: 1.1.0
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.

        .. versionadded:: 1.1.0
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated in the
        current iteration; if true all pixels are considered as candidates for
        update, regardless of what happened in the previous iteration.
        False by default.

        .. versionadded:: 1.1.0

    Returns
    -------
    binary_opening : ndarray of bools
        Opening of the input by the structuring element.

    See Also
    --------
    grey_opening, binary_closing, binary_erosion, binary_dilation,
    generate_binary_structure

    Notes
    -----
    *Opening* [1]_ is a mathematical morphology operation [2]_ that
    consists in the succession of an erosion and a dilation of the
    input with the same structuring element. Opening, therefore, removes
    objects smaller than the structuring element.

    Together with *closing* (`binary_closing`), opening can be used for
    noise removal.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Opening_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    """
    """
    使用示例代码展示了如何使用 scipy 的 ndimage 模块进行二进制形态学操作，包括开运算、闭运算等。

    >>> from scipy import ndimage  # 导入 scipy 的图像处理模块 ndimage
    >>> import numpy as np  # 导入 numpy 数组操作库
    >>> a = np.zeros((5,5), dtype=int)  # 创建一个 5x5 的整数类型全零数组 a
    >>> a[1:4, 1:4] = 1; a[4, 4] = 1  # 将数组 a 的一部分设为 1，模拟一个二值图像
    >>> a
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 1]])
    >>> # Opening removes small objects
    >>> ndimage.binary_opening(a, structure=np.ones((3,3))).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> # Opening can also smooth corners
    >>> ndimage.binary_opening(a).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> # Opening is the dilation of the erosion of the input
    >>> ndimage.binary_erosion(a).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> ndimage.binary_dilation(ndimage.binary_erosion(a)).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])

    """
    input = np.asarray(input)  # 将输入参数转换为 numpy 数组
    if structure is None:  # 如果结构元素为 None，则根据输入的维度生成一个二进制结构
        rank = input.ndim  # 获取输入数组的维度
        structure = generate_binary_structure(rank, 1)  # 生成对应维度的二进制结构

    # 对输入进行二值侵蚀操作，得到临时结果 tmp
    tmp = binary_erosion(input, structure, iterations, mask, None,
                         border_value, origin, brute_force)
    # 对 tmp 进行二值膨胀操作，返回最终结果
    return binary_dilation(tmp, structure, iterations, mask, output,
                           border_value, origin, brute_force)
# 定义一个函数，用于多维二进制图像的闭运算，使用给定的结构元素进行操作
def binary_closing(input, structure=None, iterations=1, output=None,
                   origin=0, mask=None, border_value=0, brute_force=False):
    """
    Multidimensional binary closing with the given structuring element.

    The *closing* of an input image by a structuring element is the
    *erosion* of the *dilation* of the image by the structuring element.

    Parameters
    ----------
    input : array_like
        Binary array_like to be closed. Non-zero (True) elements form
        the subset to be closed.
    structure : array_like, optional
        Structuring element used for the closing. Non-zero elements are
        considered True. If no structuring element is provided an element
        is generated with a square connectivity equal to one (i.e., only
        nearest neighbors are connected to the center, diagonally-connected
        elements are not considered neighbors).
    iterations : int, optional
        The dilation step of the closing, then the erosion step are each
        repeated `iterations` times (one, by default). If iterations is
        less than 1, each operations is repeated until the result does
        not change anymore. Only an integer of iterations is accepted.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.
        
        .. versionadded:: 1.1.0
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.
        
        .. versionadded:: 1.1.0
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated in the
        current iteration; if true al pixels are considered as candidates for
        update, regardless of what happened in the previous iteration.
        False by default.
        
        .. versionadded:: 1.1.0

    Returns
    -------
    binary_closing : ndarray of bools
        Closing of the input by the structuring element.

    See Also
    --------
    grey_closing, binary_opening, binary_dilation, binary_erosion,
    generate_binary_structure

    Notes
    -----
    *Closing* [1]_ is a mathematical morphology operation [2]_ that
    consists in the succession of a dilation and an erosion of the
    input with the same structuring element. Closing therefore fills
    holes smaller than the structuring element.

    Together with *opening* (`binary_opening`), closing can be used for
    noise removal.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Closing_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    """
    """
    input = np.asarray(input)  # 将输入数据转换为 NumPy 数组

    if structure is None:
        rank = input.ndim  # 获取输入数据的维度
        structure = generate_binary_structure(rank, 1)  # 使用默认参数生成二进制结构

    tmp = binary_dilation(input, structure, iterations, mask, None,
                          border_value, origin, brute_force)  # 对输入数据进行二进制膨胀操作，并存储结果到 tmp 变量

    return binary_erosion(tmp, structure, iterations, mask, output,
                          border_value, origin, brute_force)  # 对膨胀后的数据进行二进制侵蚀操作，并返回结果
    """
def binary_hit_or_miss(input, structure1=None, structure2=None,
                       output=None, origin1=0, origin2=None):
    """
    Multidimensional binary hit-or-miss transform.

    The hit-or-miss transform finds the locations of a given pattern
    inside the input image.

    Parameters
    ----------
    input : array_like (cast to booleans)
        Binary image where a pattern is to be detected.
    structure1 : array_like (cast to booleans), optional
        Part of the structuring element to be fitted to the foreground
        (non-zero elements) of `input`. If no value is provided, a
        structure of square connectivity 1 is chosen.
    structure2 : array_like (cast to booleans), optional
        Second part of the structuring element that has to miss completely
        the foreground. If no value is provided, the complementary of
        `structure1` is taken.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    origin1 : int or tuple of ints, optional
        Placement of the first part of the structuring element `structure1`,
        by default 0 for a centered structure.
    origin2 : int or tuple of ints, optional
        Placement of the second part of the structuring element `structure2`,
        by default 0 for a centered structure. If a value is provided for
        `origin1` and not for `origin2`, then `origin2` is set to `origin1`.

    Returns
    -------
    binary_hit_or_miss : ndarray
        Hit-or-miss transform of `input` with the given structuring
        element (`structure1`, `structure2`).

    See Also
    --------
    binary_erosion

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hit-or-miss_transform

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.zeros((7,7), dtype=int)
    >>> a[1, 1] = 1; a[2:4, 2:4] = 1; a[4:6, 4:6] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> structure1 = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
    >>> structure1
    array([[1, 0, 0],
           [0, 1, 1],
           [0, 1, 1]])
    >>> # Find the matches of structure1 in the array a
    >>> ndimage.binary_hit_or_miss(a, structure1=structure1).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> # Change the origin of the filter
    >>> # origin1=1 is equivalent to origin1=(1,1) here
    >>> ndimage.binary_hit_or_miss(a, structure1=structure1,\
    origin1=1).astype(int)
    """

    # 使用 scipy.ndimage.binary_hit_or_miss 函数进行多维二进制命中或不命中变换
    return ndimage.binary_hit_or_miss(input, structure1, structure2, output, origin1, origin2)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])

"""
将输入转换为 NumPy 数组
input = np.asarray(input)
如果结构1未定义：
    生成一个与输入维度相同的二进制结构
    structure1 = generate_binary_structure(input.ndim, 1)
如果结构2未定义：
    使用结构1生成逻辑非结果作为结构2
    structure2 = np.logical_not(structure1)
将 origin1 规范化为与输入维度相匹配的序列
origin1 = _ni_support._normalize_sequence(origin1, input.ndim)
如果 origin2 未定义：
    origin2 与 origin1 相同
    origin2 = origin1
否则：
    将 origin2 规范化为与输入维度相匹配的序列
    origin2 = _ni_support._normalize_sequence(origin2, input.ndim)

使用结构1进行二进制侵蚀操作，结果存储在 tmp1 中
tmp1 = _binary_erosion(input, structure1, 1, None, None, 0, origin1, 0, False)

检查输出是否为 NumPy 数组
inplace = isinstance(output, np.ndarray)
使用结构2进行二进制侵蚀操作，结果存储在 result 中
result = _binary_erosion(input, structure2, 1, None, output, 0, origin2, 1, False)

如果 inplace 为真：
    将 output 取逻辑非，并将 tmp1 与 output 的逻辑与结果存储在 output 中
    np.logical_not(output, output)
    np.logical_and(tmp1, output, output)
否则：
    将 result 取逻辑非，并返回 tmp1 与 result 的逻辑与结果
    np.logical_not(result, result)
    return np.logical_and(tmp1, result)
# 定义了一个函数，用于执行多维度的二进制传播，将输入图像在指定掩模内传播
def binary_propagation(input, structure=None, mask=None,
                       output=None, border_value=0, origin=0):
    """
    Multidimensional binary propagation with the given structuring element.

    Parameters
    ----------
    input : array_like
        Binary image to be propagated inside `mask`.
        待传播的二进制图像，将其传播到`mask`内部。

    structure : array_like, optional
        Structuring element used in the successive dilations. The output
        may depend on the structuring element, especially if `mask` has
        several connex components. If no structuring element is
        provided, an element is generated with a squared connectivity equal
        to one.
        用于连续膨胀操作的结构元素。输出可能依赖于结构元素，特别是当`mask`具有多个连通组件时。如果未提供结构元素，则生成一个连接度为1的方形结构元素。

    mask : array_like, optional
        Binary mask defining the region into which `input` is allowed to
        propagate.
        定义`input`可以传播到的区域的二进制掩码。

    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
        与输入具有相同形状的数组，用于放置输出结果。默认情况下，创建一个新数组。

    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.
        输出数组边界的值，会被强制转换为0或1。

    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
        滤波器的放置位置，默认为0。

    Returns
    -------
    binary_propagation : ndarray
        Binary propagation of `input` inside `mask`.
        `input`在`mask`内的二进制传播结果。

    Notes
    -----
    This function is functionally equivalent to calling binary_dilation
    with the number of iterations less than one: iterative dilation until
    the result does not change anymore.
    该函数的功能等同于调用具有小于一的迭代次数的binary_dilation函数：即迭代扩张，直到结果不再改变。

    The succession of an erosion and propagation inside the original image
    can be used instead of an *opening* for deleting small objects while
    keeping the contours of larger objects untouched.
    在原始图像内进行侵蚀和传播的连续操作可以用来替代“开运算”，用于删除小对象，同时保持较大对象的轮廓不变。

    References
    ----------
    .. [1] http://cmm.ensmp.fr/~serra/cours/pdf/en/ch6en.pdf, slide 15.
    .. [2] I.T. Young, J.J. Gerbrands, and L.J. van Vliet, "Fundamentals of
        image processing", 1998
        ftp://qiftp.tudelft.nl/DIPimage/docs/FIP2.3.pdf

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> input = np.zeros((8, 8), dtype=int)
    >>> input[2, 2] = 1
    >>> mask = np.zeros((8, 8), dtype=int)
    >>> mask[1:4, 1:4] = mask[4, 4]  = mask[6:8, 6:8] = 1
    >>> input
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])
    >>> mask
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1]])
    >>> ndimage.binary_propagation(input, mask=mask).astype(int)
    # 返回通过二进制膨胀处理后的输入数组。膨胀过程基于给定的结构元素和遮罩。
    return binary_dilation(input, structure, -1, mask, output,
                           border_value, origin)
# 定义函数 `binary_fill_holes`，用于填充二进制对象中的空洞
def binary_fill_holes(input, structure=None, output=None, origin=0):
    # 创建一个布尔掩码，其值为输入数组 `input` 的逻辑非
    mask = np.logical_not(input)
    # 创建一个与 `mask` 形状相同的零数组 `tmp`
    tmp = np.zeros(mask.shape, bool)
    # 检查是否提供了输出数组 `output`
    inplace = isinstance(output, np.ndarray)
    # 如果 `output` 已提供，则在 `output` 上进行就地操作
    if inplace:
        # 对 `tmp` 进行二进制膨胀，结果存入 `output`
        binary_dilation(tmp, structure, -1, mask, output, 1, origin)
        # 对 `output` 进行逻辑非操作，填充后的结果存储在 `output` 中
        np.logical_not(output, output)
    else:
        # 如果未提供 `output`，则创建一个新的填充后的数组，存入 `output`
        output = binary_dilation(tmp, structure, -1, mask, None, 1, origin)
        # 对 `output` 进行逻辑非操作，填充后的结果存储在 `output` 中
        np.logical_not(output, output)
        # 返回填充后的输出数组 `output`
        return output
    Grayscale erosion is a mathematical morphology operation. For the
    simple case of a full and flat structuring element, it can be viewed
    as a minimum filter over a sliding window.

    Parameters
    ----------
    input : array_like
        Array over which the grayscale erosion is to be computed.
    size : tuple of ints
        Shape of a flat and full structuring element used for the grayscale
        erosion. Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the grayscale erosion. Non-zero values give the set of
        neighbors of the center over which the minimum is chosen.
    structure : array of ints, optional
        Structuring element used for the grayscale erosion. `structure`
        may be a non-flat structuring element. The `structure` array applies a
        subtractive offset for each pixel in the neighborhood.
    output : array, optional
        An array used for storing the output of the erosion may be provided.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    output : ndarray
        Grayscale erosion of `input`.

    See Also
    --------
    binary_erosion, grey_dilation, grey_opening, grey_closing
    generate_binary_structure, minimum_filter

    Notes
    -----
    The grayscale erosion of an image input by a structuring element s defined
    over a domain E is given by:

    (input+s)(x) = min {input(y) - s(x-y), for y in E}

    In particular, for structuring elements defined as
    s(y) = 0 for y in E, the grayscale erosion computes the minimum of the
    input image inside a sliding window defined by E.

    Grayscale erosion [1]_ is a *mathematical morphology* operation [2]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.zeros((7,7), dtype=int)
    >>> a[1:6, 1:6] = 3
    >>> a[4,4] = 2; a[2,3] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 3, 3, 1, 3, 3, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 3, 3, 3, 2, 3, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.grey_erosion(a, size=(3,3))


注释：
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 3, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> footprint = ndimage.generate_binary_structure(2, 1)
    >>> footprint
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> # Diagonally-connected elements are not considered neighbors
    >>> ndimage.grey_erosion(a, footprint=footprint)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 3, 1, 2, 0, 0],
           [0, 0, 3, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])

    """
    # 如果 size、footprint 和 structure 都未指定，抛出数值错误
    if size is None and footprint is None and structure is None:
        raise ValueError("size, footprint, or structure must be specified")

    # 调用底层 C 函数进行最小或最大滤波操作，返回结果
    return _filters._min_or_max_filter(input, size, footprint, structure,
                                       output, mode, cval, origin, 1)
# 定义函数 `grey_dilation`，用于计算灰度膨胀操作，基于数学形态学中的膨胀操作，可视为滑动窗口上的最大值滤波器。

"""
Calculate a greyscale dilation, using either a structuring element,
or a footprint corresponding to a flat structuring element.

Grayscale dilation is a mathematical morphology operation. For the
simple case of a full and flat structuring element, it can be viewed
as a maximum filter over a sliding window.

Parameters
----------
input : array_like
    Array over which the grayscale dilation is to be computed.
size : tuple of ints
    Shape of a flat and full structuring element used for the grayscale
    dilation. Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the grayscale dilation. Non-zero values give the set of
    neighbors of the center over which the maximum is chosen.
structure : array of ints, optional
    Structuring element used for the grayscale dilation. `structure`
    may be a non-flat structuring element. The `structure` array applies an
    additive offset for each pixel in the neighborhood.
output : array, optional
    An array used for storing the output of the dilation may be provided.
mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
grey_dilation : ndarray
    Grayscale dilation of `input`.

See Also
--------
binary_dilation, grey_erosion, grey_closing, grey_opening
generate_binary_structure, maximum_filter

Notes
-----
The grayscale dilation of an image input by a structuring element s defined
over a domain E is given by:

(input+s)(x) = max {input(y) + s(x-y), for y in E}

In particular, for structuring elements defined as
s(y) = 0 for y in E, the grayscale dilation computes the maximum of the
input image inside a sliding window defined by E.

Grayscale dilation [1]_ is a *mathematical morphology* operation [2]_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> import numpy as np
>>> a = np.zeros((7,7), dtype=int)
>>> a[2:5, 2:5] = 1
>>> a[4,4] = 2; a[2,3] = 3
>>> a
"""

def grey_dilation(input, size=None, footprint=None, structure=None,
                  output=None, mode="reflect", cval=0.0, origin=0):
    """
    Calculate a greyscale dilation, using either a structuring element,
    or a footprint corresponding to a flat structuring element.

    Grayscale dilation is a mathematical morphology operation. For the
    simple case of a full and flat structuring element, it can be viewed
    as a maximum filter over a sliding window.

    Parameters
    ----------
    input : array_like
        Array over which the grayscale dilation is to be computed.
    size : tuple of ints
        Shape of a flat and full structuring element used for the grayscale
        dilation. Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the grayscale dilation. Non-zero values give the set of
        neighbors of the center over which the maximum is chosen.
    structure : array of ints, optional
        Structuring element used for the grayscale dilation. `structure`
        may be a non-flat structuring element. The `structure` array applies an
        additive offset for each pixel in the neighborhood.
    output : array, optional
        An array used for storing the output of the dilation may be provided.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    grey_dilation : ndarray
        Grayscale dilation of `input`.
    """
    # 检查是否未指定 size、footprint 和 structure 中的任何一个，如果是则抛出数值错误异常
    if size is None and footprint is None and structure is None:
        raise ValueError("size, footprint, or structure must be specified")
    
    # 如果指定了 structure 参数，则将其转换为 NumPy 数组，并进行维度倒序处理
    if structure is not None:
        structure = np.asarray(structure)
        structure = structure[tuple([slice(None, None, -1)] *
                                    structure.ndim)]
    
    # 如果指定了 footprint 参数，则将其转换为 NumPy 数组，并进行维度倒序处理
    if footprint is not None:
        footprint = np.asarray(footprint)
        footprint = footprint[tuple([slice(None, None, -1)] *
                                    footprint.ndim)]

    # 将输入数组转换为 NumPy 数组
    input = np.asarray(input)
    
    # 标准化 origin 参数的序列，使其与输入数组的维度匹配
    origin = _ni_support._normalize_sequence(origin, input.ndim)
    
    # 遍历 origin 序列，并对其进行处理以确保其为负值
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        
        # 如果指定了 footprint 参数，则获取其在当前维度下的形状大小
        if footprint is not None:
            sz = footprint.shape[ii]
        
        # 如果指定了 structure 参数，则获取其在当前维度下的形状大小
        elif structure is not None:
            sz = structure.shape[ii]
        
        # 如果 size 是标量，则将其作为当前维度下的大小
        elif np.isscalar(size):
            sz = size
        
        # 否则，从 size 参数中获取当前维度下的大小
        else:
            sz = size[ii]
        
        # 如果 sz 不是奇数，则将 origin 在当前维度下减小 1
        if not sz & 1:
            origin[ii] -= 1

    # 调用内部函数 _filters._min_or_max_filter 执行最小值或最大值滤波操作，返回结果
    return _filters._min_or_max_filter(input, size, footprint, structure,
                                       output, mode, cval, origin, 0)
# 定义一个多维灰度开运算函数
def grey_opening(input, size=None, footprint=None, structure=None,
                 output=None, mode="reflect", cval=0.0, origin=0):
    """
    Multidimensional grayscale opening.

    A grayscale opening consists in the succession of a grayscale erosion,
    and a grayscale dilation.

    Parameters
    ----------
    input : array_like
        Array over which the grayscale opening is to be computed.
    size : tuple of ints
        Shape of a flat and full structuring element used for the grayscale
        opening. Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the grayscale opening.
    structure : array of ints, optional
        Structuring element used for the grayscale opening. `structure`
        may be a non-flat structuring element. The `structure` array applies
        offsets to the pixels in a neighborhood (the offset is additive during
        dilation and subtractive during erosion).
    output : array, optional
        An array used for storing the output of the opening may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    grey_opening : ndarray
        Result of the grayscale opening of `input` with `structure`.

    See Also
    --------
    binary_opening, grey_dilation, grey_erosion, grey_closing
    generate_binary_structure

    Notes
    -----
    The action of a grayscale opening with a flat structuring element amounts
    to smoothen high local maxima, whereas binary opening erases small objects.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.arange(36).reshape((6,6))
    >>> a[3, 3] = 50
    >>> a
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 50, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])
    >>> ndimage.grey_opening(a, size=(3,3))
    array([[ 0,  1,  2,  3,  4,  4],
           [ 6,  7,  8,  9, 10, 10],
           [12, 13, 14, 15, 16, 16],
           [18, 19, 20, 22, 22, 22],
           [24, 25, 26, 27, 28, 28],
           [24, 25, 26, 27, 28, 28]])
    >>> # Note that the local maximum a[3,3] has disappeared

    """
    # 如果 size 和 footprint 都不为 None，则发出警告并忽略 size 参数
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=2)
    # 对输入图像进行灰度侵蚀操作，返回处理后的结果
    tmp = grey_erosion(input, size, footprint, structure, None, mode,
                       cval, origin)
    # 对灰度侵蚀结果进行灰度膨胀操作，返回最终处理后的结果
    return grey_dilation(tmp, size, footprint, structure, output, mode,
                         cval, origin)
# 定义一个多维灰度闭运算函数，对输入的数组进行处理
def grey_closing(input, size=None, footprint=None, structure=None,
                 output=None, mode="reflect", cval=0.0, origin=0):
    """
    Multidimensional grayscale closing.

    A grayscale closing consists in the succession of a grayscale dilation,
    and a grayscale erosion.

    Parameters
    ----------
    input : array_like
        Array over which the grayscale closing is to be computed.
    size : tuple of ints
        Shape of a flat and full structuring element used for the grayscale
        closing. Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the grayscale closing.
    structure : array of ints, optional
        Structuring element used for the grayscale closing. `structure`
        may be a non-flat structuring element. The `structure` array applies
        offsets to the pixels in a neighborhood (the offset is additive during
        dilation and subtractive during erosion)
    output : array, optional
        An array used for storing the output of the closing may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    grey_closing : ndarray
        Result of the grayscale closing of `input` with `structure`.

    See Also
    --------
    binary_closing, grey_dilation, grey_erosion, grey_opening,
    generate_binary_structure

    Notes
    -----
    The action of a grayscale closing with a flat structuring element amounts
    to smoothen deep local minima, whereas binary closing fills small holes.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.arange(36).reshape((6,6))
    >>> a[3,3] = 0
    >>> a
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20,  0, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])
    >>> ndimage.grey_closing(a, size=(3,3))
    array([[ 7,  7,  8,  9, 10, 11],
           [ 7,  7,  8,  9, 10, 11],
           [13, 13, 14, 15, 16, 17],
           [19, 19, 20, 20, 22, 23],
           [25, 25, 26, 27, 28, 29],
           [31, 31, 32, 33, 34, 35]])
    >>> # Note that the local minimum a[3,3] has disappeared

    """
    # 如果 `size` 和 `footprint` 都不是 None，则发出警告并忽略 `size` 参数，因为 `footprint` 参数已设置
    if (size is not None) and (footprint is not None):
        # 发出警告消息，提示用户忽略 `size` 参数的设定
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=2)
    
    # 对输入图像进行灰度膨胀操作，并将结果存储在临时变量 `tmp` 中
    tmp = grey_dilation(input, size, footprint, structure, None, mode,
                        cval, origin)
    
    # 对临时变量 `tmp` 进行灰度侵蚀操作，并返回结果
    return grey_erosion(tmp, size, footprint, structure, output, mode,
                        cval, origin)
# 定义一个多维形态梯度函数，用于计算输入数组的形态梯度
def morphological_gradient(input, size=None, footprint=None, structure=None,
                           output=None, mode="reflect", cval=0.0, origin=0):
    """
    Multidimensional morphological gradient.

    The morphological gradient is calculated as the difference between a
    dilation and an erosion of the input with a given structuring element.

    Parameters
    ----------
    input : array_like
        Array over which to compute the morphlogical gradient.
    size : tuple of ints
        Shape of a flat and full structuring element used for the mathematical
        morphology operations. Optional if `footprint` or `structure` is
        provided. A larger `size` yields a more blurred gradient.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the morphology operations. Larger footprints
        give a more blurred morphological gradient.
    structure : array of ints, optional
        Structuring element used for the morphology operations. `structure` may
        be a non-flat structuring element. The `structure` array applies
        offsets to the pixels in a neighborhood (the offset is additive during
        dilation and subtractive during erosion)
    output : array, optional
        An array used for storing the output of the morphological gradient
        may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    morphological_gradient : ndarray
        Morphological gradient of `input`.

    See Also
    --------
    grey_dilation, grey_erosion, gaussian_gradient_magnitude

    Notes
    -----
    For a flat structuring element, the morphological gradient
    computed at a given point corresponds to the maximal difference
    between elements of the input among the elements covered by the
    structuring element centered on the point.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.zeros((7,7), dtype=int)
    >>> a[2:5, 2:5] = 1
    >>> ndimage.morphological_gradient(a, size=(3,3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> # The morphological gradient is computed as the difference
    >>> # between a dilation and an erosion
    """
    # 根据给定的结构元素，计算输入数组的形态梯度，即膨胀和腐蚀之间的差异
    return _ni_morphological_gradient(input, size, footprint, structure, output,
                                      mode, cval, origin)
    """
    根据输入参数进行灰度膨胀操作，并返回结果与灰度侵蚀操作的差值。
    
    Parameters:
    - input: 输入数组，用于执行膨胀和侵蚀操作的图像数据。
    - size: 用于膨胀和侵蚀的结构元素的大小。
    - footprint, structure: 影响像素的形状或结构定义。
    - output: 可选的输出数组，用于存储侵蚀操作的结果。
    - mode: 处理边界的模式（例如'constant', 'nearest', 'reflect'等）。
    - cval: 当 mode 是'constant'时，用于填充边界的常数值。
    - origin: 结构元素的原点位置。
    
    Returns:
    - 如果输出参数是 ndarray，则返回膨胀结果与输出的差值。
    - 如果输出参数不是 ndarray，则返回膨胀结果与侵蚀结果的差值。
    
    """
    
    tmp = grey_dilation(input, size, footprint, structure, None, mode,
                        cval, origin)
    # 如果输出参数是 ndarray，执行侵蚀操作，并返回差值
    if isinstance(output, np.ndarray):
        grey_erosion(input, size, footprint, structure, output, mode,
                     cval, origin)
        return np.subtract(tmp, output, output)
    # 如果输出参数不是 ndarray，直接计算膨胀结果与侵蚀结果的差值并返回
    else:
        return (tmp - grey_erosion(input, size, footprint, structure,
                                   None, mode, cval, origin))
# 多维形态学拉普拉斯变换。

def morphological_laplace(input, size=None, footprint=None,
                          structure=None, output=None,
                          mode="reflect", cval=0.0, origin=0):
    """
    Multidimensional morphological laplace.

    Parameters
    ----------
    input : array_like
        输入数组。
    size : tuple of ints
        用于数学形态学操作的扁平且完整的结构元素的形状。如果提供了 `footprint` 或 `structure`，则此参数可选。
    footprint : array of ints, optional
        扁平结构元素的非无穷元素的位置，用于形态学操作。
    structure : array of ints, optional
        用于形态学操作的结构元素。`structure` 可能是非扁平的结构元素。`structure` 数组对邻域中的像素应用偏移
        （在膨胀期间是加法，在侵蚀期间是减法）。
    output : ndarray, optional
        可选的输出数组。
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        mode 参数确定如何处理数组边界。
        对于 'constant' 模式，超出边界的值被设置为 `cval`。
        默认为 'reflect'。
    cval : scalar, optional
        如果模式为 'constant'，则用于填充输入边缘之外的值。
        默认为 0.0。
    origin : origin, optional
        origin 参数控制滤波器的放置。

    Returns
    -------
    morphological_laplace : ndarray
        输出数组

    """
    # 使用 grey_dilation 函数对输入进行灰度膨胀操作，存储在 tmp1 中
    tmp1 = grey_dilation(input, size, footprint, structure, None, mode,
                         cval, origin)
    # 如果输出参数是 ndarray 类型
    if isinstance(output, np.ndarray):
        # 使用 grey_erosion 函数对输入进行灰度侵蚀操作，结果存储在 output 中
        grey_erosion(input, size, footprint, structure, output, mode,
                     cval, origin)
        # 将 tmp1 和 output 相加，结果存储在 output 中
        np.add(tmp1, output, output)
        # 将 output 和输入相减，结果存储在 output 中
        np.subtract(output, input, output)
        # 返回 output 减去输入的结果
        return np.subtract(output, input, output)
    else:
        # 使用 grey_erosion 函数对输入进行灰度侵蚀操作，结果存储在 tmp2 中
        tmp2 = grey_erosion(input, size, footprint, structure, None, mode,
                            cval, origin)
        # 将 tmp1 和 tmp2 相加，结果存储在 tmp2 中
        np.add(tmp1, tmp2, tmp2)
        # 将 tmp2 和输入相减，结果存储在 tmp2 中
        np.subtract(tmp2, input, tmp2)
        # 再次将 tmp2 和输入相减，结果存储在 tmp2 中
        np.subtract(tmp2, input, tmp2)
        # 返回 tmp2 减去输入的结果
        return tmp2


def white_tophat(input, size=None, footprint=None, structure=None,
                 output=None, mode="reflect", cval=0.0, origin=0):
    """
    Multidimensional white tophat filter.

    Parameters
    ----------
    input : array_like
        输入数组。
    size : tuple of ints
        用于过滤器的扁平和完整结构元素的形状。如果提供了 `footprint` 或 `structure`，则此参数可选。
    footprint : array of ints, optional
        扁平结构元素的元素位置，用于白顶帽过滤器。
    """
    structure : array of ints, optional
        结构元素用于滤波器。`structure` 可以是非扁平的结构元素。在膨胀期间，`structure` 数组对邻域中的像素应用偏移（在腐蚀期间为减法）。
    output : array, optional
        用于存储滤波器输出的数组。可选提供。
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        `mode` 参数确定如何处理数组边界，其中 `cval` 是当 `mode` 等于 'constant' 时的值。默认为 'reflect'。
    cval : scalar, optional
        如果 `mode` 是 'constant'，则用于填充输入超出边界的值。默认为 0.0。
    origin : scalar, optional
        `origin` 参数控制滤波器放置的位置。默认为 0。

    Returns
    -------
    output : ndarray
        使用 `structure` 对 `input` 进行滤波的结果。

    See Also
    --------
    black_tophat

    Examples
    --------
    从亮峰减去灰色背景。

    >>> from scipy.ndimage import generate_binary_structure, white_tophat
    >>> import numpy as np
    >>> square = generate_binary_structure(rank=2, connectivity=3)
    >>> bright_on_gray = np.array([[2, 3, 3, 3, 2],
    ...                            [3, 4, 5, 4, 3],
    ...                            [3, 5, 9, 5, 3],
    ...                            [3, 4, 5, 4, 3],
    ...                            [2, 3, 3, 3, 2]])
    >>> white_tophat(input=bright_on_gray, structure=square)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 5, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])

    """
    # 如果设置了 `size` 和 `footprint`，则发出警告，忽略 `size` 参数
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=2)
    # 对输入进行灰度腐蚀，返回临时结果
    tmp = grey_erosion(input, size, footprint, structure, None, mode,
                       cval, origin)
    # 对临时结果进行灰度膨胀，可选使用提供的 `output` 存储输出
    tmp = grey_dilation(tmp, size, footprint, structure, output, mode,
                        cval, origin)
    # 如果 `tmp` 为空，则使用提供的 `output`
    if tmp is None:
        tmp = output

    # 如果 `input` 和 `tmp` 的数据类型都是布尔型，使用按位异或运算，否则使用减法运算
    if input.dtype == np.bool_ and tmp.dtype == np.bool_:
        np.bitwise_xor(input, tmp, out=tmp)
    else:
        np.subtract(input, tmp, out=tmp)
    # 返回处理后的结果 `tmp`
    return tmp
def black_tophat(input, size=None, footprint=None,
                 structure=None, output=None, mode="reflect",
                 cval=0.0, origin=0):
    """
    Multidimensional black tophat filter.

    Parameters
    ----------
    input : array_like
        输入的数组，表示待处理的输入数据。
    size : tuple of ints, optional
        用于滤波的平坦和完整结构元素的形状。如果提供了 `footprint` 或 `structure`，则可选。
    footprint : array of ints, optional
        用于黑顶帽滤波的平坦结构元素的非无限元素的位置。
    structure : array of ints, optional
        用于滤波的结构元素。`structure` 可以是非平坦的结构元素。
        `structure` 数组在邻域内的像素应用偏移（在膨胀期间加法偏移，在侵蚀期间减法偏移）。
    output : array, optional
        可提供用于存储滤波输出的数组。
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        `mode` 参数确定如何处理数组边界，其中 `cval` 是当 `mode` 等于 'constant' 时的填充值。默认为 'reflect'。
    cval : scalar, optional
        如果 `mode` 是 'constant'，则在输入边界之外填充的值。默认为 0.0。
    origin : scalar, optional
        `origin` 参数控制滤波器放置的位置。默认为 0。

    Returns
    -------
    black_tophat : ndarray
        使用 `structure` 对 `input` 进行滤波的结果。

    See Also
    --------
    white_tophat, grey_opening, grey_closing

    Examples
    --------
    将暗峰转换为亮峰并减去背景。

    >>> from scipy.ndimage import generate_binary_structure, black_tophat
    >>> import numpy as np
    >>> square = generate_binary_structure(rank=2, connectivity=3)
    >>> dark_on_gray = np.array([[7, 6, 6, 6, 7],
    ...                          [6, 5, 4, 5, 6],
    ...                          [6, 4, 0, 4, 6],
    ...                          [6, 5, 4, 5, 6],
    ...                          [7, 6, 6, 6, 7]])
    >>> black_tophat(input=dark_on_gray, structure=square)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 5, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])

    """
    # 如果 `size` 和 `footprint` 同时提供，发出警告并忽略 `size`
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=2)
    # 进行灰度膨胀操作，得到临时结果 `tmp`
    tmp = grey_dilation(input, size, footprint, structure, None, mode,
                        cval, origin)
    # 对临时结果进行灰度侵蚀操作，得到最终的黑顶帽滤波结果 `tmp`
    tmp = grey_erosion(tmp, size, footprint, structure, output, mode,
                       cval, origin)
    # 如果 `tmp` 为 None，则使用提供的 `output`
    if tmp is None:
        tmp = output

    # 如果输入和 `tmp` 的数据类型都是布尔类型，使用位异或操作
    if input.dtype == np.bool_ and tmp.dtype == np.bool_:
        np.bitwise_xor(tmp, input, out=tmp)
    else:
        # 否则使用减法操作
        np.subtract(tmp, input, out=tmp)
    return tmp
# 定义一个距离变换函数，使用暴力算法实现

def distance_transform_bf(input, metric="euclidean", sampling=None,
                          return_distances=True, return_indices=False,
                          distances=None, indices=None):
    """
    Distance transform function by a brute force algorithm.

    This function calculates the distance transform of the `input`, by
    replacing each foreground (non-zero) element, with its
    shortest distance to the background (any zero-valued element).

    In addition to the distance transform, the feature transform can
    be calculated. In this case the index of the closest background
    element to each foreground element is returned in a separate array.

    Parameters
    ----------
    input : array_like
        输入数组

    metric : {'euclidean', 'taxicab', 'chessboard'}, optional
        距离度量方法。'cityblock' 和 'manhattan' 也是有效的，等效于 'taxicab'。
        默认值是 'euclidean'。

    sampling : float, or sequence of float, optional
        当 `metric` 是 'euclidean' 时使用的参数。指定每个维度上的元素间距。
        如果是一个序列，必须与输入数组的维度相同；如果是单个数值，将用于所有轴。
        如果未指定，假定网格间距为单位距离。

    return_distances : bool, optional
        是否计算距离变换，默认为 True。

    return_indices : bool, optional
        是否计算特征变换，默认为 False。

    distances : ndarray, optional
        用于存储计算得到的距离变换的输出数组，而不是返回它。
        当 `return_distances` 为 True 且未提供 `distances` 时返回。
        数组形状与 `input` 相同，如果 `metric` 是 'euclidean'，则为 float64 类型，否则为 uint32 类型。

    indices : int32 ndarray, optional
        用于存储计算得到的特征变换的输出数组，而不是返回它。
        当 `return_indices` 为 True 且未提供 `indices` 时返回。
        其形状必须为 ``(input.ndim,) + input.shape``。

    Returns
    -------
    distances : ndarray, optional
        计算得到的距离变换。仅在 `return_distances` 为 True 且未提供 `distances` 时返回。
        数组形状与输入数组相同。

    indices : int32 ndarray, optional
        计算得到的特征变换。每个输入维度的形状数组。详见 distance_transform_edt 文档的示例。
        仅在 `return_indices` 为 True 且未提供 `indices` 时返回。

    See Also
    --------
    distance_transform_cdt : 更快速的 taxicab 和 chessboard 距离变换算法
    distance_transform_edt : 更快速的 euclidean 距离变换算法

    Notes
    -----
    该函数采用一种较慢的暴力算法。更高效的 taxicab [1]_ 和 chessboard [2]_ 算法请参见 `distance_transform_cdt` 函数。
    """
    ...                                                    metric='taxicab')
    # 使用 distance_transform_bf 函数计算基于曼哈顿距离（taxicab metric）的距离变换
    >>> taxicab_transformation = grid[2].imshow(distance_transform_taxicab,
    ...                                           cmap='gray')
    # 在第三个子图中显示基于曼哈顿距离的距离变换结果


这段代码是一个示例，展示了如何在Python中使用SciPy库进行图像处理。注释解释了每行代码的作用，特别是如何计算不同距离度量（如欧几里得距离和曼哈顿距离）的距离变换，并且如何将结果显示在Matplotlib的图像网格中。
    # 检查 indices 和 distances 是否是 NumPy 数组的实例，返回布尔值
    ft_inplace = isinstance(indices, np.ndarray)
    dt_inplace = isinstance(distances, np.ndarray)
    
    # 调用 _distance_tranform_arg_check 函数，检查并处理距离变换的参数
    _distance_tranform_arg_check(
        dt_inplace, ft_inplace, return_distances, return_indices
    )
    
    # 将输入 input 转换为布尔值数组，并生成与其维度相匹配的二进制结构
    tmp1 = np.asarray(input) != 0
    struct = generate_binary_structure(tmp1.ndim, tmp1.ndim)
    
    # 对 tmp1 进行二值膨胀处理，并计算其与原始值的逻辑异或
    tmp2 = binary_dilation(tmp1, struct)
    tmp2 = np.logical_xor(tmp1, tmp2)
    
    # 将 tmp1 和 tmp2 转换为 int8 类型，并进行相减操作
    tmp1 = tmp1.astype(np.int8) - tmp2.astype(np.int8)
    
    # 将 metric 转换为小写，根据不同的度量标准设置相应的值：1 表示欧氏距离，2 表示曼哈顿距离，3 表示棋盘距离
    metric = metric.lower()
    if metric == 'euclidean':
        metric = 1
    elif metric in ['taxicab', 'cityblock', 'manhattan']:
        metric = 2
    elif metric == 'chessboard':
        metric = 3
    else:
        raise RuntimeError('distance metric not supported')
    
    # 如果指定了 sampling 参数，则对其进行归一化处理，并转换为 float64 类型
    if sampling is not None:
        sampling = _ni_support._normalize_sequence(sampling, tmp1.ndim)
        sampling = np.asarray(sampling, dtype=np.float64)
        if not sampling.flags.contiguous:
            sampling = sampling.copy()
    
    # 根据 return_indices 的值初始化 ft 数组
    if return_indices:
        ft = np.zeros(tmp1.shape, dtype=np.int32)
    else:
        ft = None
    
    # 如果 return_distances 为 True，则根据 metric 的不同值初始化 dt 数组
    if return_distances:
        if distances is None:
            if metric == 1:
                dt = np.zeros(tmp1.shape, dtype=np.float64)
            else:
                dt = np.zeros(tmp1.shape, dtype=np.uint32)
        else:
            if distances.shape != tmp1.shape:
                raise RuntimeError('distances array has wrong shape')
            if metric == 1:
                if distances.dtype.type != np.float64:
                    raise RuntimeError('distances array must be float64')
            else:
                if distances.dtype.type != np.uint32:
                    raise RuntimeError('distances array must be uint32')
            dt = distances
    else:
        dt = None
    
    # 调用 _nd_image.distance_transform_bf 函数进行距离变换计算
    _nd_image.distance_transform_bf(tmp1, metric, sampling, dt, ft)
    # 如果需要返回索引
    if return_indices:
        # 如果输入的索引是一个 NumPy 数组
        if isinstance(indices, np.ndarray):
            # 检查索引数组的数据类型是否为 int32
            if indices.dtype.type != np.int32:
                raise RuntimeError('indices array must be int32')
            # 检查索引数组的形状是否与 tmp1 的形状一致
            if indices.shape != (tmp1.ndim,) + tmp1.shape:
                raise RuntimeError('indices array has wrong shape')
            # 将 tmp2 设置为输入的索引数组
            tmp2 = indices
        else:
            # 如果没有输入索引数组，则使用 np.indices 创建一个 int32 类型的索引数组
            tmp2 = np.indices(tmp1.shape, dtype=np.int32)
        
        # 将 ft 展平为一维数组
        ft = np.ravel(ft)
        
        # 遍历 tmp2 的第一维度
        for ii in range(tmp2.shape[0]):
            # 根据 ft 中的索引重排 tmp2[ii, ...] 的内容
            rtmp = np.ravel(tmp2[ii, ...])[ft]
            # 将 rtmp 的形状重新设为 tmp1 的形状
            rtmp.shape = tmp1.shape
            # 更新 tmp2[ii, ...] 为重排后的 rtmp
            tmp2[ii, ...] = rtmp
        
        # 更新 ft 为重排后的 tmp2
        ft = tmp2

    # 构造并返回结果列表
    result = []
    
    # 如果需要返回距离并且不是原地操作，则将 dt 添加到结果列表中
    if return_distances and not dt_inplace:
        result.append(dt)
    
    # 如果需要返回索引并且不是原地操作，则将 ft 添加到结果列表中
    if return_indices and not ft_inplace:
        result.append(ft)

    # 根据结果列表的长度进行返回
    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None
# 定义一个函数，用于执行 chamfer 类型的距离变换
def distance_transform_cdt(input, metric='chessboard', return_distances=True,
                           return_indices=False, distances=None, indices=None):
    """
    Distance transform for chamfer type of transforms.

    This function calculates the distance transform of the `input`, by
    replacing each foreground (non-zero) element, with its
    shortest distance to the background (any zero-valued element).

    In addition to the distance transform, the feature transform can
    be calculated. In this case the index of the closest background
    element to each foreground element is returned in a separate array.

    Parameters
    ----------
    input : array_like
        Input. Values of 0 are treated as background.
    metric : {'chessboard', 'taxicab'} or array_like, optional
        The `metric` determines the type of chamfering that is done. If the
        `metric` is equal to 'taxicab' a structure is generated using
        `generate_binary_structure` with a squared distance equal to 1. If
        the `metric` is equal to 'chessboard', a `metric` is generated
        using `generate_binary_structure` with a squared distance equal to
        the dimensionality of the array. These choices correspond to the
        common interpretations of the 'taxicab' and the 'chessboard'
        distance metrics in two dimensions.
        A custom metric may be provided, in the form of a matrix where
        each dimension has a length of three.
        'cityblock' and 'manhattan' are also valid, and map to 'taxicab'.
        The default is 'chessboard'.
    return_distances : bool, optional
        Whether to calculate the distance transform.
        Default is True.
    return_indices : bool, optional
        Whether to calculate the feature transform.
        Default is False.
    distances : int32 ndarray, optional
        An output array to store the calculated distance transform, instead of
        returning it.
        `return_distances` must be True.
        It must be the same shape as `input`.
    indices : int32 ndarray, optional
        An output array to store the calculated feature transform, instead of
        returning it.
        `return_indicies` must be True.
        Its shape must be ``(input.ndim,) + input.shape``.

    Returns
    -------
    distances : int32 ndarray, optional
        The calculated distance transform. Returned only when
        `return_distances` is True, and `distances` is not supplied.
        It will have the same shape as the input array.
    indices : int32 ndarray, optional
        The calculated feature transform. It has an input-shaped array for each
        dimension of the input. See distance_transform_edt documentation for an
        example.
        Returned only when `return_indices` is True, and `indices` is not
        supplied.

    See Also
    --------
    distance_transform_edt : Fast distance transform for euclidean metric
    """
    # 函数体内没有实际的代码，仅仅是函数的文档字符串提供了详细的参数说明和返回值说明
    pass
    # 利用较慢的暴力算法计算不同度量标准下的距离变换

    Examples
    --------
    # 导入必要的模块
    >>> import numpy as np
    >>> from scipy.ndimage import distance_transform_cdt
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.axes_grid1 import ImageGrid

    # 首先，我们创建一个简单的二进制图像

    >>> def add_circle(center_x, center_y, radius, image, fillvalue=1):
    ...     # 填充圆形区域为1
    ...     xx, yy = np.mgrid[:image.shape[0], :image.shape[1]]
    ...     circle = (xx - center_x) ** 2 + (yy - center_y) ** 2
    ...     circle_shape = np.sqrt(circle) < radius
    ...     image[circle_shape] = fillvalue
    ...     return image
    >>> image = np.zeros((100, 100), dtype=np.uint8)
    >>> image[35:65, 20:80] = 1
    >>> image = add_circle(28, 65, 10, image)
    >>> image = add_circle(37, 30, 10, image)
    >>> image = add_circle(70, 45, 20, image)
    >>> image = add_circle(45, 80, 10, image)

    # 接下来，设置图形布局

    >>> fig = plt.figure(figsize=(5, 15))
    >>> grid = ImageGrid(fig, 111, nrows_ncols=(3, 1), axes_pad=(0.5, 0.3),
    ...                  label_mode="1", share_all=True,
    ...                  cbar_location="right", cbar_mode="each",
    ...                  cbar_size="7%", cbar_pad="2%")
    >>> for ax in grid:
    ...     ax.axis('off')
    >>> top, middle, bottom = grid
    >>> colorbar_ticks = [0, 10, 20]

    # 顶部图像包含原始二进制图像

    >>> binary_image = top.imshow(image, cmap='gray')
    >>> cbar_binary_image = top.cax.colorbar(binary_image)
    >>> cbar_binary_image.set_ticks([0, 1])
    >>> top.set_title("Binary image: foreground in white")

    # 中间图像包含使用“taxicab”度量标准的距离变换

    >>> distance_taxicab = distance_transform_cdt(image, metric="taxicab")
    >>> taxicab_transform = middle.imshow(distance_taxicab, cmap='gray')
    >>> cbar_taxicab = middle.cax.colorbar(taxicab_transform)
    >>> cbar_taxicab.set_ticks(colorbar_ticks)
    >>> middle.set_title("Taxicab metric")

    # 底部图像包含使用“chessboard”度量标准的距离变换

    >>> distance_chessboard = distance_transform_cdt(image,
    ...                                              metric="chessboard")
    >>> chessboard_transform = bottom.imshow(distance_chessboard, cmap='gray')
    >>> cbar_chessboard = bottom.cax.colorbar(chessboard_transform)
    >>> cbar_chessboard.set_ticks(colorbar_ticks)
    >>> bottom.set_title("Chessboard metric")
    >>> plt.tight_layout()
    >>> plt.show()

    """
    # 检查是否为 ndarray 类型，判断是否进行原地修改的距离变换参数检查
    ft_inplace = isinstance(indices, np.ndarray)
    dt_inplace = isinstance(distances, np.ndarray)
    _distance_tranform_arg_check(
        dt_inplace, ft_inplace, return_distances, return_indices
    )
    # 将输入转换为 ndarray 类型
    input = np.asarray(input)
    # 检查 metric 是否为字符串类型
    if isinstance(metric, str):
        # 如果 metric 是 'taxicab', 'cityblock', 或者 'manhattan' 中的一种
        if metric in ['taxicab', 'cityblock', 'manhattan']:
            # 获取输入数组的维度
            rank = input.ndim
            # 生成一个二进制结构的 metric
            metric = generate_binary_structure(rank, 1)
        # 如果 metric 是 'chessboard'
        elif metric == 'chessboard':
            # 获取输入数组的维度
            rank = input.ndim
            # 生成一个与输入数组维度相同的二进制结构的 metric
            metric = generate_binary_structure(rank, rank)
        else:
            # 如果 metric 不是上述任何一种，则抛出错误
            raise ValueError('invalid metric provided')
    else:
        try:
            # 尝试将 metric 转换为 NumPy 数组
            metric = np.asarray(metric)
        except Exception as e:
            # 如果转换失败，则抛出错误
            raise ValueError('invalid metric provided') from e
        # 检查 metric 的每个维度大小是否为 3
        for s in metric.shape:
            if s != 3:
                raise ValueError('metric sizes must be equal to 3')

    # 如果 metric 不是连续的，则复制一份连续的 metric
    if not metric.flags.contiguous:
        metric = metric.copy()

    # 如果 dt_inplace 为 True
    if dt_inplace:
        # 检查 distances 数组的数据类型是否为 int32
        if distances.dtype.type != np.int32:
            raise ValueError('distances must be of int32 type')
        # 检查 distances 数组的形状是否与 input 数组相同
        if distances.shape != input.shape:
            raise ValueError('distances has wrong shape')
        # 将 dt 设置为 distances 的一个副本，其中非 input 元素为 -1，其余为 0，并转换为 int32 类型
        dt = distances
        dt[...] = np.where(input, -1, 0).astype(np.int32)
    else:
        # 将 dt 设置为一个新数组，其中非 input 元素为 -1，其余为 0，并转换为 int32 类型
        dt = np.where(input, -1, 0).astype(np.int32)

    # 获取 dt 的维度
    rank = dt.ndim

    # 如果 return_indices 为 True
    if return_indices:
        # 创建一个与 dt 大小相同的索引数组 ft，数据类型为 int32
        ft = np.arange(dt.size, dtype=np.int32)
        ft.shape = dt.shape
    else:
        # 否则，将 ft 设为 None
        ft = None

    # 调用 _nd_image.distance_transform_op 函数，对 dt 和 ft 进行距离变换操作
    _nd_image.distance_transform_op(metric, dt, ft)

    # 将 dt 沿所有维度反转
    dt = dt[tuple([slice(None, None, -1)] * rank)]

    # 如果 return_indices 为 True
    if return_indices:
        # 将 ft 沿所有维度反转
        ft = ft[tuple([slice(None, None, -1)] * rank)]

    # 再次调用 _nd_image.distance_transform_op 函数，对 dt 和 ft 进行距离变换操作
    _nd_image.distance_transform_op(metric, dt, ft)

    # 将 dt 沿所有维度反转
    dt = dt[tuple([slice(None, None, -1)] * rank)]

    # 如果 return_indices 为 True
    if return_indices:
        # 将 ft 沿所有维度反转
        ft = ft[tuple([slice(None, None, -1)] * rank)]
        # 将 ft 展平为一维数组
        ft = np.ravel(ft)
        # 如果 ft_inplace 为 True
        if ft_inplace:
            # 检查 indices 数组的数据类型是否为 int32
            if indices.dtype.type != np.int32:
                raise ValueError('indices array must be int32')
            # 检查 indices 数组的形状是否为 (dt.ndim, dt.shape)
            if indices.shape != (dt.ndim,) + dt.shape:
                raise ValueError('indices array has wrong shape')
            # 将 tmp 设置为 indices 的一个副本
            tmp = indices
        else:
            # 否则，创建一个与 dt 形状相同的索引数组 tmp，数据类型为 int32
            tmp = np.indices(dt.shape, dtype=np.int32)
        # 对 tmp 的每个维度进行迭代
        for ii in range(tmp.shape[0]):
            # 根据 ft 重新排列 tmp 中的每个维度
            rtmp = np.ravel(tmp[ii, ...])[ft]
            rtmp.shape = dt.shape
            tmp[ii, ...] = rtmp
        # 将 ft 设置为 tmp
        ft = tmp

    # 构造并返回结果
    result = []
    # 如果 return_distances 为 True 且 dt_inplace 为 False，则将 dt 加入结果列表
    if return_distances and not dt_inplace:
        result.append(dt)
    # 如果 return_indices 为 True 且 ft_inplace 为 False，则将 ft 加入结果列表
    if return_indices and not ft_inplace:
        result.append(ft)

    # 根据结果列表的长度返回结果
    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None
# 定义精确的欧几里得距离变换函数，用于计算输入数组的距离变换
def distance_transform_edt(input, sampling=None, return_distances=True,
                           return_indices=False, distances=None, indices=None):
    """
    Exact Euclidean distance transform.

    This function calculates the distance transform of the `input`, by
    replacing each foreground (non-zero) element, with its
    shortest distance to the background (any zero-valued element).

    In addition to the distance transform, the feature transform can
    be calculated. In this case the index of the closest background
    element to each foreground element is returned in a separate array.

    Parameters
    ----------
    input : array_like
        Input data to transform. Can be any type but will be converted
        into binary: 1 wherever input equates to True, 0 elsewhere.
    sampling : float, or sequence of float, optional
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the input rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.
    return_distances : bool, optional
        Whether to calculate the distance transform.
        Default is True.
    return_indices : bool, optional
        Whether to calculate the feature transform.
        Default is False.
    distances : float64 ndarray, optional
        An output array to store the calculated distance transform, instead of
        returning it.
        `return_distances` must be True.
        It must be the same shape as `input`.
    indices : int32 ndarray, optional
        An output array to store the calculated feature transform, instead of
        returning it.
        `return_indices` must be True.
        Its shape must be ``(input.ndim,) + input.shape``.

    Returns
    -------
    distances : float64 ndarray, optional
        The calculated distance transform. Returned only when
        `return_distances` is True and `distances` is not supplied.
        It will have the same shape as the input array.
    indices : int32 ndarray, optional
        The calculated feature transform. It has an input-shaped array for each
        dimension of the input. See example below.
        Returned only when `return_indices` is True and `indices` is not
        supplied.

    Notes
    -----
    The Euclidean distance transform gives values of the Euclidean
    distance::

                    n
      y_i = sqrt(sum (x[i]-b[i])**2)
                    i

    where b[i] is the background point (value 0) with the smallest
    Euclidean distance to input points x[i], and n is the
    number of dimensions.

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.array(([0,1,1,1,1],
    ...               [0,0,1,1,1],
    ...               [0,1,1,1,1],
    ...               [0,1,1,1,0],
    ...               [0,1,1,0,0]))
    >>> ndimage.distance_transform_edt(a)
    """
    检查是否在原地输出中处理索引数组和距离数组
    """
    ft_inplace = isinstance(indices, np.ndarray)
    dt_inplace = isinstance(distances, np.ndarray)
    _distance_tranform_arg_check(
        dt_inplace, ft_inplace, return_distances, return_indices
    )

    """
    将输入数据转换为至少一维的数组，其中True转换为1，False转换为0，并强制转换为int8类型
    """
    input = np.atleast_1d(np.where(input, 1, 0).astype(np.int8))

    """
    如果定义了采样，将其标准化为与输入数据维度相匹配的序列，并将其转换为float64类型的数组
    """
    if sampling is not None:
        sampling = _ni_support._normalize_sequence(sampling, input.ndim)
        sampling = np.asarray(sampling, dtype=np.float64)
        if not sampling.flags.contiguous:
            sampling = sampling.copy()

    """
    如果需要原地输出，将输出数组设为indices；检查indices数组的形状是否与输入数据维度相匹配，否则抛出运行时错误；检查indices数组的数据类型是否为int32，否则抛出运行时错误
    否则，创建一个与输入数据维度相匹配的int32类型的零数组
    """
    if ft_inplace:
        ft = indices
        if ft.shape != (input.ndim,) + input.shape:
            raise RuntimeError('indices array has wrong shape')
        if ft.dtype.type != np.int32:
            raise RuntimeError('indices array must be int32')
    else:
        ft = np.zeros((input.ndim,) + input.shape, dtype=np.int32)

    """
    调用底层函数计算欧几里得特征转换
    """
    _nd_image.euclidean_feature_transform(input, sampling, ft)
    
    """
    如果需要距离转换的输出，进行距离转换计算
    """
    # if requested, calculate the distance transform
    # 如果需要返回距离信息
    if return_distances:
        # 计算每个像素点到图像中心的距离
        dt = ft - np.indices(input.shape, dtype=ft.dtype)
        dt = dt.astype(np.float64)
        # 如果指定了采样参数，对距离进行加权
        if sampling is not None:
            for ii in range(len(sampling)):
                dt[ii, ...] *= sampling[ii]
        # 计算每个像素点的距离平方
        np.multiply(dt, dt, dt)
        # 如果要求原地操作
        if dt_inplace:
            # 沿着指定轴求和距离平方
            dt = np.add.reduce(dt, axis=0)
            # 检查距离数组形状是否正确
            if distances.shape != dt.shape:
                raise RuntimeError('distances array has wrong shape')
            # 检查距离数组数据类型是否为 float64
            if distances.dtype.type != np.float64:
                raise RuntimeError('distances array must be float64')
            # 对每个像素点距离应用平方根操作
            np.sqrt(dt, distances)
        else:
            # 沿着指定轴求和距离平方，并对结果应用平方根操作
            dt = np.add.reduce(dt, axis=0)
            dt = np.sqrt(dt)

    # 构建并返回结果列表
    result = []
    # 如果需要返回距离信息且不是原地操作
    if return_distances and not dt_inplace:
        result.append(dt)
    # 如果需要返回特征索引信息且不是原地操作
    if return_indices and not ft_inplace:
        result.append(ft)

    # 根据结果列表长度返回结果
    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None
# 检查距离变换函数的参数是否有效，如果无效则引发 RuntimeError 异常
def _distance_tranform_arg_check(distances_out, indices_out,
                                 return_distances, return_indices):
    """Raise a RuntimeError if the arguments are invalid"""
    # 存储错误消息的列表
    error_msgs = []
    # 如果 return_distances 和 return_indices 都为 False，则添加错误消息
    if (not return_distances) and (not return_indices):
        error_msgs.append(
            'at least one of return_distances/return_indices must be True')
    # 如果 distances_out 为 True 且 return_distances 为 False，则添加错误消息
    if distances_out and not return_distances:
        error_msgs.append(
            'return_distances must be True if distances is supplied'
        )
    # 如果 indices_out 为 True 且 return_indices 为 False，则添加错误消息
    if indices_out and not return_indices:
        error_msgs.append('return_indices must be True if indices is supplied')
    # 如果有任何错误消息存在，则抛出 RuntimeError 异常，将所有错误消息连接成一个字符串
    if error_msgs:
        raise RuntimeError(', '.join(error_msgs))
```