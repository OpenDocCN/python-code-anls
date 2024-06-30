# `D:\src\scipysrc\scipy\scipy\ndimage\_ni_docstrings.py`

```
"""Docstring components common to several ndimage functions."""
# 导入doccer模块，用于处理文档字符串的工具函数
from scipy._lib import doccer

# 声明公共的文档字符串组件列表，用于多个ndimage函数
__all__ = ['docfiller']

# 输入参数的文档字符串
_input_doc = (
"""input : array_like
    The input array.""")
# 轴参数的文档字符串
_axis_doc = (
"""axis : int, optional
    The axis of `input` along which to calculate. Default is -1.""")
# 输出参数的文档字符串
_output_doc = (
"""output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.""")
# 尺寸和footprint参数的文档字符串
_size_foot_doc = (
"""size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.""")
# 反射模式的文档字符串
_mode_reflect_doc = (
"""mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel. This mode is also sometimes referred to as half-sample
        symmetric.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel. This mode is also sometimes referred to as whole-sample
        symmetric.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.

    For consistency with the interpolation functions, the following mode
    names can also be used:

    'grid-mirror'
        This is a synonym for 'reflect'.

    'grid-constant'
        This is a synonym for 'constant'.

    'grid-wrap'
        This is a synonym for 'wrap'.""")
# 插值和常量模式的文档字符串
_mode_interp_constant_doc = (
"""mode : {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', \
'mirror', 'grid-wrap', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    # 'reflect' (`d c b a | a b c d | d c b a`)
    # 输入被反射到最后一个像素的边缘。有时也称为半采样对称。
    
    # 'grid-mirror'
    # 'reflect' 的同义词。
    
    # 'constant' (`k k k k | a b c d | k k k k`)
    # 输入通过用常数值填充超出边缘的所有位置，常数值由 `cval` 参数定义。在输入边缘之外不执行插值。
    
    # 'grid-constant' (`k k k k | a b c d | k k k k`)
    # 输入通过用常数值填充超出边缘的所有位置，常数值由 `cval` 参数定义。超出输入范围的样本也进行插值。
    
    # 'nearest' (`a a a a | a b c d | d d d d`)
    # 输入通过复制最后一个像素来扩展。
    
    # 'mirror' (`d c b | a b c d | c b a`)
    # 输入通过在最后一个像素的中心反射来扩展。有时也称为整采样对称。
    
    # 'grid-wrap' (`a b c d | a b c d | a b c d`)
    # 输入通过绕到对立边缘来扩展。
    
    # 'wrap' (`d b c d | a b c d | b c a b`)
    # 输入通过绕到对立边缘来扩展，但以使最后一个点和初始点完全重叠的方式。在这种情况下，在重叠点选择哪个样本是不明确的。
# 创建一个新的文档字符串，将 _mode_interp_constant_doc 中的默认值 'constant' 替换为 'mirror'
_mode_interp_mirror_doc = (
    _mode_interp_constant_doc.replace("Default is 'constant'",
                                      "Default is 'mirror'")
)
# 使用断言检查 _mode_interp_mirror_doc 是否确实被替换为新的文档字符串，如果没有则抛出异常
assert _mode_interp_mirror_doc != _mode_interp_constant_doc, \
    'Default not replaced'

# 定义 mode 参数的文档字符串，描述了如何处理图像边缘的不同模式
_mode_multiple_doc = (
"""mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel. This mode is also sometimes referred to as half-sample
        symmetric.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel. This mode is also sometimes referred to as whole-sample
        symmetric.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.

    For consistency with the interpolation functions, the following mode
    names can also be used:

    'grid-constant'
        This is a synonym for 'constant'.

    'grid-mirror'
        This is a synonym for 'reflect'.

    'grid-wrap'
        This is a synonym for 'wrap'.""")
# 定义 cval 参数的文档字符串，描述了在 mode 为 'constant' 时填充边缘的常量值
_cval_doc = (
"""cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.""")

# 定义 origin 参数的文档字符串，描述了滤波器在输入数组像素上的放置控制
_origin_doc = (
"""origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.""")

# 定义 origin 参数的多维文档字符串，描述了在多维输入数组中每个轴上不同的放置控制
_origin_multiple_doc = (
"""origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.""")

# 定义 extra_arguments 参数的文档字符串，描述了传递给传入函数的额外位置参数的序列
_extra_arguments_doc = (
"""extra_arguments : sequence, optional
    Sequence of extra positional arguments to pass to passed function.""")

# 定义 extra_keywords 参数的文档字符串，描述了传递给传入函数的额外关键字参数的字典
_extra_keywords_doc = (
"""extra_keywords : dict, optional
    dict of extra keyword arguments to pass to passed function.""")

# 定义 prefilter 参数的文档字符串，描述了是否对输入数组进行预过滤的确定
_prefilter_doc = (
"""prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if ``order > 1``. If
    setting this to False, the output will be slightly blurred if
    ``order > 1``, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.
# 创建一个字典 `docdict`，用于存储不同关键词对应的文档字符串
docdict = {
    'input': _input_doc,  # 'input' 关键词对应的文档字符串
    'axis': _axis_doc,  # 'axis' 关键词对应的文档字符串
    'output': _output_doc,  # 'output' 关键词对应的文档字符串
    'size_foot': _size_foot_doc,  # 'size_foot' 关键词对应的文档字符串
    'mode_interp_constant': _mode_interp_constant_doc,  # 'mode_interp_constant' 关键词对应的文档字符串
    'mode_interp_mirror': _mode_interp_mirror_doc,  # 'mode_interp_mirror' 关键词对应的文档字符串
    'mode_reflect': _mode_reflect_doc,  # 'mode_reflect' 关键词对应的文档字符串
    'mode_multiple': _mode_multiple_doc,  # 'mode_multiple' 关键词对应的文档字符串
    'cval': _cval_doc,  # 'cval' 关键词对应的文档字符串
    'origin': _origin_doc,  # 'origin' 关键词对应的文档字符串
    'origin_multiple': _origin_multiple_doc,  # 'origin_multiple' 关键词对应的文档字符串
    'extra_arguments': _extra_arguments_doc,  # 'extra_arguments' 关键词对应的文档字符串
    'extra_keywords': _extra_keywords_doc,  # 'extra_keywords' 关键词对应的文档字符串
    'prefilter': _prefilter_doc  # 'prefilter' 关键词对应的文档字符串
}

# 使用 doccer.filldoc 函数，将 docdict 中的文档字符串填充到一个新的对象 docfiller 中
docfiller = doccer.filldoc(docdict)
```