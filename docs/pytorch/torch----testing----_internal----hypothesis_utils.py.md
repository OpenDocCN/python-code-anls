# `.\pytorch\torch\testing\_internal\hypothesis_utils.py`

```
# 忽略 mypy 的类型检查错误
# 导入必要的库和模块
from collections import defaultdict  # 导入 defaultdict 类
from collections.abc import Iterable  # 导入 Iterable 抽象类
import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库，用于深度学习

import hypothesis  # 导入 Hypothesis 库，用于测试
from functools import reduce  # 导入 reduce 函数，用于函数式编程
from hypothesis import assume  # 导入 assume 函数，用于假设条件
from hypothesis import settings  # 导入 settings 模块，用于配置测试策略
from hypothesis import strategies as st  # 导入 strategies 模块，并重命名为 st，用于定义测试数据生成策略
from hypothesis.extra import numpy as stnp  # 导入额外的 NumPy 支持模块
from hypothesis.strategies import SearchStrategy  # 导入 SearchStrategy 类，用于定义测试数据生成策略

from torch.testing._internal.common_quantized import _calculate_dynamic_qparams, _calculate_dynamic_per_channel_qparams  # 导入内部测试函数

# 为 Hypothesis 测试设置准备
# `_ALL_QINT_TYPES` 是包含所有量化数据类型的元组
# 其中包含 torch.quint8、torch.qint8 和 torch.qint32
_ALL_QINT_TYPES = (
    torch.quint8,
    torch.qint8,
    torch.qint32,
)

# `_ENFORCED_ZERO_POINT` 是一个 defaultdict，默认情况下返回 None
# 用于指定每种量化数据类型的强制零点值
# 对于 torch.quint8 和 torch.qint8，零点值为任意值
# 对于 torch.qint32，零点值强制为 0
_ENFORCED_ZERO_POINT = defaultdict(lambda: None, {
    torch.quint8: None,
    torch.qint8: None,
    torch.qint32: 0
})

def _get_valid_min_max(qparams):
    # 获取量化参数 qparams 中的 scale、zero_point 和 quantized_type
    scale, zero_point, quantized_type = qparams
    # 计算修正值，使用 torch.finfo(torch.float).eps
    adjustment = 1 + torch.finfo(torch.float).eps
    # 获取 torch.long 的信息
    _long_type_info = torch.iinfo(torch.long)
    long_min, long_max = _long_type_info.min / adjustment, _long_type_info.max / adjustment
    # 确保中间结果在 long 类型的范围内
    min_value = max((long_min - zero_point) * scale, (long_min / scale + zero_point))
    max_value = min((long_max - zero_point) * scale, (long_max / scale + zero_point))
    return np.float32(min_value), np.float32(max_value)

# `_floats_wrapper` 函数是 `st.floats` 的包装器函数
# 检查 Hypothesis 版本，如果版本过旧，则移除 `width` 参数
# `width` 参数是在 3.67.0 版本中引入的
def _floats_wrapper(*args, **kwargs):
    # 检查是否在 kwargs 中存在 'width' 参数，并且当前 hypothesis 版本低于 (3, 67, 0)
    if 'width' in kwargs and hypothesis.version.__version_info__ < (3, 67, 0):
        # 只要未指定 nan、inf、min、max，重新实现 width 参数逻辑以支持旧版本的 hypothesis
        no_nan_and_inf = (
            (('allow_nan' in kwargs and not kwargs['allow_nan']) or
             'allow_nan' not in kwargs) and
            (('allow_infinity' in kwargs and not kwargs['allow_infinity']) or
             'allow_infinity' not in kwargs))
        min_and_max_not_specified = (
            len(args) == 0 and
            'min_value' not in kwargs and
            'max_value' not in kwargs
        )
        # 如果不允许 nan 和 inf，并且未指定 min_value 和 max_value
        if no_nan_and_inf and min_and_max_not_specified:
            # 根据指定的 width 参数设置对应的 min_value 和 max_value
            if kwargs['width'] == 16:
                kwargs['min_value'] = torch.finfo(torch.float16).min
                kwargs['max_value'] = torch.finfo(torch.float16).max
            elif kwargs['width'] == 32:
                kwargs['min_value'] = torch.finfo(torch.float32).min
                kwargs['max_value'] = torch.finfo(torch.float32).max
            elif kwargs['width'] == 64:
                kwargs['min_value'] = torch.finfo(torch.float64).min
                kwargs['max_value'] = torch.finfo(torch.float64).max
        # 移除处理过的 'width' 参数
        kwargs.pop('width')
    
    # 返回使用给定参数生成的浮点数的统计测试策略
    return st.floats(*args, **kwargs)
"""Define a function to generate floating point numbers with specified width.

Args:
    *args: Positional arguments passed to `_floats_wrapper`.
    **kwargs: Keyword arguments passed to `_floats_wrapper`, including 'width' if provided.

Returns:
    Result of `_floats_wrapper` function call with provided arguments and keyword arguments.
"""
def floats(*args, **kwargs):
    # Check if 'width' keyword argument is not provided, set it to 32
    if 'width' not in kwargs:
        kwargs['width'] = 32
    # Call _floats_wrapper with the provided arguments and keyword arguments
    return _floats_wrapper(*args, **kwargs)

"""Hypothesis filter to avoid overflows with quantized tensors.

Args:
    tensor: Tensor of floats to filter
    qparams: Quantization parameters as returned by the `qparams`.

Returns:
    True

Raises:
    hypothesis.UnsatisfiedAssumption

Note: This filter is slow. Use it only when filtering of the test cases is
      absolutely necessary!
"""
def assume_not_overflowing(tensor, qparams):
    # Retrieve minimum and maximum valid values from quantization parameters
    min_value, max_value = _get_valid_min_max(qparams)
    # Assume that tensor's minimum value is greater than or equal to min_value
    assume(tensor.min() >= min_value)
    # Assume that tensor's maximum value is less than or equal to max_value
    assume(tensor.max() <= max_value)
    # Return True to indicate assumptions are met
    return True

"""Strategy for generating the quantization parameters.

Args:
    dtypes: Quantized data types to sample from.
    scale_min / scale_max: Minimum and maximum scales. If None, set to 1e-3 / 1e3.
    zero_point_min / zero_point_max: Minimum and maximum for the zero point. If None,
        set to the minimum and maximum of the quantized data type.
        Note: The min and max are only valid if the zero_point is not enforced
              by the data type itself.

Generates:
    scale: Sampled scale.
    zero_point: Sampled zero point.
    quantized_type: Sampled quantized type.
"""
@st.composite
def qparams(draw, dtypes=None, scale_min=None, scale_max=None,
            zero_point_min=None, zero_point_max=None):
    # If dtypes is not provided, use all quantized integer types
    if dtypes is None:
        dtypes = _ALL_QINT_TYPES
    # Convert dtypes to tuple if it's not already a list or tuple
    if not isinstance(dtypes, (list, tuple)):
        dtypes = (dtypes,)
    # Draw a random quantized type from dtypes
    quantized_type = draw(st.sampled_from(dtypes))

    # Get information about the chosen quantized type
    _type_info = torch.iinfo(quantized_type)
    qmin, qmax = _type_info.min, _type_info.max

    # Determine enforced zero point from predefined constants
    _zp_enforced = _ENFORCED_ZERO_POINT[quantized_type]
    if _zp_enforced is not None:
        zero_point = _zp_enforced
    else:
        # Draw a random integer for zero point within specified range
        _zp_min = qmin if zero_point_min is None else zero_point_min
        _zp_max = qmax if zero_point_max is None else zero_point_max
        zero_point = draw(st.integers(min_value=_zp_min, max_value=_zp_max))

    # Set default values for scale if not provided
    if scale_min is None:
        scale_min = torch.finfo(torch.float).eps
    if scale_max is None:
        scale_max = torch.finfo(torch.float).max
    # Draw a random float for scale within specified range
    scale = draw(floats(min_value=scale_min, max_value=scale_max, width=32))

    # Return sampled scale, zero point, and quantized type
    return scale, zero_point, quantized_type

"""Strategy to create different shapes.

Args:
    min_dims / max_dims: Minimum and maximum rank (number of dimensions).
    min_side / max_side: Minimum and maximum dimensions per rank.

Generates:
    Possible shapes for a tensor, constrained by the rank and dimensionality.

Example:
    # Generates 3D and 4D tensors.
    @given(Q = qtensor(shapes=array_shapes(min_dims=3, max_dims=4))
    some_test(self, Q):...
"""
@st.composite
def array_shapes(draw, min_dims=1, max_dims=None, min_side=1, max_side=None, max_numel=None):
    """Return a strategy for array shapes (tuples of int >= 1)."""
    # Assert that minimum dimensions are less than 32
    assert min_dims < 32
    # 如果max_dims为None，则将其设为min_dims加2和32中的较小值
    if max_dims is None:
        max_dims = min(min_dims + 2, 32)
    
    # 断言max_dims应小于32，确保维度不会超过设定的上限
    assert max_dims < 32
    
    # 如果max_side为None，则将其设为min_side加5
    if max_side is None:
        max_side = min_side + 5
    
    # 生成一个候选列表，其中元素为min_side到max_side之间的整数，列表长度为min_dims到max_dims之间
    candidate = st.lists(st.integers(min_side, max_side), min_size=min_dims, max_size=max_dims)
    
    # 如果max_numel不为None，则过滤掉候选列表中元素乘积大于max_numel的部分
    if max_numel is not None:
        candidate = candidate.filter(lambda x: reduce(int.__mul__, x, 1) <= max_numel)
    
    # 从候选列表中随机选择一个元素，并将其映射为元组返回
    return draw(candidate.map(tuple))
@st.composite
def tensor(draw, shapes=None, elements=None, qparams=None, dtype=np.float32):
    """
    生成张量测试用例的策略。
    返回的张量格式为float32。

    Args:
        shapes: 张量的测试形状。可以是假设策略，也可以是要从中采样的不同形状的可迭代对象。
        elements: 生成返回数据类型的元素。
                  如果为None，则策略解析为范围[-1e6, 1e6]内的浮点数。
        qparams: qparams策略的实例。用于过滤张量，以避免溢出。
                 如果设置了`qparams`参数，则返回张量的量化参数。
                 返回的参数是`(scale, zero_point, quantization_type)`。
                 如果`qparams`参数为None，则返回None。

    Generates:
        X: 类型为float32的张量。注意不包括NaN和+/-inf。
        qparams: (如果设置了`qparams`参数) X的量化参数。
                 返回的参数是`(scale, zero_point, quantization_type)`。
                 (如果`qparams`参数为None)，则返回None。
    """
    if isinstance(shapes, SearchStrategy):
        _shape = draw(shapes)
    else:
        _shape = draw(st.sampled_from(shapes))

    if qparams is None:
        if elements is None:
            elements = floats(-1e6, 1e6, allow_nan=False, width=32)
        X = draw(stnp.arrays(dtype=dtype, elements=elements, shape=_shape))
        assume(not (np.isnan(X).any() or np.isinf(X).any()))
        return X, None

    qparams = draw(qparams)

    if elements is None:
        min_value, max_value = _get_valid_min_max(qparams)
        elements = floats(min_value, max_value, allow_infinity=False,
                          allow_nan=False, width=32)

    X = draw(stnp.arrays(dtype=dtype, elements=elements, shape=_shape))

    # 根据X的统计数据重新计算scale和zero_points。
    scale, zp = _calculate_dynamic_qparams(X, qparams[2])

    enforced_zp = _ENFORCED_ZERO_POINT.get(qparams[2], None)
    if enforced_zp is not None:
        zp = enforced_zp

    return X, (scale, zp, qparams[2])


@st.composite
def per_channel_tensor(draw, shapes=None, elements=None, qparams=None):
    """
    生成通道级别量化的张量测试用例的策略。

    Args:
        shapes: 张量的测试形状。可以是假设策略，也可以是要从中采样的不同形状的可迭代对象。
        elements: 生成返回数据类型的元素。
                  如果为None，则策略解析为范围[-1e6, 1e6]内的浮点数。
        qparams: qparams策略的实例。用于过滤张量，以避免溢出。
                 返回的参数是`(scale, zero_point, quantization_type)`。

    Generates:
        X: 类型为float32的张量。注意不包括NaN和+/-inf。
        qparams: (如果设置了`qparams`参数) X的通道级别量化参数。
                 返回的参数是`(scale, zero_point, quantization_type)`。
    """
    if isinstance(shapes, SearchStrategy):
        _shape = draw(shapes)
    else:
        _shape = draw(st.sampled_from(shapes))

    if qparams is None:
        if elements is None:
            elements = floats(-1e6, 1e6, allow_nan=False, width=32)
        X = draw(stnp.arrays(dtype=np.float32, elements=elements, shape=_shape))
        assume(not (np.isnan(X).any() or np.isinf(X).any()))
        return X, None

    qparams = draw(qparams)

    if elements is None:
        min_value, max_value = _get_valid_min_max(qparams)
        elements = floats(min_value, max_value, allow_infinity=False,
                          allow_nan=False, width=32)

    X = draw(stnp.arrays(dtype=np.float32, elements=elements, shape=_shape))

    # 根据X的统计数据重新计算scale和zero_points。
    scale, zp = _calculate_dynamic_per_channel_qparams(X, qparams[2])

    enforced_zp = _ENFORCED_ZERO_POINT.get(qparams[2], None)
    if enforced_zp is not None:
        zp = enforced_zp

    return X, (scale, zp, qparams[2])
    # 随机选择一个整数作为轴，用于模拟沿特定轴的量化操作
    axis = int(np.random.randint(0, X.ndim, 1))
    # 创建一个包含当前数据维度的排列轴数组
    permute_axes = np.arange(X.ndim)
    # 将第一个轴设置为随机选择的轴
    permute_axes[0] = axis
    # 将随机选择的轴设置为第一个轴
    permute_axes[axis] = 0
    # 使用指定的轴顺序对数据进行转置
    X = np.transpose(X, permute_axes)

    # 返回转置后的数据以及其他相关的量化参数
    return X, (scale, zp, axis, qparams[2])
"""Strategy for generating test cases for tensors used in Conv.
The resulting tensors is in float32 format.

Args:
    spatial_dim: Spatial Dim for feature maps. If given as an iterable, randomly
                 picks one from the pool to make it the spatial dimension
    batch_size_range: Range to generate `batch_size`.
                      Must be tuple of `(min, max)`.
    input_channels_per_group_range:
        Range to generate `input_channels_per_group`.
        Must be tuple of `(min, max)`.
    output_channels_per_group_range:
        Range to generate `output_channels_per_group`.
        Must be tuple of `(min, max)`.
    feature_map_range: Range to generate feature map size for each spatial_dim.
                       Must be tuple of `(min, max)`.
    kernel_range: Range to generate kernel size for each spatial_dim. Must be
                  tuple of `(min, max)`.
    max_groups: Maximum number of groups to generate.
    elements: Elements to generate from for the returned data type.
              If None, the strategy resolves to float within range [-1e6, 1e6].
    qparams: Strategy for quantization parameters. for X, w, and b.
             Could be either a single strategy (used for all) or a list of
             three strategies for X, w, b.
Generates:
    (X, W, b, g): Tensors of type `float32` of the following drawen shapes:
        X: (`batch_size, input_channels, H, W`)
        W: (`output_channels, input_channels_per_group) + kernel_shape
        b: `(output_channels,)`
        groups: Number of groups the input is divided into
Note: X, W, b are tuples of (Tensor, qparams), where qparams could be either
      None or (scale, zero_point, quantized_type)


Example:
    @given(tensor_conv(
        spatial_dim=2,
        batch_size_range=(1, 3),
        input_channels_per_group_range=(1, 7),
        output_channels_per_group_range=(1, 7),
        feature_map_range=(6, 12),
        kernel_range=(3, 5),
        max_groups=4,
        elements=st.floats(-1.0, 1.0),
        qparams=qparams()
    ))
"""
@st.composite
def tensor_conv(
    draw, spatial_dim=2, batch_size_range=(1, 4),
    input_channels_per_group_range=(3, 7),
    output_channels_per_group_range=(3, 7), feature_map_range=(6, 12),
    kernel_range=(3, 7), max_groups=1, can_be_transposed=False,
    elements=None, qparams=None
):
    """
    Composite strategy function for generating convolutional test tensors.

    Args:
        draw: Function to draw random values from strategies.
        spatial_dim: Dimensionality of spatial features (int or iterable).
        batch_size_range: Range for batch size generation.
        input_channels_per_group_range: Range for input channels per group.
        output_channels_per_group_range: Range for output channels per group.
        feature_map_range: Range for spatial feature map sizes.
        kernel_range: Range for kernel sizes.
        max_groups: Maximum number of groups for group convolutions.
        can_be_transposed: Boolean indicating if tensors can be transposed.
        elements: Range of elements for data type.
        qparams: Quantization parameters strategy.

    Generates:
        (X, W, b, groups): Tuple of tensors and group information.

    Returns:
        Composite function for generating convolutional test tensors.
    """
    
    # Resolve the minibatch, in_channels, out_channels, iH/iW, iK/iW
    batch_size = draw(st.integers(*batch_size_range))
    input_channels_per_group = draw(
        st.integers(*input_channels_per_group_range))
    output_channels_per_group = draw(
        st.integers(*output_channels_per_group_range))
    groups = draw(st.integers(1, max_groups))
    input_channels = input_channels_per_group * groups
    output_channels = output_channels_per_group * groups

    if isinstance(spatial_dim, Iterable):
        spatial_dim = draw(st.sampled_from(spatial_dim))

    feature_map_shape = []
    # 遍历空间维度，生成特征图形状的各维度大小，并添加到列表中
    for i in range(spatial_dim):
        feature_map_shape.append(draw(st.integers(*feature_map_range)))
    
    # 创建一个空列表用于存储卷积核的大小
    kernels = []
    # 遍历空间维度，生成每个卷积核的大小，并添加到列表中
    for i in range(spatial_dim):
        kernels.append(draw(st.integers(*kernel_range)))
    
    # 初始化一个变量，表示是否进行转置操作，默认为 False
    tr = False
    # 计算权重的形状，包括输出通道数、每组输入通道数以及卷积核的大小
    weight_shape = (output_channels, input_channels_per_group) + tuple(kernels)
    # 计算偏置的形状，仅包括输出通道数
    bias_shape = output_channels
    # 如果可以进行转置操作，则随机确定是否进行转置
    if can_be_transposed:
        tr = draw(st.booleans())
        if tr:
            # 如果进行转置，则更新权重的形状和偏置的形状
            weight_shape = (input_channels, output_channels_per_group) + tuple(kernels)
            bias_shape = output_channels
    
    # 处理量化参数，确保传入的量化参数格式正确
    if qparams is not None:
        if isinstance(qparams, (list, tuple)):
            # 如果 qparams 是列表或元组，则需确保包含三个量化参数
            assert len(qparams) == 3, "Need 3 qparams for X, w, b"
        else:
            # 如果 qparams 是单个量化参数，则复制为包含三个相同参数的列表
            qparams = [qparams] * 3
    
    # 生成输入张量 X，权重张量 W，偏置张量 b，并返回对应的组数和是否转置的标志 tr
    X = draw(tensor(shapes=(
        (batch_size, input_channels) + tuple(feature_map_shape),),
        elements=elements, qparams=qparams[0]))
    W = draw(tensor(shapes=(weight_shape,), elements=elements,
                    qparams=qparams[1]))
    b = draw(tensor(shapes=(bias_shape,), elements=elements,
                    qparams=qparams[2]))
    
    # 返回生成的输入张量 X，权重张量 W，偏置张量 b，组数以及是否进行了转置操作的标志 tr
    return X, W, b, groups, tr
# 设置假设（Hypothesis）版本号
hypothesis_version = hypothesis.version.__version_info__
# 获取当前配置文件的设置信息
current_settings = settings._profiles[settings._current_profile].__dict__
# 将当前配置文件的截止时间设为 None，即禁用截止时间
current_settings['deadline'] = None

# 如果假设版本号在3.16.0到5.0.0之间
if hypothesis_version >= (3, 16, 0) and hypothesis_version < (5, 0, 0):
    # 将当前配置文件的超时设置为无限制
    current_settings['timeout'] = hypothesis.unlimited

# 定义一个函数用来断言截止时间是否被禁用
def assert_deadline_disabled():
    # 如果假设版本低于3.27.0
    if hypothesis_version < (3, 27, 0):
        # 导入警告模块
        import warnings
        # 构建警告消息
        warning_message = (
            "Your version of hypothesis is outdated. "
            "To avoid `DeadlineExceeded` errors, please update. "
            f"Current hypothesis version: {hypothesis.__version__}"
        )
        # 发出警告
        warnings.warn(warning_message)
    else:
        # 断言当前设置的截止时间为 None，即已禁用截止时间
        assert settings().deadline is None
```