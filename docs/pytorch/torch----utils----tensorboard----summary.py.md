# `.\pytorch\torch\utils\tensorboard\summary.py`

```
# 设置类型映射表，将 torch.Tensor 类型映射到对应的 Protobuf 类型及其值字段
_TENSOR_TYPE_MAP = {
    torch.half: ("DT_HALF", "half_val", _tensor_to_half_val),  # 半精度浮点数映射为 Protobuf 的半精度类型及其值
    torch.float16: ("DT_HALF", "half_val", _tensor_to_half_val),  # 同上，支持另一种命名方式
    torch.bfloat16: ("DT_BFLOAT16", "half_val", _tensor_to_half_val),  # BF16 浮点数映射为 Protobuf 的 BF16 类型及其值
    torch.float32: ("DT_FLOAT", "float_val", _tensor_to_list),  # 单精度浮点数映射为 Protobuf 的单精度类型及其值
    torch.float: ("DT_FLOAT", "float_val", _tensor_to_list),  # 同上，支持另一种命名方式
    torch.float64: ("DT_DOUBLE", "double_val", _tensor_to_list),  # 双精度浮点数映射为 Protobuf 的双精度类型及其值
    torch.double: ("DT_DOUBLE", "double_val", _tensor_to_list),  # 同上，支持另一种命名方式
    torch.int8: ("DT_INT8", "int_val", _tensor_to_list),  # 8 位整数映射为 Protobuf 的 8 位整数类型及其值
    torch.uint8: ("DT_UINT8", "int_val", _tensor_to_list),  # 无符号 8 位整数映射为 Protobuf 的 8 位整数类型及其值
    torch.qint8: ("DT_UINT8", "int_val", _tensor_to_list),  # 同上，支持另一种命名方式
    torch.int16: ("DT_INT16", "int_val", _tensor_to_list),  # 16 位整数映射为 Protobuf 的 16 位整数类型及其值
    torch.short: ("DT_INT16", "int_val", _tensor_to_list),  # 同上，支持另一种命名方式
}
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数
    torch.int: ("DT_INT32", "int_val", _tensor_to_list),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数
    torch.int32: ("DT_INT32", "int_val", _tensor_to_list),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数
    torch.qint32: ("DT_INT32", "int_val", _tensor_to_list),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数
    torch.int64: ("DT_INT64", "int64_val", _tensor_to_list),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数，处理复数类型为 32 位
    torch.complex32: ("DT_COMPLEX32", "scomplex_val", _tensor_to_complex_val),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数，处理复数类型为 16 位浮点数
    torch.chalf: ("DT_COMPLEX32", "scomplex_val", _tensor_to_complex_val),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数，处理复数类型为 64 位
    torch.complex64: ("DT_COMPLEX64", "scomplex_val", _tensor_to_complex_val),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数，处理复数类型为 32 位浮点数
    torch.cfloat: ("DT_COMPLEX64", "scomplex_val", _tensor_to_complex_val),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数，处理布尔类型
    torch.bool: ("DT_BOOL", "bool_val", _tensor_to_list),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数，处理复数类型为 128 位
    torch.complex128: ("DT_COMPLEX128", "dcomplex_val", _tensor_to_complex_val),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数，处理复数类型为 64 位浮点数
    torch.cdouble: ("DT_COMPLEX128", "dcomplex_val", _tensor_to_complex_val),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数，处理无符号 8 位整数类型
    torch.uint8: ("DT_UINT8", "uint32_val", _tensor_to_list),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数，处理无符号 8 位整数类型
    torch.quint8: ("DT_UINT8", "uint32_val", _tensor_to_list),
    # 将 torch 数据类型映射到 TensorFlow 数据类型及相应的处理函数，处理无符号 4x2 位整数类型
    torch.quint4x2: ("DT_UINT8", "uint32_val", _tensor_to_list),
}

# 计算张量的缩放因子，根据类型判断返回值
def _calc_scale_factor(tensor):
    # 如果张量不是 NumPy 数组，则转换为 NumPy 数组
    converted = tensor.numpy() if not isinstance(tensor, np.ndarray) else tensor
    # 如果数组元素类型是 np.uint8，则返回缩放因子 1；否则返回 255
    return 1 if converted.dtype == np.uint8 else 255


# 绘制单个边界框到图像上
def _draw_single_box(
    image,
    xmin,
    ymin,
    xmax,
    ymax,
    display_str,
    color="black",
    color_text="black",
    thickness=2,
):
    from PIL import ImageDraw, ImageFont

    # 加载默认字体
    font = ImageFont.load_default()
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    # 定义边界框的坐标
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    # 绘制边界框
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )
    # 如果有显示字符串
    if display_str:
        # 计算文字显示位置
        text_bottom = bottom
        _left, _top, _right, _bottom = font.getbbox(display_str)
        text_width, text_height = _right - _left, _bottom - _top
        margin = np.ceil(0.05 * text_height)
        # 绘制文字背景框
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        # 绘制文字内容
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill=color_text,
            font=font,
        )
    return image


# 输出与 TensorBoard 的超参数插件相关的 Summary protobufs
def hparams(hparam_dict=None, metric_dict=None, hparam_domain_discrete=None):
    """Output three `Summary` protocol buffers needed by hparams plugin.

    `Experiment` keeps the metadata of an experiment, such as the name of the
      hyperparameters and the name of the metrics.
    `SessionStartInfo` keeps key-value pairs of the hyperparameters
    `SessionEndInfo` describes status of the experiment e.g. STATUS_SUCCESS

    Args:
      hparam_dict: A dictionary that contains names of the hyperparameters
        and their values.
      metric_dict: A dictionary that contains names of the metrics
        and their values.
      hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
        contains names of the hyperparameters and all discrete values they can hold

    Returns:
      The `Summary` protobufs for Experiment, SessionStartInfo and
        SessionEndInfo
    """
    import torch
    from tensorboard.plugins.hparams.api_pb2 import (
        DataType,
        Experiment,
        HParamInfo,
        MetricInfo,
        MetricName,
        Status,
    )
    from tensorboard.plugins.hparams.metadata import (
        EXPERIMENT_TAG,
        PLUGIN_DATA_VERSION,
        PLUGIN_NAME,
        SESSION_END_INFO_TAG,
        SESSION_START_INFO_TAG,
    )
    from tensorboard.plugins.hparams.plugin_data_pb2 import (
        HParamsPluginData,
        SessionEndInfo,
        SessionStartInfo,
    )

    # TODO: 在将来暴露其它参数
    # hp = HParamInfo(name='lr',display_name='learning rate',
    # type=DataType.DATA_TYPE_FLOAT64, domain_interval=Interval(min_value=10,
    # max_value=100))
    # 检查 hparam_dict 是否为字典类型，如果不是则记录警告并抛出类型错误异常
    if not isinstance(hparam_dict, dict):
        logger.warning("parameter: hparam_dict should be a dictionary, nothing logged.")
        raise TypeError(
            "parameter: hparam_dict should be a dictionary, nothing logged."
        )
    
    # 检查 metric_dict 是否为字典类型，如果不是则记录警告并抛出类型错误异常
    if not isinstance(metric_dict, dict):
        logger.warning("parameter: metric_dict should be a dictionary, nothing logged.")
        raise TypeError(
            "parameter: metric_dict should be a dictionary, nothing logged."
        )
    
    # 如果 hparam_domain_discrete 为 None，则设为一个空字典
    hparam_domain_discrete = hparam_domain_discrete or {}
    
    # 检查 hparam_domain_discrete 是否为字典类型，如果不是则抛出类型错误异常
    if not isinstance(hparam_domain_discrete, dict):
        raise TypeError(
            "parameter: hparam_domain_discrete should be a dictionary, nothing logged."
        )
    
    # 遍历 hparam_domain_discrete 的键值对，检查每个键是否存在于 hparam_dict 中，
    # 值是否为列表类型，并且列表中的每个元素是否与 hparam_dict 中对应键的类型相同，
    # 如果不符合条件则抛出类型错误异常
    for k, v in hparam_domain_discrete.items():
        if (
            k not in hparam_dict
            or not isinstance(v, list)
            or not all(isinstance(d, type(hparam_dict[k])) for d in v)
        ):
            raise TypeError(
                f"parameter: hparam_domain_discrete[{k}] should be a list of same type as hparam_dict[{k}]."
            )
    
    # 初始化一个空列表 hps 用于存储后续处理的内容
    hps = []
    
    # 创建一个 SessionStartInfo 的实例对象 ssi
    ssi = SessionStartInfo()
    #`
    # 遍历 hparam_dict 字典中的每个键值对
    for k, v in hparam_dict.items():
        # 如果值 v 为 None，跳过该循环迭代
        if v is None:
            continue
        # 如果值 v 是整数或浮点数类型
        if isinstance(v, (int, float)):
            # 将整数或浮点数类型的值赋给 ssi.hparams 的对应键的 number_value 属性
            ssi.hparams[k].number_value = v

            # 检查当前键 k 是否在离散域参数列表中
            if k in hparam_domain_discrete:
                # 如果在，创建一个 ListValue 对象，包含离散域参数中的数值
                domain_discrete: Optional[struct_pb2.ListValue] = struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(number_value=d)
                        for d in hparam_domain_discrete[k]
                    ]
                )
            else:
                domain_discrete = None

            # 将当前参数信息添加到 hps 列表中，指定数据类型为 DATA_TYPE_FLOAT64 和离散域参数
            hps.append(
                HParamInfo(
                    name=k,
                    type=DataType.Value("DATA_TYPE_FLOAT64"),
                    domain_discrete=domain_discrete,
                )
            )
            continue

        # 如果值 v 是字符串类型
        if isinstance(v, str):
            # 将字符串类型的值赋给 ssi.hparams 的对应键的 string_value 属性
            ssi.hparams[k].string_value = v

            # 检查当前键 k 是否在离散域参数列表中
            if k in hparam_domain_discrete:
                # 如果在，创建一个 ListValue 对象，包含离散域参数中的字符串值
                domain_discrete = struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(string_value=d)
                        for d in hparam_domain_discrete[k]
                    ]
                )
            else:
                domain_discrete = None

            # 将当前参数信息添加到 hps 列表中，指定数据类型为 DATA_TYPE_STRING 和离散域参数
            hps.append(
                HParamInfo(
                    name=k,
                    type=DataType.Value("DATA_TYPE_STRING"),
                    domain_discrete=domain_discrete,
                )
            )
            continue

        # 如果值 v 是布尔类型
        if isinstance(v, bool):
            # 将布尔类型的值赋给 ssi.hparams 的对应键的 bool_value 属性
            ssi.hparams[k].bool_value = v

            # 检查当前键 k 是否在离散域参数列表中
            if k in hparam_domain_discrete:
                # 如果在，创建一个 ListValue 对象，包含离散域参数中的布尔值
                domain_discrete = struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(bool_value=d)
                        for d in hparam_domain_discrete[k]
                    ]
                )
            else:
                domain_discrete = None

            # 将当前参数信息添加到 hps 列表中，指定数据类型为 DATA_TYPE_BOOL 和离散域参数
            hps.append(
                HParamInfo(
                    name=k,
                    type=DataType.Value("DATA_TYPE_BOOL"),
                    domain_discrete=domain_discrete,
                )
            )
            continue

        # 如果值 v 是 torch.Tensor 类型
        if isinstance(v, torch.Tensor):
            # 将 torch.Tensor 转换为 numpy 数组
            v = make_np(v)[0]
            # 将转换后的数值赋给 ssi.hparams 的对应键的 number_value 属性
            ssi.hparams[k].number_value = v
            # 将当前参数信息添加到 hps 列表中，指定数据类型为 DATA_TYPE_FLOAT64
            hps.append(HParamInfo(name=k, type=DataType.Value("DATA_TYPE_FLOAT64")))
            continue
        # 如果值 v 不是上述任何类型，抛出 ValueError 异常
        raise ValueError(
            "value should be one of int, float, str, bool, or torch.Tensor"
        )

    # 创建 HParamsPluginData 对象，包含 session_start_info 和版本号
    content = HParamsPluginData(session_start_info=ssi, version=PLUGIN_DATA_VERSION)
    # 创建 SummaryMetadata 对象，包含插件数据和插件名称
    smd = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        )
    )
    # 创建 Summary 对象，包含一个值对象，值对象的 tag 为 SESSION_START_INFO_TAG，元数据为 smd
    ssi = Summary(value=[Summary.Value(tag=SESSION_START_INFO_TAG, metadata=smd)])

    # 创建 MetricInfo 对象列表，包含 metric_dict 中所有键的 MetricInfo 对象
    mts = [MetricInfo(name=MetricName(tag=k)) for k in metric_dict.keys()]

    # 创建 Experiment 对象，包含 hparam_infos 和 metric_infos
    exp = Experiment(hparam_infos=hps, metric_infos=mts)

    # 创建 HParamsPluginData 对象，包含 experiment 和版本号
    content = HParamsPluginData(experiment=exp, version=PLUGIN_DATA_VERSION)
    # 创建 SummaryMetadata 对象，用于存储插件数据的元数据
    smd = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        )
    )

    # 创建 Summary 对象，用于存储实验数据的摘要信息
    exp = Summary(value=[Summary.Value(tag=EXPERIMENT_TAG, metadata=smd)])

    # 创建 SessionEndInfo 对象，表示会话结束的状态信息
    sei = SessionEndInfo(status=Status.Value("STATUS_SUCCESS"))

    # 创建 HParamsPluginData 对象，用于存储超参数插件数据
    content = HParamsPluginData(session_end_info=sei, version=PLUGIN_DATA_VERSION)

    # 更新 SummaryMetadata 对象，用于存储插件数据的元数据，包括插件名称和序列化后的数据内容
    smd = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        )
    )

    # 创建 Summary 对象，用于存储会话结束信息的摘要数据
    sei = Summary(value=[Summary.Value(tag=SESSION_END_INFO_TAG, metadata=smd)])

    # 返回三个对象：实验摘要数据 (exp)、会话摘要信息 (ssi)、会话结束摘要信息 (sei)
    return exp, ssi, sei
# 输出一个包含单个标量值的 `Summary` 协议缓冲区。
#
# 生成的 Summary 包含一个包含输入 Tensor 的 Tensor.proto。
#
# Args:
#   name: 生成节点的名称。也将作为 TensorBoard 中的系列名称。
#   tensor: 包含单个值的实数型 Tensor。
#   collections: 可选的图集合键列表。新的 summary 操作会添加到这些集合中。默认为 `[GraphKeys.SUMMARIES]`。
#   new_style: 是否使用新样式（tensor 字段）或旧样式（simple_value 字段）。新样式可能会导致更快的数据加载。
#
# Returns:
#   一个类型为 `string` 的标量 `Tensor`，其中包含一个 `Summary` 协议缓冲区。
#
# Raises:
#   ValueError: 如果 tensor 的形状或类型不正确。
def scalar(name, tensor, collections=None, new_style=False, double_precision=False):
    # 转换为 NumPy 数组，并挤压成一维数组
    tensor = make_np(tensor).squeeze()
    # 断言，确保 tensor 的维度为 0（只包含一个元素）
    assert (
        tensor.ndim == 0
    ), f"Tensor should contain one element (0 dimensions). Was given size: {tensor.size} and {tensor.ndim} dimensions."
    
    # python 中的 float 在 numpy 中是双精度
    scalar = float(tensor)
    
    # 如果使用新样式
    if new_style:
        # 创建 TensorProto 对象，包含 float_val 或 double_val
        tensor_proto = TensorProto(float_val=[scalar], dtype="DT_FLOAT")
        if double_precision:
            tensor_proto = TensorProto(double_val=[scalar], dtype="DT_DOUBLE")

        # 创建插件数据和摘要元数据
        plugin_data = SummaryMetadata.PluginData(plugin_name="scalars")
        smd = SummaryMetadata(plugin_data=plugin_data)
        
        # 返回一个包含 TensorProto 的 Summary
        return Summary(
            value=[
                Summary.Value(
                    tag=name,
                    tensor=tensor_proto,
                    metadata=smd,
                )
            ]
        )
    else:
        # 返回一个包含 simple_value 的 Summary
        return Summary(value=[Summary.Value(tag=name, simple_value=scalar)])


# 输出一个包含完整 Tensor 的 `Summary` 协议缓冲区。
#
# 生成的 Summary 包含一个包含输入 Tensor 的 Tensor.proto。
#
# Args:
#   tag: 生成节点的名称。也将作为 TensorBoard 中的系列名称。
#   tensor: 要转换为 protobuf 的 Tensor。
#
# Returns:
#   一个 `Summary` 协议缓冲区中的 Tensor protobuf。
#
# Raises:
#   ValueError: 如果 tensor 太大无法转换为 protobuf，或者 tensor 数据类型不受支持。
def tensor_proto(tag, tensor):
    # 如果 tensor 大小超过 2GB 的硬限制，抛出 ValueError
    if tensor.numel() * tensor.itemsize >= (1 << 31):
        raise ValueError(
            "tensor is bigger than protocol buffer's hard limit of 2GB in size"
        )

    # 如果 tensor 数据类型在 _TENSOR_TYPE_MAP 中
    if tensor.dtype in _TENSOR_TYPE_MAP:
        dtype, field_name, conversion_fn = _TENSOR_TYPE_MAP[tensor.dtype]
        
        # 创建 TensorProto 对象，包含 dtype、tensor_shape 和 field_name
        tensor_proto = TensorProto(
            **{
                "dtype": dtype,
                "tensor_shape": TensorShapeProto(
                    dim=[TensorShapeProto.Dim(size=x) for x in tensor.shape]
                ),
                field_name: conversion_fn(tensor),
            },
        )
    else:
        # 如果代码执行到这里，说明传入的张量数据类型不受支持，抛出值错误异常
        raise ValueError(f"{tag} has unsupported tensor dtype {tensor.dtype}")

    # 创建插件数据对象，指定插件名称为"tensor"
    plugin_data = SummaryMetadata.PluginData(plugin_name="tensor")
    # 创建摘要元数据对象，将插件数据对象作为参数传入
    smd = SummaryMetadata(plugin_data=plugin_data)
    # 返回一个摘要对象，其中包含了标签、元数据和张量协议缓冲区
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor_proto)])
# 定义一个函数，生成包含直方图的 `Summary` 协议缓冲区。

def histogram_raw(name, min, max, num, sum, sum_squares, bucket_limits, bucket_counts):
    # pylint: disable=line-too-long
    """Output a `Summary` protocol buffer with a histogram.

    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      min: A float or int min value
      max: A float or int max value
      num: Int number of values
      sum: Float or int sum of all values
      sum_squares: Float or int sum of squares for all values
      bucket_limits: A numeric `Tensor` with upper value per bucket
      bucket_counts: A numeric `Tensor` with number of values per bucket
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    # 创建一个直方图的协议缓冲区 `HistogramProto`
    hist = HistogramProto(
        min=min,
        max=max,
        num=num,
        sum=sum,
        sum_squares=sum_squares,
        bucket_limit=bucket_limits,
        bucket=bucket_counts,
    )
    # 将直方图协议缓冲区添加到 `Summary` 值中，并返回
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


# 定义一个函数，生成包含直方图的 `Summary` 协议缓冲区。

def histogram(name, values, bins, max_bins=None):
    # pylint: disable=line-too-long
    """Output a `Summary` protocol buffer with a histogram.

    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.
    This op reports an `InvalidArgument` error if any value is not finite.
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      values: A real numeric `Tensor`. Any shape. Values to use to
        build the histogram.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    # 将输入的 `values` 转换成 numpy 数组
    values = make_np(values)
    # 根据输入的值、bins 和最大 bin 数量，生成直方图数据
    hist = make_histogram(values.astype(float), bins, max_bins)
    # 将直方图协议缓冲区添加到 `Summary` 值中，并返回
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


# 根据输入的值，使用 histogram.cc 中的逻辑生成直方图协议缓冲区。

def make_histogram(values, bins, max_bins=None):
    """Convert values into a histogram proto using logic from histogram.cc."""
    # 如果输入的值数量为 0，则抛出 ValueError 异常
    if values.size == 0:
        raise ValueError("The input has no element.")
    # 将输入的值重新调整为一维数组
    values = values.reshape(-1)
    # 使用 numpy 的 `histogram` 函数生成直方图的 counts 和 limits
    counts, limits = np.histogram(values, bins=bins)
    # 计算 bins 的数量
    num_bins = len(counts)
    # 如果指定了最大的箱子数，并且实际箱子数超过最大值，则进行子采样以减少箱子数
    if max_bins is not None and num_bins > max_bins:
        subsampling = num_bins // max_bins  # 计算子采样的步长
        subsampling_remainder = num_bins % subsampling  # 计算子采样余数
        if subsampling_remainder != 0:
            # 如果余数不为零，使用零填充 counts 数组，以便能够整除 subsampling
            counts = np.pad(
                counts,
                pad_width=[[0, subsampling - subsampling_remainder]],
                mode="constant",
                constant_values=0,
            )
        # 将 counts 数组重塑为子采样后的形状，并对每组进行求和，以减少箱子数
        counts = counts.reshape(-1, subsampling).sum(axis=-1)
        
        # 根据新的 counts 数组重新计算边界 limits
        new_limits = np.empty((counts.size + 1,), limits.dtype)
        new_limits[:-1] = limits[:-1:subsampling]  # 按照子采样步长重新赋值 limits 的上限
        new_limits[-1] = limits[-1]  # 最后一个边界保持不变
        limits = new_limits  # 更新 limits

    # 查找直方图支持的第一个和最后一个箱子的索引范围
    cum_counts = np.cumsum(np.greater(counts, 0))  # 计算大于零的 counts 累积和
    start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")  # 查找起始和结束索引
    start = int(start)  # 起始索引转为整数
    end = int(end) + 1  # 结束索引转为整数，并加一表示包含末尾
    del cum_counts  # 删除累积和数组，释放内存

    # TensorBoard 只包含右边界的箱子限制。为了保留左边第一个限制，我们在左侧添加一个空箱子。
    # 如果 start == 0，则需要在左侧添加一个空箱子；否则，直接在第一个非零计数的箱子左侧保留限制。
    counts = (
        counts[start - 1 : end] if start > 0 else np.concatenate([[0], counts[:end]])
    )  # 调整 counts 数组以保留左侧边界的箱子限制
    limits = limits[start : end + 1]  # 根据调整后的索引范围更新 limits

    # 如果 counts 或 limits 的大小为零，抛出 ValueError
    if counts.size == 0 or limits.size == 0:
        raise ValueError("The histogram is empty, please file a bug report.")

    # 计算 values 的平方和
    sum_sq = values.dot(values)
    
    # 返回 HistogramProto 对象，包括统计信息和箱子的限制和计数
    return HistogramProto(
        min=values.min(),
        max=values.max(),
        num=len(values),
        sum=values.sum(),
        sum_squares=sum_sq,
        bucket_limit=limits.tolist(),
        bucket=counts.tolist(),
    )
def image(tag, tensor, rescale=1, dataformats="NCHW"):
    """Output a `Summary` protocol buffer with images.

    The summary has up to `max_images` summary values containing images. The
    images are built from `tensor` which must be 3-D with shape `[height, width,
    channels]` and where `channels` can be:
    *  1: `tensor` is interpreted as Grayscale.
    *  3: `tensor` is interpreted as RGB.
    *  4: `tensor` is interpreted as RGBA.
    The `name` in the outputted Summary.Value protobufs is generated based on the
    name, with a suffix depending on the max_outputs setting:
    *  If `max_outputs` is 1, the summary value tag is '*name*/image'.
    *  If `max_outputs` is greater than 1, the summary value tags are
       generated sequentially as '*name*/image/0', '*name*/image/1', etc.
    Args:
      tag: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 3-D `uint8` or `float32` `Tensor` of shape `[height, width,
        channels]` where `channels` is 1, 3, or 4.
        'tensor' can either have values in [0, 1] (float32) or [0, 255] (uint8).
        The image() function will scale the image values to [0, 255] by applying
        a scale factor of either 1 (uint8) or 255 (float32). Out-of-range values
        will be clipped.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    # Convert the input tensor to NumPy array
    tensor = make_np(tensor)
    # Convert data format to HWC (Height x Width x Channels)
    tensor = convert_to_HWC(tensor, dataformats)
    # Calculate the scale factor based on tensor data type
    scale_factor = _calc_scale_factor(tensor)
    # Convert tensor to float32 and apply scaling, then clip values to [0, 255] and convert to uint8
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).clip(0, 255).astype(np.uint8)
    # Create image summary value using the processed tensor
    image = make_image(tensor, rescale=rescale)
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def image_boxes(
    tag, tensor_image, tensor_boxes, rescale=1, dataformats="CHW", labels=None
):
    """Output a `Summary` protocol buffer with images."""
    # Convert image tensor to NumPy array
    tensor_image = make_np(tensor_image)
    # Convert image data format to HWC (Height x Width x Channels)
    tensor_image = convert_to_HWC(tensor_image, dataformats)
    # Convert boxes tensor to NumPy array
    tensor_boxes = make_np(tensor_boxes)
    # Scale the image tensor values by calculating the scale factor
    tensor_image = tensor_image.astype(np.float32) * _calc_scale_factor(tensor_image)
    # Create image summary value with the image and optional boxes and labels
    image = make_image(
        tensor_image.clip(0, 255).astype(np.uint8),
        rescale=rescale,
        rois=tensor_boxes,
        labels=labels,
    )
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def draw_boxes(disp_image, boxes, labels=None):
    # xyxy format
    # Get the number of boxes from the shape of boxes array
    num_boxes = boxes.shape[0]
    # Create a list of box indices
    list_gt = range(num_boxes)
    # Iterate through each box and draw it on the display image
    for i in list_gt:
        disp_image = _draw_single_box(
            disp_image,
            boxes[i, 0],
            boxes[i, 1],
            boxes[i, 2],
            boxes[i, 3],
            display_str=None if labels is None else labels[i],
            color="Red",
        )
    return disp_image


def make_image(tensor, rescale=1, rois=None, labels=None):
    """
    Create an image from tensor data, optionally overlaying regions of interest (ROIs) and labels.

    Args:
      tensor: A NumPy array representing an image.
      rescale: Optional scale factor for the image.
      rois: Optional regions of interest to overlay on the image.
      labels: Optional labels corresponding to the ROIs.

    Returns:
      An image with optional overlays.
    """
    # Function implementation is missing in provided code snippet
    pass
    """Convert a numpy representation of an image to Image protobuf."""
    # 导入PIL库中的Image模块，用于处理图像
    from PIL import Image
    
    # 获取输入张量的高度、宽度和通道数
    height, width, channel = tensor.shape
    
    # 根据给定的缩放比例计算缩放后的高度和宽度
    scaled_height = int(height * rescale)
    scaled_width = int(width * rescale)
    
    # 将numpy数组表示的图像转换为PIL Image对象
    image = Image.fromarray(tensor)
    
    # 如果提供了感兴趣区域（ROIs），则在图像上绘制边界框和标签
    if rois is not None:
        image = draw_boxes(image, rois, labels=labels)
    
    # 定义图像缩放时使用的抗锯齿方法
    ANTIALIAS = Image.Resampling.LANCZOS
    
    # 缩放图像到指定的尺寸
    image = image.resize((scaled_width, scaled_height), ANTIALIAS)
    
    # 导入io模块，用于处理二进制数据
    import io
    
    # 创建一个字节流对象，用于保存图像的PNG编码结果
    output = io.BytesIO()
    image.save(output, format="PNG")  # 将图像以PNG格式保存到字节流中
    image_string = output.getvalue()  # 获取字节流中的二进制图像数据
    output.close()  # 关闭字节流对象，释放资源
    
    # 构建并返回Summary.Image对象，包含图像的相关信息和编码后的图像数据
    return Summary.Image(
        height=height,
        width=width,
        colorspace=channel,
        encoded_image_string=image_string,
    )
def video(tag, tensor, fps=4):
    tensor = make_np(tensor)  # 将输入张量转换为NumPy数组
    tensor = _prepare_video(tensor)  # 准备视频数据，可能会进行预处理
    # 如果用户传入的是uint8类型的张量，则无需按255进行重新缩放
    scale_factor = _calc_scale_factor(tensor)  # 计算缩放因子
    tensor = tensor.astype(np.float32)  # 将张量转换为float32类型
    tensor = (tensor * scale_factor).clip(0, 255).astype(np.uint8)  # 根据缩放因子进行重新缩放，并限制在0到255之间
    video = make_video(tensor, fps)  # 创建视频摘要
    return Summary(value=[Summary.Value(tag=tag, image=video)])  # 返回摘要信息，包含视频图像


def make_video(tensor, fps):
    try:
        import moviepy  # 导入moviepy包，用于视频处理
    except ImportError:
        print("add_video needs package moviepy")  # 如果导入失败，则打印错误信息
        return  # 返回空值

    try:
        from moviepy import editor as mpy  # 导入moviepy的editor模块
    except ImportError:
        print(
            "moviepy is installed, but can't import moviepy.editor.",
            "Some packages could be missing [imageio, requests]",
        )  # 如果导入失败，则打印错误信息，可能是缺少相关的依赖包
        return  # 返回空值

    import tempfile  # 导入临时文件模块

    t, h, w, c = tensor.shape  # 获取张量的形状信息

    # 将图像序列编码为GIF字符串
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    # 创建临时文件名，用于存储生成的GIF
    filename = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name

    try:  # 尝试写入GIF文件（新版本的moviepy使用logger而不是progress_bar参数）
        clip.write_gif(filename, verbose=False, logger=None)
    except TypeError:
        try:  # 如果上述写入方法失败，则尝试旧版本的写入方法（老版本的moviepy不支持progress_bar参数）
            clip.write_gif(filename, verbose=False, progress_bar=False)
        except TypeError:
            clip.write_gif(filename, verbose=False)

    with open(filename, "rb") as f:
        tensor_string = f.read()  # 读取生成的GIF文件内容为字符串

    try:
        os.remove(filename)  # 尝试删除临时文件
    except OSError:
        logger.warning("The temporary file used by moviepy cannot be deleted.")  # 如果删除失败，则记录警告信息

    return Summary.Image(
        height=h, width=w, colorspace=c, encoded_image_string=tensor_string
    )  # 返回图像的摘要信息，包含高度、宽度、颜色空间和编码后的图像字符串


def audio(tag, tensor, sample_rate=44100):
    array = make_np(tensor)  # 将输入张量转换为NumPy数组
    array = array.squeeze()  # 去除数组中的单维度条目
    if abs(array).max() > 1:
        print("warning: audio amplitude out of range, auto clipped.")  # 如果音频振幅超出范围，则打印警告信息
        array = array.clip(-1, 1)  # 将音频数组的振幅限制在-1到1之间
    assert array.ndim == 1, "input tensor should be 1 dimensional."  # 断言音频数组应为一维数组
    array = (array * np.iinfo(np.int16).max).astype("<i2")  # 将音频数组转换为16位整数格式

    import io  # 导入io模块
    import wave  # 导入wave模块

    fio = io.BytesIO()  # 创建一个字节流对象
    with wave.open(fio, "wb") as wave_write:  # 打开wave文件，以二进制写模式
        wave_write.setnchannels(1)  # 设置声道数为1
        wave_write.setsampwidth(2)  # 设置样本宽度为2字节（16位）
        wave_write.setframerate(sample_rate)  # 设置采样率
        wave_write.writeframes(array.data)  # 将音频数据写入wave文件

    audio_string = fio.getvalue()  # 获取字节流中的音频数据内容
    fio.close()  # 关闭字节流对象

    audio = Summary.Audio(
        sample_rate=sample_rate,  # 设置音频的采样率
        num_channels=1,  # 设置音频的声道数
        length_frames=array.shape[-1],  # 设置音频的帧长度
        encoded_audio_string=audio_string,  # 设置编码后的音频字符串
        content_type="audio/wav",  # 设置音频内容类型为WAV格式
    )

    return Summary(value=[Summary.Value(tag=tag, audio=audio)])  # 返回音频的摘要信息，包含音频对象和标签信息


def custom_scalars(layout):
    categories = []  # 初始化空列表，用于存储类别信息
    # 遍历布局字典中的每个键值对，键是类别标题，值是包含图表信息的字典列表
    for k, v in layout.items():
        charts = []
        # 遍历每个类别下的图表信息
        for chart_name, chart_metadata in v.items():
            # 从图表元数据中获取标签信息
            tags = chart_metadata[1]
            # 如果图表类型是"Margin"
            if chart_metadata[0] == "Margin":
                # 确保标签列表长度为3
                assert len(tags) == 3
                # 创建MarginChartContent对象
                mgcc = layout_pb2.MarginChartContent(
                    series=[
                        layout_pb2.MarginChartContent.Series(
                            value=tags[0], lower=tags[1], upper=tags[2]
                        )
                    ]
                )
                # 创建Chart对象，包含标题和MarginChartContent对象
                chart = layout_pb2.Chart(title=chart_name, margin=mgcc)
            else:
                # 创建MultilineChartContent对象
                mlcc = layout_pb2.MultilineChartContent(tag=tags)
                # 创建Chart对象，包含标题和MultilineChartContent对象
                chart = layout_pb2.Chart(title=chart_name, multiline=mlcc)
            # 将创建的Chart对象添加到charts列表中
            charts.append(chart)
        # 创建Category对象，包含类别标题和charts列表
        categories.append(layout_pb2.Category(title=k, chart=charts))

    # 创建Layout对象，包含类别信息列表
    layout = layout_pb2.Layout(category=categories)
    # 创建PluginData对象，指定插件名称
    plugin_data = SummaryMetadata.PluginData(plugin_name="custom_scalars")
    # 创建SummaryMetadata对象，包含PluginData对象
    smd = SummaryMetadata(plugin_data=plugin_data)
    # 创建TensorProto对象，指定数据类型为字符串，内容为layout对象序列化后的字符串
    tensor = TensorProto(
        dtype="DT_STRING",
        string_val=[layout.SerializeToString()],
        tensor_shape=TensorShapeProto(),
    )
    # 创建Summary对象，包含标签、TensorProto对象和Metadata对象
    return Summary(
        value=[
            Summary.Value(tag="custom_scalars__config__", tensor=tensor, metadata=smd)
        ]
    )
# 定义一个函数，用于创建包含文本摘要的 TensorBoard Summary 对象
def text(tag, text):
    # 创建插件数据对象，指定插件名称和内容，使用序列化后的文本插件数据
    plugin_data = SummaryMetadata.PluginData(
        plugin_name="text", content=TextPluginData(version=0).SerializeToString()
    )
    # 创建摘要元数据对象，包含上述插件数据
    smd = SummaryMetadata(plugin_data=plugin_data)
    # 创建字符串类型的 TensorProto 对象，用于存储编码后的文本数据
    tensor = TensorProto(
        dtype="DT_STRING",
        string_val=[text.encode(encoding="utf_8")],
        tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
    )
    # 返回包含摘要值的 Summary 对象，其中包含标签、元数据和张量数据
    return Summary(
        value=[Summary.Value(tag=tag + "/text_summary", metadata=smd, tensor=tensor)]
    )


# 定义一个函数，用于创建原始 PR 曲线的 TensorBoard Summary 对象
def pr_curve_raw(
    tag, tp, fp, tn, fn, precision, recall, num_thresholds=127, weights=None
):
    # 如果 num_thresholds 大于 127，将其设为 127，因为大于该值会破坏协议缓冲区
    if num_thresholds > 127:  
        num_thresholds = 127
    # 将所有输入数据堆叠为一个 numpy 数组
    data = np.stack((tp, fp, tn, fn, precision, recall))
    # 创建 PR 曲线插件数据，指定版本和阈值数量
    pr_curve_plugin_data = PrCurvePluginData(
        version=0, num_thresholds=num_thresholds
    ).SerializeToString()
    # 创建插件数据对象，指定插件名称和序列化后的 PR 曲线数据
    plugin_data = SummaryMetadata.PluginData(
        plugin_name="pr_curves", content=pr_curve_plugin_data
    )
    # 创建摘要元数据对象，包含上述插件数据
    smd = SummaryMetadata(plugin_data=plugin_data)
    # 创建浮点数类型的 TensorProto 对象，用于存储压平后的数据数组
    tensor = TensorProto(
        dtype="DT_FLOAT",
        float_val=data.reshape(-1).tolist(),
        tensor_shape=TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=data.shape[0]),
                TensorShapeProto.Dim(size=data.shape[1]),
            ]
        ),
    )
    # 返回包含摘要值的 Summary 对象，其中包含标签、元数据和张量数据
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])


# 定义一个函数，用于创建 PR 曲线的 TensorBoard Summary 对象
def pr_curve(tag, labels, predictions, num_thresholds=127, weights=None):
    # 如果 num_thresholds 大于 127，将其设为 127，因为大于该值会破坏协议缓冲区
    num_thresholds = min(num_thresholds, 127)
    # 计算 PR 曲线的数据，调用 compute_curve 函数获取数据
    data = compute_curve(
        labels, predictions, num_thresholds=num_thresholds, weights=weights
    )
    # 创建 PR 曲线插件数据，指定版本和阈值数量
    pr_curve_plugin_data = PrCurvePluginData(
        version=0, num_thresholds=num_thresholds
    ).SerializeToString()
    # 创建插件数据对象，指定插件名称和序列化后的 PR 曲线数据
    plugin_data = SummaryMetadata.PluginData(
        plugin_name="pr_curves", content=pr_curve_plugin_data
    )
    # 创建摘要元数据对象，包含上述插件数据
    smd = SummaryMetadata(plugin_data=plugin_data)
    # 创建浮点数类型的 TensorProto 对象，用于存储压平后的数据数组
    tensor = TensorProto(
        dtype="DT_FLOAT",
        float_val=data.reshape(-1).tolist(),
        tensor_shape=TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=data.shape[0]),
                TensorShapeProto.Dim(size=data.shape[1]),
            ]
        ),
    )
    # 返回包含摘要值的 Summary 对象，其中包含标签、元数据和张量数据
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])


# 定义一个函数，计算 PR 曲线的数据
def compute_curve(labels, predictions, num_thresholds=None, weights=None):
    # 定义一个常量，用于避免除数为零的情况
    _MINIMUM_COUNT = 1e-7

    # 如果权重为空，则设为 1.0
    if weights is None:
        weights = 1.0

    # 计算真正例和假正例的分组
    bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
    float_labels = labels.astype(np.float64)
    histogram_range = (0, num_thresholds - 1)
    # 使用 numpy 计算真正例的直方图，并按权重调整
    tp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=float_labels * weights,
    )
    # 使用 numpy 的直方图函数计算给定 bucket_indices 的频数分布
    # fp_buckets 是存储频数分布结果的数组，_ 是忽略的值
    fp_buckets, _ = np.histogram(
        bucket_indices,                    # bucket_indices：输入的索引数组
        bins=num_thresholds,               # num_thresholds：直方图的箱数
        range=histogram_range,             # histogram_range：直方图的范围
        weights=(1.0 - float_labels) * weights,  # 加权的数组，根据标签和权重计算得出
    
    )
    
    # 计算反向累积和，得到真正正样本的数量
    tp = np.cumsum(tp_buckets[::-1])[::-1]   # tp_buckets 反向累积和的数组，表示真正正样本
    fp = np.cumsum(fp_buckets[::-1])[::-1]   # fp_buckets 反向累积和的数组，表示假正样本
    tn = fp[0] - fp                          # 计算真负样本的数量
    fn = tp[0] - tp                          # 计算假负样本的数量
    
    # 计算精确率和召回率
    precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)  # 精确率的计算公式
    recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)     # 召回率的计算公式
    
    # 返回所有计算结果的堆叠数组
    return np.stack((tp, fp, tn, fn, precision, recall))
# 创建一个函数 `_get_tensor_summary`，用于生成带有摘要元数据的张量摘要。

def _get_tensor_summary(
    name, display_name, description, tensor, content_type, components, json_config
):
    """Create a tensor summary with summary metadata.

    Args:
      name: Uniquely identifiable name of the summary op. Could be replaced by
        combination of name and type to make it unique even outside of this
        summary.
        摘要操作的唯一标识名称。可以通过名称和类型的组合替换，使其在此摘要之外也能保持唯一性。
      display_name: Will be used as the display name in TensorBoard.
        Defaults to `name`.
        在 TensorBoard 中显示的名称，默认为 `name`。
      description: A longform readable description of the summary data. Markdown
        is supported.
        摘要数据的详细可读描述。支持 Markdown 格式。
      tensor: Tensor to display in summary.
        要在摘要中显示的张量。
      content_type: Type of content inside the Tensor.
        张量内部内容的类型。
      components: Bitmask representing present parts (vertices, colors, etc.) that
        belong to the summary.
        表示属于摘要的存在部分（顶点、颜色等）的位掩码。
      json_config: A string, JSON-serialized dictionary of ThreeJS classes
        configuration.
        一个字符串，ThreeJS 类配置的 JSON 序列化字典。

    Returns:
      Tensor summary with metadata.
      带有元数据的张量摘要。
    """

    import torch
    from tensorboard.plugins.mesh import metadata

    # 将输入的张量转换为 PyTorch 张量
    tensor = torch.as_tensor(tensor)

    # 使用 metadata 模块的函数创建张量摘要的元数据
    tensor_metadata = metadata.create_summary_metadata(
        name,
        display_name,
        content_type,
        components,
        tensor.shape,
        description,
        json_config=json_config,
    )

    # 将张量转换为 TensorProto 格式，以便在摘要中使用
    tensor = TensorProto(
        dtype="DT_FLOAT",
        float_val=tensor.reshape(-1).tolist(),
        tensor_shape=TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=tensor.shape[0]),
                TensorShapeProto.Dim(size=tensor.shape[1]),
                TensorShapeProto.Dim(size=tensor.shape[2]),
            ]
        ),
    )

    # 创建 Summary.Value 对象，包含标签、张量数据和元数据
    tensor_summary = Summary.Value(
        tag=metadata.get_instance_name(name, content_type),
        tensor=tensor,
        metadata=tensor_metadata,
    )

    # 返回生成的张量摘要对象
    return tensor_summary


# 创建一个函数 `_get_json_config`，用于将 Python 字典解析并返回 JSON 字符串。

def _get_json_config(config_dict):
    """Parse and returns JSON string from python dictionary.

    Args:
      config_dict: Dictionary containing configuration to be converted to JSON.

    Returns:
      JSON string representing the input dictionary.
      表示输入字典的 JSON 字符串。
    """
    json_config = "{}"
    if config_dict is not None:
        # 如果配置字典不为空，则将其转换为 JSON 字符串（按键排序）
        json_config = json.dumps(config_dict, sort_keys=True)
    return json_config


# https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/mesh/summary.py
# 创建一个函数 `mesh`，用于生成包含网格/点云的合并 `Summary` 协议缓冲区。

def mesh(
    tag, vertices, colors, faces, config_dict, display_name=None, description=None
):
    """Output a merged `Summary` protocol buffer with a mesh/point cloud.

    Args:
      tag: A name for this summary operation.
        此摘要操作的名称。
      vertices: Tensor of shape `[dim_1, ..., dim_n, 3]` representing the 3D
        coordinates of vertices.
        形状为 `[dim_1, ..., dim_n, 3]` 的张量，表示顶点的三维坐标。
      faces: Tensor of shape `[dim_1, ..., dim_n, 3]` containing indices of
        vertices within each triangle.
        形状为 `[dim_1, ..., dim_n, 3]` 的张量，包含每个三角形内顶点的索引。
      colors: Tensor of shape `[dim_1, ..., dim_n, 3]` containing colors for each
        vertex.
        形状为 `[dim_1, ..., dim_n, 3]` 的张量，包含每个顶点的颜色。
      display_name: If set, will be used as the display name in TensorBoard.
        Defaults to `name`.
        如果设置，则用作 TensorBoard 中的显示名称，默认为 `name`。
      description: A longform readable description of the summary data. Markdown
        is supported.
        摘要数据的详细可读描述。支持 Markdown 格式。
      config_dict: Dictionary with ThreeJS classes names and configuration.
        包含 ThreeJS 类名和配置的字典。
    """
    # 导入必要的模块和类
    from tensorboard.plugins.mesh import metadata
    from tensorboard.plugins.mesh.plugin_data_pb2 import MeshPluginData

    # 根据配置字典获取 JSON 配置
    json_config = _get_json_config(config_dict)

    # 初始化空列表用于存储所有的 summary
    summaries = []

    # 筛选非空的张量和它们对应的内容类型
    tensors = [
        (vertices, MeshPluginData.VERTEX),
        (faces, MeshPluginData.FACE),
        (colors, MeshPluginData.COLOR),
    ]
    tensors = [tensor for tensor in tensors if tensor[0] is not None]

    # 获取组件的位掩码，表示每个张量的组件类型
    components = metadata.get_components_bitmask(
        [content_type for (tensor, content_type) in tensors]
    )

    # 遍历筛选后的张量和内容类型列表，生成每个张量的 summary 并添加到 summaries 中
    for tensor, content_type in tensors:
        summaries.append(
            _get_tensor_summary(
                tag,
                display_name,
                description,
                tensor,
                content_type,
                components,
                json_config,
            )
        )

    # 返回所有 summaries 合并后的 Summary 对象
    return Summary(value=summaries)
```