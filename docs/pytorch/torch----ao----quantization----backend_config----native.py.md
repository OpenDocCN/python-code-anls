# `.\pytorch\torch\ao\quantization\backend_config\native.py`

```
# mypy: allow-untyped-defs
# 导入 torch 库，用于深度学习相关操作
import torch
# 从 _common_operator_config_utils 中导入多个函数，用于获取不同操作的配置信息
from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_bn_configs,
    _get_cat_config,
    _get_conv_configs,
    _get_default_op_configs,
    _get_embedding_op_configs,
    _get_fixed_qparams_op_configs,
    _get_linear_configs,
    _get_ln_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
    _get_tensor_info_op_configs,
)
# 导入 BackendConfig 和 DTypeConfig 类
from .backend_config import BackendConfig, DTypeConfig

# __all__ 列表定义了模块公开的接口名称
__all__ = [
    "get_test_only_legacy_native_backend_config",
    "default_op_quint8_dtype_config",
    "default_op_fp16_dtype_config",
    "default_dynamic_int8_dtype_config",
    "default_dynamic_float16_dtype_config",
    "input_output_only_quint8_dtype_config",
    "weight_only_quint8_dtype_config",
    "weight_only_quint4x2_dtype_config",
    "get_native_backend_config",
    "get_native_backend_config_dict",
    "get_test_only_legacy_native_backend_config_dict",
]

# ===================
# |  DTYPE CONFIGS  |
# ===================

# weighted op int8 dtype config
# 这是用于具有量化权重的操作（如线性、卷积）的配置
weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

# 默认的 op quint8 dtype config
default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

# 默认的 op fp16 dtype config
default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float16,
    weight_dtype=torch.float16,
    bias_dtype=torch.float16,
)

# 默认的动态 int8 dtype config
default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    # 当前 dtype 检查尚未启用，提供了 dtype_configs 但实际上尚未使用，
    # 我们稍后将在将所有内容移到 backend_config_dict 后启用它
    is_dynamic=True,
)

# 默认的动态 float16 dtype config
default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    # 当前 dtype 检查尚未启用，提供了 dtype_configs 但实际上尚未使用，
    # 我们稍后将在将所有内容移到 backend_config_dict 后启用它
    is_dynamic=True,
)

# LayerNorm 和 f.layer_norm 需要的配置，因为当前内核仅支持 float 权重
input_output_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.float,
    bias_dtype=torch.float,
)

# 仅权重为 quint8 的 dtype config
weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

# 仅权重为 quint4x2 的 dtype config
weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint4x2,
)

# =====================
# |  BACKEND CONFIGS  |
# =====================

# 在这里添加后续的 BackendConfig 相关定义
def get_test_only_legacy_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) with various additional fp16 ops.
    """
    # 定义卷积运算的数据类型配置列表
    conv_dtype_configs = [weighted_op_quint8_dtype_config]
    # 定义线性运算的数据类型配置列表，包括几种不同的数据类型
    linear_dtype_configs = [
        weighted_op_quint8_dtype_config,
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
        default_op_fp16_dtype_config,
    ]
    # 定义二元运算的数据类型配置列表，包括几种不同的数据类型
    binary_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config,
    ]
    # 定义默认操作的数据类型配置列表，仅包括一个数据类型
    default_op_dtype_configs = [default_op_quint8_dtype_config]
    # 定义固定量化参数操作的数据类型配置列表，包括几种不同的数据类型
    fixed_qparams_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config,
    ]
    # 定义共享量化参数操作的数据类型配置列表，包括几种不同的数据类型
    share_qparams_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config
    ]
    # 定义张量信息操作的数据类型配置列表，仅包括一个数据类型
    tensor_info_op_dtype_configs = [
        default_op_quint8_dtype_config,
    ]
    # 定义循环神经网络操作的数据类型配置列表，包括几种不同的数据类型
    rnn_op_dtype_configs = [
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    # 定义嵌入操作的数据类型配置列表，包括几种不同的数据类型
    embedding_op_dtype_configs = [
        weight_only_quint8_dtype_config,
        weight_only_quint4x2_dtype_config,
    ]
    # 定义层归一化操作的数据类型配置列表，仅包括一个数据类型
    layer_norm_op_dtype_configs = [input_output_only_quint8_dtype_config]
    # 返回一个 BackendConfig 对象，设置了多个不同模式的后端配置
    return BackendConfig("_native_and_fp16") \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)) \
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_tensor_info_op_configs(tensor_info_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_ln_configs(layer_norm_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_embedding_op_configs(embedding_op_dtype_configs))

def get_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack).
    """
    # TODO: 将此 BackendConfig 表达为 FBGEMM 和 QNNPACK BackendConfigs 的联合
    # 定义卷积运算的数据类型配置列表
    conv_dtype_configs = [weighted_op_quint8_dtype_config]
    # 定义线性运算的数据类型配置列表，包括几种不同的数据类型
    linear_dtype_configs = [
        weighted_op_quint8_dtype_config,
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    # 定义二元运算的数据类型配置列表，仅包括一个数据类型
    binary_op_dtype_configs = [default_op_quint8_dtype_config]
    # 定义默认操作的数据类型配置列表，仅包括一个数据类型
    default_op_dtype_configs = [default_op_quint8_dtype_config]
    # 定义一个固定量化参数操作的数据类型配置列表，包含一个默认的固定量化参数操作的数据类型配置
    fixed_qparams_op_dtype_configs = [default_op_quint8_dtype_config]
    # 定义一个共享量化参数操作的数据类型配置列表，包含一个默认的共享量化参数操作的数据类型配置
    share_qparams_op_dtype_configs = [default_op_quint8_dtype_config]
    # 定义一个张量信息操作的数据类型配置列表，包含一个默认的张量信息操作的数据类型配置
    tensor_info_op_dtype_configs = [default_op_quint8_dtype_config]
    # 定义一个循环神经网络操作的数据类型配置列表，包含动态整数8位和动态浮点数16位的默认数据类型配置
    rnn_op_dtype_configs = [
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    # 定义一个嵌入操作的数据类型配置列表，包含仅权重8位量化和权重4x2位量化的默认数据类型配置
    embedding_op_dtype_configs = [
        weight_only_quint8_dtype_config,
        weight_only_quint4x2_dtype_config,
    ]
    # 定义一个层归一化操作的数据类型配置列表，包含仅输入输出8位量化的默认数据类型配置
    layer_norm_op_dtype_configs = [input_output_only_quint8_dtype_config]
    # 返回一个 BackendConfig 对象，其后续调用链配置了不同操作模式的数据类型配置
    return BackendConfig("native") \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \  # 设置卷积操作的模式配置
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)) \  # 设置线性操作的模式配置
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \  # 设置二元操作的模式配置
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs)) \  # 设置 cat 操作的模式配置
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs)) \  # 设置默认操作的模式配置
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)) \  # 设置固定量化参数操作的模式配置
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \  # 设置共享量化参数操作的模式配置
        .set_backend_pattern_configs(_get_tensor_info_op_configs(tensor_info_op_dtype_configs)) \  # 设置张量信息操作的模式配置
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs)) \  # 设置批归一化操作的模式配置
        .set_backend_pattern_configs(_get_ln_configs(layer_norm_op_dtype_configs)) \  # 设置层归一化操作的模式配置
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs)) \  # 设置循环神经网络操作的模式配置
        .set_backend_pattern_configs(_get_embedding_op_configs(embedding_op_dtype_configs))  # 设置嵌入操作的模式配置
# 返回 PyTorch 原生后端（fbgemm/qnnpack）的BackendConfig对象，以字典形式表示
def get_native_backend_config_dict():
    # 调用函数获取 PyTorch 原生后端的BackendConfig对象，并将其转换为字典
    return get_native_backend_config().to_dict()

# 返回带有额外fp16操作的PyTorch原生后端（fbgemm/qnnpack）的BackendConfig对象，以字典形式表示
def get_test_only_legacy_native_backend_config_dict():
    # 调用函数获取带有额外fp16操作的PyTorch原生后端的BackendConfig对象，并将其转换为字典
    return get_test_only_legacy_native_backend_config().to_dict()
```