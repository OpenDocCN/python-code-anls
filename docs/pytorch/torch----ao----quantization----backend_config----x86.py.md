# `.\pytorch\torch\ao\quantization\backend_config\x86.py`

```
# 导入 PyTorch 库
import torch
# 导入内部的配置工具模块
from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_bn_configs,
    _get_cat_config,
    _get_conv_configs,
    _get_default_op_configs,
    _get_embedding_op_configs,
    _get_fixed_qparams_op_configs,
    _get_linear_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
    _get_tensor_info_op_configs,
)
# 导入后端配置和数据类型配置
from .backend_config import BackendConfig, DTypeConfig

__all__ = [
    "get_x86_backend_config",
]

# ===================
# |  DTYPE CONFIGS  |
# ===================

# X86 现在与 FBGEMM 对齐

# 定义 X86 权重操作的整数8位数据类型配置
x86_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

# 定义 X86 默认操作的无符号整数8位数据类型配置
x86_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

# 定义 X86 默认操作的浮点16位数据类型配置
x86_default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float16,
    weight_dtype=torch.float16,
    bias_dtype=torch.float16,
)

# 定义 X86 默认动态整数8位数据类型配置
x86_default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

# 定义 X86 默认动态浮点16位数据类型配置
x86_default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    is_dynamic=True,
)

# 定义 X86 仅权重为无符号整数8位数据类型配置
x86_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

# 定义 X86 仅权重为无符号四位每两位数据类型配置
x86_weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint4x2,
)


# =====================
# |  BACKEND CONFIGS  |
# =====================

# 定义获取 x86 后端配置的函数，返回 BackendConfig 对象
def get_x86_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native x86 backend.
    """
    # 定义不同操作的数据类型配置列表
    conv_dtype_configs = [x86_weighted_op_int8_dtype_config]
    linear_dtype_configs = [
        x86_weighted_op_int8_dtype_config,
        x86_default_dynamic_int8_dtype_config,
        x86_default_dynamic_float16_dtype_config,
    ]
    binary_op_dtype_configs = [x86_weighted_op_int8_dtype_config]
    default_op_dtype_configs = [x86_default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [x86_weighted_op_int8_dtype_config]
    share_qparams_op_dtype_configs = [x86_default_op_quint8_dtype_config]
    tensor_info_op_dtype_configs = [x86_default_op_quint8_dtype_config]
    rnn_op_dtype_configs = [
        x86_default_dynamic_int8_dtype_config,
        x86_default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        x86_weight_only_quint8_dtype_config,
        x86_weight_only_quint4x2_dtype_config,
    ]
    # 创建一个 BackendConfig 对象，指定使用 "x86" 作为后端配置
    return BackendConfig("x86") \
        # 设置卷积操作的配置，根据 conv_dtype_configs 获取
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \
        # 设置线性操作的配置，根据 linear_dtype_configs 获取
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)) \
        # 设置二元操作的配置，根据 binary_op_dtype_configs 获取
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \
        # 设置拼接操作的配置，根据 default_op_dtype_configs 获取
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs)) \
        # 设置默认操作的配置，根据 default_op_dtype_configs 获取
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs)) \
        # 设置固定量化参数操作的配置，根据 fixed_qparams_op_dtype_configs 获取
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)) \
        # 设置共享量化参数操作的配置，根据 share_qparams_op_dtype_configs 获取
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \
        # 设置张量信息操作的配置，根据 tensor_info_op_dtype_configs 获取
        .set_backend_pattern_configs(_get_tensor_info_op_configs(tensor_info_op_dtype_configs)) \
        # 设置批归一化操作的配置，根据 default_op_dtype_configs 获取
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs)) \
        # 设置循环神经网络操作的配置，根据 rnn_op_dtype_configs 获取
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs)) \
        # 设置嵌入操作的配置，根据 embedding_op_dtype_configs 获取
        .set_backend_pattern_configs(_get_embedding_op_configs(embedding_op_dtype_configs))
```