# `.\pytorch\torch\ao\quantization\backend_config\fbgemm.py`

```
# 导入PyTorch库
import torch
# 导入配置相关的工具函数
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
# 导入后端配置和数据类型配置类
from .backend_config import BackendConfig, DTypeConfig

# 将"get_fbgemm_backend_config"添加到__all__列表中，表示它是模块的公共接口之一
__all__ = [
    "get_fbgemm_backend_config",
]

# ===================
# |  DTYPE CONFIGS  |
# ===================

# TODO: 目前这些DTypeConfigs与native.py中定义的相同
# 在未来，一旦我们支持指定quant_min/quant_max和scale_min/scale_max，
# 这些配置将会有所不同。特别地，对于FBGEMM，我们将限制激活量化值在[0, 127]之间。

# 定义不同的数据类型配置实例
fbgemm_weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

fbgemm_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

fbgemm_default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float16,
    weight_dtype=torch.float16,
    bias_dtype=torch.float16,
)

fbgemm_default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

fbgemm_default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    is_dynamic=True,
)

fbgemm_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

fbgemm_weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint4x2,
)


# =====================
# |  BACKEND CONFIGS  |
# =====================

# 定义获取FBGEMM后端配置的函数，返回BackendConfig类型的对象
def get_fbgemm_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native FBGEMM backend.
    """
    # 定义不同操作的数据类型配置列表
    conv_dtype_configs = [fbgemm_weighted_op_quint8_dtype_config]
    linear_dtype_configs = [
        fbgemm_weighted_op_quint8_dtype_config,
        fbgemm_default_dynamic_int8_dtype_config,
        fbgemm_default_dynamic_float16_dtype_config,
    ]
    binary_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    default_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    share_qparams_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    tensor_info_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    rnn_op_dtype_configs = [
        fbgemm_default_dynamic_int8_dtype_config,
        fbgemm_default_dynamic_float16_dtype_config,
    ]
    # 定义嵌入操作的数据类型配置列表
    embedding_op_dtype_configs = [
        fbgemm_weight_only_quint8_dtype_config,   # 使用 fbgemm 权重仅支持的 8 位无符号整数数据类型配置
        fbgemm_weight_only_quint4x2_dtype_config,  # 使用 fbgemm 权重仅支持的 4x2 位无符号整数数据类型配置
    ]
    # 返回一个后端配置对象，并依次设置不同模式的配置项
    return BackendConfig("fbgemm") \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \  # 设置卷积操作的后端模式配置
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)) \  # 设置线性操作的后端模式配置
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \  # 设置二元操作的后端模式配置
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs)) \  # 设置合并操作的后端模式配置
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs)) \  # 设置默认操作的后端模式配置
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)) \  # 设置固定量化参数操作的后端模式配置
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \  # 设置共享量化参数操作的后端模式配置
        .set_backend_pattern_configs(_get_tensor_info_op_configs(tensor_info_op_dtype_configs)) \  # 设置张量信息操作的后端模式配置
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs)) \  # 设置批归一化操作的后端模式配置
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs)) \  # 设置循环神经网络操作的后端模式配置
        .set_backend_pattern_configs(_get_embedding_op_configs(embedding_op_dtype_configs))  # 设置嵌入操作的后端模式配置
```