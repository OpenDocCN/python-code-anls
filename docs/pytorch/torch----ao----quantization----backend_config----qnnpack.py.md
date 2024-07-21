# `.\pytorch\torch\ao\quantization\backend_config\qnnpack.py`

```py
# 引入 torch 库，用于深度学习任务
import torch
# 从内部模块导入多个函数，用于获取不同操作的配置信息
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
)
# 从 backend_config 模块中导入特定类和对象
from .backend_config import BackendConfig, DTypeConfig, DTypeWithConstraints

# __all__ 列表定义了该模块中可以导出的公共接口
__all__ = [
    "get_qnnpack_backend_config",
]

# ===================
# |  DTYPE CONFIGS  |
# ===================

# 定义 QNNPACK 权重操作的 quint8 类型配置
qnnpack_weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,   # 输入数据类型为 quint8
    output_dtype=torch.quint8,  # 输出数据类型为 quint8
    weight_dtype=torch.qint8,   # 权重数据类型为 qint8
    bias_dtype=torch.float,     # 偏置数据类型为 float
)

# 定义 QNNPACK 默认操作的 quint8 类型配置
qnnpack_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

# 定义 QNNPACK 默认操作的 fp16 类型配置
qnnpack_default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float16,
    weight_dtype=torch.float16,
    bias_dtype=torch.float16,
)

# 定义 QNNPACK 默认动态 int8 类型配置
qnnpack_default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,  # 数据类型是动态的
)

# 定义 QNNPACK 默认动态 float16 类型配置
qnnpack_default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    is_dynamic=True,
)

# 定义仅包含权重的 quint8 类型配置
qnnpack_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

# 定义仅包含权重的 quint4x2 类型配置
qnnpack_weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint4x2,
)

# xnnpack 兼容的数据类型配置

# 我们将 scale 值限制在 2 的 -12 次方，以确保重新量化的 scale 不会低于 xnnpack 的下限
# 此外，对于 qint8 权重，我们将量化值限制在 [-127, +127]，排除 -128。
# 更多详情，请参考 `default_symmetric_qnnpack_qconfig` 的描述。

# TODO: 添加对 qscheme 的额外限制，确保它是 per_tensor_symmetric 或 per_channel_symmetric

# 定义 qint8 数据类型的附加约束，包括最小 scale 为 2 的 -12 次方
qnnpack_act_qint8_scale_min_2_neg_12 = DTypeWithConstraints(
    dtype=torch.qint8,
    scale_min_lower_bound=2 ** -12,
)

# 定义 qint8 权重数据类型的附加约束，包括量化范围为 [-127, +127]，最小 scale 为 2 的 -12 次方
qnnpack_weight_qint8_neg_127_to_127_scale_min_2_neg_12 = DTypeWithConstraints(
    dtype=torch.qint8,
    quant_min_lower_bound=-127,
    quant_max_upper_bound=127,
    scale_min_lower_bound=2 ** -12,
)

# 定义 QNNPACK 权重操作的 qint8 对称数据类型配置
qnnpack_weighted_op_qint8_symmetric_dtype_config = DTypeConfig(
    input_dtype=qnnpack_act_qint8_scale_min_2_neg_12,
    output_dtype=qnnpack_act_qint8_scale_min_2_neg_12,
    weight_dtype=qnnpack_weight_qint8_neg_127_to_127_scale_min_2_neg_12,
    bias_dtype=torch.float,
)

# 定义 QNNPACK 默认操作的 qint8 对称数据类型配置
qnnpack_default_op_qint8_symmetric_dtype_config = DTypeConfig(
    input_dtype=qnnpack_act_qint8_scale_min_2_neg_12,
    output_dtype=qnnpack_act_qint8_scale_min_2_neg_12,
)

# =====================
# |  BACKEND CONFIGS  |
# =====================

def get_qnnpack_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native QNNPACK backend.
    """
    # 定义卷积层的数据类型配置列表
    conv_dtype_configs = [
        qnnpack_weighted_op_qint8_symmetric_dtype_config,
        qnnpack_weighted_op_quint8_dtype_config,
    ]
    # 定义全连接层的数据类型配置列表
    linear_dtype_configs = [
        qnnpack_weighted_op_qint8_symmetric_dtype_config,
        qnnpack_weighted_op_quint8_dtype_config,
        qnnpack_default_dynamic_int8_dtype_config,
        qnnpack_default_dynamic_float16_dtype_config,
    ]
    # 定义二元操作的数据类型配置列表
    binary_op_dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    # 定义默认操作的数据类型配置列表
    default_op_dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    # 定义固定量化参数操作的数据类型配置列表
    fixed_qparams_op_dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    # 定义共享量化参数操作的数据类型配置列表
    share_qparams_op_dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    # 定义循环神经网络操作的数据类型配置列表
    rnn_op_dtype_configs = [
        qnnpack_default_dynamic_int8_dtype_config,
        qnnpack_default_dynamic_float16_dtype_config,
    ]
    # 定义嵌入操作的数据类型配置列表
    embedding_op_dtype_configs = [
        qnnpack_weight_only_quint8_dtype_config,
        qnnpack_weight_only_quint4x2_dtype_config,
    ]
    
    # 返回配置了各种模式的 BackendConfig 对象，使用链式调用设置各个模式的配置
    return BackendConfig("qnnpack") \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)) \
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_embedding_op_configs(embedding_op_dtype_configs))
```