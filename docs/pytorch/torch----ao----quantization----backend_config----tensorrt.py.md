# `.\pytorch\torch\ao\quantization\backend_config\tensorrt.py`

```
# mypy: allow-untyped-defs
# 引入 torch 库，用于深度学习计算
import torch
# 从 backend_config 模块中导入以下配置类
from .backend_config import (
    BackendConfig,                  # 后端配置
    BackendPatternConfig,           # 后端模式配置
    DTypeConfig,                    # 数据类型配置
    ObservationType                 # 观测类型
)
# 从 _common_operator_config_utils 模块中导入以下函数
from ._common_operator_config_utils import (
    _get_binary_op_configs,         # 获取二元操作配置
    _get_linear_configs,            # 获取线性操作配置
    _get_conv_configs,              # 获取卷积操作配置
    _get_share_qparams_op_configs,  # 获取共享量化参数操作配置
    _get_tensor_info_op_configs,    # 获取张量信息操作配置
)

# 定义公开的接口列表
__all__ = [
    "get_tensorrt_backend_config",         # 获取 TensorRT 后端配置
    "get_tensorrt_backend_config_dict",    # 获取 TensorRT 后端配置的字典形式
]

# 返回 TensorRT 后端配置的函数
def get_tensorrt_backend_config() -> BackendConfig:
    """
    返回 TensorRT 后端的 `BackendConfig`。
    注意：当前的 API 将来会有变化，目前仅用于解锁新后端的实验，请勿在目前使用。
    TODO: 在更稳定时添加 README。
    """
    # 定义加权操作 qint8 数据类型配置
    weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,    # 输入数据类型为 qint8
        output_dtype=torch.qint8,   # 输出数据类型为 qint8
        weight_dtype=torch.qint8,   # 权重数据类型为 qint8
        bias_dtype=torch.float,     # 偏置数据类型为 float
    )
    # 定义非加权操作 qint8 数据类型配置
    non_weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,    # 输入数据类型为 qint8
        output_dtype=torch.qint8,   # 输出数据类型为 qint8
    )

    # 定义 addmm 操作的配置
    addmm_config = BackendPatternConfig(torch.addmm) \
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
        .add_dtype_config(weighted_op_qint8_dtype_config) \
        ._set_input_type_to_index({
            "bias": 0,              # 偏置在输入中的索引为 0
            "input": 1,             # 输入在输入中的索引为 1
            "weight": 2,            # 权重在输入中的索引为 2
        })
    
    # 定义 cat 操作的配置
    cat_config = BackendPatternConfig(torch.cat) \
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT) \
        .add_dtype_config(non_weighted_op_qint8_dtype_config)
    
    # 定义卷积操作的 qint8 数据类型配置列表
    conv_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    
    # 定义线性操作的 qint8 数据类型配置列表
    linear_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    
    # 定义二元操作的 qint8 数据类型配置列表
    binary_op_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    
    # 定义共享量化参数操作的 qint8 数据类型配置列表
    share_qparams_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    
    # 定义张量信息操作的 qint8 数据类型配置列表
    tensor_info_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    
    # 返回 TensorRT 后端配置对象
    return BackendConfig("tensorrt") \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \
        .set_backend_pattern_config(addmm_config) \
        .set_backend_pattern_config(cat_config) \
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)) \
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_tensor_info_op_configs(tensor_info_op_dtype_configs))

# 返回 TensorRT 后端配置的字典形式的函数
def get_tensorrt_backend_config_dict():
    """
    返回 TensorRT 后端配置的 `BackendConfig` 的字典形式。
    """
    return get_tensorrt_backend_config().to_dict()
```