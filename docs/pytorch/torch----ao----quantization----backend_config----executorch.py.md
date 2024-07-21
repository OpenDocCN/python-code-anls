# `.\pytorch\torch\ao\quantization\backend_config\executorch.py`

```
# TODO: 将 executorch 重命名为 qnnpack_executorch，因为 executorch 是一个通用的运行时，而不是特定的后端

import operator  # 导入 operator 模块，用于操作符相关的函数
from typing import List  # 导入 List 类型，用于声明列表类型

import torch  # 导入 PyTorch 库
import torch.ao.nn.qat as nnqat  # 导入 PyTorch AO 模块中的量化训练相关库
import torch.ao.nn.quantized.reference as nnqr  # 导入 PyTorch AO 模块中的参考量化库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 神经网络函数模块

from ..fuser_method_mappings import (  # 从相对路径中导入相关模块和函数
    _sequential_wrapper2,
    fuse_conv_bn,
    fuse_conv_bn_relu,
)
from ._common_operator_config_utils import _Conv2dMetadata  # 从当前目录下的 _common_operator_config_utils 模块导入 _Conv2dMetadata 类
from .backend_config import (  # 从当前目录下的 backend_config 模块导入多个类和常量
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
)
from .qnnpack import (  # 从当前目录下的 qnnpack 模块导入多个量化配置
    qnnpack_default_op_qint8_symmetric_dtype_config,
    qnnpack_weighted_op_qint8_symmetric_dtype_config,
)

__all__ = [  # 导出该模块的公共接口名称列表
    "get_executorch_backend_config",
]

# ===================
# |  DTYPE CONFIGS  |
# ===================

# 定义 executorch_weighted_op_int8_dtype_config 变量，用于配置 int8 权重量化的数据类型
executorch_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,  # 输入数据类型为 quint8（8 位无符号整数）
    output_dtype=torch.quint8,  # 输出数据类型为 quint8
    weight_dtype=torch.qint8,   # 权重数据类型为 qint8（8 位有符号整数）
    bias_dtype=torch.float,     # 偏置数据类型为 float（单精度浮点数）
)

# 定义 executorch_default_op_quint8_dtype_config 变量，用于配置默认的 quint8 数据类型
executorch_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,  # 输入数据类型为 quint8
    output_dtype=torch.quint8,  # 输出数据类型为 quint8
)

# 定义 executorch_default_dynamic_quint8_dtype_config 变量，用于配置动态 quint8 数据类型
executorch_default_dynamic_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,  # 输入数据类型为 quint8
    output_dtype=torch.float,  # 输出数据类型为 float（单精度浮点数）
    weight_dtype=torch.qint8,   # 权重数据类型为 qint8
    bias_dtype=torch.float,     # 偏置数据类型为 float
    is_dynamic=True,            # 表示数据类型是动态的
)

# 定义 executorch_act_qint8_scale_min_2_neg_12 变量，配置 qint8 数据类型及其约束条件
executorch_act_qint8_scale_min_2_neg_12 = DTypeWithConstraints(
    dtype=torch.qint8,          # 数据类型为 qint8
    scale_min_lower_bound=2**-12,  # 最小缩放比例下限为 2 的 -12 次方
)

# 定义 executorch_weight_qint8_neg_127_to_127_scale_min_2_neg_12 变量，配置 qint8 权重数据类型及其约束条件
executorch_weight_qint8_neg_127_to_127_scale_min_2_neg_12 = DTypeWithConstraints(
    dtype=torch.qint8,          # 数据类型为 qint8
    quant_min_lower_bound=-127,  # 量化最小值下限为 -127
    quant_max_upper_bound=127,   # 量化最大值上限为 127
    scale_min_lower_bound=2**-12,  # 最小缩放比例下限为 2 的 -12 次方
)

# 定义 executorch_default_dynamic_qint8_dtype_config 变量，配置动态 qint8 数据类型
executorch_default_dynamic_qint8_dtype_config = DTypeConfig(
    input_dtype=executorch_act_qint8_scale_min_2_neg_12,  # 输入数据类型为 qint8
    output_dtype=torch.float,  # 输出数据类型为 float
    weight_dtype=executorch_weight_qint8_neg_127_to_127_scale_min_2_neg_12,  # 权重数据类型为 qint8
    bias_dtype=torch.float,     # 偏置数据类型为 float
    is_dynamic=True,            # 表示数据类型是动态的
)

# 定义 executorch_default_dynamic_float16_dtype_config 变量，配置动态 float16 数据类型
executorch_default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,  # 输入数据类型为 float16
    output_dtype=torch.float,   # 输出数据类型为 float
    weight_dtype=torch.float16, # 权重数据类型为 float16
    bias_dtype=torch.float,     # 偏置数据类型为 float
    is_dynamic=True,            # 表示数据类型是动态的
)

# 定义 executorch_weight_only_quint8_dtype_config 变量，配置仅权重为 quint8 的数据类型
executorch_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,    # 输入数据类型为 float
    output_dtype=torch.float,   # 输出数据类型为 float
    weight_dtype=torch.quint8,  # 权重数据类型为 quint8
)

# =============================
# |  BACKEND PATTERN CONFIGS  |
# =============================

def _get_linear_configs() -> List[BackendPatternConfig]:
    """
    Return all configs related to linear modules and ops.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT  # 观察类型为使用不同的观察器作为输入
    dtype_configs = [
        qnnpack_weighted_op_qint8_symmetric_dtype_config,  # 加入 qnnpack 权重量化 qint8 对称配置
        executorch_weighted_op_int8_dtype_config,          # 加入 executorch 权重量化 int8 配置
        executorch_default_dynamic_quint8_dtype_config,    # 加入 executorch 动态 quint8 配置
        executorch_default_dynamic_qint8_dtype_config,     # 加入 executorch 动态 qint8 配置
        executorch_default_dynamic_float16_dtype_config,   # 加入 executorch 动态 float16 配置
    ]
    linear_configs: List[BackendPatternConfig] = []  # 初始化线性模块配置列表为空
    # linear module
    linear_configs.append(
        # 创建一个针对 torch.nn.Linear 的后端模式配置对象，并设置观察类型
        BackendPatternConfig(torch.nn.Linear)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)  # 设置数据类型配置
        .set_root_module(torch.nn.Linear)  # 设置根模块为 torch.nn.Linear
        .set_reference_quantized_module(nnqr.Linear)  # 设置参考量化模块为 nnqr.Linear
        .set_qat_module(nnqat.Linear)  # 设置量化训练模块为 nnqat.Linear
    )
    
    # linear qat module
    linear_configs.append(
        # 创建一个针对 nnqat.Linear 的后端模式配置对象，并设置观察类型
        BackendPatternConfig(nnqat.Linear)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)  # 设置数据类型配置
        .set_root_module(torch.nn.Linear)  # 设置根模块为 torch.nn.Linear
        .set_reference_quantized_module(nnqr.Linear)  # 设置参考量化模块为 nnqr.Linear
    )
    
    # functional linear
    linear_configs.append(
        # 创建一个针对 torch.nn.functional.linear 的后端模式配置对象，并设置观察类型
        BackendPatternConfig(torch.nn.functional.linear)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)  # 设置数据类型配置
        ._set_input_type_to_index({"weight": 1, "bias": 2})  # 设置输入类型到索引的映射
    )
    
    # 返回配置好的 linear_configs 列表
    return linear_configs
# 返回所有与卷积模块和操作相关的配置列表
def _get_conv_configs() -> List[BackendPatternConfig]:
    # 观察类型为使用不同的观察器作为输入的输出观察
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    # 数据类型配置列表
    dtype_configs = [
        qnnpack_weighted_op_qint8_symmetric_dtype_config,  # QNNPACK权重操作的QINT8对称数据类型配置
        executorch_weighted_op_int8_dtype_config,  # ExecutorCh加权操作的INT8数据类型配置
    ]
    # 空的卷积配置列表
    conv_configs = []
    return conv_configs


# 返回所有与二元操作相关的配置列表
def _get_binary_ops_configs() -> List[BackendPatternConfig]:
    # 数据类型配置列表
    dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,  # QNNPACK默认操作的QINT8对称数据类型配置
        executorch_weighted_op_int8_dtype_config,  # ExecutorCh加权操作的INT8数据类型配置
    ]
    # 张量参数数量到观察类型的映射字典
    num_tensor_args_to_observation_type_mapping = {
        # TODO: 目前未使用，因为在准备阶段有额外的检查，后续在实现张量数据类型推断后需要改为NO_OBSERVER
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    # 二元操作配置列表初始化为空
    binary_op_configs: List[BackendPatternConfig] = []
    # 遍历每个操作符
    for op in [operator.add, torch.add, operator.sub, torch.sub, operator.mul, torch.mul]:
        # 操作模式列表
        bop_patterns = [
            (op, torch.nn.ReLU),  # 添加操作与ReLU激活函数的模式
            (op, torch.nn.functional.relu),  # 添加操作与torch.nn.functional.relu的模式
            (op, torch.relu),  # 添加操作与torch.relu的模式
            op  # 添加操作本身的模式
        ]
        # 遍历每个二元操作模式
        for bop_pattern in bop_patterns:
            # 创建BackendPatternConfig对象，并设置数据类型配置和张量参数数量到观察类型的映射
            binary_op_configs.append(
                BackendPatternConfig(bop_pattern)
                .set_dtype_configs(dtype_configs)  # 设置数据类型配置
                ._set_num_tensor_args_to_observation_type(
                    num_tensor_args_to_observation_type_mapping
                )
            )
    return binary_op_configs


# 返回在输入为量化的情况下，与浮点数和量化输入均可工作的运算符配置列表，
# 输出张量与输入共享相同的量化参数。
def _get_share_qparams_ops_configs() -> List[BackendPatternConfig]:
    # 观察类型为输出与输入共享观察
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    # 数据类型配置列表
    dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,  # QNNPACK默认操作的QINT8对称数据类型配置
        executorch_default_op_quint8_dtype_config,  # ExecutorCh默认操作的QUINT8数据类型配置
    ]
    # 定义要共享量化参数的操作列表，包括 Torch 内置函数和字符串标识符
    share_qparams_ops = [
        torch.nn.Flatten,                   # 将输入张量展平的操作
        F.adaptive_avg_pool2d,              # 自适应平均池化操作
        F.elu,                              # ELU 激活函数
        F.hardtanh,                         # 硬切线激活函数
        F.max_pool2d,                       # 最大池化操作
        F.pad,                              # 填充操作
        F.relu,                             # ReLU 激活函数
        F.relu6,                            # ReLU6 激活函数
        F.leaky_relu,                       # Leaky ReLU 激活函数
        F.leaky_relu_,                      # Leaky ReLU 原位版本
        torch.nn.AdaptiveAvgPool2d,         # 自适应平均池化层
        torch.nn.ConstantPad2d,             # 常数填充层
        torch.nn.ELU,                       # ELU 激活函数层
        torch.nn.MaxPool2d,                 # 最大池化层
        torch.nn.ReLU6,                     # ReLU6 激活函数层
        torch.nn.Hardtanh,                  # 硬切线激活函数层
        torch.nn.LeakyReLU,                 # Leaky ReLU 激活函数层
        torch.clamp,                        # 张量值截取操作
        torch.flatten,                      # 张量展平操作
        torch.mean,                         # 张量均值操作
        torch.permute,                      # 张量维度重排操作
        torch.permute_copy,                 # 复制张量并重排维度操作
        torch.squeeze,                      # 去除张量中大小为 1 的维度操作
        "clamp",                            # 字符串标识符，代表张量值截取操作
        "mean",                             # 字符串标识符，代表张量均值操作
        "permute",                          # 字符串标识符，代表张量维度重排操作
        "reshape",                          # 字符串标识符，代表张量重塑操作
        "relu",                             # 字符串标识符，代表 ReLU 激活函数
        "relu_",                            # 字符串标识符，代表原位 ReLU 激活函数
        "squeeze",                          # 字符串标识符，代表去除维度为 1 的操作
        "squeeze_",                         # 字符串标识符，代表原位去除维度为 1 的操作
        "leaky_relu",                       # 字符串标识符，代表 Leaky ReLU 激活函数
    ]
    
    # 定义一个空列表，用于存储每个操作的后端模式配置
    share_qparams_op_configs: List[BackendPatternConfig] = []
    
    # 遍历共享量化参数操作列表，并为每个操作创建后端模式配置对象，设置观察类型和数据类型配置
    for op in share_qparams_ops:
        share_qparams_op_configs.append(
            BackendPatternConfig(op)
            .set_observation_type(observation_type)  # 设置观察类型，忽略 E131 错误
            .set_dtype_configs(dtype_configs)        # 设置数据类型配置
        )
    
    # 返回包含所有操作配置的列表
    return share_qparams_op_configs
def _get_bn_configs() -> List[BackendPatternConfig]:
    """
    Return all configs related to batchnorm.
    """
    # 定义观察类型为：输出使用不同的观察器作为输入
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    # 定义数据类型配置列表
    dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,  # 默认的 qint8 对称数据类型配置
        executorch_default_op_quint8_dtype_config,  # 默认的 quint8 数据类型配置
    ]
    # 初始化 batchnorm 配置列表
    bn_configs = []
    # 添加 BatchNorm2d 的配置
    bn_configs.append(
        BackendPatternConfig(nn.BatchNorm2d)
        .set_observation_type(observation_type)  # 设置观察类型为定义的观察类型
        .set_dtype_configs(dtype_configs)  # 设置数据类型配置
    )
    # 返回 batchnorm 配置列表
    return bn_configs


def _get_cat_configs() -> List[BackendPatternConfig]:
    # 定义数据类型配置列表
    dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,  # 默认的 qint8 对称数据类型配置
        executorch_default_op_quint8_dtype_config,  # 默认的 quint8 数据类型配置
    ]
    # 初始化 concatenate 配置列表
    cat_configs = []
    # 添加 torch.cat 的配置
    cat_configs.append(
        BackendPatternConfig(torch.cat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)  # 设置观察类型为输出共享观察器与输入
        .set_dtype_configs(dtype_configs)  # 设置数据类型配置
    )
    # 添加 torch.concat 的配置
    cat_configs.append(
        BackendPatternConfig(torch.concat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)  # 设置观察类型为输出共享观察器与输入
        .set_dtype_configs(dtype_configs)  # 设置数据类型配置
    )
    # 添加 torch.concatenate 的配置
    cat_configs.append(
        BackendPatternConfig(torch.concatenate)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)  # 设置观察类型为输出共享观察器与输入
        .set_dtype_configs(dtype_configs)  # 设置数据类型配置
    )
    # 返回 concatenate 配置列表
    return cat_configs


def _get_embedding_op_configs() -> List[BackendPatternConfig]:
    # 定义数据类型配置列表
    dtype_configs = [
        executorch_weight_only_quint8_dtype_config,  # 仅权重的 quint8 数据类型配置
    ]
    # 初始化 embedding 操作配置列表
    embedding_op_configs = []
    # 遍历 embedding 操作及其对应的量化训练和参考量化模块
    for embedding_op, qat_embedding_op, ref_embedding_op in [
        (nn.Embedding, nnqat.Embedding, nnqr.Embedding),  # Embedding 相关操作
        (nn.EmbeddingBag, nnqat.EmbeddingBag, nnqr.EmbeddingBag),  # EmbeddingBag 相关操作
    ]:
        # 添加 embedding 操作的配置
        embedding_op_configs.append(
            BackendPatternConfig(embedding_op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT  # 设置观察类型为：输出使用不同的观察器作为输入
            )
            .set_dtype_configs(dtype_configs)  # 设置数据类型配置
            .set_qat_module(qat_embedding_op)  # 设置量化训练模块
            .set_root_module(embedding_op)  # 设置根模块
            .set_reference_quantized_module(ref_embedding_op)  # 设置参考量化模块
        )
        # 添加量化训练操作的配置
        embedding_op_configs.append(
            BackendPatternConfig(qat_embedding_op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT  # 设置观察类型为：输出使用不同的观察器作为输入
            )
            .set_dtype_configs(dtype_configs)  # 设置数据类型配置
            .set_root_module(embedding_op)  # 设置根模块
            .set_reference_quantized_module(ref_embedding_op)  # 设置参考量化模块
        )

        # 添加功能性 embedding 操作的配置
        embedding_op_configs.append(
            BackendPatternConfig(torch.nn.functional.embedding)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT  # 设置观察类型为：输出使用不同的观察器作为输入
            )
            .set_dtype_configs(dtype_configs)  # 设置数据类型配置
            ._set_input_type_to_index({"weight": 1})  # 设置输入类型到索引的映射
        )
    # 返回 embedding 操作配置列表
    return embedding_op_configs
# =====================
# |  BACKEND CONFIGS  |
# =====================

# 返回用于 Executorch 栈通过的后端 PyTorch 的 BackendConfig
def get_executorch_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for backends PyTorch lowers to through the Executorch stack.
    """
    # 创建一个名为 "executorch" 的 BackendConfig 对象
    return (
        BackendConfig("executorch")
        # 设置线性模式的配置
        .set_backend_pattern_configs(_get_linear_configs())
        # 设置卷积模式的配置
        .set_backend_pattern_configs(_get_conv_configs())
        # 设置二元操作模式的配置
        .set_backend_pattern_configs(_get_binary_ops_configs())
        # 设置共享量化参数操作模式的配置
        .set_backend_pattern_configs(_get_share_qparams_ops_configs())
        # 设置批归一化模式的配置
        .set_backend_pattern_configs(_get_bn_configs())
        # 设置拼接模式的配置
        .set_backend_pattern_configs(_get_cat_configs())
        # 设置嵌入操作模式的配置
        .set_backend_pattern_configs(_get_embedding_op_configs())
    )
```