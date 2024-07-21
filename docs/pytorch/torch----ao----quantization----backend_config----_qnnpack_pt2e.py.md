# `.\pytorch\torch\ao\quantization\backend_config\_qnnpack_pt2e.py`

```py
# mypy: allow-untyped-defs
# 引入操作符模块
import operator
# 引入 PyTorch 库
import torch
# 从 PyTorch 的量化后端配置中导入相关类和枚举
from torch.ao.quantization.backend_config import (
    BackendConfig,
    DTypeConfig,
    ObservationType,
    BackendPatternConfig,
)
# 创建一个特定的 dtype 配置对象，用于权重操作
weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,    # 输入数据类型为 quint8
    output_dtype=torch.quint8,   # 输出数据类型为 quint8
    weight_dtype=torch.qint8,    # 权重数据类型为 qint8
    bias_dtype=torch.float,      # 偏置数据类型为 float
)
# 导入 List 类型的模块
from typing import List

# 获取线性层的配置
def get_linear_configs():
    linear_configs = []   # 初始化一个空列表，用于存储线性层的配置信息
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT  # 观察类型设定为输出使用不同的观察器作为输入

    dtype_configs = [weighted_op_quint8_dtype_config]   # 使用之前创建的 dtype 配置对象列表

    # TODO: 需要修复插入观察器的方式以适应这种模式
    # 这个问题应该在新的融合 API 中得到解决
    # 当前解决方法的问题在于：这个模式比较复杂，我们无法指定我们想要观察的模式的哪个输入
    # 模式如下：
    # bias input weight
    # \     |    /
    #  \    |   t
    #   \   |  /
    #    addmm
    # 我们想要将 "weight" 观察为权重，但当前的模式语言无法传达这个信息

    # linear_configs.append(
    #     BackendPatternConfig((torch.ops.aten.addmm.default, MatchAllNode, MatchAllNode, torch.ops.aten.t.default))
    #     .set_observation_type(observation_type)  # noqa: E131
    #     .set_dtype_configs(dtype_configs)
    #     ._set_root_node_getter(root_node_getter))

    # 将 addmm 操作作为线性配置的一部分，并设置观察类型和 dtype 配置
    linear_configs.append(
        BackendPatternConfig(torch.ops.aten.addmm.default)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 0})  # 设置输入类型到索引的映射
    )
    # 如果偏置项不存在，则将 linear 层分解为 `t - mm`
    linear_configs.append(
        BackendPatternConfig(torch.ops.aten.mm.default)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1})  # 设置输入类型到索引的映射
    )
    return linear_configs

# 获取卷积层的配置
def get_conv_configs():
    conv_configs = []   # 初始化一个空列表，用于存储卷积层的配置信息
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT  # 观察类型设定为输出使用不同的观察器作为输入

    dtype_configs = [weighted_op_quint8_dtype_config]   # 使用之前创建的 dtype 配置对象列表

    # 将卷积操作作为卷积配置的一部分，并设置观察类型和 dtype 配置
    conv_configs.append(
        BackendPatternConfig(torch.ops.aten.convolution.default)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})  # 设置输入类型到索引的映射
    )
    conv_configs.append(
        BackendPatternConfig((torch.ops.aten.convolution.default, torch.ops.aten.relu.default))
        .set_observation_type(observation_type)  # 设置观察类型，用于记录模式配置信息
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    # TODO: remove when functionalization is supported in PT2 mode
    conv_configs.append(
        BackendPatternConfig((torch.ops.aten.convolution.default, torch.ops.aten.relu_.default))
        .set_observation_type(observation_type)  # 设置观察类型，用于记录模式配置信息
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    # 返回已配置的卷积模式列表
    return conv_configs
def get_pooling_configs():
    backend_pattern_configs = []  # 创建一个空列表，用于存储后端模式配置
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT  # 设置观察类型为输出共享观察者与输入
    dtype_configs = [weighted_op_quint8_dtype_config]  # 设置数据类型配置列表，包含权重操作的quint8数据类型配置

    def root_node_getter(node_pattern):
        getitem, maxpool, index = node_pattern  # 解构节点模式元组
        return maxpool  # 返回 maxpool 元素作为根节点

    backend_pattern_configs.append(  # 向后端模式配置列表添加新配置
        BackendPatternConfig()  # 创建一个后端模式配置对象
        ._set_pattern_complex_format((operator.getitem, torch.ops.aten.max_pool2d_with_indices.default, 0))  # 设置复杂模式格式
        .set_observation_type(observation_type)  # 设置观察类型
        .set_dtype_configs(dtype_configs)  # 设置数据类型配置列表
        ._set_root_node_getter(root_node_getter)  # 设置根节点获取器
    )

    return backend_pattern_configs  # 返回配置好的后端模式配置列表


def get_relu_configs():
    backend_pattern_configs = []  # 创建一个空列表，用于存储后端模式配置
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT  # 设置观察类型为输出共享观察者与输入
    dtype_configs = [weighted_op_quint8_dtype_config]  # 设置数据类型配置列表，包含权重操作的quint8数据类型配置
    backend_pattern_configs.append(  # 向后端模式配置列表添加新配置
        BackendPatternConfig(torch.ops.aten.relu.default)  # 使用默认的 relu 操作配置一个后端模式配置对象
        .set_observation_type(observation_type)  # 设置观察类型
        .set_dtype_configs(dtype_configs))  # 设置数据类型配置列表

    return backend_pattern_configs  # 返回配置好的后端模式配置列表


def get_binary_op_configs():
    binary_op_configs: List[BackendPatternConfig] = []  # 创建一个空列表，用于存储二元操作的后端模式配置对象
    dtype_configs = [weighted_op_quint8_dtype_config]  # 设置数据类型配置列表，包含权重操作的quint8数据类型配置
    num_tensor_args_to_observation_type_mapping = {
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,  # 映射，对于0个张量参数，使用不同的观察者作为输入
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,  # 映射，对于1个张量参数，输出与输入共享观察者
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,  # 映射，对于2个张量参数，使用不同的观察者作为输入
    }
    for op_with_quantized_bop_scalar_variant in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
        bop_patterns = [
            (op_with_quantized_bop_scalar_variant, torch.ops.aten.relu.default),  # 二元操作模式列表中的模式元组
            op_with_quantized_bop_scalar_variant,  # 二元操作模式列表中的操作
            # TODO: 在 pt2_mode 支持功能化后移除此处
            (op_with_quantized_bop_scalar_variant, torch.ops.aten.relu_.default),  # 二元操作模式列表中的模式元组
        ]
        for bop_pattern in bop_patterns:
            binary_op_configs.append(  # 向二元操作配置列表添加新的后端模式配置对象
                BackendPatternConfig(bop_pattern)  # 使用二元操作模式创建后端模式配置对象
                    .set_dtype_configs(dtype_configs)  # 设置数据类型配置列表
                    ._set_num_tensor_args_to_observation_type(num_tensor_args_to_observation_type_mapping))  # 设置张量参数到观察类型的映射

    return binary_op_configs  # 返回配置好的二元操作后端模式配置列表


def get_qnnpack_pt2e_backend_config():
    return (
        BackendConfig("qnnpack_pytorch_2.0_export")  # 创建后端配置对象，命名为 "qnnpack_pytorch_2.0_export"
        .set_backend_pattern_configs(get_linear_configs())  # 设置后端模式配置列表为线性层的配置
        .set_backend_pattern_configs(get_binary_op_configs())  # 添加二元操作的后端模式配置列表
        .set_backend_pattern_configs(get_conv_configs())  # 添加卷积层的后端模式配置列表
        .set_backend_pattern_configs(get_pooling_configs())  # 添加池化层的后端模式配置列表
        .set_backend_pattern_configs(get_relu_configs())  # 添加 relu 操作的后端模式配置列表
    )
```