# `.\pytorch\torch\ao\quantization\backend_config\_common_operator_config_utils.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和库
import copy  # 导入 copy 模块，用于对象的复制操作
import operator  # 导入 operator 模块，用于函数形式的操作符
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的函数式接口模块
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
import torch.ao.nn.intrinsic as nni  # 导入 PyTorch 中的 AO（高级优化）模块
import torch.ao.nn.intrinsic.qat as nniqat  # 导入 PyTorch 中的 QAT（量化感知训练）AO 模块
import torch.ao.nn.qat as nnqat  # 导入 PyTorch 中的 QAT 模块
import torch.ao.nn.quantized.reference as nnqr  # 导入 PyTorch 中的量化参考模块
from collections import namedtuple  # 导入 collections 模块中的 namedtuple 类型
from typing import Callable, Dict, List, Union  # 导入类型提示相关的模块和类型
from .backend_config import (  # 从当前包中导入后端配置相关的模块
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
)
from ..fuser_method_mappings import (  # 从父级包中导入融合方法映射相关的模块
    _sequential_wrapper2,
    fuse_conv_bn,
    fuse_conv_bn_relu,
    fuse_linear_bn,
    fuse_convtranspose_bn,
)

__all__: List[str] = []  # 定义一个空列表，用于存放需要公开的模块成员

# TODO: rename to be more explicit, e.g. qat_conv_relu
# 定义名为 _ConvMetadata 的命名元组，用于存储卷积相关的元数据信息
_ConvMetadata = namedtuple(
    "_ConvMetadata",
    ["root", "transpose", "bn", "reference", "transpose_reference",
     "fused_conv_relu", "fused_conv_bn", "fused_conv_bn_relu",
     "qat", "relu_qat", "bn_qat", "bn_relu_qat",
     "func", "func_transpose"])
# 定义 _Conv1dMetadata 命名元组，存储 1D 卷积相关的元数据信息
_Conv1dMetadata = _ConvMetadata(
    nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d, nnqr.Conv1d, nnqr.ConvTranspose1d,
    nni.ConvReLU1d, nni.ConvBn1d, nni.ConvBnReLU1d,
    nnqat.Conv1d, nniqat.ConvReLU1d, nniqat.ConvBn1d, nniqat.ConvBnReLU1d,
    F.conv1d, F.conv_transpose1d)
# 定义 _Conv2dMetadata 命名元组，存储 2D 卷积相关的元数据信息
_Conv2dMetadata = _ConvMetadata(
    nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nnqr.Conv2d, nnqr.ConvTranspose2d,
    nni.ConvReLU2d, nni.ConvBn2d, nni.ConvBnReLU2d,
    nnqat.Conv2d, nniqat.ConvReLU2d, nniqat.ConvBn2d, nniqat.ConvBnReLU2d,
    F.conv2d, F.conv_transpose2d)
# 定义 _Conv3dMetadata 命名元组，存储 3D 卷积相关的元数据信息
_Conv3dMetadata = _ConvMetadata(
    nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d, nnqr.Conv3d, nnqr.ConvTranspose3d,
    nni.ConvReLU3d, nni.ConvBn3d, nni.ConvBnReLU3d,
    nnqat.Conv3d, nniqat.ConvReLU3d, nniqat.ConvBn3d, nniqat.ConvBnReLU3d,
    F.conv3d, F.conv_transpose3d)

# 为固定量化参数操作（如 sigmoid 和 tanh）添加约束，确保值落在适当的范围内
# 例如，sigmoid 的范围为 [0, 1]，tanh 的范围为 [-1, 1]
_FIXED_QPARAM_OP_0TO1_CONSTRAINTS = DTypeWithConstraints(
    dtype=torch.quint8,
    quant_min_lower_bound=0,
    quant_max_upper_bound=255,
    scale_exact_match=1.0 / 256.0,
    zero_point_exact_match=0,
)
_FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS = DTypeWithConstraints(
    dtype=torch.quint8,
    quant_min_lower_bound=0,
    quant_max_upper_bound=255,
    scale_exact_match=2.0 / 256.0,
    zero_point_exact_match=128,
)
# 定义一个字典，将固定量化参数操作映射到相应的约束
_FIXED_QPARAMS_OP_TO_CONSTRAINTS: Dict[Union[Callable, str], DTypeWithConstraints] = {
    torch.nn.Hardsigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    torch.nn.functional.hardsigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "hardsigmoid": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "hardsigmoid_": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    torch.nn.Sigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    torch.sigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "sigmoid": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "sigmoid_": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    torch.nn.Softmax: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    # 使用 `_FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS` 约束了 `torch.nn.Tanh` 激活函数
    torch.nn.Tanh: _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS,
    # 使用 `_FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS` 约束了 `torch.tanh` 函数
    torch.tanh: _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS,
    # 使用 `_FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS` 约束了字符串 `"tanh"`
    "tanh": _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS,
    # 使用 `_FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS` 约束了字符串 `"tanh_"`
    "tanh_": _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS,
}

# 定义一个私有函数，用于获取二元操作的配置信息
def _get_binary_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    binary_op_configs: List[BackendPatternConfig] = []
    # 定义一个映射表，将输入张量个数映射到观察类型，当前这段代码暂时未使用，
    # 因为在 prepare 函数中有额外的检查，之后实现了张量数据类型推断后将改为 NO_OBSERVER
    num_tensor_args_to_observation_type_mapping = {
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    # 遍历四种带有量化二元操作标量变体的操作：加法、乘法及它们在 torch 中的版本
    for op_with_quantized_bop_scalar_variant in [operator.add, torch.add, operator.mul, torch.mul]:
        # 定义可能的操作模式列表，包括操作符与 ReLU 激活函数的组合
        bop_patterns = [
            (op_with_quantized_bop_scalar_variant, nn.ReLU),
            (op_with_quantized_bop_scalar_variant, F.relu),
            (op_with_quantized_bop_scalar_variant, torch.relu),
            op_with_quantized_bop_scalar_variant
        ]
        # 遍历操作模式列表，创建 BackendPatternConfig 对象并添加到二元操作配置列表中
        for bop_pattern in bop_patterns:
            binary_op_configs.append(
                BackendPatternConfig(bop_pattern)
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    ._set_num_tensor_args_to_observation_type(num_tensor_args_to_observation_type_mapping))
    # 添加 matmul 操作的配置信息
    binary_op_configs.append(
        BackendPatternConfig(torch.matmul)
        .set_dtype_configs(dtype_configs)  # noqa: E131
    )
    return binary_op_configs

# 定义一个函数，用于获取线性模块和操作的配置信息
def _get_linear_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    """
    返回所有与线性模块和操作相关的配置。
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    linear_configs: List[BackendPatternConfig] = []

    # (1) 单个线性模块/函数
    # -------------------------------------
    # 添加线性模块的配置信息
    linear_configs.append(
        BackendPatternConfig(torch.nn.Linear)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(torch.nn.Linear)
            .set_reference_quantized_module(nnqr.Linear)
            .set_qat_module(nnqat.Linear))
    # 添加量化训练模式下的线性模块配置信息
    linear_configs.append(
        BackendPatternConfig(nnqat.Linear)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(torch.nn.Linear)
            .set_reference_quantized_module(nnqr.Linear))
    # 添加函数式线性操作的配置信息
    linear_configs.append(
        BackendPatternConfig(torch.nn.functional.linear)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            ._set_input_type_to_index({"weight": 1, "bias": 2}))

    # (2) 线性模块 + ReLU
    # -------------------
    # 2.1 添加线性模块与ReLU激活函数融合的配置信息
    # linear relu, 线性模块 + ReLU 模块
    linear_configs.append(
        BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))
            .set_dtype_configs(dtype_configs)  # 设置数据类型配置
            .set_fuser_method(_sequential_wrapper2(nni.LinearReLU))  # 设置融合方法为线性-ReLU模式
            .set_fused_module(nni.LinearReLU))  # 设置融合后的模块为LinearReLU

    # linear relu, linear module + functional relu
    linear_configs.append(
        BackendPatternConfig((torch.nn.Linear, torch.nn.functional.relu))
            .set_dtype_configs(dtype_configs)  # 设置数据类型配置
            .set_fuser_method(_sequential_wrapper2(nni.LinearReLU))  # 设置融合方法为线性-ReLU模式
            .set_fused_module(nni.LinearReLU))  # 设置融合后的模块为LinearReLU

    # 2.2 linear module + relu, fused module configs
    # linear relu, fused module
    linear_configs.append(
        BackendPatternConfig(nni.LinearReLU)
            .set_observation_type(observation_type)  # 设置观测类型
            .set_dtype_configs(dtype_configs)  # 设置数据类型配置
            .set_root_module(torch.nn.Linear)  # 设置根模块为Linear
            .set_reference_quantized_module(nnqr.Linear)  # 设置参考量化模块为Linear
            .set_qat_module(nniqat.LinearReLU))  # 设置QAT模块为LinearReLU

    # linear relu, qat fused module
    linear_configs.append(
        BackendPatternConfig(nniqat.LinearReLU)
            .set_observation_type(observation_type)  # 设置观测类型
            .set_dtype_configs(dtype_configs)  # 设置数据类型配置
            .set_root_module(torch.nn.Linear)  # 设置根模块为Linear
            .set_reference_quantized_module(nnqr.Linear))  # 设置参考量化模块为Linear

    # 2.3 functional linear + relu configs
    # linear relu, functional linear + relu module
    linear_configs.append(
        BackendPatternConfig((F.linear, torch.nn.ReLU))
            .set_observation_type(observation_type)  # 设置观测类型
            .set_dtype_configs(dtype_configs))  # 设置数据类型配置

    # linear relu, functional linear + functional relu
    linear_configs.append(
        BackendPatternConfig((F.linear, F.relu))
            .set_observation_type(observation_type)  # 设置观测类型
            .set_dtype_configs(dtype_configs))  # 设置数据类型配置

    # (3) Linear + batchnorm
    # ------------------------
    # 3.1 linear bn fusion
    linear_configs.append(
        BackendPatternConfig((nn.Linear, nn.BatchNorm1d))
            .set_dtype_configs(dtype_configs)  # 设置数据类型配置
            .set_fuser_method(fuse_linear_bn)  # 设置融合方法为线性-BatchNorm1d模式
            .set_fused_module(nni.LinearBn1d))  # 设置融合后的模块为LinearBn1d

    # 3.2 linear bn fused
    # linear bn, fused module
    linear_configs.append(
        BackendPatternConfig(nni.LinearBn1d)
            .set_observation_type(observation_type)  # 设置观测类型
            .set_dtype_configs(dtype_configs)  # 设置数据类型配置
            .set_root_module(torch.nn.Linear)  # 设置根模块为Linear
            .set_reference_quantized_module(nnqr.Linear)  # 设置参考量化模块为Linear
            .set_qat_module(nniqat.LinearBn1d))  # 设置QAT模块为LinearBn1d

    # linear bn, qat fused module
    linear_configs.append(
        BackendPatternConfig(nniqat.LinearBn1d)
            .set_observation_type(observation_type)  # 设置观测类型
            .set_dtype_configs(dtype_configs)  # 设置数据类型配置
            .set_root_module(torch.nn.Linear)  # 设置根模块为Linear
            .set_reference_quantized_module(nnqr.Linear))  # 设置参考量化模块为Linear

    return linear_configs
def _get_conv_configs(dtype_configs):
    """
    Return all configs related to conv modules and ops.
    """
    # 初始化一个空列表，用于存储卷积配置
    conv_configs = []
    # 设置观察类型为OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    # 返回空的卷积配置列表
    return conv_configs

def _get_cat_config(dtype_configs: List[DTypeConfig]) -> BackendPatternConfig:
    # 返回一个BackendPatternConfig对象，该对象使用torch.cat作为操作
    return BackendPatternConfig(torch.cat) \
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT) \
        .set_dtype_configs(dtype_configs)

def _get_ln_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    # 初始化一个空列表，用于存储LayerNorm配置
    ln_configs = []
    
    # 添加一个BackendPatternConfig对象到ln_configs列表，使用torch.nn.LayerNorm作为操作
    ln_configs.append(
        BackendPatternConfig(torch.nn.LayerNorm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(dtype_configs)
    )
    
    # 添加一个BackendPatternConfig对象到ln_configs列表，使用torch.nn.functional.layer_norm作为操作
    ln_configs.append(
        BackendPatternConfig(torch.nn.functional.layer_norm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 3})
    )
    
    # 返回LayerNorm配置列表ln_configs
    return ln_configs

def _get_default_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    # 初始化一个空列表，用于存储默认操作配置
    configs = []
    
    # 默认操作列表，包括一系列torch.nn和torch.nn.functional中的操作类
    default_ops = [
        torch.nn.ELU,
        torch.nn.LeakyReLU,
        torch.nn.Hardswish,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.Dropout,
        torch.nn.PReLU,
        torch.nn.functional.elu,
        torch.nn.functional.hardswish,
        torch.nn.functional.leaky_relu,
        torch.nn.functional.dropout,
    ]
    
    # 遍历默认操作列表，为每个操作创建一个BackendPatternConfig对象并添加到configs列表中
    for op in default_ops:
        configs.append(
            BackendPatternConfig(op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
                .set_dtype_configs(dtype_configs)
        )

    # 添加一个BackendPatternConfig对象到configs列表，使用torch.nn.functional.group_norm作为操作
    configs.append(
        BackendPatternConfig(torch.nn.functional.group_norm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 3})
    )

    # 添加一个BackendPatternConfig对象到configs列表，使用torch.nn.functional.instance_norm作为操作
    configs.append(
        BackendPatternConfig(torch.nn.functional.instance_norm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 3, "bias": 4})
    )
    
    # 返回默认操作配置列表configs
    return configs

def _add_fixed_qparams_to_dtype_configs(
    dtype_configs: List[DTypeConfig],
    constraints: DTypeWithConstraints,
) -> List[DTypeConfig]:
    """
    Return a copy of the list of DTypeConfigs where activations are subject to the specified
    constraints required for fixed qparams ops.

    If the data type doesn't match the one in the constraints, simply leave the corresponding
    DTypeConfig unchanged.

    If `scale_min_lower_bound` or `scale_max_upper_bound` is specified in the activations,
    """
    # 略，因为代码截断，未提供完整内容
    """
    throw an exception since these settings are incompatible with fixed qparams ops.
    """
    # 创建一个空列表，用于存储更新后的数据类型配置
    new_dtype_configs = []
    # 遍历每个数据类型配置
    for dtype_config in dtype_configs:
        # 深度复制当前数据类型配置，以防止修改原始对象
        dc = copy.deepcopy(dtype_config)
        # 遍历输入和输出数据类型约束列表
        for orig_constraints in [dc.input_dtype_with_constraints, dc.output_dtype_with_constraints]:
            # 如果约束的数据类型与给定的约束数据类型不匹配，则跳过
            if orig_constraints.dtype != constraints.dtype:
                continue
            # 如果存在最小缩放下界，抛出异常，因为对于固定的量化参数操作是无效的
            if orig_constraints.scale_min_lower_bound is not None:
                raise ValueError(f"scale_min_lower_bound is invalid for fixed qparams ops: {dtype_config}")
            # 如果存在最大缩放上界，抛出异常，因为对于固定的量化参数操作是无效的
            if orig_constraints.scale_max_upper_bound is not None:
                raise ValueError(f"scale_max_upper_bound is invalid for fixed qparams ops: {dtype_config}")
            # 更新原始约束的量化参数下界和上界，以及精确匹配标志和零点精确匹配标志
            orig_constraints.quant_min_lower_bound = constraints.quant_min_lower_bound
            orig_constraints.quant_max_upper_bound = constraints.quant_max_upper_bound
            orig_constraints.scale_exact_match = constraints.scale_exact_match
            orig_constraints.zero_point_exact_match = constraints.zero_point_exact_match
        # 将更新后的数据类型配置添加到新列表中
        new_dtype_configs.append(dc)
    # 返回更新后的数据类型配置列表
    return new_dtype_configs
# 根据给定的 dtype_configs 列表生成固定量化参数操作的配置列表
def _get_fixed_qparams_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    # 初始化空的固定量化参数操作配置列表
    fixed_qparams_op_configs = []
    # 遍历 _FIXED_QPARAMS_OP_TO_CONSTRAINTS 字典中的固定量化参数操作及其约束条件
    for fixed_qparam_op, constraints in _FIXED_QPARAMS_OP_TO_CONSTRAINTS.items():
        # 将固定量化参数添加到给定的 dtype_configs 中，生成新的 dtype 配置列表
        new_dtype_configs = _add_fixed_qparams_to_dtype_configs(dtype_configs, constraints)
        # 创建 BackendPatternConfig 对象并设置观测类型为 OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
        # 设置 dtype 配置列表，并添加到 fixed_qparams_op_configs 列表中
        fixed_qparams_op_configs.append(
            BackendPatternConfig(fixed_qparam_op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .set_dtype_configs(new_dtype_configs))
    # 返回配置好的固定量化参数操作列表
    return fixed_qparams_op_configs

# 获取可以共享量化参数的操作的配置
def _get_share_qparams_op_configs(dtype_configs):
    """ 获取操作符的配置，适用于既有浮点数输入又有量化输入
    如果输入是量化的，输出张量将与输入共享相同的量化参数。
    示例操作符：
    avgpool2d, reshape, transpose, maxpool2d
    示例观察操作符：
    observer_0 - avgpool2d - observer_0（与输入共享相同的观察器实例）
    """

    def _get_share_qprams_op_backend_config(op):
        # 创建 BackendPatternConfig 对象并设置观测类型为 OUTPUT_SHARE_OBSERVER_WITH_INPUT
        # 设置 dtype 配置列表，并返回配置对象
        return BackendPatternConfig(op) \
            .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT) \
            .set_dtype_configs(dtype_configs)

    # 定义可以共享量化参数的操作列表
    share_qparams_ops = [
        torch.nn.AdaptiveAvgPool1d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.AdaptiveAvgPool3d,
        torch.nn.AvgPool1d,
        torch.nn.AvgPool2d,
        torch.nn.AvgPool3d,
        torch.nn.Hardtanh,
        torch.nn.Identity,
        torch.nn.MaxPool1d,
        torch.nn.MaxPool2d,
        torch.nn.MaxPool3d,
        torch.nn.PixelShuffle,
        torch.nn.PixelUnshuffle,
        torch.nn.ReLU,
        torch.nn.ReLU6,
        torch.adaptive_avg_pool1d,
        torch.nn.functional.adaptive_avg_pool2d,
        torch.nn.functional.adaptive_avg_pool3d,
        torch.nn.functional.hardtanh,
        torch.nn.functional.hardtanh_,
        torch.nn.functional.interpolate,
        torch.nn.functional.max_pool1d,
        torch.nn.functional.max_pool2d,
        torch.nn.functional.max_pool3d,
        torch.nn.functional.pixel_shuffle,
        torch.nn.functional.pixel_unshuffle,
        torch.nn.functional.relu,
        torch.nn.functional.relu6,
        torch.avg_pool1d,
        torch._C._nn.avg_pool2d,
        torch._C._nn.avg_pool3d,
        torch.clamp,
        torch.flatten,
        torch.mean,
        torch.narrow,
        torch.repeat_interleave,
        torch.transpose,
        torch.squeeze,
        torch.stack,
        torch.unsqueeze,
        operator.floordiv,
        "contiguous",
        "clamp",
        "detach",
        "detach_",
        "mean",
        "permute",
        "repeat",
        "repeat_interleave",
        "reshape",
        "resize_",
        "relu",
        "relu_",
        "squeeze",
        "squeeze_",
        "transpose",
        "unsqueeze",
        "unsqueeze_",
        "view"
    ]
    # 遍历 share_qparams_ops 列表中的每个元素 op，调用 _get_share_qprams_op_backend_config 函数并返回结果组成的列表
    return [_get_share_qprams_op_backend_config(op) for op in share_qparams_ops]
# 返回与批归一化相关的配置列表
def _get_bn_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    bn_configs = []  # 初始化空的批归一化配置列表

    # 定义标准批归一化模块与融合批归一化模块之间的映射关系
    bn_to_fused_bn = {
        torch.nn.BatchNorm2d: nni.BNReLU2d,
        torch.nn.BatchNorm3d: nni.BNReLU3d,
    }

    # 遍历标准批归一化模块，生成与ReLU融合的配置
    for bn in bn_to_fused_bn.keys():
        fused_bn = bn_to_fused_bn[bn]

        # 添加标准批归一化模块与ReLU融合的配置
        bn_configs.append(
            BackendPatternConfig((bn, nn.ReLU))
                .set_dtype_configs(dtype_configs)  # 设置数据类型配置
                .set_fuser_method(_sequential_wrapper2(fused_bn))  # 设置融合方法
                .set_fused_module(fused_bn))  # 设置融合后的模块

        # 添加标准批归一化模块与F.relu融合的配置
        bn_configs.append(
            BackendPatternConfig((bn, F.relu))
                .set_dtype_configs(dtype_configs)  # 设置数据类型配置
                .set_fuser_method(_sequential_wrapper2(fused_bn))  # 设置融合方法
                .set_fused_module(fused_bn))  # 设置融合后的模块

        # 添加标准批归一化模块的配置，不进行融合
        bn_configs.append(
            BackendPatternConfig(bn)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # 设置观察类型
                .set_dtype_configs(dtype_configs))  # 设置数据类型配置

    # 遍历所有融合批归一化模块，添加配置
    for fused_bn in bn_to_fused_bn.values():
        bn_configs.append(
            BackendPatternConfig(fused_bn)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # 设置观察类型
                .set_dtype_configs(dtype_configs))  # 设置数据类型配置

    return bn_configs  # 返回批归一化配置列表


# 返回与循环神经网络操作相关的配置列表
def _get_rnn_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    rnn_op_configs = []  # 初始化空的循环神经网络操作配置列表

    # 遍历循环神经网络操作和相应的参考量化循环神经网络操作
    for rnn_op, ref_rnn_op in [
            (nn.GRUCell, nnqr.GRUCell),
            (nn.LSTMCell, nnqr.LSTMCell),
            (nn.RNNCell, nnqr.RNNCell),
            (nn.LSTM, nnqr.LSTM),
            (nn.GRU, nnqr.GRU)
    ]:
        # 添加循环神经网络操作的配置
        rnn_op_configs.append(
            BackendPatternConfig(rnn_op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # 设置观察类型
                .set_dtype_configs(dtype_configs)  # 设置数据类型配置
                .set_root_module(rnn_op)  # 设置根模块
                .set_reference_quantized_module(ref_rnn_op))  # 设置参考量化模块

    return rnn_op_configs  # 返回循环神经网络操作配置列表


# 返回与嵌入操作相关的配置列表
def _get_embedding_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    embedding_op_configs = []  # 初始化空的嵌入操作配置列表

    # 遍历嵌入操作、量化自动量化嵌入操作和参考嵌入操作
    for embedding_op, qat_embedding_op, ref_embedding_op in [
            (nn.Embedding, nnqat.Embedding, nnqr.Embedding),
            (nn.EmbeddingBag, nnqat.EmbeddingBag, nnqr.EmbeddingBag),
    ]:
        # 将配置项添加到 embedding_op_configs 列表中，用于普通操作符
        embedding_op_configs.append(
            # 创建 BackendPatternConfig 对象，并配置相关属性
            BackendPatternConfig(embedding_op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .set_dtype_configs(dtype_configs)  # 设置数据类型配置
                .set_qat_module(qat_embedding_op)  # 设置量化训练模块
                .set_root_module(embedding_op)  # 设置根模块
                .set_reference_quantized_module(ref_embedding_op))  # 设置参考量化模块

        # 配置量化训练操作符的配置项
        embedding_op_configs.append(
            BackendPatternConfig(qat_embedding_op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .set_dtype_configs(dtype_configs)  # 设置数据类型配置
                .set_root_module(embedding_op)  # 设置根模块
                .set_reference_quantized_module(ref_embedding_op))  # 设置参考量化模块
    # 返回配置好的 embedding_op_configs 列表
    return embedding_op_configs
# 定义一个函数 _get_tensor_info_op_configs，用于生成操作配置列表，这些操作处理不同数据类型的张量，
# 但返回包含输入张量信息的非张量对象。

def _get_tensor_info_op_configs(dtype_configs):
    """
    These ops work on tensors of different dtypes but return non-tensors
    containing information about the input tensor.
    """

    # 定义内部函数 _get_config，用于为指定的操作生成配置对象
    def _get_config(op):
        # 创建 BackendPatternConfig 对象，用指定操作初始化
        return BackendPatternConfig(op) \
            # 设置观察类型为输入输出未观察
            .set_observation_type(ObservationType.INPUT_OUTPUT_NOT_OBSERVED) \
            # 设置数据类型配置为给定的 dtype_configs
            .set_dtype_configs(dtype_configs)

    # 返回一个列表，其中每个元素是通过 _get_config 函数生成的配置对象，对应于操作 "shape" 和 "size"
    return [_get_config(op) for op in ("shape", "size")]
```