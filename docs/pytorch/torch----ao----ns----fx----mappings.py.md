# `.\pytorch\torch\ao\ns\fx\mappings.py`

```py
# 导入运算符模块，提供标准的运算符函数实现
import operator

# 导入 PyTorch 深度学习库
import torch
import torch.nn as nn
import torch.nn.functional as F
toq = torch.ops.quantized

# 导入量化后的神经网络模块
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.ao.nn.qat.dynamic as nnqatd

# 导入与量化后端相关的配置函数
from torch.ao.quantization.backend_config import get_native_backend_config

# 导入用于将量化模块降级为本地后端的函数
import torch.ao.quantization.fx._lower_to_native_backend as \
    _lower_to_native_backend

# 导入量化映射配置
import torch.ao.quantization.quantization_mappings as quantization_mappings

# 导入自定义类型
from .ns_types import NSNodeTargetType

# 导入类型提示
from typing import Callable, Dict, List, Optional, Set, Tuple

# 定义一个函数，返回基础名称到相关操作集合的字典
def get_base_name_to_sets_of_related_ops() -> Dict[str, Set[NSNodeTargetType]]:
    # 注意：此集合将在后面根据 backend_config 的条目进行修改

    # 获取本地后端的配置信息
    backend_config = get_native_backend_config()

    # 初始化新的连接列表，用于存储操作对
    new_connections: List[Tuple[Callable, Callable]] = [
        # 技术债务特殊情况
        (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear),
    ]

    # 遍历后端配置中的复杂模式格式及其配置项
    for pattern, config in backend_config._pattern_complex_format_to_config.items():

        # 模式格式：(c, (b, a))
        first_element = pattern
        # 从末尾开始查找，因为模式是倒序的
        while isinstance(first_element, (list, tuple)):
            first_element = first_element[-1]

        # 如果有融合模块，则添加到新连接列表中
        if config.fused_module is not None:
            # 情况1：模式将一组操作融合为一个操作
            # 示例：nn.Conv1d 和 nn.ReLU 被融合为 nni.ConvReLU1d
            new_connections.append((first_element, config.fused_module))

        # 如果有量化训练模块，则添加到新连接列表中
        if config.qat_module is not None:
            # 情况2：模式将模块替换为量化训练模块
            # 示例：nni.ConvReLU1d 替换为 nniqat.ConvReLU1d
            new_connections.append((first_element, config.qat_module))

        # 如果有参考量化模块，则添加到新连接列表中
        if config.reference_quantized_module is not None:
            # 情况3：浮点模块的参考版本，如 nn.Conv2d 和 nnqr.Conv2d
            new_connections.append((first_element, config.reference_quantized_module))

    #
    # 从默认降级路径中添加参考模块替换
    #

    # 遍历不同的降级模块映射，添加到新连接列表中
    for source_to_target in (
        _lower_to_native_backend.STATIC_LOWER_MODULE_MAP,
        _lower_to_native_backend.DYNAMIC_LOWER_MODULE_MAP,
        _lower_to_native_backend.WEIGHT_ONLY_LOWER_MODULE_MAP,
        _lower_to_native_backend.SPECIAL_PATTERN_LOWER_MODULE_MAP,
    ):
        for source, target in source_to_target.items():  # type: ignore[attr-defined]
            new_connections.append((source, target))
    # 遍历静态融合模块映射、静态双输入融合模块映射和动态融合模块映射
    for source_to_double_target in (
        _lower_to_native_backend.STATIC_LOWER_FUSED_MODULE_MAP,
        _lower_to_native_backend.STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP,
        _lower_to_native_backend.DYNAMIC_LOWER_FUSED_MODULE_MAP,
    ):
        # 遍历每个映射中的源与双目标
        for source, (target1, target2) in source_to_double_target.items():  # type: ignore[attr-defined]
            # 将源与目标1、目标2的连接添加到新连接列表中
            new_connections.append((source, target1))
            new_connections.append((source, target2))

    #
    # 添加默认降低路径中的函数交换
    #

    # 遍历静态功能映射，将源与两个目标的连接添加到新连接列表中
    for source, (target1, target2) in \
            _lower_to_native_backend.STATIC_LOWER_FUNCTIONAL_MAP.items():
        new_connections.append((source, target1))
        new_connections.append((source, target2))

    # 遍历量化操作映射、ReLU量化操作映射和默认浮点到量化操作映射
    for source_to_target in (
        _lower_to_native_backend.QBIN_OP_MAPPING,
        _lower_to_native_backend.QBIN_RELU_OP_MAPPING,
        quantization_mappings.DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS,
    ):
        # 遍历每个映射中的源与目标，将其连接添加到新连接列表中
        for source, target in source_to_target.items():
            new_connections.append((source, target))

    #
    # 添加其他交换，理想情况下，当降低代码停止使用这些交换时，可以移除这部分代码。
    #
    # 遍历默认动态量化模块映射，将源与目标的连接添加到新连接列表中
    for source_to_target in (
        quantization_mappings.DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
    ):
        for source, target in source_to_target.items():
            new_connections.append((source, target))

    # 从后端配置中添加新连接
    for item1, item2 in new_connections:
        # 遍历集合中的相关操作集合，将新连接中的项添加到相关操作集合中
        for set_of_related_ops in sets_of_related_ops:
            if item1 in set_of_related_ops or item2 in set_of_related_ops:
                set_of_related_ops.add(item1)
                set_of_related_ops.add(item2)
                break

    # 创建基础名称到相关操作集合的字典
    base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]] = {}

    counter = 0
    # 遍历集合中的相关操作集合，为每个集合创建一个唯一的基础名称，并将其与相关操作集合关联起来
    for set_of_related_ops in sets_of_related_ops:
        base_name = str(counter)
        counter += 1
        base_name_to_sets_of_related_ops[base_name] = set_of_related_ops

    # 返回基础名称到相关操作集合的字典
    return base_name_to_sets_of_related_ops
# 根据给定的操作（op）查找其对应的基本名称（base_name），并返回找到的第一个匹配项
def get_base_name_for_op(
    base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]],
    op: NSNodeTargetType,
) -> Optional[str]:
    # 遍历 base_name_to_sets_of_related_ops 字典
    for base_name, set_of_related_ops in base_name_to_sets_of_related_ops.items():
        # 如果 op 在当前 set_of_related_ops 中，则返回对应的 base_name
        if op in set_of_related_ops:
            return base_name
    # 如果没有找到匹配项，则返回 None
    return None


# 将操作（op）添加到相关操作集合中
def add_op_to_sets_of_related_ops(
    base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]],
    op: NSNodeTargetType,
    related_op: Optional[NSNodeTargetType],
) -> None:
    # 如果 related_op 不为 None，则将 op 添加到相关操作集合中
    if related_op is not None:
        # 遍历 base_name_to_sets_of_related_ops 中的所有集合
        for set_of_related_ops in base_name_to_sets_of_related_ops.values():
            # 如果 related_op 存在于当前集合中，则将 op 添加到该集合中并返回
            if related_op in set_of_related_ops:
                set_of_related_ops.add(op)
                return
        # 如果找不到 related_op，则引发断言错误
        raise AssertionError(f"{related_op} was not found")
    else:
        # 如果 related_op 为 None，则生成一个唯一的字符串作为 base_name 并将 op 添加到新集合中
        counter = 0
        while str(counter) in base_name_to_sets_of_related_ops:
            counter += 1
        base_name_to_sets_of_related_ops[str(counter)] = {op}


# 获取节点类型到输入输出类型映射的字典
# TODO(future PR): clean this up
def get_node_type_to_io_type_map() -> Dict[str, Set[NSNodeTargetType]]:
    # 定义基于 FP32 的操作集合 FUNS_IO_TYPE_FP32
    FUNS_IO_TYPE_FP32: Set[NSNodeTargetType] = {
        F.linear,
        F.conv1d,
        F.conv2d,
        F.conv3d,
        torch.cat,
        F.elu,
        F.hardswish,
        F.instance_norm,
        F.layer_norm,
        F.leaky_relu,
        F.dropout,
        F.silu,
        F.mish,
        operator.add,
        torch.add,
        operator.mul,
        torch.mul,
        torch.sum,
        F.prelu,
    }

    # 定义空的 FP16 操作集合 FUNS_IO_TYPE_FP16
    FUNS_IO_TYPE_FP16: Set[NSNodeTargetType] = set()

    # 定义基于 INT8 的操作集合 FUNS_IO_TYPE_INT8
    FUNS_IO_TYPE_INT8: Set[NSNodeTargetType] = {
        toq.linear,
        toq.linear_relu,
        toq.conv1d,
        toq.conv1d_relu,
        toq.conv2d,
        toq.conv2d_relu,
        toq.conv3d,
        toq.conv3d_relu,
        toq.cat,
        toq.elu,
        toq.hardswish,
        toq.instance_norm,
        toq.layer_norm,
        toq.leaky_relu,
        toq.dropout,
        toq.prelu,
        # TODO(future PR): implement shadowing for binary ops and
        # uncomment below
        # toq.add,
        # toq.mul,
    }
    # 定义一个集合，包含所有接受浮点型或整型输入并产生相同类型输出的函数或操作
    FUNS_IO_TYPE_FP32_OR_INT8: Set[NSNodeTargetType] = {
        F.relu,                                 # ReLU 激活函数
        F.tanh,                                 # 双曲正切激活函数
        torch.tanh,                             # PyTorch 的双曲正切函数
        F.sigmoid,                              # Sigmoid 激活函数
        torch.sigmoid,                          # PyTorch 的 Sigmoid 函数
        F.hardsigmoid,                          # Hard Sigmoid 激活函数
        operator.floordiv,                      # 整数除法操作符
        torch.adaptive_avg_pool1d,              # 一维自适应平均池化函数
        F.adaptive_avg_pool2d,                  # 二维自适应平均池化函数
        F.adaptive_avg_pool3d,                  # 三维自适应平均池化函数
        F.dropout,                              # Dropout 函数
        F.hardtanh,                             # HardTanh 激活函数
        F.hardtanh_,                            # Inplace HardTanh 激活函数
        F.interpolate,                          # 插值函数
        F.max_pool1d,                           # 一维最大池化函数
        F.max_pool2d,                           # 二维最大池化函数
        F.max_pool3d,                           # 三维最大池化函数
        F.relu6,                                # ReLU6 激活函数
        F.pixel_shuffle,                        # 像素混洗函数
        F.pixel_unshuffle,                      # 像素反混洗函数
        torch.avg_pool1d,                       # 一维平均池化函数
        torch._C._nn.avg_pool2d,                # PyTorch C++ 扩展的二维平均池化函数
        torch._C._nn.avg_pool3d,                # PyTorch C++ 扩展的三维平均池化函数
        torch.cat,                              # 张量拼接函数
        torch.chunk,                            # 张量分块函数
        torch.clamp,                            # 张量裁剪函数
        torch.flatten,                          # 张量展平函数
        torch.transpose,                        # 张量转置函数
        torch.max,                              # 张量最大值函数
        torch.mean,                             # 张量均值函数
        torch.min,                              # 张量最小值函数
        torch.narrow,                           # 张量窄化函数
        torch.repeat_interleave,                # 重复插入函数
        torch.sort,                             # 张量排序函数
        torch.squeeze,                          # 去除维度为1的张量维度函数
        torch.stack,                            # 张量堆叠函数
        torch.unsqueeze,                        # 增加维度为1的张量维度函数
        operator.add,                           # 加法操作符
    }
    
    # 定义一个集合，包含所有接受浮点型输入并产生浮点型输出的模块或类
    MODS_IO_TYPE_FP32: Set[NSNodeTargetType] = {
        nn.Linear,                              # 线性层
        nnqat.Linear,                           # QAT（量化感知训练）线性层
        nnqatd.Linear,                          # QAT（量化感知训练）带权重衰减的线性层
        nnqd.Linear,                            # QD（量化训练）线性层
        torch.nn.modules.linear.NonDynamicallyQuantizableLinear,  # 非动态量化线性层
        nn.Conv1d,                              # 一维卷积层
        nn.Conv2d,                              # 二维卷积层
        nn.Conv3d,                              # 三维卷积层
        nnqat.Conv1d,                           # QAT（量化感知训练）一维卷积层
        nnqat.Conv2d,                           # QAT（量化感知训练）二维卷积层
        nnqat.Conv3d,                           # QAT（量化感知训练）三维卷积层
        nnqat.Embedding,                        # QAT（量化感知训练）嵌入层
        nnqat.EmbeddingBag,                     # QAT（量化感知训练）嵌入袋层
        nn.LSTM,                                # LSTM 层
        nnqd.LSTM,                              # QD（量化训练）LSTM 层
        nn.BatchNorm2d,                         # 二维批标准化层
        nn.BatchNorm3d,                         # 三维批标准化层
        nn.Dropout,                             # Dropout 层
        nn.ConvTranspose1d,                     # 一维转置卷积层
        nn.ConvTranspose2d,                     # 二维转置卷积层
        nn.ConvTranspose3d,                     # 三维转置卷积层
        nn.ELU,                                 # ELU 激活函数
        nn.GroupNorm,                           # 组标准化层
        nn.InstanceNorm1d,                      # 一维实例标准化层
        nn.InstanceNorm2d,                      # 二维实例标准化层
        nn.InstanceNorm3d,                      # 三维实例标准化层
        nn.LayerNorm,                           # 层标准化层
        nn.Hardswish,                           # Hardswish 激活函数
        nn.LeakyReLU,                           # LeakyReLU 激活函数
        nn.ReLU6,                               # ReLU6 激活函数
        nn.SiLU,                                # SiLU（Swish）激活函数
        nn.Mish,                                # Mish 激活函数
        nn.Softmax,                             # Softmax 函数
        nn.PReLU,                               # PReLU 激活函数
        nni.BNReLU2d,                           # 批标准化 + ReLU 的二维结构
        nni.BNReLU3d,                           # 批标准化 + ReLU 的三维结构
        nni.ConvReLU1d,                         # 卷积 + ReLU 的一维结构
        nni.ConvReLU2d,                         # 卷积 + ReLU 的二维结构
        nni.ConvReLU3d,                         # 卷积 + ReLU 的三维结构
        nni.LinearReLU,                         # 线性层 + ReLU 结构
        nni.LinearBn1d,                         # 线性层 + 批标准化的一维结构
        nni.ConvBn1d,                           # 卷积 + 批标准化的一维结构
        nni.ConvBn2d,                           # 卷积 + 批标准化的二维结构
        nni.ConvBn3d,                           # 卷积 + 批标准化的三维结构
        nniqat.ConvBn1d,                        # QAT 卷积 + 批标准化的一维结构
        nniqat.ConvBn2d,                        # QAT 卷积 + 批标准化的二维结构
        nniqat.ConvBn3d,                        # QAT 卷积 + 批标准化的三维结构
        nniqat.ConvBnReLU1d,                    # QAT 卷积 + 批标准化 + ReLU 的一维结构
        nniqat.ConvBnReLU2d,                    # QAT 卷积 + 批标准化 + ReLU 的二维结构
        nniqat.ConvBnReLU3d,                    # QAT 卷积 + 批标准化 + ReLU 的三维结构
        nniqat.ConvReLU1d,                      # QAT 卷积 + ReLU 的一维结构
        nniqat.ConvReLU2d,                      # QAT 卷积 + ReLU 的二维结构
        nniqat.ConvReLU3d,                      # QAT
    # 定义 MODS_IO_TYPE_INT8 集合，包含多种神经网络节点类型
    MODS_IO_TYPE_INT8: Set[NSNodeTargetType] = {
        nnq.Linear,                 # 线性层
        nnq.Conv1d,                 # 一维卷积层
        nnq.Conv2d,                 # 二维卷积层
        nnq.Conv3d,                 # 三维卷积层
        nnq.BatchNorm2d,            # 二维批归一化层
        nnq.BatchNorm3d,            # 三维批归一化层
        nnq.Dropout,                # Dropout 层
        nnq.ConvTranspose1d,        # 一维转置卷积层
        nnq.ConvTranspose2d,        # 二维转置卷积层
        nnq.ELU,                    # ELU 激活函数
        nnq.InstanceNorm1d,         # 一维实例归一化层
        nnq.InstanceNorm2d,         # 二维实例归一化层
        nnq.InstanceNorm3d,         # 三维实例归一化层
        nnq.LayerNorm,              # 层归一化层
        nnq.Hardswish,              # Hardswish 激活函数
        nnq.LeakyReLU,              # LeakyReLU 激活函数
        nnq.Embedding,              # 嵌入层
        nnq.EmbeddingBag,           # 嵌入包层
        nnq.Dropout,                # Dropout 层
        nnq.Softmax,                # Softmax 层
        nnq.PReLU,                  # PReLU 激活函数
        nniq.BNReLU2d,              # 自定义二维批归一化与ReLU结合层
        nniq.BNReLU3d,              # 自定义三维批归一化与ReLU结合层
        nniq.ConvReLU1d,            # 自定义一维卷积与ReLU结合层
        nniq.ConvReLU2d,            # 自定义二维卷积与ReLU结合层
        nniq.ConvReLU3d,            # 自定义三维卷积与ReLU结合层
        nniq.LinearReLU,            # 自定义线性层与ReLU结合层
        nniq.LinearLeakyReLU,       # 自定义线性层与LeakyReLU结合层
        nniq.LinearTanh,            # 自定义线性层与Tanh激活函数结合层
        nniq.ConvAdd2d,             # 自定义二维卷积与加法结合层
        nniq.ConvAddReLU2d,         # 自定义二维卷积、加法与ReLU结合层
    }

    # 定义 MODS_IO_TYPE_FP32_OR_INT8 集合，包含多种神经网络节点类型
    MODS_IO_TYPE_FP32_OR_INT8: Set[NSNodeTargetType] = {
        nn.ReLU,                    # ReLU 激活函数
        nn.Tanh,                    # Tanh 激活函数
        nn.Sigmoid,                 # Sigmoid 激活函数
        nn.Hardsigmoid,             # Hardsigmoid 激活函数
        nn.AdaptiveAvgPool1d,       # 自适应一维平均池化层
        nn.AdaptiveAvgPool2d,       # 自适应二维平均池化层
        nn.AdaptiveAvgPool3d,       # 自适应三维平均池化层
        nn.AvgPool1d,               # 一维平均池化层
        nn.AvgPool2d,               # 二维平均池化层
        nn.AvgPool3d,               # 三维平均池化层
        nn.Dropout,                 # Dropout 层
        nn.Hardtanh,                # Hardtanh 激活函数
        nn.Identity,                # 恒等映射层
        nn.MaxPool1d,               # 一维最大池化层
        nn.MaxPool2d,               # 二维最大池化层
        nn.MaxPool3d,               # 三维最大池化层
        nn.PixelShuffle,            # 像素洗牌层
        nn.PixelUnshuffle,          # 像素反洗牌层
        nn.ReLU6,                   # ReLU6 激活函数
    }

    # 定义 METHS_IO_TYPE_FP32_OR_INT8 集合，包含多个字符串表示的方法名
    METHS_IO_TYPE_FP32_OR_INT8: Set[NSNodeTargetType] = {
        'sigmoid_',                 # sigmoid_ 方法
        'sigmoid',                  # sigmoid 方法
        'tanh_',                    # tanh_ 方法
        'tanh',                     # tanh 方法
        'hardsigmoid_',             # hardsigmoid_ 方法
        'hardsigmoid',              # hardsigmoid 方法
        'relu_',                    # relu_ 方法
        'relu',                     # relu 方法
    }

    # 返回一个包含各种类型集合的字典
    return {
        'funs_io_type_fp32': FUNS_IO_TYPE_FP32,
        'funs_io_type_fp16': FUNS_IO_TYPE_FP16,
        'funs_io_type_int8': FUNS_IO_TYPE_INT8,
        'funs_io_type_fp32_or_int8': FUNS_IO_TYPE_FP32_OR_INT8,
        'mods_io_type_fp32': MODS_IO_TYPE_FP32,
        'mods_io_type_int8': MODS_IO_TYPE_INT8,
        'mods_io_type_fp32_or_int8': MODS_IO_TYPE_FP32_OR_INT8,
        'meths_io_type_fp32_or_int8': METHS_IO_TYPE_FP32_OR_INT8,
    }
# 返回一个字典，其中包含不可匹配类型的集合，分为函数、模块和方法
def get_unmatchable_types_map() -> Dict[str, Set[NSNodeTargetType]]:
    # 不可匹配的函数集合，包括 torch.quantize_per_tensor 和 operator.getitem
    FUNS_UNMATCHABLE: Set[NSNodeTargetType] = {
        torch.quantize_per_tensor,
        operator.getitem,
    }

    # 不可匹配的模块集合，包括 nn.Identity
    MODS_UNMATCHABLE: Set[NSNodeTargetType] = {
        nn.Identity,
    }

    # 不可匹配的方法集合，包括各种操作如 'to', 'dequantize' 等
    METHS_UNMATCHABLE: Set[NSNodeTargetType] = {
        'to',
        'dequantize',
        'reshape',
        'view',
        'unsqueeze_',
        'unsqueeze',
        'transpose',
        'squeeze_',
        'squeeze',
        'size',
        'shape',
        'resize_',
        'repeat_interleave',
        'repeat',
        'permute',
        'numel',
        'mean',
        'detach_',
        'detach',
        'contiguous',
        'clamp',
        'chunk',
    }

    # 返回包含上述集合的字典，用于表示不可匹配类型的映射
    return {
        'funs_unmatchable': FUNS_UNMATCHABLE,
        'mods_unmatchable': MODS_UNMATCHABLE,
        'meths_unmatchable': METHS_UNMATCHABLE,
    }
```