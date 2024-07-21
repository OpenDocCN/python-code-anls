# `.\pytorch\torch\ao\quantization\qconfig.py`

```
# mypy: allow-untyped-defs
# 引入 namedtuple，用于创建带有名称的 tuple
from collections import namedtuple
# 引入类型相关的模块，包括 Optional, Any, Union, Type
from typing import Optional, Any, Union, Type
# 引入 deprecated 扩展，用于标记不推荐使用的功能
from typing_extensions import deprecated

# 引入 PyTorch 深度学习框架
import torch
# 引入神经网络模块
import torch.nn as nn
# 引入量化伪量化相关模块
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,  # 简单的伪量化
    FakeQuantizeBase,  # 伪量化基类
    default_fake_quant,  # 默认的伪量化
    default_dynamic_fake_quant,  # 默认的动态伪量化
    default_per_channel_weight_fake_quant,  # 默认的按通道权重伪量化
    default_weight_fake_quant,  # 默认的权重伪量化
    default_fused_act_fake_quant,  # 默认的融合激活伪量化
    default_fused_wt_fake_quant,  # 默认的融合权重伪量化
    FusedMovingAvgObsFakeQuantize,  # 融合移动平均观察伪量化
    default_fused_per_channel_wt_fake_quant,  # 默认的融合按通道权重伪量化
    default_embedding_fake_quant,  # 默认的嵌入层伪量化
    default_embedding_fake_quant_4bit,  # 默认的4位嵌入层伪量化
    fused_wt_fake_quant_range_neg_127_to_127,  # 范围为-127到127的融合权重伪量化
    fused_per_channel_wt_fake_quant_range_neg_127_to_127,  # 范围为-127到127的融合按通道权重伪量化
)

# 引入观察者模块
from .observer import (
    _PartialWrapper,  # 部分包装器
    MinMaxObserver,  # 最小-最大值观察者
    HistogramObserver,  # 直方图观察者
    MovingAverageMinMaxObserver,  # 移动平均最小-最大值观察者
    NoopObserver,  # 无操作观察者
    PlaceholderObserver,  # 占位符观察者
    ReuseInputObserver,  # 重复使用输入观察者
    default_debug_observer,  # 默认的调试观察者
    default_dynamic_quant_observer,  # 默认的动态量化观察者
    default_float_qparams_observer,  # 默认的浮点数量化参数观察者
    default_float_qparams_observer_4bit,  # 默认的4位浮点数量化参数观察者
    default_observer,  # 默认的观察者
    default_per_channel_weight_observer,  # 默认的按通道权重观察者
    default_placeholder_observer,  # 默认的占位符观察者
    default_weight_observer,  # 默认的权重观察者
    weight_observer_range_neg_127_to_127,  # 范围为-127到127的权重观察者
    per_channel_weight_observer_range_neg_127_to_127,  # 范围为-127到127的按通道权重观察者
    default_reuse_input_observer,  # 默认的重复使用输入观察者
    ObserverBase,  # 观察者基类
)
# 引入警告模块
import warnings
# 引入复制模块
import copy

# 暴露给外部的模块列表
__all__ = [
    "QConfig",  # 量化配置类
    # TODO: deprecated, remove
    "QConfigDynamic",  # 动态量化配置类，标记为过时
    "default_qconfig",  # 默认的量化配置
    "default_debug_qconfig",  # 默认的调试量化配置
    "default_per_channel_qconfig",  # 默认的按通道量化配置
    "default_dynamic_qconfig",  # 默认的动态量化配置
    "float16_dynamic_qconfig",  # 16位浮点动态量化配置
    "float16_static_qconfig",  # 16位浮点静态量化配置
    "per_channel_dynamic_qconfig",  # 按通道动态量化配置
    "float_qparams_weight_only_qconfig",  # 仅权重浮点参数量化配置
    "float_qparams_weight_only_qconfig_4bit",  # 4位仅权重浮点参数量化配置
    "default_quint8_weight_qconfig",  # 默认的8位量化权重配置
    "default_qat_qconfig",  # 默认的量化感知训练配置
    "default_dynamic_qat_qconfig",  # 默认的动态量化感知训练配置
    "default_weight_only_qconfig",  # 默认的仅权重量化配置
    "default_activation_only_qconfig",  # 默认的仅激活量化配置
    "default_qat_qconfig_v2",  # 默认的量化感知训练配置 v2
    "default_reuse_input_qconfig",  # 默认的重复使用输入量化配置
    "default_symmetric_qnnpack_qconfig",  # 默认的对称量化 QNNPACK 配置
    "default_per_channel_symmetric_qnnpack_qconfig",  # 默认的按通道对称量化 QNNPACK 配置
    "default_symmetric_qnnpack_qat_qconfig",  # 默认的对称量化感知训练 QNNPACK 配置
    "default_per_channel_symmetric_qnnpack_qat_qconfig",  # 默认的按通道对称量化感知训练 QNNPACK 配置
    "default_embedding_qat_qconfig",  # 默认的嵌入层量化感知训练配置
    "default_embedding_qat_qconfig_4bit",  # 默认的4位嵌入层量化感知训练配置
    "get_default_qconfig",  # 获取默认的量化配置
    "get_default_qat_qconfig",  # 获取默认的量化感知训练配置
    "get_default_qconfig_dict",  # 获取默认的量化配置字典
    "get_default_qat_qconfig_dict",  # 获取默认的量化感知训练配置字典
    "QConfigAny",  # 任意量化配置类
    "qconfig_equals",  # 比较量化配置是否相等的函数
]

# 定义量化配置类 QConfig，继承自 namedtuple
class QConfig(namedtuple('QConfig', ['activation', 'weight'])):
    """
    描述如何通过提供激活和权重的设置（观察者类）来量化网络的层或部分。

    注意，QConfig 需要包含观察者的类（如 MinMaxObserver）或一个在调用时返回实例的可调用对象，
    而不是具体的观察者实例本身。
    量化准备函数将为每个层多次实例化观察者。
    """
    """
    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::
    
      my_qconfig = QConfig(
          activation=MinMaxObserver.with_args(dtype=torch.qint8),
          weight=default_observer.with_args(dtype=torch.qint8))
    
    """
    # 定义一个名为 QConfig 的类，继承自默认的 object 类
    def __new__(cls, activation, weight):
        # 捕获常见错误
        # 如果 activation 或 weight 是 nn.Module 的实例，则引发 ValueError
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError("QConfig received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        # 调用父类的 __new__ 方法来创建实例
        return super().__new__(cls, activation, weight)
@deprecated(
    "`QConfigDynamic` is going to be deprecated in PyTorch 1.12, please use `QConfig` instead",
    category=FutureWarning,
)
# 定义 QConfigDynamic 类，用于描述动态量化网络层或部分的设置
class QConfigDynamic(namedtuple('QConfigDynamic', ['activation', 'weight'])):
    """
    Describes how to dynamically quantize a layer or a part of the network by providing
    settings (observer classes) for weights.

    It's like QConfig, but for dynamic quantization.

    Note that QConfigDynamic needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of the layers.

    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfigDynamic(weight=default_observer.with_args(dtype=torch.qint8))
    """
    def __new__(cls, activation=torch.nn.Identity, weight=torch.nn.Identity):
        # 捕捉常见错误
        if isinstance(weight, nn.Module):
            raise ValueError("QConfigDynamic received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        # 调用父类的构造函数创建新的 QConfigDynamic 实例
        return super().__new__(cls, activation, weight)


# 默认的 QConfig 配置
default_qconfig = QConfig(activation=default_observer,
                          weight=default_weight_observer)
"""
Default qconfig configuration.
"""

# 默认的用于调试的 QConfig 配置
default_debug_qconfig = QConfig(weight=default_weight_observer,
                                activation=default_debug_observer)
"""
Default qconfig configuration for debugging.
"""

# 默认的逐通道权重量化的 QConfig 配置
default_per_channel_qconfig = QConfig(activation=default_observer,
                                      weight=default_per_channel_weight_observer)
"""
Default qconfig configuration for per channel weight quantization.
"""

# 默认的动态 QConfig 配置
default_dynamic_qconfig = QConfig(activation=default_dynamic_quant_observer,
                                  weight=default_weight_observer)
"""
Default dynamic qconfig.
"""

# 使用 torch.float16 来量化权重的动态 QConfig 配置
float16_dynamic_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float16, is_dynamic=True),
                                  weight=PlaceholderObserver.with_args(dtype=torch.float16))
"""
Dynamic qconfig with weights quantized to `torch.float16`.
"""

# 使用 torch.float16 来量化激活函数和权重的动态 QConfig 配置
float16_static_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float16),
                                 weight=PlaceholderObserver.with_args(dtype=torch.float16))
"""
Dynamic qconfig with both activations and weights quantized to `torch.float16`.
"""

# 使用逐通道权重量化的动态 QConfig 配置
per_channel_dynamic_qconfig = QConfig(activation=default_dynamic_quant_observer,
                                      weight=default_per_channel_weight_observer)
"""
Dynamic qconfig with weights quantized per channel.
"""

# 仅对权重进行量化参数的 QConfig 配置
float_qparams_weight_only_qconfig = QConfig(
    activation=default_placeholder_observer,
    weight=default_float_qparams_observer)
"""
Dynamic qconfig with weights quantized with a floating point zero_point.
"""

# 定义一个 QConfig 对象，用于仅对权重进行量化，并使用浮点零点
float_qparams_weight_only_qconfig_4bit = QConfig(
    activation=default_placeholder_observer,
    weight=default_float_qparams_observer_4bit)

# 默认的 QAT（量化感知训练）配置
default_qat_qconfig = QConfig(activation=default_fake_quant,
                              weight=default_weight_fake_quant)
"""
Default qconfig for QAT.
"""

# 默认的动态 QAT 配置
default_dynamic_qat_qconfig = QConfig(activation=default_dynamic_fake_quant,
                                      weight=default_weight_fake_quant)
"""
Default qconfig for dynamic QAT.
"""

# 默认仅对权重进行量化的配置
default_weight_only_qconfig = QConfig(activation=torch.nn.Identity,
                                      weight=default_weight_fake_quant)
"""
Default qconfig for quantizing weights only.
"""

# 默认仅对激活值进行量化的配置
default_activation_only_qconfig = QConfig(activation=default_fake_quant,
                                          weight=torch.nn.Identity)
"""
Default qconfig for quantizing activations only.
"""

# 使用融合的观察器和伪量化模块的 QAT 配置，以优化训练性能
# 可以修改 fake_quantize.py 中的默认条目来修改激活值/权重观察器
default_qat_qconfig_v2 = QConfig(activation=default_fused_act_fake_quant, weight=default_fused_wt_fake_quant)
"""
Fused version of `default_qat_config`, has performance benefits.
"""

# 默认的重用输入张量观察器的配置，用于运算符（如 reshape）中重用输入张量的情况
default_reuse_input_qconfig = QConfig(activation=default_reuse_input_observer,
                                      weight=NoopObserver)
"""
Default qconfig for operators that reuse the observers from input Tensor, e.g. reshape
"""

def get_default_qconfig(backend='x86', version=0):
    """
    Returns the default PTQ qconfig for the specified backend.

    Args:
      * `backend` (str): a string representing the target backend. Currently supports
        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.

    Return:
        qconfig
    """
    supported_backends = ["fbgemm", "x86", "qnnpack", "onednn"]
    # 如果指定的后端不在支持的后端列表中，抛出断言错误
    if backend not in supported_backends:
        raise AssertionError(
            "backend: " + str(backend) +
            f" not supported. backend must be one of {supported_backends}"
        )
    # 如果版本号为0，则根据后端选择不同的量化配置
    if version == 0:
        # 如果后端是'fbgemm'，使用带有reduce_range=True参数的HistogramObserver观察器和默认的逐通道权重观察器
        if backend == 'fbgemm':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                              weight=default_per_channel_weight_observer)
        # 如果后端是'qnnpack'，使用带有reduce_range=False参数的HistogramObserver观察器和默认的权重观察器
        elif backend == 'qnnpack':
            # TODO: 使其兼容xnnpack的约束
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                              weight=default_weight_observer)
        # 如果后端是'onednn'，检查CPU是否支持VNNI指令集，若不支持发出警告
        elif backend == 'onednn':
            if not torch.cpu._is_cpu_support_vnni():
                warnings.warn(
                    "Default qconfig of oneDNN backend with reduce_range of false may have accuracy issues "
                    "on CPU without Vector Neural Network Instruction support.")
            # 使用带有reduce_range=False参数的HistogramObserver观察器和默认的逐通道权重观察器
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                              weight=default_per_channel_weight_observer)
        # 如果后端是'x86'，使用带有reduce_range=True参数的HistogramObserver观察器和默认的逐通道权重观察器
        elif backend == 'x86':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                              weight=default_per_channel_weight_observer)
        else:
            # 不会执行到这里，保留默认的量化配置
            qconfig = default_qconfig
    else:
        # 如果版本号不为0，抛出断言错误
        raise AssertionError("Version number: " + str(version) +
                             " in get_default_qconfig is not supported. Version number must be 0")

    # 返回选定的量化配置
    return qconfig
"""
定义了几个默认的量化配置（QConfig）变量，用于不同的量化应用场景和后端选择。

Symmetric qconfig:
symmetric_qnnpack_qconfig = QConfig(
    activation=HistogramObserver.with_args(dtype=torch.qint8, reduce_range=False, eps=2 ** -12),
    weight=weight_observer_range_neg_127_to_127
)
使用直方图观察器来量化激活值为有符号8位整数（torch.qint8），不减少范围，设置eps为2的负12次方。
权重的量化使用范围限制在[-127, 127]之间。

Per-channel symmetric qconfig:
per_channel_symmetric_qnnpack_qconfig = QConfig(
    activation=HistogramObserver.with_args(dtype=torch.qint8, reduce_range=False, eps=2 ** -12),
    weight=per_channel_weight_observer_range_neg_127_to_127
)
与对称qconfig相似，但权重使用逐通道量化，每个通道有独立的量化参数。

Embedding QAT qconfig:
default_embedding_qat_qconfig = QConfig(
    activation=NoopObserver.with_args(dtype=torch.float32),
    weight=default_embedding_fake_quant
)
用于嵌入层的量化训练感知训练（QAT）配置，激活函数观察器为NoopObserver，权重使用默认的伪量化操作。

4-bit Embedding QAT qconfig:
default_embedding_qat_qconfig_4bit = QConfig(
    activation=NoopObserver.with_args(dtype=torch.float32),
    weight=default_embedding_fake_quant_4bit
)
类似于默认嵌入层QAT配置，但权重使用4位量化的伪量化操作。

Quint8 Weight qconfig:
default_quint8_weight_qconfig = QConfig(
    activation=HistogramObserver,
    weight=MinMaxObserver
)
用于权重量化的配置，使用直方图观察器观察激活值，使用最小-最大观察器观察权重值。

函数 get_default_qat_qconfig：
def get_default_qat_qconfig(backend='x86', version=1):
    """
    返回指定后端的默认量化训练感知训练（QAT）配置。

    Args:
      * `backend` (str): 目标后端的字符串表示。当前支持`x86`（默认）、`fbgemm`、`qnnpack`和`onednn`。
      * `version`：版本，用于向后兼容。可以为`None`或`1`。

    Return:
        qconfig
    """
    supported_backends = ["fbgemm", "x86", "qnnpack", "onednn"]
    # 检查所选后端是否在支持的后端列表中
    if backend not in supported_backends:
        # 如果不在支持的后端列表中，抛出断言错误，并显示支持的后端列表信息
        raise AssertionError(
            "backend: " + str(backend) +
            f" not supported. backend must be one of {supported_backends}"
        )

    # 如果版本为0，说明执行量化感知训练时直方图观察器过慢
    if version == 0:
        # 根据后端选择合适的量化配置
        if backend == 'fbgemm':
            # 使用带参数的伪量化来配置 QConfig
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                quant_min=0,
                                                                quant_max=255,
                                                                reduce_range=True),
                              weight=default_per_channel_weight_fake_quant)
        elif backend == 'qnnpack':
            # 使用带参数的伪量化来配置 QConfig
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                quant_min=0,
                                                                quant_max=255,
                                                                reduce_range=False),
                              weight=default_weight_fake_quant)
        elif backend == 'onednn':
            # 使用带参数的伪量化来配置 QConfig
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                quant_min=0,
                                                                quant_max=255),
                              weight=default_per_channel_weight_fake_quant)
        elif backend == 'x86':
            # 使用带参数的伪量化来配置 QConfig
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                quant_min=0,
                                                                quant_max=255,
                                                                reduce_range=True),
                              weight=default_per_channel_weight_fake_quant)
        else:
            # 使用默认的量化感知训练配置
            qconfig = default_qat_qconfig
    # 对于其他版本，使用融合的观察器+伪量化模块进行量化感知训练
    # 如果版本号为1，则根据后端选择不同的量化配置
    elif version == 1:
        if backend == 'fbgemm':
            # 对于后端为 'fbgemm'，使用特定参数创建量化配置
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=0,
                                                                                 quant_max=255,
                                                                                 reduce_range=True),
                              weight=default_fused_per_channel_wt_fake_quant)
        elif backend == 'qnnpack':
            # TODO: 使其与 xnnpack 约束兼容
            # 对于后端为 'qnnpack'，使用特定参数创建量化配置
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=0,
                                                                                 quant_max=255,
                                                                                 reduce_range=False),
                              weight=default_fused_wt_fake_quant)
        elif backend == 'onednn':
            # 对于后端为 'onednn'，使用特定参数创建量化配置
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=0,
                                                                                 quant_max=255),
                              weight=default_fused_per_channel_wt_fake_quant)
        elif backend == 'x86':
            # 对于后端为 'x86'，使用特定参数创建量化配置
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=0,
                                                                                 quant_max=255,
                                                                                 reduce_range=True),
                              weight=default_fused_per_channel_wt_fake_quant)
        else:
            # 如果后端未知，则使用默认的量化配置
            qconfig = default_qat_qconfig_v2
    else:
        # 如果版本号不是 0 或 1，则抛出断言错误
        raise AssertionError("Version number: " + str(version) +
                             " in get_default_qat_qconfig is not supported. Version number must be 0 or 1")

    # 返回所选的量化配置
    return qconfig
"""
Default symmetric QAT qconfig for qnnpack. And its per channel weight variant.
"""
# 定义默认的对称 QAT QConfig 用于 qnnpack，以及其每通道权重变体。

default_symmetric_qnnpack_qat_qconfig = QConfig(
    activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                       quant_min=-128,
                                                       quant_max=127,
                                                       dtype=torch.qint8,
                                                       reduce_range=False,
                                                       eps=2 ** -12),
    weight=fused_wt_fake_quant_range_neg_127_to_127)

# 定义默认的对称 QAT QConfig 用于 qnnpack，包含每通道权重的变体。

default_per_channel_symmetric_qnnpack_qat_qconfig = QConfig(
    activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                       quant_min=-128,
                                                       quant_max=127,
                                                       dtype=torch.qint8,
                                                       reduce_range=False,
                                                       eps=2 ** -12),
    weight=fused_per_channel_wt_fake_quant_range_neg_127_to_127)

# 定义默认的 FP32 占位符 QConfig，用于未量化的 FP32 类型。

_default_fp32_placeholder_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=torch.float32),
    weight=PlaceholderObserver.with_args(dtype=torch.float32)
)

# 定义默认的 Quint8 占位符 QConfig，用于未量化的 Quint8 类型，这类操作不涉及权重。

_default_quint8_placeholder_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=torch.quint8),
    weight=None,
)

@deprecated(
    "`torch.ao.quantization.get_default_qconfig_dict` is deprecated and will be removed in "
    "a future version. Please use `torch.ao.quantization.get_default_qconfig_mapping` instead.",
    category=FutureWarning,
)
# 声明一个被废弃的函数装饰器，提示使用新的函数代替旧版本的函数。

def get_default_qconfig_dict(backend='x86', version=0):
    # 返回默认的量化配置字典，根据指定的后端和版本。
    return torch.ao.quantization.get_default_qconfig_mapping(backend, version).to_dict()

@deprecated(
    "`torch.ao.quantization.get_default_qat_qconfig_dict` is deprecated and will be removed in "
    "a future version. Please use `torch.ao.quantization.get_default_qat_qconfig_mapping` instead.",
    category=FutureWarning,
)
# 声明一个被废弃的函数装饰器，提示使用新的函数代替旧版本的函数。

def get_default_qat_qconfig_dict(backend='x86', version=1):
    # 返回默认的 QAT（Quantization Aware Training）量化配置字典，根据指定的后端和版本。
    return torch.ao.quantization.get_default_qat_qconfig_mapping(backend, version).to_dict()

def _assert_valid_qconfig(qconfig: Optional[QConfig],
                          mod: torch.nn.Module) -> None:
    """
    Verifies that this `qconfig` is valid.
    """
    # 验证给定的 `qconfig` 是否有效。
    if qconfig is None:
        return
    is_conv_transpose_mod = (
        isinstance(mod, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)))
    if is_conv_transpose_mod:
        # 检查是否为转置卷积模型，如果是，则执行以下操作
        if qconfig.weight is None:
            # 如果权重的量化配置为空，目前假设所有没有权重量化配置的转置卷积都是有效的
            return
        # 使用权重的量化配置创建一个示例观察器
        example_observer = qconfig.weight()
        # 检查示例观察器是否是按通道的观察器类型
        is_per_channel = (
            isinstance(example_observer, (torch.ao.quantization.PerChannelMinMaxObserver,
                                          torch.ao.quantization.MovingAveragePerChannelMinMaxObserver))
        )
        # 断言不支持按通道权重观察器，抛出异常信息
        assert not is_per_channel, \
            'Per channel weight observer is not supported yet for ConvTranspose{n}d.'
# 定义一个类型别名 QConfigAny，表示可选的 QConfig 类型
QConfigAny = Optional[QConfig]
# 将 QConfigAny 的 __module__ 属性设置为 "torch.ao.quantization.qconfig"
QConfigAny.__module__ = "torch.ao.quantization.qconfig"

# 定义一个辅助函数 _add_module_to_qconfig_obs_ctr，用于在量化准备过程中更新 qconfig，
# 以便构造函数在与模块 module 相同的设备上创建观察者
def _add_module_to_qconfig_obs_ctr(
        qconfig: QConfigAny,
        module: Optional[nn.Module]) -> Any:
    r"""This is a helper function for use in quantization prepare that updates a qconfig so that
    the constructors stored in the qconfig will create observers on the same device that
    'module' is on. This is intended to be used when the qconfigs are propagated to each
    module in order to avoid potential device alignment issues.

    Args:
        qconfig: QConfig with obs constructors stored in activation and weight
        module: module which the qconfig is related to

    Return:
        qconfig: configured so that obs constructors set to construct on the same device as module
    """

    # 如果 module 或 qconfig 为 None，或者 qconfig 不是 ('activation', 'weight') 字段的元组，
    # 则直接返回 qconfig
    if module is None or qconfig is None or qconfig._fields != ('activation', 'weight'):
        return qconfig

    # 定义一个函数，根据 module 的设备返回工厂参数
    def get_factory_kwargs_based_on_module_device():
        assert isinstance(module, torch.nn.Module)
        # 收集模块参数和缓冲区的设备
        devices = {p.device for p in module.parameters()} | \
            {p.device for p in module.buffers()}
        # 获取第一个设备，如果设备集合不为空
        device = next(iter(devices)) if len(devices) > 0 else None
        return None if device is None else {'device': device}

    # 定义一个函数，配置构造函数以在 module 设备上创建观察者
    def configure_constructor_to_put_obs_on_module_device(original_constructor):
        try:
            # 检查构造函数是否可以接受 factory_kwargs 参数
            check = original_constructor.with_args(factory_kwargs=None)
            check()
            # 返回使用 module 设备的可调用参数配置的构造函数
            return original_constructor.with_callable_args(factory_kwargs=get_factory_kwargs_based_on_module_device)
        except AttributeError:  # qconfig 没有 activation 或 weight
            return original_constructor
        except TypeError:  # 类不接受 factory_kwargs 参数
            return original_constructor

    # 配置 activation 和 weight 的构造函数，使其在 module 设备上创建观察者
    activation = configure_constructor_to_put_obs_on_module_device(qconfig.activation)
    weight = configure_constructor_to_put_obs_on_module_device(qconfig.weight)

    # 返回更新后的 QConfig 对象，其中 activation 和 weight 已经配置完成
    return QConfig(activation, weight)

# 定义一个类型别名 _ObserverOrFakeQuantizeConstructor，表示 Observer 或 FakeQuantize 构造函数的联合类型
_ObserverOrFakeQuantizeConstructor = Union[_PartialWrapper, Type[ObserverBase], Type[FakeQuantizeBase]]

# 定义函数 _obs_or_fq_ctr_equals，用于比较两个 Observer 或 FakeQuantize 构造函数是否相等
def _obs_or_fq_ctr_equals(obs_or_fq1: _ObserverOrFakeQuantizeConstructor, obs_or_fq2: _ObserverOrFakeQuantizeConstructor):
    # 如果两个对象都是 _PartialWrapper 类型，则调用 _partial_wrapper_equals 函数进行比较
    if isinstance(obs_or_fq1, _PartialWrapper) and isinstance(obs_or_fq2, _PartialWrapper):
        return _partial_wrapper_equals(obs_or_fq1, obs_or_fq2)
    # 否则直接比较两个对象是否相等
    return obs_or_fq1 == obs_or_fq2

# 定义函数 _partial_wrapper_equals，用于比较两个 partial wrapper 是否相等
def _partial_wrapper_equals(obs_or_fq1: _PartialWrapper, obs_or_fq2: _PartialWrapper):
    """
    Return whether the two partial wrappers are equal,
    """
    # 复制 partial wrapper 的关键字参数
    obs_or_fq1_keywords = copy.copy(obs_or_fq1.p.keywords)
    obs_or_fq2_keywords = copy.copy(obs_or_fq2.p.keywords)
    # 比较两个关键字参数是否相等
    keywords_equal = True
    # 使用 _obs_or_fq_ctr_equals 比较观察者构造函数，因为直接比较会失败
    # 如果 "observer" 关键词同时出现在 obs_or_fq1_keywords 和 obs_or_fq2_keywords 中
    if "observer" in obs_or_fq1_keywords and "observer" in obs_or_fq2_keywords:
        # 调用 _obs_or_fq_ctr_equals 函数比较两个关键词对应的值是否相等，并更新 keywords_equal
        keywords_equal = keywords_equal and _obs_or_fq_ctr_equals(obs_or_fq1_keywords["observer"], obs_or_fq2_keywords["observer"])
        # 从 obs_or_fq1_keywords 和 obs_or_fq2_keywords 中移除 "observer" 关键词及其对应的值
        obs_or_fq1_keywords.pop("observer")
        obs_or_fq2_keywords.pop("observer")
    
    # 更新 keywords_equal，检查剩余的关键词是否相等
    keywords_equal = keywords_equal and obs_or_fq1_keywords == obs_or_fq2_keywords
    
    # 返回观察器或频率计数器对象的属性 p 的函数和参数是否相等，以及关键词是否相等的结果
    return obs_or_fq1.p.func == obs_or_fq2.p.func and obs_or_fq1.p.args == obs_or_fq2.p.args and keywords_equal
def qconfig_equals(q1: QConfigAny, q2: QConfigAny):
    """
    Returns `True` if `q1` equals `q2`, and `False` otherwise.
    """
    # 如果 q1 或 q2 任意一个为 None，则直接比较它们是否相等，返回比较结果
    if q1 is None or q2 is None:
        return q1 == q2
    else:
        # 确保 q1 和 q2 都不为 None
        assert q1 is not None and q2 is not None
        try:
            # 比较 q1 和 q2 的激活函数是否相同，可能涉及到部分包装器的特殊处理
            activation_same = _obs_or_fq_ctr_equals(q1.activation, q2.activation)
            # 比较 q1 和 q2 的权重是否相同，可能涉及到部分包装器的特殊处理
            weight_same = _obs_or_fq_ctr_equals(q1.weight, q2.weight)
            # 返回激活函数和权重是否均相同的比较结果
            return activation_same and weight_same
        except AttributeError:
            # 如果无法比较，则直接比较 q1 和 q2 是否相等
            return q1 == q2

def _activation_is_memoryless(qconfig: QConfig):
    """
    Return whether the observer for activations defined in the given QConfig is memoryless.
    This means a MovingAverage observer with averaging constant equal to 1.
    """
    def _is_memoryless(observer):
        # 判断观察器是否是无记忆的，即移动平均观察器且平均常数为 1
        return hasattr(observer, "averaging_constant") and observer.averaging_constant == 1
    
    # 获得激活函数的观察器对象
    act = qconfig.activation()
    if isinstance(act, FakeQuantizeBase) and hasattr(act, "activation_post_process"):
        # 如果激活函数是 FakeQuantizeBase 类型并且具有 activation_post_process 属性
        # 则判断其 activation_post_process 属性是否是无记忆的观察器
        return _is_memoryless(act.activation_post_process)
    else:
        # 如果不满足上述条件，则直接判断激活函数的观察器是否是无记忆的
        return _is_memoryless(act)

def _is_reuse_input_qconfig(qconfig: Optional[QConfig]):
    # 判断是否为重用输入的 QConfig
    return qconfig is not None and \
        isinstance(qconfig.activation(), ReuseInputObserver) and \
        isinstance(qconfig.weight(), NoopObserver)
```