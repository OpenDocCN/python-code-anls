# `.\pytorch\torch\ao\quantization\quantizer\xnnpack_quantizer.py`

```
# mypy: allow-untyped-defs
from __future__ import annotations

import copy  # 导入复制模块用于深拷贝对象
import functools  # 导入函数工具模块提供高阶函数支持

from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING  # 导入类型提示模块

import torch  # 导入PyTorch深度学习框架
import torch._dynamo as torchdynamo  # 导入torchdynamo用于Dynamo图导出
import torch.nn.functional as F  # 导入PyTorch的函数库F

from torch.ao.quantization.fake_quantize import (  # 从fake_quantize模块导入伪量化相关类
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torch.ao.quantization.observer import (  # 从observer模块导入观察器类
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)

from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer  # 从quantizer模块导入量化规格和量化器
from torch.ao.quantization.quantizer.utils import _get_module_name_filter  # 导入获取模块名过滤器的工具函数

from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (  # 从xnnpack_quantizer_utils模块导入XNNPACK量化器相关工具
    _convert_scalars_to_attrs,
    OP_TO_ANNOTATOR,
    OperatorConfig,
    OperatorPatternType,
    propagate_annotation,
    QuantizationConfig,
)

if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor  # 类型检查：导入观察器或伪量化构造函数类型
    from torch.fx import Node  # 类型检查：导入FX库中的节点类型

__all__ = [
    "XNNPACKQuantizer",
    "get_symmetric_quantization_config",
]

# 定义私有方法：根据输入函数和参数，返回torch.fx.Graph对象
def _get_dynamo_graph(function: Callable, inputs) -> torch.fx.Graph:
    gm, _ = torchdynamo.export(function, aten_graph=True)(*inputs)
    gm.graph.eliminate_dead_code()  # 优化图结构，删除死代码
    return gm.graph  # 返回导出的Dynamo图对象

# 定义私有方法：根据输入尺寸，返回线性模式的动态图列表
def _get_linear_patterns(input_size: List[int]):
    in_channels = input_size[-1]
    out_channels = 8  # 硬编码输出通道数
    weight = torch.ones((out_channels, in_channels))  # 创建全为1的权重张量
    bias = torch.ones((out_channels,))  # 创建全为1的偏置张量
    act = torch.ones(input_size)  # 创建全为1的激活张量

    # 定义线性操作函数
    def linear_op(act, weight, bias=None):
        return F.linear(act, weight, bias)

    # 获取带偏置和不带偏置的线性操作动态图
    pattern_w_bias = _get_dynamo_graph(linear_op, (act, weight, bias))
    pattern_wo_bias = _get_dynamo_graph(linear_op, (act, weight))
    return [pattern_w_bias, pattern_wo_bias]  # 返回动态图列表

# 定义私有方法：返回支持的对称量化操作符字典
def _supported_symmetric_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    supported_operators: Dict[str, List[OperatorPatternType]] = {
        # 支持的操作符列表，包括卷积、线性、加法、最大池化、自适应平均池化
        "conv2d": [
            [torch.nn.Conv2d, torch.nn.ReLU],
            [torch.nn.Conv2d, F.relu],
            [F.conv2d, torch.nn.ReLU],
            [F.conv2d, F.relu],
        ],
        "linear": [[torch.nn.Linear], [F.linear]],
        "add": [[torch.add]],
        "max_pool2d": [[torch.nn.MaxPool2d], [F.max_pool2d]],
        "adaptive_avg_pool2d": [
            [torch.nn.AdaptiveAvgPool2d],
            [F.adaptive_avg_pool2d],
        ],
    }
    return copy.deepcopy(supported_operators)  # 深拷贝并返回支持的操作符字典

# 定义私有方法：返回支持的对称配置和操作符列表
def _get_supported_symmetric_config_and_operators() -> List[OperatorConfig]:
    supported_config_and_operators: List[OperatorConfig] = []  # 初始化支持的配置和操作符列表
    # 遍历四种量化配置的列表，每种配置调用不同的函数生成
    for quantization_config in [
        get_symmetric_quantization_config(),                                      # 调用函数获取对称量化配置
        get_symmetric_quantization_config(is_qat=True),                           # 调用函数获取带量化感知训练参数的对称量化配置
        get_symmetric_quantization_config(is_per_channel=True),                   # 调用函数获取每通道对称量化配置
        get_symmetric_quantization_config(is_per_channel=True, is_qat=True),     # 调用函数获取带量化感知训练参数和每通道对称量化配置
    ]:
        # 获取支持的对称量化操作符列表
        ops = _supported_symmetric_quantized_operators()
        # 遍历操作符模式列表，并将每个模式列表与当前量化配置关联
        for pattern_list in ops.values():
            supported_config_and_operators.append(
                OperatorConfig(quantization_config, pattern_list)  # 创建 OperatorConfig 对象，将量化配置和操作符模式列表传入
            )
    # 返回支持的量化配置和操作符的深层拷贝
    return copy.deepcopy(supported_config_and_operators)
# 使用 functools 模块中的 lru_cache 装饰器，将函数包装成带有 LRU 缓存的版本，用于优化函数调用性能
@functools.lru_cache
# 定义一个函数，返回对称量化配置的对象
def get_symmetric_quantization_config(
    is_per_channel: bool = False,  # 是否为每通道量化，默认为 False
    is_qat: bool = False,          # 是否为量化感知训练，默认为 False
    is_dynamic: bool = False,      # 是否为动态量化，默认为 False
    act_qmin: int = -128,          # 激活量化的最小值，默认为 -128
    act_qmax: int = 127,           # 激活量化的最大值，默认为 127
    weight_qmin: int = -127,       # 权重量化的最小值，默认为 -127
    weight_qmax: int = 127,        # 权重量化的最大值，默认为 127
):
    # 额外的参数字典，包含一个名为 "eps" 的键，值为 2 的负 12 次方
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    if is_qat:
        if is_dynamic:
            # 如果是量化感知训练且是动态量化，则使用 FakeQuantize 类
            act_observer_or_fake_quant_ctr = FakeQuantize
            # 动态量化的观察器使用 MovingAverageMinMaxObserver 类，并设置参数 averaging_constant 为 1
            dynamic_quant_observer = MovingAverageMinMaxObserver.with_args(
                averaging_constant=1
            )
            # 将动态量化观察器设置为额外参数字典中的值
            extra_args["observer"] = dynamic_quant_observer
        else:
            # 如果是量化感知训练但不是动态量化，则使用 FusedMovingAvgObsFakeQuantize 类
            act_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize  # type: ignore[assignment]
    else:
        if is_dynamic:
            # 如果不是量化感知训练但是动态量化，则使用 PlaceholderObserver 类
            act_observer_or_fake_quant_ctr = PlaceholderObserver  # type: ignore[assignment]
        else:
            # 否则使用 HistogramObserver 类
            act_observer_or_fake_quant_ctr = HistogramObserver  # type: ignore[assignment]

    # 定义激活量化的规范对象，使用 QuantizationSpec 类进行配置
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,                    # 数据类型为 int8
        quant_min=act_qmin,                  # 量化的最小值为 act_qmin
        quant_max=act_qmax,                  # 量化的最大值为 act_qmax
        qscheme=torch.per_tensor_affine,     # 量化方案为每张量张量仿射量化
        is_dynamic=is_dynamic,               # 是否为动态量化
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            **extra_args,                    # 使用额外的参数字典作为参数
        ),
    )

    # 定义权重量化方案，根据是否每通道量化选择不同的 qscheme
    weight_qscheme = (
        torch.per_channel_symmetric if is_per_channel else torch.per_tensor_symmetric
    )
    # 权重观察器或伪量化构造器默认为 MinMaxObserver 类
    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        MinMaxObserver
    )
    if is_qat:
        # 如果是量化感知训练，则根据权重量化方案选择不同的权重观察器或伪量化构造器
        # TODO: qat + per channel?
        weight_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize
    elif is_per_channel:
        # 如果是每通道量化，则使用 PerChannelMinMaxObserver 类
        weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver

    # 再次定义额外的参数字典，包含一个名为 "eps" 的键，值为 2 的负 12 次方
    extra_args: Dict[str, Any] = {"eps": 2**-12}
    if is_qat:
        if weight_qscheme == torch.per_tensor_symmetric:
            # 如果是量化感知训练且权重量化方案为每张张量仿射量化，则观察器使用 MovingAverageMinMaxObserver 类
            extra_args["observer"] = MovingAverageMinMaxObserver
        else:
            # 否则观察器使用 MovingAveragePerChannelMinMaxObserver 类
            extra_args["observer"] = MovingAveragePerChannelMinMaxObserver  # type: ignore[dict-item]
    # 定义权重量化的规范对象，使用 QuantizationSpec 类进行配置
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,                    # 数据类型为 int8
        quant_min=weight_qmin,               # 量化的最小值为 weight_qmin
        quant_max=weight_qmax,               # 量化的最大值为 weight_qmax
        qscheme=weight_qscheme,              # 量化方案为权重量化方案
        ch_axis=0,                           # 通道轴为 0
        is_dynamic=False,                    # 不使用动态量化
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
            **extra_args                     # 使用额外的参数字典作为参数
        ),
    )

    # 偏置量化规范对象初始化为 None
    bias_quantization_spec = None
    if is_dynamic:
        # 如果是动态量化，则创建 QuantizationConfig 对象，包含激活量化、权重量化、偏置量化配置
        quantization_config = QuantizationConfig(
            act_quantization_spec,           # 激活量化规范对象
            None,                            # 空的偏置量化规范对象
            weight_quantization_spec,        # 权重量化规范对象
            bias_quantization_spec,          # 空的偏置量化规范对象
            is_qat,                          # 是否为量化感知训练
        )
    else:
        # 否则创建 QuantizationConfig 对象，包含激活量化、权重量化、偏置量化配置
        quantization_config = QuantizationConfig(
            act_quantization_spec,           # 激活量化规范对象
            act_quantization_spec,           # 激活量化规范对象（与权重量化规范对象相同）
            weight_quantization_spec,        # 权重量化规范对象
            bias_quantization_spec,          # 空的偏置量化规范对象
            is_qat,                          # 是否为量化感知训练
        )
    # 返回量化配置对象
    return quantization_config


# 定义一个私有函数，返回支持的对称配置和运算符的列表
def _get_supported_config_and_operators() -> List[OperatorConfig]:
    return _get_supported_symmetric_config_and_operators()
# 获取给定模块类型的模块类型过滤器函数，该过滤器接受一个节点并检查节点是否来自具有特定模块类型的模块

def _get_module_type_filter(tp: Callable):
    """Get the module_type_filter function for a given module type, the filter accepts
    a node and checks if the node comes from a module that has certain module type
    
    For example:
        node: linear_op = call_function[...](...)  # comes from a module with type Block -> Sub -> Linear
    
    >> module_type_filter = _get_module_type_filter(Sub)  # submodule with type `Sub`, under the `Block` submodule
    >> print(module_type_filter(node))
    True  # the node is from the submodule `Sub` (same for `Block` and `Linear` as well)
    """
    
    tp_str = tp.__module__ + "." + tp.__qualname__  # 构建模块类型的字符串表示，格式为 模块所在模块名.模块全限定名

    def module_type_filter(n: Node) -> bool:
        # 从节点的元数据中获取神经网络模块堆栈信息
        nn_module_stack = n.meta.get("nn_module_stack", {})
        types = []
        # 遍历神经网络模块堆栈中的每一个条目，提取模块类型的字符串表示
        for _, t in nn_module_stack.values():
            # export() 返回 str，但是旧版 API（如 capture_pre_autograd_graph）返回类型。处理两种情况。
            if isinstance(t, type):
                t = t.__module__ + "." + t.__qualname__
            types.append(t)
        # 判断模块类型的字符串表示是否在预期的模块类型字符串列表中
        return tp_str in types

    return module_type_filter

# 获取不属于给定模块类型或模块名称列表的模块过滤器函数

def _get_not_module_type_or_name_filter(
    tp_list: List[Callable], module_name_list: List[str]
) -> Callable[[Node], bool]:
    # 为给定的每个模块类型创建模块类型过滤器函数
    module_type_filters = [_get_module_type_filter(tp) for tp in tp_list]
    # 为给定的每个模块名称创建模块名称过滤器函数
    module_name_list_filters = [_get_module_name_filter(m) for m in module_name_list]

    def not_module_type_or_name_filter(n: Node) -> bool:
        # 如果节点不属于任何一个模块类型过滤器或模块名称过滤器返回的结果为真，则返回真
        return not any(f(n) for f in module_type_filters + module_name_list_filters)

    return not_module_type_or_name_filter

class XNNPACKQuantizer(Quantizer):
    # 支持的配置和运算符列表
    supported_config_and_operators = _get_supported_config_and_operators()
    
    # 仅静态量化训练量化自动辨别系统（QAT）操作列表
    STATIC_QAT_ONLY_OPS = [
        "conv_bn_relu",
        "conv_bn",
        "conv_transpose_bn_relu",
        "conv_transpose_bn",
    ]
    
    # 静态量化操作（包括 PTQ 和 QAT）
    # 保持融合操作在单个操作之前的顺序
    STATIC_OPS = [
        "linear_relu",
        "linear",
        "conv_relu",
        "conv",
        "conv_transpose_relu",
        "adaptive_avg_pool2d",
        # TODO: move this to BoltNNQuantizer?
        "gru_io_only",
        "max_pool2d",
        "add_relu",
        "add",
        "mul_relu",
        "mul",
        "cat",
    ]
    
    # 动态量化操作列表
    DYNAMIC_OPS = [
        "linear",
    ]

    def __init__(self):
        super().__init__()
        # 全局配置（可选的量化配置）
        self.global_config: Optional[QuantizationConfig] = None
        # 操作符类型配置字典，映射到相应的量化配置
        self.operator_type_config: Dict[
            torch._ops.OpOverloadPacket, Optional[QuantizationConfig]
        ] = {}
        # 模块类型配置字典，映射到相应的量化配置
        self.module_type_config: Dict[Callable, Optional[QuantizationConfig]] = {}
        # 模块名称配置字典，映射到相应的量化配置
        self.module_name_config: Dict[str, Optional[QuantizationConfig]] = {}

    @classmethod
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        # 从类的支持配置和操作符中提取所有唯一的量化配置
        op_configs: Set[QuantizationConfig] = {
            spec for spec, _ in cls.supported_config_and_operators
        }
        return list(op_configs)

    @classmethod
    def get_supported_operator_for_quantization_config(
        cls, quantization_config: Optional[QuantizationConfig]
    ) -> List[OperatorPatternType]:
        if quantization_config is None:
            # 如果没有指定量化配置，返回所有支持的操作符列表
            all_ops = []
            for _, ops in cls.supported_config_and_operators:
                all_ops.extend(ops)
            return all_ops

        for config, ops in cls.supported_config_and_operators:
            # 遍历支持的配置和操作符对，找到与给定量化配置匹配的操作符列表
            # 注意：这里假设每个条目在 cls.supported_config_and_operators 中对应一个配置
            # 例如，我们没有 [(spec1, op_list1), (spec1, op_list2), (spec2, op_list3)] 这种情况
            # 其中第一个和第二个条目具有相同的 spec 但未合并 op 列表
            if config == quantization_config:
                return ops
        return []

    def set_global(self, quantization_config: QuantizationConfig) -> XNNPACKQuantizer:
        # 设置全局量化配置并返回当前对象
        self.global_config = quantization_config
        return self

    def set_operator_type(
        self,
        operator_type: torch._ops.OpOverloadPacket,
        quantization_config: QuantizationConfig,
    ) -> XNNPACKQuantizer:
        # 设置特定操作符类型的量化配置并返回当前对象
        self.operator_type_config[operator_type] = quantization_config
        return self

    def set_module_type(
        self, module_type: Callable, quantization_config: QuantizationConfig
    ):
        """为具有特定模块类型的子模块设置量化配置，例如：
        quantizer.set_module_name(Sub) 或 quantizer.set_module_name(nn.Linear)，
        它将使用给定的 `quantization_config` 量化该模块类型中的所有支持的操作符/操作符模式
        """
        self.module_type_config[module_type] = quantization_config
        return self

    def set_module_name(
        self, module_name: str, quantization_config: Optional[QuantizationConfig]
    ):
        """为具有特定模块名称的子模块设置量化配置，例如：
        quantizer.set_module_name("blocks.sub")，它将使用给定的 `quantization_config` 量化该模块名称中的所有支持的操作符/操作符模式
        """
        assert (
            quantization_config is not None
        ), " quantization_config == None is not supported yet"
        self.module_name_config[module_name] = quantization_config
        return self

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """将标量值转换为张量属性"""
        return _convert_scalars_to_attrs(model)
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """处理全局规格的函数，目前仅处理全局配置"""
        # 如果存在全局配置且输入激活为动态类型
        if self.global_config and self.global_config.input_activation.is_dynamic:  # type: ignore[union-attr]
            # 对动态量化配置进行标注处理
            model = self._annotate_for_dynamic_quantization_config(model)
        else:
            # 对静态量化配置进行标注处理
            model = self._annotate_for_static_quantization_config(model)
        # 传播标注结果
        propagate_annotation(model)
        # 返回标注后的模型
        return model

    def _annotate_all_static_patterns(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[Callable[[Node], bool]] = None,
    ) -> torch.fx.GraphModule:
        # TODO: 实现支持 None 取消先前标注的注释
        if quantization_config is None:
            return model

        # 如果是量化训练，处理仅支持静态量化的操作
        if quantization_config.is_qat:
            for op in self.STATIC_QAT_ONLY_OPS:
                # 使用特定的标注器处理静态量化 QAT 操作
                OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
        # 处理其他静态操作的标注
        for op in self.STATIC_OPS:
            OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
        # 返回标注后的模型
        return model

    def _annotate_all_dynamic_patterns(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[Callable[[Node], bool]] = None,
    ) -> torch.fx.GraphModule:
        # TODO: 实现支持 None 取消先前标注的注释
        if quantization_config is None:
            return model

        # 处理所有动态操作的标注
        for op in self.DYNAMIC_OPS:
            OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
        # 返回标注后的模型
        return model

    def _annotate_for_static_quantization_config(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        # 获取模块名称配置列表
        module_name_list = list(self.module_name_config.keys())
        # 对每个模块名称及其配置进行静态标注处理
        for module_name, config in self.module_name_config.items():
            self._annotate_all_static_patterns(
                model, config, _get_module_name_filter(module_name)
            )

        # 获取模块类型配置列表
        tp_list = list(self.module_type_config.keys())
        # 对每个模块类型及其配置进行静态标注处理
        for module_type, config in self.module_type_config.items():
            self._annotate_all_static_patterns(
                model, config, _get_module_type_filter(module_type)
            )

        # 对非模块类型或名称的配置进行静态标注处理
        self._annotate_all_static_patterns(
            model,
            self.global_config,
            _get_not_module_type_or_name_filter(tp_list, module_name_list),
        )
        # 返回标注后的模型
        return model

    def _annotate_for_dynamic_quantization_config(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        # 在动态量化配置下，对模型进行标注处理
        for op in self.DYNAMIC_OPS:
            OP_TO_ANNOTATOR[op](model, self.global_config, None)
        # 返回标注后的模型
        return model
    ) -> torch.fx.GraphModule:
        # 获取模块名列表
        module_name_list = list(self.module_name_config.keys())
        # 遍历每个模块名及其配置，对模型进行动态模式标注
        for module_name, config in self.module_name_config.items():
            self._annotate_all_dynamic_patterns(
                model, config, _get_module_name_filter(module_name)
            )

        # 获取模块类型列表
        tp_list = list(self.module_type_config.keys())
        # 遍历每个模块类型及其配置，对模型进行动态模式标注
        for module_type, config in self.module_type_config.items():
            self._annotate_all_dynamic_patterns(
                model, config, _get_module_type_filter(module_type)
            )

        # 对于未匹配到的模块类型或模块名，使用全局配置进行动态模式标注
        self._annotate_all_dynamic_patterns(
            model,
            self.global_config,
            _get_not_module_type_or_name_filter(tp_list, module_name_list),
        )
        # 返回标注后的模型
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        # 空函数，用于验证模型，无返回值
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        # 返回支持的运算符配置列表
        return cls.supported_config_and_operators
```