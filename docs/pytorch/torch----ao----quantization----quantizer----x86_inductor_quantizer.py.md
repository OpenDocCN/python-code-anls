# `.\pytorch\torch\ao\quantization\quantizer\x86_inductor_quantizer.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和函数
import copy  # 复制对象的模块
import functools  # 高阶函数工具库
import itertools  # 创建迭代器的函数工具库
import operator  # 提供一组对应Python内置操作的函数
import warnings  # 控制警告的模块
from dataclasses import dataclass  # 用于创建数据类的装饰器
from typing import (  # 引入类型提示相关的模块和工具
    Any,  # 任意类型
    Callable,  # 可调用对象类型
    Dict,  # 字典类型
    List,  # 列表类型
    Optional,  # 可选类型
    Sequence,  # 序列类型
    Set,  # 集合类型
    Tuple,  # 元组类型
    TYPE_CHECKING,  # 类型检查标记
    Union,  # 联合类型
)

from typing_extensions import TypeAlias  # 类型别名的扩展库

import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch中的函数工具库
from torch.ao.quantization.fake_quantize import (  # PyTorch AO模块中的量化仿真工具
    FakeQuantize,  # 伪量化
    FusedMovingAvgObsFakeQuantize,  # 融合移动平均的伪量化
)
from torch.ao.quantization.observer import (  # PyTorch AO模块中的观察者工具
    HistogramObserver,  # 直方图观察者
    MovingAverageMinMaxObserver,  # 移动平均最小最大值观察者
    MovingAveragePerChannelMinMaxObserver,  # 按通道移动平均最小最大值观察者
    PerChannelMinMaxObserver,  # 按通道最小最大值观察者
    PlaceholderObserver,  # 占位符观察者
)
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions  # 找到顺序分区的工具函数
from torch.ao.quantization.quantizer.quantizer import (  # PyTorch AO模块中的量化器工具
    QuantizationAnnotation,  # 量化注释
    QuantizationSpec,  # 量化规格
    Quantizer,  # 量化器
    SharedQuantizationSpec,  # 共享量化规格
)

from torch.ao.quantization.quantizer.utils import _get_module_name_filter  # 获取模块名称过滤器的工具函数
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (  # PyTorch AO模块中XNNPACK量化器工具
    get_bias_qspec,  # 获取偏置量化规格的函数
    get_input_act_qspec,  # 获取输入激活量化规格的函数
    get_output_act_qspec,  # 获取输出激活量化规格的函数
    get_weight_qspec,  # 获取权重量化规格的函数
    OperatorConfig,  # 运算符配置
    OperatorPatternType,  # 运算符模式类型
    QuantizationConfig,  # 量化配置
)
from torch.fx import Node  # PyTorch FX模块中的节点
from torch.fx.passes.utils.source_matcher_utils import (  # PyTorch FX模块中源匹配工具的函数
    get_source_partitions,  # 获取源分区的函数
    SourcePartition,  # 源分区类型
)

FilterFn: TypeAlias = Callable[[List[Node]], bool]  # 类型别名，表示过滤函数的类型

if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor  # 如果进行类型检查，引入相关的观察者或伪量化构造器

__all__ = [  # 模块公开的接口列表
    "X86InductorQuantizer",  # X86感应量化器
    "get_default_x86_inductor_quantization_config",  # 获取X86感应量化配置的函数
]


@dataclass  # 数据类装饰器，用于定义数据类
class _X86InductorQuantizationAnnotation(QuantizationAnnotation):
    # _is_output_of_quantized_pattern:
    #  * Node as output node of a fusion pattern.
    #  * The fusion pattern supports int8 data type.
    #  * The fusion pattern has inputs annotated to insert observer.
    #  * The quantization_config is not `None`.
    _is_output_of_quantized_pattern: bool = False  # 是否为量化模式输出的注释


# Operators that:
# 1. Operators are optimized to run with int8 when int8 input provided.
# 2. Operators do not support int8 input and produce fp32 output.
int8_in_int8_out_ops: Set = {
    torch.ops.aten.max_pool2d.default,  # 最大池化操作
    torch.ops.aten.cat.default,  # 连接操作
    torch.ops.aten.avg_pool2d.default,  # 平均池化操作
    torch.ops.aten.adaptive_avg_pool2d.default,  # 自适应平均池化操作
    torch.ops.aten.flatten.using_ints,  # 使用整数展平操作
}

# Operators that support the int8 data type for quantization config propagation.
# A superset of int8_in_int8_out_ops incorporating additional operators.
propagation_quantizable_ops = int8_in_int8_out_ops  # 支持量化配置传播的操作集合

# Operators support the int8 data type
# and recipe is configured by default in X86InductorQuantizer.
default_quantizable_ops = propagation_quantizable_ops | {
    torch.ops.aten.conv2d.default,  # 二维卷积操作
    torch.ops.aten.linear.default,  # 线性操作
}

# A superset of default_quantizable_ops includes operators support the int8 data type
# but not enabled by default recipe of X86InductorQuantizer.
# 将默认可量化操作与 torch.ops.aten.matmul.default 结合为一个集合
quantizable_ops = default_quantizable_ops | {
    torch.ops.aten.matmul.default,
}

# 量化注解的关键字
QUANT_ANNOTATION_KEY = "quantization_annotation"


def _skip_annotate(nodes: List[Node], filter_fn: Optional[FilterFn] = None) -> bool:
    """决定是否跳过对一组节点进行注解。"""

    # 1) 如果任何节点已经被注解，则跳过注解
    if _is_any_annotated(nodes):
        return True

    # 2) 如果 a) 提供了过滤函数
    # 并且 b) 给定的节点列表通过了过滤函数的检查，则进行注解
    if filter_fn and filter_fn(nodes):
        return False

    # 否则跳过注解
    return True


def _create_module_name_filter(module_name: str) -> FilterFn:
    """为给定的模块名称创建一个过滤函数。

    这个过滤函数接受一个节点列表（由 annotate 函数确定），如果所有节点都来自指定的模块名称，则返回 True，否则返回 False。

    例如：
        linear_1: "f32[3, 10]" = torch.ops.aten.linear.default(...) # 来自名称为 `sub.linear1` 的模块
        relu: "f32[3, 10]" = torch.ops.aten.relu.default(linear_1); # 来自名称为 `sub.relu1` 的模块

    >> module_name_filter = _create_module_name_filter_inner("sub")
    >> print(module_name_filter([relu, linear_1]))
    # True  # 这两个节点由 `_annotate_linear_unary` 函数确定，来自 "sub" 模块。
    """

    # 获取用于检查模块名称的过滤函数
    filter_fn = _get_module_name_filter(module_name)

    def check_all_nodes_from_module(nodes: List[Node]) -> bool:
        # 检查所有节点是否来自指定的模块名称
        all_nodes_from_module_name: bool = all(filter_fn(n) for n in nodes)
        return all_nodes_from_module_name

    return check_all_nodes_from_module


def _create_operator_type_filter(
    operator_type: Callable,
) -> FilterFn:
    """为给定的操作类型创建一个过滤函数。

    这个过滤函数接受一个节点列表，并返回 True 如果它包含一个具有指定操作类型的节点，否则返回 False。

    例如：
        linear_1: "f32[3, 10]" = torch.ops.aten.linear.default(...) # 来自名称为 `sub.linear1` 的模块
        relu: "f32[3, 10]" = torch.ops.aten.relu.default(linear_1); # 来自名称为 `sub.relu1` 的模块

    >> operator_type_filter = _create_operator_type_filter(torch.ops.aten.linear.default)
    >> print(operator_type_filter([relu, linear_1]))
    # True  # 这两个节点由 `_annotate_linear_unary` 函数确定，第二个节点是 `linear`。
    """

    def operator_type_filter(nodes: List[Node]):
        # 计算具有指定操作类型的节点数量
        num_nodes_with_operator_type = sum(
            node.target == operator_type for node in nodes
        )
        if num_nodes_with_operator_type > 1:
            raise NotImplementedError(
                f"Several nodes within a single pattern are {operator_type}."
            )
        return num_nodes_with_operator_type == 1

    return operator_type_filter


def _global_config_filter(nodes: List[Node]) -> bool:
    """全局配置的过滤函数。"""
    # 这是一个为全局配置定义的过滤函数，具体实现可以根据需要添加。
    # 这个过滤函数接受一个节点列表，并在列表中恰好有一个节点是默认可量化操作时返回True，否则返回False。
    """
    # 统计节点列表中是默认可量化操作的节点数量
    num_nodes_in_default_quantizable_ops = sum(
        node.target in default_quantizable_ops for node in nodes
    )
    # 如果默认可量化操作的节点数量大于1，则抛出NotImplementedError异常
    if num_nodes_in_default_quantizable_ops > 1:
        raise NotImplementedError(
            "Several nodes within a single pattern are default quantizable operations."
        )
    # 返回是否存在一个默认可量化操作的节点
    return num_nodes_in_default_quantizable_ops == 1
def _map_module_function_to_aten_operator_type():
    # 创建一个空的字典，用于映射模块函数到 ATen 运算符类型
    module_function_to_aten_operator: Dict[Callable, torch._ops.OpOverloadPacket] = {}
    # 定义要映射的函数列表及其对应的默认 ATen 运算符
    map_list = (
        ([torch.nn.Conv2d, F.conv2d], torch.ops.aten.conv2d.default),
        ([torch.nn.Linear, F.linear], torch.ops.aten.linear.default),
        ([torch.nn.MaxPool2d, F.max_pool2d], torch.ops.aten.max_pool2d.default),
        (
            [
                torch.cat,
            ],
            torch.ops.aten.cat.default,
        ),
        ([torch.nn.AvgPool2d, F.avg_pool2d], torch.ops.aten.avg_pool2d.default),
        (
            [torch.nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d],
            torch.ops.aten.adaptive_avg_pool2d.default,
        ),
        (
            [
                torch.flatten,
            ],
            torch.ops.aten.flatten.using_ints,
        ),
        (
            [
                torch.matmul,
            ],
            torch.ops.aten.matmul.default,
        ),
    )
    # 遍历映射列表，更新映射关系到字典中
    for map_item in map_list:
        module_function_to_aten_operator.update(dict.fromkeys(map_item[0], map_item[1]))  # type: ignore[call-overload]
    # 返回完成映射的字典
    return module_function_to_aten_operator


def _mark_nodes_as_annotated(nodes: List[Node]):
    # 遍历节点列表，为未注释的节点添加量化注释标记
    for node in nodes:
        if node is not None:
            if QUANT_ANNOTATION_KEY not in node.meta:
                node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation()
            node.meta[QUANT_ANNOTATION_KEY]._annotated = True


def _is_node_annotated(_node):
    """
    如果节点已经被注释，则返回 True，否则返回 False
    """
    return (
        QUANT_ANNOTATION_KEY in _node.meta
        and _node.meta[QUANT_ANNOTATION_KEY]._annotated
    )


def _is_any_annotated(nodes: List[Node]):
    """
    给定一个节点列表（表示一个操作模式），检查是否有任何节点被注释，
    如果有任何一个节点被注释，则返回 True，否则返回 False
    """
    return any(_is_node_annotated(node) for node in nodes)


def _is_all_annotated(nodes: List[Node]):
    """
    给定一个节点列表（表示一个操作模式），如果所有节点都被注释，则返回 True，否则返回 False
    """
    return all(_is_node_annotated(node) for node in nodes)


def _is_quantized_op_pt2e(node: torch.fx.Node):
    """
    用于 pt2e 流程，检查节点是否是量化节点：
    Case1: 节点已经被注释为融合模式的输出节点。
    Case2: 节点已经被注释为单个量化节点。
    """
    if not _is_any_annotated([node]):
        # 节点未被注释，直接返回 False
        return False
    quantization_annotation = node.meta.get(QUANT_ANNOTATION_KEY, None)
    assert isinstance(quantization_annotation, _X86InductorQuantizationAnnotation)
    return quantization_annotation._is_output_of_quantized_pattern


def _supported_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    # 这个函数目前没有实现，但是声明了将来会返回一个特定类型的字典
    # 定义一个字典，用于存储支持的操作符列表，键为操作名，值为操作符模式类型的列表
    supported_operators: Dict[str, List[OperatorPatternType]] = {
        "conv2d": [
            [torch.nn.Conv2d],
            [F.conv2d],
        ],
    }

    # 生成所有可能的 Conv 可选项(Add) 可选项(ReLU)
    conv_add_relu_options = itertools.product(
        [torch.nn.Conv2d, F.conv2d],
        [torch.add, operator.add, None],  # add
        [torch.nn.ReLU, F.relu, None],  # relu
    )

    # 遍历每一种可能的组合
    for conv_op, add_op, relu_op in conv_add_relu_options:
        if add_op is None:
            # 如果没有 Add 操作符，则添加 Conv + ReLU 组合到 "conv2d" 的支持操作符列表中
            supported_operators["conv2d"].append([conv_op, relu_op])  # type: ignore[list-item]
        elif relu_op is None:
            # 如果没有 ReLU 操作符，则添加 Conv + Add 组合到 "conv2d" 的支持操作符列表中
            supported_operators["conv2d"].append([conv_op, add_op])  # type: ignore[list-item]
        else:
            # 添加 Conv + Add + ReLU 组合到 "conv2d" 的支持操作符列表中
            supported_operators["conv2d"].append([conv_op, add_op, relu_op])  # type: ignore[list-item]

    # 返回支持操作符的深拷贝副本
    return copy.deepcopy(supported_operators)
# 获取支持的 x86 诱导器配置和运算符列表，返回值为 OperatorConfig 对象列表
def _get_supported_x86_inductor_config_and_operators() -> List[OperatorConfig]:
    # 初始化一个空列表，用于存储支持的配置和运算符
    supported_config_and_operators: List[OperatorConfig] = []
    
    # 遍历默认的 x86 诱导器量化配置列表，此处只包含一个配置
    for quantization_config in [
        get_default_x86_inductor_quantization_config(),
    ]:
        # 获取支持的量化操作符列表
        ops = _supported_quantized_operators()
        
        # 遍历每个操作符模式列表
        for pattern_list in ops.values():
            # 创建 OperatorConfig 对象并添加到列表中
            supported_config_and_operators.append(
                OperatorConfig(quantization_config, pattern_list)
            )
    
    # 返回深拷贝后的支持配置和运算符列表
    return copy.deepcopy(supported_config_and_operators)


# 使用 functools.lru_cache 装饰器缓存默认的 x86 诱导器量化配置函数
def get_default_x86_inductor_quantization_config(
    is_qat: bool = False,
    is_dynamic: bool = False,
):
    # 初始化额外的参数字典，设置 eps 值为 2 的负 12 次方
    extra_args: Dict[str, Any] = {"eps": 2**-12}
    
    # 根据参数配置选择激活观察器或伪量化器
    if is_qat:
        if is_dynamic:
            # 如果是 QAT 并且是动态量化，则使用 MovingAverageMinMaxObserver
            act_observer_or_fake_quant_ctr = FakeQuantize
            dynamic_quant_observer = MovingAverageMinMaxObserver.with_args(
                averaging_constant=1
            )
            extra_args["observer"] = dynamic_quant_observer
        else:
            # 如果是 QAT 但不是动态量化，则使用 FusedMovingAvgObsFakeQuantize
            act_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize  # type: ignore[assignment]
    else:
        if is_dynamic:
            # 如果不是 QAT 但是动态量化，则使用 PlaceholderObserver
            act_observer_or_fake_quant_ctr = PlaceholderObserver  # type: ignore[assignment]
        else:
            # 默认使用 HistogramObserver
            act_observer_or_fake_quant_ctr = HistogramObserver  # type: ignore[assignment]

    # 定义激活量化规格对象，设置数据类型为 torch.uint8，量化范围为 0 到 255
    act_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=255,  # reduce_range=False
        qscheme=torch.per_tensor_affine,
        is_dynamic=is_dynamic,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )

    # 根据是否是 QAT 选择权重观察器或伪量化器
    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        FusedMovingAvgObsFakeQuantize if is_qat else PerChannelMinMaxObserver
    )

    # 如果是 QAT，则设置额外参数使用 MovingAveragePerChannelMinMaxObserver
    if is_qat:
        # 目前只支持每通道量化
        extra_args["observer"] = MovingAveragePerChannelMinMaxObserver  # type: ignore[dict-item]
    
    # 定义权重量化规格对象，设置数据类型为 torch.int8，量化范围为 -128 到 127
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,  # 0 对应于权重形状为 (oc, ic, kh, kw) 的卷积
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )
    
    # 设置偏置量化规格为 None，默认使用占位符观察器
    bias_quantization_spec = None  # will use placeholder observer by default
    
    # 创建量化配置对象并返回
    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
        is_qat,
    )
    return quantization_config


# 获取支持的配置和运算符列表，调用 _get_supported_x86_inductor_config_and_operators 函数返回结果
def _get_supported_config_and_operators() -> List[OperatorConfig]:
    return _get_supported_x86_inductor_config_and_operators()


# 对节点进行非量化注释，参数 nodes 可以是单个节点或节点列表
def _annotate_nodes_not_quantize(nodes: Union[Node, List[Node]]) -> None:
    """Annotate nodes to exclude them from quantization (their `quantization_config` is `None`)."""
    # 如果 `nodes` 不是列表类型，则将其转换为列表
    if not isinstance(nodes, list):
        nodes = [nodes]
    # 遍历每个节点，并设置它们的元数据，标记为不需要量化的注释
    for node in nodes:
        # 为节点的元数据中添加一个量化注释键，值为一个特定的量化注释对象
        node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
            _annotated=True
        )
# 创建一个装饰器函数，用于检查配置的有效性
def _config_checker(method: Callable) -> Callable:
    # 定义装饰器函数的内部包装函数，保留原始方法的元数据
    @functools.wraps(method)
    def wrapper(
        quantizer: "X86InductorQuantizer",
        name: Any,
        quantization_config: Optional["QuantizationConfig"],
    ) -> "X86InductorQuantizer":
        # 检查是否需要跳过指定配置，如果是，则发出警告并返回原始量化器
        if quantizer._need_skip_config(quantization_config):
            warnings.warn(
                f"Skip the quantization config for {name}.",
            )
            return quantizer
        # 否则调用原始方法，传递量化器、名称和配置，并返回结果
        return method(quantizer, name, quantization_config)

    return wrapper


@dataclass
class _CurrentQuantizationMode:
    r"""Configuration defining the current quantization mode for the quantizer.

    All possible current quantization modes are listed below:
    ----------------------------------------------------------------------------------------------------------
                |                                       dynamic_state
     qat_state  |---------------------------------------------------------------------------------------------
                |                           None                              |    True       |  False
    ----------------------------------------------------------------------------------------------------------
        None    | quantizer does not receive a non-None `quantization_config` | \             | \
        False   | quantizer will not do QAT                                   | dynamic       | static
        True    | quantizer will do QAT                                       | QAT + dynamic | QAT + static
    """

    # 当前量化模式的数据结构，包含量化自动微调和动态量化的状态
    qat_state: Optional[bool]
    dynamic_state: Optional[bool]


class X86InductorQuantizer(Quantizer):
    # 获取支持的配置和操作符列表
    supported_config_and_operators = _get_supported_config_and_operators()
    # 映射模块函数到aten操作符类型
    module_function_to_aten_operator_type = _map_module_function_to_aten_operator_type()

    def __init__(self):
        # 初始化函数，继承Quantizer类的初始化方法
        super().__init__()
        # 全局配置，默认为None
        self.global_config: Optional[QuantizationConfig] = None
        # 操作符类型配置字典，使用torch._ops.OpOverloadPacket作为键，对应的量化配置作为值
        self.operator_type_qconfig: Dict[
            torch._ops.OpOverloadPacket, Optional[QuantizationConfig]
        ] = {}
        # 模块名称到量化配置的映射字典，键为模块名称，值为对应的量化配置
        self.module_name_qconfig: Dict[str, Optional[QuantizationConfig]] = {}

    @classmethod
    # 返回支持的量化配置列表
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        # 使用集合推导式获取支持的操作配置集合
        op_configs: Set[QuantizationConfig] = {
            spec for spec, _ in cls.supported_config_and_operators
        }
        # 将集合转换为列表并返回
        return list(op_configs)

    @classmethod
    # 返回支持指定量化配置的操作符列表
    def get_supported_operator_for_quantization_config(
        cls, quantization_config: Optional[QuantizationConfig]
    ) -> List[OperatorPatternType]:
        # 如果量化配置为None，则返回所有支持的操作符列表
        if quantization_config is None:
            all_ops = []
            for _, ops in cls.supported_config_and_operators:
                all_ops.extend(ops)
            return all_ops

        # 否则，遍历支持的配置和操作符列表，查找匹配的配置，并返回相应的操作符列表
        for config, ops in cls.supported_config_and_operators:
            if config == quantization_config:
                return ops
        # 如果没有找到匹配的配置，则返回空列表
        return []
    def _get_current_quantization_mode(self) -> _CurrentQuantizationMode:
        """Retrieves the current quantization mode based on all configurations."""
        qat_state = None  # 初始化量化训练状态为None
        dynamic_state = None  # 初始化动态量化状态为None

        # 遍历所有配置，通过 `_need_skip_config` 方法跳过无效配置
        # 可以安全地假设所有非None的配置具有相同的量化模式
        for qconfig in (
            list(self.module_name_qconfig.values())  # 遍历模块名到量化配置的映射
            + list(self.operator_type_qconfig.values())  # 遍历操作类型到量化配置的映射
            + [self.global_config]  # 全局配置
        ):
            if qconfig is not None:
                # 查询量化训练状态 `is_qat`
                if qat_state is None:
                    qat_state = qconfig.is_qat
                else:
                    assert qat_state == qconfig.is_qat, (
                        f"All non-None quantization configs should have the same `is_qat`,"
                        f"but got {qat_state} and {qconfig.is_qat}."
                    )
                # 查询动态量化状态 `is_dynamic`
                input_activation_spec = qconfig.input_activation
                if input_activation_spec is not None:
                    if dynamic_state is None:
                        dynamic_state = input_activation_spec.is_dynamic
                    else:
                        assert dynamic_state == input_activation_spec.is_dynamic, (
                            f"All non-None `input_activation_spec` should have the same `is_dynamic`,"
                            f"but got {dynamic_state} and {input_activation_spec.is_dynamic}."
                        )
        # 返回当前的量化模式，包括量化训练状态和动态量化状态
        return _CurrentQuantizationMode(
            qat_state=qat_state, dynamic_state=dynamic_state
        )

    def _need_skip_config(
        self, quantization_config: Optional[QuantizationConfig]
    ) -> bool:
        """检查提供的量化配置是否对于 X86InductorQuantizer 有效。

        不支持混合静态/动态配置或混合 QAT/非 QAT 配置。
        为避免这种混合，我们将传入的配置与当前配置状态进行比较。
        参考 `_CurrentQuantizationMode` 定义了所有可能的模式。
        """
        # 如果量化配置为 None，则返回 False
        if quantization_config is None:
            return False

        # 默认不需要跳过配置
        need_skip = False
        # 获取当前的量化模式
        current_mode = self._get_current_quantization_mode()
        
        # 如果当前模式的 QAT 状态不为空，并且与传入配置的 QAT 状态不同
        if (
            current_mode.qat_state is not None
            and current_mode.qat_state != quantization_config.is_qat
        ):
            # 发出警告信息
            warnings.warn("Mixed QAT and Non-QAT quantization config is not supported.")
            need_skip = True
        
        # 如果当前模式的动态状态不为空
        if current_mode.dynamic_state is not None:
            # 获取输入激活的量化配置
            input_activation_spec = quantization_config.input_activation
            # 如果输入激活的配置不为空，并且当前模式的动态状态与输入激活的配置状态不同
            if (
                input_activation_spec is not None
                and current_mode.dynamic_state != input_activation_spec.is_dynamic
            ):
                # 发出警告信息
                warnings.warn(
                    "Mixed dynamic and static quantization config is not supported."
                )
                need_skip = True
        
        # 返回是否需要跳过配置的结果
        return need_skip

    def set_global(self, quantization_config: QuantizationConfig):
        """设置全局量化配置。

        如果需要跳过配置，则发出警告并返回当前对象。
        否则，设置全局配置为传入的量化配置，并返回当前对象。
        """
        # 如果需要跳过配置
        if self._need_skip_config(quantization_config):
            # 发出警告信息
            warnings.warn("Skip the global quantization config.")
            # 返回当前对象
            return self
        
        # 设置全局配置为传入的量化配置
        self.global_config = quantization_config
        # 返回当前对象
        return self

    def get_global_quantization_config(self):
        """获取全局量化配置。

        如果全局配置不是 QuantizationConfig 类型，则发出警告信息。
        返回当前的全局配置。
        """
        # 如果全局配置不是 QuantizationConfig 类型
        if not isinstance(self.global_config, QuantizationConfig):
            # 发出警告信息
            warnings.warn(
                "The global_config for X86InductorQuantizer is currently invalid. \
                Please ensure that you use set_global to establish the global quantization configuration."
            )
        
        # 返回当前的全局配置
        return self.global_config

    @_config_checker
    def set_function_type_qconfig(
        self,
        function_type: Callable,
        quantization_config: Optional[QuantizationConfig],
    ) -> "X86InductorQuantizer":
        """设置函数类型的量化配置。

        根据函数类型设置相应的操作符量化配置。
        如果函数类型无法找到对应的操作符类型，发出警告信息。
        返回当前对象。
        """
        # 如果函数类型在 module_function_to_aten_operator_type 中
        if function_type in X86InductorQuantizer.module_function_to_aten_operator_type:
            # 根据函数类型设置相应的操作符量化配置
            self._set_aten_operator_qconfig(
                X86InductorQuantizer.module_function_to_aten_operator_type[
                    function_type
                ],
                quantization_config,
            )
        else:
            # 发出警告信息，说明无法定制该函数类型的量化配置
            warnings.warn(
                f"function: Unable to customize quantization config for {function_type} by X86InductorQuantizer."
            )
        
        # 返回当前对象
        return self

    @_config_checker
    def set_module_type_qconfig(
        self,
        module_type: torch.nn.Module,
        quantization_config: Optional[QuantizationConfig],
    ) -> "X86InductorQuantizer":
        """设置模块类型的量化配置。

        根据模块类型设置相应的量化配置。
        返回当前对象。
        """
        # 如果模块类型在 module_function_to_aten_operator_type 中
        if module_type in X86InductorQuantizer.module_function_to_aten_operator_type:
            # 调用内部方法设置操作符量化配置
            self._set_aten_operator_qconfig(
                X86InductorQuantizer.module_function_to_aten_operator_type[
                    module_type
                ],
                quantization_config,
            )
        # 返回当前对象
        return self
    ) -> "X86InductorQuantizer":
        # 定义函数的返回类型为 "X86InductorQuantizer"
        if module_type in X86InductorQuantizer.module_function_to_aten_operator_type:
            # 检查 module_type 是否在预先定义的映射中，设置对应的 ATen 操作符的量化配置
            self._set_aten_operator_qconfig(
                X86InductorQuantizer.module_function_to_aten_operator_type[module_type],
                quantization_config,
            )
        else:
            # 如果 module_type 不在映射中，发出警告
            warnings.warn(
                f"Module: Unable to customize quantization config for {module_type} by X86InductorQuantizer."
            )
        # 返回当前对象实例，以支持链式调用
        return self

    @_config_checker
    def set_module_name_qconfig(
        self, module_name: str, quantization_config: Optional[QuantizationConfig]
    ):
        """Set quantization_config for a submodule with name: `module_name`, for example:
        quantizer.set_module_name_qconfig("blocks.sub"), it will quantize all supported operator/operator
        patterns in the submodule with this module name with the given `quantization_config`

        The supported operators include `quantizable_ops` and `propagation_quantizable_ops`.
        """
        # 为指定的子模块名称 module_name 设置量化配置 quantization_config
        self.module_name_qconfig[module_name] = quantization_config
        # 返回当前对象实例，以支持链式调用
        return self

    def _set_aten_operator_qconfig(
        self,
        operator_type: torch._ops.OpOverloadPacket,
        quantization_config: Optional[QuantizationConfig],
    ) -> "X86InductorQuantizer":
        # 设置特定的 ATen 操作符的量化配置
        if operator_type in quantizable_ops:
            self.operator_type_qconfig[operator_type] = quantization_config
        else:
            # 如果操作符不支持量化，发出警告
            warnings.warn(
                f"operator: Unable to quantize {operator} by X86InductorQuantizer."
            )
        # 返回当前对象实例，以支持链式调用
        return self

    def _annotate_conv_node_helper(
        self,
        conv_node: torch.fx.Node,
        annotate_output: bool,
        quantization_config: Optional[QuantizationConfig],
    ) -> None:
        """Helper function to annotate the conv node"""
        # 辅助函数，用于给卷积节点添加注释
        if quantization_config is None:
            # 如果没有量化配置，调用特定函数来标记不需要量化的节点
            _annotate_nodes_not_quantize(conv_node)
            return
        input_qspec_map = {}
        input_node = conv_node.args[0]
        assert isinstance(input_node, Node)
        # 为输入节点获取输入激活量化规范
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        weight_node = conv_node.args[1]
        assert isinstance(weight_node, Node)
        # 为权重节点获取权重量化规范
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        bias_node = None if len(conv_node.args) == 2 else conv_node.args[2]
        if isinstance(bias_node, Node):
            # 如果存在偏置节点，为其获取偏置量化规范
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        if annotate_output:
            # 如果需要注释输出节点，设置特定的量化注释信息
            conv_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )
        else:
            # 否则，设置一般的量化注释信息
            conv_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
            )
    def _annotate_linear_node_helper(
        self,
        linear_node: torch.fx.Node,
        annotate_output: bool,
        quantization_config: Optional[QuantizationConfig],
    ) -> None:
        """Helper function to annotate the linear node"""
        # 如果没有量化配置，调用非量化节点注解函数并返回
        if quantization_config is None:
            _annotate_nodes_not_quantize(linear_node)
            return
        
        # 用于映射输入节点和它们的量化规格
        input_qspec_map = {}
        
        # 确保目标节点为 torch.ops.aten.linear.default
        assert linear_node.target in (torch.ops.aten.linear.default,)
        
        # 检查是否有偏置项
        has_bias = len(linear_node.args) == 3
        
        # 确定输入、权重和偏置项的索引
        input_index = 0
        weight_index = 1
        bias_index = 2
        
        # 获取输入节点，并确保其为 Node 类型
        input_node = linear_node.args[input_index]
        assert isinstance(input_node, Node)
        # 获取输入节点的激活量化规格
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        
        # 获取权重节点，并确保其为 Node 类型
        weight_node = linear_node.args[weight_index]
        assert isinstance(weight_node, Node)
        # 获取权重节点的量化规格
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        
        # 如果存在偏置节点，则获取偏置节点的量化规格
        bias_node = linear_node.args[bias_index] if has_bias else None
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        
        # 如果需要注解输出节点，则创建对应的量化注解对象
        if annotate_output:
            linear_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )
        else:
            # 否则，创建普通的量化注解对象
            linear_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map, _annotated=True
            )

    def _get_output_nodes_of_partitions(
        self,
        partition_list: List[SourcePartition],
    ) -> List[torch.fx.Node]:
        """Helper function to get the output node list from partition list"""
        # 初始化输出节点列表
        output_node_list = []
        
        # 遍历分区列表中的每个分区
        for partition in partition_list:
            # 检查分区的输出节点数是否超过一个，如果是则抛出错误
            if len(partition.output_nodes) > 1:
                raise ValueError("Input partition has more than one output node")
            
            # 获取分区的输出节点，并确保其为 Node 类型
            output_node = partition.output_nodes[0]
            assert isinstance(output_node, Node)
            
            # 将输出节点添加到输出节点列表中
            output_node_list.append(output_node)
        
        # 检查输出节点列表的长度是否与分区列表的长度相等
        if len(output_node_list) != len(partition_list):
            raise ValueError(
                "length of output_node_list should equal to length of partition_list"
            )
        
        # 返回输出节点列表
        return output_node_list

    def _get_input_idx_for_binary_node(
        self,
        conv_gemm_node: torch.fx.Node,
        binary_node: torch.fx.Node,
        input_gemm_idx: int
    ) -> Tuple[int, int]:
        """Helper function to get the input indices for the binary node"""
        # 略
    ):
        """Helper function to determine indices for conv_gemm and extra input nodes in a binary operation node.
        
        Args:
            binary_node (Node): The binary operation node to analyze.
            conv_gemm_node (Node): The node representing conv_gemm.

        Returns:
            Tuple[int, int]: Indices of conv_gemm_node and extra input node within binary_node.
        """
        conv_gemm_node_idx = None
        extra_input_node_idx = None
        
        # Check if the first argument of binary_node is conv_gemm_node
        if (binary_node.args[0].op == "call_function") and (  # type: ignore[union-attr]
            binary_node.args[0] == conv_gemm_node
        ):
            conv_gemm_node_idx = 0
            extra_input_node_idx = 1
        
        # Check if the second argument of binary_node is conv_gemm_node
        elif (binary_node.args[1].op == "call_function") and (  # type: ignore[union-attr]
            binary_node.args[1] == conv_gemm_node
        ):
            conv_gemm_node_idx = 1
            extra_input_node_idx = 0
        
        # Retrieve the node representing the extra input based on its index
        extra_input_node = binary_node.args[extra_input_node_idx]  # type: ignore[index]
        
        # Ensure that the extra input node is of type Node
        assert isinstance(extra_input_node, Node)
        
        # Return the indices of conv_gemm_node and extra input node
        return conv_gemm_node_idx, extra_input_node_idx
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Annotate the given model with quantization configurations.

        Annotation contracts:
        1. Annotate each node according to the user's qconfig in the following order:
        `module_name_qconfig`, `operator_type_qconfig`, and `global_config`.
        2. Avoid re-annotating nodes already annotated in prior stages. For example,
        if `linear1` has been annotated by `module_name_qconfig`, it won't be annotated again
        during the processing of the 'operator_type_qconfig' or 'global_config'.
        3. For config is `None`, the node will be annotated with `_X86InductorQuantizationAnnotation(_annotated=True)`.

        For each pair of (module_name_or_operator_type_or_global, qconfig), a filter function is created.
        This filter function checks if the node is marked by current stage and not annotated by the previous stage.
        """
        
        # Iterate through each (module_name, quantization_config) pair in module_name_qconfig
        for module_name, quantization_config in self.module_name_qconfig.items():
            # Annotate model nodes based on the given quantization_config and module_name filter
            self._annotate_with_config(
                model, quantization_config, _create_module_name_filter(module_name)
            )

        # Iterate through each (operator_type, quantization_config) pair in operator_type_qconfig
        for operator_type, quantization_config in self.operator_type_qconfig.items():
            # Annotate model nodes based on the given quantization_config and operator_type filter
            self._annotate_with_config(
                model, quantization_config, _create_operator_type_filter(operator_type)
            )

        # If global_config is defined, annotate model nodes using global_config and _global_config_filter
        if self.global_config:
            self._annotate_with_config(
                model,
                self.global_config,
                _global_config_filter,
            )

        # Annotate the output of quantizable operations to ensure consistent quantization handling
        self._annotate_output_for_int8_in_int8_out_pattern_entry(model)

        # Return the annotated model
        return model

    def _annotate_with_config(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: FilterFn,
    ) -> None:
        """
        为模型添加给定的量化配置注释。

        X86 Inductor 后端的量化配方概述：
        Step 1: 应用卷积/线性融合模式的量化配方，以启用活跃的 int8 数据类型。
        Step 2: 传播除卷积/线性外其他模式的量化注释。遍历模型中的模式，从开头到结尾。
        如果模式支持使用 int8 数据类型进行计算，并且其输入连接到量化模式，将其输入标注为量化模式。
        """

        # Step1: 卷积/线性融合模式的配方。
        self._annotate_conv2d_fusion_pattern(model, quantization_config, filter_fn)
        self._annotate_linear_fusion_pattern(model, quantization_config, filter_fn)
        self._annotate_matmul(model, quantization_config, filter_fn)

        # Step2: 传播注释到除卷积/线性外的其他模式。
        # 从开头到结尾遍历所有节点。
        # 配方参考 https://github.com/intel/intel-extension-for-pytorch/blob/
        # 90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_recipe.py#L538

        self._annotate_propagation_quantizable_pattern_entry(
            model, quantization_config, filter_fn
        )

    def _annotate_qat_conv2d_fusion_pattern(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        # 注释 QAT 特定模式
        self._annotate_qat_conv2d_bn_binary_unary(model, quantization_config, filter_fn)
        self._annotate_qat_conv2d_bn_binary(model, quantization_config, filter_fn)
        self._annotate_qat_conv2d_bn_unary(model, quantization_config, filter_fn)
        self._annotate_qat_conv2d_bn(model, quantization_config, filter_fn)

    def _annotate_qat_conv2d_bn_binary_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        # 注释 QAT 的卷积-批标准化-二元-一元模式
        ...

    def _annotate_qat_conv2d_bn_binary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        # 注释 QAT 的卷积-批标准化-二元模式
        ...
    # 定义一个方法用于对 QAT（量化感知训练）中的 Conv-BN 二元节点进行标注
    def _annotate_qat_conv2d_bn_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        # 查找序列化分区，这些分区包括 Conv2d、BatchNorm2d 和加法操作
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d, operator.add]
        )
        # 遍历每个融合分区
        for fused_partition in fused_partitions:
            # 将融合分区解包成单独的 Conv、BN 和二元节点
            conv_partition, bn_partition, binary_partition = fused_partition
            # 获取 Conv、BN 和二元节点的输出节点
            (
                conv_node,
                bn_output_node,
                binary_node,
            ) = self._get_output_nodes_of_partitions(
                [conv_partition, bn_partition, binary_partition]
            )
            # 如果 BN 输出节点的使用者不唯一，则跳过该模式
            if len(bn_output_node.users) != 1:
                continue
            # 获取二元节点与 BN 输出节点的输入索引
            (
                bn_output_node_idx,
                extra_input_node_idx,
            ) = self._get_input_idx_for_binary_node(bn_output_node, binary_node)
            # 如果索引不存在则跳过当前融合分区的处理
            if (bn_output_node_idx is None) or (extra_input_node_idx is None):
                continue
            # 如果 BN 输出节点不匹配二元节点的特定输入，则引发异常
            if bn_output_node != binary_node.args[bn_output_node_idx]:
                raise ValueError(f"{bn_output_node} doesn't match input of binary node")

            # 获取额外输入节点
            extra_input_node = binary_node.args[extra_input_node_idx]

            # 如果 Conv 节点不是 "call_function" 或者不是默认的 torch conv2d 操作，则跳过当前融合分区的处理
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                continue

            # 调用内部方法，为 Conv 节点标注信息，用于量化配置
            self._annotate_conv_node_helper(conv_node, False, quantization_config)

            # 如果存在量化配置，则为二元节点的输入映射创建一个映射表
            if quantization_config is not None:
                binary_node_input_qspec_map = {}
                # 将额外输入节点映射到输入激活量化配置
                binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                    quantization_config
                )
                # 设置二元节点的元数据，包括输入和输出量化配置的注释
                binary_node.meta[
                    QUANT_ANNOTATION_KEY
                ] = _X86InductorQuantizationAnnotation(
                    input_qspec_map=binary_node_input_qspec_map,
                    # TODO<leslie> 在 qat 工具支持模式匹配器时移除输出的注释。
                    output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                    _annotated=True,
                    _is_output_of_quantized_pattern=True,
                )
            else:
                # 如果没有量化配置，则标注二元节点为不量化
                _annotate_nodes_not_quantize(binary_node)

            # 将融合分区中的节点标记为已标注
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            nodes_to_mark_annotated.extend(list(binary_partition.nodes))
            _mark_nodes_as_annotated(nodes_to_mark_annotated)
    ) -> None:
        # 初始化空列表，用于存储合并后的分区
        fused_partitions = []
        # 预定义多个一元模式，每个模式都包含一组特定的神经网络层类
        unary_patterns = [
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU],
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Hardtanh],
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Hardswish],
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU6],
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.SiLU],
        ]
        # 遍历每个一元模式，寻找符合条件的连续分区
        for unary_pattern in unary_patterns:
            partitions = find_sequential_partitions(gm, unary_pattern)
            # 如果找到分区，则将其添加到合并分区列表中
            if partitions:
                # 如果分区不为空，则将其扩展到已合并分区列表中
                fused_partitions.extend(partitions)

        # 遍历所有合并的分区
        for fused_partition in fused_partitions:
            # 解构合并分区的三个部分：卷积分区、批归一化分区和一元操作分区
            conv_partition, bn_partition, unary_partition = fused_partition
            # 获取这三个分区的输出节点
            (
                conv_node,
                bn_output_node,
                unary_node,
            ) = self._get_output_nodes_of_partitions(
                [conv_partition, bn_partition, unary_partition]
            )

            # 如果卷积节点的操作类型不是 "call_function" 或目标不是默认的 torch.conv2d 操作，则跳过当前循环
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                continue

            # 如果一元节点、批归一化输出节点和卷积节点符合特定的过滤函数，则跳过当前循环
            if _skip_annotate([unary_node, bn_output_node, conv_node], filter_fn):
                continue

            # 对卷积节点进行注释，指定是否进行量化的配置
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            # 如果存在量化配置，则为一元节点添加量化相关的元数据注释
            if quantization_config is not None:
                unary_node.meta[
                    QUANT_ANNOTATION_KEY
                ] = _X86InductorQuantizationAnnotation(
                    # TODO<leslie> Remove the annotate of output in QAT when qat util support pattern matcher.
                    output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                    _annotated=True,
                    _is_output_of_quantized_pattern=True,
                )
            else:
                # 否则，对一元节点执行未量化的注释
                _annotate_nodes_not_quantize(unary_node)
            
            # 将需要标注为已注释的节点列表扩展为包含所有分区的节点
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            nodes_to_mark_annotated.extend(list(unary_partition.nodes))
            # 标记这些节点为已注释状态
            _mark_nodes_as_annotated(nodes_to_mark_annotated)
    ) -> None:
        # 查找连续的分区，这些分区包含 Conv2d 和 BatchNorm2d 层
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
        )
        # 遍历每个融合的分区
        for fused_partition in fused_partitions:
            # 分离出 Conv2d 和 BatchNorm2d 分区
            conv_partition, bn_partition = fused_partition
            # 获取 Conv2d 节点和 BatchNorm2d 输出节点
            conv_node, bn_output_node = self._get_output_nodes_of_partitions(
                [conv_partition, bn_partition]
            )

            # 如果 Conv2d 节点不是 call_function 或者不是 torch.ops.aten.conv2d.default
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                # 继续下一个融合分区的处理
                continue

            # 如果需要跳过对节点的注释（根据 filter_fn 的返回结果决定）
            if _skip_annotate([bn_output_node, conv_node], filter_fn):
                # 继续下一个融合分区的处理
                continue

            # 对 Conv2d 节点进行注释
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            
            # 如果 quantization_config 不为 None，则对 BatchNorm2d 输出节点进行量化注释
            if quantization_config is not None:
                bn_output_node.meta[
                    QUANT_ANNOTATION_KEY
                ] = _X86InductorQuantizationAnnotation(
                    # TODO<leslie> 在 QAT 中移除输出的注释，当 qat 工具支持模式匹配时。
                    output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                    _annotated=True,
                    _is_output_of_quantized_pattern=True,
                )
            else:
                # 否则对节点进行非量化的注释
                _annotate_nodes_not_quantize(bn_output_node)
            
            # 将需要标注为已注释的节点列出，并标记它们
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_conv2d_fusion_pattern(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        # 如果 quantization_config 为 None 或者是 QAT 模式
        if (quantization_config is None) or (quantization_config.is_qat):
            # 对 QAT 特定的 Conv2d 融合模式进行注释
            self._annotate_qat_conv2d_fusion_pattern(
                model, quantization_config, filter_fn
            )
        # 对二元操作的 Conv2d 进行注释
        self._annotate_conv2d_binary_unary(model, quantization_config, filter_fn)
        # 对二元操作的 Conv2d 进行注释
        self._annotate_conv2d_binary(model, quantization_config, filter_fn)
        # 对一元操作的 Conv2d 进行注释
        self._annotate_conv2d_unary(model, quantization_config, filter_fn)
        # 对 Conv2d 进行注释
        self._annotate_conv2d(model, quantization_config, filter_fn)

    def _annotate_linear_fusion_pattern(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        if (quantization_config is None) or (
            quantization_config.input_activation
            and not quantization_config.input_activation.is_dynamic
        ):
            # 如果没有指定量化配置或者输入激活函数不是动态的
            # <TODO> Weiwen: Dynamic Quant of linear unary will be supported in next step
            # 提示：动态量化线性一元将在下一步中支持
            # 对模型中的线性一元操作进行标注量化
            self._annotate_linear_binary_unary(model, quantization_config, filter_fn)
            # 对模型中的线性一元操作进行标注量化
            self._annotate_linear_unary(model, quantization_config, filter_fn)
        # 对模型中的线性操作进行标注量化
        self._annotate_linear(model, quantization_config, filter_fn)

    def _annotate_matmul(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        # 遍历模型图中的每个节点
        for node in model.graph.nodes:
            # 如果节点的目标不是 torch.ops.aten.matmul.default 则继续下一个节点
            if node.target != torch.ops.aten.matmul.default:
                continue
            # 如果节点需要跳过标注量化则继续下一个节点
            if _skip_annotate([node], filter_fn):
                continue

            # 如果没有指定量化配置
            if quantization_config is None:
                # 对不需要量化的节点进行标注
                _annotate_nodes_not_quantize(node)
                continue

            # 创建空的输入量化规格映射
            input_qspec_map = {}
            # 获取当前 matmul 节点
            matmul_node = node
            # 遍历 matmul 节点的每个输入节点
            for input_node in matmul_node.args:
                # 将每个输入节点映射到输入激活的量化规格
                input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
            # 设置 matmul 节点的量化注释键
            matmul_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_conv2d_binary_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
    ) -> None:
        # 定义函数 _annotate_conv2d_binary，用于对 Conv2d + add + unary op 进行注释和量化标注
        # 查找顺序执行的分区，包括 Conv2d、operator.add 和 torch.nn.ReLU
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, operator.add, torch.nn.ReLU]
        )
        # 遍历每个找到的顺序执行分区
        for fused_partition in fused_partitions:
            # 解包当前融合分区的三个子分区：conv_partition, binary_partition, unary_partition
            conv_partition, binary_partition, unary_partition = fused_partition
            # 获取分区中的输出节点：conv_node, binary_node, unary_node
            conv_node, binary_node, unary_node = self._get_output_nodes_of_partitions(
                [conv_partition, binary_partition, unary_partition]
            )
            # 如果 conv_node 的用户节点数量不等于 1，则跳过当前 conv_node
            if len(conv_node.users) != 1:
                # Conv Node 应只有一个用户节点
                continue
            # 获取用于二进制节点的输入索引：conv_node_idx, extra_input_node_idx
            conv_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(
                conv_node, binary_node
            )
            # 如果 conv_node_idx 或 extra_input_node_idx 为 None，则跳过当前节点
            if (conv_node_idx is None) or (extra_input_node_idx is None):
                continue
            # 检查 conv_node 是否等于 binary_node 的第 conv_node_idx 个参数
            if conv_node != binary_node.args[conv_node_idx]:
                raise ValueError(f"{conv_node} doesn't match input of binary node")
            # 获取额外输入节点：extra_input_node
            extra_input_node = binary_node.args[extra_input_node_idx]
            # 如果 conv_node 不是调用函数，或者 conv_node 的目标不是 torch.ops.aten.conv2d.default，则跳过
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                # 没有找到可以与 add 融合的 conv_node
                continue
            # 如果通过 _skip_annotate 函数跳过了节点，则继续下一个循环
            if _skip_annotate([unary_node, binary_node, conv_node], filter_fn):
                continue

            # 如果 quantization_config 为 None，则标注这些节点不进行量化
            if quantization_config is None:
                _annotate_nodes_not_quantize([conv_node, binary_node, unary_node])
                continue

            # 使用 _annotate_conv_node_helper 函数对 conv_node 进行量化标注
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            # 创建二进制节点输入量化规格映射表：binary_node_input_qspec_map
            binary_node_input_qspec_map = {}
            # 将额外输入节点及其量化规格添加到映射表中
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                quantization_config
            )
            # 为 binary_node 添加量化标注元数据：QUANT_ANNOTATION_KEY
            binary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=binary_node_input_qspec_map,
                _annotated=True,
            )
            # 为 unary_node 添加量化标注元数据：QUANT_ANNOTATION_KEY
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )
    ) -> None:
        # 定义一个方法，用于对 Conv2d 和 add 进行融合
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, operator.add]
        )
        # 遍历融合后的分区
        for fused_partition in fused_partitions:
            conv_partition, binary_partition = fused_partition
            # 获取 Conv2d 和 add 节点的输出节点
            conv_node, binary_node = self._get_output_nodes_of_partitions(
                [conv_partition, binary_partition]
            )
            if len(conv_node.users) != 1:
                # Conv 节点应该只有一个用户节点
                continue
            # 获取 Conv2d 节点和 add 节点的输入索引
            conv_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(
                conv_node, binary_node
            )
            if (conv_node_idx is None) or (extra_input_node_idx is None):
                continue
            if conv_node != binary_node.args[conv_node_idx]:
                raise ValueError(f"{conv_node} doesn't match input of binary node")
            extra_input_node = binary_node.args[extra_input_node_idx]
            assert isinstance(conv_node, Node)
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                # 没有找到要与 add 融合的 Conv 节点
                continue
            if _skip_annotate([binary_node, conv_node], filter_fn):
                continue

            if quantization_config is None:
                _annotate_nodes_not_quantize([conv_node, binary_node])
                continue

            # 对 Conv2d 节点进行标注
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                quantization_config
            )
            binary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=binary_node_input_qspec_map,
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_conv2d_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        fused_partitions = []
        unary_patterns = [
            [torch.nn.Conv2d, torch.nn.ReLU],    # 定义一组包含 Conv2d 和 ReLU 的模式列表
            [torch.nn.Conv2d, torch.nn.Hardtanh],    # 定义一组包含 Conv2d 和 Hardtanh 的模式列表
            [torch.nn.Conv2d, torch.nn.Hardswish],   # 定义一组包含 Conv2d 和 Hardswish 的模式列表
            [torch.nn.Conv2d, torch.nn.ReLU6],   # 定义一组包含 Conv2d 和 ReLU6 的模式列表
            [torch.nn.Conv2d, torch.nn.SiLU],    # 定义一组包含 Conv2d 和 SiLU 的模式列表
        ]
        for unary_pattern in unary_patterns:
            partitions = find_sequential_partitions(gm, unary_pattern)
            if partitions:
                # 如果找到了分区，则将其扩展到 fused_partitions
                fused_partitions.extend(partitions)

        for fused_partition in fused_partitions:
            conv_partition, unary_partition = fused_partition
            conv_node, unary_node = self._get_output_nodes_of_partitions(
                [conv_partition, unary_partition]
            )
            if (
                conv_node.op != "call_function"    # 如果 conv_node 不是 call_function 操作
                or conv_node.target != torch.ops.aten.conv2d.default    # 或者 conv_node 的目标不是 torch.ops.aten.conv2d.default
            ):
                continue
            if _skip_annotate([unary_node, conv_node], filter_fn):
                continue

            if quantization_config is None:
                _annotate_nodes_not_quantize([conv_node, unary_node])    # 对不进行量化的节点进行标注
                continue

            self._annotate_conv_node_helper(conv_node, False, quantization_config)    # 对 conv_node 进行帮助性的卷积节点标注
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )    # 设置 unary_node 的量化注释信息为已标注的 x86 感应量化注释

    def _annotate_conv2d(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        conv_partitions = get_source_partitions(
            gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d]
        )    # 获取包含 Conv2d 和 functional.conv2d 的源分区
        conv_partitions = list(itertools.chain.from_iterable(conv_partitions.values()))    # 展开所有分区值并放入列表中
        for conv_partition in conv_partitions:
            if len(conv_partition.output_nodes) > 1:
                raise ValueError("conv partition has more than one output node")    # 如果 conv_partition 的输出节点超过一个，则引发 ValueError
            conv_node = conv_partition.output_nodes[0]
            if (
                conv_node.op != "call_function"    # 如果 conv_node 不是 call_function 操作
                or conv_node.target != torch.ops.aten.conv2d.default    # 或者 conv_node 的目标不是 torch.ops.aten.conv2d.default
            ):
                raise ValueError(f"{conv_node} is not an aten conv2d operator")    # 如果 conv_node 不是 aten conv2d 运算符，则引发 ValueError
            # 如果已经标注了，则跳过标注
            if _skip_annotate([conv_node], filter_fn):
                continue
            self._annotate_conv_node_helper(conv_node, True, quantization_config)    # 对 conv_node 进行帮助性的卷积节点标注

    def _annotate_maxpool2d(
        self,
        node: Node,
        quantization_config: Optional[QuantizationConfig],
    ) -> None:
        # 如果节点的目标不是 torch.ops.aten.max_pool2d.default，则返回
        if node.target is not torch.ops.aten.max_pool2d.default:
            return
        # 如果量化配置为 None，则对节点进行非量化标注并返回
        if quantization_config is None:
            _annotate_nodes_not_quantize(node)
            return

        # 将当前节点标记为最大池化节点
        maxpool_node = node
        # 如果已经有任何标注过的节点存在于 [maxpool_node] 中，则返回
        if _is_any_annotated(
            [
                maxpool_node,
            ]
        ):
            return

        # 获取最大池化操作的输入节点
        input_node = maxpool_node.args[0]
        # 确保输入节点是 Node 类型
        assert isinstance(input_node, Node)
        # 创建输入节点到输入激活量化规范映射的空字典
        input_qspec_map = {}
        # 将第一个输入节点映射到其对应的输入激活量化规范
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        # 为最大池化节点添加量化注释
        maxpool_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
            _is_output_of_quantized_pattern=True,
        )

    def _annotate_cat(
        self, node: Node, quantization_config: QuantizationConfig
    ) -> None:
        # 如果量化配置为 None，则对节点进行非量化标注并返回
        if quantization_config is None:
            _annotate_nodes_not_quantize(node)
            return
        # 将当前节点标记为拼接节点
        cat_node = node
        # 获取拼接操作的输入节点列表
        input_nodes = cat_node.args[0]
        # 确保输入节点是一个序列
        assert isinstance(input_nodes, Sequence)
        # 获取第一个输入节点
        first_input_node = input_nodes[0]
        # 创建输入节点到输入激活量化规范映射的空字典
        input_qspec_map = {}
        # 确保第一个输入节点是 Node 类型
        assert isinstance(first_input_node, Node)
        # 将第一个输入节点映射到其对应的输入激活量化规范
        input_qspec_map[first_input_node] = get_input_act_qspec(quantization_config)
        # 创建一个共享的量化规范对象，用于所有拼接操作的输入节点
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
            (first_input_node, cat_node)
        )

        # 遍历除第一个输入节点外的所有输入节点
        for input_node in input_nodes[1:]:
            # 如果当前输入节点尚未在映射中，则将其映射到共享的量化规范对象
            if input_node not in input_qspec_map:
                # 确保当前输入节点是 Node 类型
                assert isinstance(input_node, Node)
                # 将当前输入节点映射到共享的量化规范对象
                input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

        # 为拼接节点添加量化注释
        cat_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
            _is_output_of_quantized_pattern=True,
        )

    def _annotate_propagation_quantizable_pattern_entry(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        # 遍历图中的所有节点，为每个节点调用量化传播模式标注方法
        for node in gm.graph.nodes:
            self._annotate_propagation_quantizable_pattern(
                node, quantization_config, filter_fn
            )

    def _annotate_propagation_quantizable_pattern(
        self, node: Node, quantization_config, filter_fn
        # 量化传播模式标注方法的定义
    ) -> None:
        # 将注解传播到可量化模式。
        if (
            (node.target in propagation_quantizable_ops)  # 如果节点目标在可传播的可量化操作列表中
            and (not _is_any_annotated([node]))  # 并且该节点没有任何注解
            and (node.op == "call_function")  # 并且节点操作是调用函数
        ):

            def is_all_inputs_connected_to_quantized_op(input_nodes):
                # 确保所有输入连接到融合模式或量化节点
                for input_node in input_nodes:
                    if not _is_quantized_op_pt2e(input_node):
                        return False
                return True

            if _skip_annotate([node], filter_fn):
                return

            if quantization_config is None:
                _annotate_nodes_not_quantize(node)
                return

            if node.target is torch.ops.aten.max_pool2d.default:
                # Maxpool2d的注解方式：检查maxpool2d的输入参数arg[0]是否被量化
                input_nodes_to_check = [node.all_input_nodes[0]]
                if not is_all_inputs_connected_to_quantized_op(input_nodes_to_check):
                    if quantization_config is not None:
                        warnings.warn(
                            f"The input of maxpool2d is not quantized, skip annotate maxpool2d with config {quantization_config}."
                        )
                    return

                self._annotate_maxpool2d(node, quantization_config)
                return
            elif node.target is torch.ops.aten.cat.default:
                input_nodes_to_check = node.all_input_nodes
                if not is_all_inputs_connected_to_quantized_op(input_nodes_to_check):
                    return
                self._annotate_cat(node, quantization_config)
            else:
                input_node = node.all_input_nodes[0]
                if not is_all_inputs_connected_to_quantized_op(
                    [
                        input_node,
                    ]
                ):
                    return
                input_qspec_map = {}
                input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
                node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                    input_qspec_map=input_qspec_map,
                    _annotated=True,
                    _is_output_of_quantized_pattern=True,
                )
        return

    def _annotate_output_share_observer_as_input(
        self, input_node: Node, source_node: Node
    ):
        source_node_quantization_annotation = (
            source_node.meta[QUANT_ANNOTATION_KEY]  # 获取源节点的量化注释信息，如果存在的话
            if QUANT_ANNOTATION_KEY in source_node.meta  # 检查源节点的元数据是否包含量化注释键
            else None  # 如果不存在量化注释，则设置为 None
        )
        if (
            source_node_quantization_annotation  # 如果存在量化注释
            and source_node_quantization_annotation._is_output_of_quantized_pattern  # 并且源节点标记为量化模式的输出节点
        ):
            edge_or_node = (input_node, source_node)  # 构建边或节点元组
            source_node_quantization_annotation.output_qspec = SharedQuantizationSpec(
                edge_or_node  # 设置输出节点的量化规格
            )
        return  # 返回结束函数

    def _annotate_output_for_int8_in_int8_out_pattern_entry(
        self,
        model: torch.fx.GraphModule,
    ):
        for node in model.graph.nodes:  # 遍历模型图中的所有节点
            self._annotate_output_for_int8_in_int8_out_pattern(node)  # 对每个节点调用量化注释函数

    def _annotate_output_for_int8_in_int8_out_pattern(
        self,
        node: Node,
    ) -> None:
        r"""
        检查并在 int8_in_int8_out_ops 中需要时在节点输出处插入观察器。
        """
        edge_or_node: Tuple[Node, Node]  # 定义边或节点元组类型
        if (node.target in int8_in_int8_out_ops) and (_is_any_annotated([node])):
            if node.target == torch.ops.aten.max_pool2d.default:  # 如果节点目标是最大池化操作
                maxpool_node = node  # 设置最大池化节点
                if not _is_all_annotated(
                    [
                        maxpool_node,
                    ]
                ):
                    return  # 如果节点没有全部被注释，则返回

                # 从 maxpool_node 获取量化注释
                maxpool_node_quantization_annotation = (
                    maxpool_node.meta[QUANT_ANNOTATION_KEY]
                    if QUANT_ANNOTATION_KEY in maxpool_node.meta
                    else None
                )
                if (
                    maxpool_node_quantization_annotation
                    and maxpool_node_quantization_annotation._is_output_of_quantized_pattern
                ):
                    # 标记 maxpool_node 的输出量化规格
                    input_act = maxpool_node.args[0]  # 获取输入激活节点
                    assert isinstance(input_act, Node)  # 断言输入激活是节点类型
                    assert isinstance(maxpool_node, Node)  # 断言 maxpool_node 是节点类型
                    edge_or_node = (input_act, maxpool_node)  # 设置边或节点元组
                    maxpool_node_quantization_annotation.output_qspec = (
                        SharedQuantizationSpec(edge_or_node)  # 设置输出量化规格
                    )
            else:
                input_node = node.all_input_nodes[0]  # 获取输入节点
                self._annotate_output_share_observer_as_input(input_node, node)  # 共享观察器作为输入节点的量化注释
        return  # 返回结束函数

    def _annotate_linear(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        # 获取包含线性操作的分区
        linear_partitions = get_source_partitions(
            gm.graph, [torch.nn.Linear, torch.nn.functional.linear]
        )
        # 展开并合并所有线性操作分区
        linear_partitions = list(
            itertools.chain.from_iterable(linear_partitions.values())
        )
        # 遍历每个线性操作分区
        for partition in linear_partitions:
            # 检查分区的输出节点数量是否大于1，若是则引发数值错误
            if len(partition.output_nodes) > 1:
                raise ValueError(
                    "Linear partition cannot have more than one output node"
                )
            # 获取线性节点（唯一的输出节点）
            linear_node = partition.output_nodes[0]
            # 检查线性节点是否为调用函数且目标为 torch.ops.aten.linear.default
            if linear_node.op != "call_function" or linear_node.target not in (
                torch.ops.aten.linear.default,
            ):
                raise ValueError(f"{linear_node} is not an aten linear operator")
            # 如果节点已经被标注，跳过标注过程
            if _skip_annotate([linear_node], filter_fn):
                continue
            # 对线性节点进行标注辅助操作
            self._annotate_linear_node_helper(linear_node, True, quantization_config)

    def _annotate_linear_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        # 定义包含一元操作的列表
        postop_list = [
            torch.nn.ReLU,
            torch.nn.LeakyReLU,
            torch.nn.Tanh,
            torch.nn.GELU,
        ]
        # 初始化融合分区列表
        fused_partitions: List[tuple] = []
        # 遍历每个一元操作
        for postop in postop_list:
            # 查找并融合包含线性和当前一元操作的分区
            fused_partitions = fused_partitions + find_sequential_partitions(
                gm, [torch.nn.Linear, postop]
            )
        # 遍历所有融合后的分区
        for fused_partition in fused_partitions:
            # 获取线性分区和一元分区的输出节点
            linear_partition, unary_partition = fused_partition
            linear_node, unary_node = self._get_output_nodes_of_partitions(
                [linear_partition, unary_partition]
            )
            # 如果线性节点不是调用函数或目标不是 torch.ops.aten.linear.default，则跳过
            if linear_node.op != "call_function" or linear_node.target not in (
                torch.ops.aten.linear.default,
            ):
                continue
            # 如果跳过标注过程，则继续下一轮循环
            if _skip_annotate([unary_node, linear_node], filter_fn):
                continue

            # 如果未指定量化配置，则对节点不进行量化标注并继续下一轮循环
            if quantization_config is None:
                _annotate_nodes_not_quantize([linear_node, unary_node])
                continue

            # 对线性节点进行标注辅助操作，不执行量化标注
            self._annotate_linear_node_helper(linear_node, False, quantization_config)
            # 添加一元节点的量化标注信息
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_linear_binary_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        # 这个方法还没有实现，仅有方法签名

    def validate(self, model: torch.fx.GraphModule) -> None:
        # 这是一个空方法，用于验证模型，不执行任何操作

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        # 返回支持的运算符配置列表
        return cls.supported_config_and_operators
```