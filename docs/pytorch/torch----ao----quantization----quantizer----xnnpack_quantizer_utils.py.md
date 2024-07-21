# `.\pytorch\torch\ao\quantization\quantizer\xnnpack_quantizer_utils.py`

```
# mypy: allow-untyped-defs
# 导入 itertools 模块，提供迭代器的工具函数
import itertools
# 导入 operator 模块，提供对内置操作符的函数形式访问
import operator
# 导入 dataclass 模块，用于创建不可变数据类
from dataclasses import dataclass
# 导入类型提示模块
from typing import Callable, Dict, List, NamedTuple, Optional

# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的函数库，用于神经网络相关的功能
import torch.nn.functional as F
# 导入 FakeTensor 类，用于模拟张量对象
from torch._subclasses import FakeTensor
# 导入 quantization.fx.utils 模块的函数，用于获取带有指定前缀的新属性名称
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
# 导入 quantization.pt2e.export_utils 模块的 _WrapperModule 类
from torch.ao.quantization.pt2e.export_utils import _WrapperModule
# 导入 quantization.pt2e.graph_utils 模块的函数，用于找到序列分区
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
# 导入 quantization.pt2e.utils 模块的各个函数，用于操作 Conv1d、Conv2d 等示例输入
from torch.ao.quantization.pt2e.utils import (
    _conv1d_bn_example_inputs,
    _conv2d_bn_example_inputs,
    _get_aten_graph_module_for_pattern,
    _is_conv_node,
    _is_conv_transpose_node,
)
# 导入 quantizer 模块的类，包括 QuantizationAnnotation、QuantizationSpec 等
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)
# 导入 quantizer.utils 模块的函数，用于给输入和输出的量化规格添加注释
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
# 导入 fx 模块的 Node 类，表示 FX 语法树中的节点
from torch.fx import Node
# 导入 passes.utils.matcher_with_name_node_map_utils 模块的 SubgraphMatcherWithNameNodeMap 类
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)
# 导入 passes.utils.source_matcher_utils 模块的 get_source_partitions 函数
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

# 定义 __all__ 列表，包含在当前模块中公开的所有类和函数
__all__ = [
    "OperatorConfig",
    "OperatorPatternType",
    "QuantizationConfig",
    "get_input_act_qspec",
    "get_output_act_qspec",
    "get_weight_qspec",
    "get_bias_qspec",
    "OP_TO_ANNOTATOR",
    "propagate_annotation",
]

# 数据类 QuantizationConfig，用于保存量化配置，冻结以确保不可变
@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]  # 输入激活量化规格的可选项
    output_activation: Optional[QuantizationSpec]  # 输出激活量化规格的可选项
    weight: Optional[QuantizationSpec]  # 权重量化规格的可选项
    bias: Optional[QuantizationSpec]  # 偏置量化规格的可选项
    is_qat: bool = False  # 是否为量化感知训练，默认为 False

# 定义 OperatorPatternType 类型别名，表示操作符模式的列表
OperatorPatternType = List[Callable]
# 将 OperatorPatternType 类的模块设置为 "torch.ao.quantization.quantizer.xnnpack_quantizer_utils"
OperatorPatternType.__module__ = (
    "torch.ao.quantization.quantizer.xnnpack_quantizer_utils"
)

# 定义 AnnotatorType 类型别名，表示注释器函数的类型
AnnotatorType = Callable[
    [
        torch.fx.GraphModule,
        Optional[QuantizationConfig],
        Optional[Callable[[Node], bool]],
    ],
    Optional[List[List[Node]]],
]
# 初始化空字典 OP_TO_ANNOTATOR，用于存储操作符到注释器函数的映射
OP_TO_ANNOTATOR: Dict[str, AnnotatorType] = {}


# register_annotator 装饰器函数，用于注册注释器函数到操作符的映射
def register_annotator(op: str):
    def decorator(annotator: AnnotatorType):
        OP_TO_ANNOTATOR[op] = annotator

    return decorator


# OperatorConfig 命名元组，用于存储操作符的配置信息
class OperatorConfig(NamedTuple):
    config: QuantizationConfig  # 量化配置对象的配置信息
    # 定义一个名为 operators 的变量，其类型为 List[OperatorPatternType]
    operators: List[OperatorPatternType]
def _is_annotated(nodes: List[Node]):
    """
    给定一个节点列表（表示操作符模式），检查是否有任何节点被注释，如果有则返回True，否则返回False
    """
    annotated = False
    for node in nodes:
        # 检查节点的元数据中是否包含量化注释，并且该注释已经标记为已注释
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated


def _mark_nodes_as_annotated(nodes: List[Node]):
    """
    给定一个节点列表，将这些节点标记为已注释
    """
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                # 如果节点的元数据中没有量化注释，则创建一个新的量化注释对象
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            # 将节点的量化注释标记为已注释
            node.meta["quantization_annotation"]._annotated = True


def get_input_act_qspec(quantization_config: Optional[QuantizationConfig]):
    """
    根据量化配置获取输入激活的量化规格
    """
    if quantization_config is None:
        return None
    if quantization_config.input_activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.input_activation
    assert quantization_spec.qscheme in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
    ]
    return quantization_spec


def get_output_act_qspec(quantization_config: Optional[QuantizationConfig]):
    """
    根据量化配置获取输出激活的量化规格
    """
    if quantization_config is None:
        return None
    if quantization_config.output_activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.output_activation
    assert quantization_spec.qscheme in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
    ]
    return quantization_spec


def get_weight_qspec(quantization_config: Optional[QuantizationConfig]):
    """
    根据量化配置获取权重的量化规格
    """
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.weight is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.weight
    if quantization_spec.qscheme not in [
        torch.per_tensor_symmetric,
        torch.per_channel_symmetric,
    ]:
        raise ValueError(
            f"Unsupported quantization_spec {quantization_spec} for weight"
        )
    return quantization_spec


def get_bias_qspec(quantization_config: Optional[QuantizationConfig]):
    """
    根据量化配置获取偏置的量化规格
    """
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.bias is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.bias
    assert (
        quantization_spec.dtype == torch.float
    ), "Only float dtype for bias is supported for bias right now"
    return quantization_spec


@register_annotator("linear")
def _annotate_linear(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None
) -> Optional[List[List[Node]]]:
    """
    注册的注释器函数，用于线性层的量化注释
    """
    annotated_partitions = []
    input_act_qspec = get_input_act_qspec(quantization_config)
    # 获取输出激活量化规范
    output_act_qspec = get_output_act_qspec(quantization_config)
    # 获取权重量化规范
    weight_qspec = get_weight_qspec(quantization_config)
    # 获取偏置量化规范
    bias_qspec = get_bias_qspec(quantization_config)
    # 遍历计算图中的节点
    for node in gm.graph.nodes:
        # 如果节点不是调用函数或者目标不是 torch.ops.aten.linear.default，则跳过
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue
        # 如果存在过滤函数并且节点不符合过滤条件，则跳过
        if filter_fn and not filter_fn(node):
            continue
        # 获取激活节点和权重节点
        act_node = node.args[0]
        weight_node = node.args[1]
        bias_node = None
        # 如果节点参数超过两个，则获取偏置节点
        if len(node.args) > 2:
            bias_node = node.args[2]

        # 如果节点没有被标记为注解的话
        if _is_annotated([node]) is False:  # type: ignore[list-item]
            # 为节点、激活节点、权重节点添加输入量化规范的注解
            _annotate_input_qspec_map(
                node,
                act_node,
                input_act_qspec,
            )
            _annotate_input_qspec_map(
                node,
                weight_node,
                weight_qspec,
            )
            nodes_to_mark_annotated = [node, weight_node]
            # 如果存在偏置节点，则为其添加输入量化规范的注解
            if bias_node:
                _annotate_input_qspec_map(
                    node,
                    bias_node,
                    bias_qspec,
                )
                nodes_to_mark_annotated.append(bias_node)
            # 为节点添加输出激活量化规范的注解
            _annotate_output_qspec(node, output_act_qspec)
            # 标记节点及其相关节点为已注解
            _mark_nodes_as_annotated(nodes_to_mark_annotated)
            # 将已标记为注解的节点列表添加到注解分区列表中
            annotated_partitions.append(nodes_to_mark_annotated)

    # 返回所有注解过的节点分区列表
    return annotated_partitions
@register_annotator("linear_relu")
def _annotate_linear_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    # 初始化空列表，用于存储已注释的分区
    annotated_partitions = []
    
    # 获取输入激活量化规范
    input_act_qspec = get_input_act_qspec(quantization_config)
    # 获取输出激活量化规范
    output_act_qspec = get_output_act_qspec(quantization_config)
    # 获取权重量化规范
    weight_qspec = get_weight_qspec(quantization_config)
    # 获取偏置量化规范
    bias_qspec = get_bias_qspec(quantization_config)
    
    # 遍历图中的每个节点
    for node in gm.graph.nodes:
        # 如果节点不是调用函数或者函数不是 torch.relu 或 torch.relu_，则继续下一个节点
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]:
            continue
        
        # 获取当前节点作为 relu_node
        relu_node = node
        # 获取作为线性函数调用的节点
        maybe_linear_node = node.args[0]
        
        # 如果节点不是 Node 类型，或者不是调用函数，或者函数不是 torch.linear.default，则继续下一个节点
        if (
            not isinstance(maybe_linear_node, Node)
            or maybe_linear_node.op != "call_function"
            or maybe_linear_node.target != torch.ops.aten.linear.default
        ):
            continue
        
        # 获取线性节点
        linear_node = maybe_linear_node
        
        # 初始化输入量化规范映射的字典
        input_qspec_map = {}
        
        # 获取线性节点的输入激活
        input_act = linear_node.args[0]
        assert isinstance(input_act, Node)
        # 将输入激活和其对应的量化规范加入映射字典
        input_qspec_map[input_act] = input_act_qspec
        
        # 获取权重节点
        weight = linear_node.args[1]
        assert isinstance(weight, Node)
        # 将权重和其对应的量化规范加入映射字典
        input_qspec_map[weight] = weight_qspec
        
        # 将权重节点也加入分区
        partition = [relu_node, linear_node, weight]
        
        # 如果线性节点的参数大于2个，则获取偏置
        bias = linear_node.args[2] if len(linear_node.args) > 2 else None
        if isinstance(bias, Node):
            # 将偏置及其对应的量化规范加入映射字典，并将偏置节点加入分区
            input_qspec_map[bias] = bias_qspec
            partition.append(bias)
        
        # 如果分区已经被注释，则继续下一个节点
        if _is_annotated(partition):
            continue
        
        # 如果有过滤函数，并且分区中有节点不符合过滤函数的条件，则继续下一个节点
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue
        
        # 给线性节点和 relu 节点添加量化注释
        linear_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        
        # 标记分区中的节点为已注释
        _mark_nodes_as_annotated(partition)
        
        # 将已注释的分区添加到注释分区列表中
        annotated_partitions.append(partition)
    
    # 返回已注释的分区列表
    return annotated_partitions
    # 遍历计算图中的节点
    for n in gm.graph.nodes:
        # 如果节点不是调用函数操作或者目标不在预定义的 torch 操作中，则跳过该节点
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
        ]:
            continue
        # 将满足条件的节点作为卷积节点进行处理
        conv_node = n

        # 初始化输入量化规格映射字典
        input_qspec_map = {}

        # 获取卷积操作的输入激活节点，并确保其为 Node 类型
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        # 获取输入激活节点的量化规格，并添加到输入量化规格映射字典中
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

        # 获取卷积操作的权重节点，并确保其为 Node 类型
        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        # 获取权重节点的量化规格，并添加到输入量化规格映射字典中
        input_qspec_map[weight] = get_weight_qspec(quantization_config)

        # 将权重节点也加入到分区中
        partition = [conv_node, conv_node.args[1]]

        # 如果卷积操作存在偏置节点，且偏置节点为 Node 类型，则获取其量化规格并添加到输入量化规格映射字典中
        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
            partition.append(bias)

        # 如果分区已经被标记为注释过的，则跳过该分区
        if _is_annotated(partition):
            continue

        # 如果存在过滤函数，并且分区中任一节点不满足过滤条件，则跳过该分区
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        # 将卷积节点的元数据中添加量化注释对象
        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )
        # 将该分区内的所有节点标记为已注释
        _mark_nodes_as_annotated(partition)
        # 将已注释的分区添加到结果列表中
        annotated_partitions.append(partition)
    
    # 返回所有已注释的分区列表
    return annotated_partitions
# 定义一个用于注释卷积层和ReLU激活函数的函数
def _do_annotate_conv_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
    is_conv_transpose: bool = False,
):
    # 存储已注释的分区列表
    annotated_partitions = []
    
    # 遍历计算图中的每个节点
    for n in gm.graph.nodes:
        # 如果节点不是函数调用或者目标函数不是标准的ReLU函数，则继续下一个节点
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]:
            continue
        
        # 获取ReLU节点
        relu_node = n
        # 获取可能的卷积节点
        maybe_conv_node = n.args[0]
        
        # 根据是否是转置卷积决定判断卷积节点的函数
        is_conv_node = _is_conv_transpose_node if is_conv_transpose else _is_conv_node
        # 如果卷积节点不是Node对象或者不满足卷积节点的判断条件，则继续下一个节点
        if not isinstance(maybe_conv_node, Node) or not is_conv_node(maybe_conv_node):
            continue
        
        # 获取卷积节点
        conv_node = maybe_conv_node
        
        # 初始化输入量化规格映射字典
        input_qspec_map = {}
        
        # 获取卷积操作的输入激活节点，并添加到输入量化规格映射中
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)
        
        # 获取卷积操作的权重节点，并添加到输入量化规格映射中
        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)
        
        # 如果卷积操作有偏置，则添加偏置节点到分区列表，并将其添加到输入量化规格映射中
        partition = [relu_node, conv_node, conv_node.args[1]]
        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
            partition.append(bias)
        
        # 如果分区已经被注释过，则继续下一个节点
        if _is_annotated(partition):
            continue
        
        # 如果存在过滤函数，并且分区中有节点不符合过滤函数的条件，则继续下一个节点
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue
        
        # 为卷积节点和ReLU节点添加量化注释信息
        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, _annotated=True
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
        
        # 将分区节点标记为已注释
        _mark_nodes_as_annotated(partition)
        
        # 将已注释的分区添加到注释分区列表中
        annotated_partitions.append(partition)
    
    # 返回所有已注释的分区列表
    return annotated_partitions


# 注册函数，为卷积层和ReLU激活函数添加量化注释
@register_annotator("conv_relu")
def _annotate_conv_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    return _do_annotate_conv_relu(
        gm, quantization_config, filter_fn, is_conv_transpose=False
    )


# 注册函数，为转置卷积层和ReLU激活函数添加量化注释
@register_annotator("conv_transpose_relu")
def _annotate_conv_transpose_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    return _do_annotate_conv_relu(
        gm, quantization_config, filter_fn, is_conv_transpose=True
    )


# 注册函数，为卷积层和批归一化添加量化注释
@register_annotator("conv_bn")
def _annotate_conv_bn(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """
    """
    # 寻找卷积层和批量归一化层的边界
    # 注意：这仅用于量化感知训练（QAT）。在量化感知训练之外（PTQ），批量归一化应已融合到卷积层中。
    """
    调用函数_do_annotate_conv_bn，传递给定的图模型（gm）、量化配置（quantization_config）、过滤函数（filter_fn），
    并指定是否有ReLU激活（has_relu=False）。
    """
    return _do_annotate_conv_bn(gm, quantization_config, filter_fn, has_relu=False)
@register_annotator("conv_bn_relu")
# 注册一个名为"conv_bn_relu"的分析器函数装饰器，用于识别 conv + batchnorm + relu 模式
def _annotate_conv_bn_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """
    Find conv + batchnorm + relu parititions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    # 找到 conv + batchnorm + relu 模式的分区
    return _do_annotate_conv_bn(gm, quantization_config, filter_fn, has_relu=True)


@register_annotator("conv_transpose_bn")
# 注册一个名为"conv_transpose_bn"的分析器函数装饰器，用于识别 conv_transpose + batchnorm 模式
def _annotate_conv_transpose_bn(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """
    Find conv_transpose + batchnorm parititions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    # 找到 conv_transpose + batchnorm 模式的分区
    return _do_annotate_conv_bn(
        gm, quantization_config, filter_fn, has_relu=False, is_conv_transpose=True
    )


@register_annotator("conv_transpose_bn_relu")
# 注册一个名为"conv_transpose_bn_relu"的分析器函数装饰器，用于识别 conv_transpose + batchnorm + relu 模式
def _annotate_conv_transpose_bn_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """
    Find conv_transpose + batchnorm + relu parititions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    # 找到 conv_transpose + batchnorm + relu 模式的分区
    return _do_annotate_conv_bn(
        gm, quantization_config, filter_fn, has_relu=True, is_conv_transpose=True
    )


def _do_annotate_conv_bn(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]],
    has_relu: bool,
    is_conv_transpose: bool = False,
) -> List[List[Node]]:
    """
    Given a function that takes in a `conv_fn` and returns a conv-bn[-relu] pattern,
    return a list of annotated partitions.

    The output of the pattern must include a dictionary from string name to node
    for the following names: "input", "conv", "weight", "bias", and "output".
    """

    def get_pattern(conv_fn: Callable, relu_is_inplace: bool):
        def _conv_bn(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_rm, bn_rv):
            conv = conv_fn(x, conv_weight, conv_bias)
            bn = F.batch_norm(conv, bn_rm, bn_rv, bn_weight, bn_bias, training=True)
            if has_relu:
                output = F.relu_(bn) if relu_is_inplace else F.relu(bn)
            else:
                output = bn
            return output, {
                "input": x,
                "conv": conv,
                "weight": conv_weight,
                "bias": conv_bias,
                "output": output,
            }

        return _WrapperModule(_conv_bn)

    # 需要进行匹配，否则由于未使用的节点被批量归一化返回而被过滤掉
    gm.graph.eliminate_dead_code()
    gm.recompile()

    # 初始化匹配结果列表
    matches = []
    # 如果是转置卷积操作，定义不同的操作组合
    if is_conv_transpose:
        combinations = [
            (F.conv_transpose1d, _conv1d_bn_example_inputs),
            (F.conv_transpose2d, _conv2d_bn_example_inputs),
        ]
    else:
        # 如果不是转置卷积操作，定义另一组操作组合
        combinations = [
            (F.conv1d, _conv1d_bn_example_inputs),  # type: ignore[list-item]
            (F.conv2d, _conv2d_bn_example_inputs),  # type: ignore[list-item]
        ]

    # 添加 `is_cuda` 和 `relu_is_inplace` 维度的组合
    combinations = itertools.product(  # type: ignore[assignment]
        combinations,
        [True, False] if torch.cuda.is_available() else [False],  # is_cuda
        [True, False] if has_relu else [False],  # relu_is_inplace
    )

    # 遍历所有卷积函数和其变种（是否CUDA加速，是否inplace relu）
    for (conv_fn, example_inputs), is_cuda, relu_is_inplace in combinations:  # type: ignore[misc]
        # 获取当前组合的模式
        pattern = get_pattern(conv_fn, relu_is_inplace)  # type: ignore[has-type]
        # 根据模式、输入示例和CUDA标记获取模式的ATen图模块
        pattern = _get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda)  # type: ignore[has-type]
        # 清除模式中的死代码
        pattern.graph.eliminate_dead_code()
        # 重新编译模式
        pattern.recompile()
        # 创建模式匹配器，并忽略文字字面量
        matcher = SubgraphMatcherWithNameNodeMap(pattern, ignore_literals=True)
        # 将匹配结果扩展到总匹配列表中
        matches.extend(matcher.match(gm.graph))

    # 标注返回匹配结果中的节点分区
    annotated_partitions = []
    # 遍历匹配列表中的每一个匹配对象
    for match in matches:
        # 从当前匹配对象中获取名称到节点的映射
        name_node_map = match.name_node_map
        # 获取匹配对象中标记为"input"的节点
        input_node = name_node_map["input"]
        # 获取匹配对象中标记为"conv"的节点
        conv_node = name_node_map["conv"]
        # 获取匹配对象中标记为"weight"的节点
        weight_node = name_node_map["weight"]
        # 获取匹配对象中标记为"bias"的节点
        bias_node = name_node_map["bias"]
        # 获取匹配对象中标记为"output"的节点
        output_node = name_node_map["output"]

        # TODO: annotate the uses of input, weight, and bias separately instead
        # of assuming they come from a single conv node. This is not possible today
        # because input may have multiple users, and we can't rely on the conv node
        # always being the first user. This was the case in models with skip
        # connections like resnet18

        # 验证卷积层的参数是否正确设置
        if conv_node.args[0] is not input_node:
            raise ValueError("Conv arg did not contain input node ", input_node)
        if conv_node.args[1] is not weight_node:
            raise ValueError("Conv arg did not contain weight node ", weight_node)
        if len(conv_node.args) > 2 and conv_node.args[2] is not bias_node:
            raise ValueError("Conv arg did not contain bias node ", bias_node)

        # 如果分区已经被注释或者被用户过滤掉，则跳过处理
        partition = [conv_node, weight_node]
        if bias_node is not None:
            partition.append(bias_node)
        if _is_annotated(partition):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        # 为卷积层的输入和模式输出添加注释
        input_qspec_map = {}
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        if bias_node is not None:
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        
        # 为卷积节点添加量化注释
        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        # 为输出节点添加量化注释
        output_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
        # 标记节点已经被注释
        _mark_nodes_as_annotated(partition)
        # 将已注释的分区添加到列表中
        annotated_partitions.append(partition)
    
    # 返回所有已注释的分区列表
    return annotated_partitions
@register_annotator("gru_io_only")
# 注册一个名为 "gru_io_only" 的函数装饰器，用于将函数 _annotate_gru_io_only 注册为一个注解器
def _annotate_gru_io_only(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    # 获取所有包含 GRU 模块的源代码分区
    gru_partitions = get_source_partitions(gm.graph, [torch.nn.GRU], filter_fn)
    # 将分区列表展开成一个一维列表
    gru_partitions = list(itertools.chain.from_iterable(gru_partitions.values()))
    # 存储已注解的分区列表
    annotated_partitions = []
    # 遍历每个 GRU 分区
    for gru_partition in gru_partitions:
        # 将当前分区的所有节点添加到注解分区列表中
        annotated_partitions.append(gru_partition.nodes)
        # 获取当前分区的输出节点和输入节点
        output_nodes = gru_partition.output_nodes
        input_nodes = gru_partition.input_nodes
        # 如果输入节点和输出节点已经被注解，则跳过注解
        if _is_annotated(input_nodes + output_nodes):
            continue
        # 获取输入激活节点和对应的用户节点，并为用户节点添加量化注解
        input_qspec_map: Dict[Node, QuantizationSpecBase] = {}
        input_act = input_nodes[0]
        input_act_user = next(iter(input_act.users.keys()))
        assert isinstance(input_act, Node)
        assert isinstance(input_act_user, Node)
        input_act_user.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                input_act: get_input_act_qspec(quantization_config),
            },
            _annotated=True,
        )
        # 获取隐藏状态节点和对应的用户节点，并为用户节点添加量化注解
        hidden_state = input_nodes[1]
        hidden_state_user = next(iter(hidden_state.users.keys()))
        assert isinstance(hidden_state, Node)
        assert isinstance(hidden_state_user, Node)
        hidden_state_user.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                hidden_state: get_input_act_qspec(quantization_config),
            },
            _annotated=True,
        )
        # 确保输出节点数为 2，因为 GRU 应有两个输出
        assert len(output_nodes) == 2, "expecting GRU to have two outputs"
        # 为每个输出节点添加量化注解
        for output in output_nodes:
            output.meta["quantization_annotation"] = QuantizationAnnotation(
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
        # 将当前分区的所有节点标记为已注解
        nodes_to_mark_annotated = list(gru_partition.nodes)
        _mark_nodes_as_annotated(nodes_to_mark_annotated)
    # 返回已注解的分区列表
    return annotated_partitions


@register_annotator("max_pool2d")
# 注册一个名为 "max_pool2d" 的函数装饰器，用于将函数 _annotate_max_pool2d 注册为一个注解器
def _annotate_max_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    # 获取所有包含 MaxPool2d 或 torch.nn.functional.max_pool2d 模块的源代码分区
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d], filter_fn
    )
    # 将分区列表展开成一个一维列表
    maxpool_partitions = list(itertools.chain.from_iterable(module_partitions.values()))
    # 存储已注解的分区列表
    annotated_partitions = []
    # 对于每个 maxpool_partition 中的分区，将其节点添加到 annotated_partitions 中
    for maxpool_partition in maxpool_partitions:
        annotated_partitions.append(maxpool_partition.nodes)
        # 取分区的输出节点作为 output_node
        output_node = maxpool_partition.output_nodes[0]
        maxpool_node = None
        # 查找分区中的 max_pool2d 默认节点
        for n in maxpool_partition.nodes:
            if n.target == torch.ops.aten.max_pool2d.default:
                maxpool_node = n
        # 确保找到了 max_pool2d 默认节点，否则抛出异常
        assert (
            maxpool_node is not None
        ), "XNNPACKQuantizer only works with torch.ops.aten.max_pool2d.default, " \
           "please make sure you are exporting the model correctly"
        # 如果 output_node 和 maxpool_node 已经被注释过，则跳过当前循环
        if _is_annotated([output_node, maxpool_node]):  # type: ignore[list-item]
            continue

        # 获取 maxpool_node 的输入激活值节点
        input_act = maxpool_node.args[0]  # type: ignore[union-attr]
        assert isinstance(input_act, Node)

        # 只有当输入节点的量化注释未被标记，或者其未被注释，或者其输出量化规范为空时，才跳过当前循环
        if (
            "quantization_annotation" not in input_act.meta
            or not input_act.meta["quantization_annotation"]._annotated
            or input_act.meta["quantization_annotation"].output_qspec is None
        ):
            continue

        # 使用输入节点的量化规范创建共享量化规范
        act_qspec = SharedQuantizationSpec(input_act)
        # 为 maxpool_node 添加量化注释
        maxpool_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
            input_qspec_map={
                input_act: act_qspec,
            },
            _annotated=True,
        )
        # 为 output_node 添加量化注释
        output_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=act_qspec,
            _annotated=True,
        )
    # 返回已注释的分区列表
    return annotated_partitions
# 注册自定义的注释器函数，用于 adaptive_avg_pool2d 操作符
@register_annotator("adaptive_avg_pool2d")
def _annotate_adaptive_avg_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """始终注释 adaptive_avg_pool2d 操作"""
    
    # 获取源代码中包含 adaptive_avg_pool2d 操作的模块分区
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d], filter_fn
    )
    
    # 将分区列表展开成单个分区的列表
    partitions = list(itertools.chain.from_iterable(module_partitions.values()))
    annotated_partitions = []
    
    # 遍历每个分区
    for partition in partitions:
        # 获取分区的输出节点
        pool_node = partition.output_nodes[0]
        
        # 检查节点是否为 call_function 操作，并且目标是 torch.ops.aten.adaptive_avg_pool2d.default
        if (
            pool_node.op != "call_function"
            or pool_node.target != torch.ops.aten.adaptive_avg_pool2d.default
        ):
            raise ValueError(f"{pool_node} 不是 aten 的 adaptive_avg_pool2d 操作符")
        
        # 如果节点已经被注释过，则继续下一个分区
        if _is_annotated([pool_node]):
            continue
        
        # 将当前分区的节点列表添加到已注释分区的列表中
        annotated_partitions.append(partition.nodes)
        
        # 获取 adaptive_avg_pool2d 操作的输入节点
        input_act = pool_node.args[0]
        assert isinstance(input_act, Node)
        
        # 仅当输入节点的输出已经被注释时，才注释输入输出共享操作符
        if (
            "quantization_annotation" not in input_act.meta
            or not input_act.meta["quantization_annotation"]._annotated
            or input_act.meta["quantization_annotation"].output_qspec is None
        ):
            input_act_qspec = get_input_act_qspec(quantization_config)
        else:
            input_act_qspec = SharedQuantizationSpec(input_act)
        
        # 创建输出与输入共享的量化规范
        output_act_qspec = SharedQuantizationSpec((input_act, pool_node))
        
        # 将量化注释信息添加到 pool_node 的元数据中
        pool_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                input_act: input_act_qspec,
            },
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    
    # 返回已注释分区的列表
    return annotated_partitions


def _is_input_large_scalar(node: Node, gm: torch.fx.GraphModule):
    """检查输入是否为大标量值。这样我们可以跳过节点的量化，因为 HistogramObserver 中的 histc 操作仅适用于特定上限以下的值"""
    if node.op == "get_attr":
        tensor = getattr(gm, node.target)  # type: ignore[arg-type]
        # torch.histc 的工作上限值
        HISTC_UPPER_BOUND = 3.4028235e15
        return tensor.numel() == 1 and abs(tensor.item()) > HISTC_UPPER_BOUND
    return False


def _is_input_non_float_tensor(node: Node):
    """检查输入是否为非浮点数张量，以便我们可以跳过节点的量化，因为观察器仅适用于浮点张量"""
    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return True
    return node.meta["val"].dtype != torch.float32


# 注册自定义的注释器函数，用于 add_relu 操作符
@register_annotator("add_relu")
def _annotate_add_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    # 定义一个参数filter_fn，类型为Optional[Callable[[Node], bool]]，默认为None
    filter_fn: Optional[Callable[[Node], bool]] = None,
# 注册一个函数注解器，用于对加法操作进行注解
@register_annotator("add")
# 定义一个函数 _annotate_add，接受以下参数
def _annotate_add(
    gm: torch.fx.GraphModule,  # 输入一个 Torch 的图形模块
    quantization_config: Optional[QuantizationConfig],  # 可选的量化配置参数
    filter_fn: Optional[Callable[[Node], bool]] = None,  # 可选的过滤函数，用于过滤节点
) -> Optional[List[List[Node]]]:  # 返回一个可选的节点列表的列表
    # 获取所有源分区，这些分区包含加法操作的分区
    add_partitions = get_source_partitions(
        gm.graph, [operator.add, torch.add, operator.iadd], filter_fn
    )
    # 展开所有分区并放入同一个列表中
    add_partitions = list(itertools.chain.from_iterable(add_partitions.values()))
    # 初始化一个空列表，用于存储已注解的分区
    annotated_partitions = []
    # 遍历需要添加的分区列表中的每一个分区
    for add_partition in add_partitions:
        # 将当前分区的节点列表添加到已注释分区列表中
        annotated_partitions.append(add_partition.nodes)
        # 获取当前分区的输出节点
        add_node = add_partition.output_nodes[0]
        # 如果已经对当前节点进行了注释，则跳过
        if _is_annotated([add_node]):
            continue

        # 根据量化配置获取输入激活量化规格
        input_act_qspec = get_input_act_qspec(quantization_config)
        # 根据量化配置获取输出激活量化规格
        output_act_qspec = get_output_act_qspec(quantization_config)

        # 初始化空的输入量化规格映射
        input_qspec_map = {}

        # 处理第一个输入节点
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            # 如果第一个输入是大型标量，则跳过
            if _is_input_large_scalar(input_act0, gm):
                continue
            # 如果第一个输入是非浮点数张量，则跳过
            if _is_input_non_float_tensor(input_act0):
                continue
            # 将第一个输入节点和其对应的激活量化规格添加到映射中
            input_qspec_map[input_act0] = input_act_qspec

        # 处理第二个输入节点
        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            # 如果第二个输入是大型标量，则跳过
            if _is_input_large_scalar(input_act1, gm):
                continue
            # 如果第二个输入是非浮点数张量，则跳过
            if _is_input_non_float_tensor(input_act1):
                continue
            # 将第二个输入节点和其对应的激活量化规格添加到映射中
            input_qspec_map[input_act1] = input_act_qspec

        # 给加法节点添加量化注释元数据
        add_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )

    # 返回已注释分区的列表
    return annotated_partitions
# 注册一个名为 "mul_relu" 的函数装饰器，用于对 Torch 的图模块进行操作
@register_annotator("mul_relu")
# 定义了一个名为 _annotate_mul_relu 的函数，接受以下参数和返回类型
def _annotate_mul_relu(
    gm: torch.fx.GraphModule,  # Torch 的图模块，包含了计算图和模型结构
    quantization_config: Optional[QuantizationConfig],  # 可选的量化配置对象
    filter_fn: Optional[Callable[[Node], bool]] = None,  # 可选的节点过滤函数
) -> Optional[List[List[Node]]]:  # 返回一个可选的节点列表的列表（用于注释的分区）

    # 查找包含顺序的分区，即连续的乘法和 ReLU 操作
    fused_partitions = find_sequential_partitions(
        gm, [torch.mul, torch.nn.ReLU], filter_fn=filter_fn
    )
    
    # 初始化一个空的注释分区列表
    annotated_partitions = []

    # 遍历每个找到的融合分区
    for fused_partition in fused_partitions:
        mul_partition, relu_partition = fused_partition  # 解包乘法和 ReLU 分区

        # 将乘法分区和 ReLU 分区的节点列表合并到一个列表中，并添加到注释分区列表中
        annotated_partitions.append(mul_partition.nodes + relu_partition.nodes)

        # 如果 ReLU 分区的输出节点数量超过一个，则抛出数值错误
        if len(relu_partition.output_nodes) > 1:
            raise ValueError("Relu partition has more than one output node")
        
        # 获取 ReLU 分区的输出节点
        relu_node = relu_partition.output_nodes[0]

        # 如果乘法分区的输出节点数量超过一个，则抛出数值错误
        if len(mul_partition.output_nodes) > 1:
            raise ValueError("mul partition has more than one output node")
        
        # 获取乘法分区的输出节点
        mul_node = mul_partition.output_nodes[0]

        # 如果这两个节点已经被注释过了，则继续下一个分区的处理
        if _is_annotated([relu_node, mul_node]):
            continue
        
        # 获取输入激活量化规范
        input_act_qspec = get_input_act_qspec(quantization_config)
        # 获取输出激活量化规范
        output_act_qspec = get_output_act_qspec(quantization_config)

        # 初始化输入量化规范映射
        input_qspec_map = {}

        # 处理乘法节点的第一个参数
        input_act0 = mul_node.args[0]
        if isinstance(input_act0, Node):
            # 如果第一个参数是一个节点且满足特定条件，则跳过当前分区的处理
            if _is_input_large_scalar(input_act0, gm):
                continue
            if _is_input_non_float_tensor(input_act0):
                continue
            # 将第一个参数及其量化规范添加到输入量化规范映射中
            input_qspec_map[input_act0] = input_act_qspec

        # 处理乘法节点的第二个参数
        input_act1 = mul_node.args[1]
        if isinstance(input_act1, Node):
            # 如果第二个参数是一个节点且满足特定条件，则跳过当前分区的处理
            if _is_input_large_scalar(input_act1, gm):
                continue
            if _is_input_non_float_tensor(input_act1):
                continue
            # 将第二个参数及其量化规范添加到输入量化规范映射中
            input_qspec_map[input_act1] = input_act_qspec

        # 为乘法节点添加量化注释元数据
        mul_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )

        # 为 ReLU 节点添加量化注释元数据
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )

    # 返回注释后的分区列表
    return annotated_partitions


# 注册一个名为 "mul" 的函数装饰器
@register_annotator("mul")
# 定义了一个名为 _annotate_mul 的函数，接受以下参数和返回类型
def _annotate_mul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:  # 返回一个可选的节点列表的列表（用于注释的分区）

    # 获取所有包含乘法操作的源分区
    mul_partitions = get_source_partitions(
        gm.graph, ["mul", "mul_", operator.mul, torch.mul, operator.imul], filter_fn
    )
    
    # 将字典值中的所有分区列表展开成一个列表
    mul_partitions = list(itertools.chain.from_iterable(mul_partitions.values()))
    
    # 初始化一个空的注释分区列表
    annotated_partitions = []
    # 对于每一个多重分区中的分区，将其节点添加到已注释的分区列表中
    annotated_partitions.append(mul_partition.nodes)
    # 取出当前多重分区的输出节点
    mul_node = mul_partition.output_nodes[0]
    # 如果该节点已经被注释过，则跳过继续处理下一个节点
    if _is_annotated([mul_node]):
        continue

    # 获取输入激活量化规范
    input_act_qspec = get_input_act_qspec(quantization_config)
    # 获取输出激活量化规范
    output_act_qspec = get_output_act_qspec(quantization_config)

    # 初始化输入量化规范映射表
    input_qspec_map = {}

    # 处理第一个输入参数
    input_act0 = mul_node.args[0]
    if isinstance(input_act0, Node):
        # 如果第一个输入参数是节点，并且是大标量输入，则跳过处理下一个节点
        if _is_input_large_scalar(input_act0, gm):
            continue
        # 如果第一个输入参数是非浮点张量，则跳过处理下一个节点
        if _is_input_non_float_tensor(input_act0):
            continue
        # 将第一个输入参数与其激活量化规范关联并添加到映射表中
        input_qspec_map[input_act0] = input_act_qspec

    # 处理第二个输入参数
    input_act1 = mul_node.args[1]
    if isinstance(input_act1, Node):
        # 如果第二个输入参数是节点，并且是大标量输入，则跳过处理下一个节点
        if _is_input_large_scalar(input_act1, gm):
            continue
        # 如果第二个输入参数是非浮点张量，则跳过处理下一个节点
        if _is_input_non_float_tensor(input_act1):
            continue
        # 将第二个输入参数与其激活量化规范关联并添加到映射表中
        input_qspec_map[input_act1] = input_act_qspec

    # 给当前乘法节点添加量化注释元数据
    mul_node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_act_qspec,
        _annotated=True,
    )

# 返回已注释的分区列表
return annotated_partitions
@register_annotator("cat")
# 注册一个用于“cat”操作的注解器函数
def _annotate_cat(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    # 获取所有包含 torch.cat 操作的源代码分区
    cat_partitions = get_source_partitions(gm.graph, [torch.cat], filter_fn)
    # 展开分区字典的值，得到所有的分区列表
    cat_partitions = list(itertools.chain.from_iterable(cat_partitions.values()))
    # 初始化一个空列表，用于存储已注解的分区
    annotated_partitions = []
    # 遍历每个 cat 分区
    for cat_partition in cat_partitions:
        # 获取当前分区的输出节点
        cat_node = cat_partition.output_nodes[0]
        # 如果输出节点已经被注解过，则跳过
        if _is_annotated([cat_node]):
            continue

        # 检查节点的目标是否为 torch.ops.aten.cat.default
        if cat_node.target != torch.ops.aten.cat.default:
            # 抛出异常，要求节点的目标应为 torch.ops.aten.cat.default
            raise Exception(
                f"Expected cat node: torch.ops.aten.cat.default, but found {cat_node.target}"
                " please check if you are calling the correct capture API"
            )

        # 将当前分区的节点列表添加到已注解分区的列表中
        annotated_partitions.append(cat_partition.nodes)

        # 获取输入激活量化规格
        input_act_qspec = get_input_act_qspec(quantization_config)
        # 获取 cat 操作的输入节点
        inputs = cat_node.args[0]

        # 初始化输入量化规格映射
        input_qspec_map = {}
        input_act0 = inputs[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec

        # 创建共享的输入量化规格对象
        shared_with_input0_qspec = SharedQuantizationSpec((input_act0, cat_node))
        # 遍历其余输入节点，将它们添加到量化规格映射中
        for input_act in inputs[1:]:
            input_qspec_map[input_act] = shared_with_input0_qspec

        # 设置输出激活量化规格
        output_act_qspec = shared_with_input0_qspec

        # 将量化注解信息添加到节点的元数据中
        cat_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    # 返回已注解的分区列表
    return annotated_partitions


def _is_share_obs_or_fq_op(op: Callable) -> bool:
    # 检查操作是否为共享观察或量化操作
    return op in [
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.mean.default,
        torch.ops.aten.mean.dim,
        torch.ops.aten.permute.default,
        torch.ops.aten.permute_copy.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze_copy.dim,
        # 可能需要移除的操作
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten.view.default,
        torch.ops.aten.slice_copy.Tensor,
        torch.ops.aten.flatten.using_ints,
    ]


def propagate_annotation(model: torch.fx.GraphModule) -> None:
    # 这个函数的具体作用未在提供的代码段中展示
    pass
    # 遍历模型图中的节点
    for n in model.graph.nodes:
        # 检查节点操作是否为函数调用，并且目标函数是否为共享观察或量化操作
        if n.op != "call_function" or not _is_share_obs_or_fq_op(n.target):
            continue
        
        # 获取当前节点的第一个参数作为前一个节点
        prev_node = n.args[0]
        # 如果前一个节点不是 Node 类型，则跳过当前节点
        if not isinstance(prev_node, Node):
            continue
        
        # 获取前一个节点的量化注释信息
        quantization_annotation = prev_node.meta.get("quantization_annotation", None)
        # 如果没有量化注释信息，则跳过当前节点
        if not quantization_annotation:
            continue
        
        # 获取前一个节点的输出量化规格
        output_qspec = quantization_annotation.output_qspec
        # 如果输出量化规格为空，则跳过当前节点
        if not output_qspec:
            continue
        
        # 确保当前节点没有被标记过量化注释
        if (
            "quantization_annotation" in n.meta
            and n.meta["quantization_annotation"]._annotated
        ):
            continue
        
        # 创建共享的量化规格对象，基于前一个节点的信息
        shared_qspec = SharedQuantizationSpec(prev_node)
        # 将前一个节点的输出量化规格传播到当前节点
        n.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                prev_node: shared_qspec,
            },
            output_qspec=shared_qspec,
            _annotated=True,
        )
# TODO: make the list of ops customizable
def _convert_scalars_to_attrs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # 遍历图中的每个节点
    for n in model.graph.nodes:
        # 如果节点不是函数调用或者目标不是指定的操作，则继续下一个节点
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
        ]:
            continue
        
        args = list(n.args)
        new_args = []
        # 遍历节点的参数列表
        for i in range(len(args)):
            if isinstance(args[i], torch.fx.Node):
                # 如果参数是一个节点对象，则直接添加到新参数列表中
                new_args.append(args[i])
                continue
            
            # 创建新的属性名前缀
            prefix = "_tensor_constant_"
            # 获取带有指定前缀的新属性名的函数
            get_new_attr_name = get_new_attr_name_with_prefix(prefix)
            # 生成新的属性名并注册为模型的缓冲区
            tensor_constant_name = get_new_attr_name(model)
            float_tensor = torch.tensor(float(args[i]))
            model.register_buffer(tensor_constant_name, float_tensor)
            
            # 获取当前节点的虚拟模式信息
            fake_mode = n.meta["val"].fake_mode
            # 在当前节点之前插入一个新节点，获取刚注册的属性值
            with model.graph.inserting_before(n):
                get_attr_node = model.graph.create_node(
                    "get_attr", tensor_constant_name, (), {}
                )
                # 将虚拟模式信息绑定到新节点的元数据中
                get_attr_node.meta["val"] = fake_mode.from_tensor(
                    float_tensor, static_shapes=True
                )
                # 将新节点添加到新参数列表中
                new_args.append(get_attr_node)
        
        # 更新当前节点的参数为新参数列表
        n.args = tuple(new_args)
    
    # 重新编译模型，以便应用变更
    model.recompile()
    # 返回更新后的模型
    return model
```