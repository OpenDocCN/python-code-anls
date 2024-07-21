# `.\pytorch\torch\ao\quantization\pt2e\port_metadata_pass.py`

```py
# 添加类型提示和声明允许未类型化的函数定义
# 导入日志模块
import logging
# 导入类型提示 Optional
from typing import Optional

# 导入 PyTorch 模块
import torch
# 导入内部错误异常
from torch._export.error import InternalError

# 导入量化工具函数
from torch.ao.quantization.pt2e.utils import (
    _filter_sym_size_users,
    _find_q_dq_node_for_user,
    _is_valid_annotation,
)

# 导入量化规范基类
from torch.ao.quantization.quantizer import QuantizationSpecBase

# 导入 FX 框架基础通行证和通行证结果
from torch.fx.passes.infra.pass_base import PassBase, PassResult

# 设置日志记录器的名称和日志级别为错误
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# 导出的模块成员变量列表
__all__ = ["PortNodeMetaForQDQ"]

# 元数据映射到端口的名称列表
_METADATA_TO_PORT = [
    "stack_trace",
    "quantization_tag",
]

# 量化操作的列表
_QUANTIZE_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
]

# 反量化操作的列表
_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]

# 选择量化参数操作的列表
_CHOOSE_QPARAMS_OPS = [
    torch.ops.quantized_decomposed.choose_qparams.tensor,
    torch.ops.quantized_decomposed.choose_qparams_symmetric.tensor,
]

# 将元数据从一个节点复制到另一个节点
def _add_metadata(to_node: torch.fx.Node, from_node: torch.fx.Node) -> None:
    from_meta = from_node.meta
    for meta_name in _METADATA_TO_PORT:
        if meta_name in from_meta:
            to_node.meta[meta_name] = from_meta[meta_name]

# 检查节点是否具有量化注释
def _has_quant_annotation(node: torch.fx.Node) -> bool:
    return "quantization_annotation" in node.meta

# 查找包含选择量化参数操作的节点
def _find_choose_qparams_node(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    # 使用广度优先搜索来查找选择量化参数节点
    from collections import deque

    queue = deque(list(node.users.keys()))
    while len(queue):
        n = queue.popleft()
        if n.op == "output":
            continue
        if n.op == "call_function" and n.target in _CHOOSE_QPARAMS_OPS:
            return n
        for k in n.users.keys():
            queue.append(k)
    return None

# 为输入量化节点添加端口元数据
def _port_metadata_for_input_quant_nodes(
    input_node: torch.fx.Node,
    node: torch.fx.Node,
    qspec: Optional[QuantizationSpecBase],
):
    # 如果没有提供量化规范，则直接返回
    if qspec is None:
        return

    # 获取是否动态量化的属性
    is_dynamic_quant = getattr(qspec, "is_dynamic", None)
    # 如果动态量化标志不为 None 并且为 True，则执行以下逻辑
    if is_dynamic_quant is not None and is_dynamic_quant is True:
        # 查找选择量化参数节点
        choose_qparams_node = _find_choose_qparams_node(input_node)
        # 如果找不到选择量化参数节点，则抛出数值错误异常
        if choose_qparams_node is None:
            raise ValueError(f"No chose qparams node found for {node}")
        # 过滤选择量化参数节点的符号大小用户
        choose_qparam_users = _filter_sym_size_users(choose_qparams_node)
        # 如果选择量化参数节点的用户数量不等于 2，则抛出内部错误异常
        if len(choose_qparam_users) != 2:
            raise InternalError(f"Expecting exactly two user for {choose_qparams_node}")
        # 弹出一个符号大小用户作为比例节点
        scale_node = choose_qparam_users.pop()
        # 获取比例节点的用户中的动态量化节点
        dynamic_q_node = next(iter(scale_node.users.keys()))
        # 过滤动态量化节点的符号大小用户
        dynamic_q_node_users = _filter_sym_size_users(dynamic_q_node)
        # 如果动态量化节点的用户数量大于 1，则抛出内部错误异常
        if len(dynamic_q_node_users) > 1:
            raise InternalError(f"Expecting single user for {dynamic_q_node}")
        # 弹出一个符号大小用户作为动态量化偏置节点
        dynamic_dq_node = dynamic_q_node_users.pop()
        # 为选择量化参数节点、动态量化节点和动态量化偏置节点添加元数据
        _add_metadata(choose_qparams_node, node)
        _add_metadata(dynamic_q_node, node)
        _add_metadata(dynamic_dq_node, node)
    else:
        # 如果不是动态量化，则查找输入节点和节点的量化节点和量化偏置节点
        q_node, dq_node = _find_q_dq_node_for_user(input_node, node)
        # 如果找不到量化节点或量化偏置节点，则直接返回
        if q_node is None or dq_node is None:
            return
        # 将量化节点和 get_attr 节点之间的所有节点添加元数据
        # 如果量化节点可以追溯到 get_attr 节点
        q_to_get_attr_nodes = [q_node]
        q_node_input = q_node.args[0]
        while (
            isinstance(q_node_input, torch.fx.Node)
            and q_node_input.op == "call_function"
            and q_node_input.target
            in [
                torch.ops.aten.flatten.using_ints,
                torch.ops.aten.permute.default,
                torch.ops.aten.permute_copy.default,
                torch.ops.aten.slice_copy.Tensor,
                torch.ops.aten.squeeze.dim,
                torch.ops.aten.squeeze_copy.dim,
                torch.ops.aten.transpose.Dimname,
                torch.ops.aten.transpose.int,
                torch.ops.aten.transpose_,
                torch.ops.aten.view_copy.default,
                torch.ops.aten.view.default,
                torch.ops.aten._mkldnn_transpose,
            ]
        ):
            q_to_get_attr_nodes.append(q_node_input)
            q_node_input = q_node_input.args[0]
        # 如果量化节点的输入是 get_attr 节点，则为所有节点添加元数据
        if isinstance(q_node_input, torch.fx.Node) and q_node_input.op == "get_attr":
            for n in q_to_get_attr_nodes:
                _add_metadata(n, q_node_input)
        # 为量化偏置节点和节点添加元数据
        _add_metadata(dq_node, node)
# 为输出量化节点的端口元数据
def _port_metadata_for_output_quant_nodes(
    node: torch.fx.Node, qspec: Optional[QuantizationSpecBase]
):
    # 如果量化规格未提供，则返回
    if qspec is None:
        return

    # 获取使用了当前节点的节点列表
    node_users = _filter_sym_size_users(node)
    # 如果节点使用者不止一个，记录警告信息
    if len(node_users) != 1:
        logger.warning(f"Expecting {node} to have single user")  # noqa: G004

    # 弹出唯一的使用了当前节点的节点
    q_node = node_users.pop()
    # 如果使用节点的操作不是函数调用或者目标不在量化操作列表中，记录警告信息
    if q_node.op != "call_function" or q_node.target not in _QUANTIZE_OPS:
        logger.warning(
            f"Expecting {node} user to be a quantized op but got {q_node}"  # noqa: G004
        )  # noqa: G004
        return

    # 为使用节点添加元数据
    _add_metadata(q_node, node)


class PortNodeMetaForQDQ(PassBase):
    """
    Port metadata for nodes added by quantization flow.
    For static quant these are:
    - quantizer_per_tensor.default, dequantize_per_tensor.default
    - quantizer_per_channel.default, dequantize_per_channel.default
    For dynamic quant these are:
    - choose_qparams.tensor
    - quantizer_per_tensor.tensor, dequantize_per_tensor.tensor
    - quantizer_per_channel.default, dequantize_per_channel.default

    Rules of porting metadata:
    - Metadata to be ported:
      - nn_module_stack
      - stack_trace
      - quantization_tag
    - Metadata to NOT be ported:
      - Everything else
    """
    # Rules:
    # - Statically quantized patterns:
    #   - Dequantize nodes on the inputs to be quantized inherit metadata of the consumer node.
    #   - Quantize nodes on the outputs inherit metadata of the producer node.
    #   - Example 1:
    #     - Original: [Conv -> AvgPool -> Linear]
    #     - Quantized [Q-> DQ -> Conv -> Q -> DQ -> AvgPool -> Q -> DQ -> Linear -> Q -> DQ]
    #     - Inner brackets specify which nodes Q/DQ inherit metadata from
    #     - [Q-> [DQ -> Conv -> Q] -> [DQ -> AvgPool -> Q] -> [DQ -> Linear -> Q] -> DQ]
    #     - Note first Q and last DQ do not inherit metadata from any nodes
    #   - Example 2:
    #     - Original: [Conv -> AvgPool -> Linear]
    #     - AvgPool is not quantized
    #     - Quantized [Q-> DQ -> Conv -> Q -> DQ -> AvgPool -> Q -> DQ -> Linear -> Q -> DQ]
    #     - Inner brackets specify which nodes Q/DQ inherit metadata from
    #     - [Q-> [DQ -> Conv -> Q] -> DQ -> [AvgPool] -> Q -> [DQ -> Linear -> Q] -> DQ]
    #     - Note DQ and Q nodes around AvgPool do not inherit metadata from AvgPool because
    #       AvgPool was not supposed to be quantized. Metadata porting relies on quantization_annotation
    #       on the nodes (in this case AvgPool node) to conclude if the node or pattern was
    #       supposed to be quantized. And subsequent decide if the preceding Q, if any, should
    #       inherit metadata from AvgPool.
    # - Dynamically quantized patterns:
    #   - Inputs that are dynamically quantized have choose_qparams, quantize, and dequantize nodes.
    #   - For example, below linear is dynamically quantized while the rest are statically:
    #     - Original: [Conv -> AvgPool -> Linear]
    #     - Quantized [Q-> DQ -> Conv -> Q -> DQ -> AvgPool -> Q -> DQ -> choose_params -> Q -> DQ -> Linear]
    #     - Quantized [Q-> [DQ -> Conv -> Q] -> [DQ -> AvgPool -> Q] -> DQ -> [choose_params -> Q -> DQ -> Linear]]
    #     - Note first Q does not inherit metadata from any nodes
    # NB:
    # - The best place for porting metadata is during observer conversion to q/dq. This is because it precisely
    #   knows which quantization spec is converted to q/dq and thus from where the metadata should be ported.
    #   However, since FX and PT2E quant workflow are on a common code-base, this hurts readability quite a bit.
    #   Doing it via a separate pass helps readability of the code. Once we are able to refactor PT2E quant
    #   code, this pass should likely be integrated in the refactored variant of "convert" step.
    # 遍历图中的每个节点
    for node in graph_module.graph.nodes:
        # 获取节点的量化注释信息，如果存在的话
        annotation = node.meta.get("quantization_annotation", None)
        # 检查注释是否有效
        if _is_valid_annotation(annotation):
            # 获取输入量化规格映射
            input_qspec_map = node.meta["quantization_annotation"].input_qspec_map
            # 获取输出量化规格
            output_qspec = node.meta["quantization_annotation"].output_qspec
            # 遍历输入节点及其对应的量化规格
            for input_node, qspec in input_qspec_map.items():
                # 为输入量化节点更新端口元数据
                _port_metadata_for_input_quant_nodes(input_node, node, qspec)
            # 更新输出量化节点的端口元数据
            _port_metadata_for_output_quant_nodes(node, output_qspec)
    # 返回处理后的图模块及处理成功标志
    return PassResult(graph_module, True)
```