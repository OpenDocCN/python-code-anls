# `.\pytorch\torch\onnx\_internal\jit_utils.py`

```
# mypy: allow-untyped-defs
# 定义了一个模块，用于处理 torch.Graph 对象和 torchscript 的操作
"""Utilities for manipulating the torch.Graph object and the torchscript."""
from __future__ import annotations

# TODO(justinchuby): Move more of the symbolic helper functions here and expose
# them to the user.
# 未来的工作：将更多符号帮助函数移至此处，并对用户公开

import dataclasses  # 导入 dataclasses 模块，用于创建数据类
import re  # 导入 re 模块，用于处理正则表达式
import typing  # 导入 typing 模块，用于类型提示
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union  # 导入各种类型提示

import torch  # 导入 torch 模块
from torch import _C  # 从 torch 模块导入 _C 子模块
from torch.onnx._globals import GLOBALS  # 从 torch.onnx._globals 导入 GLOBALS 变量
from torch.onnx._internal import _beartype, registration  # 导入 _beartype 和 registration 函数

_ATTR_PATTERN = re.compile("^(.+)_(([ifstgz])|(ty))$")  # 编译正则表达式，用于匹配属性名
_SKIP_NODE_ATTRIBUTES = {"inplace", "aten"}  # 定义要跳过的节点属性集合


@dataclasses.dataclass
class GraphContext:
    """Extra context for symbolic functions with all methods from torch.Graph.

    NOTE: This class is not meant for external consumption. Please do not depend on
    it outside of torch.onnx as the interface may evolve.

    Attributes:
        graph: The _C.Graph being constructed.
        block: The current _C.Block being constructed.
        opset: The opset version.
        original_node: Current node that is being converted from.
        params_dict: Mapping from graph initializer name to IValue.
        env: Mapping from Torch domain graph Value to ONNX domain graph Value.
        values_in_env: Set of all values in env, for constant-time lookups.
        new_nodes: List that tracks all new nodes that are added (used to make
            sure metadata is propagated to all new nodes).
    """
    graph: _C.Graph  # torch 脚本图的构造中使用的 _C.Graph 对象
    block: _C.Block  # 当前构造中的 _C.Block 对象
    opset: int  # 操作集版本号
    original_node: _C.Node  # 当前转换的原始节点
    params_dict: Dict[str, "_C.IValue"]  # 图初始化器名称到 IValue 的映射字典
    env: Dict[_C.Value, _C.Value]  # Torch 域图值到 ONNX 域图值的映射字典
    values_in_env: Set[_C.Value]  # env 中所有值的集合，用于快速查找
    new_nodes: List[_C.Node] = dataclasses.field(default_factory=list)  # 跟踪所有新增节点的列表，用于在所有新节点间传播元数据

    # 为了与期望 _C.Graph 的符号函数兼容，中继方法从 _C.Graph 中继承
    def __getattr__(self, name: str) -> Any:
        return getattr(self.graph, name)

    @_beartype.beartype
    def op(
        self,
        opname: str,
        *raw_args: Union[torch.Tensor, _C.Value],
        outputs: int = 1,
        **kwargs,
    ):
        """
        创建一个 ONNX 运算符 "opname"，使用 "raw_args" 作为输入，"kwargs" 作为属性。

        运算符及其输入/属性的集合在 https://github.com/onnx/onnx/blob/master/docs/Operators.md 中有文档记录。

        Args:
            opname: ONNX 运算符的名称，例如 `Abs` 或 `Add`，或者带命名空间的运算符，例如 `aten::add`。
            raw_args: 运算符的输入；通常作为 `symbolic` 定义的参数提供。
            outputs: 运算符返回的输出数。默认情况下，假定运算符返回单个输出。
                     如果 `outputs` 大于一，此函数返回一个输出值的元组，按顺序表示 ONNX 运算符的每个输出 `Value`。
            kwargs: ONNX 运算符的属性，其键按照以下约定命名：
                    `alpha_f` 表示类型为 `f` 的 `alpha` 属性。有效的类型指示符包括 `f` (浮点数)、`i` (整数)、
                    `s` (字符串) 或 `t` (张量)。指定为浮点类型的属性接受单个浮点数或浮点数列表
                    （例如，对于接受整数列表的 `dims` 属性，可以使用 `dims_i`）。

        Returns:
            表示此运算符单个输出的值（有关多返回节点，请参阅 `outputs` 关键字参数）。
        """
        # FIXME(justinchuby): 一旦确定如何处理 mypy，添加返回类型
        return _add_op(self, opname, *raw_args, outputs=outputs, **kwargs)

    @_beartype.beartype
    def aten_op(self, operator: str, *args, overload_name: str = "", **kwargs):
        """
        生成一个 ONNX ATen 运算符节点。

        此函数用于与旧符号函数的向后兼容性。
        """
        return self.op(
            "aten::ATen",
            *args,
            operator_s=operator,
            overload_name_s=overload_name,
            **kwargs,
        )

    # NOTE: 与旧符号函数的向后兼容性。
    # 在 fx 导出器建立后，可能会删除此部分。
    at = aten_op

    @_beartype.beartype
    def onnxscript_op(
        self,
        onnx_fn,
        *raw_args: Union[torch.Tensor, _C.Value],
        outputs: int = 1,
        **kwargs,
        ):
        """
        创建一个 ONNX 脚本运算符。

        Args:
            onnx_fn: ONNX 函数名。
            raw_args: 运算符的输入，可以是 torch.Tensor 或 _C.Value 类型。
            outputs: 运算符返回的输出数，默认为 1。
            kwargs: 其他属性。

        """
    ):
        """
        从 onnx-script 函数创建一个 ONNX 运算符，使用 "raw_args" 作为输入和 "kwargs" 作为属性。

        onnx-script 仓库地址：https://github.com/microsoft/onnx-script

        Args:
            onnx_fn: 来自 onnx-script 的 ONNXFunction；示例可在 https://github.com/microsoft/onnx-script#example 找到
            raw_args: 运算符的输入；通常作为 `symbolic` 定义的参数提供。
            outputs: 此运算符返回的输出数量。
                默认情况下，假定运算符返回单个输出。
                如果 `outputs` 大于一，则此函数返回一个输出值的元组，
                按顺序表示 ONNX 运算符的每个输出 `Value`。
            kwargs: ONNX 运算符的属性，其键按照以下约定命名：
                `alpha_f` 表示类型为 `f` 的 `alpha` 属性。有效的类型指定符号包括
                `f`（浮点数）、`i`（整数）、`s`（字符串）或 `t`（张量）。
                用浮点数类型指定的属性接受单个浮点数或浮点数列表
                （例如，对于接受整数列表的 `dims` 属性，可以说 `dims_i`）。

        Returns:
            表示此运算符单个输出的值（有关多返回节点，请参阅 `outputs` 关键字参数）。
        """
        # 注意（titaiwang）：这里使用了类属性，如果 onnx-script 对这些属性做出任何更改，则需要更新。
        symbolic_name = f"{onnx_fn.opset.domain}::{onnx_fn.name}"
        opset_version = onnx_fn.opset.version

        # 使用自定义的 onnx 符号注册函数将符号名和 opset 版本注册到 onnx_fn
        registration.custom_onnx_symbolic(symbolic_name, opset_version)(onnx_fn)

        # 调用 _add_op 函数将运算符添加到操作数，并传入符号名、raw_args、outputs 和 kwargs
        return _add_op(self, symbolic_name, *raw_args, outputs=outputs, **kwargs)
# 使用装饰器 @_beartype.beartype 对函数进行类型检查和验证
@_beartype.beartype
# 创建一个函数 add_op_with_blocks，用于添加带有子块的 ONNX 操作符
def add_op_with_blocks(
    # 图形上下文，表示当前图的上下文信息
    graph_context: GraphContext,
    # ONNX 操作符的名称，可以是简单名称如 `Abs` 或 `Add`，也可以是带有命名空间的操作符如 `aten::add`
    opname: str,
    # 输入参数列表，可以是任意数量的 _C.Value 类型
    *inputs: _C.Value,
    # 操作符输出的数量，默认为 1
    outputs: int = 1,
    # 创建的子块数量，默认为 1
    n_blocks: int = 1,
    # 其他属性参数，作为 ONNX 操作符的属性传递
    **attributes,
) -> Tuple[Any, Tuple[GraphContext, ...], _C.Node]:
    """Creates an ONNX operator "opname", taking inputs and attributes.

    Args:
        graph_context: The context for the current graph.
        opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
            with a namespace, e.g., `aten::add`.
        inputs: The inputs to the operator.
        outputs: The number of outputs this operator returns.
            By default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Value`, representing each output of the ONNX operator
            in order.
        n_blocks: The number of sub-blocks to create in the node.
        attributes: The attributes of the ONNX operator.

    Returns:
        A tuple of (output_values, new_contexts, node) where:
            output_values: One or more output value of this operator
                (see the `outputs` keyword argument for multi-return nodes).
            new_contexts: A tuple of new graph contexts for each sub-block.
            node: The node representing the operator.
    """

    # 调用 graph_context 的 op 方法，创建指定名称的 ONNX 操作符，并传入对应的参数和属性
    output_values = graph_context.op(opname, *inputs, outputs=outputs, **attributes)
    
    # 根据 output_values 的类型确定 node 的值
    if isinstance(output_values, Sequence):
        node = output_values[0].node()
    else:
        node = output_values.node()

    new_contexts = []
    # 根据 n_blocks 参数循环创建新的子块，并更新 graph_context
    for _ in range(n_blocks):
        new_block = node.addBlock()
        # 创建 graph_context 的浅拷贝，并更新其 block 属性为新创建的子块 new_block
        new_context = dataclasses.replace(graph_context, block=new_block)
        new_contexts.append(new_context)

    # 返回包含 output_values、new_contexts 和 node 的元组
    return output_values, tuple(new_contexts), node


# 使用装饰器 @_beartype.beartype 对函数进行类型检查和验证
@_beartype.beartype
# 创建一个函数 _add_op，用于创建简单的 ONNX 操作符
def _add_op(
    # 图形上下文，表示当前图的上下文信息
    graph_context: GraphContext,
    # ONNX 操作符的名称
    opname: str,
    # 输入参数列表，可以是 torch.Tensor 或 _C.Value 类型的 Union
    *args: Union[torch.Tensor, _C.Value],
    # 操作符输出的数量，默认为 1
    outputs: int = 1,
    # 其他属性参数，作为 ONNX 操作符的属性传递
    **kwargs,
):
    """Creates an ONNX operator "opname", taking "args" as inputs and attributes "kwargs".

    The set of operators and the inputs/attributes they take
    is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

    This function is monkey-patched onto Graph.
    """
    def _symbolic_op(graph_context, opname, args, outputs=1, **kwargs):
        """
        Create a symbolic node in the Torch Graph or Block for an ONNX operator.
    
        Args:
            graph_context: The Torch Graph or Block.
            opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
                with a namespace, e.g., `aten::add`.
            args: The inputs to the operator; usually provided
                as arguments to the `symbolic` definition.
            outputs: The number of outputs this operator returns.
                By default an operator is assumed to return a single output.
                If `outputs` is greater than one, this functions returns a tuple
                of output `Value`, representing each output of the ONNX operator
                in order.
            kwargs: The attributes of the ONNX operator, whose keys are named
                according to the following convention: `alpha_f` indicates
                the `alpha` attribute with type `f`.  The valid type specifiers are
                `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
                specified with type float accepts either a single float, or a
                list of floats (e.g., you would say `dims_i` for a `dims` attribute
                that takes a list of integers).
    
        Returns:
            (Union[_C.Value, Tuple[_C.Value, ...]])
            The value representing the single output of this operator (see the `outputs`
            keyword argument for multi-return nodes).
        """
        # Convert args to constants if they are tensors
        inputs = [_const_if_tensor(graph_context, arg) for arg in args]
        
        # Filter out None attributes, allowing client-side passing of None attributes
        attributes = {k: v for k, v in kwargs.items() if v is not None}
        
        # Prepend 'onnx::' if opname doesn't already include a namespace
        if "::" not in opname:
            opname = "onnx::" + opname
        
        # Create a new node in the graph context block
        node = _create_node(
            graph_context.block,
            opname,
            inputs,
            attributes,
            params_dict=graph_context.params_dict,
            opset_version=graph_context.opset,
            n_outputs=outputs,
            shape_inference=GLOBALS.onnx_shape_inference,
        )
        
        # Append the newly created node to the list of new nodes in graph context
        graph_context.new_nodes.append(node)
    
        # Return either a single output value or a tuple of outputs based on 'outputs'
        if outputs == 1:
            return node.output()
        return tuple(node.outputs())
# 使用装饰器进行类型检查和注解
@_beartype.beartype
def _const_if_tensor(graph_context: GraphContext, arg):
    # 如果参数为 None，则直接返回 None
    if arg is None:
        return arg
    # 如果参数是 _C.Value 类型，则直接返回该参数
    if isinstance(arg, _C.Value):
        return arg

    # 否则调用 _add_op 函数，创建一个 "onnx::Constant" 的操作节点，并返回结果
    return _add_op(graph_context, "onnx::Constant", value_z=arg)


def _create_node(
    graph_or_block: Union[_C.Graph, _C.Block],
    domain_op: str,
    inputs: Sequence,
    attributes: dict,
    params_dict: dict,
    opset_version: int,
    n_outputs: int,
    shape_inference: bool = True,
) -> _C.Node:
    """Creates an node 'domain_op', taking inputs and attributes."""
    # 如果 graph_or_block 是 _C.Graph 类型
    if isinstance(graph_or_block, _C.Graph):
        graph = graph_or_block
        # 在图中创建一个指定域操作的节点，接收输入和输出数量
        node = graph.create(domain_op, inputs, n_outputs)
        # 将创建的节点插入到图中
        node = graph.insertNode(node)
    # 如果 graph_or_block 是 _C.Block 类型
    elif isinstance(graph_or_block, _C.Block):
        block = graph_or_block
        # 在块中添加一个指定域操作的节点，接收输入
        node = block.addNode(domain_op, inputs)

        # 如果节点需要多个输出
        if n_outputs > 1:
            # 循环添加额外的输出
            for _ in range(1, n_outputs):
                node.addOutput()

    # 获取节点的所有输出
    node_outputs = tuple(node.outputs())  # type: ignore[possibly-undefined]
    # 断言节点的输出数量与指定的 n_outputs 相等
    assert len(node_outputs) == n_outputs

    # 检查 domain_op 是否以 "aten::" 开头
    aten = domain_op.startswith("aten::")

    # 添加所有的属性到节点中
    for key, value in sorted(attributes.items()):
        # 如果属性名在 _SKIP_NODE_ATTRIBUTES 中，则跳过
        if key in _SKIP_NODE_ATTRIBUTES:
            continue
        # 调用 _add_attribute 函数，将属性添加到节点中
        _add_attribute(node, key, value, aten=aten)
    
    # 如果需要进行形状推断
    if shape_inference:
        # 调用 _C._jit_pass_onnx_node_shape_type_inference 函数，进行形状和类型推断
        _C._jit_pass_onnx_node_shape_type_inference(node, params_dict, opset_version)
    
    # 返回创建的节点
    return node


@_beartype.beartype
def _is_onnx_list(value):
    # 检查 value 是否为可迭代对象且不是字符串、字节或者 torch.Tensor 类型
    return isinstance(value, Iterable) and not isinstance(
        value, (str, bytes, torch.Tensor)
    )


@_beartype.beartype
def _scalar(x: torch.Tensor):
    """Convert a scalar tensor into a Python value."""
    # 断言张量 x 的元素数量为 1
    assert x.numel() == 1
    # 返回张量 x 的第一个元素作为 Python 值
    return x[0]


@_beartype.beartype
def _add_attribute(node: _C.Node, key: str, value: Any, aten: bool):
    r"""Initializes the right attribute based on type of value."""
    # 使用正则表达式匹配属性名称的模式
    m = _ATTR_PATTERN.match(key)
    # 如果匹配结果为 None，则抛出 ValueError 异常
    if m is None:
        raise ValueError(
            f"Invalid attribute specifier '{key}' names "
            "must be suffixed with type, e.g. 'dim_i' or 'dims_i'"
        )
    # 提取属性名称和类型
    name, kind = m.group(1), m.group(2)
    # 如果 value 是 ONNX 列表，则将 kind 后缀加上 "s"
    if _is_onnx_list(value):
        kind += "s"

    # 根据 kind 动态调用 _C.Node 对象的对应方法，将属性值添加到节点中
    return getattr(node, f"{kind}_")(name, value)


# TODO: Expose this to user when migrating symbolic helper functions to here.
@_beartype.beartype
def _is_tensor(x: _C.Value) -> bool:
    # 判断 x 是否是 _C.Value 类型且其类型是 _C.TensorType 的子类型
    return x.type().isSubtypeOf(_C.TensorType.get())


@_beartype.beartype
def get_device_from_value(value: _C.Value) -> Optional[torch.device]:
    # 如果 value 不是张量类型，则返回 None
    if not _is_tensor(value):
        return None
    # 获取 value 的类型信息，并转换为 _C.TensorType 类型
    tensor_type = typing.cast(_C.TensorType, value.type())
    # 返回张量的设备信息
    return tensor_type.device()


@_beartype.beartype
def parse_node_kind(kind: str) -> Tuple[str, str]:
    """Parse node kind into domain and Op name."""
    # 如果节点的 kind 不包含 "::" 符号，则抛出 ValueError 异常
    if "::" not in kind:
        raise ValueError(f"Node kind: {kind} is invalid. '::' is not in node kind.")
    # 使用 "::" 分割字符串 kind，将结果分别赋给 domain 和 opname 变量
    domain, opname = kind.split("::", 1)
    
    # 如果 opname 中仍然包含 "::"，则抛出 ValueError 异常，提示 kind 的格式不正确
    if "::" in opname:
        raise ValueError(f"Node kind: {kind} is invalid. '::' should only appear once.")
    
    # 返回 domain 和 opname 变量作为结果
    return domain, opname
# 使用 @_beartype 装饰器来确保 is_aten 函数参数类型为 str，返回类型为 bool
@_beartype.beartype
def is_aten(domain: str) -> bool:
    """Check if the domain is official."""
    # 检查给定的域名是否为 "aten"
    return domain == "aten"

# 使用 @_beartype 装饰器来确保 is_prim 函数参数类型为 str，返回类型为 bool
@_beartype.beartype
def is_prim(domain: str) -> bool:
    """Check if the domain is official."""
    # 检查给定的域名是否为 "prim"
    return domain == "prim"

# 使用 @_beartype 装饰器来确保 is_onnx 函数参数类型为 str，返回类型为 bool
@_beartype.beartype
def is_onnx(domain: str) -> bool:
    """Check if the domain is official."""
    # 检查给定的域名是否为 "onnx"
    return domain == "onnx"
```