# `.\pytorch\torch\onnx\_internal\fx\fx_onnx_interpreter.py`

```
# mypy: allow-untyped-defs
from __future__ import annotations

import inspect  # 导入 inspect 模块，用于获取对象信息
import logging  # 导入 logging 模块，用于记录日志
import operator  # 导入 operator 模块，提供了对内置操作符的函数实现
import re  # 导入 re 模块，用于正则表达式操作
import types  # 导入 types 模块，用于操作类型信息
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union  # 导入类型提示相关的模块

import onnxscript  # type: ignore[import]  # 导入 onnxscript 模块，忽略类型检查
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
    graph_building as onnxscript_graph_building,  # 导入 onnxscript 模块中的 graph_building 子模块，别名为 onnxscript_graph_building
)

import torch  # 导入 torch 库
import torch.fx  # 导入 torch.fx 模块
from torch.onnx import _type_utils as jit_type_utils  # 导入 torch.onnx._type_utils 模块，别名为 jit_type_utils
from torch.onnx._internal import _beartype  # 导入 torch.onnx._internal 中的 _beartype
from torch.onnx._internal.fx import (  # 导入 torch.onnx._internal.fx 模块中的多个成员
    _pass,  # 导入 _pass
    diagnostics,  # 导入 diagnostics
    onnxfunction_dispatcher,  # 导入 onnxfunction_dispatcher
    op_validation,  # 导入 op_validation
    type_utils as fx_type_utils,  # 导入 type_utils 模块，别名为 fx_type_utils
)
from torch.utils import _pytree  # 导入 torch.utils 中的 _pytree 模块

# 装饰器函数，用于类型检查
@_beartype.beartype
def _fx_node_to_onnx_message_formatter(
    fn: Callable,
    self,
    node: torch.fx.Node,
    *args,
    **kwargs,
) -> str:
    # 格式化输出 FX 节点信息
    return f"FX Node: {node.op}:{node.target}[name={node.name}]. "


# 装饰器函数，用于类型检查
@_beartype.beartype
def _fx_graph_to_onnx_message_formatter(
    fn: Callable,
    self,
    fx_graph_module: torch.fx.GraphModule,
    *args,
    **kwargs,
) -> str:
    # 格式化输出 FX 图信息
    return f"FX Graph: {fx_graph_module._get_name()}. "


def _location_from_fx_stack_trace(
    node_stack_trace: str,
) -> Optional[diagnostics.infra.Location]:
    """Extract location from FX node stack trace.

    TODO(bowbao): Create fx utils module and move this function there.

    Args:
        node_stack_trace: The stack trace of the FX node. Example:

            File "path/file.py", line 311, in <function>
                <code>
            |   File "path/file2.py", line 389, in <function>
                <code>

    Returns:
        location: The location of the FX node.
    """
    if "File" not in node_stack_trace:
        return None

    lines = node_stack_trace.strip().split("\n")
    idx = 0
    while idx < len(lines) and "File" not in lines[idx]:
        idx += 1
    if idx + 1 >= len(lines):
        return None

    pattern = re.compile(r"^File \"(.+)\", line (\d+), in (.+)$")
    matches = pattern.match(lines[idx].strip())
    if matches:
        uri = matches.group(1)
        line_number = int(matches.group(2))
        snippet = lines[idx + 1].strip()
        # 构建并返回 FX 节点的位置信息
        return diagnostics.infra.Location(uri=uri, line=line_number, snippet=snippet)
    return None


@_beartype.beartype
def _retrieve_or_adapt_input_to_graph_set(
    fx_node_arg: fx_type_utils.Argument,
    fx_name_to_onnxscript_value: Dict[
        str,
        Union[
            onnxscript_graph_building.TorchScriptTensor,
            Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
        ],
    ],
    tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
):
    """Map FX value to TorchScript value.

    When creating TorchScript graph from FX graph, we need a mapping from FX variable
    to TorchScript variable. This function maps FX variable, fx_node_arg, to torch.jit.Value.
    """

    onnx_tensor = fx_node_arg  # 将 fx_node_arg 映射到 onnx_tensor
    # 如果 onnx_tensor 是 torch.fx.Node 类型
    if isinstance(onnx_tensor, torch.fx.Node):
        # 返回与 onnx_tensor.name 对应的 TorchScript 值
        return fx_name_to_onnxscript_value[onnx_tensor.name]
    
    # 如果 onnx_tensor 是 tuple 或者 list，并且包含至少一个 torch.fx.Node
    # 并且这些节点在 TorchScript 图中被映射到 torch.jit.Value
    elif isinstance(onnx_tensor, (tuple, list)) and any(
        isinstance(node, torch.fx.Node)
        and fx_type_utils.is_torch_symbolic_type(node.meta.get("val"))
        for node in onnx_tensor
    ):
        # 构建一个序列，其中包含每个节点对应的 TorchScriptTensor 或 None
        sequence_elements: List[
            Union[
                Optional[onnxscript_graph_building.TorchScriptTensor],
                Tuple[
                    onnxscript_graph_building.TorchScriptTensor,
                    ...,
                ],
            ]
        ] = []
        for tensor in onnx_tensor:
            sequence_elements.append(
                fx_name_to_onnxscript_value[tensor.name] if tensor is not None else None
            )
        return sequence_elements
    
    # 如果 onnx_tensor 是 tuple 或者 list，并且所有元素要么是 torch.fx.Node，要么是 None
    elif isinstance(onnx_tensor, (tuple, list)) and all(
        isinstance(node, torch.fx.Node) or node is None for node in onnx_tensor
    ):
        # 构建一个序列，其中包含每个节点对应的 TorchScriptTensor 或 None
        sequence_elements: List[
            Union[
                Optional[onnxscript_graph_building.TorchScriptTensor],
                Tuple[
                    onnxscript_graph_building.TorchScriptTensor,
                    ...,
                ],
            ]
        ] = []
        for tensor in onnx_tensor:
            sequence_elements.append(
                fx_name_to_onnxscript_value[tensor.name] if tensor is not None else None
            )
        return sequence_elements
    
    # 如果 onnx_tensor 是 torch.dtype 类型
    if isinstance(onnx_tensor, torch.dtype):
        # 将 torch.dtype 转换为对应的 ONNX 类型并返回其整数表示
        onnx_tensor = int(
            jit_type_utils.JitScalarType.from_dtype(onnx_tensor).onnx_type()
        )
    
    # 如果 onnx_tensor 是 torch.device 类型
    if isinstance(onnx_tensor, torch.device):
        # 将 torch.device 转换为字符串表示
        # 因为 onnxscript 不支持 torch.device 类型
        return str(onnx_tensor)
    
    # 对于所有其他情况，直接返回 onnx_tensor，不进行处理
    return onnx_tensor
# 过滤掉不被 onnxscript 支持且不需要数据类型转换的 kwargs
def filter_incompatible_and_dtype_convert_kwargs(kwargs):
    """Filter out kwargs that are not supported by onnxscript."""
    # 创建空字典来存储过滤后的 kwargs
    filtered = {}
    # 遍历传入的 kwargs 字典
    for key, value in kwargs.items():
        # 检查键是否在不支持列表中
        if key in {
            "layout",
            "device",
            "requires_grad",
            "pin_memory",
            "memory_format",
            "implicit",
        }:
            # 如果在列表中，则跳过这个键值对
            continue
        # 如果键是 "dtype"
        if key == "dtype":
            # 如果值是 None，则忽略，因为 onnxscript 处理默认情况
            if value is None:
                continue
            else:
                # 将值转换为对应的 onnx 数据类型
                value = int(jit_type_utils.JitScalarType.from_dtype(value).onnx_type())
        # 将符合条件的键值对加入过滤后的字典中
        filtered[key] = value
    # 返回过滤后的 kwargs 字典
    return filtered


# 装饰器，用于类型检查
@_beartype.beartype
# 填充 Tensor 的形状和类型信息
def _fill_tensor_shape_type(
    onnxscript_values: Union[
        onnxscript_graph_building.TorchScriptTensor,
        Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
    ],
    name: str,
    expected_values: Union[
        fx_type_utils.META_VALUE_TYPE,
        List[fx_type_utils.META_VALUE_TYPE],
        Tuple[Optional[fx_type_utils.META_VALUE_TYPE], ...],
    ],
):
    """Fill the meta information of onnxscript_values with that from the fx FakeTensor."""
    
    # 检查 expected_values 是否是列表或元组，而 onnxscript_values 不是
    if isinstance(expected_values, (list, tuple)) and not isinstance(
        onnxscript_values, (list, tuple)
    ):
        # 如果是这种情况，说明 onnxscript_values 是单个 Tensor，但 expected_values 是多个 Tensor 的列表
        # 返回空，表示不进行处理
        return

    # 将 onnxscript_values 和 expected_values 扁平化处理
    flat_onnxscript_values, _ = _pytree.tree_flatten(onnxscript_values)
    flat_expected_values, _ = _pytree.tree_flatten(expected_values)
    
    # 遍历扁平化后的 onnxscript_values 和 expected_values
    for i, (onnxscript_value, expected_value) in enumerate(
        zip(flat_onnxscript_values, flat_expected_values)
    )
        ):
            # 如果期望值为 None，则没有形状/类型信息。
            # 注意：根据 https://github.com/pytorch/pytorch/blob/main/torch/_meta_registrations.py，
            # None 可能是返回类型的有效值，因此我们需要处理它。
            # 例如，在 CPU 模式下的函数 meta__scaled_dot_product_flash()。
            continue
        elif fx_type_utils.is_torch_symbolic_type(expected_value):
            # aten::sym_size 的输出是一个整数，而不是张量，代表一个维度的大小。我们将其视为 1 维张量。
            onnxscript_value.dtype = fx_type_utils.from_sym_value_to_torch_dtype(
                expected_value
            )
            onnxscript_value.shape = torch.Size([1])
        elif isinstance(expected_value, (int, float, bool)):
            # 如果期望值是整数、浮点数或布尔值，设置数据类型和形状
            onnxscript_value.dtype = fx_type_utils.from_scalar_type_to_torch_dtype(
                type(expected_value)
            )
            onnxscript_value.shape = torch.Size([])
        elif isinstance(expected_value, complex):
            # 将复数标量转换为实数表示
            onnxscript_value_to_torch_dtype = (
                fx_type_utils.from_scalar_type_to_torch_dtype(type(expected_value))
            )
            onnxscript_value.dtype = (
                fx_type_utils.from_complex_to_float(onnxscript_value_to_torch_dtype)
                if onnxscript_value_to_torch_dtype is not None
                else None
            )
            onnxscript_value.shape = torch.Size([2])
        elif fx_type_utils.is_torch_complex_dtype(expected_value.dtype):
            # 类似于 torch.view_as_real，我们将复杂张量展平为具有额外最后维度的实数张量
            onnxscript_value.shape = torch.Size((*expected_value.size(), 2))
            # 复杂数类型转换为对应的浮点数类型
            onnxscript_value.dtype = fx_type_utils.from_complex_to_float(
                expected_value.dtype
            )
            # 调度器需要知道值是复数
            onnxscript_value.is_complex = True
        else:
            # 设置节点输出大小为动态以继续模型转换，同时在 add_input() 中也将输入设置为动态
            onnxscript_value.shape = expected_value.size()
            onnxscript_value.dtype = expected_value.dtype

        # 命名处理
        if i > 0:
            # 如果 i 大于 0，则为 onnxscript_value 设置带索引的名称
            onnxscript_value.name = f"{name}_{i}"
        else:
            # 如果 i 等于 0，则为 onnxscript_value 设置基本名称
            onnxscript_value.name = name
# 使用装饰器进行类型检查和验证
@_beartype.beartype
# 定义一个函数，用于填充默认的关键字参数
def _fill_in_default_kwargs(
    node: torch.fx.Node,
) -> Tuple[List[fx_type_utils.Argument], Dict[str, fx_type_utils.Argument]]:
    """Find and Fill in the not provided kwargs with default values."""
    
    # 如果节点对象具有 "_schema" 属性，则使用其作为 node_schema
    if hasattr(node.target, "_schema"):
        node_schema = node.target._schema  # type: ignore[union-attr]
    else:
        # 否则，使用 torch.ops.aten.sym_size.int._schema 作为 node_schema
        node_schema = torch.ops.aten.sym_size.int._schema  # type: ignore[union-attr]

    # 创建空列表和字典来存储完整的参数和关键字参数
    complete_args: List[fx_type_utils.Argument] = []
    complete_kwargs: Dict[str, fx_type_utils.Argument] = {}

    # 如果 node.target 是内置函数或方法，直接将 node.args 添加到 complete_args 中
    if inspect.isbuiltin(node.target):
        complete_args = list(node.args)
    else:
        # 否则，根据 node_schema 中的预期参数遍历并填充 complete_args 和 complete_kwargs
        for i, expected_arg in enumerate(node_schema.arguments):
            if i < len(node.args):
                complete_args.append(node.args[i])
            elif expected_arg.name in node.kwargs:
                complete_kwargs[expected_arg.name] = node.kwargs[expected_arg.name]
            else:
                # 如果没有在 node.kwargs 中找到对应参数，则使用 schema 中的默认值填充
                complete_kwargs[expected_arg.name] = expected_arg.default_value

    # 返回完整的参数列表和关键字参数字典
    return complete_args, complete_kwargs


# 使用装饰器进行类型检查和验证
@_beartype.beartype
# 定义一个函数，将 FX 图的参数映射到 TorchScript 图中的参数
def _wrap_fx_args_as_onnxscript_args(
    complete_args: List[fx_type_utils.Argument],
    complete_kwargs: Dict[str, fx_type_utils.Argument],
    fx_name_to_onnxscript_value: Dict[
        str,
        Union[
            onnxscript_graph_building.TorchScriptTensor,
            Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
        ],
    ],
    tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
) -> Tuple[
    Sequence[
        Optional[
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                str,
                int,
                float,
                bool,
                list,
                complex,
            ]
        ]
    ],
    Dict[str, fx_type_utils.Argument],
]:
    """Map all FX arguments of a node to arguments in TorchScript graph."""

    # 将 complete_args 中的每个参数映射或适应到 TorchScript 图中的相应参数
    onnxscript_args = tuple(
        _retrieve_or_adapt_input_to_graph_set(arg, fx_name_to_onnxscript_value, tracer)
        for arg in complete_args
    )
    
    # 过滤不兼容的关键字参数并执行数据类型转换
    onnxscript_kwargs = filter_incompatible_and_dtype_convert_kwargs(complete_kwargs)

    # 返回映射后的 TorchScript 参数元组和处理后的关键字参数字典
    return onnxscript_args, onnxscript_kwargs


# 定义一个无状态的类，用于处理 FX 图节点并将其转换为对应的 ONNX 节点
class FxOnnxInterpreter:
    """Stateless class to process FX graph Nodes and translate them into their ONNX counterparts.

    All FX nodes described by [FX Graph](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) are supported.
    Similarly to [FX Interpreter pattern](https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter), each FX node
    must be implemented on its own method in this class.
    """
    # Each operator's implementation returns either an `onnxscript.OnnxFunction` or
    # `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.
    # They can also raise RuntimeError if no overloaded functions are available for the given FX node.
    """
    TODO: Convert methods to @staticmethod when the diagnostic system supports it
          DO NOT ADD NEW ATTRIBUTES TO THIS CLASS!
    """

    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
    ):
        # THIS SHOULD BE THE ONLY STATE IN THIS CLASS (constraint from diagnosticS API)
        # TODO: Diagnostics API should be revised to get rid of this attribute.
        # DO NOT add other class-level attributes.
        self.diagnostic_context = diagnostic_context

    @_beartype.beartype
    @diagnostics.diagnose_call(
        diagnostics.rules.fx_node_to_onnx,
        diagnostic_message_formatter=_fx_node_to_onnx_message_formatter,
    )
    def run_node(
        self,
        node,
        fx_graph_module: torch.fx.GraphModule,
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,
        op_level_debug: bool,
        onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        onnxscript_tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
    ):
        # Method decorated to diagnose calls using the specified diagnostic rules and message formatter.
        # This method processes a node in the FX graph to convert it into an ONNX function representation.
        # Parameters:
        # - node: The node from the FX graph to be processed.
        # - fx_graph_module: The TorchScript GraphModule containing the FX graph.
        # - onnxfunction_dispatcher: Dispatcher for ONNX function implementations.
        # - op_level_debug: Flag indicating whether to enable operational level debugging.
        # - onnxscript_graph: Graph representation in ONNX script format for building.
        # - onnxscript_tracer: Tracing evaluator for ONNX script graph building.
        # - fx_name_to_onnxscript_value: Mapping of FX node names to corresponding ONNX script values.
        pass
    ):
        """
        Execute a single FX node to produce its ONNX counterpart.

        Args:
            node: The FX node to be translated.
            fx_graph_module: The FX graph module containing the node.
            onnxfunction_dispatcher: The dispatcher to find the best matched ONNX op.
            op_level_debug (bool): Whether to enable op level debug.
            onnxscript_graph: The ONNX graph to be populated.
            onnxscript_tracer: The tracer to trace the ONNX graph.
            fx_name_to_onnxscript_value: The mapping from FX node name to ONNX Script value.

        Raises:
            RuntimeError: When a node.op is not supported.
        """
        # 记录节点的堆栈跟踪信息
        node_stack_trace = node.stack_trace
        if node_stack_trace:
            # 创建节点在诊断信息中的诊断上下文
            diagnostic = self.diagnostic_context.inflight_diagnostic(
                rule=diagnostics.rules.fx_node_to_onnx
            )
            # 在诊断日志中记录 PyTorch 源信息
            with diagnostic.log_section(logging.INFO, "PyTorch source information"):
                diagnostic.info("```\n%s\n```", node_stack_trace)
            # 从 FX 的堆栈跟踪中获取节点的位置信息
            location = _location_from_fx_stack_trace(node_stack_trace)
            if location is not None:
                diagnostic.with_location(location)

        # 根据节点的操作类型执行相应的处理
        if node.op == "placeholder":
            self.placeholder(node, onnxscript_graph, fx_name_to_onnxscript_value)
        elif node.op == "get_attr":
            self.get_attr(
                node,
                onnxscript_graph,
                fx_name_to_onnxscript_value,
                fx_graph_module,
            )
        elif node.op == "call_function":
            self.call_function(
                node,
                onnxscript_tracer,
                fx_name_to_onnxscript_value,
                onnxfunction_dispatcher,
                op_level_debug,
                fx_graph_module,
            )
        elif node.op == "call_method":
            self.call_method(node)
        elif node.op == "call_module":
            self.call_module(
                node,
                onnxscript_graph,
                fx_name_to_onnxscript_value,
                onnxscript_tracer,
                fx_graph_module,
                onnxfunction_dispatcher,
                op_level_debug,
            )
        elif node.op == "output":
            self.output(node, onnxscript_graph, fx_name_to_onnxscript_value)
        else:
            # 抛出异常，说明发现了在 torch.fx 中未定义的节点类型
            raise RuntimeError(f"Found node type not defined in torch.fx: {node.op}")

    @_beartype.beartype
    @diagnostics.diagnose_call(
        diagnostics.rules.fx_graph_to_onnx,
        diagnostic_message_formatter=_fx_graph_to_onnx_message_formatter,
    )
    def run(
        self,
        fx_graph_module: torch.fx.GraphModule,
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,
        op_level_debug: bool,
        parent_onnxscript_graph: Optional[
            onnxscript_graph_building.TorchScriptGraph
        ] = None,
    ):
    # 定义一个方法 placeholder，用于生成图中的占位符节点
    def placeholder(
        self,
        node: torch.fx.Node,
        onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
    ):
        # 获取节点中存储的伪造张量信息（由 FakeTensorProp 生成）
        fake_tensor = node.meta.get("val", None)
        
        # 如果 fake_tensor 为 None 或者是基本类型（int, float, bool, str）之一
        # 表明这些节点代表输入常量，在转换为 ONNX 时不需要创建 TorchScriptTensor
        if fake_tensor is None or isinstance(fake_tensor, (int, float, bool, str)):
            output = onnxscript_graph.add_input(
                input_name=None,
            )
        # 如果 fake_tensor 是 torch.Tensor 类型
        elif isinstance(fake_tensor, torch.Tensor):
            # 如果 fake_tensor 的数据类型是复数类型（complex64/complex128），ONNX 不支持这些类型
            # 因此将其转换为实数类型（float32/float64）
            if fx_type_utils.is_torch_complex_dtype(fake_tensor.dtype):
                fake_tensor = torch.view_as_real(fake_tensor.resolve_conj())
            output = onnxscript_graph.add_input(
                input_name=node.name,
                shape=fake_tensor.shape,
                dtype=fake_tensor.dtype,
            )
        # 如果 fake_tensor 是 torch 符号类型
        elif fx_type_utils.is_torch_symbolic_type(fake_tensor):
            # 将符号类型转换为对应的 Torch 数据类型
            output = onnxscript_graph.add_input(
                input_name=node.name,
                shape=torch.Size([]),
                dtype=fx_type_utils.from_sym_value_to_torch_dtype(fake_tensor),
            )
        else:
            # 如果 fake_tensor 类型不受支持，则抛出运行时错误
            raise RuntimeError(
                f"Unsupported type(node.meta['val']) for placeholder: {type(fake_tensor)}"
            )
        
        # 断言输出 output 不为空，确保节点创建成功
        assert (
            output is not None
        ), f"Node creates None with target={node.target} and name={node.name}"
        
        # 断言 output 是 TorchScriptTensor 类型的实例，确保符合预期
        assert isinstance(output, onnxscript_graph_building.TorchScriptTensor)
        # 断言 output 是 onnxscript.tensor.Tensor 类型的实例，确保符合预期
        assert isinstance(output, onnxscript.tensor.Tensor)
        
        # 将节点名称和生成的 output 存入映射表 fx_name_to_onnxscript_value 中
        fx_name_to_onnxscript_value[node.name] = output
    @_beartype.beartype
    # 使用装饰器 `_beartype.beartype` 对函数进行类型检查和验证
    def call_function(
        self,
        node: torch.fx.Node,
        onnxscript_tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,
        op_level_debug: bool,
        fx_graph_module: torch.fx.GraphModule,
    ):
        # 根据节点的类型执行相应的函数调用操作
        if isinstance(node.args[0], torch.fx.Node):
            # 如果参数是一个节点对象，则获取其对应的 ONNX Script 值，并将其注册为输出
            onnx_tensor_or_tensor_tuple = fx_name_to_onnxscript_value[node.args[0].name]
            onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)
        else:
            # 如果参数是一个集合类型，例如字典、元组的元组等，将其扁平化并逐个注册为输出
            flat_args, _ = _pytree.tree_flatten(node.args[0])
            for arg in flat_args:
                assert isinstance(
                    arg, torch.fx.Node
                ), f"arg must be a torch.fx.Node, not {type(arg)}"
                onnx_tensor_or_tensor_tuple = fx_name_to_onnxscript_value[arg.name]
                onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)

    @_beartype.beartype
    # 使用装饰器 `_beartype.beartype` 对函数进行类型检查和验证
    def output(
        self,
        node: torch.fx.Node,
        onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
    ):
        # 如果节点的参数是一个 torch.fx.Node 对象，获取其对应的 ONNX Script 值并注册为输出
        if isinstance(node.args[0], torch.fx.Node):
            onnx_tensor_or_tensor_tuple = fx_name_to_onnxscript_value[node.args[0].name]
            onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)
        else:
            # 如果参数是一个集合类型，将其扁平化并逐个注册为输出
            flat_args, _ = _pytree.tree_flatten(node.args[0])
            for arg in flat_args:
                assert isinstance(
                    arg, torch.fx.Node
                ), f"arg must be a torch.fx.Node, not {type(arg)}"
                onnx_tensor_or_tensor_tuple = fx_name_to_onnxscript_value[arg.name]
                onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)

    @_beartype.beartype
    # 使用装饰器 `_beartype.beartype` 对函数进行类型检查和验证
    def call_method(self, node: torch.fx.Node):
        # 抛出运行时异常，因为当前不支持调用方法的操作
        raise RuntimeError("call_method is not supported yet.")

    @_beartype.beartype
    # 使用装饰器 `_beartype.beartype` 对函数进行类型检查和验证
    def call_module(
        self,
        node: torch.fx.Node,
        parent_onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
        tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
        root_fx_graph_module: torch.fx.GraphModule,
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,
        op_level_debug: bool,
    ):
        # 根据节点调用相应模块的方法，并传递相关参数进行处理
        pass
    # 定义一个方法用于获取节点的属性
    def get_attr(
        self,
        node: torch.fx.Node,  # node 参数是一个 torch.fx.Node 类型的对象，表示当前节点
        onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,  # onnxscript_graph 参数是一个 TorchScriptGraph 对象，用于构建 ONNX 图
        fx_name_to_onnxscript_value: Dict[  # fx_name_to_onnxscript_value 参数是一个字典，用于映射 FX 图中的名称到 TorchScript 值的关系
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,  # 值可以是 TorchScriptTensor 类型
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],  # 或者是 TorchScriptTensor 元组类型
            ],
        ],
        fx_graph_module: torch.fx.GraphModule,  # fx_graph_module 参数是一个 torch.fx.GraphModule 对象，表示 FX 图的模块
    ):
        assert isinstance(node.target, str), f"node.target {node.target} is not a str."
        # 获取节点目标对应的属性张量
        attr_tensor = getattr(fx_graph_module, node.target)
        assert isinstance(attr_tensor, torch.Tensor), f"{attr_tensor} is not a tensor."

        # Parameter/buffer name cannot contain "."
        # Revert from "/" to restore namespace formatting.
        # 将节点目标中的 "/" 替换为 "."，恢复命名空间格式，并作为初始化器添加到 ONNX 图中
        input_ = onnxscript_graph.add_initializer(
            name=node.target.replace("/", "."),
            value=attr_tensor,
        )

        assert isinstance(input_, onnxscript_graph_building.TorchScriptTensor)
        assert isinstance(input_, onnxscript.tensor.Tensor)
        # 将初始化器 input_ 与节点名称 node.name 关联，更新到 fx_name_to_onnxscript_value 字典中
        fx_name_to_onnxscript_value[node.name] = input_
```