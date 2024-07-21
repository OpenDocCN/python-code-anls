# `.\pytorch\torch\onnx\_internal\fx\onnxfunction_dispatcher.py`

```
# mypy: allow-untyped-defs
"""Dispatcher for AtenLib functions from onnx-script."""

from __future__ import annotations

import logging  # 导入 logging 模块，用于记录日志
import operator  # 导入 operator 模块，用于函数操作符
import types  # 导入 types 模块，支持类型相关操作
from typing import (  # 导入 typing 模块，支持类型提示
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import torch  # 导入 PyTorch 模块
import torch._ops  # 导入 PyTorch 私有模块 _ops
import torch.fx  # 导入 PyTorch FX 模块
from torch.onnx._internal import _beartype  # 导入 PyTorch ONNX 内部模块 _beartype
from torch.onnx._internal.fx import (  # 导入 PyTorch ONNX FX 相关模块
    diagnostics,
    registration,
    type_utils as fx_type_utils,
)

if TYPE_CHECKING:
    import onnxscript  # type: ignore[import]  # 如果类型检查开启，导入 onnxscript 模块

    from torch.onnx import OnnxRegistry  # 如果类型检查开启，导入 OnnxRegistry 类型


# For beartype
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
    graph_building as onnxscript_graph_building,  # 导入 onnxscript 中的 graph_building 模块
)


@_beartype.beartype
def _find_opschema_matched_symbolic_function_disagnostic_message_formatter(
    fn: Callable,
    self,
    node: torch.fx.Node,
    default_and_custom_functions: List[registration.ONNXFunction],
    *args,
    **kwargs,
) -> str:
    """Format the diagnostic message for the nearest match warning."""
    all_function_overload_names = ""
    for symbolic_func in default_and_custom_functions:
        overload_func = symbolic_func.onnx_function
        all_function_overload_names += f"ONNX Node: {overload_func.name}[opset={overload_func.opset};is_custom={symbolic_func.is_custom}]. \n"  # noqa: B950
    return f"FX Node: {node.target}. \n" f"{all_function_overload_names}"


@_beartype.beartype
def _find_operator_overloads_in_onnx_registry_disagnostic_message_formatter(
    fn: Callable,
    self,
    node: torch.fx.Node,
    *args,
    **kwargs,
) -> str:
    """Format the diagnostic message for the nearest match warning."""
    return f"Searching operator overload: '{node.target}' in onnx registry...\n"


class OnnxFunctionDispatcher:
    """A dispatcher that finds the best ONNX Function for ATen/Custom operators.

    It uses the `torch.ops` name to find the function. If not found, it falls back to default.
    Otherwise, the best match is found among all function overloads. An exact match has
    higher precedence over the closest ones.

    Below is a breakdown on how the dispatch mechanism works:

    1. Use the torch.ops name to find the function:
        a. Check if the ATen overload exists in the registry.
        b. If not, check if the default overload exists in the registry.

    2. Find the nearest match among all overloaded functions:
        a. If the types match perfectly, select the function.
        b. Otherwise, find the nearest one with the highest matching score. Because of
            the potential wrongly annotated dtypes and attributes matching, we use
            nearest match to find the best function once the aten name is targeted.

    3. Tie-breaker: If there are multiple nearest matches, we will select the one with
        the highest matching score.

    NOTE: The nearest match `doesn't guarantee` a correct match, and a warning message is logged.
    """
    def __init__(
        self,
        onnx_registry: "OnnxRegistry",
        diagnostic_context: diagnostics.DiagnosticContext,
    ):
        """初始化 ONNX 函数调度器。

        Args:
            onnx_registry: ONNX 注册表实例。
            diagnostic_context: 用于报告错误的诊断上下文。
        """
        self.onnx_registry = onnx_registry
        self.diagnostic_context = diagnostic_context

    @_beartype.beartype
    def dispatch(
        self,
        node: torch.fx.Node,
        onnx_args: Sequence[
            Optional[
                Union[fx_type_utils.TensorLike, str, int, float, bool, list, complex]
            ]
        ],
        onnx_kwargs: Dict[str, fx_type_utils.Argument],
        diagnostic_context: diagnostics.DiagnosticContext,
    ) -> Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"]:
        """基于给定的 FX 节点、参数和关键字参数来调度 ONNX 函数。

        Args:
            node: 要调度函数的 TorchFX 节点。
            onnx_args: ONNX 函数的参数。
            onnx_kwargs: ONNX 函数的关键字参数。
            diagnostic_context: 用于报告错误的诊断上下文。

        Returns:
            根据调度算法返回一个 `onnxscript.OnnxFunction` 或 `onnxscript.TracedOnnxFunction` 实例。

        Raises:
            RuntimeError: 如果对于给定的 FX 节点没有可用的重载函数。
        """
        # 如果对于给定的 FX 节点没有可用的重载函数，抛出不支持的错误
        default_and_custom_functions = self.get_function_overloads(
            node, diagnostic_context
        )

        # 如果有可用的重载函数，我们将找到一个最完美或最接近匹配给定参数和关键字参数的函数
        return self._find_the_perfect_or_nearest_match_onnxfunction(
            node,
            default_and_custom_functions,
            onnx_args,
            onnx_kwargs,
            diagnostic_context,
        )

    @_beartype.beartype
    def _filter_or_keep_complex(
        self,
        node,
        default_and_custom_functions: List[registration.ONNXFunction],
        diagnostic_context: diagnostics.DiagnosticContext,
        onnx_args: Sequence[
            Optional[
                Union[fx_type_utils.TensorLike, str, int, float, bool, list, complex]
            ]
        ],
        onnx_kwargs: Dict[str, fx_type_utils.Argument],
    ) -> List[registration.ONNXFunction]:
        """过滤或保留复杂函数。

        Args:
            node: 节点对象。
            default_and_custom_functions: 默认和自定义的 ONNX 函数列表。
            diagnostic_context: 用于报告错误的诊断上下文。
            onnx_args: ONNX 函数的参数。
            onnx_kwargs: ONNX 函数的关键字参数。

        Returns:
            经过过滤或保留处理后的 ONNX 函数列表。
        """
    ) -> List[registration.ONNXFunction]:
        """Filter the complex functions if the input has complex dtype."""

        # 判断节点参数中是否存在复数类型的参数
        args_with_complex_dtype = [_is_arg_with_complex_dtype(arg) for arg in node.args]
        
        # 如果存在任意一个参数是复数类型，则筛选出所有复数函数
        if any(args_with_complex_dtype):
            default_and_custom_functions = [
                func for func in default_and_custom_functions if func.is_complex
            ]
            
            # 如果找不到复数函数组，则抛出错误
            if not default_and_custom_functions:
                # 获取操作的全名
                op_full_name = self._get_aten_name(
                    node, diagnostic_context
                ).qualified_name()
                # 创建不支持的节点诊断信息
                diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
                    diagnostics.rules.no_symbolic_function_for_call_function,
                    diagnostics.levels.ERROR,
                    f"Cannot find any COMPLEX symbolic function for {op_full_name}, "
                    f"which should be registered under {node.target}.",
                    unsupported_fx_node=node,
                )
                # 记录诊断信息
                diagnostic_context.log(diagnostic)
                # 抛出带诊断信息的运行时错误
                raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
        else:
            # 如果所有参数均非复数类型，则筛选出所有非复数函数
            default_and_custom_functions = [
                func for func in default_and_custom_functions if not func.is_complex
            ]
            
            # 如果找不到非复数函数组，则抛出错误
            if not default_and_custom_functions:
                # 获取操作的全名
                op_full_name = self._get_aten_name(
                    node, diagnostic_context
                ).qualified_name()
                # 创建不支持的节点诊断信息
                diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
                    diagnostics.rules.no_symbolic_function_for_call_function,
                    diagnostics.levels.ERROR,
                    f"Can ONLY find COMPLEX symbolic function for {op_full_name}, "
                    f"which should be registered under {node.target}.",
                    unsupported_fx_node=node,
                )
                # 记录诊断信息
                diagnostic_context.log(diagnostic)
                # 抛出带诊断信息的运行时错误
                raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
        
        # 返回筛选后的函数列表
        return default_and_custom_functions

    @_beartype.beartype
    @diagnostics.diagnose_call(
        diagnostics.rules.find_opschema_matched_symbolic_function,
        diagnostic_message_formatter=_find_opschema_matched_symbolic_function_disagnostic_message_formatter,
    )
    def _find_the_perfect_or_nearest_match_onnxfunction(
        self,
        node: torch.fx.Node,  # this is used in diagnostic_message_formatter
        default_and_custom_functions: List[registration.ONNXFunction],
        onnx_args: Sequence[
            Optional[
                Union[fx_type_utils.TensorLike, str, int, float, bool, list, complex]
            ]
        ],
        onnx_kwargs: Dict[str, fx_type_utils.Argument],
        diagnostic_context: diagnostics.DiagnosticContext,
    @_beartype.beartype
    # 定义一个私有方法 _get_aten_name，接受两个参数：node 和 diagnostic_context
    def _get_aten_name(
        self, node: torch.fx.Node, diagnostic_context: diagnostics.DiagnosticContext
    ):
        # 应用装饰器 @_beartype.beartype，用于参数类型检查和自动化测试
        @_beartype.beartype
        # 应用装饰器 @diagnostics.diagnose_call，执行诊断调用并传入特定的规则和格式化器
        @diagnostics.diagnose_call(
            diagnostics.rules.find_operator_overloads_in_onnx_registry,
            diagnostic_message_formatter=_find_operator_overloads_in_onnx_registry_disagnostic_message_formatter,
        )
        # 定义方法 get_function_overloads，接受两个参数：node 和 diagnostic_context
        def get_function_overloads(
            self,
            node: torch.fx.Node,
            diagnostic_context: diagnostics.DiagnosticContext,
    ) -> List[registration.ONNXFunction]:
        """从注册表中获取函数重载。

        Args:
            node: 要获取函数重载的节点。
            diagnostic_context: 用于报告错误的诊断上下文。

        Returns:
            返回包含ONNXFunctions的列表，以默认函数为开头，后跟任何自定义函数。
        """

        internal_opname: registration.OpName = self._get_aten_name(
            node=node, diagnostic_context=diagnostic_context
        )

        # 如果ATen/Custom运算符未注册，function_group将为None。
        # 未注册的ATen/Custom运算符将在下一步触发错误。
        function_group: Optional[List[registration.ONNXFunction]] = None

        function_group = self.onnx_registry.get_op_functions(
            namespace=internal_opname.namespace,
            op_name=internal_opname.op_name,
            overload=internal_opname.overload,
        )

        # 注意：如果ONNX注册表中没有找到对应的重载，回退到默认的重载。
        if function_group is None:
            function_group = self.onnx_registry.get_op_functions(
                namespace=internal_opname.namespace,
                op_name=internal_opname.op_name,
                overload=None,
            )
            if function_group is not None:
                op_full_name = internal_opname.qualified_name()
                diagnostic = diagnostic_context.inflight_diagnostic()
                diagnostic.warning(
                    "### 在ONNX注册表中找不到运算符重载！\n"
                    "无法在ONNX注册表中找到运算符重载，但找到了默认的重载。请仔细检查ONNX输出。\n",
                )
                diagnostic.level = diagnostics.levels.WARNING

        if function_group is not None:
            # 注意：如果输入具有复杂的数据类型，将只调度到复杂函数。
            function_group = self._filter_or_keep_complex(
                node, function_group, diagnostic_context
            )
            return function_group  # type: ignore[return-value]

        op_full_name = internal_opname.qualified_name()
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
            diagnostics.rules.no_symbolic_function_for_call_function,
            diagnostics.levels.ERROR,
            f"找不到{op_full_name}的符号函数，该函数应在{node.target}下注册。",
            unsupported_fx_node=node,
        )
        diagnostic_context.log(diagnostic)
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
class _OnnxSchemaChecker:
    """
    The OnnxSchemaChecker class is a checker for ONNX OpSchema and param schema.

    It provides methods to check for input compatibility based on the OpSchema. It also
    provides a matching score to indicate how well the OpSchema matches the input and
    kwargs types. A function will be evaluated as perfect match, nearest match eligible,
    or no match.

    Here are some common examples in categories:

    1. [NOTE: Perfect match]: The number of inputs and attributes are exactly the same as
        the OpSchema. The types of inputs and attributes are exactly the same as the
        OpSchema.

        ```python
        inputs = (Tensor[2, 3], Tensor[2, 3])
        attributes = {"alpha": 1.0}

        @torch_op("aten::op")
        def aten_op(self: TReal, other: TReal, alpha: float = 1) -> TReal:
            ...

        ```
        Result: Perfect match.

    2. [NOTE: Optional input]: The dispatcher recognizes optional inputs. However,
        the input can't be ignored. None must be provided.

        ```python
        inputs = (Tensor([2, 3]), None)
        attributes = {}

        aten_op(X: TTensor, Y: Optional[INT64]):
            ...
        ```
        Result: Perfect match.
        Real example: `aten::convolution`.

    3. [NOTE: Different attributes]: If an attribute is provided with value, it's
        a must to match the attribute in function signature.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a":1, "b":2}

        aten_op(X: TTensor, a: int):
            ...
        ```
        Result: No match.
        Real example: `aten::div` vs `aten::div.Tensor_mode`.

    4. [NOTE: Default attributes]: Default attribute will fill in the value into
        inputs/attributes.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {}

        aten_op(X: TTensor, a: int = 3):
            ...
        ```
        Result: Perfect match.
        Real example: `aten::clone`

    5. [NOTE: Ignore attribute with None value]: The attributes with None value
        will be ignored in matching.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a": None}

        aten_op(X: TTensor):
            ...
        ```
        Result: Perfect match.

        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a": None}

        aten_op(X: TTensor, a: int = 3):
            ...
        ```
        Result: Nearest match eligible.
        Real example: `aten::div` vs `aten::div.Tensor_mode`.

    Attributes:
        onnxfunction: The OnnxFunction.
        param_schema: The parameter schema defined in the OnnxFunction.
        op_schema: The ONNX OpSchema.
        type_constraints: The type constraints defined in the OpSchema.
        attributes: The attributes defined in the OpSchema.
        _matching_score: The matching score of the OnnxSchemaChecker .
    """
    # 空白，这是_OnnxSchemaChecker类的头部说明性注释，解释了该类的作用和功能
    pass
    def __init__(
        self,
        onnxfunction: Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction],
    ):
        """
        Initialize the OnnxSchemaChecker instance.

        Args:
            onnxfunction: The OnnxFunction or TracedOnnxFunction object.
        """
        self.onnxfunction = onnxfunction
        # Retrieve parameter schemas from the OnnxFunction instance
        self.param_schema = self.onnxfunction.param_schemas()
        
        # Obtain operational schema from the OnnxFunction instance; op_schema is always not None
        op_schema = self.onnxfunction.op_schema
        assert op_schema is not None
        self.op_schema = op_schema
        
        # Extract type constraints from op_schema and initialize type_constraints dictionary
        self.type_constraints = {
            constraint.type_param_str: set(constraint.allowed_type_strs)
            for constraint in self.op_schema.type_constraints
        }
        
        # Fetch attributes from op_schema
        self.attributes = self.op_schema.attributes
        
        # Initialize matching score as None
        self._matching_score: Optional[int] = None

    @property
    def match_score(self) -> Optional[int]:
        """
        The matching score of the OnnxSchemaChecker instance.

        If this remains None, it means the matching score has not been calculated,
        and it's not a nearest match candidate.

        Returns:
            The matching score of the OnnxSchemaChecker instance.
        """
        return self._matching_score

    @_beartype.beartype
    def perfect_match_inputs(
        self,
        diagnostic: diagnostics.Diagnostic,
        args: Sequence[
            Optional[
                Union[fx_type_utils.TensorLike, str, int, float, bool, list, complex]
            ]
        ],
        kwargs: Dict[str, fx_type_utils.Argument],
    ) -> bool:
        """
        Check if the inputs perfectly match the expected types defined by the OnnxSchema.

        Args:
            diagnostic: The diagnostic instance to record any discrepancies.
            args: Sequence of optional inputs including tensors, primitives, or lists.
            kwargs: Dictionary of named arguments.

        Returns:
            True if the inputs match perfectly; False otherwise.
        """
        # Iterate over each attribute and verify its type against expected Onnx type
        def _match_onnx_attribute_type(
            attribute_name: str,
            attribute: Union[
                fx_type_utils.Argument, onnxscript_graph_building.TorchScriptTensor
            ],
            is_sequence: bool = False,
        ) -> bool:
            if isinstance(attribute, (int, float, bool, str)):
                # Convert Python type to Onnx attribute type and check against expected type
                attribute_onnx_type = fx_type_utils.from_python_type_to_onnx_attribute_type(
                    type(attribute), is_sequence=is_sequence
                )
                if attribute_onnx_type != self.attributes[attribute_name].type:
                    return False
            # If attribute is a non-empty list, recursively check each element's type
            elif isinstance(attribute, (list, tuple)) and attribute:
                return _match_onnx_attribute_type(
                    attribute_name, attribute[0], is_sequence=True
                )
            else:
                # Unrecognized attribute type
                return False
            return True
        
        # Validate inputs against expected types and record discrepancies in diagnostic
        for attr_name, attr_value in kwargs.items():
            if not _match_onnx_attribute_type(attr_name, attr_value):
                diagnostic.log(f"Mismatched type for attribute: {attr_name}")
                return False
        
        return True

    @_beartype.beartype
    def _record_matching_score(
        self,
        inputs: Sequence[
            Optional[
                Union[fx_type_utils.TensorLike, str, int, float, bool, list, complex]
            ]
        ],
        attributes: Dict[str, fx_type_utils.Argument],
    ) -> None:
        """
        Record the matching score based on the inputs and attributes.

        Args:
            inputs: Sequence of optional inputs.
            attributes: Dictionary of attributes.

        Returns:
            None
        """
        # Placeholder for recording matching score logic; not fully implemented here
        pass
    ):
        """
        Calculate the inputs matching score of the OpSchema requirements to find the nearest match.

        Only the functions which have the same number of inputs and attributes as the
        OpSchema are eligible to be a nearest match candidate. Thus, we don't need to
        check the length of inputs and attributes here, and only check the types of
        inputs and attributes.

        How the matching score is calculated:
            score += 1 if one input/attribute type is in the type constraints.

        Limitations:
            None/NoeType/[] could result in zero matches, and the same score of overloads,
            which will be recorded in SARIF.

        Args:
            inputs: The input arguments.
            attributes: The input keyword arguments.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """
        self._matching_score = 0
        # If they have different length of arguments, the score would be lower to those
        # functions which have the same length of arguments.
        for schema_input, torch_input in zip(self.op_schema.inputs, inputs):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if allowed_types.intersection(torch_input_compatible_types):
                # If torch_input_compatible_types is in allowed_types
                # of this input defined in the OpSchema, we know the function
                # and the input are compatible
                self._matching_score += 1
        # NOTE: The penalty is applied to those functions which have different attributes.
        for attribute_name, attribute_proto in self.attributes.items():
            attribute = attributes[attribute_name]
            attribute_onnx_type = fx_type_utils.from_python_type_to_onnx_attribute_type(
                type(attribute)
            )
            if attribute_onnx_type != attribute_proto.type:
                # If the attribute type of the OpSchema and the attribute type don't match,
                # we know the function and the input are not compatible
                self._matching_score -= 1

    # NOTE: Referenced from onnxscript internal function.
    # Importing this function makes the code less robust, as it is not a public API.
    @_beartype.beartype
    def _separate_input_attributes_from_arguments(
        self,
        param_schemas: Sequence["onnxscript.values.ParamSchema"],
        args: Sequence[
            Optional[
                Union[fx_type_utils.TensorLike, str, int, float, bool, list, complex]
            ]
        ],
        kwargs: Dict[str, fx_type_utils.Argument],
        fill_defaults: bool = True,
# 使用装饰器 @_beartype.beartype，对函数进行类型检查和装饰
@_beartype.beartype
# 检查是否输入参数具有复杂数据类型，支持递归检查
def _is_arg_with_complex_dtype(arg: fx_type_utils.Argument) -> bool:
    """Check if the node has complex dtype recursively."""
    # 如果参数是 torch.fx.Node 类型，并且其 meta 属性中包含 'val'，且 'val' 是 torch.Tensor 类型，并且是复数类型
    if (
        isinstance(arg, torch.fx.Node)
        and "val" in arg.meta
        and isinstance(arg.meta["val"], torch.Tensor)
        and torch.is_complex(arg.meta["val"])
    ):
        return True
    # 如果参数是 list 类型，则递归检查列表中的每个元素
    elif isinstance(arg, list):
        for item in arg:
            return _is_arg_with_complex_dtype(item)
    # 默认返回 False，表示参数没有复杂数据类型
    return False


# 使用装饰器 @_beartype.beartype，对函数进行类型检查和装饰
@_beartype.beartype
# 根据输入的 torch 数据类型转换为兼容的 ONNX 数据类型字符串
def _find_onnx_data_type(
    torch_input: Optional[
        Union[fx_type_utils.TensorLike, str, int, float, bool, list, tuple, complex]
    ]
) -> Set[str]:
    """Convert inputs data type from torch acceptable dtype to the compatible onnx dtype string."""
    # 如果 torch_input 是 fx_type_utils.TensorLike 类型，并且具有有效的 dtype
    if (
        isinstance(torch_input, fx_type_utils.TensorLike)
        and torch_input.dtype is not None
    ):
        return fx_type_utils.from_torch_dtype_to_onnx_dtype_str(torch_input.dtype)
    
    # 如果 torch_input 是 int、float、bool、str 或 complex 类型
    if isinstance(torch_input, (int, float, bool, str, complex)):
        return fx_type_utils.from_torch_dtype_to_onnx_dtype_str(type(torch_input))
    
    # 如果 torch_input 是 list 或 tuple 类型，并且不为空
    if isinstance(torch_input, (list, tuple)) and torch_input:
        # 找到第一个非 None 的元素
        the_first_non_none_item = next(
            (item for item in torch_input if item is not None), None
        )
        # 递归查找该元素的 ONNX 数据类型
        set_dtype = _find_onnx_data_type(the_first_non_none_item)
        # 如果列表中包含任何 fx_type_utils.TensorLike 类型的元素，则返回 seq(tensor(onnx_type)) 的集合
        if any(isinstance(input, fx_type_utils.TensorLike) for input in torch_input):
            return {f"seq({dtype})" for dtype in set_dtype}
        # 否则返回找到的 ONNX 数据类型集合
        else:
            return set_dtype
    
    # 如果 torch_input 是 None，或者是 fx_type_utils.TensorLike 类型但没有有效的 dtype，
    # 或者是空的 list 或 tuple
    if (
        torch_input is None
        or (
            isinstance(torch_input, fx_type_utils.TensorLike)
            and torch_input.dtype is None
        )
        or (isinstance(torch_input, (list, tuple)) and not torch_input)
    ):
        # 返回空集合，表示不支持的类型或边缘情况
        return set()
    
    # 如果以上条件都不符合，则抛出运行时错误，表示未知的输入类型
    raise RuntimeError(f"Unknown input type from input: {torch_input}")
```