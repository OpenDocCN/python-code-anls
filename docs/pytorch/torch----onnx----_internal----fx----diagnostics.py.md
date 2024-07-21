# `.\pytorch\torch\onnx\_internal\fx\diagnostics.py`

```py
# mypy: allow-untyped-defs
# 使用未类型化的定义允许 MyPy 放宽对类型的检查

from __future__ import annotations
# 引入未来的 annotations 特性，支持在类型提示中使用字符串类型

import dataclasses
# 导入用于创建数据类的模块

import functools
# 导入用于高阶函数操作的模块

import logging
# 导入日志记录模块

from typing import Any, Optional
# 从 typing 模块中导入 Any 和 Optional 类型提示

import onnxscript  # type: ignore[import]
# 导入 onnxscript 模块，忽略导入时的类型检查错误

from onnxscript.function_libs.torch_lib import graph_building  # type: ignore[import]
# 从 onnxscript 的 torch_lib 模块导入 graph_building 函数，忽略导入时的类型检查错误

import torch
# 导入 PyTorch 库

import torch.fx
# 导入 PyTorch FX 模块

from torch.onnx._internal import diagnostics
# 从 torch.onnx._internal 中导入 diagnostics 模块

from torch.onnx._internal.diagnostics import infra
# 从 torch.onnx._internal.diagnostics 中导入 infra 模块

from torch.onnx._internal.diagnostics.infra import decorator, formatter
# 从 torch.onnx._internal.diagnostics.infra 中导入 decorator 和 formatter 模块

from torch.onnx._internal.fx import registration, type_utils as fx_type_utils
# 从 torch.onnx._internal.fx 中导入 registration 模块，并导入 type_utils 别名为 fx_type_utils

# NOTE: The following limits are for the number of items to display in diagnostics for
# a list, tuple or dict. The limit is picked such that common useful scenarios such as
# operator arguments are covered, while preventing excessive processing loads on considerably
# large containers such as the dictionary mapping from fx to onnx nodes.
# _CONTAINER_ITEM_LIMIT 用于在诊断信息中显示列表、元组或字典的项目数量限制，
# 选择限制以涵盖常见的有用场景，如操作符参数，同时防止处理过多的大容器负荷，
# 比如从 fx 映射到 onnx 节点的字典。

_CONTAINER_ITEM_LIMIT: int = 10
# 设置容器项限制为整数类型，初始值为 10

# NOTE(bowbao): This is a shim over `torch.onnx._internal.diagnostics`, which is
# used in `torch.onnx`, and loaded with `torch`. Hence anything related to `onnxscript`
# cannot be put there.
# 这是对 `torch.onnx._internal.diagnostics` 的一个包装，它在 `torch.onnx` 中使用，并且加载到 `torch` 中。
# 因此，任何与 `onnxscript` 相关的内容都不能放在那里。

# [NOTE: `dynamo_export` diagnostics logging]
# The 'dynamo_export' diagnostics leverages the PT2 artifact logger to handle the verbosity
# level of logs that are recorded in each SARIF log diagnostic. In addition to SARIF log,
# terminal logging is by default disabled. Terminal logging can be activated by setting
# the environment variable `TORCH_LOGS="onnx_diagnostics"`. When the environment variable
# is set, it also fixes logging level to `logging.DEBUG`, overriding the verbosity level
# specified in the diagnostic options.
# See `torch/_logging/__init__.py` for more on PT2 logging.
# `dynamo_export` 诊断日志利用 PT2 工件记录器处理每个 SARIF 日志诊断中记录的日志的详细级别。
# 除了 SARIF 日志外，默认情况下禁用终端日志记录。可以通过设置环境变量 `TORCH_LOGS="onnx_diagnostics"` 来激活终端日志记录。
# 当环境变量被设置时，它还将日志级别固定为 `logging.DEBUG`，覆盖诊断选项中指定的详细级别。
# 有关 PT2 日志记录的更多信息，请参阅 `torch/_logging/__init__.py`。

_ONNX_DIAGNOSTICS_ARTIFACT_LOGGER_NAME = "onnx_diagnostics"
# 设置 onnx_diagnostics 的工件记录器名称为 _ONNX_DIAGNOSTICS_ARTIFACT_LOGGER_NAME

diagnostic_logger = torch._logging.getArtifactLogger(
    "torch.onnx", _ONNX_DIAGNOSTICS_ARTIFACT_LOGGER_NAME
)
# 获取工件记录器，用于记录 "torch.onnx" 的诊断日志，记录器名称为 _ONNX_DIAGNOSTICS_ARTIFACT_LOGGER_NAME

def is_onnx_diagnostics_log_artifact_enabled() -> bool:
    # 检查 onnx_diagnostics 的诊断日志工件记录是否启用
    return torch._logging._internal.log_state.is_artifact_enabled(
        _ONNX_DIAGNOSTICS_ARTIFACT_LOGGER_NAME
    )


@functools.singledispatch
def _format_argument(obj: Any) -> str:
    # 单分派函数，用于格式化参数对象的字符串表示
    return formatter.format_argument(obj)


def format_argument(obj: Any) -> str:
    # 格式化参数对象的字符串表示
    formatter = _format_argument.dispatch(type(obj))
    return formatter(obj)


# NOTE: EDITING BELOW? READ THIS FIRST!
#
# The below functions register the `format_argument` function for different types via
# `functools.singledispatch` registry. These are invoked by the diagnostics system
# when recording function arguments and return values as part of a diagnostic.
# Hence, code with heavy workload should be avoided. Things to avoid for example:
# `torch.fx.GraphModule.print_readable()`.

# 编辑以下内容？首先阅读此信息！
# 下面的函数通过 `functools.singledispatch` 注册 `format_argument` 函数用于不同的类型。
# 这些函数在记录函数参数和返回值作为诊断的一部分时由诊断系统调用。
# 因此，应避免重负载的代码。例如，应避免使用 `torch.fx.GraphModule.print_readable()`。

@_format_argument.register
def _torch_nn_module(obj: torch.nn.Module) -> str:
    # 注册对 torch.nn.Module 类型对象的格式化函数，返回格式化后的字符串表示
    return f"torch.nn.Module({obj.__class__.__name__})"


@_format_argument.register
def _torch_fx_graph_module(obj: torch.fx.GraphModule) -> str:
    # 注册对 torch.fx.GraphModule 类型对象的格式化函数，返回格式化后的字符串表示
    # 构造一个以给定对象类名为参数的 torch.fx.GraphModule 对象的字符串表示
    return f"torch.fx.GraphModule({obj.__class__.__name__})"
# 注册 _format_argument 函数的特定版本，用于处理 torch.fx.Node 对象
@_format_argument.register
def _torch_fx_node(obj: torch.fx.Node) -> str:
    # 创建节点字符串，包含目标和操作信息
    node_string = f"fx.Node({obj.target})[{obj.op}]:"
    # 如果 meta 中没有 'val' 键，返回节点字符串加上 'None'
    if "val" not in obj.meta:
        return node_string + "None"
    # 否则返回节点字符串加上 meta 中 'val' 的格式化字符串
    return node_string + format_argument(obj.meta["val"])

# 注册 _format_argument 函数的特定版本，用于处理 torch.SymBool 对象
@_format_argument.register
def _torch_fx_symbolic_bool(obj: torch.SymBool) -> str:
    # 返回符号布尔值的字符串表示
    return f"SymBool({obj})"

# 注册 _format_argument 函数的特定版本，用于处理 torch.SymInt 对象
@_format_argument.register
def _torch_fx_symbolic_int(obj: torch.SymInt) -> str:
    # 返回符号整数的字符串表示
    return f"SymInt({obj})"

# 注册 _format_argument 函数的特定版本，用于处理 torch.SymFloat 对象
@_format_argument.register
def _torch_fx_symbolic_float(obj: torch.SymFloat) -> str:
    # 返回符号浮点数的字符串表示
    return f"SymFloat({obj})"

# 注册 _format_argument 函数的特定版本，用于处理 torch.Tensor 对象
@_format_argument.register
def _torch_tensor(obj: torch.Tensor) -> str:
    # 返回张量的字符串表示，包含数据类型缩写和形状信息
    return f"Tensor({fx_type_utils.from_torch_dtype_to_abbr(obj.dtype)}{_stringify_shape(obj.shape)})"

# 注册 _format_argument 函数的特定版本，用于处理整数对象
@_format_argument.register
def _int(obj: int) -> str:
    # 返回整数的字符串表示
    return str(obj)

# 注册 _format_argument 函数的特定版本，用于处理浮点数对象
@_format_argument.register
def _float(obj: float) -> str:
    # 返回浮点数的字符串表示
    return str(obj)

# 注册 _format_argument 函数的特定版本，用于处理布尔值对象
@_format_argument.register
def _bool(obj: bool) -> str:
    # 返回布尔值的字符串表示
    return str(obj)

# 注册 _format_argument 函数的特定版本，用于处理字符串对象
@_format_argument.register
def _str(obj: str) -> str:
    # 返回字符串对象本身
    return obj

# 注册 _format_argument 函数的特定版本，用于处理 registration.ONNXFunction 对象
@_format_argument.register
def _registration_onnx_function(obj: registration.ONNXFunction) -> str:
    # 返回 ONNXFunction 对象的简洁显示，包含操作全名和其他属性
    return f"registration.ONNXFunction({obj.op_full_name}, is_custom={obj.is_custom}, is_complex={obj.is_complex})"

# 注册 _format_argument 函数的特定版本，用于处理列表对象
@_format_argument.register
def _list(obj: list) -> str:
    # 创建列表对象的字符串表示，包含长度信息
    list_string = f"List[length={len(obj)}](\n"
    # 如果列表为空，直接返回 None
    if not obj:
        return list_string + "None)"
    # 遍历列表中的项，格式化成字符串添加到列表表示中
    for i, item in enumerate(obj):
        if i >= _CONTAINER_ITEM_LIMIT:
            # 如果超过限制项数，添加省略号
            list_string += "...,\n"
            break
        list_string += f"{format_argument(item)},\n"
    # 返回完整的列表字符串表示
    return list_string + ")"

# 注册 _format_argument 函数的特定版本，用于处理元组对象
@_format_argument.register
def _tuple(obj: tuple) -> str:
    # 创建元组对象的字符串表示，包含长度信息
    tuple_string = f"Tuple[length={len(obj)}](\n"
    # 如果元组为空，直接返回 None
    if not obj:
        return tuple_string + "None)"
    # 遍历元组中的项，格式化成字符串添加到元组表示中
    for i, item in enumerate(obj):
        if i >= _CONTAINER_ITEM_LIMIT:
            # 如果超过限制项数，添加省略号
            tuple_string += "...,\n"
            break
        tuple_string += f"{format_argument(item)},\n"
    # 返回完整的元组字符串表示
    return tuple_string + ")"

# 注册 _format_argument 函数的特定版本，用于处理字典对象
@_format_argument.register
def _dict(obj: dict) -> str:
    # 创建字典对象的字符串表示，包含长度信息
    dict_string = f"Dict[length={len(obj)}](\n"
    # 如果字典为空，直接返回 None
    if not obj:
        return dict_string + "None)"
    # 遍历字典中的键值对，格式化成字符串添加到字典表示中
    for i, (key, value) in enumerate(obj.items()):
        if i >= _CONTAINER_ITEM_LIMIT:
            # 如果超过限制项数，添加省略号
            dict_string += "...\n"
            break
        dict_string += f"{key}: {format_argument(value)},\n"
    # 返回完整的字典字符串表示
    return dict_string + ")"

# 注册 _format_argument 函数的特定版本，用于处理 torch.nn.Parameter 对象
@_format_argument.register
def _torch_nn_parameter(obj: torch.nn.Parameter) -> str:
    # 返回神经网络参数的字符串表示，包含数据部分的格式化字符串
    return f"Parameter({format_argument(obj.data)})"

# 注册 _format_argument 函数的特定版本，用于处理 graph_building.TorchScriptTensor 对象
@_format_argument.register
def _onnxscript_torch_script_tensor(obj: graph_building.TorchScriptTensor) -> str:
    # TODO: Add implementation for _onnxscript_torch_script_tensor function
    pass  # 此处未实现，待补充
    # 构造一个字符串，表示为一个 TorchScriptTensor 对象，包括数据类型和形状信息
    return f"`TorchScriptTensor({fx_type_utils.from_torch_dtype_to_abbr(obj.dtype)}{_stringify_shape(obj.shape)})`"  # type: ignore[arg-type]  # noqa: B950
    # 注释1: `type: ignore[arg-type]` 告诉类型检查工具忽略该行的类型检查警告
    # 注释2: `noqa: B950` 告诉 linter 忽略 B950 警告，这通常表示忽略对类型注释的特定警告
@_format_argument.register
# 注册函数 `_format_argument` 的装饰器，用于处理 `onnxscript.OnnxFunction` 对象并返回其格式化后的字符串表示
def _onnxscript_onnx_function(obj: onnxscript.OnnxFunction) -> str:
    return f"`OnnxFunction({obj.name})`"

@_format_argument.register
# 注册函数 `_format_argument` 的装饰器，用于处理 `onnxscript.TracedOnnxFunction` 对象并返回其格式化后的字符串表示
def _onnxscript_traced_onnx_function(obj: onnxscript.TracedOnnxFunction) -> str:
    return f"`TracedOnnxFunction({obj.name})`"

# 定义函数 `_stringify_shape`，用于将 `torch.Size` 对象转换为字符串表示
# 如果 `shape` 为 `None`，则返回空字符串
# 否则返回以逗号分隔的形状维度列表的字符串表示
def _stringify_shape(shape: Optional[torch.Size]) -> str:
    if shape is None:
        return ""
    return f"[{', '.join(str(x) for x in shape)}]"

# 将 `diagnostics.rules` 赋值给 `rules`
# 将 `diagnostics.levels` 赋值给 `levels`
# 定义 `RuntimeErrorWithDiagnostic`、`LazyString` 和 `DiagnosticOptions` 类型别名
rules = diagnostics.rules
levels = diagnostics.levels
RuntimeErrorWithDiagnostic = infra.RuntimeErrorWithDiagnostic
LazyString = formatter.LazyString
DiagnosticOptions = infra.DiagnosticOptions

# 定义 `Diagnostic` 类，继承自 `infra.Diagnostic`
@dataclasses.dataclass
class Diagnostic(infra.Diagnostic):
    logger: logging.Logger = dataclasses.field(init=False, default=diagnostic_logger)

    def log(self, level: int, message: str, *args, **kwargs) -> None:
        if self.logger.isEnabledFor(level):
            formatted_message = message % args
            if is_onnx_diagnostics_log_artifact_enabled():
                # 当日志工件启用时，才将日志记录到终端
                # 参见 [NOTE: `dynamo_export` diagnostics logging] 获取详细信息
                self.logger.log(level, formatted_message, **kwargs)

            self.additional_messages.append(formatted_message)

# 定义 `DiagnosticContext` 类，泛型类型为 `infra.Diagnostic` 的子类
# 继承自 `infra.DiagnosticContext`
@dataclasses.dataclass
class DiagnosticContext(infra.DiagnosticContext[Diagnostic]):
    logger: logging.Logger = dataclasses.field(init=False, default=diagnostic_logger)
    _bound_diagnostic_type: type[Diagnostic] = dataclasses.field(
        init=False, default=Diagnostic
    )

    def __enter__(self):
        self._previous_log_level = self.logger.level
        # 根据 `is_onnx_diagnostics_log_artifact_enabled()` 的返回值，调整日志级别
        # 以及 `options.verbosity_level` 和环境变量 `TORCH_LOGS` 的影响
        # 参见 [NOTE: `dynamo_export` diagnostics logging] 获取详细信息
        if not is_onnx_diagnostics_log_artifact_enabled():
            return super().__enter__()
        else:
            return self

# 使用 `functools.partial` 创建 `diagnose_call` 函数
# 装饰器为 `decorator.diagnose_call`，使用 `Diagnostic` 作为诊断类型
# `format_argument` 用于格式化参数
diagnose_call = functools.partial(
    decorator.diagnose_call,
    diagnostic_type=Diagnostic,
    format_argument=format_argument,
)

# 定义 `UnsupportedFxNodeDiagnostic` 类，继承自 `Diagnostic`
@dataclasses.dataclass
class UnsupportedFxNodeDiagnostic(Diagnostic):
    unsupported_fx_node: Optional[torch.fx.Node] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        # 注意：这是一个 hack，确保额外字段必须被设置且不为 None。
        # 理想情况下，它们不应该是可选的。但这是 `dataclasses` 的已知限制。
        # 在 Python 3.10 中可以使用 `kw_only=True` 解决此问题。
        # 参见 https://stackoverflow.com/questions/69711886/python-dataclasses-inheritance-and-default-values
        if self.unsupported_fx_node is None:
            raise ValueError("unsupported_fx_node must be specified.")
```