# `.\pytorch\torch\onnx\__init__.py`

```py
# mypy: allow-untyped-defs
# 导入 torch._C 模块和 torch._C._onnx 模块下的特定成员
from torch import _C
from torch._C import _onnx as _C_onnx
from torch._C._onnx import OperatorExportTypes, TensorProtoDataType, TrainingMode

# 导入模块和子模块，同时保持指定的顺序而非按字母排序
from . import (
    _deprecation,
    errors,
    symbolic_caffe2,
    symbolic_helper,
    symbolic_opset7,
    symbolic_opset8,
    symbolic_opset9,
    symbolic_opset10,
    symbolic_opset11,
    symbolic_opset12,
    symbolic_opset13,
    symbolic_opset14,
    symbolic_opset15,
    symbolic_opset16,
    symbolic_opset17,
    symbolic_opset18,
    symbolic_opset19,
    symbolic_opset20,
    utils,
)

# TODO(After 1.13 release): Remove the deprecated SymbolicContext
# 从 _exporter_states 模块导入 ExportTypes 和 SymbolicContext
from ._exporter_states import ExportTypes, SymbolicContext

# 导入 JitScalarType 类型工具函数
from ._type_utils import JitScalarType

# 从 errors 模块中导入 CheckerError，用于向后兼容
from .errors import CheckerError

# 从 utils 模块中导入各种功能函数，包括优化图形、运行符号函数、导出等
from .utils import (
    _optimize_graph,
    _run_symbolic_function,
    _run_symbolic_method,
    export,
    export_to_pretty_string,
    is_in_onnx_export,
    register_custom_op_symbolic,
    select_model_mode_for_export,
    unregister_custom_op_symbolic,
)

# 导入内部的 exporter 模块，保持顺序并避免循环导入
from ._internal.exporter import (
    DiagnosticOptions,
    ExportOptions,
    ONNXProgram,
    ONNXProgramSerializer,
    ONNXRuntimeOptions,
    InvalidExportOptionsError,
    OnnxExporterError,
    OnnxRegistry,
    dynamo_export,
    enable_fake_mode,
)

# 导入内部的 onnxruntime 模块，包括检查是否支持 onnxruntime 后端等
from ._internal.onnxruntime import (
    is_onnxrt_backend_supported,
    OrtBackend as _OrtBackend,
    OrtBackendOptions as _OrtBackendOptions,
    OrtExecutionProvider as _OrtExecutionProvider,
)

# 将以下公开的名称列出，供外部模块使用
__all__ = [
    # Modules
    "symbolic_helper",
    "utils",
    "errors",
    # All opsets
    "symbolic_caffe2",
    "symbolic_opset7",
    "symbolic_opset8",
    "symbolic_opset9",
    "symbolic_opset10",
    "symbolic_opset11",
    "symbolic_opset12",
    "symbolic_opset13",
    "symbolic_opset14",
    "symbolic_opset15",
    "symbolic_opset16",
    "symbolic_opset17",
    "symbolic_opset18",
    "symbolic_opset19",
    "symbolic_opset20",
    # Enums
    "ExportTypes",
    "OperatorExportTypes",
    "TrainingMode",
    "TensorProtoDataType",
    "JitScalarType",
    # Public functions
    "export",
    "export_to_pretty_string",
    "is_in_onnx_export",
    "select_model_mode_for_export",
    "register_custom_op_symbolic",
    "unregister_custom_op_symbolic",
    "disable_log",
    "enable_log",
    # Errors
    "CheckerError",  # Backwards compatibility
    # Dynamo Exporter
    "DiagnosticOptions",
    "ExportOptions",
    "ONNXProgram",
    "ONNXProgramSerializer",
    "ONNXRuntimeOptions",
    "InvalidExportOptionsError",
    "OnnxExporterError",
    "OnnxRegistry",
    "dynamo_export",
    "enable_fake_mode",
    # DORT / torch.compile
    "is_onnxrt_backend_supported",
]

# 设置 ExportTypes 类和 JitScalarType 类的命名空间，用于在外部模块中使用
ExportTypes.__module__ = "torch.onnx"
JitScalarType.__module__ = "torch.onnx"
# 将 ExportOptions 类的模块名设置为 "torch.onnx"
ExportOptions.__module__ = "torch.onnx"
# 将 ONNXProgram 类的模块名设置为 "torch.onnx"
ONNXProgram.__module__ = "torch.onnx"
# 将 ONNXProgramSerializer 类的模块名设置为 "torch.onnx"
ONNXProgramSerializer.__module__ = "torch.onnx"
# 将 ONNXRuntimeOptions 类的模块名设置为 "torch.onnx"
ONNXRuntimeOptions.__module__ = "torch.onnx"
# 将 dynamo_export 函数的模块名设置为 "torch.onnx"
dynamo_export.__module__ = "torch.onnx"
# 将 InvalidExportOptionsError 类的模块名设置为 "torch.onnx"
InvalidExportOptionsError.__module__ = "torch.onnx"
# 将 OnnxExporterError 类的模块名设置为 "torch.onnx"
OnnxExporterError.__module__ = "torch.onnx"
# 将 enable_fake_mode 函数的模块名设置为 "torch.onnx"
enable_fake_mode.__module__ = "torch.onnx"
# 将 OnnxRegistry 类的模块名设置为 "torch.onnx"
OnnxRegistry.__module__ = "torch.onnx"
# 将 DiagnosticOptions 类的模块名设置为 "torch.onnx"
DiagnosticOptions.__module__ = "torch.onnx"
# 将 is_onnxrt_backend_supported 函数的模块名设置为 "torch.onnx"
is_onnxrt_backend_supported.__module__ = "torch.onnx"
# 将 _OrtExecutionProvider 类的模块名设置为 "torch.onnx"
_OrtExecutionProvider.__module__ = "torch.onnx"
# 将 _OrtBackendOptions 类的模块名设置为 "torch.onnx"
_OrtBackendOptions.__module__ = "torch.onnx"
# 将 _OrtBackend 类的模块名设置为 "torch.onnx"
_OrtBackend.__module__ = "torch.onnx"

# 设置产生 ONNX 输出的生产者名称为 "pytorch"
producer_name = "pytorch"
# 设置产生 ONNX 输出的生产者版本为 _C_onnx 模块的生产者版本
producer_version = _C_onnx.PRODUCER_VERSION

# 使用装饰器标记函数 _export 为已弃用，自版本 1.12.0 起，将在版本 2.0 中移除，建议使用 `torch.onnx.export` 代替
@_deprecation.deprecated(
    since="1.12.0", removed_in="2.0", instructions="use `torch.onnx.export` instead"
)
def _export(*args, **kwargs):
    return utils._export(*args, **kwargs)

# TODO(justinchuby): Deprecate these logging functions in favor of the new diagnostic module.

# 返回是否已启用 ONNX 日志记录的布尔值
is_onnx_log_enabled = _C._jit_is_onnx_log_enabled

# 启用 ONNX 日志记录
def enable_log() -> None:
    r"""Enables ONNX logging."""
    _C._jit_set_onnx_log_enabled(True)

# 禁用 ONNX 日志记录
def disable_log() -> None:
    r"""Disables ONNX logging."""
    _C._jit_set_onnx_log_enabled(False)

"""设置 ONNX 日志记录的输出流。

Args:
    stream_name (str, default "stdout"): 只支持 'stdout' 和 'stderr' 作为 ``stream_name``。
"""
set_log_stream = _C._jit_set_onnx_log_output_stream

"""一个用于 ONNX 导出器的简单日志记录设施。

Args:
    args: 参数被转换为字符串，连接到一起，并在末尾加上换行符后刷新到输出流。
"""
log = _C._jit_onnx_log
```