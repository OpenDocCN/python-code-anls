# `.\pytorch\torch\onnx\_internal\diagnostics\_diagnostic.py`

```
# mypy: allow-untyped-defs
"""Diagnostic components for TorchScript based ONNX export, i.e. `torch.onnx.export`."""
from __future__ import annotations

import contextlib  # 导入上下文管理工具模块
import gzip  # 导入 gzip 压缩文件处理模块
from collections.abc import Generator  # 导入生成器抽象基类
from typing import List, Optional  # 导入类型提示工具中的 List 和 Optional

import torch  # 导入 PyTorch 深度学习框架

from torch.onnx._internal.diagnostics import infra  # 导入 Torch ONNX 导出的诊断基础设施
from torch.onnx._internal.diagnostics.infra import formatter, sarif  # 导入诊断基础设施中的格式化和 SARIF 模块
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version  # 导入 SARIF 版本信息
from torch.utils import cpp_backtrace  # 导入 C++ 调用堆栈追踪工具


def _cpp_call_stack(frames_to_skip: int = 0, frames_to_log: int = 32) -> infra.Stack:
    """Returns the current C++ call stack.

    This function utilizes `torch.utils.cpp_backtrace` to get the current C++ call stack.
    The returned C++ call stack is a concatenated string of the C++ call stack frames.
    Each frame is separated by a newline character, in the same format of
    r"frame #[0-9]+: (?P<frame_info>.*)". More info at `c10/util/Backtrace.cpp`.

    """
    # NOTE: Cannot use `@_beartype.beartype`. It somehow erases the cpp stack frame info.
    frames = cpp_backtrace.get_cpp_backtrace(frames_to_skip, frames_to_log).split("\n")
    frame_messages = []
    for frame in frames:
        segments = frame.split(":", 1)
        if len(segments) == 2:
            frame_messages.append(segments[1].strip())
        else:
            frame_messages.append("<unknown frame>")
    return infra.Stack(
        frames=[
            infra.StackFrame(location=infra.Location(message=message))
            for message in frame_messages
        ]
    )


class TorchScriptOnnxExportDiagnostic(infra.Diagnostic):
    """Base class for all export diagnostics.

    This class is used to represent all export diagnostics. It is a subclass of
    infra.Diagnostic, and adds additional methods to add more information to the
    diagnostic.
    """

    python_call_stack: Optional[infra.Stack] = None  # 可选的 Python 调用堆栈信息
    cpp_call_stack: Optional[infra.Stack] = None  # 可选的 C++ 调用堆栈信息

    def __init__(
        self,
        *args,
        frames_to_skip: int = 1,  # 跳过的栈帧数，默认为 1
        cpp_stack: bool = False,  # 是否记录 C++ 调用堆栈，默认为 False
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)  # 调用父类构造函数
        self.python_call_stack = self.record_python_call_stack(
            frames_to_skip=frames_to_skip
        )  # 记录 Python 调用堆栈
        if cpp_stack:
            self.cpp_call_stack = self.record_cpp_call_stack(
                frames_to_skip=frames_to_skip
            )  # 记录 C++ 调用堆栈

    def record_cpp_call_stack(self, frames_to_skip: int) -> infra.Stack:
        """Records the current C++ call stack in the diagnostic."""
        # NOTE: Cannot use `@_beartype.beartype`. It somehow erases the cpp stack frame info.
        # No need to skip this function because python frame is not recorded
        # in cpp call stack.
        stack = _cpp_call_stack(frames_to_skip=frames_to_skip)  # 获取当前 C++ 调用堆栈
        stack.message = "C++ call stack"  # 设置堆栈信息的描述
        self.with_stack(stack)  # 将堆栈信息添加到当前诊断对象中
        return stack


class ExportDiagnosticEngine:
    """PyTorch ONNX Export diagnostic engine.
    
    This class represents the diagnostic engine for PyTorch's ONNX export functionality.
    It manages export diagnostics and provides methods to handle diagnostic data.

    """
    The only purpose of creating this class instead of using `DiagnosticContext` directly
    is to provide a background context for `diagnose` calls inside exporter.

    By design, one `torch.onnx.export` call should initialize one diagnostic context.
    All `diagnose` calls inside exporter should be made in the context of that export.
    However, since diagnostic context is currently being accessed via a global variable,
    there is no guarantee that the context is properly initialized. Therefore, we need
    to provide a default background context to fallback to, otherwise any invocation of
    exporter internals, e.g. unit tests, will fail due to missing diagnostic context.
    This can be removed once the pipeline for context to flow through the exporter is
    established.
    """



    contexts: List[infra.DiagnosticContext]
    _background_context: infra.DiagnosticContext



    def __init__(self) -> None:
        # Initialize an empty list to hold multiple diagnostic contexts
        self.contexts = []
        # Initialize a background diagnostic context for torch.onnx with the current version of torch
        self._background_context = infra.DiagnosticContext(
            name="torch.onnx",
            version=torch.__version__,
        )



    @property
    def background_context(self) -> infra.DiagnosticContext:
        # Return the background diagnostic context for torch.onnx
        return self._background_context



    def create_diagnostic_context(
        self,
        name: str,
        version: str,
        options: Optional[infra.DiagnosticOptions] = None,
    ) -> infra.DiagnosticContext:
        """Creates a new diagnostic context.

        Args:
            name: The subject name for the diagnostic context.
            version: The subject version for the diagnostic context.
            options: The options for the diagnostic context.

        Returns:
            A new diagnostic context.
        """
        # If options are not provided, create an empty DiagnosticOptions object
        if options is None:
            options = infra.DiagnosticOptions()
        # Create a new diagnostic context with the provided name, version, and options
        context: infra.DiagnosticContext[infra.Diagnostic] = infra.DiagnosticContext(
            name, version, options
        )
        # Add the newly created diagnostic context to the list of contexts
        self.contexts.append(context)
        return context



    def clear(self):
        """Clears all diagnostic contexts."""
        # Clear the list of diagnostic contexts
        self.contexts.clear()
        # Clear the diagnostics within the background context
        self._background_context.diagnostics.clear()



    def to_json(self) -> str:
        # Convert the SARIF log into a JSON formatted string
        return formatter.sarif_to_json(self.sarif_log())



    def dump(self, file_path: str, compress: bool = False) -> None:
        """Dumps the SARIF log to a file."""
        # If compress is True, write the SARIF log to a gzip-compressed file
        if compress:
            with gzip.open(file_path, "wt") as f:
                f.write(self.to_json())
        # Otherwise, write the SARIF log to a plain text file
        else:
            with open(file_path, "w") as f:
                f.write(self.to_json())



    def sarif_log(self):
        # Construct a SARIF log object
        log = sarif.SarifLog(
            version=sarif_version.SARIF_VERSION,
            schema_uri=sarif_version.SARIF_SCHEMA_LINK,
            # Convert each diagnostic context into SARIF format and append to the runs list
            runs=[context.sarif() for context in self.contexts],
        )
        # Append the SARIF representation of the background context to the runs list
        log.runs.append(self._background_context.sarif())
        return log
engine = ExportDiagnosticEngine()
_context = engine.background_context

# 创建一个 `ExportDiagnosticEngine` 的实例并赋给 `engine` 变量，用于导出诊断引擎。
# 将 `engine.background_context` 赋给 `_context`，作为全局的后台上下文变量。


@contextlib.contextmanager
def create_export_diagnostic_context() -> (
    Generator[infra.DiagnosticContext, None, None]
):
    """Create a diagnostic context for export.

    This is a workaround for code robustness since diagnostic context is accessed by
    export internals via global variable. See `ExportDiagnosticEngine` for more details.
    """
    global _context
    assert (
        _context == engine.background_context
    ), "Export context is already set. Nested export is not supported."
    _context = engine.create_diagnostic_context(
        "torch.onnx.export",
        torch.__version__,
    )
    try:
        yield _context
    finally:
        _context = engine.background_context

# 创建一个上下文管理器 `create_export_diagnostic_context`，用于导出诊断上下文的创建。
# 在函数内部使用全局变量 `_context`，确保当前的导出上下文未被设置，以避免嵌套导出。
# 将 `engine.create_diagnostic_context` 创建的上下文赋给 `_context`，用于指定导出上下文的具体信息。
# 使用 `yield _context` 使得函数可以被当作上下文管理器使用，返回 `_context` 给调用方。
# 最终在 `finally` 块中将 `_context` 重置为 `engine.background_context`，以确保上下文在退出时正确恢复。


def diagnose(
    rule: infra.Rule,
    level: infra.Level,
    message: Optional[str] = None,
    frames_to_skip: int = 2,
    **kwargs,
) -> TorchScriptOnnxExportDiagnostic:
    """Creates a diagnostic and record it in the global diagnostic context.

    This is a wrapper around `context.log` that uses the global diagnostic
    context.
    """
    # NOTE: Cannot use `@_beartype.beartype`. It somehow erases the cpp stack frame info.
    diagnostic = TorchScriptOnnxExportDiagnostic(
        rule, level, message, frames_to_skip=frames_to_skip, **kwargs
    )
    export_context().log(diagnostic)
    return diagnostic

# 定义函数 `diagnose`，用于创建诊断信息并记录到全局诊断上下文中。
# 创建 `TorchScriptOnnxExportDiagnostic` 类的实例 `diagnostic`，用指定的规则、级别和消息初始化。
# 调用 `export_context().log(diagnostic)` 将 `diagnostic` 记录到全局诊断上下文中。
# 返回创建的 `diagnostic` 实例。


def export_context() -> infra.DiagnosticContext:
    global _context
    return _context

# 定义函数 `export_context`，返回当前的全局导出上下文 `_context`。
# `_context` 是一个 `infra.DiagnosticContext` 类型的变量，用于管理导出时的诊断信息。
```