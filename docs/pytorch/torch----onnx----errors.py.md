# `.\pytorch\torch\onnx\errors.py`

```
# 引入ONNX导出器的异常模块
"""ONNX exporter exceptions."""

# 启用Future特性，允许在类中使用'annotations'
from __future__ import annotations

# 引入标准库中的textwrap模块和Optional类型提示
import textwrap
from typing import Optional

# 引入torch库中的_C模块，和torch.onnx模块中的常量_constants
from torch import _C
from torch.onnx import _constants
# 引入torch.onnx._internal中的诊断模块
from torch.onnx._internal import diagnostics

# 指定公开的异常类名列表
__all__ = [
    "OnnxExporterError",
    "OnnxExporterWarning",
    "CheckerError",
    "SymbolicValueError",
    "UnsupportedOperatorError",
]

# ONNX导出器警告基类
class OnnxExporterWarning(UserWarning):
    """Base class for all warnings in the ONNX exporter."""
    pass

# ONNX导出器运行时错误基类
class OnnxExporterError(RuntimeError):
    """Errors raised by the ONNX exporter."""
    pass

# ONNX检查器错误类，当ONNX检查器检测到无效模型时引发
class CheckerError(OnnxExporterError):
    """Raised when ONNX checker detects an invalid model."""
    pass

# 不支持的运算符错误类，当导出器不支持某运算符时引发
class UnsupportedOperatorError(OnnxExporterError):
    """Raised when an operator is unsupported by the exporter."""

    def __init__(self, name: str, version: int, supported_version: Optional[int]):
        # 如果支持的版本不为None，则生成诊断信息
        if supported_version is not None:
            diagnostic_rule: diagnostics.infra.Rule = (
                diagnostics.rules.operator_supported_in_newer_opset_version
            )
            msg = diagnostic_rule.format_message(name, version, supported_version)
            diagnostics.diagnose(diagnostic_rule, diagnostics.levels.ERROR, msg)
        else:
            # 如果运算符名称以"aten::", "prim::", "quantized::"开头，则生成标准符号函数缺失的诊断信息
            if name.startswith(("aten::", "prim::", "quantized::")):
                diagnostic_rule = diagnostics.rules.missing_standard_symbolic_function
                msg = diagnostic_rule.format_message(
                    name, version, _constants.PYTORCH_GITHUB_ISSUES_URL
                )
                diagnostics.diagnose(diagnostic_rule, diagnostics.levels.ERROR, msg)
            else:
                # 否则生成自定义符号函数缺失的诊断信息
                diagnostic_rule = diagnostics.rules.missing_custom_symbolic_function
                msg = diagnostic_rule.format_message(name)
                diagnostics.diagnose(diagnostic_rule, diagnostics.levels.ERROR, msg)
        
        # 调用父类初始化方法，传递错误信息msg
        super().__init__(msg)

# 符号值错误类，围绕TorchScript值和节点的错误
class SymbolicValueError(OnnxExporterError):
    """Errors around TorchScript values and nodes."""
    def __init__(self, msg: str, value: _C.Value):
        # 构造错误消息，描述由 TorchScript 图中的值引起的问题
        message = (
            f"{msg}  [Caused by the value '{value}' (type '{value.type()}') in the "
            f"TorchScript graph. The containing node has kind '{value.node().kind()}'.] "
        )

        # 获取值节点的代码位置范围，并将其添加到消息中
        code_location = value.node().sourceRange()
        if code_location:
            message += f"\n    (node defined in {code_location})"

        try:
            # 将值节点的输入和输出添加到消息中
            message += "\n\n"
            message += textwrap.indent(
                (
                    "Inputs:\n"
                    + (
                        "\n".join(
                            f"    #{i}: {input_}  (type '{input_.type()}')"
                            for i, input_ in enumerate(value.node().inputs())
                        )
                        or "    Empty"
                    )
                    + "\n"
                    + "Outputs:\n"
                    + (
                        "\n".join(
                            f"    #{i}: {output}  (type '{output.type()}')"
                            for i, output in enumerate(value.node().outputs())
                        )
                        or "    Empty"
                    )
                ),
                "    ",
            )
        except AttributeError:
            # 如果无法获取输入和输出，则记录错误消息
            message += (
                " Failed to obtain its input and output for debugging. "
                "Please refer to the TorchScript graph for debugging information."
            )

        # 调用父类的初始化方法，传递构造好的错误消息
        super().__init__(message)
```