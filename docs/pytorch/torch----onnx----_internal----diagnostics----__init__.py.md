# `.\pytorch\torch\onnx\_internal\diagnostics\__init__.py`

```
# 从 _diagnostic 模块中导入以下函数和类：
# - create_export_diagnostic_context: 创建导出诊断上下文的函数
# - diagnose: 诊断函数
# - engine: 引擎对象
# - export_context: 导出上下文对象
# - ExportDiagnosticEngine: 导出诊断引擎类
# - TorchScriptOnnxExportDiagnostic: TorchScript 到 ONNX 导出的诊断类
# 从 _rules 模块中导入 rules 变量
# 从 infra 模块中导入 levels 变量
from ._diagnostic import (
    create_export_diagnostic_context,
    diagnose,
    engine,
    export_context,
    ExportDiagnosticEngine,
    TorchScriptOnnxExportDiagnostic,
)
from ._rules import rules
from .infra import levels

# 将以下标识符添加到模块的导出列表中，以便外部访问：
# - TorchScriptOnnxExportDiagnostic: TorchScript 到 ONNX 导出的诊断类
# - ExportDiagnosticEngine: 导出诊断引擎类
# - rules: 规则变量
# - levels: 等级变量
# - engine: 引擎对象
# - export_context: 导出上下文对象
# - create_export_diagnostic_context: 创建导出诊断上下文的函数
# - diagnose: 诊断函数
__all__ = [
    "TorchScriptOnnxExportDiagnostic",
    "ExportDiagnosticEngine",
    "rules",
    "levels",
    "engine",
    "export_context",
    "create_export_diagnostic_context",
    "diagnose",
]
```