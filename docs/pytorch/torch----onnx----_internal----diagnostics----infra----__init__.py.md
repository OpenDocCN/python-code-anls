# `.\pytorch\torch\onnx\_internal\diagnostics\infra\__init__.py`

```py
# 从._infra模块中导入多个类和变量，用于当前模块的使用
from ._infra import (
    DiagnosticOptions,    # 导入诊断选项类
    Graph,                # 导入图类
    Invocation,           # 导入调用类
    Level,                # 导入级别类
    levels,               # 导入级别变量
    Location,             # 导入位置类
    Rule,                 # 导入规则类
    RuleCollection,       # 导入规则集合类
    Stack,                # 导入栈类
    StackFrame,           # 导入栈帧类
    Tag,                  # 导入标签类
    ThreadFlowLocation,   # 导入线程流位置类
)

# 从.context模块中导入特定的类，用于当前模块的使用
from .context import Diagnostic, DiagnosticContext, RuntimeErrorWithDiagnostic

# __all__列表指定了在使用from ... import *语句时导入的符号（类、函数等）
__all__ = [
    "Diagnostic",                   # 导出诊断类
    "DiagnosticContext",            # 导出诊断上下文类
    "DiagnosticOptions",            # 导出诊断选项类
    "Graph",                        # 导出图类
    "Invocation",                   # 导出调用类
    "Level",                        # 导出级别类
    "levels",                       # 导出级别变量
    "Location",                     # 导出位置类
    "Rule",                         # 导出规则类
    "RuleCollection",               # 导出规则集合类
    "RuntimeErrorWithDiagnostic",   # 导出带有诊断的运行时错误类
    "Stack",                        # 导出栈类
    "StackFrame",                   # 导出栈帧类
    "Tag",                          # 导出标签类
    "ThreadFlowLocation",           # 导出线程流位置类
]
```