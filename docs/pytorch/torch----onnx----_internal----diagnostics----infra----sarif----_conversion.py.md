# `.\pytorch\torch\onnx\_internal\diagnostics\infra\sarif\_conversion.py`

```py
# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

# 导入必要的模块和类
from __future__ import annotations

import dataclasses
from typing import List, Optional

# 导入 SARIF 相关模块
from torch.onnx._internal.diagnostics.infra.sarif import (
    _artifact_location,
    _invocation,
    _property_bag,
    _tool,
)


@dataclasses.dataclass
class Conversion(object):
    """描述转换器如何将静态分析工具的输出从其原生输出格式转换为 SARIF 格式。"""

    # 工具对象，用于描述进行转换的工具
    tool: _tool.Tool = dataclasses.field(metadata={"schema_property_name": "tool"})
    
    # 可选字段：静态分析工具的日志文件列表
    analysis_tool_log_files: Optional[
        List[_artifact_location.ArtifactLocation]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "analysisToolLogFiles"}
    )
    
    # 可选字段：描述转换操作的调用信息
    invocation: Optional[_invocation.Invocation] = dataclasses.field(
        default=None, metadata={"schema_property_name": "invocation"}
    )
    
    # 可选字段：其他属性信息
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )


# flake8: noqa
```