# `.\pytorch\torch\onnx\_internal\diagnostics\infra\sarif\_artifact_change.py`

```py
# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.
# 导入必要的模块和类
from __future__ import annotations

import dataclasses
from typing import List, Optional

# 导入需要的模块和类
from torch.onnx._internal.diagnostics.infra.sarif import (
    _artifact_location,
    _property_bag,
    _replacement,
)

# 使用 dataclasses 装饰器定义一个数据类 ArtifactChange
@dataclasses.dataclass
class ArtifactChange(object):
    """A change to a single artifact."""

    # 定义 artifact_location 属性，指定元数据 schema_property_name
    artifact_location: _artifact_location.ArtifactLocation = dataclasses.field(
        metadata={"schema_property_name": "artifactLocation"}
    )
    # 定义 replacements 属性，指定元数据 schema_property_name
    replacements: List[_replacement.Replacement] = dataclasses.field(
        metadata={"schema_property_name": "replacements"}
    )
    # 定义 properties 属性，可选，默认为 None，指定元数据 schema_property_name
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )

# flake8: noqa
# 禁用 flake8 对此文件的检查
```