# `.\pytorch\torch\onnx\_internal\diagnostics\infra\sarif\_message.py`

```py
# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

# 导入必要的模块和类
from __future__ import annotations  # 使用未来的类型注解特性

import dataclasses  # 导入dataclasses模块，用于创建数据类
from typing import List, Optional  # 导入List和Optional类型提示

from torch.onnx._internal.diagnostics.infra.sarif import _property_bag  # 导入自定义模块中的_property_bag类


@dataclasses.dataclass
class Message(object):
    """Encapsulates a message intended to be read by the end user."""

    arguments: Optional[List[str]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "arguments"}  # 定义字段arguments，可选的字符串列表，默认为None
    )
    id: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "id"}  # 定义字段id，可选的字符串，默认为None
    )
    markdown: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "markdown"}  # 定义字段markdown，可选的字符串，默认为None
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}  # 定义字段properties，可选的_property_bag.PropertyBag对象，默认为None
    )
    text: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "text"}  # 定义字段text，可选的字符串，默认为None
    )


# flake8: noqa  # 忽略flake8的警告
```