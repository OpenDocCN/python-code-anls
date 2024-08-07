# `.\pytorch\torch\onnx\_internal\diagnostics\infra\sarif\_run_automation_details.py`

```py
# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.
# 导入必要的模块和类

from __future__ import annotations

import dataclasses  # 导入用于定义数据类的模块
from typing import Optional  # 导入用于类型提示的 Optional 类型

from torch.onnx._internal.diagnostics.infra.sarif import _message, _property_bag  # 导入用于 SARIF 文件处理的模块


@dataclasses.dataclass
class RunAutomationDetails(object):
    """Information that describes a run's identity and role within an engineering system process."""
    
    correlation_guid: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "correlationGuid"}
    )  # 用于表示相关性 GUID 的可选字符串字段

    description: Optional[_message.Message] = dataclasses.field(
        default=None, metadata={"schema_property_name": "description"}
    )  # 用于表示描述信息的可选 _message.Message 对象字段

    guid: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "guid"}
    )  # 用于表示 GUID 的可选字符串字段

    id: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "id"}
    )  # 用于表示 ID 的可选字符串字段

    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )  # 用于表示属性集合的可选 _property_bag.PropertyBag 对象字段

# flake8: noqa
```