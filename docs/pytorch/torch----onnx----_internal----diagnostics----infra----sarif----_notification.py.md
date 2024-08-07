# `.\pytorch\torch\onnx\_internal\diagnostics\infra\sarif\_notification.py`

```py
# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

# 导入必要的模块
from __future__ import annotations

import dataclasses
from typing import List, Literal, Optional

# 导入来自torch.onnx._internal.diagnostics.infra.sarif模块的特定对象
from torch.onnx._internal.diagnostics.infra.sarif import (
    _exception,
    _location,
    _message,
    _property_bag,
    _reporting_descriptor_reference,
)

# 使用dataclasses装饰器定义通知(Notification)类
@dataclasses.dataclass
class Notification(object):
    """Describes a condition relevant to the tool itself, as opposed to being relevant to a target being analyzed by the tool."""

    # 消息对象，指定其对应的架构属性名为"message"
    message: _message.Message = dataclasses.field(
        metadata={"schema_property_name": "message"}
    )
    # 可选字段，关联的规则描述符引用，对应的架构属性名为"associatedRule"
    associated_rule: Optional[
        _reporting_descriptor_reference.ReportingDescriptorReference
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "associatedRule"}
    )
    # 可选字段，描述符引用，对应的架构属性名为"descriptor"
    descriptor: Optional[
        _reporting_descriptor_reference.ReportingDescriptorReference
    ] = dataclasses.field(default=None, metadata={"schema_property_name": "descriptor"})
    # 可选字段，异常对象，对应的架构属性名为"exception"
    exception: Optional[_exception.Exception] = dataclasses.field(
        default=None, metadata={"schema_property_name": "exception"}
    )
    # 字面量类型字段，表示级别，对应的架构属性名为"level"
    level: Literal["none", "note", "warning", "error"] = dataclasses.field(
        default="warning", metadata={"schema_property_name": "level"}
    )
    # 可选字段，位置列表，元素为_location.Location类型，对应的架构属性名为"locations"
    locations: Optional[List[_location.Location]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "locations"}
    )
    # 可选字段，属性包对象，对应的架构属性名为"properties"
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    # 可选字段，线程ID，对应的架构属性名为"threadId"
    thread_id: Optional[int] = dataclasses.field(
        default=None, metadata={"schema_property_name": "threadId"}
    )
    # 可选字段，UTC时间字符串，对应的架构属性名为"timeUtc"
    time_utc: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "timeUtc"}
    )

# 禁用flake8对本文件的检查，以确保生成的代码不受格式检查影响
# flake8: noqa
```