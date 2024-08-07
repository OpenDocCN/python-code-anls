# `.\pytorch\torch\onnx\_internal\diagnostics\infra\sarif\_physical_location.py`

```py
# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.
# 从未来导入必要的注释功能

from __future__ import annotations

# 导入必要的模块和类
import dataclasses
from typing import Optional

# 从 torch.onnx._internal.diagnostics.infra.sarif 中导入以下几个类
from torch.onnx._internal.diagnostics.infra.sarif import (
    _address,
    _artifact_location,
    _property_bag,
    _region,
)

# 使用 dataclasses 装饰器定义 PhysicalLocation 类
@dataclasses.dataclass
class PhysicalLocation(object):
    """A physical location relevant to a result. Specifies a reference to a programming artifact together with a range of bytes or characters within that artifact."""

    # 物理地址，可选的 _address.Address 类型，默认为 None
    address: Optional[_address.Address] = dataclasses.field(
        default=None, metadata={"schema_property_name": "address"}
    )
    # 与结果相关的编程工件的物理位置，可选的 _artifact_location.ArtifactLocation 类型，默认为 None
    artifact_location: Optional[
        _artifact_location.ArtifactLocation
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "artifactLocation"}
    )
    # 上下文区域，可选的 _region.Region 类型，默认为 None
    context_region: Optional[_region.Region] = dataclasses.field(
        default=None, metadata={"schema_property_name": "contextRegion"}
    )
    # 属性集合，可选的 _property_bag.PropertyBag 类型，默认为 None
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    # 区域，可选的 _region.Region 类型，默认为 None
    region: Optional[_region.Region] = dataclasses.field(
        default=None, metadata={"schema_property_name": "region"}
    )

# flake8: noqa
```