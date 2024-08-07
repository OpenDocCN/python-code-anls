# `.\pytorch\torch\onnx\_internal\diagnostics\infra\sarif\_version_control_details.py`

```py
# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

# 引入未来的注解支持，使得类型注解可以在 Python 3.7 及更早版本中使用
from __future__ import annotations

# 引入数据类支持，用于创建不可变对象
import dataclasses

# 引入可选类型的支持
from typing import Optional

# 从 torch.onnx._internal.diagnostics.infra.sarif 模块中引入特定成员
from torch.onnx._internal.diagnostics.infra.sarif import (
    _artifact_location,
    _property_bag,
)

# 使用数据类装饰器 dataclass 创建一个数据类 VersionControlDetails
@dataclasses.dataclass
class VersionControlDetails(object):
    """Specifies the information necessary to retrieve a desired revision from a version control system."""

    # 仓库 URI，用于指定版本控制仓库的地址
    repository_uri: str = dataclasses.field(
        metadata={"schema_property_name": "repositoryUri"}
    )

    # 可选项：指定 UTC 时间的字符串，用于指示版本的时间点
    as_of_time_utc: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "asOfTimeUtc"}
    )

    # 可选项：指定分支的名称，用于指示从哪个分支获取版本
    branch: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "branch"}
    )

    # 可选项：指定一个 ArtifactLocation 对象，用于映射到版本库中的某个特定位置
    mapped_to: Optional[_artifact_location.ArtifactLocation] = dataclasses.field(
        default=None, metadata={"schema_property_name": "mappedTo"}
    )

    # 可选项：指定一个 PropertyBag 对象，用于携带与版本相关的属性信息
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )

    # 可选项：指定版本的 ID，用于唯一标识版本
    revision_id: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "revisionId"}
    )

    # 可选项：指定版本的标签，用于描述版本的标识信息
    revision_tag: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "revisionTag"}
    )

# flake8: noqa
```