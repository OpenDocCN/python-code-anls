# `.\pytorch\torch\onnx\_internal\diagnostics\infra\sarif\_run.py`

```py
# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

# 引入必要的模块和类
from __future__ import annotations

import dataclasses
from typing import Any, List, Literal, Optional

# 从特定模块中导入多个类
from torch.onnx._internal.diagnostics.infra.sarif import (
    _address,
    _artifact,
    _conversion,
    _external_property_file_references,
    _graph,
    _invocation,
    _logical_location,
    _property_bag,
    _result,
    _run_automation_details,
    _special_locations,
    _thread_flow_location,
    _tool,
    _tool_component,
    _version_control_details,
    _web_request,
    _web_response,
)

# 使用dataclasses装饰器定义类
@dataclasses.dataclass
class Run(object):
    """Describes a single run of an analysis tool, and contains the reported output of that run."""

    # 描述工具运行的对象
    tool: _tool.Tool = dataclasses.field(metadata={"schema_property_name": "tool"})
    # 可选的地址列表
    addresses: Optional[List[_address.Address]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "addresses"}
    )
    # 可选的工件列表
    artifacts: Optional[List[_artifact.Artifact]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "artifacts"}
    )
    # 可选的自动化详情对象
    automation_details: Optional[
        _run_automation_details.RunAutomationDetails
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "automationDetails"}
    )
    # 可选的基线GUID字符串
    baseline_guid: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "baselineGuid"}
    )
    # 列的种类，可以是utf16CodeUnits或unicodeCodePoints之一
    column_kind: Optional[
        Literal["utf16CodeUnits", "unicodeCodePoints"]
    ] = dataclasses.field(default=None, metadata={"schema_property_name": "columnKind"})
    # 可选的转换对象
    conversion: Optional[_conversion.Conversion] = dataclasses.field(
        default=None, metadata={"schema_property_name": "conversion"}
    )
    # 可选的默认编码字符串
    default_encoding: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "defaultEncoding"}
    )
    # 可选的默认源语言字符串
    default_source_language: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "defaultSourceLanguage"}
    )
    # 可选的外部属性文件引用对象
    external_property_file_references: Optional[
        _external_property_file_references.ExternalPropertyFileReferences
    ] = dataclasses.field(
        default=None,
        metadata={"schema_property_name": "externalPropertyFileReferences"},
    )
    # 可选的图列表
    graphs: Optional[List[_graph.Graph]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "graphs"}
    )
    # 可选的调用列表
    invocations: Optional[List[_invocation.Invocation]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "invocations"}
    )
    # 语言字符串，默认为en-US
    language: str = dataclasses.field(
        default="en-US", metadata={"schema_property_name": "language"}
    )
    # 可选的逻辑位置列表
    logical_locations: Optional[
        List[_logical_location.LogicalLocation]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "logicalLocations"}
    )
    # 定义一个字符串列表类型的字段，用于存储换行符序列，默认包括"\r\n"和"\n"
    newline_sequences: List[str] = dataclasses.field(
        default_factory=lambda: ["\r\n", "\n"],
        metadata={"schema_property_name": "newlineSequences"},
    )
    # 定义一个任意类型的字段，用于存储原始 URI 基础 ID，初始值为 None
    original_uri_base_ids: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "originalUriBaseIds"}
    )
    # 定义一个可选的工具组件列表类型的字段，用于存储策略信息，初始值为 None
    policies: Optional[List[_tool_component.ToolComponent]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "policies"}
    )
    # 定义一个可选的属性包类型的字段，用于存储属性信息，初始值为 None
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    # 定义一个可选的字符串列表类型的字段，用于存储用于红action的token列表，初始值为 None
    redaction_tokens: Optional[List[str]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "redactionTokens"}
    )
    # 定义一个可选的结果列表类型的字段，用于存储结果信息，初始值为 None
    results: Optional[List[_result.Result]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "results"}
    )
    # 定义一个可选的运行聚合详情列表类型的字段，用于存储运行聚合信息，初始值为 None
    run_aggregates: Optional[
        List[_run_automation_details.RunAutomationDetails]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "runAggregates"}
    )
    # 定义一个可选的特殊位置信息类型的字段，用于存储特殊位置信息，初始值为 None
    special_locations: Optional[
        _special_locations.SpecialLocations
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "specialLocations"}
    )
    # 定义一个可选的工具组件列表类型的字段，用于存储分类信息，初始值为 None
    taxonomies: Optional[List[_tool_component.ToolComponent]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "taxonomies"}
    )
    # 定义一个可选的线程流位置列表类型的字段，用于存储线程流位置信息，初始值为 None
    thread_flow_locations: Optional[
        List[_thread_flow_location.ThreadFlowLocation]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "threadFlowLocations"}
    )
    # 定义一个可选的工具组件列表类型的字段，用于存储翻译信息，初始值为 None
    translations: Optional[List[_tool_component.ToolComponent]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "translations"}
    )
    # 定义一个可选的版本控制详情列表类型的字段，用于存储版本控制来源信息，初始值为 None
    version_control_provenance: Optional[
        List[_version_control_details.VersionControlDetails]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "versionControlProvenance"}
    )
    # 定义一个可选的 Web 请求列表类型的字段，用于存储 Web 请求信息，初始值为 None
    web_requests: Optional[List[_web_request.WebRequest]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "webRequests"}
    )
    # 定义一个可选的 Web 响应列表类型的字段，用于存储 Web 响应信息，初始值为 None
    web_responses: Optional[List[_web_response.WebResponse]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "webResponses"}
    )
# flake8: noqa
```