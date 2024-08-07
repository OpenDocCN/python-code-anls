# `.\pytorch\torch\onnx\_internal\diagnostics\infra\sarif\_result.py`

```py
# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

# 导入必要的模块和类
from __future__ import annotations

import dataclasses
from typing import Any, List, Literal, Optional

# 导入 SARIF 规范中定义的多个类
from torch.onnx._internal.diagnostics.infra.sarif import (
    _artifact_location,
    _attachment,
    _code_flow,
    _fix,
    _graph,
    _graph_traversal,
    _location,
    _message,
    _property_bag,
    _reporting_descriptor_reference,
    _result_provenance,
    _stack,
    _suppression,
    _web_request,
    _web_response,
)


# 使用 dataclasses 装饰器定义 Result 类
@dataclasses.dataclass
class Result(object):
    """A result produced by an analysis tool."""

    # 消息对象，使用 _message.Message 类型
    message: _message.Message = dataclasses.field(
        metadata={"schema_property_name": "message"}
    )
    # 分析目标位置，可选的 _artifact_location.ArtifactLocation 类型
    analysis_target: Optional[_artifact_location.ArtifactLocation] = dataclasses.field(
        default=None, metadata={"schema_property_name": "analysisTarget"}
    )
    # 附件列表，可选的 _attachment.Attachment 类型的列表
    attachments: Optional[List[_attachment.Attachment]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "attachments"}
    )
    # 基准状态，字符串字面值类型，取值为 "new", "unchanged", "updated", "absent" 中的一个
    baseline_state: Optional[
        Literal["new", "unchanged", "updated", "absent"]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "baselineState"}
    )
    # 代码流列表，可选的 _code_flow.CodeFlow 类型的列表
    code_flows: Optional[List[_code_flow.CodeFlow]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "codeFlows"}
    )
    # 相关性 GUID，字符串类型
    correlation_guid: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "correlationGuid"}
    )
    # 指纹信息，任意类型
    fingerprints: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "fingerprints"}
    )
    # 修复列表，可选的 _fix.Fix 类型的列表
    fixes: Optional[List[_fix.Fix]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "fixes"}
    )
    # 图遍历列表，可选的 _graph_traversal.GraphTraversal 类型的列表
    graph_traversals: Optional[
        List[_graph_traversal.GraphTraversal]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "graphTraversals"}
    )
    # 图列表，可选的 _graph.Graph 类型的列表
    graphs: Optional[List[_graph.Graph]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "graphs"}
    )
    # 唯一标识符 GUID，字符串类型
    guid: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "guid"}
    )
    # 托管的查看器 URI，字符串类型
    hosted_viewer_uri: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "hostedViewerUri"}
    )
    # 结果类型，字面值类型，取值为 "notApplicable", "pass", "fail", "review", "open", "informational" 中的一个，默认为 "fail"
    kind: Literal[
        "notApplicable", "pass", "fail", "review", "open", "informational"
    ] = dataclasses.field(default="fail", metadata={"schema_property_name": "kind"})
    # 严重程度，字面值类型，取值为 "none", "note", "warning", "error" 中的一个，默认为 "warning"
    level: Literal["none", "note", "warning", "error"] = dataclasses.field(
        default="warning", metadata={"schema_property_name": "level"}
    )
    # 位置列表，可选的 _location.Location 类型的列表
    locations: Optional[List[_location.Location]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "locations"}
    )
    # 定义一个可选的整数型变量 occurrence_count，用于存储数据出现次数，默认为 None
    occurrence_count: Optional[int] = dataclasses.field(
        default=None, metadata={"schema_property_name": "occurrenceCount"}
    )
    # 定义一个任意类型的变量 partial_fingerprints，存储部分指纹信息，默认为 None
    partial_fingerprints: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "partialFingerprints"}
    )
    # 定义一个可选的 _property_bag.PropertyBag 对象 properties，存储属性信息，默认为 None
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    # 定义一个可选的 _result_provenance.ResultProvenance 对象 provenance，存储结果来源信息，默认为 None
    provenance: Optional[_result_provenance.ResultProvenance] = dataclasses.field(
        default=None, metadata={"schema_property_name": "provenance"}
    )
    # 定义一个浮点数变量 rank，存储排名信息，默认为 -1.0
    rank: float = dataclasses.field(
        default=-1.0, metadata={"schema_property_name": "rank"}
    )
    # 定义一个可选的 _location.Location 对象列表 related_locations，存储相关位置信息，默认为 None
    related_locations: Optional[List[_location.Location]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "relatedLocations"}
    )
    # 定义一个可选的 _reporting_descriptor_reference.ReportingDescriptorReference 对象 rule，存储规则信息，默认为 None
    rule: Optional[
        _reporting_descriptor_reference.ReportingDescriptorReference
    ] = dataclasses.field(default=None, metadata={"schema_property_name": "rule"})
    # 定义一个可选的字符串变量 rule_id，存储规则ID信息，默认为 None
    rule_id: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "ruleId"}
    )
    # 定义一个整数型变量 rule_index，存储规则索引信息，默认为 -1
    rule_index: int = dataclasses.field(
        default=-1, metadata={"schema_property_name": "ruleIndex"}
    )
    # 定义一个可选的 _stack.Stack 对象列表 stacks，存储堆栈信息，默认为 None
    stacks: Optional[List[_stack.Stack]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "stacks"}
    )
    # 定义一个可选的 _suppression.Suppression 对象列表 suppressions，存储抑制信息，默认为 None
    suppressions: Optional[List[_suppression.Suppression]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "suppressions"}
    )
    # 定义一个可选的 _reporting_descriptor_reference.ReportingDescriptorReference 对象列表 taxa，存储分类信息，默认为 None
    taxa: Optional[
        List[_reporting_descriptor_reference.ReportingDescriptorReference]
    ] = dataclasses.field(default=None, metadata={"schema_property_name": "taxa"})
    # 定义一个可选的 _web_request.WebRequest 对象 web_request，存储网络请求信息，默认为 None
    web_request: Optional[_web_request.WebRequest] = dataclasses.field(
        default=None, metadata={"schema_property_name": "webRequest"}
    )
    # 定义一个可选的 _web_response.WebResponse 对象 web_response，存储网络响应信息，默认为 None
    web_response: Optional[_web_response.WebResponse] = dataclasses.field(
        default=None, metadata={"schema_property_name": "webResponse"}
    )
    # 定义一个可选的字符串列表 work_item_uris，存储工作项URI信息，默认为 None
    work_item_uris: Optional[List[str]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "workItemUris"}
    )
# flake8: noqa
```