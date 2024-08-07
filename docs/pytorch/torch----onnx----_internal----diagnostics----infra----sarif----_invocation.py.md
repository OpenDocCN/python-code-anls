# `.\pytorch\torch\onnx\_internal\diagnostics\infra\sarif\_invocation.py`

```py
# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

# 导入必要的模块
from __future__ import annotations

import dataclasses  # 导入用于数据类的模块
from typing import Any, List, Optional  # 导入类型提示相关的类

# 导入 SARIF 格式相关的类
from torch.onnx._internal.diagnostics.infra.sarif import (
    _artifact_location,
    _configuration_override,
    _notification,
    _property_bag,
)

# 定义数据类 Invocation，用于表示分析工具运行时的执行环境
@dataclasses.dataclass
class Invocation(object):
    """The runtime environment of the analysis tool run."""

    # 标识分析工具运行是否成功
    execution_successful: bool = dataclasses.field(
        metadata={"schema_property_name": "executionSuccessful"}
    )
    # 可选字段：运行所在的账户
    account: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "account"}
    )
    # 可选字段：运行时的命令行参数列表
    arguments: Optional[List[str]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "arguments"}
    )
    # 可选字段：运行时的命令行
    command_line: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "commandLine"}
    )
    # 可选字段：运行结束的 UTC 时间
    end_time_utc: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "endTimeUtc"}
    )
    # 可选字段：运行时的环境变量
    environment_variables: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "environmentVariables"}
    )
    # 可选字段：可执行文件的位置信息
    executable_location: Optional[
        _artifact_location.ArtifactLocation
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "executableLocation"}
    )
    # 可选字段：运行结束时的退出码
    exit_code: Optional[int] = dataclasses.field(
        default=None, metadata={"schema_property_name": "exitCode"}
    )
    # 可选字段：退出码的描述信息
    exit_code_description: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "exitCodeDescription"}
    )
    # 可选字段：退出信号的名称
    exit_signal_name: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "exitSignalName"}
    )
    # 可选字段：退出信号的编号
    exit_signal_number: Optional[int] = dataclasses.field(
        default=None, metadata={"schema_property_name": "exitSignalNumber"}
    )
    # 可选字段：运行的机器名称
    machine: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "machine"}
    )
    # 可选字段：通知配置的覆盖项列表
    notification_configuration_overrides: Optional[
        List[_configuration_override.ConfigurationOverride]
    ] = dataclasses.field(
        default=None,
        metadata={"schema_property_name": "notificationConfigurationOverrides"},
    )
    # 可选字段：进程的 ID
    process_id: Optional[int] = dataclasses.field(
        default=None, metadata={"schema_property_name": "processId"}
    )
    # 可选字段：进程启动失败时的错误消息
    process_start_failure_message: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "processStartFailureMessage"}
    )
    # 可选字段：属性信息
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    # 可选字段：响应文件的位置信息列表
    response_files: Optional[
        List[_artifact_location.ArtifactLocation]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "responseFiles"}
    )
    # 定义一个可选的规则配置覆盖列表，用于配置规则的特定设置
    rule_configuration_overrides: Optional[
        List[_configuration_override.ConfigurationOverride]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "ruleConfigurationOverrides"}
    )
    
    # 定义一个可选的 UTC 开始时间字符串，用于记录任务或过程的开始时间
    start_time_utc: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "startTimeUtc"}
    )
    
    # 定义一个可选的标准错误输出位置对象，用于指定标准错误输出的存储位置
    stderr: Optional[_artifact_location.ArtifactLocation] = dataclasses.field(
        default=None, metadata={"schema_property_name": "stderr"}
    )
    
    # 定义一个可选的标准输入位置对象，用于指定标准输入的存储位置
    stdin: Optional[_artifact_location.ArtifactLocation] = dataclasses.field(
        default=None, metadata={"schema_property_name": "stdin"}
    )
    
    # 定义一个可选的标准输出位置对象，用于指定标准输出的存储位置
    stdout: Optional[_artifact_location.ArtifactLocation] = dataclasses.field(
        default=None, metadata={"schema_property_name": "stdout"}
    )
    
    # 定义一个可选的标准输出与标准错误输出位置对象，用于指定二者的存储位置
    stdout_stderr: Optional[_artifact_location.ArtifactLocation] = dataclasses.field(
        default=None, metadata={"schema_property_name": "stdoutStderr"}
    )
    
    # 定义一个可选的工具配置通知列表，用于记录与工具配置相关的通知事件
    tool_configuration_notifications: Optional[
        List[_notification.Notification]
    ] = dataclasses.field(
        default=None,
        metadata={"schema_property_name": "toolConfigurationNotifications"},
    )
    
    # 定义一个可选的工具执行通知列表，用于记录与工具执行相关的通知事件
    tool_execution_notifications: Optional[
        List[_notification.Notification]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "toolExecutionNotifications"}
    )
    
    # 定义一个可选的工作目录位置对象，用于指定任务或过程的工作目录
    working_directory: Optional[
        _artifact_location.ArtifactLocation
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "workingDirectory"}
    )
# flake8: noqa
```