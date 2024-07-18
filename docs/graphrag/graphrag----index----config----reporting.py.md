# `.\graphrag\graphrag\index\config\reporting.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'PipelineReportingConfig', 'PipelineFileReportingConfig' and 'PipelineConsoleReportingConfig' models."""

# 导入所需模块和类
from __future__ import annotations

from typing import Generic, Literal, TypeVar

# 导入 Pydantic 模块和 Field 类
from pydantic import BaseModel
from pydantic import Field as pydantic_Field

# 导入自定义的 ReportingType 枚举类型
from graphrag.config.enums import ReportingType

# 创建类型变量 T
T = TypeVar("T")


# 定义 PipelineReportingConfig 泛型基类模型，表示管道的报告配置
class PipelineReportingConfig(BaseModel, Generic[T]):
    """Represent the reporting configuration for the pipeline."""
    
    # 报告类型，泛型参数 T
    type: T


# 定义 PipelineFileReportingConfig 派生类模型，表示管道的文件报告配置
class PipelineFileReportingConfig(PipelineReportingConfig[Literal[ReportingType.file]]):
    """Represent the file reporting configuration for the pipeline."""
    
    # 报告类型为文件类型
    type: Literal[ReportingType.file] = ReportingType.file
    """The type of reporting."""

    # 基础目录路径，用于报告
    base_dir: str | None = pydantic_Field(
        description="The base directory for the reporting.", default=None
    )
    """The base directory for the reporting."""


# 定义 PipelineConsoleReportingConfig 派生类模型，表示管道的控制台报告配置
class PipelineConsoleReportingConfig(
    PipelineReportingConfig[Literal[ReportingType.console]]
):
    """Represent the console reporting configuration for the pipeline."""
    
    # 报告类型为控制台类型
    type: Literal[ReportingType.console] = ReportingType.console
    """The type of reporting."""


# 定义 PipelineBlobReportingConfig 派生类模型，表示管道的 Blob 存储报告配置
class PipelineBlobReportingConfig(PipelineReportingConfig[Literal[ReportingType.blob]]):
    """Represents the blob reporting configuration for the pipeline."""
    
    # 报告类型为 Blob 存储类型
    type: Literal[ReportingType.blob] = ReportingType.blob
    """The type of reporting."""

    # Blob 存储连接字符串
    connection_string: str | None = pydantic_Field(
        description="The blob reporting connection string for the reporting.",
        default=None,
    )
    """The blob reporting connection string for the reporting."""

    # Blob 容器名称
    container_name: str = pydantic_Field(
        description="The container name for reporting", default=None
    )
    """The container name for reporting"""

    # 存储账户 Blob URL
    storage_account_blob_url: str | None = pydantic_Field(
        description="The storage account blob url for reporting", default=None
    )
    """The storage account blob url for reporting"""

    # 基础目录路径，用于报告
    base_dir: str | None = pydantic_Field(
        description="The base directory for the reporting.", default=None
    )
    """The base directory for the reporting."""


# 定义一个联合类型，包含所有可能的 PipelineReportingConfig 派生类
PipelineReportingConfigTypes = (
    PipelineFileReportingConfig
    | PipelineConsoleReportingConfig
    | PipelineBlobReportingConfig
)
```