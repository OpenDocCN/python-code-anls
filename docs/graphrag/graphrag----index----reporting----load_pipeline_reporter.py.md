# `.\graphrag\graphrag\index\reporting\load_pipeline_reporter.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Load pipeline reporter method."""

from pathlib import Path
from typing import cast

from datashaper import WorkflowCallbacks  # 导入WorkflowCallbacks类，用于定义回调接口

from graphrag.config import ReportingType  # 导入ReportingType枚举
from graphrag.index.config import (
    PipelineBlobReportingConfig,  # 导入PipelineBlobReportingConfig类，用于Blob类型报告配置
    PipelineFileReportingConfig,  # 导入PipelineFileReportingConfig类，用于File类型报告配置
    PipelineReportingConfig,  # 导入PipelineReportingConfig类，定义了报告配置的通用接口
)

from .blob_workflow_callbacks import BlobWorkflowCallbacks  # 导入BlobWorkflowCallbacks类，用于Blob类型的工作流回调
from .console_workflow_callbacks import ConsoleWorkflowCallbacks  # 导入ConsoleWorkflowCallbacks类，用于控制台类型的工作流回调
from .file_workflow_callbacks import FileWorkflowCallbacks  # 导入FileWorkflowCallbacks类，用于File类型的工作流回调


def load_pipeline_reporter(
    config: PipelineReportingConfig | None, root_dir: str | None
) -> WorkflowCallbacks:
    """Create a reporter for the given pipeline config."""
    # 如果config为None，则使用默认的PipelineFileReportingConfig配置对象
    config = config or PipelineFileReportingConfig(base_dir="reports")

    match config.type:  # 根据config对象的type属性进行匹配
        case ReportingType.file:  # 如果type为file
            config = cast(PipelineFileReportingConfig, config)  # 将config强制转换为PipelineFileReportingConfig类型
            # 返回一个FileWorkflowCallbacks对象，根据root_dir和config.base_dir构造路径
            return FileWorkflowCallbacks(
                str(Path(root_dir or "") / (config.base_dir or ""))
            )
        case ReportingType.console:  # 如果type为console
            # 返回一个ConsoleWorkflowCallbacks对象，不需要额外的配置
            return ConsoleWorkflowCallbacks()
        case ReportingType.blob:  # 如果type为blob
            config = cast(PipelineBlobReportingConfig, config)  # 将config强制转换为PipelineBlobReportingConfig类型
            # 返回一个BlobWorkflowCallbacks对象，使用配置中的connection_string、container_name等参数
            return BlobWorkflowCallbacks(
                config.connection_string,
                config.container_name,
                base_dir=config.base_dir,
                storage_account_blob_url=config.storage_account_blob_url,
            )
        case _:  # 对于其他未知的type
            msg = f"Unknown reporting type: {config.type}"  # 构造错误消息
            raise ValueError(msg)  # 抛出ValueError异常，提示未知的报告类型
```