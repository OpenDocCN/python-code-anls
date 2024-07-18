# `.\graphrag\graphrag\index\config\workflow.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'PipelineWorkflowReference' model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import Field as pydantic_Field

# 定义 PipelineWorkflowStep 类型为字典，表示工作流程中的一个步骤
PipelineWorkflowStep = dict[str, Any]
"""Represent a step in a workflow."""

# 定义 PipelineWorkflowConfig 类型为字典，表示工作流程的配置
PipelineWorkflowConfig = dict[str, Any]
"""Represent a configuration for a workflow."""

# 定义 PipelineWorkflowReference 类，继承自 Pydantic 的 BaseModel
class PipelineWorkflowReference(BaseModel):
    """Represent a reference to a workflow, and can optionally be the workflow itself."""

    # 工作流的名称，可选项，默认为 None
    name: str | None = pydantic_Field(description="Name of the workflow.", default=None)
    """Name of the workflow."""

    # 工作流的步骤列表，可选项，默认为 None
    steps: list[PipelineWorkflowStep] | None = pydantic_Field(
        description="The optional steps for the workflow.", default=None
    )
    """The optional steps for the workflow."""

    # 工作流的配置信息，可选项，默认为 None
    config: PipelineWorkflowConfig | None = pydantic_Field(
        description="The optional configuration for the workflow.", default=None
    )
    """The optional configuration for the workflow."""
```