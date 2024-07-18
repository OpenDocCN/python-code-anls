# `.\graphrag\graphrag\index\context.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# isort: skip_file
"""A module containing the 'PipelineRunStats' and 'PipelineRunContext' models."""

# 导入 dataclass 模块中的 dataclass 和 field 函数
from dataclasses import dataclass as dc_dataclass
from dataclasses import field

# 从当前包中导入 PipelineCache 类
from .cache import PipelineCache
# 从当前包中导入 PipelineStorage 类
from .storage.typing import PipelineStorage


@dc_dataclass
class PipelineRunStats:
    """Pipeline running stats."""

    # 表示总运行时间的浮点数，默认为 0
    total_runtime: float = field(default=0)
    """Float representing the total runtime."""

    # 表示文档数量的整数，默认为 0
    num_documents: int = field(default=0)
    """Number of documents."""

    # 表示输入加载时间的浮点数，默认为 0
    input_load_time: float = field(default=0)
    """Float representing the input load time."""

    # 一个字典，存储工作流的详细信息，结构为 dict[str, dict[str, float]]，默认为空字典
    workflows: dict[str, dict[str, float]] = field(default_factory=dict)
    """A dictionary of workflows."""


@dc_dataclass
class PipelineRunContext:
    """Provides the context for the current pipeline run."""

    # 包含 PipelineRunStats 对象的属性，用于存储运行统计信息
    stats: PipelineRunStats
    # 包含 PipelineStorage 对象的属性，用于提供存储相关的功能
    storage: PipelineStorage
    # 包含 PipelineCache 对象的属性，用于提供缓存相关的功能
    cache: PipelineCache


# TODO: For now, just has the same props available to it
# VerbRunContext 类型别名，与 PipelineRunContext 具有相同的属性结构
VerbRunContext = PipelineRunContext
"""Provides the context for the current verb run."""
```