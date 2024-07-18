# `.\graphrag\graphrag\index\cache\__init__.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Indexing Engine cache package root."""

# 导入所需模块和类
from .json_pipeline_cache import JsonPipelineCache
from .load_cache import load_cache
from .memory_pipeline_cache import InMemoryCache
from .noop_pipeline_cache import NoopPipelineCache
from .pipeline_cache import PipelineCache

# 定义公开接口列表，包含所有可以从此包导入的符号名称
__all__ = [
    "InMemoryCache",          # 将 InMemoryCache 类添加到公开接口列表中
    "JsonPipelineCache",      # 将 JsonPipelineCache 类添加到公开接口列表中
    "NoopPipelineCache",      # 将 NoopPipelineCache 类添加到公开接口列表中
    "PipelineCache",          # 将 PipelineCache 类添加到公开接口列表中
    "load_cache",             # 将 load_cache 函数添加到公开接口列表中
]
```