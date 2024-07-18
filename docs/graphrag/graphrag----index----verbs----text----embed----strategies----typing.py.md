# `.\graphrag\graphrag\index\verbs\text\embed\strategies\typing.py`

```py
#`
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 模块注释，描述包含 'TextEmbeddingResult' 模型的模块
"""A module containing 'TextEmbeddingResult' model."""

# 从 collections 模块导入 Awaitable 和 Callable，这些是用于类型提示的类型
from collections.abc import Awaitable, Callable

# 从 dataclasses 模块导入 dataclass 装饰器，用于定义数据类
from dataclasses import dataclass

# 导入 VerbCallbacks 类或类型，用于数据转换的回调
from datashaper import VerbCallbacks

# 从 graphrag.index.cache 模块导入 PipelineCache 类
from graphrag.index.cache import PipelineCache

# 定义一个数据类 TextEmbeddingResult，表示文本嵌入结果
@dataclass
class TextEmbeddingResult:
    """Text embedding result class definition."""

    # 字段 embeddings，表示嵌入的列表，列表中元素是浮点数的列表，或为 None，或为 None
    embeddings: list[list[float] | None] | None

# 定义一个类型别名 TextEmbeddingStrategy，表示文本嵌入策略的函数类型
TextEmbeddingStrategy = Callable[
    [
        list[str],          # 输入参数，字符串列表
        VerbCallbacks,       # 输入参数，VerbCallbacks 类型
        PipelineCache,       # 输入参数，PipelineCache 类型
        dict,                # 输入参数，字典
    ],
    Awaitable[TextEmbeddingResult],  # 返回值，异步 Awaitable 对象，返回 TextEmbeddingResult 类型的实例
]
```