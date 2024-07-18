# `.\graphrag\graphrag\index\verbs\entities\extraction\strategies\typing.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'Document' and 'EntityExtractionResult' models."""

# 导入必要的模块和类
from collections.abc import Awaitable, Callable  # 导入Awaitable和Callable类
from dataclasses import dataclass  # 导入dataclass装饰器
from typing import Any  # 导入Any类型

# 导入外部模块
from datashaper import VerbCallbacks  # 从datashaper模块导入VerbCallbacks类

# 导入自定义模块
from graphrag.index.cache import PipelineCache  # 从graphrag.index.cache模块导入PipelineCache类

# 定义类型别名
ExtractedEntity = dict[str, Any]  # ExtractedEntity类型为字典，键为str，值为Any类型
StrategyConfig = dict[str, Any]  # StrategyConfig类型为字典，键为str，值为Any类型
EntityTypes = list[str]  # EntityTypes类型为字符串列表，每个元素为str类型


@dataclass
class Document:
    """Document class definition."""
    
    text: str  # 文档内容文本，类型为str
    id: str  # 文档的唯一标识符，类型为str


@dataclass
class EntityExtractionResult:
    """Entity extraction result class definition."""
    
    entities: list[ExtractedEntity]  # 提取的实体列表，每个实体为ExtractedEntity类型的字典
    graphml_graph: str | None  # 表示GraphML图形的字符串，或者为None


EntityExtractStrategy = Callable[
    [
        list[Document],  # 接受的文档列表，每个元素为Document类的实例
        EntityTypes,      # 用于实体提取的实体类型列表，每个元素为str类型
        VerbCallbacks,    # 用于动词回调的VerbCallbacks实例
        PipelineCache,    # 用于管道缓存的PipelineCache实例
        StrategyConfig,   # 策略配置选项，类型为StrategyConfig
    ],
    Awaitable[EntityExtractionResult],  # 返回一个异步等待的EntityExtractionResult实例
]
```