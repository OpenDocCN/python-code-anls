# `.\graphrag\graphrag\index\verbs\graph\report\strategies\typing.py`

```py
# 版权声明和许可证信息，指出此代码的版权和许可证（MIT License）
"""A module containing 'Finding' and 'CommunityReport' models."""
# 包含 'Finding' 和 'CommunityReport' 模型的模块说明

from collections.abc import Awaitable, Callable
# 导入 Awaitable 和 Callable 类型定义，用于异步和可调用对象的支持
from typing import Any
# 导入 Any 类型，表示任意类型

from datashaper import VerbCallbacks
# 从 datashaper 模块导入 VerbCallbacks 类
from typing_extensions import TypedDict
# 导入 TypedDict 类型，用于定义类型安全的字典结构

from graphrag.index.cache import PipelineCache
# 从 graphrag.index.cache 模块导入 PipelineCache 类

ExtractedEntity = dict[str, Any]
# 定义 ExtractedEntity 类型别名，表示包含任意值的字典结构
StrategyConfig = dict[str, Any]
# 定义 StrategyConfig 类型别名，表示包含任意值的字典结构
RowContext = dict[str, Any]
# 定义 RowContext 类型别名，表示包含任意值的字典结构
EntityTypes = list[str]
# 定义 EntityTypes 类型别名，表示包含字符串元素的列表结构
Claim = dict[str, Any]
# 定义 Claim 类型别名，表示包含任意值的字典结构

class Finding(TypedDict):
    """Finding class definition."""
    # Finding 类型字典的定义
    summary: str
    # 字符串类型的摘要字段
    explanation: str
    # 字符串类型的解释字段

class CommunityReport(TypedDict):
    """Community report class definition."""
    # CommunityReport 类型字典的定义
    community: str | int
    # 字符串或整数类型的社区字段
    title: str
    # 字符串类型的标题字段
    summary: str
    # 字符串类型的摘要字段
    full_content: str
    # 字符串类型的完整内容字段
    full_content_json: str
    # 字符串类型的完整内容的 JSON 字符串字段
    rank: float
    # 浮点数类型的排名字段
    level: int
    # 整数类型的级别字段
    rank_explanation: str
    # 字符串类型的排名解释字段
    findings: list[Finding]
    # 包含 Finding 对象的列表结构的 findings 字段

CommunityReportsStrategy = Callable[
    [
        str | int,
        str,
        int,
        VerbCallbacks,
        PipelineCache,
        StrategyConfig,
    ],
    Awaitable[CommunityReport | None],
]
# CommunityReportsStrategy 类型别名，表示可调用对象，接受一组参数并异步返回 CommunityReport 或 None
```