# `.\graphrag\graphrag\query\structured_search\base.py`

```py
# 从 abc 模块导入 ABC 抽象基类和 abstractmethod 抽象方法装饰器
from abc import ABC, abstractmethod
# 从 dataclasses 模块导入 dataclass 装饰器，用于创建数据类
from dataclasses import dataclass
# 从 typing 模块导入 Any 类型
from typing import Any

# 导入 pandas 库，并使用 pd 别名
import pandas as pd
# 导入 tiktoken 模块
import tiktoken

# 从 graphrag.query.context_builder.builders 模块导入 GlobalContextBuilder 和 LocalContextBuilder 类
from graphrag.query.context_builder.builders import (
    GlobalContextBuilder,
    LocalContextBuilder,
)
# 从 graphrag.query.context_builder.conversation_history 模块导入 ConversationHistory 类
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
# 从 graphrag.query.llm.base 模块导入 BaseLLM 类
from graphrag.query.llm.base import BaseLLM


# 使用 dataclass 装饰器定义 SearchResult 数据类
@dataclass
class SearchResult:
    """A Structured Search Result."""

    # 响应数据，可以是字符串、字典或字典列表
    response: str | dict[str, Any] | list[dict[str, Any]]
    # 上下文数据，可以是字符串列表、DataFrame 列表或字典映射到 DataFrame
    context_data: str | list[pd.DataFrame] | dict[str, pd.DataFrame]
    # 上下文文本，可以是字符串列表、字符串映射
    context_text: str | list[str] | dict[str, str]
    # 搜索完成时间（秒）
    completion_time: float
    # 调用的 LLM 模型次数
    llm_calls: int
    # 使用的提示标记数量
    prompt_tokens: int


# 定义 BaseSearch 抽象基类
class BaseSearch(ABC):
    """The Base Search implementation."""

    # 构造函数，初始化 BaseSearch 类的实例
    def __init__(
        self,
        llm: BaseLLM,
        context_builder: GlobalContextBuilder | LocalContextBuilder,
        token_encoder: tiktoken.Encoding | None = None,
        llm_params: dict[str, Any] | None = None,
        context_builder_params: dict[str, Any] | None = None,
    ):
        # LLM 模型实例
        self.llm = llm
        # 上下文构建器实例，可以是全局或本地上下文构建器
        self.context_builder = context_builder
        # 标记编码器实例，可选参数
        self.token_encoder = token_encoder
        # LLM 模型参数，如果为 None 则设置为空字典
        self.llm_params = llm_params or {}
        # 上下文构建器参数，如果为 None 则设置为空字典
        self.context_builder_params = context_builder_params or {}

    # 抽象方法，用于执行同步搜索
    @abstractmethod
    def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """Search for the given query."""

    # 抽象方法，用于执行异步搜索
    @abstractmethod
    async def asearch(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """Search for the given query asynchronously."""
```