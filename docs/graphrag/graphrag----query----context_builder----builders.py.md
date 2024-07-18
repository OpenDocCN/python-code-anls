# `.\graphrag\graphrag\query\context_builder\builders.py`

```py
# 从 abc 模块导入 ABC 和 abstractmethod 装饰器
from abc import ABC, abstractmethod

# 导入 pandas 库，并使用 pd 别名
import pandas as pd

# 从 graphrag.query.context_builder.conversation_history 模块导入 ConversationHistory 类
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)


# 定义全局搜索上下文构建器的抽象基类 GlobalContextBuilder
class GlobalContextBuilder(ABC):
    """Base class for global-search context builders."""

    @abstractmethod
    # 定义抽象方法 build_context，用于构建全局搜索模式的上下文
    def build_context(
        self, conversation_history: ConversationHistory | None = None, **kwargs
    ) -> tuple[str | list[str], dict[str, pd.DataFrame]]:
        """Build the context for the global search mode."""


# 定义局部搜索上下文构建器的抽象基类 LocalContextBuilder
class LocalContextBuilder(ABC):
    """Base class for local-search context builders."""

    @abstractmethod
    # 定义抽象方法 build_context，用于构建局部搜索模式的上下文
    def build_context(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> tuple[str | list[str], dict[str, pd.DataFrame]]:
        """Build the context for the local search mode."""
```