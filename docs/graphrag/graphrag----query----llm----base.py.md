# `.\graphrag\graphrag\query\llm\base.py`

```py
# 版权声明和许可信息，指明代码的版权归属和使用许可
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块
"""Base classes for LLM and Embedding models."""
from abc import ABC, abstractmethod  # 导入ABC类和abstractmethod装饰器
from typing import Any  # 导入Any类型用于灵活的类型声明


class BaseLLMCallback:
    """Base class for LLM callbacks."""

    def __init__(self):
        self.response = []  # 初始化响应列表

    def on_llm_new_token(self, token: str):
        """Handle when a new token is generated."""
        self.response.append(token)  # 当生成新令牌时处理，将令牌添加到响应列表中


class BaseLLM(ABC):
    """The Base LLM implementation."""

    @abstractmethod
    def generate(
        self,
        messages: str | list[Any],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response."""
        # 生成响应的抽象方法声明，接受消息，支持流式处理，回调函数列表和额外参数

    @abstractmethod
    async def agenerate(
        self,
        messages: str | list[Any],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response asynchronously."""
        # 异步生成响应的抽象方法声明，接受消息，支持流式处理，回调函数列表和额外参数


class BaseTextEmbedding(ABC):
    """The text embedding interface."""

    @abstractmethod
    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a text string."""
        # 将文本嵌入向量的抽象方法声明，接受文本和额外参数，返回浮点数列表

    @abstractmethod
    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a text string asynchronously."""
        # 异步将文本嵌入向量的抽象方法声明，接受文本和额外参数，返回浮点数列表
```