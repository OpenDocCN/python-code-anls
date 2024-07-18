# `.\graphrag\graphrag\query\question_gen\base.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块和库
"""Base classes for generating questions based on previously asked questions and most recent context data."""

from abc import ABC, abstractmethod  # 导入ABC类和abstractmethod装饰器
from dataclasses import dataclass  # 导入dataclass装饰器
from typing import Any  # 导入Any类型

import tiktoken  # 导入tiktoken模块

from graphrag.query.context_builder.builders import (  # 导入context_builder模块中的GlobalContextBuilder和LocalContextBuilder
    GlobalContextBuilder,
    LocalContextBuilder,
)
from graphrag.query.llm.base import BaseLLM  # 导入BaseLLM类


@dataclass
class QuestionResult:
    """A Structured Question Result."""
    
    response: list[str]  # 响应结果，一个字符串列表
    context_data: str | dict[str, Any]  # 上下文数据，可以是字符串或者字典类型
    completion_time: float  # 完成时间，浮点数类型
    llm_calls: int  # LLM调用次数，整数类型
    prompt_tokens: int  # 提示标记数，整数类型


class BaseQuestionGen(ABC):
    """The Base Question Gen implementation."""

    def __init__(
        self,
        llm: BaseLLM,  # LLM对象，基于BaseLLM类
        context_builder: GlobalContextBuilder | LocalContextBuilder,  # 全局或局部上下文构建器对象，基于GlobalContextBuilder或LocalContextBuilder
        token_encoder: tiktoken.Encoding | None = None,  # Token编码器，基于tiktoken.Encoding或者None
        llm_params: dict[str, Any] | None = None,  # LLM参数字典或者None
        context_builder_params: dict[str, Any] | None = None,  # 上下文构建器参数字典或者None
    ):
        self.llm = llm  # 初始化LLM对象
        self.context_builder = context_builder  # 初始化上下文构建器对象
        self.token_encoder = token_encoder  # 初始化Token编码器对象
        self.llm_params = llm_params or {}  # 初始化LLM参数，如果为None则设为空字典
        self.context_builder_params = context_builder_params or {}  # 初始化上下文构建器参数，如果为None则设为空字典

    @abstractmethod
    def generate(
        self,
        question_history: list[str],  # 问题历史记录，字符串列表
        context_data: str | None,  # 上下文数据，可以是字符串或者None
        question_count: int,  # 问题数量，整数类型
        **kwargs,  # 其他关键字参数
    ) -> QuestionResult:
        """Generate questions."""
        # 抽象方法：生成问题

    @abstractmethod
    async def agenerate(
        self,
        question_history: list[str],  # 问题历史记录，字符串列表
        context_data: str | None,  # 上下文数据，可以是字符串或者None
        question_count: int,  # 问题数量，整数类型
        **kwargs,  # 其他关键字参数
    ) -> QuestionResult:
        """Generate questions asynchronously."""
        # 抽象方法：异步生成问题
```