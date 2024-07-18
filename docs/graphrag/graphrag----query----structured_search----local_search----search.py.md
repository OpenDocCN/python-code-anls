# `.\graphrag\graphrag\query\structured_search\local_search\search.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LocalSearch implementation."""

# 导入必要的库和模块
import logging   # 导入日志模块
import time      # 导入时间模块
from typing import Any   # 导入类型提示的Any类型

import tiktoken   # 导入tiktoken模块

# 导入本地上下文构建器和对话历史等模块
from graphrag.query.context_builder.builders import LocalContextBuilder
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
# 导入基础LLM类和回调函数接口
from graphrag.query.llm.base import BaseLLM, BaseLLMCallback
# 导入文本处理工具函数
from graphrag.query.llm.text_utils import num_tokens
# 导入结构化搜索基础类和搜索结果类
from graphrag.query.structured_search.base import BaseSearch, SearchResult
# 导入本地搜索系统提示
from graphrag.query.structured_search.local_search.system_prompt import (
    LOCAL_SEARCH_SYSTEM_PROMPT,
)

# 默认LLM参数设置
DEFAULT_LLM_PARAMS = {
    "max_tokens": 1500,   # 最大生成token数
    "temperature": 0.0,   # 生成文本的温度
}

# 设置日志记录器
log = logging.getLogger(__name__)


class LocalSearch(BaseSearch):
    """Search orchestration for local search mode."""

    def __init__(
        self,
        llm: BaseLLM,   # 基础LLM模型
        context_builder: LocalContextBuilder,   # 本地上下文构建器
        token_encoder: tiktoken.Encoding | None = None,   # token编码器，可选
        system_prompt: str = LOCAL_SEARCH_SYSTEM_PROMPT,   # 系统提示语句，默认为本地搜索提示
        response_type: str = "multiple paragraphs",   # 响应类型，默认为多段落
        callbacks: list[BaseLLMCallback] | None = None,   # 回调函数列表，可选
        llm_params: dict[str, Any] = DEFAULT_LLM_PARAMS,   # LLM模型参数，默认使用默认LLM参数
        context_builder_params: dict | None = None,   # 上下文构建器参数，可选
    ):
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=context_builder_params or {},
        )
        self.system_prompt = system_prompt   # 设置系统提示语句
        self.callbacks = callbacks   # 设置回调函数列表
        self.response_type = response_type   # 设置响应类型

    async def asearch(
        self,
        query: str,   # 查询字符串
        conversation_history: ConversationHistory | None = None,   # 对话历史记录，可选
        **kwargs,   # 其他关键字参数
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user query."""
        # 记录函数开始时间
        start_time = time.time()
        # 初始化搜索提示字符串
        search_prompt = ""

        # 使用上下文构建器构建上下文文本和记录
        context_text, context_records = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )
        # 记录生成答案的日志信息
        log.info("GENERATE ANSWER: %s. QUERY: %s", start_time, query)
        try:
            # 格式化系统提示字符串，将上下文数据和响应类型插入
            search_prompt = self.system_prompt.format(
                context_data=context_text, response_type=self.response_type
            )
            # 构建搜索消息列表
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]

            # 调用语言模型生成答案
            response = await self.llm.agenerate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            # 返回搜索结果对象
            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

        except Exception:
            # 记录异常信息
            log.exception("Exception in _asearch")
            # 返回空的搜索结果对象
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

    def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """定义函数签名，指定返回类型为SearchResult。"""
        start_time = time.time()  # 记录当前时间，用于计算函数执行时间
        search_prompt = ""  # 初始化搜索提示字符串

        # 使用self.context_builder构建上下文文本和记录
        context_text, context_records = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )

        log.info("GENERATE ANSWER: %d. QUERY: %s", start_time, query)  # 记录日志，输出生成答案的相关信息

        try:
            # 使用self.system_prompt格式化上下文文本和响应类型，生成搜索提示
            search_prompt = self.system_prompt.format(
                context_data=context_text, response_type=self.response_type
            )

            # 构建消息列表，包含系统生成的搜索提示和用户的查询内容
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]

            # 调用self.llm生成响应消息
            response = self.llm.generate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            # 返回SearchResult对象，包含生成的响应、上下文记录、上下文文本、函数执行时间、LLM调用次数和搜索提示的token数
            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

        except Exception:
            log.exception("Exception in _map_response_single_batch")  # 记录异常情况
            # 返回空响应的SearchResult对象，但包含上下文记录、上下文文本、函数执行时间、LLM调用次数和搜索提示的token数
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )
```