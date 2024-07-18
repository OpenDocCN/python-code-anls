# `.\graphrag\graphrag\llm\openai\factories.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Factory functions for creating OpenAI LLMs."""

import asyncio  # 导入异步IO库

from graphrag.llm.base import CachingLLM, RateLimitingLLM  # 导入缓存和限流相关的基础类
from graphrag.llm.limiting import LLMLimiter  # 导入限流器
from graphrag.llm.types import (  # 导入各种类型定义
    LLM,
    CompletionLLM,
    EmbeddingLLM,
    ErrorHandlerFn,
    LLMCache,
    LLMInvocationFn,
    OnCacheActionFn,
)

from .json_parsing_llm import JsonParsingLLM  # 导入处理JSON解析的LLM
from .openai_chat_llm import OpenAIChatLLM  # 导入OpenAI聊天LLM
from .openai_completion_llm import OpenAICompletionLLM  # 导入OpenAI完成LLM
from .openai_configuration import OpenAIConfiguration  # 导入OpenAI配置
from .openai_embeddings_llm import OpenAIEmbeddingsLLM  # 导入OpenAI嵌入LLM
from .openai_history_tracking_llm import OpenAIHistoryTrackingLLM  # 导入OpenAI历史追踪LLM
from .openai_token_replacing_llm import OpenAITokenReplacingLLM  # 导入OpenAI令牌替换LLM
from .types import OpenAIClientTypes  # 导入OpenAI客户端类型定义
from .utils import (  # 导入各种工具函数
    RATE_LIMIT_ERRORS,
    RETRYABLE_ERRORS,
    get_completion_cache_args,
    get_sleep_time_from_error,
    get_token_counter,
)


def create_openai_chat_llm(
    client: OpenAIClientTypes,
    config: OpenAIConfiguration,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> CompletionLLM:
    """Create an OpenAI chat LLM."""
    operation = "chat"  # 操作类型为聊天
    result = OpenAIChatLLM(client, config)  # 创建OpenAI聊天LLM对象
    result.on_error(on_error)  # 设置错误处理函数
    if limiter is not None or semaphore is not None:
        result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)  # 若有限流器或信号量，则对LLM对象进行限流处理
    if cache is not None:
        result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)  # 若有缓存对象，则对LLM对象进行缓存处理
    result = OpenAIHistoryTrackingLLM(result)  # 对LLM对象进行历史追踪处理
    result = OpenAITokenReplacingLLM(result)  # 对LLM对象进行令牌替换处理
    return JsonParsingLLM(result)  # 返回处理完的LLM对象


def create_openai_completion_llm(
    client: OpenAIClientTypes,
    config: OpenAIConfiguration,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> CompletionLLM:
    """Create an OpenAI completion LLM."""
    operation = "completion"  # 操作类型为完成
    result = OpenAICompletionLLM(client, config)  # 创建OpenAI完成LLM对象
    result.on_error(on_error)  # 设置错误处理函数
    if limiter is not None or semaphore is not None:
        result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)  # 若有限流器或信号量，则对LLM对象进行限流处理
    if cache is not None:
        result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)  # 若有缓存对象，则对LLM对象进行缓存处理
    return OpenAITokenReplacingLLM(result)  # 返回令牌替换后的LLM对象


def create_openai_embedding_llm(
    client: OpenAIClientTypes,
    config: OpenAIConfiguration,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> EmbeddingLLM:
    """Create an OpenAI embedding LLM."""
    # 略
    pass  # 此函数未完待续，因此用pass占位符表示暂无更多代码
    # 定义一个名为 semaphore 的变量，类型为 asyncio.Semaphore 或 None
    semaphore: asyncio.Semaphore | None = None,
    # 定义一个名为 on_invoke 的变量，类型为 LLMInvocationFn 或 None
    on_invoke: LLMInvocationFn | None = None,
    # 定义一个名为 on_error 的变量，类型为 ErrorHandlerFn 或 None
    on_error: ErrorHandlerFn | None = None,
    # 定义一个名为 on_cache_hit 的变量，类型为 OnCacheActionFn 或 None
    on_cache_hit: OnCacheActionFn | None = None,
    # 定义一个名为 on_cache_miss 的变量，类型为 OnCacheActionFn 或 None
    on_cache_miss: OnCacheActionFn | None = None,
# 创建一个 OpenAI embeddings LLM 对象的函数
def create_openai_embeddings_llm(
    client: OpenAIClient,
    config: OpenAIConfiguration,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    cache: LLMCache | None = None,
    on_error: LLMErrorFn | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> EmbeddingLLM:
    """Create an OpenAI embeddings LLM."""
    # 操作类型为 embedding
    operation = "embedding"
    # 创建基础的 OpenAIEmbeddingsLLM 实例
    result = OpenAIEmbeddingsLLM(client, config)
    # 设置错误处理回调函数
    result.on_error(on_error)
    
    # 如果有限流器或信号量参数，对结果进行限流处理
    if limiter is not None or semaphore is not None:
        result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)
    
    # 如果有缓存参数，对结果进行缓存处理
    if cache is not None:
        result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)
    
    return result


# 创建一个进行限流处理的 LLM 对象
def _rate_limited(
    delegate: LLM,
    config: OpenAIConfiguration,
    operation: str,
    limiter: LLMLimiter | None,
    semaphore: asyncio.Semaphore | None,
    on_invoke: LLMInvocationFn | None,
) -> LLM:
    result = RateLimitingLLM(
        delegate,
        config,
        operation,
        RETRYABLE_ERRORS,
        RATE_LIMIT_ERRORS,
        limiter,
        semaphore,
        get_token_counter(config),
        get_sleep_time_from_error,
    )
    # 设置调用时的回调函数
    result.on_invoke(on_invoke)
    return result


# 创建一个进行缓存处理的 LLM 对象
def _cached(
    delegate: LLM,
    config: OpenAIConfiguration,
    operation: str,
    cache: LLMCache,
    on_cache_hit: OnCacheActionFn | None,
    on_cache_miss: OnCacheActionFn | None,
) -> LLM:
    # 获取完成时缓存参数
    cache_args = get_completion_cache_args(config)
    # 创建缓存处理的实例
    result = CachingLLM(delegate, cache_args, operation, cache)
    # 设置缓存命中时的回调函数
    result.on_cache_hit(on_cache_hit)
    # 设置缓存未命中时的回调函数
    result.on_cache_miss(on_cache_miss)
    return result
```