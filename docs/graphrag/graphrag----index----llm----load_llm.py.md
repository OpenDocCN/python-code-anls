# `.\graphrag\graphrag\index\llm\load_llm.py`

```py
# 版权声明和许可证声明
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 加载 llm 工具模块
"""Load llm utilities."""

# 引入未来版本支持的语法特性
from __future__ import annotations

# 引入 asyncio 异步编程库
import asyncio
# 引入日志记录模块
import logging
# 引入类型检查相关功能
from typing import TYPE_CHECKING, Any

# 引入 llm 类型枚举
from graphrag.config.enums import LLMType
# 引入 llm 模块
from graphrag.llm import (
    CompletionLLM,
    EmbeddingLLM,
    LLMCache,
    LLMLimiter,
    MockCompletionLLM,
    OpenAIConfiguration,
    create_openai_chat_llm,
    create_openai_client,
    create_openai_completion_llm,
    create_openai_embedding_llm,
    create_tpm_rpm_limiters,
)

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 引入数据形状模块中的动词回调接口
    from datashaper import VerbCallbacks
    # 引入图形索引缓存模块
    from graphrag.index.cache import PipelineCache
    # 引入图形索引类型模块中的错误处理器函数
    from graphrag.index.typing import ErrorHandlerFn

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 用于存储信号量的字典
_semaphores: dict[str, asyncio.Semaphore] = {}
# 用于存储速率限制器的字典
_rate_limiters: dict[str, LLMLimiter] = {}


def load_llm(
    name: str,
    llm_type: LLMType,
    callbacks: VerbCallbacks,
    cache: PipelineCache | None,
    llm_config: dict[str, Any] | None = None,
    chat_only=False,
) -> CompletionLLM:
    """Load the LLM for the entity extraction chain."""
    # 创建错误处理函数
    on_error = _create_error_handler(callbacks)

    # 如果 llm_type 存在于 loaders 中
    if llm_type in loaders:
        # 如果仅支持聊天但当前类型不支持聊天
        if chat_only and not loaders[llm_type]["chat"]:
            msg = f"LLM type {llm_type} does not support chat"
            raise ValueError(msg)
        # 如果有缓存，为缓存创建子级
        if cache is not None:
            cache = cache.child(name)

        # 获取对应 llm_type 的加载器
        loader = loaders[llm_type]
        # 调用加载函数并返回结果
        return loader["load"](on_error, cache, llm_config or {})

    # 抛出未知 llm_type 错误
    msg = f"Unknown LLM type {llm_type}"
    raise ValueError(msg)


def load_llm_embeddings(
    name: str,
    llm_type: LLMType,
    callbacks: VerbCallbacks,
    cache: PipelineCache | None,
    llm_config: dict[str, Any] | None = None,
    chat_only=False,
) -> EmbeddingLLM:
    """Load the LLM for the entity extraction chain."""
    # 创建错误处理函数
    on_error = _create_error_handler(callbacks)
    
    # 如果 llm_type 存在于 loaders 中
    if llm_type in loaders:
        # 如果仅支持聊天但当前类型不支持聊天
        if chat_only and not loaders[llm_type]["chat"]:
            msg = f"LLM type {llm_type} does not support chat"
            raise ValueError(msg)
        # 如果有缓存，为缓存创建子级
        if cache is not None:
            cache = cache.child(name)

        # 返回对应 llm_type 的加载函数的调用结果
        return loaders[llm_type]["load"](on_error, cache, llm_config or {})

    # 抛出未知 llm_type 错误
    msg = f"Unknown LLM type {llm_type}"
    raise ValueError(msg)


def _create_error_handler(callbacks: VerbCallbacks) -> ErrorHandlerFn:
    """Create an error handler function."""
    # 定义错误处理函数，调用回调函数进行错误记录
    def on_error(
        error: BaseException | None = None,
        stack: str | None = None,
        details: dict | None = None,
    ) -> None:
        callbacks.error("Error Invoking LLM", error, stack, details)

    return on_error


def _load_openai_completion_llm(
    on_error: ErrorHandlerFn,
    cache: LLMCache,
    config: dict[str, Any],
    azure=False,
):
    # 留待实现，用于加载 OpenAI 完成 LLM
    pass
    # 调用 _create_openai_completion_llm 函数，传入一个 OpenAIConfiguration 对象作为参数，该对象包含以下配置：
    # - 使用 _get_base_config 函数获取的基础配置
    # - "model": 从 config 中获取，若未指定则默认为 "gpt-4-turbo-preview"
    # - "deployment_name": 从 config 中获取
    # - "temperature": 从 config 中获取，若未指定则默认为 0.0
    # - "frequency_penalty": 从 config 中获取，若未指定则默认为 0
    # - "presence_penalty": 从 config 中获取，若未指定则默认为 0
    # - "top_p": 从 config 中获取，若未指定则默认为 1
    # - "max_tokens": 从 config 中获取，若未指定则默认为 4000
    # - "n": 从 config 中获取
    # on_error: 错误处理函数
    # cache: 缓存对象
    # azure: Azure 相关配置
    return _create_openai_completion_llm(
        OpenAIConfiguration({
            **_get_base_config(config),
            "model": config.get("model", "gpt-4-turbo-preview"),
            "deployment_name": config.get("deployment_name"),
            "temperature": config.get("temperature", 0.0),
            "frequency_penalty": config.get("frequency_penalty", 0),
            "presence_penalty": config.get("presence_penalty", 0),
            "top_p": config.get("top_p", 1),
            "max_tokens": config.get("max_tokens", 4000),
            "n": config.get("n"),
        }),
        on_error,
        cache,
        azure,
    )
# 定义一个私有函数来加载 OpenAI 对话型语言模型
def _load_openai_chat_llm(
    on_error: ErrorHandlerFn,  # 错误处理函数
    cache: LLMCache,  # 缓存对象
    config: dict[str, Any],  # 配置参数字典
    azure=False,  # 是否使用 Azure 平台，默认为 False
):
    # 调用 _create_openai_chat_llm 函数创建 OpenAI 对话型语言模型
    return _create_openai_chat_llm(
        OpenAIConfiguration({
            # 使用 _get_base_config 函数获取基础配置，将其作为默认值
            **_get_base_config(config),
            "model": config.get("model", "gpt-4-turbo-preview"),  # 设置模型名称，默认为 "gpt-4-turbo-preview"
            "deployment_name": config.get("deployment_name"),  # 获取部署名称
            "temperature": config.get("temperature", 0.0),  # 设置 temperature 参数，默认为 0.0
            "frequency_penalty": config.get("frequency_penalty", 0),  # 设置 frequency_penalty 参数，默认为 0
            "presence_penalty": config.get("presence_penalty", 0),  # 设置 presence_penalty 参数，默认为 0
            "top_p": config.get("top_p", 1),  # 设置 top_p 参数，默认为 1
            "max_tokens": config.get("max_tokens"),  # 获取 max_tokens 参数
            "n": config.get("n"),  # 获取 n 参数
        }),
        on_error,  # 错误处理函数
        cache,  # 缓存对象
        azure,  # 是否使用 Azure 平台
    )


# 定义一个私有函数来加载 OpenAI 嵌入型语言模型
def _load_openai_embeddings_llm(
    on_error: ErrorHandlerFn,  # 错误处理函数
    cache: LLMCache,  # 缓存对象
    config: dict[str, Any],  # 配置参数字典
    azure=False,  # 是否使用 Azure 平台，默认为 False
):
    # TODO: Inject Cache 注入缓存对象
    return _create_openai_embeddings_llm(
        OpenAIConfiguration({
            **_get_base_config(config),  # 使用 _get_base_config 函数获取基础配置
            "model": config.get(
                "embeddings_model", config.get("model", "text-embedding-3-small")
            ),  # 设置模型名称，默认为 config 中的 "embeddings_model" 或者 "model"，默认为 "text-embedding-3-small"
            "deployment_name": config.get("deployment_name"),  # 获取部署名称
        }),
        on_error,  # 错误处理函数
        cache,  # 缓存对象
        azure,  # 是否使用 Azure 平台
    )


# 定义一个私有函数来加载 Azure 平台的 OpenAI 完成型语言模型
def _load_azure_openai_completion_llm(
    on_error: ErrorHandlerFn,  # 错误处理函数
    cache: LLMCache,  # 缓存对象
    config: dict[str, Any],  # 配置参数字典
):
    # 调用 _load_openai_completion_llm 函数加载 OpenAI 完成型语言模型，并设置 azure 参数为 True
    return _load_openai_completion_llm(on_error, cache, config, True)


# 定义一个私有函数来加载 Azure 平台的 OpenAI 对话型语言模型
def _load_azure_openai_chat_llm(
    on_error: ErrorHandlerFn,  # 错误处理函数
    cache: LLMCache,  # 缓存对象
    config: dict[str, Any],  # 配置参数字典
):
    # 调用 _load_openai_chat_llm 函数加载 OpenAI 对话型语言模型，并设置 azure 参数为 True
    return _load_openai_chat_llm(on_error, cache, config, True)


# 定义一个私有函数来加载 Azure 平台的 OpenAI 嵌入型语言模型
def _load_azure_openai_embeddings_llm(
    on_error: ErrorHandlerFn,  # 错误处理函数
    cache: LLMCache,  # 缓存对象
    config: dict[str, Any],  # 配置参数字典
):
    # 调用 _load_openai_embeddings_llm 函数加载 OpenAI 嵌入型语言模型，并设置 azure 参数为 True
    return _load_openai_embeddings_llm(on_error, cache, config, True)


# 定义一个私有函数来获取基础配置信息
def _get_base_config(config: dict[str, Any]) -> dict[str, Any]:
    api_key = config.get("api_key")  # 获取配置中的 API 密钥

    return {
        # 将所有参数化的值传入
        **config,
        # 设置默认值
        "api_key": api_key,  # 设置 API 密钥
        "api_base": config.get("api_base"),  # 获取 API 基础路径
        "api_version": config.get("api_version"),  # 获取 API 版本
        "organization": config.get("organization"),  # 获取组织名称
        "proxy": config.get("proxy"),  # 获取代理设置
        "max_retries": config.get("max_retries", 10),  # 设置最大重试次数，默认为 10
        "request_timeout": config.get("request_timeout", 60.0),  # 设置请求超时时间，默认为 60.0 秒
        "model_supports_json": config.get("model_supports_json"),  # 模型是否支持 JSON
        "concurrent_requests": config.get("concurrent_requests", 4),  # 设置并发请求数量，默认为 4
        "encoding_model": config.get("encoding_model", "cl100k_base"),  # 设置编码模型，默认为 "cl100k_base"
        "cognitive_services_endpoint": config.get("cognitive_services_endpoint"),  # 获取认知服务端点
    }


# 定义一个函数来加载静态响应数据，返回 MockCompletionLLM 实例
def _load_static_response(
    _on_error: ErrorHandlerFn,  # 错误处理函数（未使用）
    _cache: PipelineCache,  # 缓存对象（未使用）
    config: dict[str, Any],  # 配置参数字典
) -> CompletionLLM:
    return MockCompletionLLM(config.get("responses", []))  # 使用配置中的 responses 字段初始化 MockCompletionLLM 实例


# loaders 字典，用于存储不同类型语言模型加载函数的映射
loaders = {
    LLMType.OpenAI: {
        "load": _load_openai_completion_llm,  # 加载 OpenAI 完成型语言模型的函数
        "chat": False,  # 设置 chat 字段为 False，表示不支持对话型功能
    },
    LLMType.AzureOpenAI: {
        # 加载 Azure OpenAI 完形填空语言模型
        "load": _load_azure_openai_completion_llm,
        # 不支持实时聊天
        "chat": False,
    },
    LLMType.OpenAIChat: {
        # 加载 OpenAI 聊天语言模型
        "load": _load_openai_chat_llm,
        # 支持实时聊天
        "chat": True,
    },
    LLMType.AzureOpenAIChat: {
        # 加载 Azure OpenAI 聊天语言模型
        "load": _load_azure_openai_chat_llm,
        # 支持实时聊天
        "chat": True,
    },
    LLMType.OpenAIEmbedding: {
        # 加载 OpenAI 嵌入语言模型
        "load": _load_openai_embeddings_llm,
        # 不支持实时聊天
        "chat": False,
    },
    LLMType.AzureOpenAIEmbedding: {
        # 加载 Azure OpenAI 嵌入语言模型
        "load": _load_azure_openai_embeddings_llm,
        # 不支持实时聊天
        "chat": False,
    },
    LLMType.StaticResponse: {
        # 加载静态响应语言模型
        "load": _load_static_response,
        # 不支持实时聊天
        "chat": False,
    },
}

# 创建一个基于 OpenAI 的聊天模型
def _create_openai_chat_llm(
    configuration: OpenAIConfiguration,
    on_error: ErrorHandlerFn,
    cache: LLMCache,
    azure=False,
) -> CompletionLLM:
    """Create an openAI chat llm."""
    # 创建 OpenAI 客户端
    client = create_openai_client(configuration=configuration, azure=azure)
    # 创建限制器对象
    limiter = _create_limiter(configuration)
    # 创建信号量对象
    semaphore = _create_semaphore(configuration)
    # 调用函数创建并返回基于聊天的 OpenAI 完成模型对象
    return create_openai_chat_llm(
        client, configuration, cache, limiter, semaphore, on_error=on_error
    )


# 创建一个基于 OpenAI 的完成模型
def _create_openai_completion_llm(
    configuration: OpenAIConfiguration,
    on_error: ErrorHandlerFn,
    cache: LLMCache,
    azure=False,
) -> CompletionLLM:
    """Create an openAI completion llm."""
    # 创建 OpenAI 客户端
    client = create_openai_client(configuration=configuration, azure=azure)
    # 创建限制器对象
    limiter = _create_limiter(configuration)
    # 创建信号量对象
    semaphore = _create_semaphore(configuration)
    # 调用函数创建并返回基于完成的 OpenAI 完成模型对象
    return create_openai_completion_llm(
        client, configuration, cache, limiter, semaphore, on_error=on_error
    )


# 创建一个基于 OpenAI 的嵌入模型
def _create_openai_embeddings_llm(
    configuration: OpenAIConfiguration,
    on_error: ErrorHandlerFn,
    cache: LLMCache,
    azure=False,
) -> EmbeddingLLM:
    """Create an openAI embeddings llm."""
    # 创建 OpenAI 客户端
    client = create_openai_client(configuration=configuration, azure=azure)
    # 创建限制器对象
    limiter = _create_limiter(configuration)
    # 创建信号量对象
    semaphore = _create_semaphore(configuration)
    # 调用函数创建并返回基于嵌入的 OpenAI 完成模型对象
    return create_openai_embedding_llm(
        client, configuration, cache, limiter, semaphore, on_error=on_error
    )


# 创建一个限制器对象
def _create_limiter(configuration: OpenAIConfiguration) -> LLMLimiter:
    # 获取模型名称或部署名称，如果都为空则使用默认值 "default"
    limit_name = configuration.model or configuration.deployment_name or "default"
    
    # 如果限制器对象尚未创建，则根据配置信息创建并保存在全局字典中
    if limit_name not in _rate_limiters:
        tpm = configuration.tokens_per_minute
        rpm = configuration.requests_per_minute
        log.info("create TPM/RPM limiter for %s: TPM=%s, RPM=%s", limit_name, tpm, rpm)
        _rate_limiters[limit_name] = create_tpm_rpm_limiters(configuration)
    
    # 返回相应的限制器对象
    return _rate_limiters[limit_name]


# 创建一个信号量对象或返回 None
def _create_semaphore(configuration: OpenAIConfiguration) -> asyncio.Semaphore | None:
    # 获取模型名称或部署名称，如果都为空则使用默认值 "default"
    limit_name = configuration.model or configuration.deployment_name or "default"
    # 获取并设置并发请求数
    concurrency = configuration.concurrent_requests

    # 如果并发数为零，则绕过信号量对象的创建并记录日志
    if not concurrency:
        log.info("no concurrency limiter for %s", limit_name)
        return None

    # 如果信号量对象尚未创建，则根据配置信息创建并保存在全局字典中
    if limit_name not in _semaphores:
        log.info("create concurrency limiter for %s: %s", limit_name, concurrency)
        _semaphores[limit_name] = asyncio.Semaphore(concurrency)

    # 返回相应的信号量对象
    return _semaphores[limit_name]
```