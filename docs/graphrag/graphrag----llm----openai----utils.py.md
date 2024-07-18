# `.\graphrag\graphrag\llm\openai\utils.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utility functions for the OpenAI API."""

# 导入所需模块和库
import json  # 导入处理 JSON 的模块
import logging  # 导入日志记录模块
from collections.abc import Callable  # 导入抽象基类中的 Callable 类型
from typing import Any  # 导入通用的类型提示

import tiktoken  # 导入自定义的 tiktoken 模块
from openai import (  # 从 openai 模块中导入指定异常类
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)

from .openai_configuration import OpenAIConfiguration  # 导入自定义的 OpenAIConfiguration 类

DEFAULT_ENCODING = "cl100k_base"  # 默认编码类型为 "cl100k_base"

_encoders: dict[str, tiktoken.Encoding] = {}  # 创建空字典 _encoders 用于存储编码器实例

RETRYABLE_ERRORS: list[type[Exception]] = [  # 定义可重试的异常类型列表
    RateLimitError,
    APIConnectionError,
    InternalServerError,
]
RATE_LIMIT_ERRORS: list[type[Exception]] = [RateLimitError]  # 定义仅包含速率限制错误的异常类型列表

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


def get_token_counter(config: OpenAIConfiguration) -> Callable[[str], int]:
    """Get a function that counts the number of tokens in a string."""
    model = config.encoding_model or "cl100k_base"  # 获取配置中的编码模型，默认为 "cl100k_base"
    enc = _encoders.get(model)  # 从 _encoders 字典中获取指定模型的编码器实例
    if enc is None:
        enc = tiktoken.get_encoding(model)  # 如果未找到，则使用 tiktoken 模块获取编码器实例
        _encoders[model] = enc  # 将新实例存入 _encoders 字典中

    return lambda s: len(enc.encode(s))  # 返回一个匿名函数，用于计算字符串 s 中的标记数


def perform_variable_replacements(
    input: str, history: list[dict], variables: dict | None
) -> str:
    """Perform variable replacements on the input string and in a chat log."""
    result = input  # 将输入字符串赋值给结果变量

    def replace_all(input: str) -> str:
        """Replace all variables in the input string."""
        result = input
        if variables:
            for entry in variables:
                result = result.replace(f"{{{entry}}}", variables[entry])  # 使用变量字典替换输入中的所有变量
        return result

    result = replace_all(result)  # 执行一次变量替换
    for i in range(len(history)):
        entry = history[i]
        if entry.get("role") == "system":
            history[i]["content"] = replace_all(entry.get("content") or "")  # 替换聊天历史记录中系统角色的内容

    return result  # 返回替换后的结果字符串


def get_completion_cache_args(configuration: OpenAIConfiguration) -> dict:
    """Get the cache arguments for a completion LLM."""
    return {
        "model": configuration.model,
        "temperature": configuration.temperature,
        "frequency_penalty": configuration.frequency_penalty,
        "presence_penalty": configuration.presence_penalty,
        "top_p": configuration.top_p,
        "max_tokens": configuration.max_tokens,
        "n": configuration.n,
    }  # 返回用于完成语言模型的缓存参数字典


def get_completion_llm_args(
    parameters: dict | None, configuration: OpenAIConfiguration
) -> dict:
    """Get the arguments for a completion LLM."""
    return {
        **get_completion_cache_args(configuration),  # 使用配置对象获取缓存参数
        **(parameters or {}),  # 使用参数字典（如果存在）更新缓存参数
    }  # 返回用于完成语言模型的参数字典


def try_parse_json_object(input: str) -> dict:
    """Generate JSON-string output using best-attempt prompting & parsing techniques."""
    try:
        result = json.loads(input)  # 尝试解析输入字符串为 JSON 对象
    except json.JSONDecodeError:
        log.exception("error loading json, json=%s", input)  # 记录 JSON 解析错误日志
        raise  # 抛出异常
    else:
        if not isinstance(result, dict):
            raise TypeError  # 如果解析结果不是字典类型，则抛出类型错误
        return result  # 返回解析后的 JSON 字典对象


def get_sleep_time_from_error(e: Any) -> float:
    """Extract the sleep time value from a RateLimitError. This is usually only available in Azure."""
    sleep_time = 0.0  # 初始化睡眠时间为 0.0
    # 检查异常对象 e 是否是 RateLimitError 类型，并且异常消息包含 _please_retry_after 字符串
    if isinstance(e, RateLimitError) and _please_retry_after in str(e):
        # 从异常消息中提取出需要休眠的时间，可能是 "second" 或 "seconds"
        sleep_time = int(str(e).split(_please_retry_after)[1].split(" second")[0])

    # 返回计算得到的休眠时间
    return sleep_time
# 定义一个全局变量 "_please_retry_after"，用于存储字符串 "Please retry after "
_please_retry_after = "Please retry after "
```