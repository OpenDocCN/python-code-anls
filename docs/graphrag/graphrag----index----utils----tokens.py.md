# `.\graphrag\graphrag\index\utils\tokens.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utilities for working with tokens."""

# 引入日志记录模块
import logging

# 引入自定义的 token 处理模块
import tiktoken

# 默认编码名称
DEFAULT_ENCODING_NAME = "cl100k_base"

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


def num_tokens_from_string(
    string: str, model: str | None = None, encoding_name: str | None = None
) -> int:
    """Return the number of tokens in a text string."""
    # 如果指定了模型名称，则尝试获取该模型的编码方式
    if model is not None:
        try:
            encoding = tiktoken.encoding_for_model(model)
        # 捕获可能的 KeyError 异常
        except KeyError:
            # 如果获取编码失败，则记录警告信息，并使用默认编码名称
            msg = f"Failed to get encoding for {model} when getting num_tokens_from_string. Fall back to default encoding {DEFAULT_ENCODING_NAME}"
            log.warning(msg)
            encoding = tiktoken.get_encoding(DEFAULT_ENCODING_NAME)
    else:
        # 如果没有指定模型名称，则使用指定的编码名称或者默认编码名称获取编码方式
        encoding = tiktoken.get_encoding(encoding_name or DEFAULT_ENCODING_NAME)
    
    # 返回字符串编码后的 token 数量
    return len(encoding.encode(string))


def string_from_tokens(
    tokens: list[int], model: str | None = None, encoding_name: str | None = None
) -> str:
    """Return a text string from a list of tokens."""
    # 如果指定了模型名称，则获取该模型对应的编码方式
    if model is not None:
        encoding = tiktoken.encoding_for_model(model)
    # 如果指定了编码名称，则使用指定的编码名称获取编码方式
    elif encoding_name is not None:
        encoding = tiktoken.get_encoding(encoding_name)
    else:
        # 如果既没有指定模型名称也没有指定编码名称，则抛出 ValueError 异常
        msg = "Either model or encoding_name must be specified."
        raise ValueError(msg)
    
    # 解码 token 列表，返回解码后的字符串
    return encoding.decode(tokens)
```