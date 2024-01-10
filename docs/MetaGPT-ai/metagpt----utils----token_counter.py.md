# `MetaGPT\metagpt\utils\token_counter.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/18 00:40
@Author  : alexanderwu
@File    : token_counter.py
ref1: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
ref2: https://github.com/Significant-Gravitas/Auto-GPT/blob/master/autogpt/llm/token_counter.py
ref3: https://github.com/hwchase17/langchain/blob/master/langchain/chat_models/openai.py
ref4: https://ai.google.dev/models/gemini
"""
import tiktoken  # 导入tiktoken模块

TOKEN_COSTS = {  # 定义TOKEN_COSTS字典，存储不同模型的token成本
    # 不同模型及其对应的token成本
}

TOKEN_MAX = {  # 定义TOKEN_MAX字典，存储不同模型的最大token数
    # 不同模型及其对应的最大token数
}

def count_message_tokens(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)  # 获取指定模型的编码方式
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")  # 如果模型未找到，则使用默认编码方式
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {  # 根据模型类型设置不同的tokens_per_message和tokens_per_name
        # 根据模型类型设置tokens_per_message和tokens_per_name
    }:
        # 计算消息中的token数
    return num_tokens

def count_string_tokens(string: str, model_name: str) -> int:
    """
    Returns the number of tokens in a text string.

    Args:
        string (str): The text string.
        model_name (str): The name of the encoding to use. (e.g., "gpt-3.5-turbo")

    Returns:
        int: The number of tokens in the text string.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)  # 获取指定模型的编码方式
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")  # 如果模型未找到，则使用默认编码方式
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))  # 返回文本字符串中的token数

def get_max_completion_tokens(messages: list[dict], model: str, default: int) -> int:
    """Calculate the maximum number of completion tokens for a given model and list of messages.

    Args:
        messages: A list of messages.
        model: The model name.

    Returns:
        The maximum number of completion tokens.
    """
    if model not in TOKEN_MAX:  # 如果模型不在TOKEN_MAX中，则返回默认值
        return default
    return TOKEN_MAX[model] - count_message_tokens(messages) - 1  # 计算给定模型和消息列表的最大完成token数

```