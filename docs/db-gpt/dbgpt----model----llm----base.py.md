# `.\DB-GPT-src\dbgpt\model\llm\base.py`

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入必要的模块
from dataclasses import dataclass
from typing import TypedDict

# 定义一个 TypedDict 类型的子类 Message，用于表示 Vicuna 消息对象
class Message(TypedDict):
    """Vicuna Message object containing a role and the message content"""

    role: str      # 消息角色的字符串字段
    content: str   # 消息内容的字符串字段

# 使用 dataclass 装饰器定义一个数据类 ModelInfo，用于表示模型信息
@dataclass
class ModelInfo:
    """Struct for model information.

    Would be lovely to eventually get this directly from APIs
    """

    name: str           # 模型名称的字符串字段
    max_tokens: int     # 最大标记数的整数字段

# 使用 dataclass 装饰器定义一个数据类 LLMResponse，表示来自LLM模型的标准响应结构
@dataclass
class LLMResponse:
    """Standard response struct for a response from a LLM model."""

    model_info = ModelInfo   # LLM 响应的模型信息字段

# 使用 dataclass 装饰器定义一个数据类 ChatModelResponse，表示来自聊天模型的标准响应结构
@dataclass
class ChatModelResponse(LLMResponse):
    """Standard response struct for a response from an LLM model."""

    content: str = None   # 聊天模型响应的内容字段，默认为 None
```