# `.\graphrag\graphrag\llm\openai\types.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
一个基于 OpenAI 的 LLM（Language Language Model）的基类。
"""

# 从 openai 模块中导入 AsyncAzureOpenAI 和 AsyncOpenAI 类
from openai import (
    AsyncAzureOpenAI,
    AsyncOpenAI,
)

# 定义 OpenAIClientTypes 变量，其值为 AsyncOpenAI 和 AsyncAzureOpenAI 类型的并集
OpenAIClientTypes = AsyncOpenAI | AsyncAzureOpenAI
```