# `.\graphrag\graphrag\llm\openai\__init__.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入 OpenAI LLM 实现所需的模块和函数

# 导入创建 OpenAI 客户端的函数
from .create_openai_client import create_openai_client

# 导入创建不同类型 OpenAI LLM 的工厂函数
from .factories import (
    create_openai_chat_llm,
    create_openai_completion_llm,
    create_openai_embedding_llm,
)

# 导入 OpenAIChatLLM 类，用于聊天式语言模型
from .openai_chat_llm import OpenAIChatLLM

# 导入 OpenAICompletionLLM 类，用于完成式语言模型
from .openai_completion_llm import OpenAICompletionLLM

# 导入 OpenAIConfiguration 类，用于配置 OpenAI LLM 的参数
from .openai_configuration import OpenAIConfiguration

# 导入 OpenAIEmbeddingsLLM 类，用于嵌入式语言模型
from .openai_embeddings_llm import OpenAIEmbeddingsLLM

# 导入 OpenAIClientTypes 类型，定义 OpenAI 客户端的类型
from .types import OpenAIClientTypes

# 指定可以通过 from <module> import * 导入的所有公共接口
__all__ = [
    "OpenAIChatLLM",
    "OpenAIClientTypes",
    "OpenAICompletionLLM",
    "OpenAIConfiguration",
    "OpenAIEmbeddingsLLM",
    "create_openai_chat_llm",
    "create_openai_client",
    "create_openai_completion_llm",
    "create_openai_embedding_llm",
]
```