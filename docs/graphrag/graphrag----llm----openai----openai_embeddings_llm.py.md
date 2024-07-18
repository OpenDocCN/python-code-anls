# `.\graphrag\graphrag\llm\openai\openai_embeddings_llm.py`

```py
# 版权所有 (c) 2024 Microsoft Corporation.
# 根据 MIT 许可证授权

"""EmbeddingsLLM 类。"""

# 导入必要的模块和类型
from typing_extensions import Unpack

# 导入基础语言模型类和类型定义
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)

# 导入 OpenAI 配置和类型定义
from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes

# OpenAIEmbeddingsLLM 类，继承自 BaseLLM，用于生成文本嵌入
class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """一个文本嵌入生成器 LLM。"""

    _client: OpenAIClientTypes  # OpenAI 客户端类型注解
    _configuration: OpenAIConfiguration  # OpenAI 配置类型注解

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client  # 初始化 OpenAI 客户端
        self.configuration = configuration  # 初始化 OpenAI 配置

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        # 准备调用语言模型的参数，包括模型和可能的其他参数
        args = {
            "model": self.configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        # 调用 OpenAI 客户端的 embeddings.create 方法生成嵌入
        embedding = await self.client.embeddings.create(
            input=input,
            **args,
        )
        # 返回生成的嵌入数据列表
        return [d.embedding for d in embedding.data]
```