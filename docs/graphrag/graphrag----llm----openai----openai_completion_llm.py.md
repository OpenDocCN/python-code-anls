# `.\graphrag\graphrag\llm\openai\openai_completion_llm.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A text-completion based LLM."""

# 引入日志模块
import logging

# 导入类型扩展模块
from typing_extensions import Unpack

# 导入基础的语言模型类
from graphrag.llm.base import BaseLLM
# 导入相关类型定义
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
)

# 导入 OpenAI 的配置类
from .openai_configuration import OpenAIConfiguration
# 导入 OpenAI 的客户端类型定义
from .types import OpenAIClientTypes
# 导入工具函数，用于获取语言模型的完成参数
from .utils import get_completion_llm_args

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


class OpenAICompletionLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A text-completion based LLM."""

    # 定义私有变量 _client 和 _configuration，分别表示 OpenAI 客户端和配置对象
    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        # 初始化方法，接受 OpenAI 客户端和配置对象作为参数
        self.client = client  # 设置客户端属性
        self.configuration = configuration  # 设置配置属性

    async def _execute_llm(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput | None:
        # 异步方法，执行语言模型完成操作
        # 调用工具函数获取完成操作所需的参数
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self.configuration
        )
        # 使用 OpenAI 客户端创建完成操作
        completion = self.client.completions.create(prompt=input, **args)
        # 返回完成操作结果中的第一个选择文本
        return completion.choices[0].text
```