# `.\graphrag\graphrag\llm\openai\openai_token_replacing_llm.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块和类
"""The Chat-based language model."""
from typing_extensions import Unpack

# 导入类型定义
from graphrag.llm.types import (
    LLM,
    CompletionInput,
    CompletionLLM,
    CompletionOutput,
    LLMInput,
    LLMOutput,
)

# 导入局部的工具函数
from .utils import perform_variable_replacements


# 定义一个继承自LLM类的新类，用于OpenAI的历史追踪语言模型
class OpenAITokenReplacingLLM(LLM[CompletionInput, CompletionOutput]):
    """An OpenAI History-Tracking LLM."""

    # 内部变量，保存委托的CompletionLLM对象
    _delegate: CompletionLLM

    # 构造函数，接受一个CompletionLLM对象作为参数
    def __init__(self, delegate: CompletionLLM):
        self._delegate = delegate

    # 异步调用方法，接受一个CompletionInput对象和其他关键字参数
    async def __call__(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> LLMOutput[CompletionOutput]:
        """Call the LLM with the input and kwargs."""
        # 获取variables和history参数，如果不存在则设置为空列表和空字典
        variables = kwargs.get("variables")
        history = kwargs.get("history") or []
        # 执行变量替换的工具函数，返回替换后的输入对象
        input = perform_variable_replacements(input, history, variables)
        # 调用委托的CompletionLLM对象处理替换后的输入对象，并返回结果
        return await self._delegate(input, **kwargs)
```