# `.\graphrag\graphrag\llm\mock\mock_chat_llm.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A mock ChatLLM that returns the given responses."""

# 从 typing_extensions 导入 Unpack 类型
from typing_extensions import Unpack

# 导入基础 LLM 类和类型定义
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
    LLMOutput,
)


class MockChatLLM(
    BaseLLM[
        CompletionInput,  # 泛型参数：输入为 CompletionInput 类型
        CompletionOutput,  # 泛型参数：输出为 CompletionOutput 类型
    ]
):
    """A mock LLM that returns the given responses."""

    responses: list[str]  # 声明一个列表 responses，存储字符串类型的响应内容
    i: int = 0  # 初始化计数器 i 为 0

    def __init__(self, responses: list[str]):
        self.i = 0  # 初始化计数器 i
        self.responses = responses  # 将传入的 responses 参数赋值给实例变量 responses

    def _create_output(
        self,
        output: CompletionOutput | None,  # 函数参数 output 可以是 CompletionOutput 或 None
        **kwargs: Unpack[LLMInput],  # **kwargs 接受除 output 外的所有 LLMInput 参数
    ) -> LLMOutput[CompletionOutput]:  # 函数返回类型为 LLMOutput[CompletionOutput]

        history = kwargs.get("history") or []  # 从 kwargs 中获取 history 参数，若不存在则使用空列表
        return LLMOutput[CompletionOutput](  # 返回一个 LLMOutput 实例
            output=output,  # 输出内容为参数 output
            history=[*history, {"content": output}]  # 历史记录为当前 history 加上新的记录 {"content": output}
        )

    async def _execute_llm(
        self,
        input: CompletionInput,  # 函数参数 input，类型为 CompletionInput
        **kwargs: Unpack[LLMInput],  # **kwargs 接受除 input 外的所有 LLMInput 参数
    ) -> CompletionOutput:  # 函数返回类型为 CompletionOutput

        if self.i >= len(self.responses):  # 如果计数器 i 大于等于 responses 列表长度
            msg = f"No more responses, requested {self.i} but only have {len(self.responses)}"
            raise ValueError(msg)  # 抛出 ValueError 异常，指示没有更多响应可用

        response = self.responses[self.i]  # 获取当前计数器指向的响应内容
        self.i += 1  # 计数器加 1，指向下一个响应内容
        return response  # 返回获取的响应内容
```