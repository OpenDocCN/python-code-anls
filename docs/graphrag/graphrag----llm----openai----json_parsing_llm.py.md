# `.\graphrag\graphrag\llm\openai\json_parsing_llm.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""An LLM that unpacks cached JSON responses."""

# 从 typing_extensions 导入 Unpack 类型提示
from typing_extensions import Unpack

# 导入 LLM 相关类型定义
from graphrag.llm.types import (
    LLM,
    CompletionInput,
    CompletionLLM,
    CompletionOutput,
    LLMInput,
    LLMOutput,
)

# 导入本地的 try_parse_json_object 工具函数
from .utils import try_parse_json_object

# 定义 JsonParsingLLM 类，继承自 LLM[CompletionInput, CompletionOutput]
class JsonParsingLLM(LLM[CompletionInput, CompletionOutput]):
    """An OpenAI History-Tracking LLM."""

    # 私有属性 _delegate，类型为 CompletionLLM
    _delegate: CompletionLLM

    # 初始化方法，接受一个 delegate 参数
    def __init__(self, delegate: CompletionLLM):
        self._delegate = delegate

    # 异步调用方法，接受 input 和 kwargs 参数
    async def __call__(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> LLMOutput[CompletionOutput]:
        """Call the LLM with the input and kwargs."""
        # 调用 _delegate 方法处理 input 和 kwargs，并获取结果
        result = await self._delegate(input, **kwargs)
        
        # 如果 kwargs 中包含 "json" 并且 result 的 json 属性为 None，但 output 属性不为 None
        if kwargs.get("json") and result.json is None and result.output is not None:
            # 尝试解析 result.output 为 JSON 对象，并将结果赋给 result.json
            result.json = try_parse_json_object(result.output)
        
        # 返回处理后的 result 对象
        return result
```