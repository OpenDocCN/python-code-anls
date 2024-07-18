# `.\graphrag\graphrag\llm\openai\openai_history_tracking_llm.py`

```py
# 从 typing_extensions 导入 Unpack，用于解包参数类型
from typing_extensions import Unpack

# 从 graphrag.llm.types 导入所需的类型
from graphrag.llm.types import (
    LLM,                    # 导入 LLM 类型
    CompletionInput,        # 导入 CompletionInput 类型
    CompletionLLM,          # 导入 CompletionLLM 类型
    CompletionOutput,       # 导入 CompletionOutput 类型
    LLMInput,               # 导入 LLMInput 类型
    LLMOutput,              # 导入 LLMOutput 类型
)

# 定义一个 OpenAIHistoryTrackingLLM 类，继承自 LLM 类，其输入和输出分别为 CompletionInput 和 CompletionOutput
class OpenAIHistoryTrackingLLM(LLM[CompletionInput, CompletionOutput]):
    """An OpenAI History-Tracking LLM."""
    
    _delegate: CompletionLLM   # 类的私有属性 _delegate，类型为 CompletionLLM

    def __init__(self, delegate: CompletionLLM):
        self._delegate = delegate  # 初始化方法，接受一个 delegate 参数作为 CompletionLLM 类型的实例

    async def __call__(
        self,
        input: CompletionInput,          # 异步调用方法，接受 CompletionInput 类型的 input 参数
        **kwargs: Unpack[LLMInput],      # **kwargs 接受任意数量的关键字参数，类型为 Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:   # 返回类型为 LLMOutput[CompletionOutput]

        """Call the LLM."""
        
        history = kwargs.get("history") or []  # 获取关键字参数中的 "history"，如果不存在则为一个空列表
        output = await self._delegate(input, **kwargs)  # 调用 self._delegate 方法处理输入和参数，获取输出
        
        # 返回一个 LLMOutput 对象，包括 output.output、output.json 和更新后的 history
        return LLMOutput(
            output=output.output,
            json=output.json,
            history=[*history, {"role": "system", "content": output.output}],
        )
```