# `.\graphrag\graphrag\llm\types\llm.py`

```py
# 版权声明：2024年由 Microsoft Corporation 版权所有，根据 MIT 许可证授权

"""LLM 类型定义."""

# 引入类型相关的库和模块
from typing import Generic, Protocol, TypeVar

# 引入类型扩展
from typing_extensions import Unpack

# 从当前包导入 LLM 输入和输出类型
from .llm_io import (
    LLMInput,
    LLMOutput,
)

# 定义一个协变的类型变量 TIn，用于输入类型
TIn = TypeVar("TIn", contravariant=True)

# 定义一个不受限制的类型变量 TOut，用于输出类型
TOut = TypeVar("TOut")

# 定义 LLM 协议，支持泛型 TIn 和 TOut
class LLM(Protocol, Generic[TIn, TOut]):
    """LLM 协议的定义."""

    async def __call__(
        self,
        input: TIn,
        **kwargs: Unpack[LLMInput],
    ) -> LLMOutput[TOut]:
        """调用 LLM，将其视为函数处理."""
        ...
```