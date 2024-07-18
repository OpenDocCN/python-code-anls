# `.\graphrag\graphrag\llm\mock\mock_completion_llm.py`

```py
# 版权声明，指明代码版权属于 2024 年的 Microsoft Corporation，使用 MIT 许可证
# 引入日志记录模块
import logging

# 引入类型扩展 Unpack
from typing_extensions import Unpack

# 引入基础语言模型类 BaseLLM 和类型定义 CompletionInput、CompletionOutput、LLMInput
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
)

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


class MockCompletionLLM(
    BaseLLM[
        CompletionInput,
        CompletionOutput,
    ]
):
    """用于测试目的的模拟完成语言模型。"""

    def __init__(self, responses: list[str]):
        # 初始化模型对象，设置响应列表
        self.responses = responses
        # 错误处理回调函数，默认为空
        self._on_error = None

    async def _execute_llm(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput:
        # 执行语言模型，返回第一个响应
        return self.responses[0]
```