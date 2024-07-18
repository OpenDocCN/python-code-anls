# `.\graphrag\tests\unit\indexing\verbs\helpers\mock_llm.py`

```py
# 导入所需模块和类
from graphrag.llm import CompletionLLM, MockChatLLM

# 定义一个函数，用于创建一个模拟的语言模型（LLM）
# 函数接受一个字符串列表作为参数，表示模型需要返回的响应
def create_mock_llm(
    responses: list[str],
) -> CompletionLLM:
    """Creates a mock LLM that returns the given responses."""
    # 返回一个 MockChatLLM 对象，该对象能够根据输入的 responses 来模拟返回对话
    return MockChatLLM(responses)
```