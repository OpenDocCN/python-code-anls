# `.\graphrag\graphrag\llm\mock\__init__.py`

```py
# 版权声明和许可证信息，声明版权归 Microsoft Corporation 所有，使用 MIT 许可证
# Licensed under the MIT License

# 导入模拟的聊天语言模型和完成语言模型
from .mock_chat_llm import MockChatLLM
from .mock_completion_llm import MockCompletionLLM

# 定义了 __all__ 列表，用于指定在使用 `from module import *` 语法时应该导入的符号
__all__ = [
    "MockChatLLM",           # 导出 MockChatLLM 符号
    "MockCompletionLLM",     # 导出 MockCompletionLLM 符号
]
```