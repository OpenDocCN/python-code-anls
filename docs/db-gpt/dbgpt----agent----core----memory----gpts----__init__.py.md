# `.\DB-GPT-src\dbgpt\agent\core\memory\gpts\__init__.py`

```py
"""
Memory module for GPTS messages and plans.

It stores the messages and plans generated of multiple agents in the conversation.

It is different from the agent memory as it is a formatted structure to store the
messages and plans, and it can be stored in a database or a file.
"""

# 从当前目录中导入以下模块和类，忽略未使用的变量警告
from .base import (  # noqa: F401
    GptsMessage,           # 导入 GptsMessage 类
    GptsMessageMemory,     # 导入 GptsMessageMemory 类
    GptsPlan,              # 导入 GptsPlan 类
    GptsPlansMemory,       # 导入 GptsPlansMemory 类
)
from .default_gpts_memory import (  # noqa: F401
    DefaultGptsMessageMemory,   # 导入 DefaultGptsMessageMemory 类
    DefaultGptsPlansMemory,     # 导入 DefaultGptsPlansMemory 类
)
from .gpts_memory import GptsMemory  # noqa: F401
# 导入 GptsMemory 类
```