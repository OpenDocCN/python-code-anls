# `.\graphrag\graphrag\query\llm\oai\typing.py`

```py
# Copyright (`
# 版权声明，版权归 Microsoft Corporation 所有，使用 MIT 许可证授权

"""OpenAI 包装器选项。"""

# 引入所需模块和类
from enum import Enum
from typing import Any, cast

# 引入 OpenAI 模块
import openai

# 定义 OpenAI 错误类型的元组，用于重试机制
OPENAI_RETRY_ERROR_TYPES = (
    # 当更新到 OpenAI 1+ 库时更新这些
    cast(Any, openai).RateLimitError,
    cast(Any, openai).APIConnectionError,
    # 用可比较的 OpenAI 1+ 错误替换这些
)

# 定义 OpenAI API 类型枚举
class OpenaiApiType(str, Enum):
    """OpenAI 的不同类型。"""
    
    # OpenAI 类型
    OpenAI = "openai"
    # Azure OpenAI 类型
    AzureOpenAI = "azure"
```