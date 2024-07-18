# `.\graphrag\graphrag\query\llm\oai\__init__.py`

```py
# 2024年版权归 Microsoft 公司所有。
# 根据 MIT 许可证授权

"""GraphRAG Orchestration OpenAI Wrappers."""
# 导入基础模块和类
from .base import BaseOpenAILLM, OpenAILLMImpl, OpenAITextEmbeddingImpl
# 导入聊天相关的类
from .chat_openai import ChatOpenAI
# 导入嵌入相关的类
from .embedding import OpenAIEmbedding
# 导入 OpenAI 相关的类和函数
from .openai import OpenAI
# 导入类型相关定义
from .typing import OPENAI_RETRY_ERROR_TYPES, OpenaiApiType

# 将所有公开的符号放入 __all__ 列表，以便在 from module import * 时导入
__all__ = [
    "OPENAI_RETRY_ERROR_TYPES",
    "BaseOpenAILLM",
    "ChatOpenAI",
    "OpenAI",
    "OpenAIEmbedding",
    "OpenAILLMImpl",
    "OpenAITextEmbeddingImpl",
    "OpenaiApiType",
]
```