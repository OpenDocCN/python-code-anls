# `.\DB-GPT-src\dbgpt\rag\__init__.py`

```py
"""Module of RAG."""

# 导入必要的模块和类，包括 Chunk 和 Document
from dbgpt.core import Chunk, Document  # noqa: F401

# 从当前包导入 ChunkParameters 类
from .chunk_manager import ChunkParameters  # noqa: F401

# 定义导出的全部变量列表，包括 Chunk、Document 和 ChunkParameters
__ALL__ = [
    "Chunk",
    "Document",
    "ChunkParameters",
]
```