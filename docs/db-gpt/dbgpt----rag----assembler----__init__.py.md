# `.\DB-GPT-src\dbgpt\rag\assembler\__init__.py`

```py
"""
Assembler Module For RAG.

The Assembler is a module that is responsible for assembling the knowledge.
"""

# 导入基础装配器类BaseAssembler
from .base import BaseAssembler  # noqa: F401
# 导入数据库模式装配器类DBSchemaAssembler
from .db_schema import DBSchemaAssembler  # noqa: F401
# 导入嵌入式装配器类EmbeddingAssembler
from .embedding import EmbeddingAssembler  # noqa: F401
# 导入摘要装配器类SummaryAssembler
from .summary import SummaryAssembler  # noqa: F401

# 定义导出的全部符号列表，包括所有导入的装配器类名
__all__ = [
    "BaseAssembler",
    "DBSchemaAssembler",
    "EmbeddingAssembler",
    "SummaryAssembler",
]
```