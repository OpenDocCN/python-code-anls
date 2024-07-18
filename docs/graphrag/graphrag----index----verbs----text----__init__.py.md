# `.\graphrag\graphrag\index\verbs\text\__init__.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Indexing Engine text package root."""

# 从当前包中导入以下模块和函数
from .chunk.text_chunk import chunk
from .embed import text_embed
from .replace import replace
from .split import text_split
from .translate import text_translate

# 定义一个列表，包含了当前模块中所有公开的符号（模块、类、函数等）
__all__ = [
    "chunk",            # 将 chunk 模块导出
    "replace",          # 将 replace 函数导出
    "text_embed",       # 将 text_embed 函数导出
    "text_split",       # 将 text_split 函数导出
    "text_translate",   # 将 text_translate 函数导出
]
```