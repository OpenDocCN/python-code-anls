# `.\graphrag\graphrag\index\verbs\text\chunk\__init__.py`

```py
# 版权声明，声明代码版权归 Microsoft Corporation 所有，使用 MIT 许可证授权
# Licensed under the MIT License

"""索引引擎文本块包的根目录。"""
# 引入当前包中的 text_chunk 模块中的 ChunkStrategy, ChunkStrategyType, chunk 三个成员
from .text_chunk import ChunkStrategy, ChunkStrategyType, chunk

# 声明当前模块中所有公开的符号，包括 ChunkStrategy, ChunkStrategyType, chunk
__all__ = ["ChunkStrategy", "ChunkStrategyType", "chunk"]
```