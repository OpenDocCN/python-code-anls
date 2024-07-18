# `.\graphrag\graphrag\index\verbs\text\chunk\typing.py`

```py
# 版权声明，版权归 Microsoft Corporation 所有，基于 MIT 许可证发布

"""包含 'TextChunk' 模型的模块。"""

# 导入必要的数据类模块
from dataclasses import dataclass

# 定义数据类 'TextChunk'
@dataclass
class TextChunk:
    """文本块类的定义。"""

    # 文本块的内容，为字符串类型
    text_chunk: str
    # 来源文档的索引列表，为整数列表
    source_doc_indices: list[int]
    # 令牌数目，可选，可以是整数或者 None
    n_tokens: int | None = None


# 定义 'ChunkInput' 类型别名，表示用于块分割策略的输入
ChunkInput = str | list[str] | list[tuple[str, str]]
"""块分割策略的输入。可以是字符串、字符串列表或者元组列表，每个元组包含 (id, text)。"""
```