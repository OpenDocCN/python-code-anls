# `.\AutoGPT\autogpts\autogpt\tests\integration\memory\conftest.py`

```py
# 导入 pytest 模块
import pytest

# 从 autogpt.memory.vector.memory_item 模块中导入 MemoryItem 类
from autogpt.memory.vector.memory_item import MemoryItem
# 从 autogpt.memory.vector.utils 模块中导入 Embedding 类
from autogpt.memory.vector.utils import Embedding

# 定义一个 pytest 的 fixture，用于创建 MemoryItem 对象
@pytest.fixture
def memory_item(mock_embedding: Embedding):
    # 创建一个 MemoryItem 对象，传入原始内容、内容摘要、内容块、内容块摘要、摘要的嵌入、内容块的嵌入和元数据
    return MemoryItem(
        raw_content="test content",
        summary="test content summary",
        chunks=["test content"],
        chunk_summaries=["test content summary"],
        e_summary=mock_embedding,
        e_chunks=[mock_embedding],
        metadata={},
    )
```