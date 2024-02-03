# `.\AutoGPT\autogpts\autogpt\autogpt\core\memory\__init__.py`

```py
"""The memory subsystem manages the Agent's long-term memory."""
# 该内存子系统管理Agent的长期记忆

from autogpt.core.memory.base import Memory
# 从autogpt.core.memory.base模块导入Memory类
from autogpt.core.memory.simple import MemorySettings, SimpleMemory
# 从autogpt.core.memory.simple模块导入MemorySettings和SimpleMemory类

__all__ = [
    "Memory",
    "MemorySettings",
    "SimpleMemory",
]
# 将Memory、MemorySettings和SimpleMemory类添加到__all__列表中，表示这些类是该模块的公共接口
```