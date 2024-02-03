# `.\AutoGPT\autogpts\autogpt\autogpt\memory\vector\providers\no_memory.py`

```py
# 一个不存储任何数据的类。这是默认的内存提供程序。
from __future__ import annotations

from typing import Iterator, Optional

from autogpt.config.config import Config

from .. import MemoryItem
from .base import VectorMemoryProvider


class NoMemory(VectorMemoryProvider):
    """
    一个不存储任何数据的类。这是默认的内存提供程序。
    """

    def __init__(self, config: Optional[Config] = None):
        # 初始化方法，不执行任何操作
        pass

    def __iter__(self) -> Iterator[MemoryItem]:
        # 返回一个空的迭代器
        return iter([])

    def __contains__(self, x: MemoryItem) -> bool:
        # 判断是否包含指定的 MemoryItem，始终返回 False
        return False

    def __len__(self) -> int:
        # 返回内存中的数据数量，始终返回 0
        return 0

    def add(self, item: MemoryItem):
        # 向内存中添加数据的方法，不执行任何操作
        pass

    def discard(self, item: MemoryItem):
        # 从内存中删除数据的方法，不执行任何操作
        pass

    def clear(self):
        # 清空内存中的数据，不执行任何操作
        pass
```