# `.\graphrag\graphrag\llm\types\llm_cache.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Typing definitions for the OpenAI DataShaper package."""

# 引入必要的类型定义模块
from typing import Any, Protocol

# 定义一个名为 LLMCache 的协议（接口），包含异步方法签名
class LLMCache(Protocol):
    """LLM Cache interface."""

    # 异步方法，检查缓存中是否存在指定键的值
    async def has(self, key: str) -> bool:
        """Check if the cache has a value."""
        ...

    # 异步方法，从缓存中获取指定键的值，返回任意类型或 None
    async def get(self, key: str) -> Any | None:
        """Retrieve a value from the cache."""
        ...

    # 异步方法，将指定键值对写入缓存，可以附加调试数据
    async def set(self, key: str, value: Any, debug_data: dict | None = None) -> None:
        """Write a value into the cache."""
        ...
```