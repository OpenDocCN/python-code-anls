# `.\graphrag\graphrag\llm\limiting\composite_limiter.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing Composite Limiter class definition."""

# 从当前目录导入LLMLimiter类
from .llm_limiter import LLMLimiter

# 定义一个继承自LLMLimiter类的CompositeLLMLimiter类
class CompositeLLMLimiter(LLMLimiter):
    """Composite Limiter class definition."""

    # 类变量_limiters，类型为LLMLimiter类的列表
    _limiters: list[LLMLimiter]

    # 初始化方法，接受一个LLMLimiter类的列表作为参数
    def __init__(self, limiters: list[LLMLimiter]):
        """Init method definition."""
        self._limiters = limiters

    # 属性方法，返回布尔值，表示是否需要传入token计数
    @property
    def needs_token_count(self) -> bool:
        """Whether this limiter needs the token count to be passed in."""
        # 使用任意一个limiter对象的needs_token_count属性来确定返回值
        return any(limiter.needs_token_count for limiter in self._limiters)

    # 异步方法，用于获取指定数量的token
    async def acquire(self, num_tokens: int = 1) -> None:
        """Call method definition."""
        # 遍历_limiters列表中的每个limiter对象，并调用其acquire方法
        for limiter in self._limiters:
            await limiter.acquire(num_tokens)
```