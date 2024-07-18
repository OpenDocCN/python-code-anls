# `.\graphrag\graphrag\llm\limiting\noop_llm_limiter.py`

```py
# 版权声明和许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入LLMLimiter类
from .llm_limiter import LLMLimiter

# 定义一个名为NoopLLMLimiter的类，继承自LLMLimiter类
class NoopLLMLimiter(LLMLimiter):
    """TPM RPM Limiter class definition."""
    
    # 定义一个属性方法，用于判断此限制器是否需要传入令牌计数
    @property
    def needs_token_count(self) -> bool:
        """Whether this limiter needs the token count to be passed in."""
        return False
    
    # 定义一个异步方法，用于获取令牌（但实际上什么也不做）
    async def acquire(self, num_tokens: int = 1) -> None:
        """Call method definition."""
        # do nothing
```