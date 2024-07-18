# `.\graphrag\graphrag\llm\limiting\llm_limiter.py`

```py
# 引入 ABC 抽象基类和 abstractmethod 装饰器
from abc import ABC, abstractmethod

# 定义一个抽象基类 LLMLimiter，继承自 ABC 抽象基类
class LLMLimiter(ABC):
    """LLM Limiter Interface."""
    
    # 定义属性装饰器，标记 needs_token_count 方法为抽象属性方法，返回布尔值
    @property
    @abstractmethod
    def needs_token_count(self) -> bool:
        """Whether this limiter needs the token count to be passed in."""
    
    # 标记 acquire 方法为抽象方法，表示派生类必须实现此方法，异步方法，用于获取通过限制器的通行证
    @abstractmethod
    async def acquire(self, num_tokens: int = 1) -> None:
        """Acquire a pass through the limiter."""
```