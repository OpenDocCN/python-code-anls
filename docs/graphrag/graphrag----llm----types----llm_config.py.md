# `.\graphrag\graphrag\llm\types\llm_config.py`

```py
# 导入必要的模块和类型声明
"""LLM Configuration Protocol definition."""
# 定义一个接口 LLMConfig，用于描述LLM配置协议的属性和行为
from typing import Protocol


class LLMConfig(Protocol):
    """LLM Configuration Protocol definition."""

    @property
    def max_retries(self) -> int | None:
        """Get the maximum number of retries."""
        # 获取最大重试次数的属性，返回一个整数或者None
        ...

    @property
    def max_retry_wait(self) -> float | None:
        """Get the maximum retry wait time."""
        # 获取最大重试等待时间的属性，返回一个浮点数或者None
        ...

    @property
    def sleep_on_rate_limit_recommendation(self) -> bool | None:
        """Get whether to sleep on rate limit recommendation."""
        # 获取是否应该在速率限制推荐时进行睡眠的属性，返回一个布尔值或者None
        ...

    @property
    def tokens_per_minute(self) -> int | None:
        """Get the number of tokens per minute."""
        # 获取每分钟令牌数量的属性，返回一个整数或者None
        ...

    @property
    def requests_per_minute(self) -> int | None:
        """Get the number of requests per minute."""
        # 获取每分钟请求次数的属性，返回一个整数或者None
        ...
```