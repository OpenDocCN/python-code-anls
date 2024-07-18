# `.\graphrag\graphrag\index\utils\rate_limiter.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 引入 asyncio 和 time 模块
import asyncio
import time

# 定义一个速率限制器类 RateLimiter
class RateLimiter:
    """
    The original TpmRpmLLMLimiter strategy did not account for minute-based rate limiting when scheduled.

    The RateLimiter was introduced to ensure that the CommunityReportsExtractor could be scheduled to adhere to rate configurations on a per-minute basis.
    """

    # 初始化方法，接受两个参数 rate 和 per，表示速率和时间段
    def __init__(self, rate: int, per: int):
        self.rate = rate            # 每 per 秒允许的请求数
        self.per = per              # 时间段长度，以秒为单位
        self.allowance = rate       # 初始剩余可用请求数等于总请求数
        self.last_check = time.monotonic()  # 上次检查时间，初始化为当前时间

    # 异步方法，用于获取一个令牌
    async def acquire(self):
        """Acquire a token from the rate limiter."""
        current = time.monotonic()   # 获取当前时间
        elapsed = current - self.last_check  # 计算距离上次检查经过的时间
        self.last_check = current    # 更新上次检查时间
        self.allowance += elapsed * (self.rate / self.per)  # 根据经过时间补充剩余请求数

        if self.allowance > self.rate:
            self.allowance = self.rate  # 限制剩余请求数不超过总请求数

        if self.allowance < 1.0:
            sleep_time = (1.0 - self.allowance) * (self.per / self.rate)  # 计算需要等待的时间
            await asyncio.sleep(sleep_time)  # 异步等待指定的时间
            self.allowance = 0.0    # 等待后，剩余请求数归零
        else:
            self.allowance -= 1.0   # 否则，消耗一个请求数量单位
```