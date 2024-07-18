# `.\graphrag\graphrag\llm\limiting\tpm_rpm_limiter.py`

```py
# 版权声明，版权归 Microsoft Corporation 所有，基于 MIT 许可证发布

"""TPM RPM Limiter module."""
# 导入必要的模块和类
from aiolimiter import AsyncLimiter
from .llm_limiter import LLMLimiter

# 定义 TPM RPM 限速器类，继承自 LLMLimiter 类
class TpmRpmLLMLimiter(LLMLimiter):
    """TPM RPM Limiter class definition."""

    # 异步限速器对象，用于 TPM 限速
    _tpm_limiter: AsyncLimiter | None
    # 异步限速器对象，用于 RPM 限速
    _rpm_limiter: AsyncLimiter | None

    def __init__(
        self, tpm_limiter: AsyncLimiter | None, rpm_limiter: AsyncLimiter | None
    ):
        """Init method definition."""
        # 初始化方法，接收 TPM 和 RPM 限速器对象作为参数
        self._tpm_limiter = tpm_limiter
        self._rpm_limiter = rpm_limiter

    @property
    def needs_token_count(self) -> bool:
        """Whether this limiter needs the token count to be passed in."""
        # 判断是否需要传入令牌计数
        return self._tpm_limiter is not None

    async def acquire(self, num_tokens: int = 1) -> None:
        """Call method definition."""
        # 异步获取令牌方法，用于 TPM 限速器
        if self._tpm_limiter is not None:
            await self._tpm_limiter.acquire(num_tokens)
        # 异步获取令牌方法，用于 RPM 限速器
        if self._rpm_limiter is not None:
            await self._rpm_limiter.acquire()
```