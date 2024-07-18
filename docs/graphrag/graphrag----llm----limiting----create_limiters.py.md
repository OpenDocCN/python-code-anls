# `.\graphrag\graphrag\llm\limiting\create_limiters.py`

```py
# 导入日志模块，用于记录程序运行时的信息
import logging

# 导入异步限制器模块，用于控制并发请求的频率
from aiolimiter import AsyncLimiter

# 导入配置类型LLMConfig，用于获取配置信息
from graphrag.llm.types import LLMConfig

# 导入本地定义的限制器类LLMLimiter，用于封装模型的请求限制
from .llm_limiter import LLMLimiter

# 导入本地定义的TPM/RPM限制器类TpmRpmLLMLimiter，用于封装TPM和RPM的限制策略
from .tpm_rpm_limiter import TpmRpmLLMLimiter

# 获取当前模块的日志记录器对象
log = logging.getLogger(__name__)

# 创建函数，用于根据给定模型名称创建TPM和RPM的限制器
def create_tpm_rpm_limiters(
    configuration: LLMConfig,
) -> LLMLimiter:
    """Get the limiters for a given model name."""
    # 从配置对象中获取TPM和RPM的限制值
    tpm = configuration.tokens_per_minute
    rpm = configuration.requests_per_minute
    # 根据获取的TPM和RPM值，创建TPM/RPM限制器对象
    return TpmRpmLLMLimiter(
        # 如果TPM为0，则使用None；否则创建一个异步限制器对象，限制速率为tpm或最大值50,000
        None if tpm == 0 else AsyncLimiter(tpm or 50_000),
        # 如果RPM为0，则使用None；否则创建一个异步限制器对象，限制速率为rpm或最大值10,000
        None if rpm == 0 else AsyncLimiter(rpm or 10_000),
    )
```