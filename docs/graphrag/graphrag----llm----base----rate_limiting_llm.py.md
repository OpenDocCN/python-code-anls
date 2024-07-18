# `.\graphrag\graphrag\llm\base\rate_limiting_llm.py`

```py
# 导入 asyncio 异步编程库
import asyncio
# 导入日志记录库 logging
import logging
# 导入泛型类型的支持模块
from collections.abc import Callable
from typing import Any, Generic, TypeVar

# 导入 tenacity 库中的重试策略相关模块
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
# 导入 typing_extensions 库中的 Unpack 类型
from typing_extensions import Unpack

# 导入自定义错误模块中的 RetriesExhaustedError 异常类
from graphrag.llm.errors import RetriesExhaustedError
# 导入限流器模块中的 LLMLimiter 类
from graphrag.llm.limiting import LLMLimiter
# 导入类型定义模块中的自定义类型
from graphrag.llm.types import (
    LLM,
    LLMConfig,
    LLMInput,
    LLMInvocationFn,
    LLMInvocationResult,
    LLMOutput,
)

# 定义泛型类型 TIn 和 TOut
TIn = TypeVar("TIn")
TOut = TypeVar("TOut")
# 定义泛型类型 TRateLimitError
TRateLimitError = TypeVar("TRateLimitError", bound=BaseException)

# 输入无法衡量输入令牌的错误信息
_CANNOT_MEASURE_INPUT_TOKENS_MSG = "cannot measure input tokens"
# 输入无法衡量输出令牌的错误信息
_CANNOT_MEASURE_OUTPUT_TOKENS_MSG = "cannot measure output tokens"

# 设置日志记录器对象
log = logging.getLogger(__name__)


class RateLimitingLLM(LLM[TIn, TOut], Generic[TIn, TOut]):
    """用于限流的低延迟模型类。"""

    # 委托的低延迟模型对象
    _delegate: LLM[TIn, TOut]
    # 限流器对象或 None
    _rate_limiter: LLMLimiter | None
    # 异步信号量对象或 None
    _semaphore: asyncio.Semaphore | None
    # 计算令牌数量的回调函数
    _count_tokens: Callable[[str], int]
    # LLM 配置对象
    _config: LLMConfig
    # 操作名称字符串
    _operation: str
    # 可重试错误类型列表
    _retryable_errors: list[type[Exception]]
    # 限流错误类型列表
    _rate_limit_errors: list[type[Exception]]
    # 调用时的回调函数
    _on_invoke: LLMInvocationFn
    # 获取推荐休眠时间的回调函数
    _extract_sleep_recommendation: Callable[[Any], float]

    def __init__(
        self,
        delegate: LLM[TIn, TOut],
        config: LLMConfig,
        operation: str,
        retryable_errors: list[type[Exception]],
        rate_limit_errors: list[type[Exception]],
        rate_limiter: LLMLimiter | None = None,
        semaphore: asyncio.Semaphore | None = None,
        count_tokens: Callable[[str], int] | None = None,
        get_sleep_time: Callable[[BaseException], float] | None = None,
    ):
        """初始化方法，设置各种参数和回调函数。"""
        self._delegate = delegate
        self._rate_limiter = rate_limiter
        self._semaphore = semaphore
        self._config = config
        self._operation = operation
        self._retryable_errors = retryable_errors
        self._rate_limit_errors = rate_limit_errors
        # 如果未提供计算令牌数量的回调函数，则默认返回 -1
        self._count_tokens = count_tokens or (lambda _s: -1)
        # 如果未提供获取推荐休眠时间的回调函数，则默认返回 0.0
        self._extract_sleep_recommendation = get_sleep_time or (lambda _e: 0.0)
        # 默认设置调用时的回调函数为空函数
        self._on_invoke = lambda _v: None

    def on_invoke(self, fn: LLMInvocationFn | None) -> None:
        """设置调用时的回调函数。"""
        self._on_invoke = fn or (lambda _v: None)
    # 计算输入请求中的令牌数
    def count_request_tokens(self, input: TIn) -> int:
        """Count the request tokens on an input request."""
        # 如果输入是字符串，调用私有方法计算其令牌数并返回
        if isinstance(input, str):
            return self._count_tokens(input)
        # 如果输入是列表
        if isinstance(input, list):
            result = 0
            # 遍历列表中的每个元素
            for item in input:
                # 如果元素是字符串，计算其令牌数并累加到结果中
                if isinstance(item, str):
                    result += self._count_tokens(item)
                # 如果元素是字典，获取键为"content"的值（默认为空字符串），计算其令牌数并累加到结果中
                elif isinstance(item, dict):
                    result += self._count_tokens(item.get("content", ""))
                else:
                    # 抛出类型错误异常，无法计算输入的令牌数消息
                    raise TypeError(_CANNOT_MEASURE_INPUT_TOKENS_MSG)
            return result
        # 如果输入类型不支持，抛出类型错误异常，无法计算输入的令牌数消息
        raise TypeError(_CANNOT_MEASURE_INPUT_TOKENS_MSG)

    # 计算输出响应中的令牌数
    def count_response_tokens(self, output: TOut | None) -> int:
        """Count the request tokens on an output response."""
        # 如果输出为 None，返回令牌数为 0
        if output is None:
            return 0
        # 如果输出是字符串，调用私有方法计算其令牌数并返回
        if isinstance(output, str):
            return self._count_tokens(output)
        # 如果输出是字符串列表且所有元素都是字符串，计算列表中所有字符串的令牌数并返回
        if isinstance(output, list) and all(isinstance(x, str) for x in output):
            return sum(self._count_tokens(item) for item in output)
        # 如果输出是列表，但不是所有元素都是字符串，认为是嵌入式响应，返回令牌数为 0
        if isinstance(output, list):
            # 嵌入式响应，不计数令牌
            return 0
        # 如果输出类型不支持，抛出类型错误异常，无法计算输出的令牌数消息
        raise TypeError(_CANNOT_MEASURE_OUTPUT_TOKENS_MSG)

    # 处理异步调用结果的私有方法
    def _handle_invoke_result(
        self, result: LLMInvocationResult[LLMOutput[TOut]]
    ) -> None:
        # 记录调试信息，包括操作名称、结果名称、重试次数、总时间以及输入和输出的令牌数
        log.info(
            'perf - llm.%s "%s" with %s retries took %s. input_tokens=%d, output_tokens=%d',
            self._operation,
            result.name,
            result.num_retries,
            result.total_time,
            result.input_tokens,
            result.output_tokens,
        )
        # 处理调用结果的其他操作
        self._on_invoke(result)
```