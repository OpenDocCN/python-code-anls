# `.\DB-GPT-src\dbgpt\util\retry.py`

```py
# 导入 asyncio 异步编程库
import asyncio
# 导入 logging 日志记录库
import logging
# 导入 traceback 用于打印异常堆栈信息
import traceback

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)


# 定义异步重试装饰器函数 async_retry
def async_retry(
    retries: int = 1, parallel_executions: int = 1, catch_exceptions=(Exception,)
):
    """Async retry decorator.

    Examples:
        .. code-block:: python

            @async_retry(retries=3, parallel_executions=2)
            async def my_func():
                # Some code that may raise exceptions
                pass

    Args:
        retries (int): Number of retries.
        parallel_executions (int): Number of parallel executions.
        catch_exceptions (tuple): Tuple of exceptions to catch.
    """

    # 定义装饰器函数 decorator
    def decorator(func):
        # 定义装饰后的异步函数 wrapper
        async def wrapper(*args, **kwargs):
            # 初始化最后一次异常变量
            last_exception = None
            # 循环尝试 retries 次
            for attempt in range(retries):
                # 创建并发执行的任务列表
                tasks = [func(*args, **kwargs) for _ in range(parallel_executions)]
                # 并发执行任务，返回结果列表，并捕获所有异常
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 遍历并处理每个任务的结果
                for result in results:
                    # 如果结果不是异常，则直接返回结果
                    if not isinstance(result, Exception):
                        return result
                    # 如果结果是指定的异常类型，则记录该异常，并打印错误日志和调试堆栈信息
                    if isinstance(result, catch_exceptions):
                        last_exception = result
                        logger.error(
                            f"Attempt {attempt + 1} of {retries} failed with error: "
                            f"{type(result).__name__}, {str(result)}"
                        )
                        logger.debug(traceback.format_exc())

                # 记录重试信息
                logger.info(f"Retrying... (Attempt {attempt + 1} of {retries})")

            # 若所有重试均未成功，则抛出最后一次捕获的异常
            raise last_exception  # After all retries, raise the last caught exception

        return wrapper

    return decorator
```