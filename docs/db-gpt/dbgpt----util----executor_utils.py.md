# `.\DB-GPT-src\dbgpt\util\executor_utils.py`

```py
import asyncio
import contextvars
from abc import ABC, abstractmethod
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Optional

from dbgpt.component import BaseComponent, ComponentType, SystemApp


class ExecutorFactory(BaseComponent, ABC):
    name = ComponentType.EXECUTOR_DEFAULT.value

    @abstractmethod
    def create(self) -> "Executor":
        """Create executor"""


class DefaultExecutorFactory(ExecutorFactory):
    def __init__(self, system_app: SystemApp | None = None, max_workers=None):
        super().__init__(system_app)
        # 创建一个线程池执行器，用于执行并发任务
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=self.name
        )

    def init_app(self, system_app: SystemApp):
        pass

    def create(self) -> Executor:
        # 返回创建的执行器实例
        return self._executor


BlockingFunction = Callable[..., Any]


async def blocking_func_to_async(
    executor: Executor, func: BlockingFunction, *args, **kwargs
):
    """Run a potentially blocking function within an executor.

    Args:
        executor (Executor): The concurrent.futures.Executor to run the function within.
        func (ApplyFunction): The callable function, which should be a synchronous function.
            It should accept any number and type of arguments and return an asynchronous coroutine.
        *args (Any): Any additional arguments to pass to the function.
        **kwargs (Any): Other arguments to pass to the function

    Returns:
        Any: The result of the function's execution.

    Raises:
        ValueError: If the provided function 'func' is an asynchronous coroutine function.

    This function allows you to execute a potentially blocking function within an executor.
    It expects 'func' to be a synchronous function and will raise an error if 'func' is an asynchronous coroutine.
    """
    if asyncio.iscoroutinefunction(func):
        # 如果 func 是异步协程函数，则抛出值错误异常
        raise ValueError(f"The function {func} is not a blocking function")

    # 创建一个闭包函数，在新线程中运行，并捕获当前上下文
    ctx = contextvars.copy_context()

    def run_with_context():
        return ctx.run(partial(func, *args, **kwargs))

    loop = asyncio.get_event_loop()
    # 在执行器中异步运行闭包函数
    return await loop.run_in_executor(executor, run_with_context)


async def blocking_func_to_async_no_executor(func: BlockingFunction, *args, **kwargs):
    """Run a potentially blocking function within an executor."""
    # 调用 blocking_func_to_async 函数，传入 None 作为执行器，转换为异步执行
    return await blocking_func_to_async(None, func, *args, **kwargs)  # type: ignore


class AsyncToSyncIterator:
    def __init__(self, async_iterable, loop: asyncio.BaseEventLoop):
        # 初始化 AsyncToSyncIterator 类，接收一个异步可迭代对象和事件循环
        self.async_iterable = async_iterable
        self.async_iterator = None
        self._loop = loop

    def __iter__(self):
        # 实现迭代器协议，返回一个迭代器对象
        self.async_iterator = self.async_iterable.__aiter__()
        return self
    # 定义一个特殊方法 __next__，用于迭代器的下一个元素获取
    def __next__(self):
        # 如果 async_iterator 属性为 None，则抛出 StopIteration 异常
        if self.async_iterator is None:
            raise StopIteration

        # 尝试通过事件循环执行 async_iterator 的 __anext__() 方法来获取下一个元素
        try:
            return self._loop.run_until_complete(self.async_iterator.__anext__())
        # 如果 async_iterator 已经停止异步迭代，则捕获 StopAsyncIteration 异常并抛出 StopIteration
        except StopAsyncIteration:
            raise StopIteration
```