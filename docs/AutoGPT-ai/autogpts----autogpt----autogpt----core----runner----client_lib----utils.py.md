# `.\AutoGPT\autogpts\autogpt\autogpt\core\runner\client_lib\utils.py`

```py
# 导入必要的模块
import asyncio
import functools
from bdb import BdbQuit
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

import click

# 定义参数规范
P = ParamSpec("P")
T = TypeVar("T")

# 定义异常处理装饰器函数
def handle_exceptions(
    application_main: Callable[P, T],
    with_debugger: bool,
) -> Callable[P, T]:
    """Wraps a function so that it drops a user into a debugger if it raises an error.

    This is intended to be used as a wrapper for the main function of a CLI application.
    It will catch all errors and drop a user into a debugger if the error is not a
    `KeyboardInterrupt`. If the error is a `KeyboardInterrupt`, it will raise the error.
    If the error is not a `KeyboardInterrupt`, it will log the error and drop a user
    into a debugger if `with_debugger` is `True`.
    If `with_debugger` is `False`, it will raise the error.

    Parameters
    ----------
    application_main
        The function to wrap.
    with_debugger
        Whether to drop a user into a debugger if an error is raised.

    Returns
    -------
    Callable
        The wrapped function.

    """
    # 使用 functools.wraps 装饰器保留原始函数的元数据
    @functools.wraps(application_main)
    # 异步函数装饰器
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            # 尝试执行被装饰的函数
            return await application_main(*args, **kwargs)
        except (BdbQuit, KeyboardInterrupt, click.Abort):
            # 如果捕获到 BdbQuit, KeyboardInterrupt, click.Abort 异常，则重新抛出
            raise
        except Exception as e:
            if with_debugger:
                # 如果捕获到其他异常且 with_debugger 为 True，则打印异常信息并进入调试器
                print(f"Uncaught exception {e}")
                import pdb

                pdb.post_mortem()
            else:
                # 如果 with_debugger 为 False，则重新抛出异常
                raise

    return wrapped

# 定义协程装饰器函数
def coroutine(f: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, T]:
    @functools.wraps(f)
    # 包装函数，使用 asyncio.run 运行协程
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper
```