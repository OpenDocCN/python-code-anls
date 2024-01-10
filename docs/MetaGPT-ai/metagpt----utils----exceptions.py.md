# `MetaGPT\metagpt\utils\exceptions.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19 14:46
@Author  : alexanderwu
@File    : exceptions.py
"""

# 导入必要的模块
import asyncio
import functools
import traceback
from typing import Any, Callable, Tuple, Type, TypeVar, Union

# 导入自定义模块
from metagpt.logs import logger

# 定义一个类型变量
ReturnType = TypeVar("ReturnType")

# 定义一个装饰器函数，用于处理异常并返回默认值
def handle_exception(
    _func: Callable[..., ReturnType] = None,
    *,
    exception_type: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    exception_msg: str = "",
    default_return: Any = None,
) -> Callable[..., ReturnType]:
    """handle exception, return default value"""

    # 定义装饰器函数
    def decorator(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        # 如果被装饰的函数是异步函数
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            try:
                return await func(*args, **kwargs)
            except exception_type as e:
                # 记录异常信息和调用栈
                logger.opt(depth=1).error(
                    f"{e}: {exception_msg}, "
                    f"\nCalling {func.__name__} with args: {args}, kwargs: {kwargs} "
                    f"\nStack: {traceback.format_exc()}"
                )
                return default_return

        # 如果被装饰的函数是同步函数
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                # 记录异常信息和调用栈
                logger.opt(depth=1).error(
                    f"Calling {func.__name__} with args: {args}, kwargs: {kwargs} failed: {e}, "
                    f"stack: {traceback.format_exc()}"
                )
                return default_return

        # 根据被装饰的函数类型返回相应的包装函数
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    # 如果没有传入被装饰的函数，则返回装饰器函数
    if _func is None:
        return decorator
    else:
        return decorator(_func)

```