# `.\DB-GPT-src\dbgpt\util\cache_utils.py`

```py
"""
Cache utils.

Adapted from https://github.com/hephex/asyncache/blob/master/asyncache/__init__.py.
It has stopped updating since 2022. So I copied the code here for future reference.
"""

import asyncio
import functools
from contextlib import AbstractContextManager
from typing import Any, Callable, MutableMapping, Optional, Protocol, TypeVar

from cachetools import keys

_KT = TypeVar("_KT")
_T = TypeVar("_T")


class IdentityFunction(Protocol):  # pylint: disable=too-few-public-methods
    """
    Type for a function returning the same type as the one it received.
    """

    def __call__(self, __x: _T) -> _T:
        ...


class NullContext:
    """A class for noop context managers."""

    def __enter__(self):
        """Return ``self`` upon entering the runtime context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Raise any exception triggered within the runtime context."""
        return None

    async def __aenter__(self):
        """Return ``self`` upon entering the runtime context."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Raise any exception triggered within the runtime context."""
        return None


def cached(
    cache: Optional[MutableMapping[_KT, Any]],
    # ignoring the mypy error to be consistent with the type used
    # in https://github.com/python/typeshed/tree/master/stubs/cachetools
    key: Callable[..., _KT] = keys.hashkey,  # type:ignore
    lock: Optional["AbstractContextManager[Any]"] = None,
) -> IdentityFunction:
    """
    Decorator to wrap a function or a coroutine with a memoizing callable
    that saves results in a cache.

    When ``lock`` is provided for a standard function, it's expected to
    implement ``__enter__`` and ``__exit__`` that will be used to lock
    the cache when gets updated. If it wraps a coroutine, ``lock``
    must implement ``__aenter__`` and ``__aexit__``.
    """
    # Set default value for lock if not provided
    lock = lock or NullContext()
    # 定义一个装饰器函数，接受一个函数作为参数
    def decorator(func):
        # 如果传入的函数是异步函数
        if asyncio.iscoroutinefunction(func):

            # 定义一个异步函数作为包装器，接受任意参数
            async def wrapper(*args, **kwargs):
                # 根据参数生成一个键
                k = key(*args, **kwargs)
                try:
                    # 使用异步锁，尝试从缓存中获取值
                    async with lock:
                        return cache[k]

                except KeyError:
                    pass  # key not found

                # 调用传入的异步函数
                val = await func(*args, **kwargs)

                try:
                    # 使用异步锁，尝试将值存入缓存
                    async with lock:
                        cache[k] = val

                except ValueError:
                    pass  # val too large

                return val

        else:

            # 定义一个普通函数作为包装器，接受任意参数
            def wrapper(*args, **kwargs):
                # 根据参数生成一个键
                k = key(*args, **kwargs)
                try:
                    # 使用普通锁，尝试从缓存中获取值
                    with lock:
                        return cache[k]

                except KeyError:
                    pass  # key not found

                # 调用传入的普通函数
                val = func(*args, **kwargs)

                try:
                    # 使用普通锁，尝试将值存入缓存
                    with lock:
                        cache[k] = val

                except ValueError:
                    pass  # val too large

                return val

        # 返回经过包装的函数
        return functools.wraps(func)(wrapper)

    # 返回装饰器函数
    return decorator
```