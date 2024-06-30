# `D:\src\scipysrc\scipy\scipy\_lib\_threadsafety.py`

```
import threading  # 导入线程模块

import scipy._lib.decorator  # 导入装饰器模块


__all__ = ['ReentrancyError', 'ReentrancyLock', 'non_reentrant']  # 公开的类和函数列表


class ReentrancyError(RuntimeError):
    pass  # 定义一个继承自 RuntimeError 的异常类


class ReentrancyLock:
    """
    Threading lock that raises an exception for reentrant calls.

    Calls from different threads are serialized, and nested calls from the
    same thread result to an error.

    The object can be used as a context manager or to decorate functions
    via the decorate() method.

    """

    def __init__(self, err_msg):
        self._rlock = threading.RLock()  # 初始化一个可重入锁对象
        self._entered = False  # 标志变量，记录锁的进入状态
        self._err_msg = err_msg  # 锁定时发生错误时的错误信息

    def __enter__(self):
        self._rlock.acquire()  # 获取锁
        if self._entered:
            self._rlock.release()  # 如果已经进入锁状态，则释放锁
            raise ReentrancyError(self._err_msg)  # 抛出重入错误异常
        self._entered = True  # 标记已经进入锁状态

    def __exit__(self, type, value, traceback):
        self._entered = False  # 重置进入锁状态标志
        self._rlock.release()  # 释放锁

    def decorate(self, func):
        def caller(func, *a, **kw):
            with self:
                return func(*a, **kw)
        return scipy._lib.decorator.decorate(func, caller)  # 使用 scipy 装饰器装饰函数


def non_reentrant(err_msg=None):
    """
    Decorate a function with a threading lock and prevent reentrant calls.
    """
    def decorator(func):
        msg = err_msg
        if msg is None:
            msg = "%s is not re-entrant" % func.__name__
        lock = ReentrancyLock(msg)  # 创建一个 ReentrancyLock 实例
        return lock.decorate(func)  # 使用 ReentrancyLock 实例装饰函数
    return decorator  # 返回装饰器函数
```