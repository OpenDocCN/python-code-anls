# `D:\src\scipysrc\scipy\scipy\_lib\_gcutils.py`

```
"""
Module for testing automatic garbage collection of objects

.. autosummary::
   :toctree: generated/

   set_gc_state - enable or disable garbage collection
   gc_state - context manager for given state of garbage collector
   assert_deallocated - context manager to check for circular references on object

"""
import weakref  # 导入 weakref 模块，用于创建弱引用对象
import gc       # 导入 gc 模块，用于垃圾回收操作

from contextlib import contextmanager   # 导入 contextmanager 类，用于创建上下文管理器
from platform import python_implementation  # 导入 python_implementation 函数，用于获取 Python 实现的名称

__all__ = ['set_gc_state', 'gc_state', 'assert_deallocated']   # 定义模块公开的接口列表

IS_PYPY = python_implementation() == 'PyPy'   # 检查当前 Python 是否为 PyPy 实现


class ReferenceError(AssertionError):
    """ Exception class for reference errors during deallocation checks """
    pass


def set_gc_state(state):
    """ Set status of garbage collector """
    if gc.isenabled() == state:
        return
    if state:
        gc.enable()   # 启用垃圾回收
    else:
        gc.disable()  # 禁用垃圾回收


@contextmanager
def gc_state(state):
    """ Context manager to set state of garbage collector to `state`

    Parameters
    ----------
    state : bool
        True for gc enabled, False for disabled

    Examples
    --------
    >>> with gc_state(False):
    ...     assert not gc.isenabled()
    >>> with gc_state(True):
    ...     assert gc.isenabled()
    """
    orig_state = gc.isenabled()
    set_gc_state(state)   # 设置垃圾回收状态
    yield   # 返回上下文管理器的控制权，用于执行嵌套的代码块
    set_gc_state(orig_state)   # 恢复原始的垃圾回收状态


@contextmanager
def assert_deallocated(func, *args, **kwargs):
    """Context manager to check that object is deallocated

    This is useful for checking that an object can be freed directly by
    reference counting, without requiring gc to break reference cycles.
    GC is disabled inside the context manager.

    This check is not available on PyPy.

    Parameters
    ----------
    func : callable
        Callable to create object to check
    \\*args : sequence
        positional arguments to `func` in order to create object to check
    \\*\\*kwargs : dict
        keyword arguments to `func` in order to create object to check

    Examples
    --------
    >>> class C: pass
    >>> with assert_deallocated(C) as c:
    ...     # do something
    ...     del c

    >>> class C:
    ...     def __init__(self):
    ...         self._circular = self # Make circular reference
    >>> with assert_deallocated(C) as c: #doctest: +IGNORE_EXCEPTION_DETAIL
    ...     # do something
    ...     del c
    Traceback (most recent call last):
        ...
    ReferenceError: Remaining reference(s) to object
    """
    if IS_PYPY:
        raise RuntimeError("assert_deallocated is unavailable on PyPy")

    with gc_state(False):   # 在嵌套的代码块中禁用垃圾回收
        obj = func(*args, **kwargs)   # 调用给定函数创建对象
        ref = weakref.ref(obj)   # 创建对对象的弱引用
        yield obj   # 返回对象，允许在 with 语句中使用对象
        del obj   # 删除对象的引用
        if ref() is not None:   # 如果仍然有对象的引用存在
            raise ReferenceError("Remaining reference(s) to object")   # 抛出引用错误异常
```