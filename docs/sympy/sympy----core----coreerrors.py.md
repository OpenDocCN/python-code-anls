# `D:\src\scipysrc\sympy\sympy\core\coreerrors.py`

```
"""Definitions of common exceptions for :mod:`sympy.core` module. """

# 引入 Callable 类型提示，用于指定回调函数的类型
from typing import Callable


# 定义一个基础的核心相关异常类
class BaseCoreError(Exception):
    """Base class for core related exceptions. """


# 定义一个特定的异常类，表示表达式不具有交换性质
class NonCommutativeExpression(BaseCoreError):
    """Raised when expression didn't have commutative property. """


# 定义一个延迟异常消息的包装类
class LazyExceptionMessage:
    """Wrapper class that lets you specify an expensive to compute
    error message that is only evaluated if the error is rendered."""
    
    # callback 属性，用于存储一个返回字符串的回调函数
    callback: Callable[[], str]

    # 初始化方法，接收一个回调函数作为参数
    def __init__(self, callback: Callable[[], str]):
        self.callback = callback

    # 字符串表示方法，返回回调函数的结果
    def __str__(self):
        return self.callback()
```