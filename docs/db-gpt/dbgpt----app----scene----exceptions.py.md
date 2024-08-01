# `.\DB-GPT-src\dbgpt\app\scene\exceptions.py`

```py
"""Exceptions for Application."""
# 引入日志模块
import logging

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义基础应用程序异常类
class BaseAppException(Exception):
    """Base Exception for App"""

    def __init__(self, message: str, view: str):
        """Base Exception for App"""
        # 调用父类的初始化方法，并传入异常消息
        super().__init__(message)
        # 将异常消息和视图保存为实例属性
        self.message = message
        self.view = view

# 定义应用程序动作异常类，继承自基础应用程序异常类
class AppActionException(BaseAppException):
    """Exception for App Action."""

    def __init__(self, message: str, view: str):
        """Exception for App Action"""
        # 调用父类的初始化方法，并传入异常消息和视图
        super().__init__(message, view)
```