# `.\DB-GPT-src\dbgpt\util\singleton.py`

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The singleton metaclass for ensuring only one instance of a class."""
# 导入必要的模块
import abc
from typing import Any

# 定义一个元类 Singleton，用于确保类的实例只有一个
class Singleton(abc.ABCMeta, type):
    """Singleton metaclass for ensuring only one instance of a class"""

    # 存储类的实例
    _instances = {}

    # 定义类的调用方法
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Call method for the singleton metaclass"""
        # 如果类的实例不存在，则创建一个新实例并存储
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# 定义一个抽象类 AbstractSingleton，使用 Singleton 元类确保只有一个实例
class AbstractSingleton(abc.ABC, metaclass=Singleton):
    """Abstract singleton class for ensuring only one instance of a class"""

    pass
```