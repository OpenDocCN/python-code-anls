# `MetaGPT\metagpt\utils\singleton.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 16:15
@Author  : alexanderwu
@File    : singleton.py
"""
# 导入 abc 模块
import abc

# 定义 Singleton 类，继承自 abc.ABCMeta 和 type
class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    # 存储类实例的字典
    _instances = {}

    # 定义 __call__ 方法，用于创建类的实例
    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        # 如果类不在实例字典中，则创建实例并存储在字典中
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        # 返回类的实例
        return cls._instances[cls]

```