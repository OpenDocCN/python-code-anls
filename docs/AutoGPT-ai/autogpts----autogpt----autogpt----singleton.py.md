# `.\AutoGPT\autogpts\autogpt\autogpt\singleton.py`

```py
"""The singleton metaclass for ensuring only one instance of a class."""
# 导入 abc 模块，用于定义抽象基类
import abc

# 定义 Singleton 类，继承自 ABCMeta 和 type 类
class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    # 存储类实例的字典
    _instances = {}

    # 定义 __call__ 方法，用于创建类的实例
    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        # 如果类不在实例字典中，则创建一个新实例并存储在字典中
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        # 返回类的实例
        return cls._instances[cls]
```