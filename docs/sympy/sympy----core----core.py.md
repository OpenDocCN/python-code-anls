# `D:\src\scipysrc\sympy\sympy\core\core.py`

```
""" The core's core. """
# 导入未来支持的注释语法
from __future__ import annotations

# 定义一个注册表基类
class Registry:
    """
    Base class for registry objects.

    Registries map a name to an object using attribute notation. Registry
    classes behave singletonically: all their instances share the same state,
    which is stored in the class object.

    All subclasses should set `__slots__ = ()`.
    """
    # 设置类的 __slots__ 属性为空元组，以限制实例属性的动态创建
    __slots__ = ()

    # 自定义设置属性的方法，将属性设置到类对象上
    def __setattr__(self, name, obj):
        setattr(self.__class__, name, obj)

    # 自定义删除属性的方法，从类对象上删除属性
    def __delattr__(self, name):
        delattr(self.__class__, name)
```