# `D:\src\scipysrc\sympy\sympy\multipledispatch\core.py`

```
# 引入未来的注解支持
from __future__ import annotations
# 引入类型提示中的 Any 类型
from typing import Any

# 导入用于检查函数和方法的调度器、方法调度器和模糊警告
import inspect
from .dispatcher import Dispatcher, MethodDispatcher, ambiguity_warn

# 全局命名空间字典，用于存储函数的分派实现
global_namespace: dict[str, Any] = {}


def dispatch(*types, namespace=global_namespace, on_ambiguity=ambiguity_warn):
    """根据输入的参数类型分派函数

    支持对所有非关键字参数进行分派。

    根据函数名收集不同实现。忽略命名空间。

    如果存在模糊的类型签名，则在定义函数时会发出警告，建议添加额外的方法以解决歧义。

    Examples
    --------

    >>> from sympy.multipledispatch import dispatch
    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1

    >>> @dispatch(float)
    ... def f(x): # noqa: F811
    ...     return x - 1

    >>> f(3)
    4
    >>> f(3.0)
    2.0

    使用 namespace 关键字参数指定一个独立的命名空间

    >>> my_namespace = dict()
    >>> @dispatch(int, namespace=my_namespace)
    ... def foo(x):
    ...     return x + 1

    在类中分派实例方法

    >>> class MyClass(object):
    ...     @dispatch(list)
    ...     def __init__(self, data):
    ...         self.data = data
    ...     @dispatch(int)
    ...     def __init__(self, datum): # noqa: F811
    ...         self.data = [datum]
    """
    types = tuple(types)

    def _(func):
        name = func.__name__

        # 如果是方法，则获取当前帧的局部变量中的调度器或创建一个新的方法调度器
        if ismethod(func):
            dispatcher = inspect.currentframe().f_back.f_locals.get(
                name,
                MethodDispatcher(name))
        else:
            # 如果函数名不在命名空间中，则创建一个新的调度器
            if name not in namespace:
                namespace[name] = Dispatcher(name)
            dispatcher = namespace[name]

        # 将类型和函数绑定到调度器上
        dispatcher.add(types, func, on_ambiguity=on_ambiguity)
        return dispatcher
    return _


def ismethod(func):
    """检查函数是否为方法

    注意，此函数需要在定义方法但类尚未定义时工作。此时方法看起来像普通函数。
    """
    signature = inspect.signature(func)
    return signature.parameters.get('self', None) is not None
```