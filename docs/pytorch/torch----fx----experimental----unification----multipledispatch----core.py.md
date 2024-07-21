# `.\pytorch\torch\fx\experimental\unification\multipledispatch\core.py`

```py
# mypy: allow-untyped-defs
# 导入检查模块和系统模块
import inspect
import sys

# 导入本地模块中的Dispatcher和MethodDispatcher类
from .dispatcher import Dispatcher, MethodDispatcher

# 定义全局命名空间字典，类型注释忽略警告
global_namespace = {}  # type: ignore[var-annotated]

# 导出的函数名列表
__all__ = ["dispatch", "ismethod"]

# 定义dispatch装饰器函数，支持根据输入类型进行函数分派
def dispatch(*types, **kwargs):
    """ Dispatch function on the types of the inputs
    Supports dispatch on all non-keyword arguments.
    Collects implementations based on the function name.  Ignores namespaces.
    If ambiguous type signatures occur a warning is raised when the function is
    defined suggesting the additional method to break the ambiguity.

    Example:
        >>> # xdoctest: +SKIP
        >>> @dispatch(int)
        ... def f(x):
        ...     return x + 1
        >>> @dispatch(float)
        ... def f(x):
        ...     return x - 1
        >>> # xdoctest: +SKIP
        >>> f(3)
        4
        >>> f(3.0)
        2.0
        >>> # Specify an isolated namespace with the namespace keyword argument
        >>> my_namespace = {}
        >>> @dispatch(int, namespace=my_namespace)
        ... def foo(x):
        ...     return x + 1
        >>> # Dispatch on instance methods within classes
        >>> class MyClass(object):
        ...     @dispatch(list)
        ...     def __init__(self, data):
        ...         self.data = data
        ...     @dispatch(int)
        ...     def __init__(self, datum):
        ...         self.data = [datum]
        >>> MyClass([1, 2, 3]).data
        [1, 2, 3]
        >>> MyClass(3).data
        [3]
    """
    # 从关键字参数中获取命名空间，如果未指定则使用全局命名空间
    namespace = kwargs.get('namespace', global_namespace)

    # 将types转换为元组
    types = tuple(types)

    # 定义内部函数_df，接受一个函数作为参数
    def _df(func):
        name = func.__name__

        # 检查函数是否是方法
        if ismethod(func):
            # 获取当前函数调用的上下文，尝试从局部变量中获取与函数名相匹配的MethodDispatcher对象
            dispatcher = inspect.currentframe().f_back.f_locals.get(
                name,
                MethodDispatcher(name),
            )
        else:
            # 如果函数名不在命名空间中，则创建一个新的Dispatcher对象
            if name not in namespace:
                namespace[name] = Dispatcher(name)
            dispatcher = namespace[name]

        # 向dispatcher对象中添加类型和函数的映射关系
        dispatcher.add(types, func)
        return dispatcher

    return _df


# 判断函数是否是方法的辅助函数
def ismethod(func):
    """ Is func a method?
    Note that this has to work as the method is defined but before the class is
    defined.  At this stage methods look like functions.
    """
    # 使用inspect模块判断是否有'self'参数来确定是否是方法
    if hasattr(inspect, "signature"):
        signature = inspect.signature(func)
        return signature.parameters.get('self', None) is not None
    else:
        # 兼容Python 2.x的情况，使用inspect模块获取函数的参数规范
        if sys.version_info.major < 3:
            spec = inspect.getargspec(func)  # type: ignore[attr-defined]
        else:
            spec = inspect.getfullargspec(func)  # type: ignore[union-attr, assignment]
        return spec and spec.args and spec.args[0] == 'self'
```