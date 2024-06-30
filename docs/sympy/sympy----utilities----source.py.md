# `D:\src\scipysrc\sympy\sympy\utilities\source.py`

```
"""
This module adds several functions for interactive source code inspection.
"""


# 将字符串形式的类名转换为实际的类对象
def get_class(lookup_view):
    """
    Convert a string version of a class name to the object.

    For example, get_class('sympy.core.Basic') will return
    class Basic located in module sympy.core
    """
    # 如果 lookup_view 是字符串类型
    if isinstance(lookup_view, str):
        # 获取模块名和函数名
        mod_name, func_name = get_mod_func(lookup_view)
        # 如果函数名不为空
        if func_name != '':
            # 动态导入模块，并获取函数对象
            lookup_view = getattr(
                __import__(mod_name, {}, {}, ['*']), func_name)
            # 如果获取的对象不可调用（不是函数或方法）
            if not callable(lookup_view):
                # 抛出属性错误异常
                raise AttributeError(
                    "'%s.%s' is not a callable." % (mod_name, func_name))
    # 返回最终的查找对象
    return lookup_view


# 分解类路径字符串，返回模块路径和类名
def get_mod_func(callback):
    """
    splits the string path to a class into a string path to the module
    and the name of the class.

    Examples
    ========

    >>> from sympy.utilities.source import get_mod_func
    >>> get_mod_func('sympy.core.basic.Basic')
    ('sympy.core.basic', 'Basic')

    """
    # 找到最后一个点的位置
    dot = callback.rfind('.')
    # 如果找不到点，则直接返回整个字符串作为模块名，类名为空字符串
    if dot == -1:
        return callback, ''
    # 否则，返回点之前的部分作为模块名，点之后的部分作为类名
    return callback[:dot], callback[dot + 1:]
```