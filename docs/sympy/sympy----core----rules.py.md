# `D:\src\scipysrc\sympy\sympy\core\rules.py`

```
"""
Replacement rules.
"""

# 定义一个Transform类，表示不可变的映射，可以用作通用的转换规则。
class Transform:
    """
    Immutable mapping that can be used as a generic transformation rule.

    Parameters
    ==========

    transform : callable
        Computes the value corresponding to any key.

    filter : callable, optional
        If supplied, specifies which objects are in the mapping.

    Examples
    ========

    >>> from sympy.core.rules import Transform
    >>> from sympy.abc import x

    This Transform will return, as a value, one more than the key:

    >>> add1 = Transform(lambda x: x + 1)
    >>> add1[1]
    2
    >>> add1[x]
    x + 1

    By default, all values are considered to be in the dictionary. If a filter
    is supplied, only the objects for which it returns True are considered as
    being in the dictionary:

    >>> add1_odd = Transform(lambda x: x + 1, lambda x: x%2 == 1)
    >>> 2 in add1_odd
    False
    >>> add1_odd.get(2, 0)
    0
    >>> 3 in add1_odd
    True
    >>> add1_odd[3]
    4
    >>> add1_odd.get(3, 0)
    4
    """

    # 初始化方法，接受一个transform函数和一个可选的filter函数
    def __init__(self, transform, filter=lambda x: True):
        self._transform = transform  # 将transform函数存储在实例变量_transform中
        self._filter = filter        # 将filter函数存储在实例变量_filter中

    # 实现__contains__方法，判断传入的item是否符合filter条件
    def __contains__(self, item):
        return self._filter(item)

    # 实现__getitem__方法，根据key获取对应的值，如果key符合filter条件则使用transform进行转换，否则抛出KeyError异常
    def __getitem__(self, key):
        if self._filter(key):
            return self._transform(key)
        else:
            raise KeyError(key)

    # 定义get方法，根据item获取对应的值，如果item在映射中则返回对应值，否则返回默认值default
    def get(self, item, default=None):
        if item in self:
            return self[item]
        else:
            return default
```