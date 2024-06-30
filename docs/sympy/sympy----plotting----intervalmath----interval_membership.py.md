# `D:\src\scipysrc\sympy\sympy\plotting\intervalmath\interval_membership.py`

```
# 导入模糊逻辑操作函数 fuzzy_and, fuzzy_or, fuzzy_not, fuzzy_xor
from sympy.core.logic import fuzzy_and, fuzzy_or, fuzzy_not, fuzzy_xor

# 定义 intervalMembership 类，表示通过比较间隔对象返回的布尔表达式
class intervalMembership:
    """Represents a boolean expression returned by the comparison of
    the interval object.

    Parameters
    ==========

    (a, b) : (bool, bool)
        The first value determines the comparison as follows:
        - True: If the comparison is True throughout the intervals.
        - False: If the comparison is False throughout the intervals.
        - None: If the comparison is True for some part of the intervals.

        The second value is determined as follows:
        - True: If both the intervals in comparison are valid.
        - False: If at least one of the intervals is False, else
        - None
    """

    # 初始化方法，接受两个参数 a 和 b，并将它们封装到 _wrapped 元组中
    def __init__(self, a, b):
        self._wrapped = (a, b)

    # 实现 __getitem__ 方法，用于获取 _wrapped 中的元素
    def __getitem__(self, i):
        try:
            return self._wrapped[i]
        except IndexError:
            raise IndexError(
                "{} must be a valid indexing for the 2-tuple."
                .format(i))

    # 实现 __len__ 方法，返回 _wrapped 的长度，始终为 2
    def __len__(self):
        return 2

    # 实现 __iter__ 方法，使对象可迭代，返回 _wrapped 的迭代器
    def __iter__(self):
        return iter(self._wrapped)

    # 实现 __str__ 方法，返回对象的字符串表示，格式为 intervalMembership(a, b)
    def __str__(self):
        return "intervalMembership({}, {})".format(*self)

    # __repr__ 方法与 __str__ 方法相同
    __repr__ = __str__

    # 实现 __and__ 方法，支持与操作符 &，对两个 intervalMembership 对象进行模糊与操作
    def __and__(self, other):
        if not isinstance(other, intervalMembership):
            raise ValueError(
                "The comparison is not supported for {}.".format(other))

        a1, b1 = self
        a2, b2 = other
        return intervalMembership(fuzzy_and([a1, a2]), fuzzy_and([b1, b2]))

    # 实现 __or__ 方法，支持或操作符 |，对两个 intervalMembership 对象进行模糊或操作
    def __or__(self, other):
        if not isinstance(other, intervalMembership):
            raise ValueError(
                "The comparison is not supported for {}.".format(other))

        a1, b1 = self
        a2, b2 = other
        return intervalMembership(fuzzy_or([a1, a2]), fuzzy_and([b1, b2]))

    # 实现 __invert__ 方法，支持取反操作符 ~，对 intervalMembership 对象进行模糊非操作
    def __invert__(self):
        a, b = self
        return intervalMembership(fuzzy_not(a), b)

    # 实现 __xor__ 方法，支持异或操作符 ^，对两个 intervalMembership 对象进行模糊异或操作
    def __xor__(self, other):
        if not isinstance(other, intervalMembership):
            raise ValueError(
                "The comparison is not supported for {}.".format(other))

        a1, b1 = self
        a2, b2 = other
        return intervalMembership(fuzzy_xor([a1, a2]), fuzzy_and([b1, b2]))

    # 实现 __eq__ 方法，支持相等比较操作符 ==
    def __eq__(self, other):
        return self._wrapped == other

    # 实现 __ne__ 方法，支持不等比较操作符 !=
    def __ne__(self, other):
        return self._wrapped != other
```