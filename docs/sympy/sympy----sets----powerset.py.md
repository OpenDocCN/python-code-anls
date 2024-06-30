# `D:\src\scipysrc\sympy\sympy\sets\powerset.py`

```
from sympy.core.decorators import _sympifyit
from sympy.core.parameters import global_parameters
from sympy.core.logic import fuzzy_bool
from sympy.core.singleton import S
from sympy.core.sympify import _sympify

from .sets import Set, FiniteSet, SetKind  # 导入相关的符号和类


class PowerSet(Set):
    r"""A symbolic object representing a power set.

    Parameters
    ==========

    arg : Set
        The set to take power of.

    evaluate : bool
        The flag to control evaluation.

        If the evaluation is disabled for finite sets, it can take
        advantage of using subset test as a membership test.

    Notes
    =====

    Power set `\mathcal{P}(S)` is defined as a set containing all the
    subsets of `S`.

    If the set `S` is a finite set, its power set would have
    `2^{\left| S \right|}` elements, where `\left| S \right|` denotes
    the cardinality of `S`.

    Examples
    ========

    >>> from sympy import PowerSet, S, FiniteSet

    A power set of a finite set:

    >>> PowerSet(FiniteSet(1, 2, 3))
    PowerSet({1, 2, 3})

    A power set of an empty set:

    >>> PowerSet(S.EmptySet)
    PowerSet(EmptySet)
    >>> PowerSet(PowerSet(S.EmptySet))
    PowerSet(PowerSet(EmptySet))

    A power set of an infinite set:

    >>> PowerSet(S.Reals)
    PowerSet(Reals)

    Evaluating the power set of a finite set to its explicit form:

    >>> PowerSet(FiniteSet(1, 2, 3)).rewrite(FiniteSet)
    FiniteSet(EmptySet, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3})

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Power_set

    .. [2] https://en.wikipedia.org/wiki/Axiom_of_power_set
    """
    
    def __new__(cls, arg, evaluate=None):  # 创建一个新的 PowerSet 对象
        if evaluate is None:
            evaluate=global_parameters.evaluate  # 如果没有显式提供评估参数，则使用全局默认值

        arg = _sympify(arg)  # 将参数符号化处理

        if not isinstance(arg, Set):  # 如果参数不是一个集合，则抛出异常
            raise ValueError('{} must be a set.'.format(arg))

        return super().__new__(cls, arg)  # 调用父类的 __new__ 方法来实例化对象

    @property
    def arg(self):  # 返回 PowerSet 对象的第一个参数（即原始集合）
        return self.args[0]

    def _eval_rewrite_as_FiniteSet(self, *args, **kwargs):  # 将 PowerSet 重写为 FiniteSet 的形式进行评估
        arg = self.arg
        if arg.is_FiniteSet:
            return arg.powerset()  # 如果原始集合是有限集，则返回其幂集
        return None

    @_sympifyit('other', NotImplemented)
    def _contains(self, other):  # 判断某个集合是否包含于当前 PowerSet 对象中
        if not isinstance(other, Set):
            return None

        return fuzzy_bool(self.arg.is_superset(other))  # 使用模糊布尔运算判断是否为超集

    def _eval_is_subset(self, other):  # 判断当前 PowerSet 对象是否是另一个 PowerSet 对象的子集
        if isinstance(other, PowerSet):
            return self.arg.is_subset(other.arg)

    def __len__(self):  # 返回当前 PowerSet 的元素数量，即 2 的集合长度次方
        return 2 ** len(self.arg)

    def __iter__(self):  # 迭代生成当前 PowerSet 对象的所有子集
        found = [S.EmptySet]  # 初始化发现的子集列表，包含空集
        yield S.EmptySet  # 首先生成空集

        for x in self.arg:  # 对原始集合中的每个元素进行处理
            temp = []
            x = FiniteSet(x)  # 将元素转换为有限集
            for y in found:  # 遍历已发现的子集列表
                new = x + y  # 生成新的子集
                yield new  # 生成新的子集
                temp.append(new)  # 将新生成的子集添加到临时列表中
            found.extend(temp)  # 将临时列表中的子集扩展到发现的子集列表中

    @property
    def kind(self):  # 返回当前 PowerSet 对象所包含集合的类型
        return SetKind(self.arg.kind)
```