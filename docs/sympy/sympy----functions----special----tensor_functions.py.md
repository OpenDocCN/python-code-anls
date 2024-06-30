# `D:\src\scipysrc\sympy\sympy\functions\special\tensor_functions.py`

```
# 从 math 模块导入 prod 函数，用于计算序列的乘积
from math import prod

# 从 sympy.core 模块导入 S 和 Integer 类
from sympy.core import S, Integer

# 从 sympy.core.function 模块导入 Function 类
from sympy.core.function import Function

# 从 sympy.core.logic 模块导入 fuzzy_not 函数，用于模糊逻辑中的否定操作
from sympy.core.logic import fuzzy_not

# 从 sympy.core.relational 模块导入 Ne 类，用于创建不等式表达式
from sympy.core.relational import Ne

# 从 sympy.core.sorting 模块导入 default_sort_key 函数，用于排序操作
from sympy.core.sorting import default_sort_key

# 从 sympy.external.gmpy 模块导入 SYMPY_INTS，这是 gmpy 库的整数类型
from sympy.external.gmpy import SYMPY_INTS

# 从 sympy.functions.combinatorial.factorials 模块导入 factorial 函数，用于计算阶乘
from sympy.functions.combinatorial.factorials import factorial

# 从 sympy.functions.elementary.piecewise 模块导入 Piecewise 类，用于创建分段函数
from sympy.functions.elementary.piecewise import Piecewise

# 从 sympy.utilities.iterables 模块导入 has_dups 函数，用于检查可迭代对象中是否有重复元素
from sympy.utilities.iterables import has_dups


###############################################################################
###################### Kronecker Delta, Levi-Civita etc. ######################
###############################################################################


def Eijk(*args, **kwargs):
    """
    Represent the Levi-Civita symbol.

    This is a compatibility wrapper to ``LeviCivita()``.

    See Also
    ========

    LeviCivita

    """
    # 返回 Levi-Civita 符号的值，通过调用 LeviCivita 类实现
    return LeviCivita(*args, **kwargs)


def eval_levicivita(*args):
    """Evaluate Levi-Civita symbol."""
    # 获取参数个数
    n = len(args)
    # 计算 Levi-Civita 符号的值
    return prod(
        prod(args[j] - args[i] for j in range(i + 1, n))
        / factorial(i) for i in range(n))
    # converting factorial(i) to int is slightly faster


class LeviCivita(Function):
    """
    Represent the Levi-Civita symbol.

    Explanation
    ===========

    For even permutations of indices it returns 1, for odd permutations -1, and
    for everything else (a repeated index) it returns 0.

    Thus it represents an alternating pseudotensor.

    Examples
    ========

    >>> from sympy import LeviCivita
    >>> from sympy.abc import i, j, k
    >>> LeviCivita(1, 2, 3)
    1
    >>> LeviCivita(1, 3, 2)
    -1
    >>> LeviCivita(1, 2, 2)
    0
    >>> LeviCivita(i, j, k)
    LeviCivita(i, j, k)
    >>> LeviCivita(i, j, i)
    0

    See Also
    ========

    Eijk

    """

    # 设置类属性，表示 Levi-Civita 符号为整数类型
    is_integer = True

    @classmethod
    def eval(cls, *args):
        # 如果所有参数均为整数类型或符号类型
        if all(isinstance(a, (SYMPY_INTS, Integer)) for a in args):
            # 调用 eval_levicivita 函数计算 Levi-Civita 符号的值
            return eval_levicivita(*args)
        # 如果存在重复的索引
        if has_dups(args):
            return S.Zero

    def doit(self, **hints):
        # 调用 eval_levicivita 函数计算 Levi-Civita 符号的值
        return eval_levicivita(*self.args)


class KroneckerDelta(Function):
    """
    The discrete, or Kronecker, delta function.

    Explanation
    ===========

    A function that takes in two integers $i$ and $j$. It returns $0$ if $i$
    and $j$ are not equal, or it returns $1$ if $i$ and $j$ are equal.

    Examples
    ========

    An example with integer indices:

        >>> from sympy import KroneckerDelta
        >>> KroneckerDelta(1, 2)
        0
        >>> KroneckerDelta(3, 3)
        1

    Symbolic indices:

        >>> from sympy.abc import i, j, k
        >>> KroneckerDelta(i, j)
        KroneckerDelta(i, j)
        >>> KroneckerDelta(i, i)
        1
        >>> KroneckerDelta(i, i + 1)
        0
        >>> KroneckerDelta(i, i + 1 + k)
        KroneckerDelta(i, i + k + 1)

    Parameters
    ==========

    i : Number, Symbol
        The first index of the delta function.

    """
    j : Number, Symbol
        The second index of the delta function.

    See Also
    ========

    eval
    DiracDelta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kronecker_delta

    """
    # 初始化一个布尔值，表示参数是否为整数
    is_integer = True

    @classmethod
    def eval(cls, i, j, delta_range=None):
        """
        Evaluates the discrete delta function.

        Examples
        ========

        >>> from sympy import KroneckerDelta
        >>> from sympy.abc import i, j, k

        >>> KroneckerDelta(i, j)
        KroneckerDelta(i, j)
        >>> KroneckerDelta(i, i)
        1
        >>> KroneckerDelta(i, i + 1)
        0
        >>> KroneckerDelta(i, i + 1 + k)
        KroneckerDelta(i, i + k + 1)

        # indirect doctest

        """
        # 如果提供了 delta_range 参数，则检查 i 和 j 是否在指定范围内
        if delta_range is not None:
            dinf, dsup = delta_range
            if (dinf - i > 0) == True:
                return S.Zero
            if (dinf - j > 0) == True:
                return S.Zero
            if (dsup - i < 0) == True:
                return S.Zero
            if (dsup - j < 0) == True:
                return S.Zero

        # 计算 i 和 j 的差异
        diff = i - j
        # 如果 i 和 j 相等，则返回 1
        if diff.is_zero:
            return S.One
        # 如果 i 和 j 不相等，则返回 0
        elif fuzzy_not(diff.is_zero):
            return S.Zero

        # 根据符号假设检查是否为零
        if i.assumptions0.get("below_fermi") and \
                j.assumptions0.get("above_fermi"):
            return S.Zero
        if j.assumptions0.get("below_fermi") and \
                i.assumptions0.get("above_fermi"):
            return S.Zero
        
        # 使 KroneckerDelta 规范化
        # 检查输入是否按顺序，如果不是，则返回正确顺序的 KroneckerDelta
        if default_sort_key(j) < default_sort_key(i):
            if delta_range:
                return cls(j, i, delta_range)
            else:
                return cls(j, i)

    @property
    def delta_range(self):
        # 如果参数数量大于 2，则返回第三个参数作为范围
        if len(self.args) > 2:
            return self.args[2]

    def _eval_power(self, expt):
        # 如果指数是正数，返回自身
        if expt.is_positive:
            return self
        # 如果指数是负数且不是 -1，则返回 1/self
        if expt.is_negative and expt is not S.NegativeOne:
            return 1/self

    @property
    def is_above_fermi(self):
        """
        True if Delta can be non-zero above fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, a).is_above_fermi
        True
        >>> KroneckerDelta(p, i).is_above_fermi
        False
        >>> KroneckerDelta(p, q).is_above_fermi
        True

        See Also
        ========

        is_below_fermi, is_only_below_fermi, is_only_above_fermi

        """
        # 如果第一个参数或第二个参数以下 fermi 能级，则返回 False
        if self.args[0].assumptions0.get("below_fermi"):
            return False
        if self.args[1].assumptions0.get("below_fermi"):
            return False
        # 否则返回 True
        return True

    @property
    @property
    def is_above_fermi(self):
        """
        True if Delta is restricted to above fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, a).is_above_fermi
        True
        >>> KroneckerDelta(p, q).is_above_fermi
        False
        >>> KroneckerDelta(p, i).is_above_fermi
        False

        See Also
        ========

        is_below_fermi, is_only_above_fermi, is_only_below_fermi

        """
        return (self.args[0].assumptions0.get("above_fermi")
                or
                self.args[1].assumptions0.get("above_fermi")
                ) or False
    def indices_contain_equal_information(self):
        """
        Returns True if indices are either both above or below fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, q).indices_contain_equal_information
        True
        >>> KroneckerDelta(p, q+1).indices_contain_equal_information
        True
        >>> KroneckerDelta(i, p).indices_contain_equal_information
        False

        """
        # 检查第一个和第二个参数是否都在费米面以下，并返回 True
        if (self.args[0].assumptions0.get("below_fermi") and
                self.args[1].assumptions0.get("below_fermi")):
            return True
        # 检查第一个和第二个参数是否都在费米面以上，并返回 True
        if (self.args[0].assumptions0.get("above_fermi")
                and self.args[1].assumptions0.get("above_fermi")):
            return True

        # 如果两个索引一个在费米面以下一个在费米面以上，则返回 self.is_below_fermi and self.is_above_fermi
        return self.is_below_fermi and self.is_above_fermi

    @property
    def preferred_index(self):
        """
        Returns the index which is preferred to keep in the final expression.

        Explanation
        ===========

        The preferred index is the index with more information regarding fermi
        level. If indices contain the same information, 'a' is preferred before
        'b'.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> j = Symbol('j', below_fermi=True)
        >>> p = Symbol('p')
        >>> KroneckerDelta(p, i).preferred_index
        i
        >>> KroneckerDelta(p, a).preferred_index
        a
        >>> KroneckerDelta(i, j).preferred_index
        i

        See Also
        ========

        killable_index

        """
        # 返回具有更多关于费米面信息的索引作为首选索引
        if self._get_preferred_index():
            return self.args[1]
        else:
            return self.args[0]
    # 返回在最终表达式中首选替换的索引。
    def killable_index(self):
        """
        Returns the index which is preferred to substitute in the final
        expression.
        
        Explanation
        ===========
        
        The index to substitute is the index with less information regarding
        fermi level. If indices contain the same information, 'a' is preferred
        before 'b'.
        
        Examples
        ========
        
        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> j = Symbol('j', below_fermi=True)
        >>> p = Symbol('p')
        >>> KroneckerDelta(p, i).killable_index
        p
        >>> KroneckerDelta(p, a).killable_index
        p
        >>> KroneckerDelta(i, j).killable_index
        j
        
        See Also
        ========
        
        preferred_index
        
        """
        # 如果存在首选索引，则返回第一个参数
        if self._get_preferred_index():
            return self.args[0]
        else:
            return self.args[1]

    def _get_preferred_index(self):
        """
        Returns the index which is preferred to keep in the final expression.
        
        The preferred index is the index with more information regarding fermi
        level. If indices contain the same information, index 0 is returned.
        
        """
        # 如果索引不在 fermi 上方，则根据下方 fermi 级别返回首选索引
        if not self.is_above_fermi:
            if self.args[0].assumptions0.get("below_fermi"):
                return 0
            else:
                return 1
        # 如果索引不在 fermi 下方，则根据上方 fermi 级别返回首选索引
        elif not self.is_below_fermi:
            if self.args[0].assumptions0.get("above_fermi"):
                return 0
            else:
                return 1
        else:
            return 0

    @property
    def indices(self):
        # 返回参数中的前两个索引
        return self.args[0:2]

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        # 将参数 i 和 j 用 Piecewise 形式重写
        i, j = args
        return Piecewise((0, Ne(i, j)), (1, True))
```