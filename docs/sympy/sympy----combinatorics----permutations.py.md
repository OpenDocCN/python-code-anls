# `D:\src\scipysrc\sympy\sympy\combinatorics\permutations.py`

```
# 导入random模块，用于生成随机数
import random
# 导入defaultdict类，用于创建默认字典
from collections import defaultdict
# 导入Iterable抽象基类，用于判断对象是否可迭代
from collections.abc import Iterable
# 导入reduce函数，用于对可迭代对象进行累积操作
from functools import reduce

# 导入global_parameters全局参数
from sympy.core.parameters import global_parameters
# 导入Atom类，表示符号表达式树的原子元素
from sympy.core.basic import Atom
# 导入Expr类，表示符号表达式树的基类
from sympy.core.expr import Expr
# 导入int_valued函数，用于检查数值是否为整数
from sympy.core.numbers import int_valued
# 导入Integer类，表示符号表达式树中的整数
from sympy.core.numbers import Integer
# 导入_sympify函数，用于将对象转换为符号表达式
from sympy.core.sympify import _sympify
# 导入zeros函数，用于创建零矩阵
from sympy.matrices import zeros
# 导入lcm函数，用于计算多项式的最小公倍数
from sympy.polys.polytools import lcm
# 导入srepr函数，用于生成对象的可打印表示
from sympy.printing.repr import srepr
# 导入flatten函数，用于将嵌套序列展平
# 导入has_variety函数，用于检查序列中是否存在多样性
# 导入minlex函数，用于在有限集合上对序列进行最小词典序排序
# 导入has_dups函数，用于检查序列是否包含重复元素
# 导入runs函数，用于查找序列中的有序子序列
# 导入is_sequence函数，用于检查对象是否为序列
from sympy.utilities.iterables import (flatten, has_variety, minlex,
    has_dups, runs, is_sequence)
# 导入as_int函数，用于将输入转换为整数
from sympy.utilities.misc import as_int
# 导入ifac函数，用于计算阶乘的倒数
from mpmath.libmp.libintmath import ifac
# 导入dispatch函数，用于实现多分派方法
from sympy.multipledispatch import dispatch


def _af_rmul(a, b):
    """
    Return the product b*a; input and output are array forms. The ith value
    is a[b[i]].

    Examples
    ========

    >>> from sympy.combinatorics.permutations import _af_rmul, Permutation

    >>> a, b = [1, 0, 2], [0, 2, 1]
    >>> _af_rmul(a, b)
    [1, 2, 0]
    >>> [a[b[i]] for i in range(3)]
    [1, 2, 0]

    This handles the operands in reverse order compared to the ``*`` operator:

    >>> a = Permutation(a)
    >>> b = Permutation(b)
    >>> list(a*b)
    [2, 0, 1]
    >>> [b(a(i)) for i in range(3)]
    [2, 0, 1]

    See Also
    ========

    rmul, _af_rmuln
    """
    return [a[i] for i in b]


def _af_rmuln(*abc):
    """
    Given [a, b, c, ...] return the product of ...*c*b*a using array forms.
    The ith value is a[b[c[i]]].

    Examples
    ========

    >>> from sympy.combinatorics.permutations import _af_rmul, Permutation

    >>> a, b = [1, 0, 2], [0, 2, 1]
    >>> _af_rmul(a, b)
    [1, 2, 0]
    >>> [a[b[i]] for i in range(3)]
    [1, 2, 0]

    This handles the operands in reverse order compared to the ``*`` operator:

    >>> a = Permutation(a); b = Permutation(b)
    >>> list(a*b)
    [2, 0, 1]
    >>> [b(a(i)) for i in range(3)]
    [2, 0, 1]

    See Also
    ========

    rmul, _af_rmul
    """
    a = abc
    m = len(a)
    if m == 3:
        p0, p1, p2 = a
        return [p0[p1[i]] for i in p2]
    if m == 4:
        p0, p1, p2, p3 = a
        return [p0[p1[p2[i]]] for i in p3]
    if m == 5:
        p0, p1, p2, p3, p4 = a
        return [p0[p1[p2[p3[i]]]] for i in p4]
    if m == 6:
        p0, p1, p2, p3, p4, p5 = a
        return [p0[p1[p2[p3[p4[i]]]]] for i in p5]
    if m == 7:
        p0, p1, p2, p3, p4, p5, p6 = a
        return [p0[p1[p2[p3[p4[p5[i]]]]]] for i in p6]
    if m == 8:
        p0, p1, p2, p3, p4, p5, p6, p7 = a
        return [p0[p1[p2[p3[p4[p5[p6[i]]]]]]] for i in p7]
    if m == 1:
        return a[0][:]
    if m == 2:
        a, b = a
        return [a[i] for i in b]
    if m == 0:
        raise ValueError("String must not be empty")
    p0 = _af_rmuln(*a[:m//2])
    p1 = _af_rmuln(*a[m//2:])
    return [p0[i] for i in p1]


def _af_parity(pi):
    """
    Computes the parity of a permutation in array form.

    Explanation
    ===========
    
    This function computes the parity (evenness or oddness) of a given permutation 
    represented in array form.

    """
    # 省略具体的实现细节，因为这部分代码不在注释的范围内
    # 计算排列的奇偶性，反映了排列中逆序对的奇偶性，即满足 x > y 但是 pi[x] < pi[y] 的 x 和 y 对的数量。
    
    n = len(pi)  # 获取排列 pi 的长度
    a = [0] * n   # 创建长度为 n 的列表 a，初始化为全0
    c = 0         # 初始化计数器 c 为 0
    
    for j in range(n):  # 遍历排列 pi 的索引范围
        if a[j] == 0:  # 如果 a[j] 等于 0（即未被标记）
            c += 1     # 计数器 c 加1
            a[j] = 1   # 将 a[j] 标记为 1
            i = j      # 设定 i 为 j
            while pi[i] != j:  # 当 pi[i] 不等于 j 时，执行以下循环
                i = pi[i]      # 更新 i 为 pi[i]
                a[i] = 1       # 标记 a[i] 为 1（表示已访问过）
    
    # 返回排列的奇偶性，即 (n - c) % 2
    return (n - c) % 2
def _af_invert(a):
    """
    Finds the inverse, ~A, of a permutation, A, given in array form.

    Examples
    ========

    >>> from sympy.combinatorics.permutations import _af_invert, _af_rmul
    >>> A = [1, 2, 0, 3]
    >>> _af_invert(A)
    [2, 0, 1, 3]
    >>> _af_rmul(_, A)
    [0, 1, 2, 3]

    See Also
    ========

    Permutation, __invert__
    """
    # 创建一个长度与数组a相同的列表，初始值全部为0
    inv_form = [0] * len(a)
    # 遍历数组a，将索引i与元素ai的对应关系反转存储到inv_form中
    for i, ai in enumerate(a):
        inv_form[ai] = i
    return inv_form


def _af_pow(a, n):
    """
    Routine for finding powers of a permutation.

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.permutations import _af_pow
    >>> p = Permutation([2, 0, 3, 1])
    >>> p.order()
    4
    >>> _af_pow(p._array_form, 4)
    [0, 1, 2, 3]
    """
    # 若n为0，则返回数组a的长度范围内的列表
    if n == 0:
        return list(range(len(a)))
    # 若n为负数，则返回数组a的逆序列的n次幂
    if n < 0:
        return _af_pow(_af_invert(a), -n)
    # 若n为1，则返回数组a的复制
    if n == 1:
        return a[:]
    elif n == 2:
        # 对数组a进行二次幂操作
        b = [a[i] for i in a]
    elif n == 3:
        # 对数组a进行三次幂操作
        b = [a[a[i]] for i in a]
    elif n == 4:
        # 对数组a进行四次幂操作
        b = [a[a[a[i]]] for i in a]
    else:
        # 使用二进制乘法计算更高次幂
        b = list(range(len(a)))
        while 1:
            if n & 1:
                b = [b[i] for i in a]
                n -= 1
                if not n:
                    break
            if n % 4 == 0:
                a = [a[a[a[i]]] for i in a]
                n = n // 4
            elif n % 2 == 0:
                a = [a[i] for i in a]
                n = n // 2
    return b


def _af_commutes_with(a, b):
    """
    Checks if the two permutations with array forms
    given by ``a`` and ``b`` commute.

    Examples
    ========

    >>> from sympy.combinatorics.permutations import _af_commutes_with
    >>> _af_commutes_with([1, 2, 0], [0, 2, 1])
    False

    See Also
    ========

    Permutation, commutes_with
    """
    # 检查给定的两个数组形式的排列a和b是否可交换
    return not any(a[b[i]] != b[a[i]] for i in range(len(a) - 1))


class Cycle(dict):
    """
    Wrapper around dict which provides the functionality of a disjoint cycle.

    Explanation
    ===========

    A cycle shows the rule to use to move subsets of elements to obtain
    a permutation. The Cycle class is more flexible than Permutation in
    that 1) all elements need not be present in order to investigate how
    multiple cycles act in sequence and 2) it can contain singletons:

    >>> from sympy.combinatorics.permutations import Perm, Cycle

    A Cycle will automatically parse a cycle given as a tuple on the rhs:

    >>> Cycle(1, 2)(2, 3)
    (1 3 2)

    The identity cycle, Cycle(), can be used to start a product:

    >>> Cycle()(1, 2)(2, 3)
    (1 3 2)

    The array form of a Cycle can be obtained by calling the list
    method (or passing it to the list function) and all elements from
    0 will be shown:

    >>> a = Cycle(1, 2)
    >>> a.list()
    [0, 2, 1]
    >>> list(a)
    [0, 2, 1]

    If a larger (or smaller) range is desired use the list method and
    """
    def __missing__(self, arg):
        """Enter arg into dictionary and return arg."""
        # 当访问字典中不存在的键时，将该键插入字典并返回该键的值
        return as_int(arg)

    def __iter__(self):
        """Iterate over the Cycle in array form."""
        # 使用生成器迭代循环对象，产生其数组形式的元素
        yield from self.list()

    def __call__(self, *other):
        """Compose cycles from right to left and return the result.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2)(2, 3)
        (1 3 2)

        An instance of a Cycle will automatically parse list-like
        objects and Permutations that are on the right. It is more
        flexible than the Permutation in that all elements need not
        be present:

        >>> a = Cycle(1, 2)
        >>> a(2, 3)
        (1 3 2)
        >>> a(2, 3)(4, 5)
        (1 3 2)(4 5)

        """
        # 返回从右到左处理的循环组合结果
        rv = Cycle(*other)
        # 遍历当前循环对象的键列表，使用新循环对象对应的值替换原循环对象的值
        for k, v in zip(list(self.keys()), [rv[self[k]] for k in self.keys()]):
            rv[k] = v
        return rv
    def list(self, size=None):
        """
        Return the cycles as an explicit list starting from 0 up
        to the greater of the largest value in the cycles and size.

        Truncation of trailing unmoved items will occur when size
        is less than the maximum element in the cycle; if this is
        desired, setting ``size=-1`` will guarantee such trimming.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> p = Cycle(2, 3)(4, 5)
        >>> p.list()
        [0, 1, 3, 2, 5, 4]
        >>> p.list(10)
        [0, 1, 3, 2, 5, 4, 6, 7, 8, 9]

        Passing a length too small will trim trailing, unchanged elements
        in the permutation:

        >>> Cycle(2, 4)(1, 2, 4).list(-1)
        [0, 2, 1]
        """
        # 如果循环为空且未指定大小，则抛出异常
        if not self and size is None:
            raise ValueError('must give size for empty Cycle')
        # 如果指定了大小，则找到循环中最大的值并确保返回列表长度不小于此值
        if size is not None:
            big = max([i for i in self.keys() if self[i] != i] + [0])
            size = max(size, big + 1)
        else:
            # 否则使用循环对象的大小作为返回列表的长度
            size = self.size
        # 返回从 0 到 size-1 的循环列表
        return [self[i] for i in range(size)]

    def __repr__(self):
        """
        We want it to print as a Cycle, not as a dict.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2)
        (1 2)
        >>> print(_)
        (1 2)
        >>> list(Cycle(1, 2).items())
        [(1, 2), (2, 1)]
        """
        # 如果循环为空，则返回字符串 'Cycle()'
        if not self:
            return 'Cycle()'
        # 将循环转换为置换并获取循环形式
        cycles = Permutation(self).cyclic_form
        # 将循环形式转换为字符串表示
        s = ''.join(str(tuple(c)) for c in cycles)
        # 找到循环中的最大值
        big = self.size - 1
        # 如果最大值不在任何循环中，则在结果字符串中添加括号
        if not any(i == big for c in cycles for i in c):
            s += '(%s)' % big
        # 返回格式化后的字符串
        return 'Cycle%s' % s

    def __str__(self):
        """
        We want it to be printed in a Cycle notation with no
        comma in-between.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2)
        (1 2)
        >>> Cycle(1, 2, 4)(5, 6)
        (1 2 4)(5 6)
        """
        # 如果循环为空，则返回字符串 '()'
        if not self:
            return '()'
        # 将循环转换为置换并获取循环形式
        cycles = Permutation(self).cyclic_form
        # 将循环形式转换为字符串表示
        s = ''.join(str(tuple(c)) for c in cycles)
        # 找到循环中的最大值
        big = self.size - 1
        # 如果最大值不在任何循环中，则在结果字符串中添加括号
        if not any(i == big for c in cycles for i in c):
            s += '(%s)' % big
        # 移除字符串中的逗号并返回结果
        s = s.replace(',', '')
        return s
    # 初始化 Cycle 类的实例，接受任意数量的参数作为循环的值
    def __init__(self, *args):
        """Load up a Cycle instance with the values for the cycle.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2, 6)
        (1 2 6)
        """

        # 如果没有参数传入，则直接返回
        if not args:
            return

        # 如果只有一个参数，并且是 Permutation 类型的实例
        if len(args) == 1:
            if isinstance(args[0], Permutation):
                # 遍历参数 args[0] 的循环形式，更新当前 Cycle 对象
                for c in args[0].cyclic_form:
                    self.update(self(*c))
                return
            # 如果参数是 Cycle 类型的实例
            elif isinstance(args[0], Cycle):
                # 将参数 args[0] 的键值对更新到当前 Cycle 对象
                for k, v in args[0].items():
                    self[k] = v
                return
        
        # 将参数 args 转换为整数列表
        args = [as_int(a) for a in args]
        
        # 如果 args 中存在负数，则抛出 ValueError 异常
        if any(i < 0 for i in args):
            raise ValueError('negative integers are not allowed in a cycle.')
        
        # 如果 args 中有重复的元素，则抛出 ValueError 异常
        if has_dups(args):
            raise ValueError('All elements must be unique in a cycle.')
        
        # 根据 args 中的元素创建循环结构
        for i in range(-len(args), 0):
            self[args[i]] = args[i + 1]

    @property
    # 返回 Cycle 实例中元素的最大值加一，即循环的大小
    def size(self):
        if not self:
            return 0
        return max(self.keys()) + 1

    # 返回当前 Cycle 实例的副本
    def copy(self):
        return Cycle(self)
# 定义一个 Permutation 类，继承自 Atom 类
class Permutation(Atom):
    r"""
    表示一个排列，也称为 'arrangement number' 或 'ordering'，
    将有序列表的元素重新排列为与其自身的一对一映射。给定排列的表示方式
    是指出重新排列后元素的位置 [2]_。例如，如果开始时的元素是 ``[x, y, a, b]``
    （按照这个顺序），并且它们被重新排列为 ``[x, y, b, a]``，那么排列就是
    ``[0, 1, 3, 2]``。注意（在 SymPy 中）第一个元素始终被称为 0，排列使用
    元素在原始顺序中的索引，而不是元素本身的 ``(a, b, ...)``。

    >>> from sympy.combinatorics import Permutation
    >>> from sympy import init_printing
    >>> init_printing(perm_cyclic=False, pretty_print=False)

    排列表示法
    =====================

    排列通常用不相交循环或数组形式表示。

    数组表示和两行形式
    ------------------------------------

    在两行形式中，元素及其最终位置被显示为一个具有两行的矩阵：

    [0    1    2     ... n-1]
    [p(0) p(1) p(2)  ... p(n-1)]

    第一行始终是 ``range(n)``, 其中 n 是 p 的大小，因此仅用第二行表示排列足以，
    这称为排列的 "数组形式"。这在 Permutation 类的参数中用方括号表示：

    >>> p = Permutation([0, 2, 1]); p
    Permutation([0, 2, 1])

    对于范围为 p.size 的 i，排列将 i 映射到 i^p

    >>> [i^p for i in range(p.size)]
    [0, 2, 1]

    两个排列 p*q 的复合意味着先应用 p，然后是 q，因此 i^(p*q) = (i^p)^q，
    根据 Python 的优先规则是 i^p^q：

    >>> q = Permutation([2, 1, 0])
    >>> [i^p^q for i in range(3)]
    [2, 0, 1]
    >>> [i^(p*q) for i in range(3)]
    [2, 0, 1]

    也可以使用 p(i) = i^p 的表示法，但是复合规则是 (p*q)(i) = q(p(i))，
    而不是 p(q(i))：

    >>> [(p*q)(i) for i in range(p.size)]
    [2, 0, 1]
    >>> [q(p(i)) for i in range(p.size)]
    [2, 0, 1]
    >>> [p(q(i)) for i in range(p.size)]
    [1, 2, 0]

    不相交循环表示法
    -----------------------

    在不相交循环表示法中，只有移动的元素被指示。

    例如，[1, 3, 2, 0] 可以表示为 (0, 1, 3)(2)。
    这可以从给定排列的两行形式理解。
    在两行形式中，
    [0    1    2   3]
    [1    3    2   0]

    第 0 位置的元素是 1，因此 0 -> 1。第 1 位置的元素是 3，因此 1 -> 3。
    第 3 位置的元素再次是 0，因此 3 -> 0。因此，0 -> 1 -> 3 -> 0，而 2 -> 2。
    因此，这可以表示为两个循环：(0, 1, 3)(2)。
    在通常的表示法中，单个循环不会显式写出，因为它们可以
    inferred implicitly.

    Only the relative ordering of elements in a cycle matter:

    >>> Permutation(1,2,3) == Permutation(2,3,1) == Permutation(3,1,2)
    True

    The disjoint cycle notation is convenient when representing
    permutations that have several cycles in them:

    >>> Permutation(1, 2)(3, 5) == Permutation([[1, 2], [3, 5]])
    True

    It also provides some economy in entry when computing products of
    permutations that are written in disjoint cycle notation:

    >>> Permutation(1, 2)(1, 3)(2, 3)
    Permutation([0, 3, 2, 1])
    >>> _ == Permutation([[1, 2]])*Permutation([[1, 3]])*Permutation([[2, 3]])
    True

        Caution: when the cycles have common elements between them then the order
        in which the permutations are applied matters. This module applies
        the permutations from *left to right*.

        >>> Permutation(1, 2)(2, 3) == Permutation([(1, 2), (2, 3)])
        True
        >>> Permutation(1, 2)(2, 3).list()
        [0, 3, 1, 2]
        # 对于 (1,2) 先被计算再是 (2,3)。

        In the above case, (1,2) is computed before (2,3).
        As 0 -> 0, 0 -> 0, element in position 0 is 0.
        As 1 -> 2, 2 -> 3, element in position 1 is 3.
        As 2 -> 1, 1 -> 1, element in position 2 is 1.
        As 3 -> 3, 3 -> 2, element in position 3 is 2.

        If the first and second elements had been
        swapped first, followed by the swapping of the second
        and third, the result would have been [0, 2, 3, 1].
        If, you want to apply the cycles in the conventional
        right to left order, call the function with arguments in reverse order
        as demonstrated below:

        >>> Permutation([(1, 2), (2, 3)][::-1]).list()
        [0, 2, 3, 1]

    Entering a singleton in a permutation is a way to indicate the size of the
    permutation. The ``size`` keyword can also be used.

    Array-form entry:

    >>> Permutation([[1, 2], [9]])
    Permutation([0, 2, 1], size=10)
    >>> Permutation([[1, 2]], size=10)
    Permutation([0, 2, 1], size=10)

    Cyclic-form entry:

    >>> Permutation(1, 2, size=10)
    Permutation([0, 2, 1], size=10)
    >>> Permutation(9)(1, 2)
    Permutation([0, 2, 1], size=10)

    Caution: no singleton containing an element larger than the largest
    in any previous cycle can be entered. This is an important difference
    in how Permutation and Cycle handle the ``__call__`` syntax. A singleton
    argument at the start of a Permutation performs instantiation of the
    Permutation and is permitted:

    >>> Permutation(5)
    Permutation([], size=6)

    A singleton entered after instantiation is a call to the permutation
    -- a function call -- and if the argument is out of range it will
    trigger an error. For this reason, it is better to start the cycle
    with the singleton:

    The following fails because there is no element 3:

    >>> Permutation(1, 2)(3)
    Traceback (most recent call last):
    ...
    IndexError: list index out of range
    # 这部分代码段似乎是文档或者示例，没有实际的程序逻辑。以下是关于排列操作的不同示例和说明。
    
    # 1. 示例：创建排列并进行组合
    # 例如，创建一个大小为3的排列，然后对其进行组合，返回新的排列
    >>> Permutation(3)(1, 2)
    Permutation([0, 2, 1, 3])
    
    # 2. 示例：比较排列的相等性
    # 数组形式必须相同才能判断排列是否相等
    >>> Permutation([1, 0, 2, 3]) == Permutation([1, 0])
    False
    
    # 3. 示例：单位排列
    # 单位排列中没有元素错位，可以通过不同方式创建大小为4的单位排列，并验证它们是否相等
    >>> I = Permutation([0, 1, 2, 3])
    >>> all(p == I for p in [
    ... Permutation(3),
    ... Permutation(range(4)),
    ... Permutation([], size=4),
    ... Permutation(size=4)])
    True
    
    # 4. 注意事项：在方括号内部输入范围可能会导致误解为循环表示法，而不是创建排列
    >>> I == Permutation([range(4)])
    False
    
    # 5. 排列打印
    # 关于排列打印有几点需要注意
    # 5.1 如果倾向于数组形式或循环形式的其中一种，可以使用 init_printing 设置 perm_cyclic 标志
    # 例如，初始化打印为循环形式
    >>> from sympy import init_printing
    >>> p = Permutation(1, 2)(4, 5)(3, 4)
    >>> p
    Permutation([0, 2, 1, 4, 5, 3])
    
    # 5.2 无论设置如何，可以获取数组形式和循环形式的排列，并用它们作为 Permutation 的参数
    >>> p.array_form
    [0, 2, 1, 4, 5, 3]
    >>> p.cyclic_form
    [[1, 2], [3, 4, 5]]
    >>> Permutation(_) == p
    True
    
    # 5.3 打印经济，尽可能少地打印信息，同时保留排列的大小信息
    # 例如，设置打印为数组形式
    >>> init_printing(perm_cyclic=False, pretty_print=False)
    >>> Permutation([1, 0, 2, 3])
    Permutation([1, 0, 2, 3])
    
    # 5.4 其他方法简介
    # 排列可以作为双射函数，告诉给定位置上的元素是什么
    >>> q = Permutation([5, 2, 3, 4, 1, 0])
    # 创建一个 Permutation 对象 q，表示排列 [5, 2, 3, 4, 1, 0]

    >>> q.array_form[1] # the hard way
    # 访问排列 q 的 array_form 属性中索引为 1 的元素，即第 2 个元素，返回 2

    >>> q(1) # the easy way
    # 使用 Permutation 对象 q 的函数调用方式，查询其作用在 1 上的结果，即 q(1) 返回 2

    >>> {i: q(i) for i in range(q.size)} # showing the bijection
    # 构建一个字典，键为范围在 0 到 q.size-1 的整数，值为 Permutation 对象 q 作用在该键上的结果

    The full cyclic form (including singletons) can be obtained:

    >>> p.full_cyclic_form
    # 获取 Permutation 对象 p 的完整循环形式（包括单元素循环）

    Any permutation can be factored into transpositions of pairs of elements:

    >>> Permutation([[1, 2], [3, 4, 5]]).transpositions()
    # 将任意排列分解为元素对的置换（交换）

    >>> Permutation.rmul(*[Permutation([ti], size=6) for ti in _]).cyclic_form
    # 将一系列置换对象按顺序合成新的 Permutation 对象，并获取其循环形式

    The number of permutations on a set of n elements is given by n! and is
    called the cardinality.

    >>> p.size
    # 获取 Permutation 对象 p 的大小（元素个数）

    >>> p.cardinality
    # 获取 Permutation 对象 p 的基数（排列的总数）

    A given permutation has a rank among all the possible permutations of the
    same elements, but what that rank is depends on how the permutations are
    enumerated. (There are a number of different methods of doing so.) The
    lexicographic rank is given by the rank method and this rank is used to
    increment a permutation with addition/subtraction:

    >>> p.rank()
    # 获取 Permutation 对象 p 在全排列中的排名（按字典序）

    >>> p + 1
    # 将 Permutation 对象 p 的排名加一，得到一个新的 Permutation 对象

    >>> p.next_lex()
    # 获取按字典序下一个排列对象

    >>> _.rank()
    # 获取上一个操作得到的 Permutation 对象的排名

    >>> p.unrank_lex(p.size, rank=7)
    # 获取在指定大小和排名条件下的字典序排列对象

    The product of two permutations p and q is defined as their composition as
    functions, (p*q)(i) = q(p(i)) [6]_.

    >>> p = Permutation([1, 0, 2, 3])
    # 创建 Permutation 对象 p，表示排列 [1, 0, 2, 3]

    >>> q = Permutation([2, 3, 1, 0])
    # 创建 Permutation 对象 q，表示排列 [2, 3, 1, 0]

    >>> list(q*p)
    # 计算排列 q 与 p 的函数组合并转换为列表形式

    >>> list(p*q)
    # 计算排列 p 与 q 的函数组合并转换为列表形式

    >>> [q(p(i)) for i in range(p.size)]
    # 列表推导式，展示排列 q 作用在排列 p 的所有元素上的结果

    The permutation can be 'applied' to any list-like object, not only
    Permutations:

    >>> p(['zero', 'one', 'four', 'two'])
    # 使用排列 p 对列表-like 对象进行应用

    >>> p('zo42')
    # 使用排列 p 对字符串 'zo42' 进行应用

    If you have a list of arbitrary elements, the corresponding permutation
    can be found with the from_sequence method:

    >>> Permutation.from_sequence('SymPy')
    # 使用 from_sequence 方法从任意元素列表中构建相应的排列对象

    Checking if a Permutation is contained in a Group
    =================================================

    Generally if you have a group of permutations G on n symbols, and
    you're checking if a permutation on less than n symbols is part
    of that group, the check will fail.

    Here is an example for n=5 and we check if the cycle
    (1,2,3) is in G:

    >>> from sympy import init_printing
    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> from sympy.combinatorics import Cycle, Permutation
    >>> from sympy.combinatorics.perm_groups import PermutationGroup
    >>> G = PermutationGroup(Cycle(2, 3)(4, 5), Cycle(1, 2, 3, 4, 5))
    # 创建一个置换群 G，包含指定的循环置换 Cycle 对象

    >>> p1 = Permutation(Cycle(2, 5, 3))
    # 创建 Permutation 对象 p1，表示循环置换 (2, 5, 3)

    >>> p2 = Permutation(Cycle(1, 2, 3))
    # 创建 Permutation 对象 p2，表示循环置换 (1, 2, 3)

    >>> a1 = Permutation(Cycle(1, 2, 3).list(6))
    # 创建 Permutation 对象 a1，表示循环置换 (1, 2, 3)，设置大小为 6

    >>> a2 = Permutation(Cycle(1, 2, 3)(5))
    # 创建 Permutation 对象 a2，表示循环置换 (1, 2, 3)，并将元素 5 放入循环

    >>> a3 = Permutation(Cycle(1, 2, 3),size=6)
    # 创建 Permutation 对象 a3，表示循环置换 (1, 2, 3)，设置大小为 6
    # 下面是一些关于置换在群中包含性检查的例子和解释

    # 检查置换 p1, p2, a1, a2, a3 是否在群 G 中
    >>> for p in [p1,p2,a1,a2,a3]: p, G.contains(p)
    ((2 5 3), True)  # p1 在群 G 中
    ((1 2 3), False)  # p2 不在群 G 中，检查失败
    ((5)(1 2 3), True)  # a1 在群 G 中
    ((5)(1 2 3), True)  # a2 在群 G 中
    ((5)(1 2 3), True)  # a3 在群 G 中

    The check for p2 above will fail.
    对于 p2 的检查失败。

    Checking if p1 is in G works because SymPy knows
    G is a group on 5 symbols, and p1 is also on 5 symbols
    (its largest element is 5).
    对于 p1，SymPy 可以检查它是否在群 G 中，因为 SymPy 知道 G 是一个包含 5 个符号的群，而 p1 也是在 5 个符号上的置换（它最大的元素是 5）。

    For ``a1``, the ``.list(6)`` call will extend the permutation to 5
    symbols, so the test will work as well. In the case of ``a2`` the
    permutation is being extended to 5 symbols by using a singleton,
    and in the case of ``a3`` it's extended through the constructor
    argument ``size=6``.
    对于 a1，通过 ``.list(6)`` 调用将置换扩展到 5 个符号，因此测试也会成功。在 a2 的情况下，通过使用单例将置换扩展到 5 个符号，而在 a3 的情况下，则通过构造函数参数 ``size=6`` 进行扩展。

    There is another way to do this, which is to tell the ``contains``
    method that the number of symbols the group is on does not need to
    match perfectly the number of symbols for the permutation:

    >>> G.contains(p2,strict=False)
    True

    This can be via the ``strict`` argument to the ``contains`` method,
    and SymPy will try to extend the permutation on its own and then
    perform the containment check.
    还有另一种方法可以做到这一点，就是告诉 ``contains`` 方法群的符号数不需要完全匹配置换的符号数：

    通过 ``strict`` 参数传递给 ``contains`` 方法，SymPy 将尝试自行扩展置换，然后执行包含性检查。

    See Also
    ========

    Cycle

    References
    ==========

    .. [1] Skiena, S. 'Permutations.' 1.1 in Implementing Discrete Mathematics
           Combinatorics and Graph Theory with Mathematica.  Reading, MA:
           Addison-Wesley, pp. 3-16, 1990.

    .. [2] Knuth, D. E. The Art of Computer Programming, Vol. 4: Combinatorial
           Algorithms, 1st ed. Reading, MA: Addison-Wesley, 2011.

    .. [3] Wendy Myrvold and Frank Ruskey. 2001. Ranking and unranking
           permutations in linear time. Inf. Process. Lett. 79, 6 (September 2001),
           281-284. DOI=10.1016/S0020-0190(01)00141-7

    .. [4] D. L. Kreher, D. R. Stinson 'Combinatorial Algorithms'
           CRC Press, 1999

    .. [5] Graham, R. L.; Knuth, D. E.; and Patashnik, O.
           Concrete Mathematics: A Foundation for Computer Science, 2nd ed.
           Reading, MA: Addison-Wesley, 1994.

    .. [6] https://en.wikipedia.org/w/index.php?oldid=499948155#Product_and_inverse

    .. [7] https://en.wikipedia.org/wiki/Lehmer_code
    def _af_new(cls, perm):
        """
        A method to produce a Permutation object from a list;
        the list is bound to the _array_form attribute, so it must
        not be modified; this method is meant for internal use only;
        the list ``a`` is supposed to be generated as a temporary value
        in a method, so p = Perm._af_new(a) is the only object
        to hold a reference to ``a``::

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Perm
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> a = [2, 1, 3, 0]
        >>> p = Perm._af_new(a)
        >>> p
        Permutation([2, 1, 3, 0])

        """
        # 创建一个新的 Permutation 对象，将 perm 赋给其 _array_form 属性
        p = super().__new__(cls)
        p._array_form = perm
        p._size = len(perm)
        return p

    def copy(self):
        """
        Return a copy of the current Permutation object.

        """
        # 使用当前类的构造函数创建一个新的对象，以当前对象的 array_form 属性作为参数
        return self.__class__(self.array_form)

    def __getnewargs__(self):
        """
        Return the arguments for creating a new instance of the Permutation object.

        """
        return (self.array_form,)

    def _hashable_content(self):
        """
        Return a hashable representation of the Permutation object.

        """
        # 返回 _array_form 属性的元组形式，用于哈希计算
        return tuple(self.array_form)

    @property
    def array_form(self):
        """
        Return a copy of the attribute _array_form.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([[2, 0], [3, 1]])
        >>> p.array_form
        [2, 3, 0, 1]
        >>> Permutation([[2, 0, 3, 1]]).array_form
        [3, 2, 0, 1]
        >>> Permutation([2, 0, 3, 1]).array_form
        [2, 0, 3, 1]
        >>> Permutation([[1, 2], [4, 5]]).array_form
        [0, 2, 1, 3, 5, 4]

        """
        # 返回 _array_form 属性的副本
        return self._array_form[:]

    def list(self, size=None):
        """
        Return the permutation as an explicit list, possibly
        trimming unmoved elements if size is less than the maximum
        element in the permutation; if this is desired, setting
        ``size=-1`` will guarantee such trimming.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation(2, 3)(4, 5)
        >>> p.list()
        [0, 1, 3, 2, 5, 4]
        >>> p.list(10)
        [0, 1, 3, 2, 5, 4, 6, 7, 8, 9]

        Passing a length too small will trim trailing, unchanged elements
        in the permutation:

        >>> Permutation(2, 4)(1, 2, 4).list(-1)
        [0, 2, 1]
        >>> Permutation(3).list(-1)
        []

        """
        if not self and size is None:
            raise ValueError('must give size for empty Cycle')
        rv = self.array_form
        if size is not None:
            if size > self.size:
                rv.extend(list(range(self.size, size)))
            else:
                # 找到从右到左第一个 rv[i] != i 的值
                i = self.size - 1
                while rv:
                    if rv[-1] != i:
                        break
                    rv.pop()
                    i -= 1
        return rv
    def cyclic_form(self):
        """
        This is used to convert to the cyclic notation
        from the canonical notation. Singletons are omitted.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 3, 1, 2])
        >>> p.cyclic_form
        [[1, 3, 2]]
        >>> Permutation([1, 0, 2, 4, 3, 5]).cyclic_form
        [[0, 1], [3, 4]]

        See Also
        ========

        array_form, full_cyclic_form
        """
        # 如果已经计算过循环形式，则直接返回其副本
        if self._cyclic_form is not None:
            return list(self._cyclic_form)
        # 获取排列的数组形式
        array_form = self.array_form
        # 创建未检查标记列表，用于标记是否已处理每个位置
        unchecked = [True] * len(array_form)
        # 存储循环形式的结果列表
        cyclic_form = []
        # 遍历每个位置
        for i in range(len(array_form)):
            # 如果位置未检查过
            if unchecked[i]:
                # 新建一个循环列表，并将当前位置加入其中
                cycle = []
                cycle.append(i)
                unchecked[i] = False
                j = i
                # 找到当前位置的下一个位置，构成一个完整的循环
                while unchecked[array_form[j]]:
                    j = array_form[j]
                    cycle.append(j)
                    unchecked[j] = False
                # 如果循环长度大于1，则加入到循环形式列表中
                if len(cycle) > 1:
                    cyclic_form.append(cycle)
                    # 断言循环是最小字典序
                    assert cycle == list(minlex(cycle))
        # 对循环形式列表进行排序
        cyclic_form.sort()
        # 缓存计算结果，并返回循环形式列表
        self._cyclic_form = cyclic_form[:]
        return cyclic_form

    @property
    def full_cyclic_form(self):
        """Return permutation in cyclic form including singletons.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([0, 2, 1]).full_cyclic_form
        [[0], [1, 2]]
        """
        # 计算出现在的循环形式
        need = set(range(self.size)) - set(flatten(self.cyclic_form))
        # 将单个元素作为单元素循环加入到结果中
        rv = self.cyclic_form + [[i] for i in need]
        # 对结果进行排序并返回
        rv.sort()
        return rv

    @property
    def size(self):
        """
        Returns the number of elements in the permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([[3, 2], [0, 1]]).size
        4

        See Also
        ========

        cardinality, length, order, rank
        """
        # 返回排列中元素的数量
        return self._size

    def support(self):
        """Return the elements in permutation, P, for which P[i] != i.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([[3, 2], [0, 1], [4]])
        >>> p.array_form
        [1, 0, 3, 2, 4]
        >>> p.support()
        [0, 1, 2, 3]
        """
        # 获取排列的数组形式
        a = self.array_form
        # 返回所有 P[i] != i 的元素索引列表
        return [i for i, e in enumerate(a) if e != i]
    def __add__(self, other):
        """
        Return permutation that is other higher in rank than self.

        The rank is the lexicographical rank, with the identity permutation
        having rank of 0.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> I = Permutation([0, 1, 2, 3])
        >>> a = Permutation([2, 1, 3, 0])
        >>> I + a.rank() == a
        True

        See Also
        ========

        __sub__, inversion_vector

        """
        # Calculate the new rank using lexicographical ordering
        rank = (self.rank() + other) % self.cardinality
        # Generate the permutation corresponding to the new rank
        rv = self.unrank_lex(self.size, rank)
        # Set the rank attribute of the resulting permutation
        rv._rank = rank
        return rv

    def __sub__(self, other):
        """
        Return the permutation that is other lower in rank than self.

        See Also
        ========

        __add__
        """
        # Subtracting other from self is equivalent to adding its negative
        return self.__add__(-other)

    @staticmethod
    def rmul(*args):
        """
        Return product of Permutations [a, b, c, ...] as the Permutation whose
        ith value is a(b(c(i))).

        a, b, c, ... can be Permutation objects or tuples.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation

        >>> a, b = [1, 0, 2], [0, 2, 1]
        >>> a = Permutation(a); b = Permutation(b)
        >>> list(Permutation.rmul(a, b))
        [1, 2, 0]
        >>> [a(b(i)) for i in range(3)]
        [1, 2, 0]

        This handles the operands in reverse order compared to the ``*`` operator:

        >>> a = Permutation(a); b = Permutation(b)
        >>> list(a*b)
        [2, 0, 1]
        >>> [b(a(i)) for i in range(3)]
        [2, 0, 1]

        Notes
        =====

        All items in the sequence will be parsed by Permutation as
        necessary as long as the first item is a Permutation:

        >>> Permutation.rmul(a, [0, 2, 1]) == Permutation.rmul(a, b)
        True

        The reverse order of arguments will raise a TypeError.

        """
        # Initialize the result with the first permutation
        rv = args[0]
        # Multiply the permutations in reverse order
        for i in range(1, len(args)):
            rv = args[i]*rv
        return rv

    @classmethod
    def rmul_with_af(cls, *args):
        """
        Same as rmul, but the elements of args are Permutation objects
        which have _array_form.
        """
        # Extract _array_form from each Permutation object in args
        a = [x._array_form for x in args]
        # Compute the product of permutations using _af_rmuln function
        rv = cls._af_new(_af_rmuln(*a))
        return rv

    def mul_inv(self, other):
        """
        Compute the inverse permutation of other multiplied by self.

        Here, self and other are expected to have _array_form.
        """
        # Compute the inverse of self's _array_form
        a = _af_invert(self._array_form)
        # Get other's _array_form
        b = other._array_form
        # Return the new permutation constructed from the product
        return self._af_new(_af_rmul(a, b))

    def __rmul__(self, other):
        """
        This is needed to coerce other to Permutation in rmul.

        Returns the result of multiplying other by self.
        """
        # Ensure other is coerced into a Permutation and then multiply
        cls = type(self)
        return cls(other)*self
    def __mul__(self, other):
        """
        Return the product a*b as a Permutation; the ith value is b(a(i)).

        Examples
        ========

        >>> from sympy.combinatorics.permutations import _af_rmul, Permutation

        >>> a, b = [1, 0, 2], [0, 2, 1]
        >>> a = Permutation(a); b = Permutation(b)
        >>> list(a*b)
        [2, 0, 1]
        >>> [b(a(i)) for i in range(3)]
        [2, 0, 1]

        This handles operands in reverse order compared to _af_rmul and rmul:

        >>> al = list(a); bl = list(b)
        >>> _af_rmul(al, bl)
        [1, 2, 0]
        >>> [al[bl[i]] for i in range(3)]
        [1, 2, 0]

        It is acceptable for the arrays to have different lengths; the shorter
        one will be padded to match the longer one:

        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> b*Permutation([1, 0])
        Permutation([1, 2, 0])
        >>> Permutation([1, 0])*b
        Permutation([2, 0, 1])

        It is also acceptable to allow coercion to handle conversion of a
        single list to the left of a Permutation:

        >>> [0, 1]*a # no change: 2-element identity
        Permutation([1, 0, 2])
        >>> [[0, 1]]*a # exchange first two elements
        Permutation([0, 1, 2])

        You cannot use more than 1 cycle notation in a product of cycles
        since coercion can only handle one argument to the left. To handle
        multiple cycles it is convenient to use Cycle instead of Permutation:

        >>> [[1, 2]]*[[2, 3]]*Permutation([]) # doctest: +SKIP
        >>> from sympy.combinatorics.permutations import Cycle
        >>> Cycle(1, 2)(2, 3)
        (1 3 2)

        """
        from sympy.combinatorics.perm_groups import PermutationGroup, Coset
        # 如果other是PermutationGroup类型，则返回与其交换的Coset对象
        if isinstance(other, PermutationGroup):
            return Coset(self, other, dir='-')
        # 获取自身的数组形式
        a = self.array_form
        # __rmul__确保other是一个Permutation对象
        b = other.array_form
        # 如果b为空，则结果为a
        if not b:
            perm = a
        else:
            # 将b扩展到与a相同的长度
            b.extend(list(range(len(b), len(a))))
            # 计算乘积并生成新的排列
            perm = [b[i] for i in a] + b[len(a):]
        # 返回新生成的排列对象
        return self._af_new(perm)

    def commutes_with(self, other):
        """
        Checks if the elements are commuting.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> a = Permutation([1, 4, 3, 0, 2, 5])
        >>> b = Permutation([0, 1, 2, 3, 4, 5])
        >>> a.commutes_with(b)
        True
        >>> b = Permutation([2, 3, 5, 4, 1, 0])
        >>> a.commutes_with(b)
        False
        """
        # 获取自身和另一个排列对象的数组形式
        a = self.array_form
        b = other.array_form
        # 使用_af_commutes_with函数检查两个排列元素是否对易
        return _af_commutes_with(a, b)
    def __pow__(self, n):
        """
        Routine for finding powers of a permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([2, 0, 3, 1])
        >>> p.order()
        4
        >>> p**4
        Permutation([0, 1, 2, 3])
        """
        # 如果 n 是 Permutation 对象，则抛出未实现的错误
        if isinstance(n, Permutation):
            raise NotImplementedError(
                'p**p is not defined; do you mean p^p (conjugate)?')
        # 将 n 转换为整数
        n = int(n)
        # 调用 _af_pow 函数计算置换的 n 次幂，并返回一个新的置换对象
        return self._af_new(_af_pow(self.array_form, n))

    def __rxor__(self, i):
        """Return self(i) when ``i`` is an int.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation(1, 2, 9)
        >>> 2^p == p(2) == 9
        True
        """
        # 如果 i 是整数，则返回 self(i)
        if int_valued(i):
            return self(i)
        else:
            # 如果 i 不是整数，则抛出未实现的错误
            raise NotImplementedError(
                "i^p = p(i) when i is an integer, not %s." % i)
    def __xor__(self, h):
        """
        Return the conjugate permutation ``~h*self*h``.

        Explanation
        ===========

        If ``a`` and ``b`` are conjugates, ``a = h*b*~h`` and
        ``b = ~h*a*h`` and both have the same cycle structure.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation(1, 2, 9)
        >>> q = Permutation(6, 9, 8)
        >>> p*q != q*p
        True

        Calculate and check properties of the conjugate:

        >>> c = p^q
        >>> c == ~q*p*q and p == q*c*~q
        True

        The expression q^p^r is equivalent to q^(p*r):

        >>> r = Permutation(9)(4, 6, 8)
        >>> q^p^r == q^(p*r)
        True

        If the term to the left of the conjugate operator, i, is an integer
        then this is interpreted as selecting the ith element from the
        permutation to the right:

        >>> all(i^p == p(i) for i in range(p.size))
        True

        Note that the * operator has higher precedence than the ^ operator:

        >>> q^r*p^r == q^(r*p)^r == Permutation(9)(1, 6, 4)
        True

        Notes
        =====

        In Python, the precedence rule is p^q^r = (p^q)^r which differs
        in general from p^(q^r).

        >>> q^p^r
        (9)(1 4 8)
        >>> q^(p^r)
        (9)(1 8 6)

        For a given r and p, both of the following are conjugates of p:
        ~r*p*r and r*p*~r. But these are not necessarily the same:

        >>> ~r*p*r == r*p*~r
        True

        >>> p = Permutation(1, 2, 9)(5, 6)
        >>> ~r*p*r == r*p*~r
        False

        The conjugate ~r*p*r was chosen so that ``p^q^r`` would be equivalent
        to ``p^(q*r)`` rather than ``p^(r*q)``. To obtain r*p*~r, pass ~r to
        this method:

        >>> p^~r == r*p*~r
        True
        """
        
        # 检查两个排列的大小是否相等，若不等则引发错误
        if self.size != h.size:
            raise ValueError("The permutations must be of equal size.")
        
        # 初始化一个数组，用于存储结果排列的数组形式
        a = [None]*self.size
        
        # 获取排列 h 和 self 的数组形式
        h = h._array_form
        p = self._array_form
        
        # 根据公式 a[h[i]] = h[p[i]] 计算结果排列的数组形式
        for i in range(self.size):
            a[h[i]] = h[p[i]]
        
        # 根据数组形式 a 创建一个新的排列对象并返回
        return self._af_new(a)
    def transpositions(self):
        """
        Return the permutation decomposed into a list of transpositions.

        Explanation
        ===========

        It is always possible to express a permutation as the product of
        transpositions, see [1]

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([[1, 2, 3], [0, 4, 5, 6, 7]])
        >>> t = p.transpositions()
        >>> t
        [(0, 7), (0, 6), (0, 5), (0, 4), (1, 3), (1, 2)]
        >>> print(''.join(str(c) for c in t))
        (0, 7)(0, 6)(0, 5)(0, 4)(1, 3)(1, 2)
        >>> Permutation.rmul(*[Permutation([ti], size=p.size) for ti in t]) == p
        True

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Transposition_%28mathematics%29#Properties

        """
        a = self.cyclic_form  # 获取循环形式的表示
        res = []  # 初始化结果列表
        for x in a:  # 遍历循环形式中的每一个循环
            nx = len(x)  # 获取当前循环的长度
            if nx == 2:
                res.append(tuple(x))  # 如果长度为2，则直接作为一个转位加入结果列表
            elif nx > 2:
                first = x[0]  # 获取循环的第一个元素
                res.extend((first, y) for y in x[nx - 1:0:-1])  # 将循环展开成转位并加入结果列表
        return res  # 返回由转位组成的列表

    @classmethod
    def from_sequence(self, i, key=None):
        """Return the permutation needed to obtain ``i`` from the sorted
        elements of ``i``. If custom sorting is desired, a key can be given.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation

        >>> Permutation.from_sequence('SymPy')
        (4)(0 1 3)
        >>> _(sorted("SymPy"))
        ['S', 'y', 'm', 'P', 'y']
        >>> Permutation.from_sequence('SymPy', key=lambda x: x.lower())
        (4)(0 2)(1 3)
        """
        ic = list(zip(i, list(range(len(i)))))  # 构建元组列表，包含元素和其索引
        if key:
            ic.sort(key=lambda x: key(x[0]))  # 如果指定了排序关键字，则按照关键字排序
        else:
            ic.sort()  # 否则按照默认排序
        return ~Permutation([i[1] for i in ic])  # 返回排序后的元素索引构成的排列的逆排列

    def __invert__(self):
        """
        Return the inverse of the permutation.

        A permutation multiplied by its inverse is the identity permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([[2, 0], [3, 1]])
        >>> ~p
        Permutation([2, 3, 0, 1])
        >>> _ == p**-1
        True
        >>> p*~p == ~p*p == Permutation([0, 1, 2, 3])
        True
        """
        return self._af_new(_af_invert(self._array_form))  # 返回当前排列的逆排列

    def __iter__(self):
        """Yield elements from array form.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> list(Permutation(range(3)))
        [0, 1, 2]
        """
        yield from self.array_form  # 从数组形式中逐个生成元素

    def __repr__(self):
        return srepr(self)  # 返回当前对象的字符串表示形式
    # 允许将置换实例作为双射函数应用
    def __call__(self, *i):
        """
        Allows applying a permutation instance as a bijective function.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([[2, 0], [3, 1]])
        >>> p.array_form
        [2, 3, 0, 1]
        >>> [p(i) for i in range(4)]
        [2, 3, 0, 1]

        If an array is given then the permutation selects the items
        from the array (i.e. the permutation is applied to the array):

        >>> from sympy.abc import x
        >>> p([x, 1, 0, x**2])
        [0, x**2, x, 1]
        """
        # 如果只有一个参数 i，则将其作为单个参数处理
        if len(i) == 1:
            i = i[0]
            # 如果 i 不是可迭代对象，则尝试将其转换为整数
            if not isinstance(i, Iterable):
                i = as_int(i)
                # 检查 i 是否在合法范围内，否则抛出异常
                if i < 0 or i > self.size:
                    raise TypeError(
                        "{} should be an integer between 0 and {}"
                        .format(i, self.size-1))
                # 返回置换后的元素
                return self._array_form[i]
            # 如果 i 的长度与置换大小不符，则抛出异常
            if len(i) != self.size:
                raise TypeError(
                    "{} should have the length {}.".format(i, self.size))
            # 返回置换后的数组
            return [i[j] for j in self._array_form]
        # 如果有多个参数 i，则将其视为循环结构的置换，返回组合后的置换结果
        return self * Permutation(Cycle(*i), size=self.size)

    # 返回置换的所有元素构成的集合
    def atoms(self):
        """
        Returns all the elements of a permutation

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([0, 1, 2, 3, 4, 5]).atoms()
        {0, 1, 2, 3, 4, 5}
        >>> Permutation([[0, 1], [2, 3], [4, 5]]).atoms()
        {0, 1, 2, 3, 4, 5}
        """
        return set(self.array_form)
    # 定义一个方法，用于对表达式应用排列
    def apply(self, i):
        r"""Apply the permutation to an expression.

        Parameters
        ==========

        i : Expr
            It should be an integer between $0$ and $n-1$ where $n$
            is the size of the permutation.

            If it is a symbol or a symbolic expression that can
            have integer values, an ``AppliedPermutation`` object
            will be returned which can represent an unevaluated
            function.

        Notes
        =====

        Any permutation can be defined as a bijective function
        $\sigma : \{ 0, 1, \dots, n-1 \} \rightarrow \{ 0, 1, \dots, n-1 \}$
        where $n$ denotes the size of the permutation.

        The definition may even be extended for any set with distinctive
        elements, such that the permutation can even be applied for
        real numbers or such, however, it is not implemented for now for
        computational reasons and the integrity with the group theory
        module.

        This function is similar to the ``__call__`` magic, however,
        ``__call__`` magic already has some other applications like
        permuting an array or attaching new cycles, which would
        not always be mathematically consistent.

        This also guarantees that the return type is a SymPy integer,
        which guarantees the safety to use assumptions.
        """
        # 将输入的i转换成符号表达式
        i = _sympify(i)
        # 如果i不是整数，抛出未实现错误
        if i.is_integer is False:
            raise NotImplementedError("{} should be an integer.".format(i))

        # 获取排列的大小n
        n = self.size
        # 如果i小于0或者大于等于n，抛出未实现错误
        if (i < 0) == True or (i >= n) == True:
            raise NotImplementedError(
                "{} should be an integer between 0 and {}".format(i, n-1))

        # 如果i是整数，返回排列中第i个元素的整数表示
        if i.is_Integer:
            return Integer(self._array_form[i])
        # 否则返回一个应用了排列的对象
        return AppliedPermutation(self, i)

    # 定义一个方法，返回字典序下一个排列
    def next_lex(self):
        """
        Returns the next permutation in lexicographical order.
        If self is the last permutation in lexicographical order
        it returns None.
        See [4] section 2.4.


        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([2, 3, 1, 0])
        >>> p = Permutation([2, 3, 1, 0]); p.rank()
        17
        >>> p = p.next_lex(); p.rank()
        18

        See Also
        ========

        rank, unrank_lex
        """
        # 复制排列的数组形式
        perm = self.array_form[:]
        # 获取排列的长度n
        n = len(perm)
        # 初始化i为倒数第二个位置
        i = n - 2
        # 寻找下一个字典序排列
        while perm[i + 1] < perm[i]:
            i -= 1
        # 如果没有找到，返回None
        if i == -1:
            return None
        else:
            # 找到后交换元素位置
            j = n - 1
            while perm[j] < perm[i]:
                j -= 1
            perm[j], perm[i] = perm[i], perm[j]
            i += 1
            j = n - 1
            while i < j:
                perm[j], perm[i] = perm[i], perm[j]
                i += 1
                j -= 1
        # 返回新排列对象
        return self._af_new(perm)
    def unrank_nonlex(self, n, r):
        """
        This is a linear time unranking algorithm that does not
        respect lexicographic order [3].

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> Permutation.unrank_nonlex(4, 5)
        Permutation([2, 0, 3, 1])
        >>> Permutation.unrank_nonlex(4, -1)
        Permutation([0, 1, 2, 3])

        See Also
        ========

        next_nonlex, rank_nonlex
        """
        # Helper function to perform the unranking
        def _unrank1(n, r, a):
            if n > 0:
                # Swap elements to generate the permutation
                a[n - 1], a[r % n] = a[r % n], a[n - 1]
                _unrank1(n - 1, r//n, a)

        # Initialize the identity permutation list
        id_perm = list(range(n))
        n = int(n)  # Ensure n is treated as an integer
        r = r % ifac(n)  # Calculate r modulo factorial of n
        # Call the helper function to generate the permutation
        _unrank1(n, r, id_perm)
        # Return the permutation object created from the permutation list
        return self._af_new(id_perm)

    def rank_nonlex(self, inv_perm=None):
        """
        This is a linear time ranking algorithm that does not
        enforce lexicographic order [3].

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.rank_nonlex()
        23

        See Also
        ========

        next_nonlex, unrank_nonlex
        """
        # Helper function to perform the ranking
        def _rank1(n, perm, inv_perm):
            if n == 1:
                return 0
            s = perm[n - 1]
            t = inv_perm[n - 1]
            perm[n - 1], perm[t] = perm[t], s
            inv_perm[n - 1], inv_perm[s] = inv_perm[s], t
            return s + n*_rank1(n - 1, perm, inv_perm)

        # If inv_perm is not provided, generate it from the inverse of self
        if inv_perm is None:
            inv_perm = (~self).array_form
        # If inv_perm is empty, return 0 (default ranking for identity permutation)
        if not inv_perm:
            return 0
        # Create a copy of the permutation as a list
        perm = self.array_form[:]
        # Calculate the rank using the helper function
        r = _rank1(len(perm), perm, inv_perm)
        return r

    def next_nonlex(self):
        """
        Returns the next permutation in nonlex order [3].
        If self is the last permutation in this order it returns None.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([2, 0, 3, 1]); p.rank_nonlex()
        5
        >>> p = p.next_nonlex(); p
        Permutation([3, 0, 1, 2])
        >>> p.rank_nonlex()
        6

        See Also
        ========

        rank_nonlex, unrank_nonlex
        """
        # Get the rank of the current permutation
        r = self.rank_nonlex()
        # If the rank is the last permutation, return None
        if r == ifac(self.size) - 1:
            return None
        # Otherwise, return the next permutation
        return self.unrank_nonlex(self.size, r + 1)
    def rank(self):
        """
        Returns the lexicographic rank of the permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.rank()
        0
        >>> p = Permutation([3, 2, 1, 0])
        >>> p.rank()
        23

        See Also
        ========

        next_lex, unrank_lex, cardinality, length, order, size
        """
        # 如果已经计算过排列的秩，则直接返回保存的结果
        if self._rank is not None:
            return self._rank
        rank = 0
        rho = self.array_form[:]  # 复制排列的数组表示
        n = self.size - 1  # 排列的大小减一
        size = n + 1  # 排列的大小
        psize = int(ifac(n))  # 排列的大小的阶乘
        # 计算排列的秩
        for j in range(size - 1):
            rank += rho[j]*psize  # 根据排列的数组形式计算当前位置的贡献
            for i in range(j + 1, size):
                if rho[i] > rho[j]:  # 如果后面的元素比当前元素大，则减小后面元素的值
                    rho[i] -= 1
            psize //= n  # 更新阶乘
            n -= 1  # 更新排列大小
        self._rank = rank  # 保存计算结果
        return rank  # 返回计算结果

    @property
    def cardinality(self):
        """
        Returns the number of all possible permutations.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.cardinality
        24

        See Also
        ========

        length, order, rank, size
        """
        return int(ifac(self.size))  # 返回排列的大小的阶乘，即所有可能的排列数量

    def parity(self):
        """
        Computes the parity of a permutation.

        Explanation
        ===========

        The parity of a permutation reflects the parity of the
        number of inversions in the permutation, i.e., the
        number of pairs of x and y such that ``x > y`` but ``p[x] < p[y]``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.parity()
        0
        >>> p = Permutation([3, 2, 0, 1])
        >>> p.parity()
        1

        See Also
        ========

        _af_parity
        """
        # 如果已经计算过置换的循环形式，则直接返回计算的奇偶性
        if self._cyclic_form is not None:
            return (self.size - self.cycles) % 2

        return _af_parity(self.array_form)  # 否则调用函数计算排列的奇偶性

    @property
    def is_even(self):
        """
        Checks if a permutation is even.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.is_even
        True
        >>> p = Permutation([3, 2, 1, 0])
        >>> p.is_even
        True

        See Also
        ========

        is_odd
        """
        return not self.is_odd  # 判断排列是否为偶排列

    @property
    def is_odd(self):
        """
        Checks if a permutation is odd.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.is_odd
        False
        >>> p = Permutation([3, 2, 0, 1])
        >>> p.is_odd
        True

        See Also
        ========

        is_even
        """
        return bool(self.parity() % 2)  # 判断排列是否为奇排列
    def is_Singleton(self):
        """
        检查排列是否只包含一个数字，因此是该数字集合的唯一可能排列

        示例
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([0]).is_Singleton
        True
        >>> Permutation([0, 1]).is_Singleton
        False

        参见
        ========

        is_Empty
        """
        return self.size == 1

    @property
    def is_Empty(self):
        """
        检查排列是否是一个空集

        示例
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([]).is_Empty
        True
        >>> Permutation([0]).is_Empty
        False

        参见
        ========

        is_Singleton
        """
        return self.size == 0

    @property
    def is_identity(self):
        return self.is_Identity

    @property
    def is_Identity(self):
        """
        如果排列是单位排列，则返回True。

        示例
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([])
        >>> p.is_Identity
        True
        >>> p = Permutation([[0], [1], [2]])
        >>> p.is_Identity
        True
        >>> p = Permutation([0, 1, 2])
        >>> p.is_Identity
        True
        >>> p = Permutation([0, 2, 1])
        >>> p.is_Identity
        False

        参见
        ========

        order
        """
        af = self.array_form
        return not af or all(i == af[i] for i in range(self.size))

    def ascents(self):
        """
        返回排列中升序位置的列表，即位置满足 p[i] < p[i+1]

        示例
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([4, 0, 1, 3, 2])
        >>> p.ascents()
        [1, 2]

        参见
        ========

        descents, inversions, min, max
        """
        a = self.array_form
        pos = [i for i in range(len(a) - 1) if a[i] < a[i + 1]]
        return pos

    def descents(self):
        """
        返回排列中降序位置的列表，即位置满足 p[i] > p[i+1]

        示例
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([4, 0, 1, 3, 2])
        >>> p.descents()
        [0, 3]

        参见
        ========

        ascents, inversions, min, max
        """
        a = self.array_form
        pos = [i for i in range(len(a) - 1) if a[i] > a[i + 1]]
        return pos
    def max(self) -> int:
        """
        返回置换中移动的最大元素。

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([1, 0, 2, 3, 4])
        >>> p.max()
        1

        See Also
        ========

        min, descents, ascents, inversions
        """
        # 获取置换的数组形式
        a = self.array_form
        # 如果数组为空，返回 0
        if not a:
            return 0
        # 返回不等于索引的元素中的最大值
        return max(_a for i, _a in enumerate(a) if _a != i)

    def min(self) -> int:
        """
        返回置换中移动的最小元素。

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 4, 3, 2])
        >>> p.min()
        2

        See Also
        ========

        max, descents, ascents, inversions
        """
        # 获取置换的数组形式
        a = self.array_form
        # 如果数组为空，返回 0
        if not a:
            return 0
        # 返回不等于索引的元素中的最小值
        return min(_a for i, _a in enumerate(a) if _a != i)

    def inversions(self):
        """
        计算置换的逆序数。

        Explanation
        ===========

        逆序数是指在索引 i > j 但 p[i] < p[j] 的情况数量。

        对于较小长度的置换 p，通过遍历所有的 i 和 j 值计算逆序数。
        对于较大长度的置换 p，使用改进的归并排序方法计算逆序数。

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3, 4, 5])
        >>> p.inversions()
        0
        >>> Permutation([3, 2, 1, 0]).inversions()
        6

        See Also
        ========

        descents, ascents, min, max

        References
        ==========

        .. [1] https://www.cp.eng.chula.ac.th/~prabhas//teaching/algo/algo2008/count-inv.htm

        """
        inversions = 0
        # 获取置换的数组形式
        a = self.array_form
        # 置换数组的长度
        n = len(a)
        # 对于长度小于 130 的置换，使用简单的遍历方法计算逆序数
        if n < 130:
            for i in range(n - 1):
                b = a[i]
                for c in a[i + 1:]:
                    if b > c:
                        inversions += 1
        else:
            # 对于长度大于等于 130 的置换，使用改进的归并排序方法计算逆序数
            k = 1
            right = 0
            arr = a[:]
            temp = a[:]
            while k < n:
                i = 0
                while i + k < n:
                    right = i + k * 2 - 1
                    if right >= n:
                        right = n - 1
                    inversions += _merge(arr, temp, i, i + k, right)
                    i = i + k * 2
                k = k * 2
        return inversions
    def commutator(self, x):
        """
        Return the commutator of ``self`` and ``x``: ``~x*~self*x*self``

        If f and g are part of a group, G, then the commutator of f and g
        is the group identity iff f and g commute, i.e. fg == gf.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([0, 2, 3, 1])
        >>> x = Permutation([2, 0, 3, 1])
        >>> c = p.commutator(x); c
        Permutation([2, 1, 3, 0])
        >>> c == ~x*~p*x*p
        True

        >>> I = Permutation(3)
        >>> p = [I + i for i in range(6)]
        >>> for i in range(len(p)):
        ...     for j in range(len(p)):
        ...         c = p[i].commutator(p[j])
        ...         if p[i]*p[j] == p[j]*p[i]:
        ...             assert c == I
        ...         else:
        ...             assert c != I
        ...

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Commutator
        """

        # Convert permutations to array form
        a = self.array_form
        b = x.array_form
        n = len(a)
        # Ensure both permutations are of equal size
        if len(b) != n:
            raise ValueError("The permutations must be of equal size.")
        # Compute inverse arrays
        inva = [None]*n
        for i in range(n):
            inva[a[i]] = i
        invb = [None]*n
        for i in range(n):
            invb[b[i]] = i
        # Compute and return the commutator permutation
        return self._af_new([a[b[inva[i]]] for i in invb])

    def signature(self):
        """
        Gives the signature of the permutation needed to place the
        elements of the permutation in canonical order.

        The signature is calculated as (-1)^<number of inversions>

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2])
        >>> p.inversions()
        0
        >>> p.signature()
        1
        >>> q = Permutation([0,2,1])
        >>> q.inversions()
        1
        >>> q.signature()
        -1

        See Also
        ========

        inversions
        """
        # Return 1 if the permutation is even, otherwise return -1
        if self.is_even:
            return 1
        return -1

    def order(self):
        """
        Computes the order of a permutation.

        When the permutation is raised to the power of its
        order it equals the identity permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([3, 1, 5, 2, 4, 0])
        >>> p.order()
        4
        >>> (p**(p.order()))
        Permutation([], size=6)

        See Also
        ========

        identity, cardinality, length, rank, size
        """

        # Compute the order of the permutation using the least common multiple (LCM)
        return reduce(lcm, [len(cycle) for cycle in self.cyclic_form], 1)
    def length(self):
        """
        Returns the number of integers moved by a permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([0, 3, 2, 1]).length()
        2
        >>> Permutation([[0, 1], [2, 3]]).length()
        4

        See Also
        ========

        min, max, support, cardinality, order, rank, size
        """

        # 返回置换中移动的整数的数量，即置换的支持集合的长度
        return len(self.support())

    @property
    def cycle_structure(self):
        """Return the cycle structure of the permutation as a dictionary
        indicating the multiplicity of each cycle length.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation(3).cycle_structure
        {1: 4}
        >>> Permutation(0, 4, 3)(1, 2)(5, 6).cycle_structure
        {2: 2, 3: 1}
        """
        if self._cycle_structure:
            rv = self._cycle_structure
        else:
            rv = defaultdict(int)
            singletons = self.size
            # 计算置换的循环形式，并统计每种循环长度的个数
            for c in self.cyclic_form:
                rv[len(c)] += 1
                singletons -= len(c)
            if singletons:
                rv[1] = singletons
            self._cycle_structure = rv
        return dict(rv)  # 返回循环结构的副本

    @property
    def cycles(self):
        """
        Returns the number of cycles contained in the permutation
        (including singletons).

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([0, 1, 2]).cycles
        3
        >>> Permutation([0, 1, 2]).full_cyclic_form
        [[0], [1], [2]]
        >>> Permutation(0, 1)(2, 3).cycles
        2

        See Also
        ========
        sympy.functions.combinatorial.numbers.stirling
        """
        # 返回置换中包含的循环的数量，包括单个元素的循环
        return len(self.full_cyclic_form)

    def index(self):
        """
        Returns the index of a permutation.

        The index of a permutation is the sum of all subscripts j such
        that p[j] is greater than p[j+1].

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([3, 0, 2, 1, 4])
        >>> p.index()
        2
        """
        a = self.array_form

        # 返回置换的指标，即所有满足 p[j] > p[j+1] 的下标 j 的和
        return sum(j for j in range(len(a) - 1) if a[j] > a[j + 1])

    def runs(self):
        """
        Returns the runs of a permutation.

        An ascending sequence in a permutation is called a run [5].


        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([2, 5, 7, 3, 6, 0, 1, 4, 8])
        >>> p.runs()
        [[2, 5, 7], [3, 6], [0, 1, 4, 8]]
        >>> q = Permutation([1,3,2,0])
        >>> q.runs()
        [[1, 3], [2], [0]]
        """
        # 返回置换的运行（即升序序列）的列表
        return runs(self.array_form)
    def inversion_vector(self):
        """Return the inversion vector of the permutation.

        The inversion vector consists of elements whose value
        indicates the number of elements in the permutation
        that are lesser than it and lie on its right hand side.

        The inversion vector is the same as the Lehmer encoding of a
        permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([4, 8, 0, 7, 1, 5, 3, 6, 2])
        >>> p.inversion_vector()
        [4, 7, 0, 5, 0, 2, 1, 1]
        >>> p = Permutation([3, 2, 1, 0])
        >>> p.inversion_vector()
        [3, 2, 1]

        The inversion vector increases lexicographically with the rank
        of the permutation, the -ith element cycling through 0..i.

        >>> p = Permutation(2)
        >>> while p:
        ...     print('%s %s %s' % (p, p.inversion_vector(), p.rank()))
        ...     p = p.next_lex()
        (2) [0, 0] 0
        (1 2) [0, 1] 1
        (2)(0 1) [1, 0] 2
        (0 1 2) [1, 1] 3
        (0 2 1) [2, 0] 4
        (0 2) [2, 1] 5

        See Also
        ========

        from_inversion_vector
        """
        # 获取置换的数组形式
        self_array_form = self.array_form
        # 置换的长度
        n = len(self_array_form)
        # 初始化逆序向量为全零列表，长度为 n-1
        inversion_vector = [0] * (n - 1)

        # 计算逆序向量的每个元素
        for i in range(n - 1):
            val = 0
            for j in range(i + 1, n):
                # 如果后面的元素比当前元素小，则逆序数加一
                if self_array_form[j] < self_array_form[i]:
                    val += 1
            inversion_vector[i] = val
        # 返回计算得到的逆序向量
        return inversion_vector

    def rank_trotterjohnson(self):
        """
        Returns the Trotter Johnson rank, which we get from the minimal
        change algorithm. See [4] section 2.4.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.rank_trotterjohnson()
        0
        >>> p = Permutation([0, 2, 1, 3])
        >>> p.rank_trotterjohnson()
        7

        See Also
        ========

        unrank_trotterjohnson, next_trotterjohnson
        """
        # 如果置换为空或者为恒等置换，则排名为0
        if self.array_form == [] or self.is_Identity:
            return 0
        # 如果置换为 [1, 0]，则排名为1
        if self.array_form == [1, 0]:
            return 1
        # 否则，根据置换计算排名
        perm = self.array_form
        n = self.size
        rank = 0
        for j in range(1, n):
            k = 1
            i = 0
            while perm[i] != j:
                if perm[i] < j:
                    k += 1
                i += 1
            j1 = j + 1
            if rank % 2 == 0:
                rank = j1*rank + j1 - k
            else:
                rank = j1*rank + k - 1
        # 返回计算得到的排名
        return rank
    def unrank_trotterjohnson(cls, size, rank):
        """
        Trotter Johnson permutation unranking. See [4] section 2.4.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> Permutation.unrank_trotterjohnson(5, 10)
        Permutation([0, 3, 1, 2, 4])

        See Also
        ========

        rank_trotterjohnson, next_trotterjohnson
        """
        # 初始化排列为 [0, 0, ..., 0]，长度为 size
        perm = [0]*size
        r2 = 0
        # 计算 size 的阶乘
        n = ifac(size)
        # pj 为累乘变量，初始为 1
        pj = 1
        # 循环生成排列
        for j in range(2, size + 1):
            # 更新 pj
            pj *= j
            # 计算 r1，按照公式 (rank * pj) // n 计算
            r1 = (rank * pj) // n
            # 计算 k
            k = r1 - j*r2
            # 根据 r2 的奇偶性分情况处理
            if r2 % 2 == 0:
                # 将一部分元素右移
                for i in range(j - 1, j - k - 1, -1):
                    perm[i] = perm[i - 1]
                perm[j - k - 1] = j - 1
            else:
                # 将一部分元素左移
                for i in range(j - 1, k, -1):
                    perm[i] = perm[i - 1]
                perm[k] = j - 1
            # 更新 r2
            r2 = r1
        # 调用 _af_new 方法生成新的排列对象并返回
        return cls._af_new(perm)

    def next_trotterjohnson(self):
        """
        Returns the next permutation in Trotter-Johnson order.
        If self is the last permutation it returns None.
        See [4] section 2.4. If it is desired to generate all such
        permutations, they can be generated in order more quickly
        with the ``generate_bell`` function.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([3, 0, 2, 1])
        >>> p.rank_trotterjohnson()
        4
        >>> p = p.next_trotterjohnson(); p
        Permutation([0, 3, 2, 1])
        >>> p.rank_trotterjohnson()
        5

        See Also
        ========

        rank_trotterjohnson, unrank_trotterjohnson, sympy.utilities.iterables.generate_bell
        """
        # 复制排列到 pi
        pi = self.array_form[:]
        # 排列长度 n
        n = len(pi)
        # st 为起始位置，rho 复制 pi
        st = 0
        rho = pi[:]
        done = False
        m = n-1
        # 循环生成下一个排列
        while m > 0 and not done:
            # 找到最大的 m
            d = rho.index(m)
            # 向左移动元素
            for i in range(d, m):
                rho[i] = rho[i + 1]
            # 计算排列的奇偶性
            par = _af_parity(rho[:m])
            # 根据奇偶性移动元素
            if par == 1:
                if d == m:
                    m -= 1
                else:
                    pi[st + d], pi[st + d + 1] = pi[st + d + 1], pi[st + d]
                    done = True
            else:
                if d == 0:
                    m -= 1
                    st += 1
                else:
                    pi[st + d], pi[st + d - 1] = pi[st + d - 1], pi[st + d]
                    done = True
        # 如果 m == 0，则返回 None，否则生成新的排列对象并返回
        if m == 0:
            return None
        return self._af_new(pi)
    def get_precedence_matrix(self):
        """
        Gets the precedence matrix. This is used for computing the
        distance between two permutations.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation.josephus(3, 6, 1)
        >>> p
        Permutation([2, 5, 3, 1, 4, 0])
        >>> p.get_precedence_matrix()
        Matrix([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0]])

        See Also
        ========

        get_precedence_distance, get_adjacency_matrix, get_adjacency_distance
        """
        # 创建一个大小为 self.size x self.size 的零矩阵
        m = zeros(self.size)
        # 获取当前置换的数组形式
        perm = self.array_form
        # 遍历矩阵的行
        for i in range(m.rows):
            # 遍历当前行之后的列
            for j in range(i + 1, m.cols):
                # 在矩阵中标记出当前排列中 perm[i] 在 perm[j] 之前的关系
                m[perm[i], perm[j]] = 1
        # 返回生成的先行关系矩阵
        return m

    def get_precedence_distance(self, other):
        """
        Computes the precedence distance between two permutations.

        Explanation
        ===========

        Suppose p and p' represent n jobs. The precedence metric
        counts the number of times a job j is preceded by job i
        in both p and p'. This metric is commutative.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([2, 0, 4, 3, 1])
        >>> q = Permutation([3, 1, 2, 4, 0])
        >>> p.get_precedence_distance(q)
        7
        >>> q.get_precedence_distance(p)
        7

        See Also
        ========

        get_precedence_matrix, get_adjacency_matrix, get_adjacency_distance
        """
        # 检查两个置换是否具有相同的大小
        if self.size != other.size:
            raise ValueError("The permutations must be of equal size.")
        # 获取当前置换和其他置换的先行关系矩阵
        self_prec_mat = self.get_precedence_matrix()
        other_prec_mat = other.get_precedence_matrix()
        # 初始化先行距离为0
        n_prec = 0
        # 遍历矩阵的行和列
        for i in range(self.size):
            for j in range(self.size):
                # 排除对角线元素，因为它们不表示先行关系
                if i == j:
                    continue
                # 如果当前位置的元素在两个矩阵中都为1，增加先行关系计数
                if self_prec_mat[i, j] * other_prec_mat[i, j] == 1:
                    n_prec += 1
        # 根据先行关系的数量计算先行距离
        d = self.size * (self.size - 1)//2 - n_prec
        # 返回计算得到的先行距离
        return d
    def get_adjacency_matrix(self):
        """
        Computes the adjacency matrix of a permutation.

        Explanation
        ===========

        If job i is adjacent to job j in a permutation p
        then we set m[i, j] = 1 where m is the adjacency
        matrix of p.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation.josephus(3, 6, 1)
        >>> p.get_adjacency_matrix()
        Matrix([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]])
        >>> q = Permutation([0, 1, 2, 3])
        >>> q.get_adjacency_matrix()
        Matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]])

        See Also
        ========

        get_precedence_matrix, get_precedence_distance, get_adjacency_distance
        """
        # 初始化大小为 self.size 的零矩阵
        m = zeros(self.size)
        # 获取排列的数组形式
        perm = self.array_form
        # 遍历排列，设置邻接矩阵中相邻位置为1
        for i in range(self.size - 1):
            m[perm[i], perm[i + 1]] = 1
        # 返回邻接矩阵
        return m

    def get_adjacency_distance(self, other):
        """
        Computes the adjacency distance between two permutations.

        Explanation
        ===========

        This metric counts the number of times a pair i,j of jobs is
        adjacent in both p and p'. If n_adj is this quantity then
        the adjacency distance is n - n_adj - 1 [1]

        [1] Reeves, Colin R. Landscapes, Operators and Heuristic search, Annals
        of Operational Research, 86, pp 473-490. (1999)


        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 3, 1, 2, 4])
        >>> q = Permutation.josephus(4, 5, 2)
        >>> p.get_adjacency_distance(q)
        3
        >>> r = Permutation([0, 2, 1, 4, 3])
        >>> p.get_adjacency_distance(r)
        4

        See Also
        ========

        get_precedence_matrix, get_precedence_distance, get_adjacency_matrix
        """
        # 检查两个排列是否大小相同
        if self.size != other.size:
            raise ValueError("The permutations must be of the same size.")
        # 获取自身和另一个排列的邻接矩阵
        self_adj_mat = self.get_adjacency_matrix()
        other_adj_mat = other.get_adjacency_matrix()
        n_adj = 0
        # 计算在两个邻接矩阵中相邻位置相同的次数
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    continue
                if self_adj_mat[i, j] * other_adj_mat[i, j] == 1:
                    n_adj += 1
        # 计算邻接距离
        d = self.size - n_adj - 1
        return d
    def get_positional_distance(self, other):
        """
        Computes the positional distance between two permutations.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 3, 1, 2, 4])
        >>> q = Permutation.josephus(4, 5, 2)
        >>> r = Permutation([3, 1, 4, 0, 2])
        >>> p.get_positional_distance(q)
        12
        >>> p.get_positional_distance(r)
        12

        See Also
        ========

        get_precedence_distance, get_adjacency_distance
        """
        # 将自身和另一个排列转换为数组形式
        a = self.array_form
        b = other.array_form
        # 检查排列长度是否相同，如果不同则抛出异常
        if len(a) != len(b):
            raise ValueError("The permutations must be of the same size.")
        # 计算两个排列之间的位置距离，即各对应位置元素差的绝对值之和
        return sum(abs(a[i] - b[i]) for i in range(len(a)))

    @classmethod
    def josephus(cls, m, n, s=1):
        """Return as a permutation the shuffling of range(n) using the Josephus
        scheme in which every m-th item is selected until all have been chosen.
        The returned permutation has elements listed by the order in which they
        were selected.

        The parameter ``s`` stops the selection process when there are ``s``
        items remaining and these are selected by continuing the selection,
        counting by 1 rather than by ``m``.

        Consider selecting every 3rd item from 6 until only 2 remain::

            choices    chosen
            ========   ======
              012345
              01 345   2
              01 34    25
              01  4    253
              0   4    2531
              0        25314
                       253140

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation.josephus(3, 6, 2).array_form
        [2, 5, 3, 1, 4, 0]

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Flavius_Josephus
        .. [2] https://en.wikipedia.org/wiki/Josephus_problem
        .. [3] https://web.archive.org/web/20171008094331/http://www.wou.edu/~burtonl/josephus.html

        """
        # 导入双端队列数据结构
        from collections import deque
        # 对 m 进行修正，以符合内部计算逻辑
        m -= 1
        # 初始化双端队列 Q，包含从 0 到 n-1 的整数
        Q = deque(list(range(n)))
        # 初始化排列 perm 为空列表
        perm = []
        # 当队列 Q 的长度大于 s 和 1 中的最大值时，执行以下操作
        while len(Q) > max(s, 1):
            # 将队列 Q 中的元素按照 m 的间隔循环移动，并将移除的元素添加到 perm 中
            for dp in range(m):
                Q.append(Q.popleft())
            perm.append(Q.popleft())
        # 将队列 Q 剩余的元素添加到 perm 中
        perm.extend(list(Q))
        # 返回以 perm 为参数创建的 Permutation 类对象
        return cls(perm)

    @classmethod
    def from_inversion_vector(cls, inversion):
        """
        Calculates the permutation from the inversion vector.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> Permutation.from_inversion_vector([3, 2, 1, 0, 0])
        Permutation([3, 2, 1, 0, 4, 5])

        """
        size = len(inversion)  # 获取反序向量的长度
        N = list(range(size + 1))  # 创建包含0到size的列表
        perm = []  # 初始化置换列表
        try:
            for k in range(size):
                val = N[inversion[k]]  # 根据反序向量中的值获取对应的N中的元素
                perm.append(val)  # 将该值添加到置换列表中
                N.remove(val)  # 移除已经使用过的值
        except IndexError:
            raise ValueError("The inversion vector is not valid.")  # 处理索引错误的异常
        perm.extend(N)  # 将剩余的N中的元素添加到置换列表的末尾
        return cls._af_new(perm)  # 使用新的置换列表创建并返回一个Permutation对象

    @classmethod
    def random(cls, n):
        """
        Generates a random permutation of length ``n``.

        Uses the underlying Python pseudo-random number generator.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation.random(2) in (Permutation([1, 0]), Permutation([0, 1]))
        True

        """
        perm_array = list(range(n))  # 创建包含0到n-1的列表
        random.shuffle(perm_array)  # 打乱列表中元素的顺序
        return cls._af_new(perm_array)  # 使用打乱后的列表创建并返回一个Permutation对象

    @classmethod
    def unrank_lex(cls, size, rank):
        """
        Lexicographic permutation unranking.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> a = Permutation.unrank_lex(5, 10)
        >>> a.rank()
        10
        >>> a
        Permutation([0, 2, 4, 1, 3])

        See Also
        ========

        rank, next_lex
        """
        perm_array = [0] * size  # 创建一个大小为size的全0列表
        psize = 1  # 初始化排列大小
        for i in range(size):
            new_psize = psize * (i + 1)  # 更新排列大小
            d = (rank % new_psize) // psize  # 计算当前位置的值
            rank -= d * psize  # 更新排名
            perm_array[size - i - 1] = d  # 设置当前位置的值
            for j in range(size - i, size):
                if perm_array[j] > d - 1:
                    perm_array[j] += 1  # 调整后续位置的值
            psize = new_psize  # 更新排列大小
        return cls._af_new(perm_array)  # 使用生成的排列数组创建并返回一个Permutation对象
    # 调整置换对象的大小到新的大小 ``n``。
    def resize(self, n):
        """Resize the permutation to the new size ``n``.

        Parameters
        ==========

        n : int
            The new size of the permutation.

        Raises
        ======

        ValueError
            If the permutation cannot be resized to the given size.
            This may only happen when resized to a smaller size than
            the original.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation

        Increasing the size of a permutation:

        >>> p = Permutation(0, 1, 2)
        >>> p = p.resize(5)
        >>> p
        (4)(0 1 2)

        Decreasing the size of the permutation:

        >>> p = p.resize(4)
        >>> p
        (3)(0 1 2)

        If resizing to the specific size breaks the cycles:

        >>> p.resize(2)
        Traceback (most recent call last):
        ...
        ValueError: The permutation cannot be resized to 2 because the
        cycle (0, 1, 2) may break.
        """
        aform = self.array_form
        l = len(aform)
        # 如果目标大小大于当前大小，扩展数组形式的置换
        if n > l:
            aform += list(range(l, n))
            return Permutation._af_new(aform)

        # 如果目标大小小于当前大小，检查循环形式的置换
        elif n < l:
            cyclic_form = self.full_cyclic_form
            new_cyclic_form = []
            for cycle in cyclic_form:
                cycle_min = min(cycle)
                cycle_max = max(cycle)
                if cycle_min <= n-1:
                    if cycle_max > n-1:
                        # 如果某个循环的最大元素超过目标大小减一，抛出异常
                        raise ValueError(
                            "The permutation cannot be resized to {} "
                            "because the cycle {} may break."
                            .format(n, tuple(cycle)))

                    new_cyclic_form.append(cycle)
            return Permutation(new_cyclic_form)

        # 如果目标大小等于当前大小，返回当前置换对象
        return self

    # XXX 已弃用的标记
    print_cyclic = None
def _merge(arr, temp, left, mid, right):
    """
    Merges two sorted arrays and calculates the inversion count.

    Helper function for calculating inversions. This method is
    for internal use only.

    Parameters:
    arr : list
        The input array to be merged.
    temp : list
        Temporary storage for merged results.
    left : int
        Starting index of the left subarray.
    mid : int
        Ending index of the left subarray and starting index of the right subarray.
    right : int
        Ending index of the right subarray.

    Returns:
    int
        The inversion count during the merge process.
    """
    i = k = left
    j = mid
    inv_count = 0
    while i < mid and j <= right:
        if arr[i] < arr[j]:
            temp[k] = arr[i]
            k += 1
            i += 1
        else:
            temp[k] = arr[j]
            k += 1
            j += 1
            inv_count += (mid - i)
    while i < mid:
        temp[k] = arr[i]
        k += 1
        i += 1
    if j <= right:
        k += right - j + 1
        j += right - j + 1
        arr[left:k + 1] = temp[left:k + 1]
    else:
        arr[left:right + 1] = temp[left:right + 1]
    return inv_count

# Define alias for Permutation class
Perm = Permutation
# Alias for private method _af_new from Perm
_af_new = Perm._af_new

# Class representing a symbolic permutation applied to a variable
class AppliedPermutation(Expr):
    """
    A permutation applied to a symbolic variable.

    Parameters
    ==========
    perm : Permutation
        The permutation to be applied.
    x : Expr
        The symbolic variable to which the permutation is applied.

    Examples
    ========
    >>> from sympy import Symbol
    >>> from sympy.combinatorics import Permutation

    Creating a symbolic permutation function application:

    >>> x = Symbol('x')
    >>> p = Permutation(0, 1, 2)
    >>> p.apply(x)
    AppliedPermutation((0 1 2), x)
    >>> _.subs(x, 1)
    2
    """
    def __new__(cls, perm, x, evaluate=None):
        # Determine if evaluation of the permutation should occur
        if evaluate is None:
            evaluate = global_parameters.evaluate

        # Ensure perm and x are sympified (converted to symbolic objects)
        perm = _sympify(perm)
        x = _sympify(x)

        # Check if perm is a Permutation object
        if not isinstance(perm, Permutation):
            raise ValueError("{} must be a Permutation instance."
                .format(perm))

        # If evaluation is enabled and x is an integer, return the result of applying perm to x
        if evaluate:
            if x.is_Integer:
                return perm.apply(x)

        # Create and return an instance of AppliedPermutation
        obj = super().__new__(cls, perm, x)
        return obj

# Define a specialized dispatch function for comparing two Permutation objects for equality
@dispatch(Permutation, Permutation)
def _eval_is_eq(lhs, rhs):
    # Check if the size of lhs and rhs permutations match
    if lhs._size != rhs._size:
        return None
    # Compare the array forms of lhs and rhs permutations for equality
    return lhs._array_form == rhs._array_form
```