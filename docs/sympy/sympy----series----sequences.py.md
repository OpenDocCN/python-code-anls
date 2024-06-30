# `D:\src\scipysrc\sympy\sympy\series\sequences.py`

```
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import call_highest_priority
from sympy.core.parameters import global_parameters
from sympy.core.function import AppliedUndef, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S, Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.core.sympify import sympify
from sympy.matrices import Matrix
from sympy.polys import lcm, factor
from sympy.sets.sets import Interval, Intersection
from sympy.tensor.indexed import Idx
from sympy.utilities.iterables import flatten, is_sequence, iterable


###############################################################################
#                            SEQUENCES                                        #
###############################################################################


class SeqBase(Basic):
    """Base class for sequences"""

    is_commutative = True  # 设置属性 is_commutative 为 True，表示对象可以进行交换
    _op_priority = 15  # 设置操作优先级为 15

    @staticmethod
    def _start_key(expr):
        """Return start (if possible) else S.Infinity.

        adapted from Set._infimum_key
        """
        try:
            start = expr.start  # 尝试获取 expr 对象的起始点
        except NotImplementedError:
            start = S.Infinity  # 如果获取失败，返回无穷大
        return start  # 返回起始点或无穷大

    def _intersect_interval(self, other):
        """Returns start and stop.

        Takes intersection over the two intervals.
        """
        interval = Intersection(self.interval, other.interval)  # 计算两个区间的交集
        return interval.inf, interval.sup  # 返回交集的最小值和最大值

    @property
    def gen(self):
        """Returns the generator for the sequence"""
        raise NotImplementedError("(%s).gen" % self)  # 返回序列的生成器，但未实现具体方法

    @property
    def interval(self):
        """The interval on which the sequence is defined"""
        raise NotImplementedError("(%s).interval" % self)  # 返回序列定义的区间，但未实现具体方法

    @property
    def start(self):
        """The starting point of the sequence. This point is included"""
        raise NotImplementedError("(%s).start" % self)  # 返回序列的起始点，但未实现具体方法

    @property
    def stop(self):
        """The ending point of the sequence. This point is included"""
        raise NotImplementedError("(%s).stop" % self)  # 返回序列的结束点，但未实现具体方法

    @property
    def length(self):
        """Length of the sequence"""
        raise NotImplementedError("(%s).length" % self)  # 返回序列的长度，但未实现具体方法

    @property
    def variables(self):
        """Returns a tuple of variables that are bounded"""
        return ()  # 返回一个空元组，表示没有被约束的变量

    @property
    ```
    # 返回对象中的符号集合，不包括那些具有特定值（即虚拟符号）
    def free_symbols(self):
        return ({j for i in self.args for j in i.free_symbols
                   .difference(self.variables)})

    # 返回指定点 pt 处的系数
    @cacheit
    def coeff(self, pt):
        if pt < self.start or pt > self.stop:
            # 如果 pt 超出范围则抛出索引错误
            raise IndexError("Index %s out of bounds %s" % (pt, self.interval))
        # 调用 _eval_coeff 方法计算并返回系数
        return self._eval_coeff(pt)

    # 抽象方法，应该被子类实现以返回指定点的系数
    def _eval_coeff(self, pt):
        raise NotImplementedError("The _eval_coeff method should be added to"
                                  "%s to return coefficient so it is available"
                                  "when coeff calls it."
                                  % self.func)

    # 返回第 i 个点的值
    def _ith_point(self, i):
        """
        如果起始点是负无穷，则从末尾返回点。
        假设第一个点的索引为零。
        """
        if self.start is S.NegativeInfinity:
            initial = self.stop
        else:
            initial = self.start

        if self.start is S.NegativeInfinity:
            step = -1
        else:
            step = 1

        # 返回第 i 个点的值
        return initial + i*step

    # 内部使用的方法，用于返回两个序列的逐项相加结果
    def _add(self, other):
        """
        只应在内部使用。

        self._add(other) 如果 self 知道如何与 other 相加，则返回一个新的逐项相加的序列，
        否则返回 None。

        other 应该是一个序列对象。

        仅在 SeqAdd 类中使用。
        """
        return None

    # 内部使用的方法，用于返回两个序列的逐项相乘结果
    def _mul(self, other):
        """
        只应在内部使用。

        self._mul(other) 如果 self 知道如何与 other 相乘，则返回一个新的逐项相乘的序列，
        否则返回 None。

        other 应该是一个序列对象。

        仅在 SeqMul 类中使用。
        """
        return None
    def coeff_mul(self, other):
        """
        当 ``other`` 不是序列时使用。应该定义以定义自定义行为。

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2).coeff_mul(2)
        SeqFormula(2*n**2, (n, 0, oo))

        Notes
        =====

        '*' 仅定义序列与序列之间的乘法。
        """
        return Mul(self, other)

    def __add__(self, other):
        """返回 'self' 和 'other' 逐项相加的结果。

        ``other`` 应该是一个序列。

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) + SeqFormula(n**3)
        SeqFormula(n**3 + n**2, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot add sequence and %s' % type(other))
        return SeqAdd(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """返回 ``self`` 和 ``other`` 逐项相减的结果。

        ``other`` 应该是一个序列。

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) - (SeqFormula(n))
        SeqFormula(n**2 - n, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot subtract sequence and %s' % type(other))
        return SeqAdd(self, -other)

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        """对序列进行取反操作。

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> -SeqFormula(n**2)
        SeqFormula(-n**2, (n, 0, oo))
        """
        return self.coeff_mul(-1)

    def __mul__(self, other):
        """返回 'self' 和 'other' 逐项相乘的结果。

        ``other`` 应该是一个序列。若 ``other`` 不是序列，参见 :func:`coeff_mul` 方法。

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) * (SeqFormula(n))
        SeqFormula(n**3, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot multiply sequence and %s' % type(other))
        return SeqMul(self, other)

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self * other

    def __iter__(self):
        for i in range(self.length):
            pt = self._ith_point(i)
            yield self.coeff(pt)
    # 定义特殊方法 __getitem__，用于支持对象的索引访问
    def __getitem__(self, index):
        # 如果索引是整数，将其转换为对应数据点的索引，然后返回对应系数
        if isinstance(index, int):
            index = self._ith_point(index)
            return self.coeff(index)
        # 如果索引是切片对象
        elif isinstance(index, slice):
            # 获取切片的起始和结束位置
            start, stop = index.start, index.stop
            # 如果起始位置为None，则设置为0
            if start is None:
                start = 0
            # 如果结束位置为None，则设置为数据的长度
            if stop is None:
                stop = self.length
            # 返回从起始到结束位置的系数列表，步长为切片对象的步长（如果有的话），默认为1
            return [self.coeff(self._ith_point(i)) for i in range(start, stop, index.step or 1)]
    def find_linear_recurrence(self,n,d=None,gfvar=None):
        r"""
        找到满足前 n 项的最短线性递推关系，其阶数 `\leq` ``n/2`` 如果可能的话。
        如果指定了 ``d``，则找到阶数 `\leq` min(d, n/2) 的最短线性递推关系，如果可能的话。
        返回系数列表 ``[b(1), b(2), ...]``，对应于递推关系 ``x(n) = b(1)*x(n-1) + b(2)*x(n-2) + ...``
        如果找不到递推关系，则返回 ``[]``。
        如果指定了 gfvar，则返回普通生成函数作为关于 gfvar 的函数。

        Examples
        ========

        >>> from sympy import sequence, sqrt, oo, lucas
        >>> from sympy.abc import n, x, y
        >>> sequence(n**2).find_linear_recurrence(10, 2)
        []
        >>> sequence(n**2).find_linear_recurrence(10)
        [3, -3, 1]
        >>> sequence(2**n).find_linear_recurrence(10)
        [2]
        >>> sequence(23*n**4+91*n**2).find_linear_recurrence(10)
        [5, -10, 10, -5, 1]
        >>> sequence(sqrt(5)*(((1 + sqrt(5))/2)**n - (-(1 + sqrt(5))/2)**(-n))/5).find_linear_recurrence(10)
        [1, 1]
        >>> sequence(x+y*(-2)**(-n), (n, 0, oo)).find_linear_recurrence(30)
        [1/2, 1/2]
        >>> sequence(3*5**n + 12).find_linear_recurrence(20,gfvar=x)
        ([6, -5], 3*(5 - 21*x)/((x - 1)*(5*x - 1)))
        >>> sequence(lucas(n)).find_linear_recurrence(15,gfvar=x)
        ([1, 1], (x - 2)/(x**2 + x - 1))
        """
        from sympy.simplify import simplify
        # 将前 n 项的表达式逐个简化并展开，存储在列表 x 中
        x = [simplify(expand(t)) for t in self[:n]]
        # 获取列表 x 的长度
        lx = len(x)
        # 计算递推关系的最大阶数 r
        if d is None:
            r = lx//2
        else:
            r = min(d,lx//2)
        coeffs = []  # 初始化系数列表
        # 从阶数 1 到 r 遍历可能的递推关系
        for l in range(1, r+1):
            l2 = 2*l
            mlist = []
            # 构造矩阵 m，从列表 x 中提取子列表，用于求解递推关系
            for k in range(l):
                mlist.append(x[k:k+l])
            m = Matrix(mlist)
            # 判断矩阵 m 的行列式是否为非零，如果是，则可以求解递推关系
            if m.det() != 0:
                y = simplify(m.LUsolve(Matrix(x[l:l2])))
                # 如果 lx 等于 l2，说明所有项均已求解，直接获取系数并结束
                if lx == l2:
                    coeffs = flatten(y[::-1])
                    break
                mlist = []
                # 继续构造矩阵 m，用于检验递推关系是否正确
                for k in range(l,lx-l):
                    mlist.append(x[k:k+l])
                m = Matrix(mlist)
                # 如果通过检验，获取系数并结束
                if m*y == Matrix(x[l2:]):
                    coeffs = flatten(y[::-1])
                    break
        # 如果没有指定 gfvar，则返回系数列表
        if gfvar is None:
            return coeffs
        else:
            l = len(coeffs)
            # 如果系数列表为空，返回空列表和 None
            if l == 0:
                return [], None
            else:
                # 计算普通生成函数
                n, d = x[l-1]*gfvar**(l-1), 1 - coeffs[l-1]*gfvar**l
                for i in range(l-1):
                    n += x[i]*gfvar**i
                    for j in range(l-i-1):
                        n -= coeffs[i]*x[j]*gfvar**(i+j+1)
                    d -= coeffs[i]*gfvar**(i+1)
                return coeffs, simplify(factor(n)/factor(d))
class EmptySequence(SeqBase, metaclass=Singleton):
    """Represents an empty sequence.

    The empty sequence is also available as a singleton as
    ``S.EmptySequence``.

    Examples
    ========

    >>> from sympy import EmptySequence, SeqPer
    >>> from sympy.abc import x
    >>> EmptySequence
    EmptySequence
    >>> SeqPer((1, 2), (x, 0, 10)) + EmptySequence
    SeqPer((1, 2), (x, 0, 10))
    >>> SeqPer((1, 2)) * EmptySequence
    EmptySequence
    >>> EmptySequence.coeff_mul(-1)
    EmptySequence
    """

    @property
    def interval(self):
        # 返回空集合作为空序列的区间
        return S.EmptySet

    @property
    def length(self):
        # 返回零作为空序列的长度
        return S.Zero

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        # 系数乘法对空序列返回自身
        return self

    def __iter__(self):
        # 返回一个空迭代器，表示空序列无元素可迭代
        return iter([])


class SeqExpr(SeqBase):
    """Sequence expression class.

    Various sequences should inherit from this class.

    Examples
    ========

    >>> from sympy.series.sequences import SeqExpr
    >>> from sympy.abc import x
    >>> from sympy import Tuple
    >>> s = SeqExpr(Tuple(1, 2, 3), Tuple(x, 0, 10))
    >>> s.gen
    (1, 2, 3)
    >>> s.interval
    Interval(0, 10)
    >>> s.length
    11

    See Also
    ========

    sympy.series.sequences.SeqPer
    sympy.series.sequences.SeqFormula
    """

    @property
    def gen(self):
        # 返回序列的生成器，即序列的第一个参数
        return self.args[0]

    @property
    def interval(self):
        # 根据参数返回序列的区间
        return Interval(self.args[1][1], self.args[1][2])

    @property
    def start(self):
        # 返回序列区间的下界，即区间的起始值
        return self.interval.inf

    @property
    def stop(self):
        # 返回序列区间的上界，即区间的结束值
        return self.interval.sup

    @property
    def length(self):
        # 返回序列的长度，计算为区间的长度加一
        return self.stop - self.start + 1

    @property
    def variables(self):
        # 返回序列中的变量，即区间的第一个参数
        return (self.args[1][0],)


class SeqPer(SeqExpr):
    """
    Represents a periodic sequence.

    The elements are repeated after a given period.

    Examples
    ========

    >>> from sympy import SeqPer, oo
    >>> from sympy.abc import k

    >>> s = SeqPer((1, 2, 3), (0, 5))
    >>> s.periodical
    (1, 2, 3)
    >>> s.period
    3

    For value at a particular point

    >>> s.coeff(3)
    1

    supports slicing

    >>> s[:]
    [1, 2, 3, 1, 2, 3]

    iterable

    >>> list(s)
    [1, 2, 3, 1, 2, 3]

    sequence starts from negative infinity

    >>> SeqPer((1, 2, 3), (-oo, 0))[0:6]
    [1, 2, 3, 1, 2, 3]

    Periodic formulas

    >>> SeqPer((k, k**2, k**3), (k, 0, oo))[0:6]
    [0, 1, 8, 3, 16, 125]

    See Also
    ========

    sympy.series.sequences.SeqFormula
    """
    def __new__(cls, periodical, limits=None):
        # 定义一个新的实例构造方法，接受周期序列和可选的限制条件

        periodical = sympify(periodical)
        # 将周期序列转换为符号表达式

        def _find_x(periodical):
            # 内部函数：查找周期序列中的自由变量
            free = periodical.free_symbols
            if len(periodical.free_symbols) == 1:
                return free.pop()
            else:
                return Dummy('k')

        x, start, stop = None, None, None
        # 初始化变量 x, start, stop

        if limits is None:
            # 如果没有给定限制条件
            x, start, stop = _find_x(periodical), 0, S.Infinity
            # 从周期序列中找到自由变量，并设置默认的起始和结束限制

        if is_sequence(limits, Tuple):
            # 如果给定的限制条件是一个元组序列
            if len(limits) == 3:
                x, start, stop = limits
                # 设置自定义的起始和结束限制
            elif len(limits) == 2:
                x = _find_x(periodical)
                start, stop = limits
                # 设置自定义的起始和结束限制

        if not isinstance(x, (Symbol, Idx)) or start is None or stop is None:
            # 如果 x 不是符号或索引对象，或者起始和结束限制条件任意一个未设置
            raise ValueError('Invalid limits given: %s' % str(limits))
            # 抛出值错误异常，说明给定的限制条件无效

        if start is S.NegativeInfinity and stop is S.Infinity:
                raise ValueError("Both the start and end value"
                                 "cannot be unbounded")
                # 如果起始和结束值均为无穷，则抛出值错误异常，说明无法同时为无界

        limits = sympify((x, start, stop))
        # 将限制条件转换为符号表达式

        if is_sequence(periodical, Tuple):
            periodical = sympify(tuple(flatten(periodical)))
            # 如果周期序列是元组序列，则展平后转换为符号表达式
        else:
            raise ValueError("invalid period %s should be something "
                             "like e.g (1, 2) " % periodical)
            # 否则，抛出值错误异常，说明周期序列无效，应该类似于 (1, 2)

        if Interval(limits[1], limits[2]) is S.EmptySet:
            return S.EmptySequence
            # 如果限制条件的区间为空集，返回空序列

        return Basic.__new__(cls, periodical, limits)
        # 调用基类的构造方法创建新的实例

    @property
    def period(self):
        # 定义一个属性方法，返回周期长度
        return len(self.gen)

    @property
    def periodical(self):
        # 定义一个属性方法，返回周期序列
        return self.gen

    def _eval_coeff(self, pt):
        # 内部方法：计算给定点的系数
        if self.start is S.NegativeInfinity:
            idx = (self.stop - pt) % self.period
            # 如果起始值为负无穷，计算索引为 (结束值 - 点) % 周期长度
        else:
            idx = (pt - self.start) % self.period
            # 否则，计算索引为 (点 - 起始值) % 周期长度
        return self.periodical[idx].subs(self.variables[0], pt)
        # 返回周期序列中对应索引的元素，替换变量后的结果

    def _add(self, other):
        """See docstring of SeqBase._add"""
        # 内部方法：查看 SeqBase._add 的文档字符串
        if isinstance(other, SeqPer):
            # 如果参数是 SeqPer 类的实例
            per1, lper1 = self.periodical, self.period
            per2, lper2 = other.periodical, other.period

            per_length = lcm(lper1, lper2)
            # 计算两个周期长度的最小公倍数

            new_per = []
            for x in range(per_length):
                ele1 = per1[x % lper1]
                ele2 = per2[x % lper2]
                new_per.append(ele1 + ele2)
                # 对应位置元素相加，生成新的周期序列

            start, stop = self._intersect_interval(other)
            # 计算与另一个对象的交集区间

            return SeqPer(new_per, (self.variables[0], start, stop))
            # 返回一个新的 SeqPer 对象，传入新的周期序列和交集区间

    def _mul(self, other):
        """See docstring of SeqBase._mul"""
        # 内部方法：查看 SeqBase._mul 的文档字符串
        if isinstance(other, SeqPer):
            # 如果参数是 SeqPer 类的实例
            per1, lper1 = self.periodical, self.period
            per2, lper2 = other.periodical, other.period

            per_length = lcm(lper1, lper2)
            # 计算两个周期长度的最小公倍数

            new_per = []
            for x in range(per_length):
                ele1 = per1[x % lper1]
                ele2 = per2[x % lper2]
                new_per.append(ele1 * ele2)
                # 对应位置元素相乘，生成新的周期序列

            start, stop = self._intersect_interval(other)
            # 计算与另一个对象的交集区间

            return SeqPer(new_per, (self.variables[0], start, stop))
            # 返回一个新的 SeqPer 对象，传入新的周期序列和交集区间
    def coeff_mul(self, coeff):
        """对序列中的每个元素乘以给定的系数，并返回新的周期序列对象。

        Args:
            coeff: 用于乘以序列每个元素的系数。

        Returns:
            SeqPer: 返回一个新的周期序列对象，其中每个元素都乘以给定的系数后的结果。

        Raises:
            TypeError: 如果无法将 coeff 转换为 sympy 对象时抛出异常。

        """
        # 将 coeff 转换为 sympy 对象
        coeff = sympify(coeff)
        # 对周期序列中的每个元素乘以 coeff，并存储结果在 per 列表中
        per = [x * coeff for x in self.periodical]
        # 返回一个新的 SeqPer 对象，将 per 列表作为周期序列传入，并保持原始序列的第二个参数不变
        return SeqPer(per, self.args[1])
class SeqFormula(SeqExpr):
    """
    表示基于公式的序列。

    元素是使用公式生成的。

    Examples
    ========

    >>> from sympy import SeqFormula, oo, Symbol
    >>> n = Symbol('n')
    >>> s = SeqFormula(n**2, (n, 0, 5))
    >>> s.formula
    n**2

    对特定点的值

    >>> s.coeff(3)
    9

    支持切片

    >>> s[:]
    [0, 1, 4, 9, 16, 25]

    可迭代

    >>> list(s)
    [0, 1, 4, 9, 16, 25]

    序列从负无穷开始

    >>> SeqFormula(n**2, (-oo, 0))[0:6]
    [0, 1, 4, 9, 16, 25]

    See Also
    ========

    sympy.series.sequences.SeqPer
    """

    def __new__(cls, formula, limits=None):
        # 将公式转换为Sympy表达式
        formula = sympify(formula)

        def _find_x(formula):
            # 查找公式中的自由变量
            free = formula.free_symbols
            if len(free) == 1:
                return free.pop()
            elif not free:
                return Dummy('k')
            else:
                raise ValueError(
                    " specify dummy variables for %s. If the formula contains"
                    " more than one free symbol, a dummy variable should be"
                    " supplied explicitly e.g., SeqFormula(m*n**2, (n, 0, 5))"
                    % formula)

        x, start, stop = None, None, None
        # 如果未指定限制条件，则自动查找变量x，并设定默认起止值
        if limits is None:
            x, start, stop = _find_x(formula), 0, S.Infinity
        # 如果限制条件为元组，则解析其内容
        if is_sequence(limits, Tuple):
            if len(limits) == 3:
                x, start, stop = limits
            elif len(limits) == 2:
                x = _find_x(formula)
                start, stop = limits

        # 检查变量x是否为Symbol或Idx，以及起止值是否有效
        if not isinstance(x, (Symbol, Idx)) or start is None or stop is None:
            raise ValueError('Invalid limits given: %s' % str(limits))

        # 如果起始值为负无穷，结束值为正无穷，则抛出异常
        if start is S.NegativeInfinity and stop is S.Infinity:
                raise ValueError("Both the start and end value "
                                 "cannot be unbounded")
        limits = sympify((x, start, stop))

        # 如果限制条件表示的区间为空集，则返回空序列
        if Interval(limits[1], limits[2]) is S.EmptySet:
            return S.EmptySequence

        # 调用父类构造函数生成对象
        return Basic.__new__(cls, formula, limits)

    @property
    def formula(self):
        return self.gen

    def _eval_coeff(self, pt):
        # 获取变量并用特定点求解公式值
        d = self.variables[0]
        return self.formula.subs(d, pt)

    def _add(self, other):
        """See docstring of SeqBase._add"""
        # 如果other是SeqFormula类型，则进行公式的加法操作
        if isinstance(other, SeqFormula):
            form1, v1 = self.formula, self.variables[0]
            form2, v2 = other.formula, other.variables[0]
            formula = form1 + form2.subs(v2, v1)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (v1, start, stop))
    # 定义一个方法 `_mul`，用于计算序列乘法
    def _mul(self, other):
        """See docstring of SeqBase._mul"""
        # 如果 `other` 是 SeqFormula 类的实例
        if isinstance(other, SeqFormula):
            # 获取当前对象和 `other` 对象的公式和变量
            form1, v1 = self.formula, self.variables[0]
            form2, v2 = other.formula, other.variables[0]
            # 计算两个公式的乘积，并用 `v2` 替换 `v1` 后的结果
            formula = form1 * form2.subs(v2, v1)
            # 获取当前对象和 `other` 对象的交集区间
            start, stop = self._intersect_interval(other)
            # 返回一个新的 SeqFormula 对象，其公式为 `formula`，变量范围为 `(v1, start, stop)`
            return SeqFormula(formula, (v1, start, stop))

    # 定义一个方法 `coeff_mul`，用于对序列的公式乘以一个系数
    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        # 将 `coeff` 转换为 sympy 的表达式
        coeff = sympify(coeff)
        # 计算当前对象的公式乘以 `coeff` 后的结果
        formula = self.formula * coeff
        # 返回一个新的 SeqFormula 对象，其公式为 `formula`，变量范围与当前对象相同
        return SeqFormula(formula, self.args[1])

    # 定义一个方法 `expand`，用于对序列的公式进行展开
    def expand(self, *args, **kwargs):
        # 使用 sympy 的 `expand` 函数对当前对象的公式进行展开，支持额外的参数和关键字参数
        expanded_formula = expand(self.formula, *args, **kwargs)
        # 返回一个新的 SeqFormula 对象，其公式为 `expanded_formula`，变量范围与当前对象相同
        return SeqFormula(expanded_formula, self.args[1])
class RecursiveSeq(SeqBase):
    """
    A finite degree recursive sequence.

    Explanation
    ===========
    
    That is, a sequence a(n) that depends on a fixed, finite number of its
    previous values. The general form is

        a(n) = f(a(n - 1), a(n - 2), ..., a(n - d))

    for some fixed, positive integer d, where f is some function defined by a
    SymPy expression.

    Parameters
    ==========

    recurrence : SymPy expression defining recurrence
        This is *not* an equality, only the expression that the nth term is
        equal to. For example, if :code:`a(n) = f(a(n - 1), ..., a(n - d))`,
        then the expression should be :code:`f(a(n - 1), ..., a(n - d))`.

    yn : applied undefined function
        Represents the nth term of the sequence as e.g. :code:`y(n)` where
        :code:`y` is an undefined function and `n` is the sequence index.

    n : symbolic argument
        The name of the variable that the recurrence is in, e.g., :code:`n` if
        the recurrence function is :code:`y(n)`.

    initial : iterable with length equal to the degree of the recurrence
        The initial values of the recurrence.

    start : start value of sequence (inclusive)

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy.series.sequences import RecursiveSeq
    >>> y = Function("y")
    >>> n = symbols("n")
    >>> fib = RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1])

    >>> fib.coeff(3) # Value at a particular point
    2

    >>> fib[:6] # supports slicing
    [0, 1, 1, 2, 3, 5]

    >>> fib.recurrence # inspect recurrence
    Eq(y(n), y(n - 2) + y(n - 1))

    >>> fib.degree # automatically determine degree
    2

    >>> for x in zip(range(10), fib): # supports iteration
    ...     print(x)
    (0, 0)
    (1, 1)
    (2, 1)
    (3, 2)
    (4, 3)
    (5, 5)
    (6, 8)
    (7, 13)
    (8, 21)
    (9, 34)

    See Also
    ========

    sympy.series.sequences.SeqFormula

    """
    def __new__(cls, recurrence, yn, n, initial=None, start=0):
        # 检查 yn 是否为 AppliedUndef 的实例，否则抛出类型错误
        if not isinstance(yn, AppliedUndef):
            raise TypeError("recurrence sequence must be an applied undefined function"
                            ", found `{}`".format(yn))

        # 检查 n 是否为 Basic 类型且为符号，否则抛出类型错误
        if not isinstance(n, Basic) or not n.is_symbol:
            raise TypeError("recurrence variable must be a symbol"
                            ", found `{}`".format(n))

        # 检查 yn 的参数是否仅为 n，否则抛出类型错误
        if yn.args != (n,):
            raise TypeError("recurrence sequence does not match symbol")

        # 获取 yn 的函数名
        y = yn.func

        # 创建一个 Wild 对象 k，用于匹配偏移量
        k = Wild("k", exclude=(n,))
        degree = 0

        # 查找 recurrence 中所有 y 的应用，并检查：
        #   1. 函数 y 只能用单个参数；
        #   2. 所有参数必须为 n + k，其中 k 是常数且为负整数。
        prev_ys = recurrence.find(y)
        for prev_y in prev_ys:
            if len(prev_y.args) != 1:
                raise TypeError("Recurrence should be in a single variable")

            shift = prev_y.args[0].match(n + k)[k]
            if not (shift.is_constant() and shift.is_integer and shift < 0):
                raise TypeError("Recurrence should have constant,"
                                " negative, integer shifts"
                                " (found {})".format(prev_y))

            if -shift > degree:
                degree = -shift

        # 如果没有提供 initial，则生成 degree 个 Dummy 变量作为初始值
        if not initial:
            initial = [Dummy("c_{}".format(k)) for k in range(degree)]

        # 检查初始值的数量是否等于 degree
        if len(initial) != degree:
            raise ValueError("Number of initial terms must equal degree")

        # 将 degree 和 start 转换为 Integer 和 sympify 后的对象
        degree = Integer(degree)
        start = sympify(start)

        # 将 initial 转换为 Tuple 类型
        initial = Tuple(*(sympify(x) for x in initial))

        # 创建 Basic 类的新实例 seq
        seq = Basic.__new__(cls, recurrence, yn, n, initial, start)

        # 缓存 seq 的初始值
        seq.cache = {y(start + k): init for k, init in enumerate(initial)}
        seq.degree = degree

        return seq

    @property
    def _recurrence(self):
        """Equation defining recurrence."""
        return self.args[0]

    @property
    def recurrence(self):
        """Equation defining recurrence."""
        return Eq(self.yn, self.args[0])

    @property
    def yn(self):
        """Applied function representing the nth term"""
        return self.args[1]

    @property
    def y(self):
        """Undefined function for the nth term of the sequence"""
        return self.yn.func

    @property
    def n(self):
        """Sequence index symbol"""
        return self.args[2]

    @property
    def initial(self):
        """The initial values of the sequence"""
        return self.args[3]

    @property
    def start(self):
        """The starting point of the sequence. This point is included"""
        return self.args[4]

    @property
    def stop(self):
        """The ending point of the sequence. (oo)"""
        return S.Infinity

    @property
    def interval(self):
        """Interval on which sequence is defined."""
        return (self.start, S.Infinity)
    # 根据给定的索引计算系数值，如果缓存中已有相应值则直接返回，否则计算并缓存新值
    def _eval_coeff(self, index):
        # 如果索引减去起始值小于缓存长度，则直接返回缓存中的值
        if index - self.start < len(self.cache):
            return self.cache[self.y(index)]

        # 如果需要计算的索引超出了当前缓存的范围，开始计算新的系数值并存入缓存
        for current in range(len(self.cache), index + 1):
            # 使用 xreplace 替代 subs 来提升性能。
            # 参考问题 #10697。
            seq_index = self.start + current
            # 计算当前递推公式在给定索引处的值
            current_recurrence = self._recurrence.xreplace({self.n: seq_index})
            # 使用缓存中的已计算值来替换递推公式中的变量，计算新的项
            new_term = current_recurrence.xreplace(self.cache)

            # 将计算得到的新值存入缓存
            self.cache[self.y(seq_index)] = new_term

        # 返回所需索引处的系数值
        return self.cache[self.y(self.start + current)]

    # 实现迭代器协议，使对象可迭代
    def __iter__(self):
        # 从起始索引开始迭代
        index = self.start
        while True:
            # 使用 _eval_coeff 方法计算当前索引处的系数值，并通过 yield 返回
            yield self._eval_coeff(index)
            # 增加索引以准备计算下一个系数值
            index += 1
# 定义函数 sequence，用于生成合适的序列对象
def sequence(seq, limits=None):
    """
    Returns appropriate sequence object.

    Explanation
    ===========

    If ``seq`` is a SymPy sequence, returns :class:`SeqPer` object
    otherwise returns :class:`SeqFormula` object.

    Examples
    ========

    >>> from sympy import sequence
    >>> from sympy.abc import n
    >>> sequence(n**2, (n, 0, 5))
    SeqFormula(n**2, (n, 0, 5))
    >>> sequence((1, 2, 3), (n, 0, 5))
    SeqPer((1, 2, 3), (n, 0, 5))

    See Also
    ========

    sympy.series.sequences.SeqPer
    sympy.series.sequences.SeqFormula
    """
    # 将 seq 转换为 SymPy 对象
    seq = sympify(seq)

    # 检查 seq 是否是 Tuple 类型的序列
    if is_sequence(seq, Tuple):
        # 如果是，返回 SeqPer 对象
        return SeqPer(seq, limits)
    else:
        # 否则，返回 SeqFormula 对象
        return SeqFormula(seq, limits)


###############################################################################
#                            OPERATIONS                                       #
###############################################################################


class SeqExprOp(SeqBase):
    """
    Base class for operations on sequences.

    Examples
    ========

    >>> from sympy.series.sequences import SeqExprOp, sequence
    >>> from sympy.abc import n
    >>> s1 = sequence(n**2, (n, 0, 10))
    >>> s2 = sequence((1, 2, 3), (n, 5, 10))
    >>> s = SeqExprOp(s1, s2)
    >>> s.gen
    (n**2, (1, 2, 3))
    >>> s.interval
    Interval(5, 10)
    >>> s.length
    6

    See Also
    ========

    sympy.series.sequences.SeqAdd
    sympy.series.sequences.SeqMul
    """
    
    @property
    def gen(self):
        """Generator for the sequence.

        returns a tuple of generators of all the argument sequences.
        """
        # 返回所有参数序列的生成器的元组
        return tuple(a.gen for a in self.args)

    @property
    def interval(self):
        """Sequence is defined on the intersection
        of all the intervals of respective sequences
        """
        # 返回所有参数序列各自区间的交集
        return Intersection(*(a.interval for a in self.args))

    @property
    def start(self):
        # 返回序列的起始点
        return self.interval.inf

    @property
    def stop(self):
        # 返回序列的终止点
        return self.interval.sup

    @property
    def variables(self):
        """Cumulative of all the bound variables"""
        # 返回所有绑定变量的累积
        return tuple(flatten([a.variables for a in self.args]))

    @property
    def length(self):
        # 返回序列的长度
        return self.stop - self.start + 1


class SeqAdd(SeqExprOp):
    """Represents term-wise addition of sequences.

    Rules:
        * The interval on which sequence is defined is the intersection
          of respective intervals of sequences.
        * Anything + :class:`EmptySequence` remains unchanged.
        * Other rules are defined in ``_add`` methods of sequence classes.

    Examples
    ========

    >>> from sympy import EmptySequence, oo, SeqAdd, SeqPer, SeqFormula
    >>> from sympy.abc import n
    >>> SeqAdd(SeqPer((1, 2), (n, 0, oo)), EmptySequence)
    SeqPer((1, 2), (n, 0, oo))
    >>> SeqAdd(SeqPer((1, 2), (n, 0, 5)), SeqPer((1, 2), (n, 6, 10)))
    EmptySequence
    >>> SeqAdd(SeqPer((1, 2), (n, 0, oo)), SeqFormula(n**2, (n, 0, oo)))

    """
    # 创建一个 SeqAdd 对象，将 SeqFormula(n**2, (n, 0, oo)) 和 SeqPer((1, 2), (n, 0, oo)) 作为参数传入
    SeqAdd(SeqFormula(n**2, (n, 0, oo)), SeqPer((1, 2), (n, 0, oo)))
    # 创建一个 SeqAdd 对象，将 SeqFormula(n**3) 和 SeqFormula(n**2) 作为参数传入
    >>> SeqAdd(SeqFormula(n**3), SeqFormula(n**2))
    # 创建一个 SeqFormula 对象，表示 n**3 + n**2 的求和序列，区间为 (n, 0, oo)
    SeqFormula(n**3 + n**2, (n, 0, oo))

    # 查看以下内容的相关信息
    See Also
    ========

    sympy.series.sequences.SeqMul
    """

    # 定义一个新的类方法 __new__
    def __new__(cls, *args, **kwargs):
        # 获取关键字参数 'evaluate'，如果不存在则使用全局参数中的 evaluate
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)

        # 将位置参数 args 转换为列表
        args = list(args)

        # 从 sympy.sets.sets.Union 适配而来，用于扁平化输入参数
        def _flatten(arg):
            # 如果参数是 SeqBase 类的实例
            if isinstance(arg, SeqBase):
                # 如果是 SeqAdd 类的实例，则递归地对其参数进行扁平化并求和
                if isinstance(arg, SeqAdd):
                    return sum(map(_flatten, arg.args), [])
                else:
                    return [arg]
            # 如果参数可迭代，则递归地对其元素进行扁平化
            if iterable(arg):
                return sum(map(_flatten, arg), [])
            # 抛出类型错误，要求输入必须是 Sequence 或可迭代的 Sequence
            raise TypeError("Input must be Sequences or "
                            " iterables of Sequences")
        # 对 args 应用 _flatten 函数
        args = _flatten(args)

        # 过滤掉空的 Sequence 对象
        args = [a for a in args if a is not S.EmptySequence]

        # 如果没有有效的 Sequence 对象，返回空序列 S.EmptySequence
        if not args:
            return S.EmptySequence

        # 如果所有序列的区间的交集为空集，返回空序列 S.EmptySequence
        if Intersection(*(a.interval for a in args)) is S.EmptySet:
            return S.EmptySequence

        # 使用已知规则对 args 序列进行简化
        if evaluate:
            return SeqAdd.reduce(args)

        # 根据 _start_key 对 args 进行排序
        args = list(ordered(args, SeqBase._start_key))

        # 调用父类 Basic 的构造方法，返回新创建的 SeqAdd 对象
        return Basic.__new__(cls, *args)

    @staticmethod
    # 静态方法：使用已知规则简化 SeqAdd 的实例列表 args
    def reduce(args):
        """Simplify :class:`SeqAdd` using known rules.

        Iterates through all pairs and ask the constituent
        sequences if they can simplify themselves with any other constituent.

        Notes
        =====

        adapted from ``Union.reduce``

        """
        # 循环直到不再有新的序列可以简化
        new_args = True
        while new_args:
            for id1, s in enumerate(args):
                new_args = False
                for id2, t in enumerate(args):
                    if id1 == id2:
                        continue
                    # 尝试将序列 s 和 t 相加并得到新的序列
                    new_seq = s._add(t)
                    # 如果成功得到新序列，则更新 args
                    if new_seq is not None:
                        new_args = [a for a in args if a not in (s, t)]
                        new_args.append(new_seq)
                        break
                if new_args:
                    args = new_args
                    break

        # 如果最终 args 只剩下一个序列，则返回该序列
        if len(args) == 1:
            return args.pop()
        else:
            # 否则返回包含简化后序列的 SeqAdd 对象
            return SeqAdd(args, evaluate=False)

    # 实例方法：计算在给定点 pt 处所有序列的系数之和
    def _eval_coeff(self, pt):
        """adds up the coefficients of all the sequences at point pt"""
        return sum(a.coeff(pt) for a in self.args)
    class SeqMul(SeqExprOp):
        r"""Represents term-wise multiplication of sequences.
        
        Explanation
        ===========
        
        Handles multiplication of sequences only. For multiplication
        with other objects see :func:`SeqBase.coeff_mul`.
        
        Rules:
            * The interval on which sequence is defined is the intersection
              of respective intervals of sequences.
            * Anything \* :class:`EmptySequence` returns :class:`EmptySequence`.
            * Other rules are defined in ``_mul`` methods of sequence classes.
        
        Examples
        ========
        
        >>> from sympy import EmptySequence, oo, SeqMul, SeqPer, SeqFormula
        >>> from sympy.abc import n
        >>> SeqMul(SeqPer((1, 2), (n, 0, oo)), EmptySequence)
        EmptySequence
        >>> SeqMul(SeqPer((1, 2), (n, 0, 5)), SeqPer((1, 2), (n, 6, 10)))
        EmptySequence
        >>> SeqMul(SeqPer((1, 2), (n, 0, oo)), SeqFormula(n**2))
        SeqMul(SeqFormula(n**2, (n, 0, oo)), SeqPer((1, 2), (n, 0, oo)))
        >>> SeqMul(SeqFormula(n**3), SeqFormula(n**2))
        SeqFormula(n**5, (n, 0, oo))
        
        See Also
        ========
        
        sympy.series.sequences.SeqAdd
        """

        def __new__(cls, *args, **kwargs):
            evaluate = kwargs.get('evaluate', global_parameters.evaluate)
    
            # flatten inputs
            args = list(args)
    
            # adapted from sympy.sets.sets.Union
            def _flatten(arg):
                if isinstance(arg, SeqBase):
                    if isinstance(arg, SeqMul):
                        return sum(map(_flatten, arg.args), [])
                    else:
                        return [arg]
                elif iterable(arg):
                    return sum(map(_flatten, arg), [])
                raise TypeError("Input must be Sequences or "
                                " iterables of Sequences")
            args = _flatten(args)
    
            # Multiplication of no sequences is EmptySequence
            if not args:
                return S.EmptySequence
    
            if Intersection(*(a.interval for a in args)) is S.EmptySet:
                return S.EmptySequence
    
            # reduce using known rules
            if evaluate:
                return SeqMul.reduce(args)
    
            args = list(ordered(args, SeqBase._start_key))
    
            return Basic.__new__(cls, *args)
    
        @staticmethod


注释：
    # 函数定义，用于简化 SeqMul 对象使用已知规则
    def reduce(args):
        """Simplify a :class:`SeqMul` using known rules.

        Explanation
        ===========

        Iterates through all pairs and ask the constituent
        sequences if they can simplify themselves with any other constituent.

        Notes
        =====

        adapted from ``Union.reduce``

        """
        # 初始化标志位，表示是否发现新的简化结果
        new_args = True
        # 当仍然存在新的简化结果时循环进行简化
        while new_args:
            # 遍历所有序列对
            for id1, s in enumerate(args):
                # 假设没有找到新的简化结果
                new_args = False
                for id2, t in enumerate(args):
                    if id1 == id2:
                        continue
                    # 尝试用序列 s 和 t 进行乘法运算，返回新的乘积序列或者 None
                    new_seq = s._mul(t)
                    # 如果乘法操作成功，更新 new_args 列表
                    if new_seq is not None:
                        # 将原始序列中的 s 和 t 替换为新的乘积序列
                        new_args = [a for a in args if a not in (s, t)]
                        new_args.append(new_seq)
                        break
                # 如果发现新的简化结果，退出当前循环
                if new_args:
                    args = new_args
                    break

        # 如果简化后只剩下一个序列，直接返回该序列
        if len(args) == 1:
            return args.pop()
        else:
            # 否则返回一个新的 SeqMul 对象，包含简化后的序列列表
            return SeqMul(args, evaluate=False)

    def _eval_coeff(self, pt):
        """multiplies the coefficients of all the sequences at point pt"""
        # 初始化系数值为 1
        val = 1
        # 遍历当前对象的所有序列，依次计算它们在给定点 pt 处的系数并相乘
        for a in self.args:
            val *= a.coeff(pt)
        # 返回最终的系数值
        return val
```