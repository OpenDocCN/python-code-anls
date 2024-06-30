# `D:\src\scipysrc\sympy\sympy\functions\special\hyper.py`

```
# 导入 Counter 类，用于计数
from collections import Counter

# 导入 SymPy 核心库中的各种符号、操作和函数
from sympy.core import S, Mod  # S 表示未知符号，Mod 是模运算
from sympy.core.add import Add  # 加法表达式
from sympy.core.expr import Expr  # 表达式基类
from sympy.core.function import Function, Derivative, ArgumentIndexError  # 函数、导数、参数错误
from sympy.core.containers import Tuple  # 元组容器
from sympy.core.mul import Mul  # 乘法表达式
from sympy.core.numbers import I, pi, oo, zoo  # 虚数单位、圆周率、无穷大、无穷小
from sympy.core.parameters import global_parameters  # 全局参数
from sympy.core.relational import Ne  # 不等式
from sympy.core.sorting import default_sort_key  # 默认排序键
from sympy.core.symbol import Dummy  # 虚拟符号

# 导入 SymPy 外部依赖的函数库
from sympy.external.gmpy import lcm  # 最小公倍数函数

# 导入各种数学函数，如平方根、指数、对数、三角函数等
from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
        sinh, cosh, asinh, acosh, atanh, acoth)
from sympy.functions import factorial, RisingFactorial  # 阶乘、升阶乘函数
from sympy.functions.elementary.complexes import Abs, re, unpolarify  # 复数相关函数
from sympy.functions.elementary.exponential import exp_polar  # 极坐标指数函数
from sympy.functions.elementary.integers import ceiling  # 向上取整函数
from sympy.functions.elementary.piecewise import Piecewise  # 分段函数
from sympy.logic.boolalg import (And, Or)  # 逻辑运算符

# 从 sympy 模块中导入 ordered 函数
from sympy import ordered


class TupleArg(Tuple):
    # 这个方法仅仅是因为 hyper._eval_as_leading_term 回退到使用 Function._eval_as_leading_term，
    # 而 Function._eval_as_leading_term 又调用 hyper 的参数的 as_leading_term 方法。
    # 理想情况下，hyper 应该有一个 _eval_as_leading_term 方法来处理所有情况，
    # 这个方法应该被移除，因为元组的首项概念在这里并不合理。
    def as_leading_term(self, *x, logx=None, cdir=0):
        return TupleArg(*[f.as_leading_term(*x, logx=logx, cdir=cdir) for f in self.args])

    # 计算极限 x -> xlim
    def limit(self, x, xlim, dir='+'):
        """ Compute limit x->xlim.
        """
        from sympy.series.limits import limit
        return TupleArg(*[limit(f, x, xlim, dir) for f in self.args])


# TODO 应该让 __new__ 接受 **options 吗？
# TODO 构造函数是否应该检查参数是否合理？

def _prep_tuple(v):
    """
    将可迭代参数 *v* 转换为元组，并解极坐标化，因为超几何和梅杰函数在其参数上是无分支的。

    Examples
    ========

    >>> from sympy.functions.special.hyper import _prep_tuple
    >>> _prep_tuple([1, 2, 3])
    (1, 2, 3)
    >>> _prep_tuple((4, 5))
    (4, 5)
    >>> _prep_tuple((7, 8, 9))
    (7, 8, 9)

    """
    return TupleArg(*[unpolarify(x) for x in v])


class TupleParametersBase(Function):
    """ 处理带有元组参数的基类，包括对微分的处理。 """
    # 由于参数中包含元组，这个属性不能自动推断
    is_commutative = True
    # 定义一个方法 `_eval_derivative`，用于计算导数
    def _eval_derivative(self, s):
        # 尝试执行以下代码块，捕获可能的异常
        try:
            # 初始化结果变量为0
            res = 0
            # 如果参数列表中的第一个或第二个参数包含变量 s
            if self.args[0].has(s) or self.args[1].has(s):
                # 遍历 `_diffargs` 列表中的索引和元素对
                for i, p in enumerate(self._diffargs):
                    # 计算第 i 个参数的偏导数
                    m = self._diffargs[i].diff(s)
                    # 如果偏导数不为0，则将其乘以相应的偏导数和添加到结果中
                    if m != 0:
                        res += self.fdiff((1, i)) * m
            # 返回结果加上对第三个参数关于 s 的偏导数乘以 `self.fdiff(3)` 的值
            return res + self.fdiff(3) * self.args[2].diff(s)
        # 捕获到 `ArgumentIndexError` 或 `NotImplementedError` 异常时执行以下操作
        except (ArgumentIndexError, NotImplementedError):
            # 返回一个新的 Derivative 对象，表示对当前对象关于变量 s 的导数
            return Derivative(self, s)
class hyper(TupleParametersBase):
    r"""
    The generalized hypergeometric function is defined by a series where
    the ratios of successive terms are a rational function of the summation
    index. When convergent, it is continued analytically to the largest
    possible domain.

    Explanation
    ===========

    The hypergeometric function depends on two vectors of parameters, called
    the numerator parameters $a_p$, and the denominator parameters
    $b_q$. It also has an argument $z$. The series definition is

    .. math ::
        {}_pF_q\left(\begin{matrix} a_1, \cdots, a_p \\ b_1, \cdots, b_q \end{matrix}
                     \middle| z \right)
        = \sum_{n=0}^\infty \frac{(a_1)_n \cdots (a_p)_n}{(b_1)_n \cdots (b_q)_n}
                            \frac{z^n}{n!},

    where $(a)_n = (a)(a+1)\cdots(a+n-1)$ denotes the rising factorial.

    If one of the $b_q$ is a non-positive integer then the series is
    undefined unless one of the $a_p$ is a larger (i.e., smaller in
    magnitude) non-positive integer. If none of the $b_q$ is a
    non-positive integer and one of the $a_p$ is a non-positive
    integer, then the series reduces to a polynomial. To simplify the
    following discussion, we assume that none of the $a_p$ or
    $b_q$ is a non-positive integer. For more details, see the
    references.

    The series converges for all $z$ if $p \le q$, and thus
    defines an entire single-valued function in this case. If $p =
    q+1$ the series converges for $|z| < 1$, and can be continued
    analytically into a half-plane. If $p > q+1$ the series is
    divergent for all $z$.

    Please note the hypergeometric function constructor currently does *not*
    check if the parameters actually yield a well-defined function.

    Examples
    ========

    The parameters $a_p$ and $b_q$ can be passed as arbitrary
    iterables, for example:

    >>> from sympy import hyper
    >>> from sympy.abc import x, n, a
    创建一个 hypergeometric function 对象 h，使用参数 (1, 2, 3) 作为 numerator，[3, 4] 作为 denominator，并传入变量 x 作为参数；返回结果 h
    >>> h = hyper((1, 2, 3), [3, 4], x); h
    生成 hypergeometric function 对象 h，使用参数 (1, 2, 3) 作为 numerator，[3, 4] 作为 denominator，并传入变量 x 作为参数，返回结果 h
    hyper((1, 2), (4,), x)
    创建 hypergeometric function 对象，使用参数 (3, 1, 2) 作为 numerator，[3, 4] 作为 denominator，变量 x 作为参数，evaluate=False 表示不移除重复的参数
    >>> hyper((3, 1, 2), [3, 4], x, evaluate=False)  # don't remove duplicates
    返回 hypergeometric function 对象，使用参数 (3, 1, 2) 作为 numerator，[3, 4] 作为 denominator，变量 x 作为参数，evaluate=False 表示不移除重复的参数

    There is also pretty printing (it looks better using Unicode):

    >>> from sympy import pprint
    >>> pprint(h, use_unicode=False)
      _
     |_  /1, 2 |  \
     |   |     | x|
    2  1 \  4  |  /
    
    The parameters must always be iterables, even if they are vectors of
    length one or zero:

    >>> hyper((1, ), [], x)
    返回 hypergeometric function 对象，使用参数 (1,) 作为 numerator，空的 denominator，变量 x 作为参数

    But of course they may be variables (but if they depend on $x$ then you
    should not expect much implemented functionality):

    >>> hyper((n, a), (n**2,), x)
    返回 hypergeometric function 对象，使用参数 (n, a) 作为 numerator，(n**2,) 作为 denominator，变量 x 作为参数

    The hypergeometric function generalizes many named special functions.
    The function ``hyperexpand()`` tries to express a hypergeometric function
    using named special functions. For example:

    >>> from sympy import hyperexpand
    >>> hyperexpand(hyper([], [], x))
    返回将 hypergeometric function 对象 hyper([][], x) 通过特定的命名特殊函数展开后的结果 exp(x)
    exp(x)
    """
    """
    You can also use ``expand_func()``:

    >>> from sympy import expand_func
    >>> expand_func(x*hyper([1, 1], [2], -x))
    log(x + 1)

    More examples:

    >>> from sympy import S
    >>> hyperexpand(hyper([], [S(1)/2], -x**2/4))
    cos(x)
    >>> hyperexpand(x*hyper([S(1)/2, S(1)/2], [S(3)/2], x**2))
    asin(x)

    We can also sometimes ``hyperexpand()`` parametric functions:

    >>> from sympy.abc import a
    >>> hyperexpand(hyper([-a], [], x))
    (1 - x)**a

    See Also
    ========

    sympy.simplify.hyperexpand
    gamma
    meijerg

    References
    ==========

    .. [1] Luke, Y. L. (1969), The Special Functions and Their Approximations,
           Volume 1
    .. [2] https://en.wikipedia.org/wiki/Generalized_hypergeometric_function

    """

    # 创建一个新的特殊超几何函数对象
    def __new__(cls, ap, bq, z, **kwargs):
        # TODO should we check convergence conditions? 是否需要检查收敛条件？
        # 根据参数 'evaluate' 的设置或全局参数决定是否评估函数
        if kwargs.pop('evaluate', global_parameters.evaluate):
            # 使用 Counter 统计并集和差集来简化参数列表
            ca = Counter(Tuple(*ap))
            cb = Counter(Tuple(*bq))
            common = ca & cb
            arg = ap, bq = [], []
            for i, c in enumerate((ca, cb)):
                c -= common
                for k in ordered(c):
                    arg[i].extend([k]*c[k])
        else:
            # 对参数列表进行排序和处理
            ap = list(ordered(ap))
            bq = list(ordered(bq))
        # 调用父类的构造函数生成对象
        return Function.__new__(cls, _prep_tuple(ap), _prep_tuple(bq), z, **kwargs)

    # 类方法，用于计算特殊超几何函数的值
    @classmethod
    def eval(cls, ap, bq, z):
        # 根据参数长度关系和 z 的值来评估特殊超几何函数
        if len(ap) <= len(bq) or (len(ap) == len(bq) + 1 and (Abs(z) <= 1) == True):
            # 去极化处理 z
            nz = unpolarify(z)
            if z != nz:
                return hyper(ap, bq, nz)

    # 计算特殊超几何函数的偏导数
    def fdiff(self, argindex=3):
        # 只接受第三个参数的偏导数计算，否则引发错误
        if argindex != 3:
            raise ArgumentIndexError(self, argindex)
        # 递增每个参数中的每个指数
        nap = Tuple(*[a + 1 for a in self.ap])
        nbq = Tuple(*[b + 1 for b in self.bq])
        # 计算函数的系数
        fac = Mul(*self.ap)/Mul(*self.bq)
        return fac*hyper(nap, nbq, self.argument)

    # 内部方法，用于展开特殊超几何函数的表达式
    def _eval_expand_func(self, **hints):
        # 导入必要的函数
        from sympy.functions.special.gamma_functions import gamma
        from sympy.simplify.hyperexpand import hyperexpand
        # 当参数满足特定条件时，返回特殊表达式
        if len(self.ap) == 2 and len(self.bq) == 1 and self.argument == 1:
            a, b = self.ap
            c = self.bq[0]
            return gamma(c)*gamma(c - a - b)/gamma(c - a)/gamma(c - b)
        # 否则，调用默认的展开函数
        return hyperexpand(self)

    # 内部方法，将特殊超几何函数重写为求和形式
    def _eval_rewrite_as_Sum(self, ap, bq, z, **kwargs):
        # 导入必要的函数
        from sympy.concrete.summations import Sum
        # 创建一个整数变量 n
        n = Dummy("n", integer=True)
        # 创建升幂阶乘序列
        rfap = [RisingFactorial(a, n) for a in ap]
        rfbq = [RisingFactorial(b, n) for b in bq]
        # 计算系数
        coeff = Mul(*rfap) / Mul(*rfbq)
        # 返回特定形式的求和
        return Piecewise((Sum(coeff * z**n / factorial(n), (n, 0, oo)),
                         self.convergence_statement), (self, True))
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 获取第三个参数
        arg = self.args[2]
        # 计算参数在 x=0 处的极限值
        x0 = arg.subs(x, 0)
        # 如果极限值为 NaN，则计算 x->0 时的极限
        if x0 is S.NaN:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')

        # 如果极限值为 0，则返回 S.One
        if x0 is S.Zero:
            return S.One
        # 否则调用父类的方法计算作为主导项
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # 导入必要的模块和类
        from sympy.series.order import Order

        # 获取第三个参数
        arg = self.args[2]
        # 计算参数在 x=0 处的极限值
        x0 = arg.limit(x, 0)
        # 获取第一个和第二个参数
        ap = self.args[0]
        bq = self.args[1]

        # 如果参数不是 x，或者极限值不为 0
        if not (arg == x and x0 == 0):
            # 最好在这里使用 arg.nseries 来做一些处理，而不是回退到 Function._eval_nseries。
            # 以下的代码如果 arg 是像 x/(x+1) 这样的表达式是不够的。
            from sympy.simplify.hyperexpand import hyperexpand
            return hyperexpand(super()._eval_nseries(x, n, logx))

        # 初始化一个空列表来存储级数展开的项
        terms = []

        # 循环计算前 n 项
        for i in range(n):
            # 计算分子的升幂阶乘
            num = Mul(*[RisingFactorial(a, i) for a in ap])
            # 计算分母的升幂阶乘
            den = Mul(*[RisingFactorial(b, i) for b in bq])
            # 计算当前项的值并添加到 terms 中
            terms.append(((num/den) * (arg**i)) / factorial(i))

        # 返回级数展开结果加上高阶无穷小量
        return (Add(*terms) + Order(x**n,x))

    @property
    def argument(self):
        """ Argument of the hypergeometric function. """
        # 返回第三个参数，即超几何函数的参数
        return self.args[2]

    @property
    def ap(self):
        """ Numerator parameters of the hypergeometric function. """
        # 返回超几何函数的分子参数
        return Tuple(*self.args[0])

    @property
    def bq(self):
        """ Denominator parameters of the hypergeometric function. """
        # 返回超几何函数的分母参数
        return Tuple(*self.args[1])

    @property
    def _diffargs(self):
        # 返回分子和分母参数的元组
        return self.ap + self.bq

    @property
    def eta(self):
        """ A quantity related to the convergence of the series. """
        # 返回级数收敛性相关的量，即分子参数之和减去分母参数之和
        return sum(self.ap) - sum(self.bq)
    # 计算定义级数的收敛半径

    def radius_of_convergence(self):
        """
        计算定义级数的收敛半径。

        Explanation
        ===========

        即使收敛半径不是无穷大，函数可能仍通过解析延拓在收敛半径外部被求值。
        但如果收敛半径为零，则函数实际上在任何其他地方都没有定义。

        Examples
        ========

        >>> from sympy import hyper
        >>> from sympy.abc import z
        >>> hyper((1, 2), [3], z).radius_of_convergence
        1
        >>> hyper((1, 2, 3), [4], z).radius_of_convergence
        0
        >>> hyper((1, 2), (3, 4), z).radius_of_convergence
        oo

        """
        # 检查分子参数和分母参数中是否有任何整数且小于等于零的情况
        if any(a.is_integer and (a <= 0) == True for a in self.ap + self.bq):
            # 提取所有小于等于零的整数分子参数
            aints = [a for a in self.ap if a.is_Integer and (a <= 0) == True]
            # 提取所有小于等于零的整数分母参数
            bints = [a for a in self.bq if a.is_Integer and (a <= 0) == True]
            # 如果分子参数中小于等于零的整数个数少于分母参数中的个数，返回零
            if len(aints) < len(bints):
                return S.Zero
            popped = False
            # 对每一个分母参数进行检查
            for b in bints:
                cancelled = False
                while aints:
                    a = aints.pop()
                    # 如果存在分子参数大于等于当前分母参数，则标记为已取消
                    if a >= b:
                        cancelled = True
                        break
                    popped = True
                # 如果没有取消的情况，返回零
                if not cancelled:
                    return S.Zero
            # 如果还有未处理的分子参数或者曾经有取消的情况，说明这是一个多项式
            return oo
        # 如果分子参数数量等于分母参数数量加一，返回一
        if len(self.ap) == len(self.bq) + 1:
            return S.One
        # 如果分子参数数量小于等于分母参数数量，返回无穷大
        elif len(self.ap) <= len(self.bq):
            return oo
        else:
            # 否则返回零
            return S.Zero

    @property
    def convergence_statement(self):
        """ 返回一个关于 z 收敛级数的条件。 """
        R = self.radius_of_convergence
        # 如果收敛半径为零，返回 False
        if R == 0:
            return False
        # 如果收敛半径为无穷大，返回 True
        if R == oo:
            return True
        # 计算特殊函数及其近似，参考第44页
        e = self.eta
        z = self.argument
        c1 = And(re(e) < 0, abs(z) <= 1)
        c2 = And(0 <= re(e), re(e) < 1, abs(z) <= 1, Ne(z, 1))
        c3 = And(re(e) >= 1, abs(z) < 1)
        # 返回条件 c1, c2, c3 的逻辑或结果
        return Or(c1, c2, c3)

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.hyperexpand import hyperexpand
        # 调用 sympy.simplify.hyperexpand 模块中的 hyperexpand 函数进行简化
        return hyperexpand(self)
class meijerg(TupleParametersBase):
    r"""
    The Meijer G-function is defined by a Mellin-Barnes type integral that
    resembles an inverse Mellin transform. It generalizes the hypergeometric
    functions.

    Explanation
    ===========

    The Meijer G-function depends on four sets of parameters. There are
    "*numerator parameters*"
    $a_1, \ldots, a_n$ and $a_{n+1}, \ldots, a_p$, and there are
    "*denominator parameters*"
    $b_1, \ldots, b_m$ and $b_{m+1}, \ldots, b_q$.
    Confusingly, it is traditionally denoted as follows (note the position
    of $m$, $n$, $p$, $q$, and how they relate to the lengths of the four
    parameter vectors):

    .. math ::
        G_{p,q}^{m,n} \left(\begin{matrix}a_1, \cdots, a_n & a_{n+1}, \cdots, a_p \\
                                        b_1, \cdots, b_m & b_{m+1}, \cdots, b_q
                          \end{matrix} \middle| z \right).

    However, in SymPy the four parameter vectors are always available
    separately (see examples), so that there is no need to keep track of the
    decorating sub- and super-scripts on the G symbol.

    The G function is defined as the following integral:

    .. math ::
         \frac{1}{2 \pi i} \int_L \frac{\prod_{j=1}^m \Gamma(b_j - s)
         \prod_{j=1}^n \Gamma(1 - a_j + s)}{\prod_{j=m+1}^q \Gamma(1- b_j +s)
         \prod_{j=n+1}^p \Gamma(a_j - s)} z^s \mathrm{d}s,

    where $\Gamma(z)$ is the gamma function. There are three possible
    contours which we will not describe in detail here (see the references).
    If the integral converges along more than one of them, the definitions
    agree. The contours all separate the poles of $\Gamma(1-a_j+s)$
    from the poles of $\Gamma(b_k-s)$, so in particular the G function
    is undefined if $a_j - b_k \in \mathbb{Z}_{>0}$ for some
    $j \le n$ and $k \le m$.

    The conditions under which one of the contours yields a convergent integral
    are complicated and we do not state them here, see the references.

    Please note currently the Meijer G-function constructor does *not* check any
    convergence conditions.

    Examples
    ========

    You can pass the parameters either as four separate vectors:

    >>> from sympy import meijerg, Tuple, pprint
    >>> from sympy.abc import x, a
    >>> pprint(meijerg((1, 2), (a, 4), (5,), [], x), use_unicode=False)
     __1, 2 /1, 2  4, a |  \
    /__     |           | x|
    \_|4, 1 \ 5         |  /

    Or as two nested vectors:

    >>> pprint(meijerg([(1, 2), (3, 4)], ([5], Tuple()), x), use_unicode=False)
     __1, 2 /1, 2  3, 4 |  \
    /__     |           | x|
    \_|4, 1 \ 5         |  /

    As with the hypergeometric function, the parameters may be passed as
    arbitrary iterables. Vectors of length zero and one also have to be
    passed as iterables. The parameters need not be constants, but if they
    depend on the argument then not much implemented functionality should be
    expected.
    """
    All the subvectors of parameters are available:

    >>> from sympy import pprint  # 导入pprint函数，用于美化输出
    >>> g = meijerg([1], [2], [3], [4], x)  # 创建Meijer G函数对象g，传入参数
    >>> pprint(g, use_unicode=False)  # 使用pprint函数打印Meijer G函数g，不使用Unicode字符

     __1, 1 /1  2 |  \         # Meijer G函数的标准表示形式
    /__     |     | x|         # 参数和变量
    \_|2, 2 \3  4 |  /         # 下标表示参数的分组

    >>> g.an  # 访问Meijer G函数g的参数an
    (1,)     # 返回Meijer G函数g的参数an的元组

    >>> g.ap  # 访问Meijer G函数g的参数ap
    (1, 2)   # 返回Meijer G函数g的参数ap的元组

    >>> g.aother  # 访问Meijer G函数g的参数aother
    (2,)     # 返回Meijer G函数g的参数aother的元组

    >>> g.bm  # 访问Meijer G函数g的参数bm
    (3,)     # 返回Meijer G函数g的参数bm的元组

    >>> g.bq  # 访问Meijer G函数g的参数bq
    (3, 4)   # 返回Meijer G函数g的参数bq的元组

    >>> g.bother  # 访问Meijer G函数g的参数bother
    (4,)     # 返回Meijer G函数g的参数bother的元组

    The Meijer G-function generalizes the hypergeometric functions.
    In some cases it can be expressed in terms of hypergeometric functions,
    using Slater's theorem. For example:

    >>> from sympy import hyperexpand  # 导入hyperexpand函数，用于展开超几何函数
    >>> from sympy.abc import a, b, c  # 导入a, b, c作为符号变量
    >>> hyperexpand(meijerg([a], [], [c], [b], x), allow_hyper=True)
    x**c*gamma(-a + c + 1)*hyper((-a + c + 1,),
                                 (-b + c + 1,), -x)/gamma(-b + c + 1)

    Thus the Meijer G-function also subsumes many named functions as special
    cases. You can use ``expand_func()`` or ``hyperexpand()`` to (try to)
    rewrite a Meijer G-function in terms of named special functions. For
    example:

    >>> from sympy import expand_func, S  # 导入expand_func函数和S符号
    >>> expand_func(meijerg([[],[]], [[0],[]], -x))  # 展开Meijer G函数
    exp(x)  # 返回展开后的表达式

    >>> hyperexpand(meijerg([[],[]], [[S(1)/2],[0]], (x/2)**2))  # 使用hyperexpand展开Meijer G函数
    sin(x)/sqrt(pi)  # 返回展开后的表达式

    See Also
    ========

    hyper  # 参考超几何函数的文档
    sympy.simplify.hyperexpand  # 参考hyperexpand函数的文档

    References
    ==========

    .. [1] Luke, Y. L. (1969), The Special Functions and Their Approximations,
           Volume 1
    .. [2] https://en.wikipedia.org/wiki/Meijer_G-function

    """

    # 实现Meijer G函数类的构造函数
    def __new__(cls, *args, **kwargs):
        # 如果参数长度为5，转换为标准形式
        if len(args) == 5:
            args = [(args[0], args[1]), (args[2], args[3]), args[4]]
        # 如果参数长度不为3，抛出类型错误异常
        if len(args) != 3:
            raise TypeError("args must be either as, as', bs, bs', z or "
                            "as, bs, z")

        # 定义参数转换函数
        def tr(p):
            # 如果参数长度不为2，抛出类型错误异常
            if len(p) != 2:
                raise TypeError("wrong argument")
            # 转换为排序后的列表，并返回TupleArg对象
            p = [list(ordered(i)) for i in p]
            return TupleArg(_prep_tuple(p[0]), _prep_tuple(p[1]))

        # 转换第一个和第二个参数
        arg0, arg1 = tr(args[0]), tr(args[1])
        # 如果参数中包含oo, zoo, -oo，抛出值错误异常
        if Tuple(arg0, arg1).has(oo, zoo, -oo):
            raise ValueError("G-function parameters must be finite")
        # 如果任何(a - b)是整数并且大于0，抛出值错误异常
        if any((a - b).is_Integer and a - b > 0
               for a in arg0[0] for b in arg1[0]):
            raise ValueError("no parameter a1, ..., an may differ from "
                             "any b1, ..., bm by a positive integer")

        # TODO 应该检查收敛条件吗？
        # 返回函数的新实例
        return Function.__new__(cls, arg0, arg1, args[2], **kwargs)
    # 定义一个方法，计算 meijerg 函数关于其参数的导数
    def fdiff(self, argindex=3):
        # 如果参数索引不等于 3，调用私有方法 _diff_wrt_parameter 处理
        if argindex != 3:
            return self._diff_wrt_parameter(argindex[1])
        
        # 如果 self.an 列表的长度大于等于 1
        if len(self.an) >= 1:
            # 将 self.an 转换为列表 a，并将第一个元素减去 1
            a = list(self.an)
            a[0] -= 1
            # 调用 meijerg 函数，传入修改后的参数列表 a 和其他参数，计算 G
            G = meijerg(a, self.aother, self.bm, self.bother, self.argument)
            # 返回表达式：1/self.argument * ((self.an[0] - 1)*self + G)
            return 1/self.argument * ((self.an[0] - 1)*self + G)
        
        # 如果 self.an 列表长度为 0 且 self.bm 列表长度大于等于 1
        elif len(self.bm) >= 1:
            # 将 self.bm 转换为列表 b，并将第一个元素加上 1
            b = list(self.bm)
            b[0] += 1
            # 调用 meijerg 函数，传入参数 self.an、self.aother、修改后的参数列表 b、self.bother 和 self.argument，计算 G
            G = meijerg(self.an, self.aother, b, self.bother, self.argument)
            # 返回表达式：1/self.argument * (self.bm[0]*self - G)
            return 1/self.argument * (self.bm[0]*self - G)
        
        # 如果 self.an 和 self.bm 列表长度均为 0，则返回零
        else:
            return S.Zero

    def get_period(self):
        """
        返回一个数 P，使得 G(x*exp(I*P)) == G(x) 成立。

        Examples
        ========

        >>> from sympy import meijerg, pi, S
        >>> from sympy.abc import z

        >>> meijerg([1], [], [], [], z).get_period()
        2*pi
        >>> meijerg([pi], [], [], [], z).get_period()
        oo
        >>> meijerg([1, 2], [], [], [], z).get_period()
        oo
        >>> meijerg([1,1], [2], [1, S(1)/2, S(1)/3], [1], z).get_period()
        12*pi

        """
        # 根据 Slater 定理进行计算。
        
        # 定义一个函数 compute(l)，用于计算给定列表 l 中元素的最小公倍数
        def compute(l):
            # 首先检查是否有两个元素之差为整数
            for i, b in enumerate(l):
                if not b.is_Rational:
                    return oo
                for j in range(i + 1, len(l)):
                    if not Mod((b - l[j]).simplify(), 1):
                        return oo
            # 如果都是有理数，则返回这些有理数的最小公倍数
            return lcm(*(x.q for x in l))
        
        # 计算 self.bm 和 self.an 中元素的最小公倍数
        beta = compute(self.bm)
        alpha = compute(self.an)
        p, q = len(self.ap), len(self.bq)
        
        # 如果 p 等于 q
        if p == q:
            # 如果 alpha 或 beta 中有无穷大，则返回无穷大
            if oo in (alpha, beta):
                return oo
            # 否则返回 2*pi*alpha 和 2*pi*beta 的最小公倍数
            return 2*pi*lcm(alpha, beta)
        
        # 如果 p 小于 q，则返回 2*pi*beta
        elif p < q:
            return 2*pi*beta
        
        # 如果 p 大于 q，则返回 2*pi*alpha
        else:
            return 2*pi*alpha

    def _eval_expand_func(self, **hints):
        # 导入 hyperexpand 函数，然后调用它对 self 进行超级展开
        from sympy.simplify.hyperexpand import hyperexpand
        return hyperexpand(self)
    # 对 Meijer G 函数进行评估和数值评估
    def _eval_evalf(self, prec):
        # 默认的代码对极坐标参数不足够
        # mpmath 提供了一个可选参数 "r"，它评估 G(z**(1/r))。
        # 我们在这里使用它的方式是：为了在一个参数 z 的值的 |argument| 小于 (比如) n*pi 的范围内评估，
        # 我们设置 r=1/n，计算 z' = root(z, n)（要小心不丢失分支信息），并评估 G(z'**(1/r)) = G(z'**n) = G(z)。
        import mpmath
        # 获取参数的数值评估 znum
        znum = self.argument._eval_evalf(prec)
        # 如果 znum 包含 exp_polar
        if znum.has(exp_polar):
            # 将 znum 拆分为系数和乘法项，获取分支信息
            znum, branch = znum.as_coeff_mul(exp_polar)
            # 如果分支信息长度不为 1，则返回
            if len(branch) != 1:
                return
            # 计算分支信息
            branch = branch[0].args[0] / I
        else:
            branch = S.Zero
        # 计算 n，使得 |argument| < n*pi
        n = ceiling(abs(branch / pi)) + 1
        # 计算 znum 的 n 次方根和分支调整后的值
        znum = znum**(S.One / n) * exp(I * branch / n)

        # 将所有参数转换为 mpf 或 mpc 类型
        try:
            [z, r, ap, bq] = [arg._to_mpmath(prec)
                    for arg in [znum, 1/n, self.args[0], self.args[1]]]
        except ValueError:
            return

        # 使用指定的精度进行计算
        with mpmath.workprec(prec):
            v = mpmath.meijerg(ap, bq, z, r)

        # 将结果转换为 Expr 对象
        return Expr._from_mpmath(v, prec)

    # 将 Meijer G 函数扩展为其主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.simplify.hyperexpand import hyperexpand
        return hyperexpand(self).as_leading_term(x, logx=logx, cdir=cdir)

    # 获取定义积分 D(s)
    def integrand(self, s):
        """ Get the defining integrand D(s). """
        from sympy.functions.special.gamma_functions import gamma
        return self.argument**s \
            * Mul(*(gamma(b - s) for b in self.bm)) \
            * Mul(*(gamma(1 - a + s) for a in self.an)) \
            / Mul(*(gamma(1 - b + s) for b in self.bother)) \
            / Mul(*(gamma(a - s) for a in self.aother))

    # 返回 Meijer G 函数的参数
    @property
    def argument(self):
        """ Argument of the Meijer G-function. """
        return self.args[2]

    # 返回分子参数的第一组参数
    @property
    def an(self):
        """ First set of numerator parameters. """
        return Tuple(*self.args[0][0])

    # 返回组合的分子参数
    @property
    def ap(self):
        """ Combined numerator parameters. """
        return Tuple(*(self.args[0][0] + self.args[0][1]))

    # 返回分子参数的第二组参数
    @property
    def aother(self):
        """ Second set of numerator parameters. """
        return Tuple(*self.args[0][1])

    # 返回分母参数的第一组参数
    @property
    def bm(self):
        """ First set of denominator parameters. """
        return Tuple(*self.args[1][0])

    # 返回组合的分母参数
    @property
    def bq(self):
        """ Combined denominator parameters. """
        return Tuple(*(self.args[1][0] + self.args[1][1]))

    # 返回分母参数的第二组参数
    @property
    def bother(self):
        """ Second set of denominator parameters. """
        return Tuple(*self.args[1][1])

    # 返回用于差分的参数列表
    @property
    def _diffargs(self):
        return self.ap + self.bq
    # 定义一个方法 nu，返回一个与积分收敛区域相关的量，参考文献见上文。
    def nu(self):
        """ A quantity related to the convergence region of the integral,
            c.f. references. """
        # 返回 self.bq 列表的和减去 self.ap 列表的和
        return sum(self.bq) - sum(self.ap)

    # 定义一个属性 delta，返回一个与积分收敛区域相关的量，参考文献见上文。
    @property
    def delta(self):
        """ A quantity related to the convergence region of the integral,
            c.f. references. """
        # 返回 self.bm 列表的长度加上 self.an 列表的长度，减去 S(len(self.ap) + len(self.bq))/2 的结果
        return len(self.bm) + len(self.an) - S(len(self.ap) + len(self.bq))/2

    # 定义一个属性 is_number，如果表达式仅包含数值数据则返回 True。
    @property
    def is_number(self):
        """ Returns true if expression has numeric data only. """
        # 如果 self.free_symbols 为空，则返回 True；否则返回 False
        return not self.free_symbols
class HyperRep(Function):
    """
    A base class for "hyper representation functions".

    This is used exclusively in ``hyperexpand()``, but fits more logically here.

    pFq is branched at 1 if p == q+1. For use with slater-expansion, we want
    define an "analytic continuation" to all polar numbers, which is
    continuous on circles and on the ray t*exp_polar(I*pi). Moreover, we want
    a "nice" expression for the various cases.

    This base class contains the core logic, concrete derived classes only
    supply the actual functions.
    """

    @classmethod
    def eval(cls, *args):
        # 对传入参数进行去极性处理，保证参数在函数内部一致性
        newargs = tuple(map(unpolarify, args[:-1])) + args[-1:]
        # 如果处理后的参数与原参数不同，则返回一个新的类实例
        if args != newargs:
            return cls(*newargs)

    @classmethod
    def _expr_small(cls, x):
        """ An expression for F(x) which holds for |x| < 1. """
        # 小于1的情况下 F(x) 的表达式，需要在具体的派生类中实现
        raise NotImplementedError

    @classmethod
    def _expr_small_minus(cls, x):
        """ An expression for F(-x) which holds for |x| < 1. """
        # 小于1的情况下 F(-x) 的表达式，需要在具体的派生类中实现
        raise NotImplementedError

    @classmethod
    def _expr_big(cls, x, n):
        """ An expression for F(exp_polar(2*I*pi*n)*x), |x| > 1. """
        # 大于1的情况下 F(exp_polar(2*I*pi*n)*x) 的表达式，需要在具体的派生类中实现
        raise NotImplementedError

    @classmethod
    def _expr_big_minus(cls, x, n):
        """ An expression for F(exp_polar(2*I*pi*n + pi*I)*x), |x| > 1. """
        # 大于1的情况下 F(exp_polar(2*I*pi*n + pi*I)*x) 的表达式，需要在具体的派生类中实现
        raise NotImplementedError

    def _eval_rewrite_as_nonrep(self, *args, **kwargs):
        # 提取参数中的枝分支因子，允许半整数
        x, n = self.args[-1].extract_branch_factor(allow_half=True)
        minus = False
        newargs = self.args[:-1] + (x,)
        # 如果 n 不是整数，则设置 minus 为 True，并调整 n
        if not n.is_Integer:
            minus = True
            n -= S.Half
        newerargs = newargs + (n,)
        # 根据是否有负号，选择不同的表达式计算方法
        if minus:
            small = self._expr_small_minus(*newargs)
            big = self._expr_big_minus(*newerargs)
        else:
            small = self._expr_small(*newargs)
            big = self._expr_big(*newerargs)

        # 如果大表达式等于小表达式，返回小表达式；否则使用分段函数返回大表达式或小表达式
        if big == small:
            return small
        return Piecewise((big, abs(x) > 1), (small, True))

    def _eval_rewrite_as_nonrepsmall(self, *args, **kwargs):
        # 提取参数中的枝分支因子，允许半整数
        x, n = self.args[-1].extract_branch_factor(allow_half=True)
        args = self.args[:-1] + (x,)
        # 如果 n 不是整数，返回小表达式的计算结果
        if not n.is_Integer:
            return self._expr_small_minus(*args)
        return self._expr_small(*args)


class HyperRep_power1(HyperRep):
    """ Return a representative for hyper([-a], [], z) == (1 - z)**a. """

    @classmethod
    def _expr_small(cls, a, x):
        # 对于小于1的情况，返回 (1 - x)**a 的表达式
        return (1 - x)**a

    @classmethod
    def _expr_small_minus(cls, a, x):
        # 对于小于1的情况，返回 (1 + x)**a 的表达式
        return (1 + x)**a

    @classmethod
    def _expr_big(cls, a, x, n):
        # 如果 a 是整数，则返回小表达式的计算结果
        if a.is_integer:
            return cls._expr_small(a, x)
        # 否则返回大表达式的计算结果
        return (x - 1)**a*exp((2*n - 1)*pi*I*a)

    @classmethod
    def _expr_big_minus(cls, a, x, n):
        # 如果 a 是整数，则返回小表达式的计算结果
        if a.is_integer:
            return cls._expr_small_minus(a, x)
        # 否则返回大表达式的计算结果
        return (1 + x)**a*exp(2*n*pi*I*a)


class HyperRep_power2(HyperRep):
    """ Return a representative for hyper([a, a - 1/2], [2*a], z). """
    
    @classmethod
    def _expr_small(cls, a, x):
        # 对于小于1的情况，返回 (1 - x)**a 的表达式
        return (1 - x)**a

    @classmethod
    def _expr_small_minus(cls, a, x):
        # 对于小于1的情况，返回 (1 + x)**a 的表达式
        return (1 + x)**a

    @classmethod
    def _expr_big(cls, a, x, n):
        # 如果 a 是整数，则返回小表达式的计算结果
        if a.is_integer:
            return cls._expr_small(a, x)
        # 否则返回大表达式的计算结果
        return (x - 1)**a*exp((2*n - 1)*pi*I*a)

    @classmethod
    def _expr_big_minus(cls, a, x, n):
        # 如果 a 是整数，则返回小表达式的计算结果
        if a.is_integer:
            return cls._expr_small_minus(a, x)
        # 否则返回大表达式的计算结果
        return (1 + x)**a*exp(2*n*pi*I*a)
    # 计算表达式 2^(2*a - 1) * (1 + sqrt(1 - x))^(1 - 2*a)
    def _expr_small(cls, a, x):
        return 2**(2*a - 1)*(1 + sqrt(1 - x))**(1 - 2*a)

    # 计算表达式 2^(2*a - 1) * (1 + sqrt(1 + x))^(1 - 2*a)
    @classmethod
    def _expr_small_minus(cls, a, x):
        return 2**(2*a - 1)*(1 + sqrt(1 + x))**(1 - 2*a)

    # 计算表达式 2^(2*a - 1) * (1 + sgn*I*sqrt(x - 1))^(1 - 2*a) * exp(-2*n*pi*I*a)
    @classmethod
    def _expr_big(cls, a, x, n):
        sgn = -1
        # 如果 n 是奇数，则改变 sgn 的值为 1，并将 n 减去 1
        if n.is_odd:
            sgn = 1
            n -= 1
        return 2**(2*a - 1)*(1 + sgn*I*sqrt(x - 1))**(1 - 2*a) \
            *exp(-2*n*pi*I*a)

    # 计算表达式 sgn*2^(2*a - 1)*(sqrt(1 + x) + sgn)^(1 - 2*a)*exp(-2*pi*I*a*n)
    @classmethod
    def _expr_big_minus(cls, a, x, n):
        sgn = 1
        # 如果 n 是奇数，则改变 sgn 的值为 -1
        if n.is_odd:
            sgn = -1
        return sgn*2**(2*a - 1)*(sqrt(1 + x) + sgn)**(1 - 2*a)*exp(-2*pi*I*a*n)
class HyperRep_log1(HyperRep):
    """ Represent -z*hyper([1, 1], [2], z) == log(1 - z). """

    @classmethod
    def _expr_small(cls, x):
        # 返回 log(1 - x)，表示小的参数情况下的对数表达式
        return log(1 - x)

    @classmethod
    def _expr_small_minus(cls, x):
        # 返回 log(1 + x)，表示小的参数情况下的对数表达式（负号情况）
        return log(1 + x)

    @classmethod
    def _expr_big(cls, x, n):
        # 返回 log(x - 1) + (2*n - 1)*pi*I，表示大的参数情况下的对数表达式
        return log(x - 1) + (2*n - 1)*pi*I

    @classmethod
    def _expr_big_minus(cls, x, n):
        # 返回 log(1 + x) + 2*n*pi*I，表示大的参数情况下的对数表达式（负号情况）
        return log(1 + x) + 2*n*pi*I


class HyperRep_atanh(HyperRep):
    """ Represent hyper([1/2, 1], [3/2], z) == atanh(sqrt(z))/sqrt(z). """

    @classmethod
    def _expr_small(cls, x):
        # 返回 atanh(sqrt(x))/sqrt(x)，表示小的参数情况下的反双曲正切表达式
        return atanh(sqrt(x))/sqrt(x)

    @classmethod
    def _expr_small_minus(cls, x):
        # 返回 atan(sqrt(x))/sqrt(x)，表示小的参数情况下的反正切表达式（负号情况）
        return atan(sqrt(x))/sqrt(x)

    @classmethod
    def _expr_big(cls, x, n):
        # 根据 n 的奇偶性返回不同的表达式，表示大的参数情况下的反双曲余切表达式
        if n.is_even:
            return (acoth(sqrt(x)) + I*pi/2)/sqrt(x)
        else:
            return (acoth(sqrt(x)) - I*pi/2)/sqrt(x)

    @classmethod
    def _expr_big_minus(cls, x, n):
        # 根据 n 的奇偶性返回不同的表达式，表示大的参数情况下的反正切表达式（负号情况）
        if n.is_even:
            return atan(sqrt(x))/sqrt(x)
        else:
            return (atan(sqrt(x)) - pi)/sqrt(x)


class HyperRep_asin1(HyperRep):
    """ Represent hyper([1/2, 1/2], [3/2], z) == asin(sqrt(z))/sqrt(z). """

    @classmethod
    def _expr_small(cls, z):
        # 返回 asin(sqrt(z))/sqrt(z)，表示小的参数情况下的反正弦表达式
        return asin(sqrt(z))/sqrt(z)

    @classmethod
    def _expr_small_minus(cls, z):
        # 返回 asinh(sqrt(z))/sqrt(z)，表示小的参数情况下的反双曲正弦表达式
        return asinh(sqrt(z))/sqrt(z)

    @classmethod
    def _expr_big(cls, z, n):
        # 返回 S.NegativeOne**n*((S.Half - n)*pi/sqrt(z) + I*acosh(sqrt(z))/sqrt(z))，表示大的参数情况下的表达式
        return S.NegativeOne**n*((S.Half - n)*pi/sqrt(z) + I*acosh(sqrt(z))/sqrt(z))

    @classmethod
    def _expr_big_minus(cls, z, n):
        # 返回 S.NegativeOne**n*(asinh(sqrt(z))/sqrt(z) + n*pi*I/sqrt(z))，表示大的参数情况下的表达式（负号情况）
        return S.NegativeOne**n*(asinh(sqrt(z))/sqrt(z) + n*pi*I/sqrt(z))


class HyperRep_asin2(HyperRep):
    """ Represent hyper([1, 1], [3/2], z) == asin(sqrt(z))/sqrt(z)/sqrt(1-z). """

    @classmethod
    def _expr_small(cls, z):
        # 返回 HyperRep_asin1._expr_small(z) / HyperRep_power1._expr_small(S.Half, z)，表示小的参数情况下的表达式
        return HyperRep_asin1._expr_small(z) / HyperRep_power1._expr_small(S.Half, z)

    @classmethod
    def _expr_small_minus(cls, z):
        # 返回 HyperRep_asin1._expr_small_minus(z) / HyperRep_power1._expr_small_minus(S.Half, z)，表示小的参数情况下的表达式（负号情况）
        return HyperRep_asin1._expr_small_minus(z) / HyperRep_power1._expr_small_minus(S.Half, z)

    @classmethod
    def _expr_big(cls, z, n):
        # 返回 HyperRep_asin1._expr_big(z, n) / HyperRep_power1._expr_big(S.Half, z, n)，表示大的参数情况下的表达式
        return HyperRep_asin1._expr_big(z, n) / HyperRep_power1._expr_big(S.Half, z, n)

    @classmethod
    def _expr_big_minus(cls, z, n):
        # 返回 HyperRep_asin1._expr_big_minus(z, n) / HyperRep_power1._expr_big_minus(S.Half, z, n)，表示大的参数情况下的表达式（负号情况）
        return HyperRep_asin1._expr_big_minus(z, n) / HyperRep_power1._expr_big_minus(S.Half, z, n)


class HyperRep_sqrts1(HyperRep):
    """ Return a representative for hyper([-a, 1/2 - a], [1/2], z). """

    @classmethod
    def _expr_small(cls, a, z):
        # 返回 ((1 - sqrt(z))**(2*a) + (1 + sqrt(z))**(2*a))/2，表示小的参数情况下的表达式
        return ((1 - sqrt(z))**(2*a) + (1 + sqrt(z))**(2*a))/2

    @classmethod
    def _expr_small_minus(cls, a, z):
        # 返回 (1 + z)**a*cos(2*a*atan(sqrt(z)))，表示小的参数情况下的表达式（负号情况）
        return (1 + z)**a*cos(2*a*atan(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        # TODO: Implement this method for big parameters
        pass

    @classmethod
    def _expr_big_minus(cls, a, z, n):
        # TODO: Implement this method for big parameters (minus case)
        pass
    # 定义一个类方法，计算表达式 `_expr_big` 的值
    def _expr_big(cls, a, z, n):
        # 如果 n 是偶数，计算第一种复杂表达式
        if n.is_even:
            return ((sqrt(z) + 1)**(2*a)*exp(2*pi*I*n*a) +
                    (sqrt(z) - 1)**(2*a)*exp(2*pi*I*(n - 1)*a))/2
        else:
            # 如果 n 是奇数，将 n 减去 1 并计算第二种复杂表达式
            n -= 1
            return ((sqrt(z) - 1)**(2*a)*exp(2*pi*I*a*(n + 1)) +
                    (sqrt(z) + 1)**(2*a)*exp(2*pi*I*a*n))/2
    
    # 定义一个类方法，计算表达式 `_expr_big_minus` 的值
    @classmethod
    def _expr_big_minus(cls, a, z, n):
        # 如果 n 是偶数，计算第一种简化的表达式
        if n.is_even:
            return (1 + z)**a*exp(2*pi*I*n*a)*cos(2*a*atan(sqrt(z)))
        else:
            # 如果 n 是奇数，计算第二种简化的表达式
            return (1 + z)**a*exp(2*pi*I*n*a)*cos(2*a*atan(sqrt(z)) - 2*pi*a)
class HyperRep_sqrts2(HyperRep):
    """ Represent a hypergeometric representation for
          sqrt(z)/2*[(1-sqrt(z))**2a - (1 + sqrt(z))**2a]
          == -2*z/(2*a+1) d/dz hyper([-a - 1/2, -a], [1/2], z) """

    @classmethod
    def _expr_small(cls, a, z):
        # Compute the expression for small z:
        return sqrt(z)*((1 - sqrt(z))**(2*a) - (1 + sqrt(z))**(2*a))/2

    @classmethod
    def _expr_small_minus(cls, a, z):
        # Compute the expression for small -z:
        return sqrt(z)*(1 + z)**a*sin(2*a*atan(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        # Compute the expression for large z based on n being even or odd:
        if n.is_even:
            return sqrt(z)/2*((sqrt(z) - 1)**(2*a)*exp(2*pi*I*a*(n - 1)) -
                              (sqrt(z) + 1)**(2*a)*exp(2*pi*I*a*n))
        else:
            n -= 1
            return sqrt(z)/2*((sqrt(z) - 1)**(2*a)*exp(2*pi*I*a*(n + 1)) -
                              (sqrt(z) + 1)**(2*a)*exp(2*pi*I*a*n))

    def _expr_big_minus(cls, a, z, n):
        # Compute the expression for large -z based on n being even or odd:
        if n.is_even:
            return (1 + z)**a*exp(2*pi*I*n*a)*sqrt(z)*sin(2*a*atan(sqrt(z)))
        else:
            return (1 + z)**a*exp(2*pi*I*n*a)*sqrt(z) \
                *sin(2*a*atan(sqrt(z)) - 2*pi*a)


class HyperRep_log2(HyperRep):
    """ Represent log(1/2 + sqrt(1 - z)/2) == -z/4*hyper([3/2, 1, 1], [2, 2], z) """

    @classmethod
    def _expr_small(cls, z):
        # Compute the expression for small z:
        return log(S.Half + sqrt(1 - z)/2)

    @classmethod
    def _expr_small_minus(cls, z):
        # Compute the expression for small -z:
        return log(S.Half + sqrt(1 + z)/2)

    @classmethod
    def _expr_big(cls, z, n):
        # Compute the expression for large z based on n being even or odd:
        if n.is_even:
            return (n - S.Half)*pi*I + log(sqrt(z)/2) + I*asin(1/sqrt(z))
        else:
            return (n - S.Half)*pi*I + log(sqrt(z)/2) - I*asin(1/sqrt(z))

    def _expr_big_minus(cls, z, n):
        # Compute the expression for large -z based on n being even or odd:
        if n.is_even:
            return pi*I*n + log(S.Half + sqrt(1 + z)/2)
        else:
            return pi*I*n + log(sqrt(1 + z)/2 - S.Half)


class HyperRep_cosasin(HyperRep):
    """ Represent hyper([a, -a], [1/2], z) == cos(2*a*asin(sqrt(z))). """
    # Note there are many alternative expressions, e.g. as powers of a sum of
    # square roots.

    @classmethod
    def _expr_small(cls, a, z):
        # Compute the expression for small z:
        return cos(2*a*asin(sqrt(z)))

    @classmethod
    def _expr_small_minus(cls, a, z):
        # Compute the expression for small -z:
        return cosh(2*a*asinh(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        # Compute the expression for large z:
        return cosh(2*a*acosh(sqrt(z)) + a*pi*I*(2*n - 1))

    @classmethod
    def _expr_big_minus(cls, a, z, n):
        # Compute the expression for large -z:
        return cosh(2*a*asinh(sqrt(z)) + 2*a*pi*I*n)


class HyperRep_sinasin(HyperRep):
    """ Represent 2*a*z*hyper([1 - a, 1 + a], [3/2], z)
        == sqrt(z)/sqrt(1-z)*sin(2*a*asin(sqrt(z))) """

    @classmethod
    def _expr_small(cls, a, z):
        # Compute the expression for small z:
        return sqrt(z)/sqrt(1 - z)*sin(2*a*asin(sqrt(z)))

    @classmethod
    def _expr_small_minus(cls, a, z):
        # Compute the expression for small -z:
        return -sqrt(z)/sqrt(1 + z)*sinh(2*a*asinh(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        # Compute the expression for large z:
        return -1/sqrt(1 - 1/z)*sinh(2*a*acosh(sqrt(z)) + a*pi*I*(2*n - 1))

    @classmethod
    # 定义一个类方法 `_expr_big_minus`，用于计算表达式的负数值
    def _expr_big_minus(cls, a, z, n):
        # 返回表达式 -1/sqrt(1 + 1/z) * sinh(2*a*asinh(sqrt(z)) + 2*a*pi*I*n) 的值
        return -1/sqrt(1 + 1/z)*sinh(2*a*asinh(sqrt(z)) + 2*a*pi*I*n)
# 定义一个 Appell 超几何函数的类，继承自 sympy 的 Function 类
class appellf1(Function):
    r"""
    This is the Appell hypergeometric function of two variables as:

    .. math ::
        F_1(a,b_1,b_2,c,x,y) = \sum_{m=0}^{\infty} \sum_{n=0}^{\infty}
        \frac{(a)_{m+n} (b_1)_m (b_2)_n}{(c)_{m+n}}
        \frac{x^m y^n}{m! n!}.

    Examples
    ========

    >>> from sympy import appellf1, symbols
    >>> x, y, a, b1, b2, c = symbols('x y a b1 b2 c')
    >>> appellf1(2., 1., 6., 4., 5., 6.)
    0.0063339426292673
    >>> appellf1(12., 12., 6., 4., 0.5, 0.12)
    172870711.659936
    >>> appellf1(40, 2, 6, 4, 15, 60)
    appellf1(40, 2, 6, 4, 15, 60)
    >>> appellf1(20., 12., 10., 3., 0.5, 0.12)
    15605338197184.4
    >>> appellf1(40, 2, 6, 4, x, y)
    appellf1(40, 2, 6, 4, x, y)
    >>> appellf1(a, b1, b2, c, x, y)
    appellf1(a, b1, b2, c, x, y)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Appell_series
    .. [2] https://functions.wolfram.com/HypergeometricFunctions/AppellF1/

    """

    @classmethod
    # 类方法，用于计算 Appell 函数的值
    def eval(cls, a, b1, b2, c, x, y):
        # 检查 b1 和 b2 的默认排序，确保 b1 <= b2，如果不是，则交换它们
        if default_sort_key(b1) > default_sort_key(b2):
            b1, b2 = b2, b1
            x, y = y, x  # 交换 x 和 y 的值
            return cls(a, b1, b2, c, x, y)  # 递归调用 eval 方法
        # 如果 b1 == b2 且 x > y 的默认排序，则交换 x 和 y 的值
        elif b1 == b2 and default_sort_key(x) > default_sort_key(y):
            x, y = y, x
            return cls(a, b1, b2, c, x, y)  # 递归调用 eval 方法
        # 如果 x 和 y 都等于 0，则返回 1
        if x == 0 and y == 0:
            return S.One

    # 计算偏导数的方法
    def fdiff(self, argindex=5):
        # 获取函数的参数 a, b1, b2, c, x, y
        a, b1, b2, c, x, y = self.args
        # 根据参数索引 argindex 计算偏导数
        if argindex == 5:
            return (a*b1/c)*appellf1(a + 1, b1 + 1, b2, c + 1, x, y)
        elif argindex == 6:
            return (a*b2/c)*appellf1(a + 1, b1, b2 + 1, c + 1, x, y)
        elif argindex in (1, 2, 3, 4):
            return Derivative(self, self.args[argindex-1])  # 返回关于对应参数的导数
        else:
            raise ArgumentIndexError(self, argindex)  # 报错，表示参数索引无效
```