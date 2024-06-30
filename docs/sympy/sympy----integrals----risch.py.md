# `D:\src\scipysrc\sympy\sympy\integrals\risch.py`

```
from types import GeneratorType
from functools import reduce

from sympy.core.function import Lambda  # 导入Lambda函数对象
from sympy.core.mul import Mul  # 导入Mul类，用于表示乘法操作
from sympy.core.intfunc import ilcm  # 导入ilcm函数，计算整数最小公倍数
from sympy.core.numbers import I  # 导入虚数单位I
from sympy.core.power import Pow  # 导入Pow类，用于表示幂运算
from sympy.core.relational import Ne  # 导入Ne类，用于表示不等式
from sympy.core.singleton import S  # 导入S对象，表示符号无穷小量
from sympy.core.sorting import ordered, default_sort_key  # 导入排序相关函数
from sympy.core.symbol import Dummy, Symbol  # 导入Dummy和Symbol类，用于表示符号变量
from sympy.functions.elementary.exponential import log, exp  # 导入对数和指数函数
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh,  # 导入双曲函数
    tanh)
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.functions.elementary.trigonometric import (atan, sin, cos,  # 导入三角函数
    tan, acot, cot, asin, acos)
from .integrals import integrate, Integral  # 导入积分相关函数和积分对象
from .heurisch import _symbols  # 导入Risch算法的符号集合
from sympy.polys.polyerrors import PolynomialError  # 导入多项式错误类
from sympy.polys.polytools import (real_roots, cancel, Poly, gcd,  # 导入多项式相关工具函数
    reduced)
from sympy.polys.rootoftools import RootSum  # 导入根式和工具类
from sympy.utilities.iterables import numbered_symbols  # 导入用于生成编号符号的函数


def integer_powers(exprs):
    """
    Rewrites a list of expressions as integer multiples of each other.

    Explanation
    ===========

    For example, if you have [x, x/2, x**2 + 1, 2*x/3], then you can rewrite
    this as [(x/6) * 6, (x/6) * 3, (x**2 + 1) * 1, (x/6) * 4]. This is useful
    in the Risch integration algorithm, where we must write exp(x) + exp(x/2)

    参数：
    exprs -- 表达式列表，需要重写为整数倍的形式

    返回：
    重写后的表达式列表

    """
    pass  # 函数体未实现，保留pass用于占位
    """
    as (exp(x/2))**2 + exp(x/2), but not as exp(x) + sqrt(exp(x)) (this is
    because only the transcendental case is implemented and we therefore cannot
    integrate algebraic extensions). The integer multiples returned by this
    function for each term are the smallest possible (their content equals 1).

    Returns a list of tuples where the first element is the base term and the
    second element is a list of `(item, factor)` terms, where `factor` is the
    integer multiplicative factor that must multiply the base term to obtain
    the original item.

    The easiest way to understand this is to look at an example:

    >>> from sympy.abc import x
    >>> from sympy.integrals.risch import integer_powers
    >>> integer_powers([x, x/2, x**2 + 1, 2*x/3])
    [(x/6, [(x, 6), (x/2, 3), (2*x/3, 4)]), (x**2 + 1, [(x**2 + 1, 1)])]

    We can see how this relates to the example at the beginning of the
    docstring.  It chose x/6 as the first base term.  Then, x can be written as
    (x/2) * 2, so we get (0, 2), and so on. Now only element (x**2 + 1)
    remains, and there are no other terms that can be written as a rational
    multiple of that, so we get that it can be written as (x**2 + 1) * 1.

    """
    # Here is the strategy:

    # First, go through each term and determine if it can be rewritten as a
    # rational multiple of any of the terms gathered so far.
    # cancel(a/b).is_Rational is sufficient for this.  If it is a multiple, we
    # add its multiple to the dictionary.

    terms = {}
    for term in exprs:
        for trm, trm_list in terms.items():
            a = cancel(term/trm)
            if a.is_Rational:
                trm_list.append((term, a))
                break
        else:
            terms[term] = [(term, S.One)]

    # After we have done this, we have all the like terms together, so we just
    # need to find a common denominator so that we can get the base term and
    # integer multiples such that each term can be written as an integer
    # multiple of the base term, and the content of the integers is 1.

    newterms = {}
    for term, term_list in terms.items():
        common_denom = reduce(ilcm, [i.as_numer_denom()[1] for _, i in
            term_list])
        newterm = term/common_denom
        newmults = [(i, j*common_denom) for i, j in term_list]
        newterms[newterm] = newmults

    # Return the sorted list of tuples, where each tuple consists of the base
    # term and its corresponding list of (item, factor) terms.
    return sorted(iter(newterms.items()), key=lambda item: item[0].sort_key())
# DifferentialExtension 类是一个包含与微分扩展相关信息的容器。

"""
A container for all the information relating to a differential extension.

Explanation
===========

The attributes of this object are (see also the docstring of __init__):

- f: The original (Expr) integrand.
- x: The variable of integration.
- T: List of variables in the extension.
- D: List of derivations in the extension; corresponds to the elements of T.
- fa: Poly of the numerator of the integrand.
- fd: Poly of the denominator of the integrand.
- Tfuncs: Lambda() representations of each element of T (except for x).
  For back-substitution after integration.
- backsubs: A (possibly empty) list of further substitutions to be made on
  the final integral to make it look more like the integrand.
- exts:
- extargs:
- cases: List of string representations of the cases of T.
- t: The top level extension variable, as defined by the current level
  (see level below).
- d: The top level extension derivation, as defined by the current
  derivation (see level below).
- case: The string representation of the case of self.d.
(Note that self.T and self.D will always contain the complete extension,
regardless of the level.  Therefore, you should ALWAYS use DE.t and DE.d
instead of DE.T[-1] and DE.D[-1].  If you want to have a list of the
derivations or variables only up to the current level, use
DE.D[:len(DE.D) + DE.level + 1] and DE.T[:len(DE.T) + DE.level + 1].  Note
that, in particular, the derivation() function does this.)

The following are also attributes, but will probably not be useful other
than in internal use:
- newf: Expr form of fa/fd.
- level: The number (between -1 and -len(self.T)) such that
  self.T[self.level] == self.t and self.D[self.level] == self.d.
  Use the methods self.increment_level() and self.decrement_level() to change
  the current level.
"""

# __slots__ 被定义主要是为了能够轻松地迭代类的所有属性（内存使用不太重要，因为我们每次只创建一个 DifferentialExtension 对象）。此外，在调试时有一个保护措施也很好。
__slots__ = ('f', 'x', 'T', 'D', 'fa', 'fd', 'Tfuncs', 'backsubs',
    'exts', 'extargs', 'cases', 'case', 't', 'd', 'newf', 'level',
    'ts', 'dummy')

def __getattr__(self, attr):
    # 当调试时避免 AttributeError
    if attr not in self.__slots__:
        raise AttributeError("%s has no attribute %s" % (repr(self), repr(attr)))
    return None
    def _rewrite_logs(self, logs, symlogs):
        """
        Rewrite logs for better processing.
        """
        # 获得 self.newf 的原子，并将其用于更新 logs
        atoms = self.newf.atoms(log)
        logs = update_sets(logs, atoms,
            lambda i: i.args[0].is_rational_function(*self.T) and
            i.args[0].has(*self.T))
        # 使用 atoms 更新 symlogs，处理符号日志的特定情况
        symlogs = update_sets(symlogs, atoms,
            lambda i: i.has(*self.T) and i.args[0].is_Pow and
            i.args[0].base.is_rational_function(*self.T) and
            not i.args[0].exp.is_Integer)

        # 处理符号日志列表
        # 可以处理类似 log(x**y) 的情况，通过转换为 y*log(x)
        # 这不仅修复了参数的符号指数，还修复了任何非整数指数，如 log(sqrt(x))
        # 指数也可以依赖于 x，如 log(x**x)
        for i in ordered(symlogs):
            # 与上面指数的情况不同，这里我们不会添加新的单项式（上面我们需要添加 log(a)）
            # 因此，不需要运行任何 is_deriv 函数。只需将 log(a**b) 转换为 b*log(a)
            # 然后让 log_new_extension() 处理后续步骤。
            lbase = log(i.args[0].base)
            logs.append(lbase)
            new = i.args[0].exp * lbase
            self.newf = self.newf.xreplace({i: new})
            self.backsubs.append((new, i))

        # 移除 logs 中的重复项并按默认排序键排序
        logs = sorted(set(logs), key=default_sort_key)

        # 返回处理后的 logs 和 symlogs
        return logs, symlogs

    def _auto_attrs(self):
        """
        Set attributes that are generated automatically.
        """
        # 如果 self.T 为空，则使用扩展标志并且未给出 T
        if not self.T:
            self.T = [i.gen for i in self.D]
        # 如果 self.x 未设置，则使用 self.T 的第一个元素作为默认值
        if not self.x:
            self.x = self.T[0]
        # 为每个 (d, t) 对应生成 case 列表
        self.cases = [get_case(d, t) for d, t in zip(self.D, self.T)]
        # 初始化 level、t、d 和 case 属性
        self.level = -1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]
    def _log_part(self, logs):
        """
        Try to build a logarithmic extension.

        Returns
        =======

        Returns True if there was a new extension and False if there was no new
        extension but it was able to rewrite the given logarithms in terms
        of the existing extension.  Unlike with exponential extensions, there
        is no way that a logarithm is not transcendental over and cannot be
        rewritten in terms of an already existing extension in a non-algebraic
        way, so this function does not ever return None or raise
        NotImplementedError.
        """
        from .prde import is_deriv_k  # 导入从文件 .prde 中的 is_deriv_k 函数

        new_extension = False  # 初始化一个变量，用于标记是否有新的扩展

        logargs = [i.args[0] for i in logs]  # 提取 logs 中每个元素的第一个参数，组成列表 logargs
        for arg in ordered(logargs):  # 对 logargs 中的参数按顺序进行迭代

            # 尝试将参数 arg 分数化
            arga, argd = frac_in(arg, self.t)

            # 调用 is_deriv_k 函数判断是否可以将 arga 视为导数的形式
            A = is_deriv_k(arga, argd, self)

            if A is not None:  # 如果 A 不为 None，表示找到了合适的导数形式
                ans, u, const = A
                newterm = log(const) + u  # 构造新的项，以 log(const) + u 形式
                self.newf = self.newf.xreplace({log(arg): newterm})  # 使用新项替换 self.newf 中的 log(arg)
                continue

            else:  # 如果无法通过 is_deriv_k 函数找到合适的导数形式
                arga, argd = frac_in(arg, self.t)
                darga = (argd * derivation(Poly(arga, self.t), self) -
                         arga * derivation(Poly(argd, self.t), self))
                dargd = argd ** 2
                darg = darga.as_expr() / dargd.as_expr()

                self.t = next(self.ts)  # 获取下一个 t 值并更新 self.t
                self.T.append(self.t)  # 将当前 t 值加入到 T 列表中
                self.extargs.append(arg)  # 将当前参数 arg 加入到 extargs 列表中
                self.exts.append('log')  # 将字符串 'log' 加入到 exts 列表中，表示是对数类型扩展
                self.D.append(cancel(darg.as_expr() / arg).as_poly(self.t, expand=False))  # 计算并加入 D 列表中的表达式
                if self.dummy:
                    i = Dummy("i")
                else:
                    i = Symbol('i')
                self.Tfuncs += [Lambda(i, log(arg.subs(self.x, i)))]  # 构造 Lambda 函数并加入到 Tfuncs 列表中
                self.newf = self.newf.xreplace({log(arg): self.t})  # 在 self.newf 中使用 self.t 替换 log(arg)
                new_extension = True  # 标记找到了新的扩展

        return new_extension  # 返回是否找到了新的扩展的标志

    @property
    def _important_attrs(self):
        """
        Returns some of the more important attributes of self.

        Explanation
        ===========

        Used for testing and debugging purposes.

        The attributes are (fa, fd, D, T, Tfuncs, backsubs,
        exts, extargs).
        """
        return (self.fa, self.fd, self.D, self.T, self.Tfuncs,
                self.backsubs, self.exts, self.extargs)

    # NOTE: this printing doesn't follow the Python's standard
    # eval(repr(DE)) == DE, where DE is the DifferentialExtension object,
    # also this printing is supposed to contain all the important
    # attributes of a DifferentialExtension object
    # 返回对象的字符串表示形式，包含类中非生成器对象的属性
    def __repr__(self):
        r = [(attr, getattr(self, attr)) for attr in self.__slots__
                if not isinstance(getattr(self, attr), GeneratorType)]
        return self.__class__.__name__ + '(dict(%r))' % (r)

    # 返回对象的漂亮字符串表示形式，格式化包含特定属性的信息
    def __str__(self):
        return (self.__class__.__name__ + '({fa=%s, fd=%s, D=%s})' %
                (self.fa, self.fd, self.D))

    # 比较两个对象是否相等，遍历所有的__slots__属性，确保非生成器类型的属性相等
    def __eq__(self, other):
        for attr in self.__class__.__slots__:
            d1, d2 = getattr(self, attr), getattr(other, attr)
            if not (isinstance(d1, GeneratorType) or d1 == d2):
                return False
        return True

    def reset(self):
        """
        将对象重置为初始状态，用于__init__方法中调用。
        """
        self.t = self.x
        self.T = [self.x]
        self.D = [Poly(1, self.x)]
        self.level = -1
        self.exts = [None]
        self.extargs = [None]
        if self.dummy:
            self.ts = numbered_symbols('t', cls=Dummy)
        else:
            # 用于测试
            self.ts = numbered_symbols('t')
        # 用于各种需要改变以使事物正常工作的变量，在完成时需要还原。
        self.backsubs = []
        self.Tfuncs = []
        self.newf = self.f

    def indices(self, extension):
        """
        返回包含特定扩展类型的索引列表。

        Parameters
        ==========

        extension : str
            表示有效扩展类型的字符串。

        Returns
        =======

        list
            'exts' 中扩展类型为 'extension' 的索引列表。

        Examples
        ========

        >>> from sympy.integrals.risch import DifferentialExtension
        >>> from sympy import log, exp
        >>> from sympy.abc import x
        >>> DE = DifferentialExtension(log(x) + exp(x), x, handle_first='exp')
        >>> DE.indices('log')
        [2]
        >>> DE.indices('exp')
        [1]

        """
        return [i for i, ext in enumerate(self.exts) if ext == extension]

    def increment_level(self):
        """
        增加对象的级别。

        Explanation
        ===========

        这会增加工作差分扩展的大小。self.level 相对于列表末端给出 (-1, -2 等)，
        因此在构建扩展时不需要担心它。
        """
        if self.level >= -1:
            raise ValueError("The level of the differential extension cannot "
                "be incremented any further.")

        self.level += 1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]
        return None
    def decrement_level(self):
        """
        Decrease the level of self.

        Explanation
        ===========

        This method reduces the level of the differential extension currently
        being worked on. The level indicates the position relative to the end of
        the list (-1, -2, etc.).

        Raises a ValueError if the current level is already at the end of the list
        (i.e., <= -len(self.T)), indicating that further decrementing is not possible.
        """

        # Check if the current level is already at the minimum possible level
        if self.level <= -len(self.T):
            raise ValueError("The level of the differential extension cannot "
                "be decremented any further.")

        # Decrease the level by 1
        self.level -= 1

        # Update attributes based on the new level
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]

        # Method does not return any value explicitly (returns None)
        return None
def update_sets(seq, atoms, func):
    # 将输入的序列转换为集合
    s = set(seq)
    # 计算集合 `s` 和 `atoms` 的交集
    s = atoms.intersection(s)
    # 计算 `atoms` 中不在 `s` 中的元素
    new = atoms - s
    # 对 `new` 中满足条件 `func` 的元素进行过滤，并添加到集合 `s` 中
    s.update(list(filter(func, new)))
    # 将集合 `s` 转换为列表并返回
    return list(s)


class DecrementLevel:
    """
    A context manager for decrementing the level of a DifferentialExtension.
    """
    __slots__ = ('DE',)

    def __init__(self, DE):
        # 初始化上下文管理器，保存 DifferentialExtension 的引用
        self.DE = DE
        return

    def __enter__(self):
        # 在进入上下文时，减小 DifferentialExtension 的级别
        self.DE.decrement_level()

    def __exit__(self, exc_type, exc_value, traceback):
        # 在退出上下文时，增加 DifferentialExtension 的级别
        self.DE.increment_level()


class NonElementaryIntegralException(Exception):
    """
    Exception used by subroutines within the Risch algorithm to indicate to one
    another that the function being integrated does not have an elementary
    integral in the given differential field.
    """
    # TODO: Rewrite algorithms below to use this (?)

    # TODO: Pass through information about why the integral was nonelementary,
    # and store that in the resulting NonElementaryIntegral somehow.
    pass


def gcdex_diophantine(a, b, c):
    """
    Extended Euclidean Algorithm, Diophantine version.

    Explanation
    ===========

    Given ``a``, ``b`` in K[x] and ``c`` in (a, b), the ideal generated by ``a`` and
    ``b``, return (s, t) such that s*a + t*b == c and either s == 0 or s.degree()
    < b.degree().
    """
    # 扩展欧几里得算法（丢番图版本），参考文献中的第 13 页
    # TODO: This should go in densetools.py.
    # XXX: Bettter name?

    # 使用 `half_gcdex` 方法计算 `a` 和 `b` 的半扩展欧几里得算法
    s, g = a.half_gcdex(b)
    # 计算 `s * c / g`，这里的 `exquo` 表示精确除法，如果结果不在 (a, b) 中则表示 `c` 不在 (a, b) 中
    s *= c.exquo(g)
    # 如果 `s` 不为零且 `s` 的次数大于等于 `b` 的次数，则执行除法操作
    if s and s.degree() >= b.degree():
        _, s = s.div(b)
    # 计算 `t = (c - s*a) / b`
    t = (c - s*a).exquo(b)
    return (s, t)


def frac_in(f, t, *, cancel=False, **kwargs):
    """
    Returns the tuple (fa, fd), where fa and fd are Polys in t.

    Explanation
    ===========

    This is a common idiom in the Risch Algorithm functions, so we abstract
    it out here. ``f`` should be a basic expression, a Poly, or a tuple (fa, fd),
    where fa and fd are either basic expressions or Polys, and f == fa/fd.
    **kwargs are applied to Poly.
    """
    # 如果 `f` 是元组，则将其分解为分子 `fa` 和分母 `fd`
    if isinstance(f, tuple):
        fa, fd = f
        f = fa.as_expr() / fd.as_expr()
    # 将 `f` 转换为分子分母形式，并确保分子 `fa` 和分母 `fd` 是多项式
    fa, fd = f.as_expr().as_numer_denom()
    fa, fd = fa.as_poly(t, **kwargs), fd.as_poly(t, **kwargs)
    # 如果指定了 `cancel` 参数，则进行约分操作
    if cancel:
        fa, fd = fa.cancel(fd, include=True)
    # 如果分子或分母为 `None`，则抛出值错误
    if fa is None or fd is None:
        raise ValueError("Could not turn %s into a fraction in %s." % (f, t))
    return (fa, fd)


def as_poly_1t(p, t, z):
    """
    (Hackish) way to convert an element ``p`` of K[t, 1/t] to K[t, z].

    In other words, ``z == 1/t`` will be a dummy variable that Poly can handle
    better.

    See issue 5131.

    Examples
    ========

    >>> from sympy import random_poly
    >>> from sympy.integrals.risch import as_poly_1t
    >>> from sympy.abc import x, z

    >>> p1 = random_poly(x, 10, -10, 10)
    >>> p2 = random_poly(x, 10, -10, 10)
    >>> p = p1 + p2.subs(x, 1/x)
    """
    # TODO: 在最终结果上使用这个函数。这样我们可以避免出现类似于(...)*exp(-x)的答案。
    # 将多项式 p 关于 t 进行分式分解，尝试取消公因子。
    pa, pd = frac_in(p, t, cancel=True)

    # 如果 pd 不是单项式（即不是 t 的整数次幂），则抛出多项式错误。
    # 这种情况通常来自 Risch 算法，表明存在 bug。
    if not pd.is_monomial:
        raise PolynomialError("%s is not an element of K[%s, 1/%s]." % (p, t, t))

    # 将 pa 对 pd 进行除法，得到商和余数。
    t_part, remainder = pa.div(pd)

    # 将 t_part 转换成关于 t 和 z 的多项式，不进行展开。
    ans = t_part.as_poly(t, z, expand=False)

    # 如果余数 remainder 非空，则进行进一步处理。
    if remainder:
        # 获取 remainder 的领头项系数 one，并构造 t*one。
        one = remainder.one
        tp = t * one
        # 计算 pd 和 remainder 的次数之差。
        r = pd.degree() - remainder.degree()
        # 构造关于 tp 的 remainder 的 z 部分，并将 t 替换为 z。
        z_part = remainder.transform(one, tp) * tp**r
        # 将 z_part 替换 t 为 z 后转换为域上的多项式，并对 pd 的领头系数取商。
        z_part = z_part.replace(t, z).to_field().quo_ground(pd.LC())
        # 将 z_part 添加到答案 ans 中。
        ans += z_part.as_poly(t, z, expand=False)

    # 返回最终的多项式表达式答案。
    return ans
def derivation(p, DE, coefficientD=False, basic=False):
    """
    Computes Dp.

    Explanation
    ===========

    Given the derivation D with D = d/dx and p is a polynomial in t over
    K(x), return Dp.

    If coefficientD is True, it computes the derivation kD
    (kappaD), which is defined as kD(sum(ai*Xi**i, (i, 0, n))) ==
    sum(Dai*Xi**i, (i, 1, n)) (Definition 3.2.2, page 80).  X in this case is
    T[-1], so coefficientD computes the derivative just with respect to T[:-1],
    with T[-1] treated as a constant.

    If ``basic=True``, the returns a Basic expression.  Elements of D can still be
    instances of Poly.
    """
    if basic:
        r = 0  # Initialize result as zero if basic flag is set
    else:
        r = Poly(0, DE.t)  # Initialize result as a polynomial with zero coefficients in terms of DE.t

    t = DE.t  # Assign DE.t to variable t
    if coefficientD:
        if DE.level <= -len(DE.T):
            # 'base' case, the answer is 0.
            return r  # Return the initialized result if coefficientD flag and DE.level conditions are met
        DE.decrement_level()  # Decrease the level attribute of DE

    D = DE.D[:len(DE.D) + DE.level + 1]  # Slice DE.D based on DE.level
    T = DE.T[:len(DE.T) + DE.level + 1]  # Slice DE.T based on DE.level

    for d, v in zip(D, T):
        pv = p.as_poly(v)  # Convert p into a polynomial in variable v
        if pv is None or basic:
            pv = p.as_expr()  # If pv is None or basic flag is set, convert p into a symbolic expression

        if basic:
            r += d.as_expr() * pv.diff(v)  # If basic flag is set, add the product of d as an expression and the derivative of pv with respect to v to r
        else:
            r += (d.as_expr() * pv.diff(v).as_expr()).as_poly(t)  # Otherwise, add the product of d as an expression and the derivative of pv with respect to v, converted to a polynomial in t, to r

    if basic:
        r = cancel(r)  # If basic flag is set, cancel out common factors in r

    if coefficientD:
        DE.increment_level()  # Increment the level attribute of DE if coefficientD flag is set

    return r  # Return the computed result


def get_case(d, t):
    """
    Returns the type of the derivation d.

    Returns one of {'exp', 'tan', 'base', 'primitive', 'other_linear',
    'other_nonlinear'}.
    """
    if not d.expr.has(t):
        if d.is_one:
            return 'base'  # Return 'base' if d does not contain t and is equal to 1
        return 'primitive'  # Otherwise, return 'primitive'

    if d.rem(Poly(t, t)).is_zero:
        return 'exp'  # Return 'exp' if d modulo t^2 is zero

    if d.rem(Poly(1 + t**2, t)).is_zero:
        return 'tan'  # Return 'tan' if d modulo (1 + t^2) is zero

    if d.degree(t) > 1:
        return 'other_nonlinear'  # Return 'other_nonlinear' if the degree of d with respect to t is greater than 1

    return 'other_linear'  # Default case, return 'other_linear'


def splitfactor(p, DE, coefficientD=False, z=None):
    """
    Splitting factorization.

    Explanation
    ===========

    Given a derivation D on k[t] and ``p`` in k[t], return (p_n, p_s) in
    k[t] x k[t] such that p = p_n*p_s, p_s is special, and each square
    factor of p_n is normal.

    Page. 100
    """
    kinv = [1/x for x in DE.T[:DE.level]]  # Create a list kinv containing 1/x for each x in DE.T up to DE.level
    if z:
        kinv.append(z)  # If z is provided, append it to kinv

    One = Poly(1, DE.t, domain=p.get_domain())  # Create a polynomial One equal to 1 in terms of DE.t and with the domain of p
    Dp = derivation(p, DE, coefficientD=coefficientD)  # Compute the derivation of p with respect to DE

    # XXX: Is this right?
    if p.is_zero:
        return (p, One)  # Return (p, One) if p is zero

    if not p.expr.has(DE.t):
        s = p.as_poly(*kinv).gcd(Dp.as_poly(*kinv)).as_poly(DE.t)  # Compute gcd of p and Dp in terms of kinv, then convert to polynomial in DE.t
        n = p.exquo(s)  # Compute quotient of p by s
        return (n, s)  # Return (n, s)

    if not Dp.is_zero:
        h = p.gcd(Dp).to_field()  # Compute gcd of p and Dp and convert to field
        g = p.gcd(p.diff(DE.t)).to_field()  # Compute gcd of p and its derivative with respect to DE.t and convert to field
        s = h.exquo(g)  # Compute quotient of h by g

        if s.degree(DE.t) == 0:
            return (p, One)  # Return (p, One) if degree of s with respect to DE.t is zero

        q_split = splitfactor(p.exquo(s), DE, coefficientD=coefficientD)  # Recursively compute splitfactor on p divided by s

        return (q_split[0], q_split[1] * s)  # Return the product of q_split[0] and q_split[1] multiplied by s

    else:
        return (p, One)  # Return (p, One) if Dp is zero


def splitfactor_sqf(p, DE, coefficientD=False, z=None, basic=False):
    """
    Splitting Square-free Factorization.

    Explanation
    ===========

    Given a derivation D on k[t] and ``p`` in k[t], return (p_n, p_s) in
    k[t] x k[t] such that p = p_n*p_s, p_s is special, and each square
    factor of p_n is normal.
    """
    """
    Given a derivation D on k[t] and ``p`` in k[t], returns (N1, ..., Nm)
    and (S1, ..., Sm) in k[t]^m such that p =
    (N1*N2**2*...*Nm**m)*(S1*S2**2*...*Sm**m) is a splitting
    factorization of ``p`` and the Ni and Si are square-free and coprime.
    """
    # TODO: This algorithm appears to be faster in every case
    # TODO: Verify this and splitfactor() for multiple extensions

    # Compute the reciprocal of each element in DE.T[:DE.level] and concatenate them with the original list
    kkinv = [1/x for x in DE.T[:DE.level]] + DE.T[:DE.level]

    # If z is given, replace kkinv with a list containing only z
    if z:
        kkinv = [z]

    # Initialize empty lists to store the square-free factors
    S = []
    N = []

    # Obtain the square-free factorization of p including multiplicities
    p_sqf = p.sqf_list_include()

    # If p is the zero polynomial, return a tuple with a nested tuple containing (p, 1) and an empty tuple
    if p.is_zero:
        return (((p, 1),), ())

    # Iterate over each square-free factor pi and its index i in the square-free factorization of p
    for pi, i in p_sqf:
        # Compute Si as the greatest common divisor of pi and its derivative with respect to DE, then convert to polynomial
        Si = pi.as_poly(*kkinv).gcd(derivation(pi, DE,
            coefficientD=coefficientD,basic=basic).as_poly(*kkinv)).as_poly(DE.t)
        
        # Convert pi and Si to polynomial objects
        pi = Poly(pi, DE.t)
        Si = Poly(Si, DE.t)

        # Compute Ni as the quotient of pi divided by Si
        Ni = pi.exquo(Si)

        # If Si is not equal to 1, append (Si, i) to list S
        if not Si.is_one:
            S.append((Si, i))
        
        # If Ni is not equal to 1, append (Ni, i) to list N
        if not Ni.is_one:
            N.append((Ni, i))

    # Return the result as a tuple of tuples for N and S
    return (tuple(N), tuple(S))
def canonical_representation(a, d, DE):
    """
    Canonical Representation.

    Explanation
    ===========

    Given a derivation D on k[t] and f = a/d in k(t), return (f_p, f_s,
    f_n) in k[t] x k(t) x k(t) such that f = f_p + f_s + f_n is the
    canonical representation of f (f_p is a polynomial, f_s is reduced
    (has a special denominator), and f_n is simple (has a normal
    denominator).
    """
    # 将 d 变为首一多项式
    l = Poly(1/d.LC(), DE.t)
    a, d = a.mul(l), d.mul(l)

    # 对 f = a/d 进行除法得到商 q 和余数 r
    q, r = a.div(d)

    # 对 d 进行分解为 dn 和 ds
    dn, ds = splitfactor(d, DE)

    # 调用 gcdex_diophantine 函数计算 b 和 c
    b, c = gcdex_diophantine(dn.as_poly(DE.t), ds.as_poly(DE.t), r.as_poly(DE.t))
    b, c = b.as_poly(DE.t), c.as_poly(DE.t)

    return (q, (b, ds), (c, dn))


def hermite_reduce(a, d, DE):
    """
    Hermite Reduction - Mack's Linear Version.

    Given a derivation D on k(t) and f = a/d in k(t), returns g, h, r in
    k(t) such that f = Dg + h + r, h is simple, and r is reduced.

    """
    # 将 d 变为首一多项式
    l = Poly(1/d.LC(), DE.t)
    a, d = a.mul(l), d.mul(l)

    # 调用 canonical_representation 函数获取 fp, fs, fn
    fp, fs, fn = canonical_representation(a, d, DE)
    a, d = fn

    # 再次将 d 变为首一多项式
    l = Poly(1/d.LC(), DE.t)
    a, d = a.mul(l), d.mul(l)

    # 初始化 ga 和 gd
    ga = Poly(0, DE.t)
    gd = Poly(1, DE.t)

    # 计算 d 的导数
    dd = derivation(d, DE)
    dm = gcd(d.to_field(), dd.to_field()).as_poly(DE.t)
    ds, _ = d.div(dm)

    # 进入循环，直到 dm 的次数为 0
    while dm.degree(DE.t) > 0:
        ddm = derivation(dm, DE)
        dm2 = gcd(dm.to_field(), ddm.to_field())
        dms, _ = dm.div(dm2)
        ds_ddm = ds.mul(ddm)
        ds_ddm_dm, _ = ds_ddm.div(dm)

        # 调用 gcdex_diophantine 函数计算 b 和 c
        b, c = gcdex_diophantine(-ds_ddm_dm.as_poly(DE.t),
            dms.as_poly(DE.t), a.as_poly(DE.t))
        b, c = b.as_poly(DE.t), c.as_poly(DE.t)

        db = derivation(b, DE).as_poly(DE.t)
        ds_dms, _ = ds.div(dms)
        a = c.as_poly(DE.t) - db.mul(ds_dms).as_poly(DE.t)

        ga = ga*dm + b*gd
        gd = gd*dm
        ga, gd = ga.cancel(gd, include=True)
        dm = dm2

    # 计算最终的商 q 和余数 r
    q, r = a.div(ds)
    ga, gd = ga.cancel(gd, include=True)

    # 化简余数 r
    r, d = r.cancel(ds, include=True)
    rra = q*fs[1] + fp*fs[1] + fs[0]
    rrd = fs[1]
    rra, rrd = rra.cancel(rrd, include=True)

    return ((ga, gd), (r, d), (rra, rrd))


def polynomial_reduce(p, DE):
    """
    Polynomial Reduction.

    Explanation
    ===========

    Given a derivation D on k(t) and p in k[t] where t is a nonlinear
    monomial over k, return q, r in k[t] such that p = Dq  + r, and
    deg(r) < deg_t(Dt).
    """
    # 初始化 q 为 0
    q = Poly(0, DE.t)

    # 当 p 的次数大于等于 DE.d 的次数时，执行循环
    while p.degree(DE.t) >= DE.d.degree(DE.t):
        m = p.degree(DE.t) - DE.d.degree(DE.t) + 1
        q0 = Poly(DE.t**m, DE.t).mul(Poly(p.as_poly(DE.t).LC()/
            (m*DE.d.LC()), DE.t))
        q += q0
        p = p - derivation(q0, DE)

    return (q, p)


def laurent_series(a, d, F, n, DE):
    """
    Contribution of ``F`` to the full partial fraction decomposition of A/D.

    Explanation
    ===========

    Given a field K of characteristic 0 and ``A``,``D``,``F`` in K[x] with D monic,
    ```
    # 如果 F 的次数为零，则返回 0
    if F.degree()==0:
        return 0
    
    # 创建符号变量列表 Z，其中包含 n 个变量 'z'，并添加一个额外的符号 'z'
    Z = _symbols('z', n)
    z = Symbol('z')
    Z.insert(0, z)
    
    # 初始化 delta_a 和 delta_d 为多项式 Poly(0, DE.t) 和 Poly(1, DE.t)
    delta_a = Poly(0, DE.t)
    delta_d = Poly(1, DE.t)

    # 计算 E = d.quo(F**n)，其中 d 是 F 除以 F**n 的商
    E = d.quo(F**n)
    ha, hd = (a, E*Poly(z**n, DE.t))
    
    # 计算 F 的导数 dF 和其与 F 的最大公因数 B 和 C
    dF = derivation(F, DE)
    B, _ = gcdex_diophantine(E, F, Poly(1,DE.t))
    C, _ = gcdex_diophantine(dF, F, Poly(1,DE.t))

    # 初始化
    F_store = F
    V, DE_D_list, H_list = [], [], []

    # 对于 j 从 0 到 n 的循环
    for j in range(0, n):
        # 计算 F 的 j 次导数 F_store
        F_store = derivation(F_store, DE)
        v = (F_store.as_expr())/(j + 1)
        V.append(v)
        DE_D_list.append(Poly(Z[j + 1], Z[j]))

    # 创建 DifferentialExtension 对象 DE_new
    DE_new = DifferentialExtension(extension={'D': DE_D_list})

    # 对于 j 从 0 到 n 的循环
    for j in range(0, n):
        # 计算 z^(n+j)*E^(j+1)*ha 和 hd
        zEha = Poly(z**(n + j), DE.t)*E**(j + 1)*ha
        zEhd = hd
        
        # 计算 Pa 和 Pd
        Pa, Pd = cancel((zEha, zEhd))[1], cancel((zEha, zEhd))[2]
        Q = Pa.quo(Pd)
        
        # 用 V[i] 替换 Q 中的 Z[i]，其中 i 从 0 到 j
        for i in range(0, j + 1):
            Q = Q.subs(Z[i], V[i])
        
        # 计算 Dha 和 Dhd
        Dha = (hd*derivation(ha, DE, basic=True).as_poly(DE.t)
             + ha*derivation(hd, DE, basic=True).as_poly(DE.t)
             + hd*derivation(ha, DE_new, basic=True).as_poly(DE.t)
             + ha*derivation(hd, DE_new, basic=True).as_poly(DE.t))
        Dhd = Poly(j + 1, DE.t)*hd**2
        ha, hd = Dha, Dhd
        
        # 计算 Ff，并将 Ff 除以 gcd(F, Q)
        Ff, _ = F.div(gcd(F, Q))
        F_stara, F_stard = frac_in(Ff, DE.t)
        
        # 如果 F_stara 的次数减去 F_stard 的次数大于 0，则进行以下操作
        if F_stara.degree(DE.t) - F_stard.degree(DE.t) > 0:
            # 计算 QBC 和 H
            QBC = Poly(Q, DE.t)*B**(1 + j)*C**(n + j)
            H = QBC
            H_list.append(H)
            H = (QBC*F_stard).rem(F_stara)
            alphas = real_roots(F_stara)
            
            # 对于 F_stara 的每个实根 alpha
            for alpha in list(alphas):
                delta_a = delta_a*Poly((DE.t - alpha)**(n - j), DE.t) + Poly(H.eval(alpha), DE.t)
                delta_d = delta_d*Poly((DE.t - alpha)**(n - j), DE.t)
    
    # 返回 delta_a, delta_d 和 H_list
    return (delta_a, delta_d, H_list)
# 判断给定多项式是否是有理函数的导数
def recognize_derivative(a, d, DE, z=None):
    # 设置标志为True
    flag = True
    # 对输入的多项式a和d进行约分，返回约分后的结果和余数
    a, d = a.cancel(d, include=True)
    # 计算多项式a除以d的商和余数
    _, r = a.div(d)
    # 将d分解为其平方自由分解形式
    Np, Sp = splitfactor_sqf(d, DE, coefficientD=True, z=z)

    j = 1
    # 遍历每个分解项
    for s, _ in Sp:
        # 计算Laurent级数，并返回相关的增量delta_a, delta_d和H
        delta_a, delta_d, H = laurent_series(r, d, s, j, DE)
        # 计算d和H[-1]的最大公约数，将结果表示为多项式g
        g = gcd(d, H[-1]).as_poly()
        # 如果g不等于d，则将标志设置为False并跳出循环
        if g is not d:
            flag = False
            break
        j = j + 1
    # 返回标志，指示多项式是否为有理函数的导数
    return flag


# 判断给定多项式是否是有理函数的对数导数
def recognize_log_derivative(a, d, DE, z=None):
    # 如果z未指定，则设置为虚拟符号'z'
    z = z or Dummy('z')
    # 对输入的多项式a和d进行约分，返回约分后的结果和余数
    a, d = a.cancel(d, include=True)
    _, a = a.div(d)

    # 创建关于z的多项式对象
    pz = Poly(z, DE.t)
    # 计算d的导数Dd
    Dd = derivation(d, DE)
    # 构造新的多项式q
    q = a - pz * Dd
    # 计算d与q的结果式，包括PRS
    r, _ = d.resultant(q, includePRS=True)
    r = Poly(r, z)
    # 将结果式r进行平方自由分解
    Np, Sp = splitfactor_sqf(r, DE, coefficientD=True, z=z)

    for s, _ in Sp:
        # 考虑实数根以外的复数根，如果有则返回False
        a = real_roots(s.as_poly(z))
        if not all(j.is_Integer for j in a):
            return False
    # 如果所有根都是整数，则返回True
    return True


# 执行Lazard-Rioboo-Rothstein-Trager剩余结果简化
def residue_reduce(a, d, DE, z=None, invert=True):
    """
    Lazard-Rioboo-Rothstein-Trager resultant reduction.

    Explanation
    ===========

    Given a derivation ``D`` on k(t) and f in k(t) simple, return g
    elementary over k(t) and a Boolean b in {True, False} such that f -
    Dg in k[t] if b == True or f + h and f + h - Dg do not have an
    elementary integral over k(t) for any h in k<t> (reduced) if b ==
    False.

    Returns (G, b), where G is a tuple of tuples of the form (s_i, S_i),
    such that g = Add(*[RootSum(s_i, lambda z: z*log(S_i(z, t))) for
    S_i, s_i in G]). f - Dg is the remaining integral, which is elementary
    only if b == True, and hence the integral of f is elementary only if
    b == True.

    f - Dg is not calculated in this function because that would require
    explicitly calculating the RootSum.  Use residue_reduce_derivation().
    """
    # TODO: Use log_to_atan() from rationaltools.py
    # 如果r = residue_reduce(...)，则对数部分由以下给出：
    # sum([RootSum(a[0].as_poly(z), lambda i: i*log(a[1].as_expr()).subs(z,
    # 如果 z 为空，则创建一个虚拟符号 'z'
    z = z or Dummy('z')
    # 对 a 和 d 进行因式分解并取消公因式，包括取消共同因子
    a, d = a.cancel(d, include=True)
    # 将 a 和 d 转换为域，并将它们的首项系数变为 1
    a, d = a.to_field().mul_ground(1/d.LC()), d.to_field().mul_ground(1/d.LC())
    # 生成 DE.T 中前 DE.level 个元素的倒数，并形成列表 kkinv
    kkinv = [1/x for x in DE.T[:DE.level]] + DE.T[:DE.level]

    # 如果 a 是零多项式，则返回空列表和 True
    if a.is_zero:
        return ([], True)
    # 计算 a 除以 d 的商和余数，将余数赋值给 a
    _, a = a.div(d)

    # 创建关于 z 的多项式 pz
    pz = Poly(z, DE.t)

    # 计算 d 的导数 Dd
    Dd = derivation(d, DE)
    # 计算多项式 q = a - pz*Dd
    q = a - pz*Dd

    # 根据条件选择使用 d.resultant(q) 或者 q.resultant(d)，并返回结果 r 和 R
    if Dd.degree(DE.t) <= d.degree(DE.t):
        r, R = d.resultant(q, includePRS=True)
    else:
        r, R = q.resultant(d, includePRS=True)

    # 创建空字典 R_map 和空列表 H
    R_map, H = {}, []

    # 将 R 中的多项式按照次数映射到 R_map 中
    for i in R:
        R_map[i.degree()] = i

    # 将 r 转换为关于 z 的多项式
    r = Poly(r, z)
    # 对 r 进行因式分解，返回 Np 和 Sp
    Np, Sp = splitfactor_sqf(r, DE, coefficientD=True, z=z)

    # 遍历 Sp 中的每对 (s, i)
    for s, i in Sp:
        # 如果 i 等于 d 关于 DE.t 的次数，则将 s 转换为首一多项式并加入 H
        if i == d.degree(DE.t):
            s = Poly(s, z).monic()
            H.append((s, d))
        else:
            # 否则，从 R_map 中获取次数为 i 的多项式 h
            h = R_map.get(i)
            if h is None:
                continue

            # 获取 h 的首项系数并将其转换为首一因子分解形式
            h_lc = Poly(h.as_poly(DE.t).LC(), DE.t, field=True)
            h_lc_sqf = h_lc.sqf_list_include(all=True)

            # 对 h_lc_sqf 中的每对 (a, j) 进行处理
            for a, j in h_lc_sqf:
                # 计算 gcd(a, s^j, *kkinv) 并将其作为多项式与 DE.t 变量的商
                h = Poly(h, DE.t, field=True).exquo(Poly(gcd(a, s**j, *kkinv), DE.t))

            # 将 s 转换为首一多项式
            s = Poly(s, z).monic()

            # 如果 invert 为真，则对 h 进行进一步处理
            if invert:
                h_lc = Poly(h.as_poly(DE.t).LC(), DE.t, field=True, expand=False)
                inv, coeffs = h_lc.as_poly(z, field=True).invert(s), [S.One]

                # 对 h 的系数进行处理并重新构建 h
                for coeff in h.coeffs()[1:]:
                    L = reduced(inv*coeff.as_poly(inv.gens), [s])[1]
                    coeffs.append(L.as_expr())

                h = Poly(dict(list(zip(h.monoms(), coeffs))), DE.t)

            # 将 (s, h) 加入 H
            H.append((s, h))

    # 判断是否存在任何 Np 中的多项式在其表达式中含有 DE.t 和 z
    b = not any(cancel(i.as_expr()).has(DE.t, z) for i, _ in Np)

    # 返回结果元组 (H, b)
    return (H, b)
def residue_reduce_to_basic(H, DE, z):
    """
    Converts the tuple returned by residue_reduce() into a Basic expression.
    """
    # TODO: check what Lambda does with RootOf
    # 创建一个虚拟变量 i
    i = Dummy('i')
    # 反转 DE.T 和 DE.Tfuncs 中的元素，并以元组的形式对齐
    s = list(zip(reversed(DE.T), reversed([f(DE.x) for f in DE.Tfuncs])))

    # 对于 H 中的每个元素 a，计算 RootSum，并将结果求和
    return sum(RootSum(a[0].as_poly(z), Lambda(i, i*log(a[1].as_expr()).subs(
        {z: i}).subs(s))) for a in H)


def residue_reduce_derivation(H, DE, z):
    """
    Computes the derivation of an expression returned by residue_reduce().

    In general, this is a rational function in t, so this returns an
    as_expr() result.
    """
    # TODO: verify that this is correct for multiple extensions
    # 创建一个虚拟变量 i
    i = Dummy('i')
    # 对于 H 中的每个元素 a，计算 RootSum，并将结果求和
    return S(sum(RootSum(a[0].as_poly(z), Lambda(i, i*derivation(a[1],
        DE).as_expr().subs(z, i)/a[1].as_expr().subs(z, i))) for a in H))


def integrate_primitive_polynomial(p, DE):
    """
    Integration of primitive polynomials.

    Explanation
    ===========

    Given a primitive monomial t over k, and ``p`` in k[t], return q in k[t],
    r in k, and a bool b in {True, False} such that r = p - Dq is in k if b is
    True, or r = p - Dq does not have an elementary integral over k(t) if b is
    False.
    """
    # 使用 Zero 和 q 初始化变量
    Zero = Poly(0, DE.t)
    q = Poly(0, DE.t)

    # 如果 p.expr 不包含 DE.t，则返回 (Zero, p, True)
    if not p.expr.has(DE.t):
        return (Zero, p, True)

    from .prde import limited_integrate
    # 循环处理直到 p.expr 不再包含 DE.t
    while True:
        if not p.expr.has(DE.t):
            return (q, p, True)

        # 计算 Dta 和 Dtb
        Dta, Dtb = frac_in(DE.d, DE.T[DE.level - 1])

        with DecrementLevel(DE):  # 我们最好使用 ratint() 来积分最低扩展 (x)
                                  # with ratint().
            # 计算 a，并将其转换为 aa 和 ad
            a = p.LC()
            aa, ad = frac_in(a, DE.t)

            try:
                # 尝试有限积分 limited_integrate
                rv = limited_integrate(aa, ad, [(Dta, Dtb)], DE)
                if rv is None:
                    raise NonElementaryIntegralException
                (ba, bd), c = rv
            except NonElementaryIntegralException:
                return (q, p, False)

        # 计算 p 的次数 m
        m = p.degree(DE.t)
        # 计算 q0
        q0 = c[0].as_poly(DE.t)*Poly(DE.t**(m + 1)/(m + 1), DE.t) + \
            (ba.as_expr()/bd.as_expr()).as_poly(DE.t)*Poly(DE.t**m, DE.t)

        # 更新 p 和 q
        p = p - derivation(q0, DE)
        q = q + q0


def integrate_primitive(a, d, DE, z=None):
    """
    Integration of primitive functions.

    Explanation
    ===========

    Given a primitive monomial t over k and f in k(t), return g elementary over
    k(t), i in k(t), and b in {True, False} such that i = f - Dg is in k if b
    is True or i = f - Dg does not have an elementary integral over k(t) if b
    is False.

    This function returns a Basic expression for the first argument.  If b is
    True, the second argument is Basic expression in k to recursively integrate.
    If b is False, the second argument is an unevaluated Integral, which has
    been proven to be nonelementary.
    """
    # XXX: a and d must be canceled, or this might return incorrect results
    # 如果 z 为 None，则创建一个虚拟变量 z
    z = z or Dummy("z")
    # 将 DE.T 的反转结果与 DE.Tfuncs 中每个函数应用于 DE.x 后的结果进行组合
    s = list(zip(reversed(DE.T), reversed([f(DE.x) for f in DE.Tfuncs])))

    # 对给定的参数 a, d 和 DE 进行 Hermite 归约操作，得到 g1, h 和 r
    g1, h, r = hermite_reduce(a, d, DE)

    # 对 h[0], h[1] 和 DE 进行残余约化操作，得到 g2 和 b
    g2, b = residue_reduce(h[0], h[1], DE, z=z)

    # 如果 b 为假（即 False）
    if not b:
        # 计算表达式 a/d - (g1[1]*derivation(g1[0], DE) - g1[0]*derivation(g1[1], DE))/(g1[1]**2) 的因式分解
        i = cancel(a.as_expr()/d.as_expr() - (g1[1]*derivation(g1[0], DE) -
            g1[0]*derivation(g1[1], DE)).as_expr()/(g1[1]**2).as_expr() -
            residue_reduce_derivation(g2, DE, z))
        # 将得到的结果转换为 NonElementaryIntegral 对象
        i = NonElementaryIntegral(cancel(i).subs(s), DE.x)
        # 返回 g1[0]/g1[1] 和 g2 的基本残余约化结果及 i, b
        return ((g1[0].as_expr()/g1[1].as_expr()).subs(s) +
            residue_reduce_to_basic(g2, DE, z), i, b)

    # 计算表达式 h[0]/h[1] - residue_reduce_derivation(g2, DE, z) + r[0]/r[1] 的因式分解
    p = cancel(h[0].as_expr()/h[1].as_expr() - residue_reduce_derivation(g2,
        DE, z) + r[0].as_expr()/r[1].as_expr())
    # 将结果转换为 DE.t 的多项式
    p = p.as_poly(DE.t)

    # 对 p 进行原始多项式的积分操作，得到 q, i 和 b
    q, i, b = integrate_primitive_polynomial(p, DE)

    # 计算表达式 g1[0]/g1[1] + q 的结果并应用 s，再加上 g2 的基本残余约化结果
    ret = ((g1[0].as_expr()/g1[1].as_expr() + q.as_expr()).subs(s) +
        residue_reduce_to_basic(g2, DE, z))

    # 如果 b 为假（即 False）
    if not b:
        # TODO: 当 b 为 False 时，此处并不会产生正确的结果
        i = NonElementaryIntegral(cancel(i.as_expr()).subs(s), DE.x)
    else:
        # 对 i 进行表达式的简化
        i = cancel(i.as_expr())

    # 返回 ret, i 和 b 的元组结果
    return (ret, i, b)
# 定义一个函数，用于积分超指数多项式
def integrate_hyperexponential_polynomial(p, DE, z):
    """
    Integration of hyperexponential polynomials.

    Explanation
    ===========

    Given a hyperexponential monomial t over k and ``p`` in k[t, 1/t], return q in
    k[t, 1/t] and a bool b in {True, False} such that p - Dq in k if b is True,
    or p - Dq does not have an elementary integral over k(t) if b is False.
    """
    # 获取变量 t1，DE.t 即为所需的变量 t
    t1 = DE.t
    # 计算 DE.d / t 的商，结果存储在 dtt 中
    dtt = DE.d.exquo(Poly(DE.t, DE.t))
    # 初始化 qa 和 qd 分别为 Poly(0, DE.t) 和 Poly(1, DE.t)
    qa = Poly(0, DE.t)
    qd = Poly(1, DE.t)
    # 初始化布尔变量 b 为 True
    b = True

    # 如果 p 是零多项式，则直接返回 qa, qd, b
    if p.is_zero:
        return (qa, qd, b)

    # 导入 rischDE 函数用于 Risch 算法的应用
    from sympy.integrals.rde import rischDE

    # 使用 DecrementLevel 类来限制一些处理
    with DecrementLevel(DE):
        # 循环处理多项式 p 的次数范围
        for i in range(-p.degree(z), p.degree(t1) + 1):
            # 如果 i 为 0，则继续下一次循环
            if not i:
                continue
            # 如果 i < 0，则从 p 中提取 z 的 i 次项系数 a
            elif i < 0:
                # 注意：如果出现 AttributeError: 'NoneType' object has no attribute 'nth'
                # 这应该是 expand=False 引起的问题，但是不应该发生，因为 p 已经是关于 t 和 z 的 Poly
                a = p.as_poly(z, expand=False).nth(-i)
            # 如果 i > 0，则从 p 中提取 t1 的 i 次项系数 a
            else:
                # 注意：如果出现 AttributeError: 'NoneType' object has no attribute 'nth'
                # 这应该是 expand=False 引起的问题，但是不应该发生，因为 p 已经是关于 t 和 z 的 Poly
                a = p.as_poly(t1, expand=False).nth(i)

            # 将 a 分解为分子和分母的有理函数形式 aa/ad
            aa, ad = frac_in(a, DE.t, field=True)
            # 对 aa/ad 进行约分
            aa, ad = aa.cancel(ad, include=True)
            # 构建 i*t1*dtt 的多项式形式
            iDt = Poly(i, t1)*dtt
            # 将 iDt 分解为分子和分母的有理函数形式 iDta/iDtd
            iDta, iDtd = frac_in(iDt, DE.t, field=True)

            try:
                # 尝试使用 rischDE 函数进行 Risch 算法的应用，获取 va/vd
                va, vd = rischDE(iDta, iDtd, Poly(aa, DE.t), Poly(ad, DE.t), DE)
                # 将 (va, vd) 分解为分子和分母的有理函数形式，并进行约分
                va, vd = frac_in((va, vd), t1, cancel=True)
            except NonElementaryIntegralException:
                # 如果捕获到 NonElementaryIntegralException 异常，则设置 b 为 False
                b = False
            else:
                # 否则更新 qa 和 qd 的值
                qa = qa*vd + va*Poly(t1**i)*qd
                qd *= vd

    # 返回结果 qa, qd, b
    return (qa, qd, b)


# 定义一个函数，用于积分超指数函数
def integrate_hyperexponential(a, d, DE, z=None, conds='piecewise'):
    """
    Integration of hyperexponential functions.

    Explanation
    ===========

    Given a hyperexponential monomial t over k and f in k(t), return g
    elementary over k(t), i in k(t), and a bool b in {True, False} such that
    i = f - Dg is in k if b is True or i = f - Dg does not have an elementary
    integral over k(t) if b is False.

    This function returns a Basic expression for the first argument.  If b is
    True, the second argument is Basic expression in k to recursively integrate.
    If b is False, the second argument is an unevaluated Integral, which has
    been proven to be nonelementary.
    """
    # XXX: a and d must be canceled, or this might return incorrect results
    # 如果 z 为 None，则创建一个虚拟符号 z
    z = z or Dummy("z")
    # 将 DE.T 和 DE.Tfuncs 的组合结果反转后存储在 s 中
    s = list(zip(reversed(DE.T), reversed([f(DE.x) for f in DE.Tfuncs])))

    # 调用 hermite_reduce 函数处理 a 和 d，得到 g1, h, r
    g1, h, r = hermite_reduce(a, d, DE)
    # 调用 residue_reduce 函数处理 h[0], h[1]，得到 g2, b
    g2, b = residue_reduce(h[0], h[1], DE, z=z)
    # 如果 b 不为真，则执行以下操作
    if not b:
        # 计算表达式 a / d - (g1[1] * derivation(g1[0], DE) - g1[0] * derivation(g1[1], DE)) / (g1[1]**2) - residue_reduce_derivation(g2, DE, z)
        i = cancel(a.as_expr() / d.as_expr() - (g1[1] * derivation(g1[0], DE) -
            g1[0] * derivation(g1[1], DE)).as_expr() / (g1[1]**2).as_expr() -
            residue_reduce_derivation(g2, DE, z))
        # 将 i 替换为其在符号 s 下的表达式，并封装成 NonElementaryIntegral 对象
        i = NonElementaryIntegral(cancel(i.subs(s)), DE.x)
        # 返回元组，包含三个元素
        return ((g1[0].as_expr() / g1[1].as_expr()).subs(s) +
            residue_reduce_to_basic(g2, DE, z), i, b)

    # p 应该是关于 t 和 1/t 的多项式，因为 Sirr == k[t, 1/t]
    # 计算 p = h[0] / h[1] - residue_reduce_derivation(g2, DE, z) + r[0] / r[1]
    p = cancel(h[0].as_expr() / h[1].as_expr() - residue_reduce_derivation(g2,
        DE, z) + r[0].as_expr() / r[1].as_expr())
    # 将 p 转换为关于 DE.t 和 z 的多项式
    pp = as_poly_1t(p, DE.t, z)

    # 对 pp 进行超指数多项式积分，返回结果为 qa, qd 和 b
    qa, qd, b = integrate_hyperexponential_polynomial(pp, DE, z)

    # 获取 p 在 t 的 0 次项系数
    i = pp.nth(0, 0)

    # 计算 ret = (g1[0] / g1[1]).subs(s) + residue_reduce_to_basic(g2, DE, z)
    ret = ((g1[0].as_expr() / g1[1].as_expr()).subs(s) \
        + residue_reduce_to_basic(g2, DE, z))

    # 计算 qa 和 qd 在符号 s 下的表达式
    qas = qa.as_expr().subs(s)
    qds = qd.as_expr().subs(s)
    # 如果 conds 等于 'piecewise' 并且 DE.x 不在 qds 的自由符号中
    if conds == 'piecewise' and DE.x not in qds.free_symbols:
        # 我们需要小心指数是否为 S.Zero！

        # XXX: 当 qd = 0 时，是否总是对应指数等于 1？
        ret += Piecewise(
                (qas / qds, Ne(qds, 0)),
                (integrate((p - i).subs(DE.t, 1).subs(s), DE.x), True)
            )
    else:
        # 否则直接将 qas / qds 加到 ret 中
        ret += qas / qds

    # 如果 b 不为真，则执行以下操作
    if not b:
        # 计算 i = p - (qd * derivation(qa, DE) - qa * derivation(qd, DE)) / (qd**2)
        i = p - (qd * derivation(qa, DE) - qa * derivation(qd, DE)).as_expr() / \
            (qd**2).as_expr()
        # 将 i 替换为其在符号 s 下的表达式，并封装成 NonElementaryIntegral 对象
        i = NonElementaryIntegral(cancel(i).subs(s), DE.x)
    
    # 返回元组，包含三个元素
    return (ret, i, b)
def integrate_hypertangent_polynomial(p, DE):
    """
    Integration of hypertangent polynomials.

    Explanation
    ===========

    Given a differential field k such that sqrt(-1) is not in k, a
    hypertangent monomial t over k, and p in k[t], return q in k[t] and
    c in k such that p - Dq - c*D(t**2 + 1)/(t**1 + 1) is in k and p -
    Dq does not have an elementary integral over k(t) if Dc != 0.
    """
    # XXX: Make sure that sqrt(-1) is not in k.
    # 要确保 k 中没有 sqrt(-1)
    q, r = polynomial_reduce(p, DE)
    # 对 p 进行多项式简化，返回简化后的多项式 q 和余项 r
    a = DE.d.exquo(Poly(DE.t**2 + 1, DE.t))
    # 计算 D(t**2 + 1)/(t**1 + 1) 的商并转换为多项式 a
    c = Poly(r.nth(1)/(2*a.as_expr()), DE.t)
    # 计算 c = r 的第二个系数除以 2*a，得到 c 作为 DE.t 的多项式
    return (q, c)


def integrate_nonlinear_no_specials(a, d, DE, z=None):
    """
    Integration of nonlinear monomials with no specials.

    Explanation
    ===========

    Given a nonlinear monomial t over k such that Sirr ({p in k[t] | p is
    special, monic, and irreducible}) is empty, and f in k(t), returns g
    elementary over k(t) and a Boolean b in {True, False} such that f - Dg is
    in k if b == True, or f - Dg does not have an elementary integral over k(t)
    if b == False.

    This function is applicable to all nonlinear extensions, but in the case
    where it returns b == False, it will only have proven that the integral of
    f - Dg is nonelementary if Sirr is empty.

    This function returns a Basic expression.
    """
    # TODO: Integral from k?
    # TODO: split out nonelementary integral
    # XXX: a and d must be canceled, or this might not return correct results
    # 确保 a 和 d 被消除，否则可能不会返回正确结果

    z = z or Dummy("z")
    # 如果 z 为 None，则创建一个虚拟符号 z

    s = list(zip(reversed(DE.T), reversed([f(DE.x) for f in DE.Tfuncs])))
    # 创建一个反向列表 s，将 DE.T 和 DE.Tfuncs 的反向元素对应起来

    g1, h, r = hermite_reduce(a, d, DE)
    # 使用 hermite_reduce 函数处理 a, d, DE，返回 g1, h, r

    g2, b = residue_reduce(h[0], h[1], DE, z=z)
    # 使用 residue_reduce 处理 h[0], h[1], DE，返回 g2 和 b

    if not b:
        # 如果 b 为 False
        return ((g1[0].as_expr()/g1[1].as_expr()).subs(s) +
            residue_reduce_to_basic(g2, DE, z), b)
        # 返回 g1[0]/g1[1] 替换 s 后的表达式加上 residue_reduce_to_basic(g2, DE, z) 和 b

    # Because f has no specials, this should be a polynomial in t, or else
    # there is a bug.
    # 因为 f 没有特殊项，所以这应该是关于 t 的多项式，否则就有 bug。
    p = cancel(h[0].as_expr()/h[1].as_expr() - residue_reduce_derivation(g2,
        DE, z).as_expr() + r[0].as_expr()/r[1].as_expr()).as_poly(DE.t)
    # 计算 h[0]/h[1] - residue_reduce_derivation(g2, DE, z) + r[0]/r[1] 的表达式并化简为 DE.t 的多项式
    q1, q2 = polynomial_reduce(p, DE)
    # 对 p 进行多项式简化，返回简化后的多项式 q1 和余项 q2

    if q2.expr.has(DE.t):
        # 如果 q2.expr 中包含 DE.t
        b = False
    else:
        b = True

    ret = (cancel(g1[0].as_expr()/g1[1].as_expr() + q1.as_expr()).subs(s) +
        residue_reduce_to_basic(g2, DE, z))
    # 返回 g1[0]/g1[1] + q1 替换 s 后的表达式加上 residue_reduce_to_basic(g2, DE, z)

    return (ret, b)


class NonElementaryIntegral(Integral):
    """
    Represents a nonelementary Integral.

    Explanation
    ===========

    If the result of integrate() is an instance of this class, it is
    guaranteed to be nonelementary.  Note that integrate() by default will try
    to find any closed-form solution, even in terms of special functions which
    may themselves not be elementary.  To make integrate() only give
    elementary solutions, or, in the cases where it can prove the integral to
    be nonelementary, instances of this class, use integrate(risch=True).
    In this case, integrate() may raise NotImplementedError if it cannot make
    such a determination.
    """
    pass
    # 表示一个非初等积分的类，继承自 Integral 类
    """
    integrate() uses the deterministic Risch algorithm to integrate elementary
    functions or prove that they have no elementary integral.  In some cases,
    this algorithm can split an integral into an elementary and nonelementary
    part, so that the result of integrate will be the sum of an elementary
    expression and a NonElementaryIntegral.
    
    Examples
    ========
    
    >>> from sympy import integrate, exp, log, Integral
    >>> from sympy.abc import x
    
    >>> a = integrate(exp(-x**2), x, risch=True)
    >>> print(a)
    Integral(exp(-x**2), x)
    >>> type(a)
    <class 'sympy.integrals.risch.NonElementaryIntegral'>
    
    >>> expr = (2*log(x)**2 - log(x) - x**2)/(log(x)**3 - x**2*log(x))
    >>> b = integrate(expr, x, risch=True)
    >>> print(b)
    -log(-x + log(x))/2 + log(x + log(x))/2 + Integral(1/log(x), x)
    >>> type(b.atoms(Integral).pop())
    <class 'sympy.integrals.risch.NonElementaryIntegral'>
    """
    
    # TODO: This is useful in and of itself, because isinstance(result,
    # NonElementaryIntegral) will tell if the integral has been proven to be
    # elementary. But should we do more?  Perhaps a no-op .doit() if
    # elementary=True?  Or maybe some information on why the integral is
    # nonelementary.
    pass
# 定义 Risch 积分算法函数，用于计算积分
def risch_integrate(f, x, extension=None, handle_first='log',
                    separate_integral=False, rewrite_complex=None,
                    conds='piecewise'):
    """
    The Risch Integration Algorithm.

    Explanation
    ===========

    Only transcendental functions are supported.  Currently, only exponentials
    and logarithms are supported, but support for trigonometric functions is
    forthcoming.

    If this function returns an unevaluated Integral in the result, it means
    that it has proven that integral to be nonelementary.  Any errors will
    result in raising NotImplementedError.  The unevaluated Integral will be
    an instance of NonElementaryIntegral, a subclass of Integral.

    handle_first may be either 'exp' or 'log'.  This changes the order in
    which the extension is built, and may result in a different (but
    equivalent) solution (for an example of this, see issue 5109).  It is also
    possible that the integral may be computed with one but not the other,
    because not all cases have been implemented yet.  It defaults to 'log' so
    that the outer extension is exponential when possible, because more of the
    exponential case has been implemented.

    If ``separate_integral`` is ``True``, the result is returned as a tuple (ans, i),
    where the integral is ans + i, ans is elementary, and i is either a
    NonElementaryIntegral or 0.  This useful if you want to try further
    integrating the NonElementaryIntegral part using other algorithms to
    possibly get a solution in terms of special functions.  It is False by
    default.

    Examples
    ========

    >>> from sympy.integrals.risch import risch_integrate
    >>> from sympy import exp, log, pprint
    >>> from sympy.abc import x

    First, we try integrating exp(-x**2). Except for a constant factor of
    2/sqrt(pi), this is the famous error function.

    >>> pprint(risch_integrate(exp(-x**2), x))
      /
     |
     |    2
     |  -x
     | e    dx
     |
    /

    The unevaluated Integral in the result means that risch_integrate() has
    proven that exp(-x**2) does not have an elementary anti-derivative.

    In many cases, risch_integrate() can split out the elementary
    anti-derivative part from the nonelementary anti-derivative part.
    For example,

    >>> pprint(risch_integrate((2*log(x)**2 - log(x) - x**2)/(log(x)**3 -
    ... x**2*log(x)), x))
                                             /
                                            |
      log(-x + log(x))   log(x + log(x))    |   1
    - ---------------- + --------------- +  | ------ dx
             2                  2           | log(x)
                                            |
                                           /

    This means that it has proven that the integral of 1/log(x) is
    nonelementary.  This function is also known as the logarithmic integral,
    and is often denoted as Li(x).
    """
    risch_integrate() currently only accepts purely transcendental functions
    with exponentials and logarithms, though note that this can include
    nested exponentials and logarithms, as well as exponentials with bases
    other than E.

    >>> pprint(risch_integrate(exp(x)*exp(exp(x)), x))
     / x\
     \e /
    e
    >>> pprint(risch_integrate(exp(exp(x)), x))
      /
     |
     |  / x\
     |  \e /
     | e     dx
     |
    /

    >>> pprint(risch_integrate(x*x**x*log(x) + x**x + x*x**x, x))
       x
    x*x
    >>> pprint(risch_integrate(x**x, x))
      /
     |
     |  x
     | x  dx
     |
    /

    >>> pprint(risch_integrate(-1/(x*log(x)*log(log(x))**2), x))
         1
    -----------
    log(log(x))

    """
    f = S(f)  # 将输入的表达式 f 转换为符号表达式 S(f)

    # 使用 DifferentialExtension 处理 f 的微分扩展
    DE = extension or DifferentialExtension(f, x, handle_first=handle_first,
            dummy=True, rewrite_complex=rewrite_complex)
    fa, fd = DE.fa, DE.fd  # 获取微分扩展后的 fa 和 fd

    result = S.Zero  # 初始化结果为零
    for case in reversed(DE.cases):  # 反向遍历微分扩展的情况
        # 如果 fa 和 fd 均不包含 DE.t，并且 case 不是 'base'，则减小级别并继续下一个循环
        if not fa.expr.has(DE.t) and not fd.expr.has(DE.t) and not case == 'base':
            DE.decrement_level()
            fa, fd = frac_in((fa, fd), DE.t)
            continue

        fa, fd = fa.cancel(fd, include=True)  # 对 fa 和 fd 进行约分
        if case == 'exp':  # 如果 case 是 'exp'，则调用 integrate_hyperexponential 处理
            ans, i, b = integrate_hyperexponential(fa, fd, DE, conds=conds)
        elif case == 'primitive':  # 如果 case 是 'primitive'，则调用 integrate_primitive 处理
            ans, i, b = integrate_primitive(fa, fd, DE)
        elif case == 'base':  # 如果 case 是 'base'
            # XXX: 这里不能直接调用 ratint()，因为它不能正确处理多项式。
            # 直接对 fa/fd 求积分，risch=False 表示不使用 Risch 算法
            ans = integrate(fa.as_expr()/fd.as_expr(), DE.x, risch=False)
            b = False
            i = S.Zero
        else:
            raise NotImplementedError("Only exponential and logarithmic "
            "extensions are currently supported.")  # 抛出未实现错误信息

        result += ans  # 将 ans 加到结果中
        if b:  # 如果 b 为真
            DE.decrement_level()  # 减小级别
            fa, fd = frac_in(i, DE.t)  # 对 i 进行 frac_in 处理
        else:
            result = result.subs(DE.backsubs)  # 使用 backsubs 替换结果中的表达式
            if not i.is_zero:
                i = NonElementaryIntegral(i.function.subs(DE.backsubs), i.limits)
            if not separate_integral:
                result += i
                return result  # 如果不需要分开积分，则返回最终结果
            else:
                if isinstance(i, NonElementaryIntegral):
                    return (result, i)  # 如果 i 是 NonElementaryIntegral 类型，则返回结果和 i
                else:
                    return (result, 0)  # 否则返回结果和 0
```