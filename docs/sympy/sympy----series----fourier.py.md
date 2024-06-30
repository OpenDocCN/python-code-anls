# `D:\src\scipysrc\sympy\sympy\series\fourier.py`

```
# 导入必要的符号和函数库，用于傅里叶级数的计算
from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import Wild
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import sin, cos, sinc
from sympy.series.series_class import SeriesBase
from sympy.series.sequences import SeqFormula
from sympy.sets.sets import Interval
from sympy.utilities.iterables import is_sequence

# 定义文档测试依赖，需要使用matplotlib
__doctest_requires__ = {('fourier_series',): ['matplotlib']}

# 定义傅里叶余弦级数的函数，返回余弦项序列
def fourier_cos_seq(func, limits, n):
    """Returns the cos sequence in a Fourier series"""
    from sympy.integrals import integrate
    x, L = limits[0], limits[2] - limits[1]
    cos_term = cos(2*n*pi*x / L)
    formula = 2 * cos_term * integrate(func * cos_term, limits) / L
    a0 = formula.subs(n, S.Zero) / 2
    return a0, SeqFormula(2 * cos_term * integrate(func * cos_term, limits)
                          / L, (n, 1, oo))

# 定义傅里叶正弦级数的函数，返回正弦项序列
def fourier_sin_seq(func, limits, n):
    """Returns the sin sequence in a Fourier series"""
    from sympy.integrals import integrate
    x, L = limits[0], limits[2] - limits[1]
    sin_term = sin(2*n*pi*x / L)
    return SeqFormula(2 * sin_term * integrate(func * sin_term, limits)
                      / L, (n, 1, oo))

# 处理限制条件的内部函数
def _process_limits(func, limits):
    """
    Limits should be of the form (x, start, stop).
    x should be a symbol. Both start and stop should be bounded.

    Explanation
    ===========

    * If x is not given, x is determined from func.
    * If limits is None. Limit of the form (x, -pi, pi) is returned.

    Examples
    ========

    >>> from sympy.series.fourier import _process_limits as pari
    >>> from sympy.abc import x
    >>> pari(x**2, (x, -2, 2))
    (x, -2, 2)
    >>> pari(x**2, (-2, 2))
    (x, -2, 2)
    >>> pari(x**2, None)
    (x, -pi, pi)
    """
    # 内部函数：从函数中找到自由符号x
    def _find_x(func):
        free = func.free_symbols
        if len(free) == 1:
            return free.pop()
        elif not free:
            return Dummy('k')
        else:
            raise ValueError(
                " specify dummy variables for %s. If the function contains"
                " more than one free symbol, a dummy variable should be"
                " supplied explicitly e.g. FourierSeries(m*n**2, (n, -pi, pi))"
                % func)

    x, start, stop = None, None, None
    # 处理限制条件为空的情况
    if limits is None:
        x, start, stop = _find_x(func), -pi, pi
    # 处理限制条件为序列的情况
    if is_sequence(limits, Tuple):
        if len(limits) == 3:
            x, start, stop = limits
        elif len(limits) == 2:
            x = _find_x(func)
            start, stop = limits

    # 检查符号x是否为Symbol类型，以及start和stop是否已定义
    if not isinstance(x, Symbol) or start is None or stop is None:
        raise ValueError('Invalid limits given: %s' % str(limits))

    # 定义无界限的情况
    unbounded = [S.NegativeInfinity, S.Infinity]
    # 检查起始值和结束值是否在未限定的集合中，如果是则抛出值错误异常
    if start in unbounded or stop in unbounded:
        raise ValueError("Both the start and end value should be bounded")

    # 使用 sympify 函数将给定的元组 (x, start, stop) 转换为符号表达式
    return sympify((x, start, stop))
def finite_check(f, x, L):
    # 定义内部函数，用于检查表达式是否不包含自由变量 x
    def check_fx(exprs, x):
        return x not in exprs.free_symbols

    # 定义内部函数，用于检查是否是 sin 或 cos 函数，并且其参数匹配特定模式
    def check_sincos(_expr, x, L):
        if isinstance(_expr, (sin, cos)):
            sincos_args = _expr.args[0]

            # 检查参数是否匹配模式 a*(pi/L)*x + b
            if sincos_args.match(a*(pi/L)*x + b) is not None:
                return True
            else:
                return False

    # 导入必要的函数和类
    from sympy.simplify.fu import TR2, TR1, sincos_to_sum
    # 将表达式 f 化简成 sin 和 cos 的和的形式
    _expr = sincos_to_sum(TR2(TR1(f)))
    # 将表达式分解成常数项和非常数项的和
    add_coeff = _expr.as_coeff_add()

    # 定义通配符 a 和 b，用于匹配特定模式
    a = Wild('a', properties=[lambda k: k.is_Integer, lambda k: k != S.Zero, ])
    b = Wild('b', properties=[lambda k: x not in k.free_symbols, ])

    # 遍历非常数项的每一部分
    for s in add_coeff[1]:
        # 将每一部分分解成乘法项，并检查每一个乘法项是否满足条件
        mul_coeffs = s.as_coeff_mul()[1]
        for t in mul_coeffs:
            # 如果乘法项中存在自由变量 x 或者不符合 sincos 模式，则返回 False
            if not (check_fx(t, x) or check_sincos(t, x, L)):
                return False, f

    # 如果所有乘法项都符合条件，则返回 True 和简化后的表达式 _expr
    return True, _expr


class FourierSeries(SeriesBase):
    r"""Represents Fourier sine/cosine series.

    Explanation
    ===========

    This class only represents a fourier series.
    No computation is performed.

    For how to compute Fourier series, see the :func:`fourier_series`
    docstring.

    See Also
    ========

    sympy.series.fourier.fourier_series
    """
    # 定义类的构造函数
    def __new__(cls, *args):
        # 将参数 args 映射为 sympy 表达式
        args = map(sympify, args)
        # 调用父类 Expr 的构造函数
        return Expr.__new__(cls, *args)

    # 定义函数属性，返回该 FourierSeries 对象的第一个参数（即表达式）
    @property
    def function(self):
        return self.args[0]

    # 定义 x 属性，返回 FourierSeries 对象的第二个参数的第一个元素（即自变量 x）
    @property
    def x(self):
        return self.args[1][0]

    # 定义 period 属性，返回 FourierSeries 对象的第二个参数的后两个元素（即周期的上下限）
    @property
    def period(self):
        return (self.args[1][1], self.args[1][2])

    # 定义 a0 属性，返回 FourierSeries 对象的第三个参数的第一个元素（即常数项 a0）
    @property
    def a0(self):
        return self.args[2][0]

    # 定义 an 属性，返回 FourierSeries 对象的第三个参数的第二个元素（即 cos 项系数 an）
    @property
    def an(self):
        return self.args[2][1]

    # 定义 bn 属性，返回 FourierSeries 对象的第三个参数的第三个元素（即 sin 项系数 bn）
    @property
    def bn(self):
        return self.args[2][2]

    # 定义 interval 属性，返回表示无穷大的区间对象
    @property
    def interval(self):
        return Interval(0, oo)

    # 定义 start 属性，返回 interval 属性的下界（即 0）
    @property
    def start(self):
        return self.interval.inf

    # 定义 stop 属性，返回 interval 属性的上界（即无穷大）
    @property
    def stop(self):
        return self.interval.sup

    # 定义 length 属性，返回无穷大
    @property
    def length(self):
        return oo

    # 定义 L 属性，返回周期长度的一半
    @property
    def L(self):
        return abs(self.period[1] - self.period[0]) / 2

    # 定义 _eval_subs 方法，用于符号替换
    def _eval_subs(self, old, new):
        x = self.x
        if old.has(x):
            return self
    `
        def truncate(self, n=3):
            """
            Return the first n nonzero terms of the series.
    
            If ``n`` is None return an iterator.
    
            Parameters
            ==========
    
            n : int or None
                Amount of non-zero terms in approximation or None.
    
            Returns
            =======
    
            Expr or iterator :
                Approximation of function expanded into Fourier series.
    
            Examples
            ========
    
            >>> from sympy import fourier_series, pi
            >>> from sympy.abc import x
            >>> s = fourier_series(x, (x, -pi, pi))
            >>> s.truncate(4)
            2*sin(x) - sin(2*x) + 2*sin(3*x)/3 - sin(4*x)/2
    
            See Also
            ========
    
            sympy.series.fourier.FourierSeries.sigma_approximation
            """
            # If n is None, return an iterator over the series terms
            if n is None:
                return iter(self)
    
            # Initialize an empty list to store the non-zero terms
            terms = []
            
            # Iterate over each term in the series
            for t in self:
                # Stop appending terms once we have collected n non-zero terms
                if len(terms) == n:
                    break
                # Only append the term if it is non-zero
                if t is not S.Zero:
                    terms.append(t)
    
            # Return the sum of the collected non-zero terms
            return Add(*terms)
    def sigma_approximation(self, n=3):
        r"""
        Return :math:`\sigma`-approximation of Fourier series with respect
        to order n.

        Explanation
        ===========

        Sigma approximation adjusts a Fourier summation to eliminate the Gibbs
        phenomenon which would otherwise occur at discontinuities.
        A sigma-approximated summation for a Fourier series of a T-periodical
        function can be written as

        .. math::
            s(\theta) = \frac{1}{2} a_0 + \sum _{k=1}^{m-1}
            \operatorname{sinc} \Bigl( \frac{k}{m} \Bigr) \cdot
            \left[ a_k \cos \Bigl( \frac{2\pi k}{T} \theta \Bigr)
            + b_k \sin \Bigl( \frac{2\pi k}{T} \theta \Bigr) \right],

        where :math:`a_0, a_k, b_k, k=1,\ldots,{m-1}` are standard Fourier
        series coefficients and
        :math:`\operatorname{sinc} \Bigl( \frac{k}{m} \Bigr)` is a Lanczos
        :math:`\sigma` factor (expressed in terms of normalized
        :math:`\operatorname{sinc}` function).

        Parameters
        ==========

        n : int
            Highest order of the terms taken into account in approximation.

        Returns
        =======

        Expr :
            Sigma approximation of function expanded into Fourier series.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x, (x, -pi, pi))
        >>> s.sigma_approximation(4)
        2*sin(x)*sinc(pi/4) - 2*sin(2*x)/pi + 2*sin(3*x)*sinc(3*pi/4)/3

        See Also
        ========

        sympy.series.fourier.FourierSeries.truncate

        Notes
        =====

        The behaviour of
        :meth:`~sympy.series.fourier.FourierSeries.sigma_approximation`
        is different from :meth:`~sympy.series.fourier.FourierSeries.truncate`
        - it takes all nonzero terms of degree smaller than n, rather than
        first n nonzero ones.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Gibbs_phenomenon
        .. [2] https://en.wikipedia.org/wiki/Sigma_approximation
        """
        # 生成包含最高 n 阶非零项的 sigma-近似表达式列表
        terms = [sinc(pi * i / n) * t for i, t in enumerate(self[:n])
                 if t is not S.Zero]
        # 返回所有项的和，作为函数展开的 sigma-近似
        return Add(*terms)
    def shift(self, s):
        """
        Shift the function by a term independent of x.

        Explanation
        ===========

        f(x) -> f(x) + s

        This method shifts the Fourier series of the function by adding a constant term 's' to the series.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.shift(1).truncate()
        -4*cos(x) + cos(2*x) + 1 + pi**2/3
        """
        s, x = sympify(s), self.x

        # Check if the shift term 's' depends on the variable 'x'
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        # Calculate the new coefficients and function after shifting
        a0 = self.a0 + s
        sfunc = self.function + s

        # Return a new instance of the function class with the shifted parameters
        return self.func(sfunc, self.args[1], (a0, self.an, self.bn))

    def shiftx(self, s):
        """
        Shift x by a term independent of x.

        Explanation
        ===========

        f(x) -> f(x + s)

        This method shifts the variable 'x' in the Fourier series of the function by adding a constant term 's'.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.shiftx(1).truncate()
        -4*cos(x + 1) + cos(2*x + 2) + pi**2/3
        """
        s, x = sympify(s), self.x

        # Check if the shift term 's' depends on the variable 'x'
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        # Calculate the shifted coefficients and function
        an = self.an.subs(x, x + s)
        bn = self.bn.subs(x, x + s)
        sfunc = self.function.subs(x, x + s)

        # Return a new instance of the function class with the shifted parameters
        return self.func(sfunc, self.args[1], (self.a0, an, bn))

    def scale(self, s):
        """
        Scale the function by a term independent of x.

        Explanation
        ===========

        f(x) -> s * f(x)

        This method scales the Fourier series of the function by multiplying all coefficients and the function itself by a scalar 's'.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.scale(2).truncate()
        -8*cos(x) + 2*cos(2*x) + 2*pi**2/3
        """
        s, x = sympify(s), self.x

        # Check if the scale factor 's' depends on the variable 'x'
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        # Scale the coefficients and the function
        an = self.an.coeff_mul(s)
        bn = self.bn.coeff_mul(s)
        a0 = self.a0 * s
        sfunc = self.args[0] * s

        # Return a new instance of the function class with the scaled parameters
        return self.func(sfunc, self.args[1], (a0, an, bn))
    # 定义一个方法，用于将自身的 x 缩放为 s*x
    def scalex(self, s):
        """
        Scale x by a term independent of x.

        Explanation
        ===========

        f(x) -> f(s*x)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.scalex(2).truncate()
        -4*cos(2*x) + cos(4*x) + pi**2/3
        """
        s, x = sympify(s), self.x

        # 如果 s 中包含自由符号 x，则抛出异常
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        # 将 Fourier 级数中的参数 x 替换为 s*x，获取新的系数 an 和 bn，以及替换后的函数
        an = self.an.subs(x, x * s)
        bn = self.bn.subs(x, x * s)
        sfunc = self.function.subs(x, x * s)

        # 返回一个新的 Fourier 级数对象，表示缩放后的函数
        return self.func(sfunc, self.args[1], (self.a0, an, bn))

    # 对 Fourier 级数中的每一项进行遍历，返回第一个非零项作为主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        for t in self:
            if t is not S.Zero:
                return t

    # 对给定的点 pt 进行处理，如果是 0，则返回常数项 a0，否则返回对应的 an 和 bn 系数之和
    def _eval_term(self, pt):
        if pt == 0:
            return self.a0
        return self.an.coeff(pt) + self.bn.coeff(pt)

    # 实现负号运算，返回当前 Fourier 级数的相反数
    def __neg__(self):
        return self.scale(-1)

    # 实现加法运算，支持 Fourier 级数与另一个 Fourier 级数的加法
    def __add__(self, other):
        if isinstance(other, FourierSeries):
            # 检查两个级数的周期是否相同
            if self.period != other.period:
                raise ValueError("Both the series should have same periods")

            x, y = self.x, other.x
            # 将两个级数的函数部分相加，并将其中一个级数中的自变量替换为另一个级数的自变量
            function = self.function + other.function.subs(y, x)

            # 如果结果中不再包含自变量 x，则返回结果
            if self.x not in function.free_symbols:
                return function

            # 分别计算新的常数项 a0，以及新的 an 和 bn 系数
            an = self.an + other.an
            bn = self.bn + other.bn
            a0 = self.a0 + other.a0

            # 返回一个新的 Fourier 级数对象，表示两个级数的和
            return self.func(function, self.args[1], (a0, an, bn))

        # 如果不是 Fourier 级数对象，则返回一个加法表达式对象
        return Add(self, other)

    # 实现减法运算，通过将减数取负然后调用加法运算来实现
    def __sub__(self, other):
        return self.__add__(-other)
class FiniteFourierSeries(FourierSeries):
    r"""Represents Finite Fourier sine/cosine series.

    For how to compute Fourier series, see the :func:`fourier_series`
    docstring.

    Parameters
    ==========

    f : Expr
        Expression for finding fourier_series

    limits : ( x, start, stop)
        x is the independent variable for the expression f
        (start, stop) is the period of the fourier series

    exprs: (a0, an, bn) or Expr
        a0 is the constant term a0 of the fourier series
        an is a dictionary of coefficients of cos terms
         an[k] = coefficient of cos(pi*(k/L)*x)
        bn is a dictionary of coefficients of sin terms
         bn[k] = coefficient of sin(pi*(k/L)*x)

        or exprs can be an expression to be converted to fourier form

    Methods
    =======

    This class is an extension of FourierSeries class.
    Please refer to sympy.series.fourier.FourierSeries for
    further information.

    See Also
    ========

    sympy.series.fourier.FourierSeries
    sympy.series.fourier.fourier_series
    """

    def __new__(cls, f, limits, exprs):
        # Ensure f, limits, and exprs are converted to sympy expressions if they aren't already
        f = sympify(f)
        limits = sympify(limits)
        exprs = sympify(exprs)

        # Check if exprs is of the form (a0, an, bn)
        if not (isinstance(exprs, Tuple) and len(exprs) == 3):  # exprs is not of form (a0, an, bn)
            # Converts the expression to Fourier form
            c, e = exprs.as_coeff_add()
            from sympy.simplify.fu import TR10
            rexpr = c + Add(*[TR10(i) for i in e])
            # Expand and separate the constant term and the terms with trigonometric functions
            a0, exp_ls = rexpr.expand(trig=False, power_base=False, power_exp=False, log=False).as_coeff_add()

            x = limits[0]  # independent variable x
            L = abs(limits[2] - limits[1]) / 2  # half the period of the Fourier series

            a = Wild('a', properties=[lambda k: k.is_Integer, lambda k: k is not S.Zero, ])
            b = Wild('b', properties=[lambda k: x not in k.free_symbols, ])

            an = {}
            bn = {}

            # Separate coefficients of sin and cos terms into dictionaries an and bn
            for p in exp_ls:
                t = p.match(b * cos(a * (pi / L) * x))
                q = p.match(b * sin(a * (pi / L) * x))
                if t:
                    an[t[a]] = t[b] + an.get(t[a], S.Zero)
                elif q:
                    bn[q[a]] = q[b] + bn.get(q[a], S.Zero)
                else:
                    a0 += p  # accumulate terms that are not sin or cos

            exprs = Tuple(a0, an, bn)

        # Call the __new__ method of the superclass (Expr) to create the instance
        return Expr.__new__(cls, f, limits, exprs)

    @property
    def interval(self):
        # Calculate the interval [0, _length] for the Fourier series
        _length = 1 if self.a0 else 0  # set _length to 1 if there is a constant term a0
        _length += max(set(self.an.keys()).union(set(self.bn.keys()))) + 1  # add the highest harmonic term
        return Interval(0, _length)

    @property
    def length(self):
        # Calculate the length of the interval of the Fourier series
        return self.stop - self.start

    def shiftx(self, s):
        # Shift the Fourier series by s units
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        # Shift the Fourier series and return the shifted series
        _expr = self.truncate().subs(x, x + s)
        sfunc = self.function.subs(x, x + s)

        return self.func(sfunc, self.args[1], _expr)
    # 对 FourierSeries 对象进行缩放操作，将其系数乘以给定的标量 s
    def scale(self, s):
        # 将输入的 s 转换为符号表达式
        s, x = sympify(s), self.x
        
        # 检查符号表达式 s 中是否包含变量 x，如果包含则抛出错误
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))
        
        # 计算缩放后的系数表达式 _expr
        _expr = self.truncate() * s
        # 缩放函数部分 sfunc
        sfunc = self.function * s
        
        # 返回缩放后的 FourierSeries 对象，保持其他参数不变
        return self.func(sfunc, self.args[1], _expr)

    # 对 FourierSeries 对象在 x 方向进行缩放操作
    def scalex(self, s):
        # 将输入的 s 转换为符号表达式
        s, x = sympify(s), self.x
        
        # 检查符号表达式 s 中是否包含变量 x，如果包含则抛出错误
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))
        
        # 计算缩放后的函数表达式 _expr
        _expr = self.truncate().subs(x, x * s)
        # 缩放函数部分 sfunc
        sfunc = self.function.subs(x, x * s)
        
        # 返回缩放后的 FourierSeries 对象，保持其他参数不变
        return self.func(sfunc, self.args[1], _expr)

    # 对 FourierSeries 对象的每一项进行求值
    def _eval_term(self, pt):
        # 如果 pt 为 0，则直接返回常数项 self.a0
        if pt == 0:
            return self.a0
        
        # 计算 FourierSeries 的每一项 _term
        _term = self.an.get(pt, S.Zero) * cos(pt * (pi / self.L) * self.x) \
                + self.bn.get(pt, S.Zero) * sin(pt * (pi / self.L) * self.x)
        
        # 返回计算结果
        return _term

    # 重载加法操作，实现 FourierSeries 对象与其他对象的加法
    def __add__(self, other):
        # 如果 other 是 FourierSeries 类型的对象，则返回其与当前对象的和
        if isinstance(other, FourierSeries):
            return other.__add__(fourier_series(self.function, self.args[1], finite=False))
        # 如果 other 是 FiniteFourierSeries 类型的对象
        elif isinstance(other, FiniteFourierSeries):
            # 检查两个 FourierSeries 对象是否具有相同的周期
            if self.period != other.period:
                raise ValueError("Both the series should have same periods")
            
            # 获取两个对象的自变量
            x, y = self.x, other.x
            # 计算两个函数的和 function
            function = self.function + other.function.subs(y, x)
            
            # 如果和 function 不再依赖于自变量 x，则直接返回结果
            if self.x not in function.free_symbols:
                return function
            
            # 否则，返回重新计算 FourierSeries 的结果
            return fourier_series(function, limits=self.args[1])
def fourier_series(f, limits=None, finite=True):
    r"""Computes the Fourier trigonometric series expansion.

    Explanation
    ===========

    Fourier trigonometric series of $f(x)$ over the interval $(a, b)$
    is defined as:

    .. math::
        \frac{a_0}{2} + \sum_{n=1}^{\infty}
        (a_n \cos(\frac{2n \pi x}{L}) + b_n \sin(\frac{2n \pi x}{L}))

    where the coefficients are:

    .. math::
        L = b - a

    .. math::
        a_0 = \frac{2}{L} \int_{a}^{b}{f(x) dx}

    .. math::
        a_n = \frac{2}{L} \int_{a}^{b}{f(x) \cos(\frac{2n \pi x}{L}) dx}

    .. math::
        b_n = \frac{2}{L} \int_{a}^{b}{f(x) \sin(\frac{2n \pi x}{L}) dx}

    The condition whether the function $f(x)$ given should be periodic
    or not is more than necessary, because it is sufficient to consider
    the series to be converging to $f(x)$ only in the given interval,
    not throughout the whole real line.

    This also brings a lot of ease for the computation because
    you do not have to make $f(x)$ artificially periodic by
    wrapping it with piecewise, modulo operations,
    but you can shape the function to look like the desired periodic
    function only in the interval $(a, b)$, and the computed series will
    automatically become the series of the periodic version of $f(x)$.

    This property is illustrated in the examples section below.

    Parameters
    ==========

    limits : (sym, start, end), optional
        *sym* denotes the symbol the series is computed with respect to.

        *start* and *end* denotes the start and the end of the interval
        where the fourier series converges to the given function.

        Default range is specified as $-\pi$ and $\pi$.

    Returns
    =======

    FourierSeries
        A symbolic object representing the Fourier trigonometric series.

    Examples
    ========

    Computing the Fourier series of $f(x) = x^2$:

    >>> from sympy import fourier_series, pi
    >>> from sympy.abc import x
    >>> f = x**2
    >>> s = fourier_series(f, (x, -pi, pi))
    >>> s1 = s.truncate(n=3)
    >>> s1
    -4*cos(x) + cos(2*x) + pi**2/3

    Shifting of the Fourier series:

    >>> s.shift(1).truncate()
    -4*cos(x) + cos(2*x) + 1 + pi**2/3
    >>> s.shiftx(1).truncate()
    -4*cos(x + 1) + cos(2*x + 2) + pi**2/3

    Scaling of the Fourier series:

    >>> s.scale(2).truncate()
    -8*cos(x) + 2*cos(2*x) + 2*pi**2/3
    >>> s.scalex(2).truncate()
    -4*cos(2*x) + cos(4*x) + pi**2/3

    Computing the Fourier series of $f(x) = x$:

    This illustrates how truncating to the higher order gives better
    convergence.
    """

    # 实现计算 Fourier 级数的函数

    # 如果未提供 limits 参数，则默认取 $-\pi$ 到 $\pi$ 的范围
    if limits is None:
        limits = (sym, -pi, pi)

    # 返回 FourierSeries 对象，表示计算得到的 Fourier 三角级数
    return FourierSeries(f, limits, finite)
    # 将输入的函数符号化
    f = sympify(f)

    # 处理限制条件，确保它们适用于函数 f
    limits = _process_limits(f, limits)
    x = limits[0]

    # 如果变量 x 不在 f 的自由符号中，则返回原始函数 f
    if x not in f.free_symbols:
        return f

    # 如果计划进行有限傅里叶级数计算
    if finite:
        # 计算限制范围的长度 L
        L = abs(limits[2] - limits[1]) / 2
        # 检查函数是否在指定范围内是有限的，并获取结果
        is_finite, res_f = finite_check(f, x, L)
        # 如果函数在指定范围内是有限的，则返回有限傅里叶级数对象
        if is_finite:
            return FiniteFourierSeries(f, limits, res_f)

    # 创建一个虚拟符号 n 用于傅里叶级数展开
    n = Dummy('n')
    # 计算函数定义域的中心点
    center = (limits[1] + limits[2]) / 2

    # 如果中心点是零
    if center.is_zero:
        # 判断函数是否为其自身的负值
        neg_f = f.subs(x, -x)
        if f == neg_f:
            # 计算函数的余弦系数
            a0, an = fourier_cos_seq(f, limits, n)
            # 声明傅里叶级数的正弦系数为零
            bn = SeqFormula(0, (1, oo))
            # 返回对应的傅里叶级数对象
            return FourierSeries(f, limits, (a0, an, bn))
        elif f == -neg_f:
            # 声明傅里叶级数的常数项 a0 为零
            a0 = S.Zero
            # 计算函数的正弦系数
            an = SeqFormula(0, (1, oo))
            bn = fourier_sin_seq(f, limits, n)
            # 返回对应的傅里叶级数对象
            return FourierSeries(f, limits, (a0, an, bn))
    # 如果函数不满足中心点为零的条件，则计算其余弦系数
    a0, an = fourier_cos_seq(f, limits, n)
    # 调用函数 `fourier_sin_seq`，传入参数 `f`、`limits`、`n`，得到傅里叶正弦级数的系数 bn
    bn = fourier_sin_seq(f, limits, n)
    # 返回一个 `FourierSeries` 对象，该对象表示给定函数 `f` 的傅里叶级数，包括其限制 `limits`，
    # 以及包括的系数元组 `(a0, an, bn)`
    return FourierSeries(f, limits, (a0, an, bn))
```