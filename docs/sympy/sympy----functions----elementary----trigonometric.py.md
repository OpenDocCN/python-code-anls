# `D:\src\scipysrc\sympy\sympy\functions\elementary\trigonometric.py`

```
# 引入类型提示，重命名导入类型别名
from typing import Tuple as tTuple, Union as tUnion
# 导入加法操作
from sympy.core.add import Add
# 导入缓存装饰器
from sympy.core.cache import cacheit
# 导入表达式基类
from sympy.core.expr import Expr
# 导入函数相关模块，包括函数基类和一些异常
from sympy.core.function import Function, ArgumentIndexError, PoleError, expand_mul
# 导入逻辑运算模块，包括模糊逻辑运算和模糊布尔类型
from sympy.core.logic import fuzzy_not, fuzzy_or, FuzzyBool, fuzzy_and
# 导入取模运算模块
from sympy.core.mod import Mod
# 导入数值模块，包括有理数、圆周率、整数、浮点数等
from sympy.core.numbers import Rational, pi, Integer, Float, equal_valued
# 导入关系运算模块，包括不等号和等号
from sympy.core.relational import Ne, Eq
# 导入单例对象模块
from sympy.core.singleton import S
# 导入符号模块，包括符号类和虚拟符号
from sympy.core.symbol import Symbol, Dummy
# 导入符号化模块，将输入转换为符号表达式
from sympy.core.sympify import sympify
# 导入组合数学函数，如阶乘和升阶乘
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
# 导入组合数学函数，如贝努利数和欧拉数
from sympy.functions.combinatorial.numbers import bernoulli, euler
# 导入复数函数，如辐角、实部和虚部
from sympy.functions.elementary.complexes import arg as arg_f, im, re
# 导入指数函数，如自然对数和指数函数
from sympy.functions.elementary.exponential import log, exp
# 导入整数函数，如向下取整
from sympy.functions.elementary.integers import floor
# 导入杂项函数，如平方根、最小值和最大值
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
# 导入分段函数
from sympy.functions.elementary.piecewise import Piecewise
# 导入三角特殊函数模块，如余弦表、整数部分分数和费马坐标
from sympy.functions.elementary._trigonometric_special import (
    cos_table, ipartfrac, fermat_coords)
# 导入布尔代数模块中的与运算
from sympy.logic.boolalg import And
# 导入数论模块中的因数分解
from sympy.ntheory import factorint
# 导入特殊多项式模块，如对称多项式
from sympy.polys.specialpolys import symmetric_poly
# 导入可枚举工具模块，用于生成编号的符号
from sympy.utilities.iterables import numbered_symbols


###############################################################################
########################## UTILITIES ##########################################
###############################################################################


def _imaginary_unit_as_coefficient(arg):
    """ Helper to extract symbolic coefficient for imaginary unit """
    # 如果参数是浮点数，返回 None
    if isinstance(arg, Float):
        return None
    else:
        # 否则尝试提取参数中虚数单位的系数
        return arg.as_coefficient(S.ImaginaryUnit)

###############################################################################
########################## TRIGONOMETRIC FUNCTIONS ############################
###############################################################################


class TrigonometricFunction(Function):
    """Base class for trigonometric functions. """

    # 定义非分叉特性和奇点（仅有无穷远点）
    unbranched = True
    _singularities = (S.ComplexInfinity,)

    def _eval_is_rational(self):
        # 计算函数是否为有理数
        s = self.func(*self.args)
        if s.func == self.func:
            # 如果函数的自身调用，且第一个参数为有理数且非零，返回 False
            if s.args[0].is_rational and fuzzy_not(s.args[0].is_zero):
                return False
        else:
            # 否则返回函数是否为有理数
            return s.is_rational

    def _eval_is_algebraic(self):
        # 计算函数是否为代数数
        s = self.func(*self.args)
        if s.func == self.func:
            # 如果函数的自身调用，且第一个参数非零且为代数数，返回 False
            if fuzzy_not(self.args[0].is_zero) and self.args[0].is_algebraic:
                return False
            # 检查第一个参数中是否有圆周率系数，并且该系数为有理数
            pi_coeff = _pi_coeff(self.args[0])
            if pi_coeff is not None and pi_coeff.is_rational:
                return True
        else:
            # 否则返回函数是否为代数数
            return s.is_algebraic

    def _eval_expand_complex(self, deep=True, **hints):
        # 展开函数的复数部分
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*S.ImaginaryUnit
    # 将复数表达式转换为实部和虚部的形式
    def _as_real_imag(self, deep=True, **hints):
        # 检查第一个参数是否为扩展实数
        if self.args[0].is_extended_real:
            # 如果需要深度展开
            if deep:
                # 设置复数标志为False，展开第一个参数，并将虚部设为零
                hints['complex'] = False
                return (self.args[0].expand(deep, **hints), S.Zero)
            else:
                # 如果不需要深度展开，则直接返回第一个参数和零作为虚部
                return (self.args[0], S.Zero)
        
        # 如果第一个参数不是扩展实数，则获取其实部和虚部
        if deep:
            # 深度展开参数并获取其实部和虚部
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            # 直接获取参数的实部和虚部
            re, im = self.args[0].as_real_imag()
        
        # 返回实部和虚部的元组
        return (re, im)

    # 计算表达式的周期
    def _period(self, general_period, symbol=None):
        # 将表达式扩展成乘法表达式
        f = expand_mul(self.args[0])
        
        # 如果未指定符号，则选择表达式中的第一个自由符号作为符号
        if symbol is None:
            symbol = tuple(f.free_symbols)[0]

        # 如果表达式不包含给定符号，则返回零
        if not f.has(symbol):
            return S.Zero

        # 如果表达式与符号相等，则返回给定的通用周期
        if f == symbol:
            return general_period

        # 如果表达式中包含给定符号，则进一步处理
        if symbol in f.free_symbols:
            # 如果表达式是乘法类型，则分离出独立的因子并检查
            if f.is_Mul:
                g, h = f.as_independent(symbol)
                if h == symbol:
                    return general_period / abs(g)

            # 如果表达式是加法类型，则分离出独立的项并检查
            if f.is_Add:
                a, h = f.as_independent(symbol)
                g, h = h.as_independent(symbol, as_Add=False)
                if h == symbol:
                    return general_period / abs(g)

        # 如果无法处理给定的表达式类型，则引发未实现错误
        raise NotImplementedError("Use the periodicity function instead.")
# 使用装饰器 @cacheit 使函数具有缓存功能，即结果可以被缓存以提高性能
@cacheit
# 定义一个名为 _table2 的函数，返回一个字典，表示特定整数到元组的映射关系
def _table2():
    # 返回包含特定整数到元组的映射关系的字典
    return {
        12: (3, 4),   # 整数 12 映射到元组 (3, 4)
        20: (4, 5),   # 整数 20 映射到元组 (4, 5)
        30: (5, 6),   # 整数 30 映射到元组 (5, 6)
        15: (6, 10),  # 整数 15 映射到元组 (6, 10)
        24: (6, 8),   # 整数 24 映射到元组 (6, 8)
        40: (8, 10),  # 整数 40 映射到元组 (8, 10)
        60: (20, 30), # 整数 60 映射到元组 (20, 30)
        120: (40, 60) # 整数 120 映射到元组 (40, 60)
    }


def _peeloff_pi(arg):
    r"""
    将 ARG 拆分为两部分，一个“剩余部分”和一个$\pi$的倍数。
    假设 ARG 是一个 Add 类型的对象。
    第二个位置返回的$\pi$的倍数始终是一个有理数。

    Examples
    ========

    >>> from sympy.functions.elementary.trigonometric import _peeloff_pi
    >>> from sympy import pi
    >>> from sympy.abc import x, y
    >>> _peeloff_pi(x + pi/2)
    (x, 1/2)
    >>> _peeloff_pi(x + 2*pi/3 + pi*y)
    (x + pi*y + pi/6, 1/2)

    """
    pi_coeff = S.Zero  # 初始化$\pi$的系数为零
    rest_terms = []    # 初始化剩余部分的列表

    # 遍历 ARG 中的每一个项
    for a in Add.make_args(arg):
        K = a.coeff(pi)  # 获取当前项 a 关于$\pi$的系数
        if K and K.is_rational:
            pi_coeff += K  # 如果系数 K 存在且是有理数，则加到 pi_coeff 中
        else:
            rest_terms.append(a)  # 否则将该项添加到剩余部分列表中

    # 如果$\pi$的系数为零，则返回原始 ARG 和 0
    if pi_coeff is S.Zero:
        return arg, S.Zero

    # 计算$\pi$的系数的两部分 m1 和 m2
    m1 = (pi_coeff % S.Half)
    m2 = pi_coeff - m1

    # 如果 m2 是整数或者 m2 是偶数且 m2 不是偶数，则返回处理后的 ARG 和 m2
    if m2.is_integer or ((2*m2).is_integer and m2.is_even is False):
        return Add(*(rest_terms + [m1*pi])), m2
    # 否则返回原始 ARG 和 0
    return arg, S.Zero


def _pi_coeff(arg: Expr, cycles: int = 1) -> tUnion[Expr, None]:
    r"""
    当 ARG 是$\pi$的倍数时（例如$3\pi/2$），返回归一化到区间$[0, 2]$的数值，
    否则返回 `None`。

    当遇到偶数倍的$\pi$时，如果它与已知奇偶性的乘积，则将其返回为 0，否则返回 2。

    Examples
    ========

    >>> from sympy.functions.elementary.trigonometric import _pi_coeff
    >>> from sympy import pi, Dummy
    >>> from sympy.abc import x
    >>> _pi_coeff(3*x*pi)
    3*x
    >>> _pi_coeff(11*pi/7)
    11/7
    >>> _pi_coeff(-11*pi/7)
    3/7
    >>> _pi_coeff(4*pi)
    0
    >>> _pi_coeff(5*pi)
    1
    >>> _pi_coeff(5.0*pi)
    1
    >>> _pi_coeff(5.5*pi)
    3/2
    >>> _pi_coeff(2 + pi)

    >>> _pi_coeff(2*Dummy(integer=True)*pi)
    2
    >>> _pi_coeff(2*Dummy(even=True)*pi)
    0

    """
    if arg is pi:
        return S.One  # 如果 ARG 就是 $\pi$，则返回 1
    elif not arg:
        return S.Zero  # 如果 ARG 是空，则返回 0
    # 检查参数是否为乘法表达式
    elif arg.is_Mul:
        # 获取乘法表达式中的系数乘以 pi 的部分
        cx = arg.coeff(pi)
        # 如果存在系数
        if cx:
            # 将系数分解为整数部分和乘积表达式部分
            c, x = cx.as_coeff_Mul()  # pi is not included as coeff
            # 如果系数 c 是浮点数
            if c.is_Float:
                # 将精确的二进制分数重新转换为有理数
                f = abs(c) % 1
                # 如果存在小数部分
                if f != 0:
                    # 计算小数部分的对数并近似取整
                    p = -int(round(log(f, 2).evalf()))
                    m = 2**p
                    cm = c*m
                    i = int(cm)
                    # 如果近似值与整数部分乘积相等，则转换为有理数
                    if equal_valued(i, cm):
                        c = Rational(i, m)
                        cx = c*x
                else:
                    # 将 c 转换为有理数
                    c = Rational(int(c))
                    cx = c*x
            # 如果变量 x 是整数
            if x.is_integer:
                # 计算系数 c 对 2 取模的结果
                c2 = c % 2
                # 如果余数为 1，则返回变量 x
                if c2 == 1:
                    return x
                # 如果余数为 0
                elif not c2:
                    # 如果 x 已知为偶数
                    if x.is_even is not None:  # known parity
                        return S.Zero
                    # 否则返回整数 2
                    return Integer(2)
                # 如果余数不为 0 或 1，则返回余数乘以 x
                else:
                    return c2*x
            # 返回计算得到的乘积表达式
            return cx
    # 如果参数 arg 是零
    elif arg.is_zero:
        # 返回零
        return S.Zero
    # 如果参数 arg 不符合以上条件，返回 None
    return None
class sin(TrigonometricFunction):
    r"""
    The sine function.

    Returns the sine of x (measured in radians).

    Explanation
    ===========

    This function will evaluate automatically in the
    case $x/\pi$ is some rational number [4]_.  For example,
    if $x$ is a multiple of $\pi$, $\pi/2$, $\pi/3$, $\pi/4$, and $\pi/6$.

    Examples
    ========

    >>> from sympy import sin, pi
    >>> from sympy.abc import x
    >>> sin(x**2).diff(x)
    2*x*cos(x**2)
    >>> sin(1).diff(x)
    0
    >>> sin(pi)
    0
    >>> sin(pi/2)
    1
    >>> sin(pi/6)
    1/2
    >>> sin(pi/12)
    -sqrt(2)/4 + sqrt(6)/4


    See Also
    ========

    csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Sin
    .. [4] https://mathworld.wolfram.com/TrigonometryAngles.html

    """

    # 定义周期函数，使用父类方法计算周期
    def period(self, symbol=None):
        return self._period(2*pi, symbol)

    # 求偏导数的方法，对于正弦函数是余弦函数
    def fdiff(self, argindex=1):
        if argindex == 1:
            return cos(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    # 计算泰勒级数的静态方法，根据奇数阶数返回相应的泰勒项
    @classmethod
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            # 根据前两项计算新的泰勒项
            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return -p*x**2/(n*(n - 1))
            else:
                return S.NegativeOne**(n//2)*x**n/factorial(n)

    # 对正弦函数进行 nseries 展开的方法
    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        if logx is not None:
            arg = arg.subs(log(x), logx)
        if arg.subs(x, 0).has(S.NaN, S.ComplexInfinity):
            raise PoleError("Cannot expand %s around 0" % (self))
        return Function._eval_nseries(self, x, n=n, logx=logx, cdir=cdir)

    # 将正弦函数重写为指数函数的形式
    def _eval_rewrite_as_exp(self, arg, **kwargs):
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        I = S.ImaginaryUnit
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp)
        return (exp(arg*I) - exp(-arg*I))/(2*I)

    # 将正弦函数重写为幂函数的形式
    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return I*x**-I/2 - I*x**I /2

    # 将正弦函数重写为余弦函数的形式
    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return cos(arg - pi/2, evaluate=False)

    # 将正弦函数重写为正切函数的形式
    def _eval_rewrite_as_tan(self, arg, **kwargs):
        tan_half = tan(S.Half*arg)
        return 2*tan_half/(1 + tan_half**2)

    # 将正弦函数重写为正弦余弦函数的形式
    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg)*cos(arg)/cos(arg)
    # 重新定义函数 `_eval_rewrite_as_cot`，将表达式重写为 cot 函数的形式
    def _eval_rewrite_as_cot(self, arg, **kwargs):
        # 计算 cot(S.Half*arg)
        cot_half = cot(S.Half*arg)
        # 根据条件返回 Piecewise 对象，根据 arg 的实部为零且模运算为零来判断返回值为 0，否则返回 2*cot_half/(1 + cot_half**2)
        return Piecewise((0, And(Eq(im(arg), 0), Eq(Mod(arg, pi), 0))),
                         (2*cot_half/(1 + cot_half**2), True))

    # 重新定义函数 `_eval_rewrite_as_pow`，将表达式重写为 pow 函数的形式
    def _eval_rewrite_as_pow(self, arg, **kwargs):
        # 调用 cos 的重写方法，并将结果再重写为 pow 函数的形式
        return self.rewrite(cos, **kwargs).rewrite(pow, **kwargs)

    # 重新定义函数 `_eval_rewrite_as_sqrt`，将表达式重写为 sqrt 函数的形式
    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        # 调用 cos 的重写方法，并将结果再重写为 sqrt 函数的形式
        return self.rewrite(cos, **kwargs).rewrite(sqrt, **kwargs)

    # 重新定义函数 `_eval_rewrite_as_csc`，将表达式重写为 csc 函数的形式
    def _eval_rewrite_as_csc(self, arg, **kwargs):
        # 返回 csc(arg) 的倒数
        return 1/csc(arg)

    # 重新定义函数 `_eval_rewrite_as_sec`，将表达式重写为 sec 函数的形式
    def _eval_rewrite_as_sec(self, arg, **kwargs):
        # 返回 sec(arg - pi/2) 的值，evaluate=False 表示不进行计算
        return 1/sec(arg - pi/2, evaluate=False)

    # 重新定义函数 `_eval_rewrite_as_sinc`，将表达式重写为 sinc 函数的形式
    def _eval_rewrite_as_sinc(self, arg, **kwargs):
        # 返回 arg*sinc(arg)
        return arg*sinc(arg)

    # 重新定义函数 `_eval_rewrite_as_besselj`，将表达式重写为 besselj 函数的形式
    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        # 导入 besselj 函数并返回 sqrt(pi*arg/2)*besselj(S.Half, arg) 的结果
        from sympy.functions.special.bessel import besselj
        return sqrt(pi*arg/2)*besselj(S.Half, arg)

    # 重新定义函数 `_eval_conjugate`，返回函数自身的共轭
    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    # 将函数表示为实部和虚部的形式
    def as_real_imag(self, deep=True, **hints):
        # 导入所需的函数
        from sympy.functions.elementary.hyperbolic import cosh, sinh
        # 获取表达式的实部和虚部
        re, im = self._as_real_imag(deep=deep, **hints)
        # 返回以 sin(re)*cosh(im) 和 cos(re)*sinh(im) 表示的实部和虚部
        return (sin(re)*cosh(im), cos(re)*sinh(im))

    # 重新定义函数 `_eval_expand_trig`，将表达式进行三角函数展开
    def _eval_expand_trig(self, **hints):
        # 导入所需的函数
        from sympy.functions.special.polynomials import chebyshevt, chebyshevu
        # 获取函数的参数
        arg = self.args[0]
        x = None
        if arg.is_Add:  # 如果参数是加法表达式
            # TODO: 在此处实现更深层次的处理
            # TODO: 对于超过两项的情况需要更高效的处理方式
            # 将参数分解为两项
            x, y = arg.as_two_terms()
            # 对两项分别进行 sin 和 cos 的展开
            sx = sin(x, evaluate=False)._eval_expand_trig()
            sy = sin(y, evaluate=False)._eval_expand_trig()
            cx = cos(x, evaluate=False)._eval_expand_trig()
            cy = cos(y, evaluate=False)._eval_expand_trig()
            # 返回展开后的结果
            return sx*cy + sy*cx
        elif arg.is_Mul:  # 如果参数是乘法表达式
            # 将参数分解为系数和变量
            n, x = arg.as_coeff_Mul(rational=True)
            if n.is_Integer:  # 如果系数是整数
                # 根据 Multiple-Angle Formulas 进行展开
                if n.is_odd:
                    return S.NegativeOne**((n - 1)/2)*chebyshevt(n, sin(x))
                else:
                    return expand_mul(S.NegativeOne**(n/2 - 1)*cos(x)*
                                      chebyshevu(n - 1, sin(x)), deep=False)
        # 返回参数的 sin 展开结果
        return sin(arg)

    # 重新定义函数 `_eval_as_leading_term`，返回表达式的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 导入所需的函数
        from sympy.calculus.accumulationbounds import AccumBounds
        # 获取函数的参数
        arg = self.args[0]
        # 获取在 x = 0 处的值并进行简化
        x0 = arg.subs(x, 0).cancel()
        # 计算 x0/pi 的整数部分
        n = x0/pi
        # 如果 n 是整数
        if n.is_integer:
            # 计算 (arg - n*pi).as_leading_term(x) 并乘以 (-1)^n
            lt = (arg - n*pi).as_leading_term(x)
            return (S.NegativeOne**n)*lt
        # 如果 x0 是复无穷
        if x0 is S.ComplexInfinity:
            # 根据 cdir 的符号确定 x0 的值
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        # 如果 x0 是无穷大，则返回 AccumBounds(-1, 1)
        if x0 in [S.Infinity, S.NegativeInfinity]:
            return AccumBounds(-1, 1)
        # 如果 x0 是有限值，则返回原函数在 x0 处的值，否则返回原函数自身
        return self.func(x0) if x0.is_finite else self
    # 检查表达式是否是扩展实数（包括实数和正负无穷）
    def _eval_is_extended_real(self):
        # 如果表达式的第一个参数是扩展实数，则返回True
        if self.args[0].is_extended_real:
            return True

    # 检查表达式是否是有限数
    def _eval_is_finite(self):
        # 获取表达式的第一个参数
        arg = self.args[0]
        # 如果第一个参数是扩展实数，则返回True
        if arg.is_extended_real:
            return True

    # 检查表达式是否为零
    def _eval_is_zero(self):
        # 将表达式的第一个参数拆解，分离其中的零部分和π的倍数部分
        rest, pi_mult = _peeloff_pi(self.args[0])
        # 如果零部分为零，则返回π的倍数部分是否为整数的结果
        if rest.is_zero:
            return pi_mult.is_integer

    # 检查表达式是否是复数
    def _eval_is_complex(self):
        # 如果表达式的第一个参数是扩展实数或者复数，则返回True
        if self.args[0].is_extended_real \
                or self.args[0].is_complex:
            return True
class cos(TrigonometricFunction):
    """
    The cosine function.

    Returns the cosine of x (measured in radians).

    Explanation
    ===========

    See :func:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import cos, pi
    >>> from sympy.abc import x
    >>> cos(x**2).diff(x)
    -2*x*sin(x**2)
    >>> cos(1).diff(x)
    0
    >>> cos(pi)
    -1
    >>> cos(pi/2)
    0
    >>> cos(2*pi/3)
    -1/2
    >>> cos(pi/12)
    sqrt(2)/4 + sqrt(6)/4

    See Also
    ========

    sin, csc, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Cos

    """

    def period(self, symbol=None):
        # 返回周期，如果未指定符号，则默认为 2*pi
        return self._period(2*pi, symbol)

    def fdiff(self, argindex=1):
        # 求导函数，对于参数索引为 1 的情况，返回负的 sin(self.args[0])
        if argindex == 1:
            return -sin(self.args[0])
        else:
            # 对于其他索引，抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    @classmethod
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # 计算泰勒展开项，如果 n 小于 0 或者 n 是奇数，则返回零
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            # 如果 previous_terms 的长度大于 2，则根据公式返回值
            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return -p*x**2/(n*(n - 1))
            else:
                # 否则，根据泰勒展开的标准公式返回值
                return S.NegativeOne**(n//2)*x**n/factorial(n)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # 对函数进行泰勒级数展开，处理特殊情况，如 logx 不为 None
        arg = self.args[0]
        if logx is not None:
            arg = arg.subs(log(x), logx)
        # 如果在 x = 0 处有不确定值，则抛出极点错误
        if arg.subs(x, 0).has(S.NaN, S.ComplexInfinity):
            raise PoleError("Cannot expand %s around 0" % (self))
        return Function._eval_nseries(self, x, n=n, logx=logx, cdir=cdir)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        # 将余弦函数重写为指数函数的形式，处理特定的参数类型
        I = S.ImaginaryUnit
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp, **kwargs)
        return (exp(arg*I) + exp(-arg*I))/2

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        # 将余弦函数重写为幂函数的形式，处理特定的参数类型
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return x**I/2 + x**-I/2

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        # 将余弦函数重写为正弦函数的形式，加上 pi/2 的偏移
        return sin(arg + pi/2, evaluate=False)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        # 将余弦函数重写为正切函数的形式
        tan_half = tan(S.Half*arg)**2
        return (1 - tan_half)/(1 + tan_half)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        # 将余弦函数重写为正弦函数与余弦函数的商的形式
        return sin(arg)*cos(arg)/sin(arg)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        # 将余弦函数重写为余切函数的形式，考虑特定的参数情况
        cot_half = cot(S.Half*arg)**2
        return Piecewise((1, And(Eq(im(arg), 0), Eq(Mod(arg, 2*pi), 0))),
                         ((cot_half - 1)/(cot_half + 1), True))

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        # 将余弦函数重写为幂函数的形式，调用 _eval_rewrite_as_sqrt 方法
        return self._eval_rewrite_as_sqrt(arg, **kwargs)
    # 将表达式重写为以 sqrt 为基础的形式
    def _eval_rewrite_as_sqrt(self, arg: Expr, **kwargs):
        # 导入特殊多项式函数 chebyshevt
        from sympy.functions.special.polynomials import chebyshevt

        # 获取 arg 的 π 系数
        pi_coeff = _pi_coeff(arg)
        # 如果没有找到 π 系数，则返回 None
        if pi_coeff is None:
            return None

        # 如果 π 系数是整数，则返回 None
        if isinstance(pi_coeff, Integer):
            return None

        # 如果 π 系数不是有理数，则返回 None
        if not isinstance(pi_coeff, Rational):
            return None

        # 获取余弦表格
        cst_table_some = cos_table()

        # 如果 π 系数的分母在余弦表格中
        if pi_coeff.q in cst_table_some:
            # 使用 chebyshevt 函数计算结果
            rv = chebyshevt(pi_coeff.p, cst_table_some[pi_coeff.q]())
            # 如果 π 系数的分母小于 257，则展开结果
            if pi_coeff.q < 257:
                rv = rv.expand()
            return rv

        # 如果 π 系数的分母是偶数，递归移除因子 2
        if not pi_coeff.q % 2:
            pico2 = pi_coeff * 2
            # 使用 cos 函数重写为以 sqrt 为基础的形式
            nval = cos(pico2 * pi).rewrite(sqrt, **kwargs)
            # 计算 x 值
            x = (pico2 + 1) / 2
            # 确定符号
            sign_cos = -1 if int(x) % 2 else 1
            # 返回符号乘以 sqrt((1 + nval) / 2) 的结果
            return sign_cos * sqrt((1 + nval) / 2)

        # 计算 fermat_coords 的结果
        FC = fermat_coords(pi_coeff.q)
        # 如果 FC 存在，则使用 FC 的值
        if FC:
            denoms = FC
        else:
            # 否则，使用 factorint 的结果作为分母
            denoms = [b**e for b, e in factorint(pi_coeff.q).items()]

        # 计算 ipartfrac 的结果
        apart = ipartfrac(*denoms)
        # 生成分解的元组
        decomp = (pi_coeff.p * Rational(n, d) for n, d in zip(apart, denoms))
        # 生成 X 列表
        X = [(x[1], x[0]*pi) for x in zip(decomp, numbered_symbols('z'))]
        # 计算 cos 的和，并展开三角函数
        pcls = cos(sum(x[0] for x in X))._eval_expand_trig().subs(X)

        # 如果 FC 不存在或者 FC 的长度为 1，则返回 pcls
        if not FC or len(FC) == 1:
            return pcls
        # 否则，使用 sqrt 为基础的形式重写 pcls 的结果
        return pcls.rewrite(sqrt, **kwargs)

    # 将表达式重写为 sec 函数的形式
    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return 1/sec(arg)

    # 将表达式重写为 csc 函数的形式
    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return 1/sec(arg).rewrite(csc, **kwargs)

    # 将表达式重写为 besselj 函数的形式
    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        # 导入贝塞尔函数 besselj
        from sympy.functions.special.bessel import besselj
        # 返回 Piecewise 对象，根据条件返回不同的表达式
        return Piecewise(
                (sqrt(pi*arg/2)*besselj(-S.Half, arg), Ne(arg, 0)),
                (1, True)
            )

    # 计算共轭复数
    def _eval_conjugate(self):
        # 返回当前函数的共轭结果
        return self.func(self.args[0].conjugate())

    # 将表达式转换为实部和虚部的形式
    def as_real_imag(self, deep=True, **hints):
        # 导入双曲函数 cosh 和 sinh
        from sympy.functions.elementary.hyperbolic import cosh, sinh
        # 计算实部和虚部
        re, im = self._as_real_imag(deep=deep, **hints)
        # 返回 cos(re)*cosh(im) 和 -sin(re)*sinh(im) 的结果
        return (cos(re)*cosh(im), -sin(re)*sinh(im))

    # 展开三角函数表达式
    def _eval_expand_trig(self, **hints):
        # 导入特殊多项式函数 chebyshevt
        from sympy.functions.special.polynomials import chebyshevt
        # 获取参数 arg
        arg = self.args[0]
        # 初始化 x 为 None
        x = None
        # 如果 arg 是加法表达式
        if arg.is_Add:  # TODO: Do this more efficiently for more than two terms
            # 将 arg 拆分为两项
            x, y = arg.as_two_terms()
            # 展开 sin(x) 和 cos(x)
            sx = sin(x, evaluate=False)._eval_expand_trig()
            sy = sin(y, evaluate=False)._eval_expand_trig()
            cx = cos(x, evaluate=False)._eval_expand_trig()
            cy = cos(y, evaluate=False)._eval_expand_trig()
            # 返回展开后的结果
            return cx*cy - sx*sy
        # 如果 arg 是乘法表达式
        elif arg.is_Mul:
            # 将 arg 拆分为系数和项
            coeff, terms = arg.as_coeff_Mul(rational=True)
            # 如果系数是整数，则使用 chebyshevt 函数
            if coeff.is_Integer:
                return chebyshevt(coeff, cos(terms))
        # 对于其他情况，直接返回 cos(arg) 的结果
        return cos(arg)
    # 计算作为主导项时的表达式值
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 导入累积边界模块中的AccumBounds类
        from sympy.calculus.accumulationbounds import AccumBounds
        # 获取表达式中的第一个参数
        arg = self.args[0]
        # 计算在 x = 0 处的极限并化简
        x0 = arg.subs(x, 0).cancel()
        # 计算 n = (x0 + pi/2)/pi，其中 n 是 x0/pi 的整数部分
        n = (x0 + pi/2)/pi
        # 如果 n 是整数
        if n.is_integer:
            # 计算作为主导项的表达式
            lt = (arg - n*pi + pi/2).as_leading_term(x)
            return (S.NegativeOne**n)*lt
        # 如果 x0 是复无穷大
        if x0 is S.ComplexInfinity:
            # 根据方向 cdir 计算 x0 的极限
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        # 如果 x0 是正无穷大或负无穷大
        if x0 in [S.Infinity, S.NegativeInfinity]:
            # 返回累积边界 (-1, 1)
            return AccumBounds(-1, 1)
        # 如果 x0 是有限的则返回 self.func(x0)，否则返回 self 本身
        return self.func(x0) if x0.is_finite else self

    # 判断表达式是否是扩展实数
    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    # 判断表达式是否是有限的
    def _eval_is_finite(self):
        arg = self.args[0]
        # 如果表达式的参数是扩展实数则返回 True
        if arg.is_extended_real:
            return True

    # 判断表达式是否是复数
    def _eval_is_complex(self):
        # 如果表达式的参数是扩展实数或复数则返回 True
        if self.args[0].is_extended_real \
            or self.args[0].is_complex:
            return True

    # 判断表达式是否为零
    def _eval_is_zero(self):
        # 使用 _peeloff_pi 函数将表达式的参数分离为 rest 和 pi_mult
        rest, pi_mult = _peeloff_pi(self.args[0])
        # 如果 rest 是零并且 pi_mult 不为空
        if rest.is_zero and pi_mult:
            # 返回 (pi_mult - S.Half) 是否为整数
            return (pi_mult - S.Half).is_integer
class tan(TrigonometricFunction):
    """
    The tangent function.

    Returns the tangent of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import tan, pi
    >>> from sympy.abc import x
    >>> tan(x**2).diff(x)
    2*x*(tan(x**2)**2 + 1)
    >>> tan(1).diff(x)
    0
    >>> tan(pi/8).expand()
    -1 + sqrt(2)

    See Also
    ========

    sin, csc, cos, sec, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Tan

    """

    def period(self, symbol=None):
        """
        Returns the period of the tangent function.
        """
        return self._period(pi, symbol)

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of the tangent function.
        """
        if argindex == 1:
            return S.One + self**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of the tangent function, which is arctangent.
        """
        return atan

    @classmethod
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Returns the nth term of the Taylor series expansion of tangent.
        """
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            a, b = ((n - 1)//2), 2**(n + 1)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return S.NegativeOne**a*b*(b - 1)*B/F*x**n

    def _eval_nseries(self, x, n, logx, cdir=0):
        """
        Evaluates the tangent function using its series expansion.
        """
        i = self.args[0].limit(x, 0)*2/pi
        if i and i.is_Integer:
            return self.rewrite(cos)._eval_nseries(x, n=n, logx=logx)
        return Function._eval_nseries(self, x, n=n, logx=logx)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        """
        Rewrites the tangent function using the exponential form.
        """
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return I*(x**-I - x**I)/(x**-I + x**I)

    def _eval_conjugate(self):
        """
        Returns the conjugate of the tangent function.
        """
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        """
        Returns the tangent function in terms of its real and imaginary components.
        """
        re, im = self._as_real_imag(deep=deep, **hints)
        if im:
            from sympy.functions.elementary.hyperbolic import cosh, sinh
            denom = cos(2*re) + cosh(2*im)
            return (sin(2*re)/denom, sinh(2*im)/denom)
        else:
            return (self.func(re), S.Zero)
    # 将三角函数表达式展开成更简单的形式，根据参数中的提示
    def _eval_expand_trig(self, **hints):
        # 获取表达式的第一个参数
        arg = self.args[0]
        x = None
        # 如果参数是加法表达式
        if arg.is_Add:
            # 获取加法表达式中的项数
            n = len(arg.args)
            TX = []
            # 对每个加法表达式的项进行处理
            for x in arg.args:
                # 对 tan 函数应用 _eval_expand_trig 方法，生成展开后的结果
                tx = tan(x, evaluate=False)._eval_expand_trig()
                TX.append(tx)

            # 创建一系列唯一符号作为 Y 的标记
            Yg = numbered_symbols('Y')
            Y = [next(Yg) for i in range(n)]

            p = [0, 0]
            # 构建一个对称多项式
            for i in range(n + 1):
                p[1 - i % 2] += symmetric_poly(i, Y)*(-1)**((i % 4)//2)
            # 返回多项式的比值，并对 Y 和 TX 进行替换
            return (p[0]/p[1]).subs(list(zip(Y, TX)))

        # 如果参数是乘法表达式
        elif arg.is_Mul:
            # 将参数拆分为系数和项
            coeff, terms = arg.as_coeff_Mul(rational=True)
            # 如果系数是整数且大于 1
            if coeff.is_Integer and coeff > 1:
                I = S.ImaginaryUnit
                z = Symbol('dummy', real=True)
                # 计算 ((1 + I*z)**coeff) 的展开形式
                P = ((1 + I*z)**coeff).expand()
                # 返回虚部和实部的比值，并对 z 替换为 tan(terms)
                return (im(P)/re(P)).subs([(z, tan(terms))])
        # 对于其他情况，直接返回 tan(arg)
        return tan(arg)

    # 将三角函数表达式重写为指数函数的形式
    def _eval_rewrite_as_exp(self, arg, **kwargs):
        I = S.ImaginaryUnit
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        # 如果参数是三角函数或双曲函数的实例
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            # 将参数应用 func 方法，并以指数函数形式重写
            arg = arg.func(arg.args[0]).rewrite(exp)
        # 计算负指数和正指数
        neg_exp, pos_exp = exp(-arg*I), exp(arg*I)
        # 返回重写后的表达式
        return I*(neg_exp - pos_exp)/(neg_exp + pos_exp)

    # 将三角函数表达式重写为 sin(x) 的形式
    def _eval_rewrite_as_sin(self, x, **kwargs):
        return 2*sin(x)**2/sin(2*x)

    # 将三角函数表达式重写为 cos(x) 的形式
    def _eval_rewrite_as_cos(self, x, **kwargs):
        return cos(x - pi/2, evaluate=False)/cos(x)

    # 将三角函数表达式重写为 sin(x)/cos(x) 的形式
    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg)/cos(arg)

    # 将三角函数表达式重写为 1/cot(x) 的形式
    def _eval_rewrite_as_cot(self, arg, **kwargs):
        return 1/cot(arg)

    # 将三角函数表达式重写为 sec(x) 的形式
    def _eval_rewrite_as_sec(self, arg, **kwargs):
        # 将 sin(x) 和 cos(x) 以 sec(x) 形式重写
        sin_in_sec_form = sin(arg).rewrite(sec, **kwargs)
        cos_in_sec_form = cos(arg).rewrite(sec, **kwargs)
        # 返回 sin(x)/cos(x) 在 sec(x) 形式中的比值
        return sin_in_sec_form/cos_in_sec_form

    # 将三角函数表达式重写为 csc(x) 的形式
    def _eval_rewrite_as_csc(self, arg, **kwargs):
        # 将 sin(x) 和 cos(x) 以 csc(x) 形式重写
        sin_in_csc_form = sin(arg).rewrite(csc, **kwargs)
        cos_in_csc_form = cos(arg).rewrite(csc, **kwargs)
        # 返回 sin(x)/cos(x) 在 csc(x) 形式中的比值
        return sin_in_csc_form/cos_in_csc_form

    # 将三角函数表达式重写为 pow(x) 的形式
    def _eval_rewrite_as_pow(self, arg, **kwargs):
        # 将表达式先以 cos(x) 形式重写，再以 pow(x) 形式重写
        y = self.rewrite(cos, **kwargs).rewrite(pow, **kwargs)
        # 如果结果中仍然包含 cos 函数，则返回 None
        if y.has(cos):
            return None
        # 否则返回重写后的结果
        return y

    # 将三角函数表达式重写为 sqrt(x) 的形式
    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        # 将表达式先以 cos(x) 形式重写，再以 sqrt(x) 形式重写
        y = self.rewrite(cos, **kwargs).rewrite(sqrt, **kwargs)
        # 如果结果中仍然包含 cos 函数，则返回 None
        if y.has(cos):
            return None
        # 否则返回重写后的结果
        return y

    # 将三角函数表达式重写为 besselj(x) 的形式
    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        # 导入贝塞尔函数 besselj
        from sympy.functions.special.bessel import besselj
        # 返回 besselj(1/2, arg) / besselj(-1/2, arg)
        return besselj(S.Half, arg)/besselj(-S.Half, arg)
    # 计算作为主导项的表达式的值
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 导入累积边界模块中的AccumBounds类
        from sympy.calculus.accumulationbounds import AccumBounds
        # 导入复数函数模块中的实部函数re
        from sympy.functions.elementary.complexes import re
        # 获取表达式的第一个参数
        arg = self.args[0]
        # 计算表达式在 x=0 处的极限并化简
        x0 = arg.subs(x, 0).cancel()
        # 计算 n = 2*x0/pi
        n = 2*x0/pi
        # 如果 n 是整数
        if n.is_integer:
            # 计算表达式减去 n*pi/2 的主导项
            lt = (arg - n*pi/2).as_leading_term(x)
            # 如果 n 是偶数，返回 lt，否则返回 -1/lt
            return lt if n.is_even else -1/lt
        # 如果 x0 是无穷大
        if x0 is S.ComplexInfinity:
            # 计算表达式在 x=0 处的极限，方向由 cdir 决定
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        # 如果 x0 是正无穷大或负无穷大，返回负无穷到正无穷的累积边界
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        # 如果 x0 是有限的实数，则返回 self.func(x0)，否则返回 self 本身
        return self.func(x0) if x0.is_finite else self

    # 判断 tan 函数是否是扩展实数
    def _eval_is_extended_real(self):
        # FIXME: 目前 tan(pi/2) 返回 zoo
        return self.args[0].is_extended_real

    # 判断 tan 函数是否是实数
    def _eval_is_real(self):
        arg = self.args[0]
        # 如果 arg 是实数且 (arg/pi - S.Half) 不是整数，返回 True
        if arg.is_real and (arg/pi - S.Half).is_integer is False:
            return True

    # 判断 tan 函数是否是有限的
    def _eval_is_finite(self):
        arg = self.args[0]
        # 如果 arg 是实数且 (arg/pi - S.Half) 不是整数，返回 True
        if arg.is_real and (arg/pi - S.Half).is_integer is False:
            return True
        # 如果 arg 是虚数，返回 True
        if arg.is_imaginary:
            return True

    # 判断 tan 函数是否为零
    def _eval_is_zero(self):
        # 将参数传递给辅助函数 _peeloff_pi 处理，获取余数 rest 和 pi 的倍数 pi_mult
        rest, pi_mult = _peeloff_pi(self.args[0])
        # 如果余数 rest 是零，返回 pi 的倍数 pi_mult 是否是整数
        if rest.is_zero:
            return pi_mult.is_integer

    # 判断 tan 函数是否是复数
    def _eval_is_complex(self):
        arg = self.args[0]
        # 如果 arg 是实数且 (arg/pi - S.Half) 不是整数，返回 True
        if arg.is_real and (arg/pi - S.Half).is_integer is False:
            return True
class cot(TrigonometricFunction):
    """
    The cotangent function.

    Returns the cotangent of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import cot, pi
    >>> from sympy.abc import x
    >>> cot(x**2).diff(x)
    2*x*(-cot(x**2)**2 - 1)
    >>> cot(1).diff(x)
    0
    >>> cot(pi/12)
    sqrt(3) + 2

    See Also
    ========

    sin, csc, cos, sec, tan
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Cot

    """

    def period(self, symbol=None):
        """
        Returns the period of the cotangent function.
        """
        return self._period(pi, symbol)

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of the cotangent function.
        """
        if argindex == 1:
            return S.NegativeOne - self**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of the cotangent function.
        """
        return acot

    @classmethod
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Returns the nth term of the Taylor series expansion of the cotangent function.
        """
        if n == 0:
            return 1/sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return S.NegativeOne**((n + 1)//2)*2**(n + 1)*B/F*x**n

    def _eval_nseries(self, x, n, logx, cdir=0):
        """
        Evaluates the cotangent function using its series expansion.
        """
        i = self.args[0].limit(x, 0)/pi
        if i and i.is_Integer:
            return self.rewrite(cos)._eval_nseries(x, n=n, logx=logx)
        return self.rewrite(tan)._eval_nseries(x, n=n, logx=logx)

    def _eval_conjugate(self):
        """
        Returns the conjugate of the cotangent function.
        """
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        """
        Returns the cotangent function in terms of its real and imaginary parts.
        """
        re, im = self._as_real_imag(deep=deep, **hints)
        if im:
            from sympy.functions.elementary.hyperbolic import cosh, sinh
            denom = cos(2*re) - cosh(2*im)
            return (-sin(2*re)/denom, sinh(2*im)/denom)
        else:
            return (self.func(re), S.Zero)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        """
        Rewrites the cotangent function in terms of exponentials.
        """
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        I = S.ImaginaryUnit
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp, **kwargs)
        neg_exp, pos_exp = exp(-arg*I), exp(arg*I)
        return I*(pos_exp + neg_exp)/(pos_exp - neg_exp)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        """
        Rewrites the cotangent function in terms of powers.
        """
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return -I*(x**-I + x**I)/(x**-I - x**I)

    def _eval_rewrite_as_sin(self, x, **kwargs):
        """
        Rewrites the cotangent function in terms of sine.
        """
        return sin(2*x)/(2*(sin(x)**2))

    def _eval_rewrite_as_cos(self, x, **kwargs):
        """
        Rewrites the cotangent function in terms of cosine.
        """
        return cos(x)/cos(x - pi/2, evaluate=False)
    # 将表达式重写为 cos(arg)/sin(arg) 的形式
    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return cos(arg)/sin(arg)

    # 将表达式重写为 1/tan(arg) 的形式
    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return 1/tan(arg)

    # 将表达式重写为 cos(arg)/sin(arg) 的形式，利用 sec 函数的重写
    def _eval_rewrite_as_sec(self, arg, **kwargs):
        cos_in_sec_form = cos(arg).rewrite(sec, **kwargs)
        sin_in_sec_form = sin(arg).rewrite(sec, **kwargs)
        return cos_in_sec_form/sin_in_sec_form

    # 将表达式重写为 cos(arg)/sin(arg) 的形式，利用 csc 函数的重写
    def _eval_rewrite_as_csc(self, arg, **kwargs):
        cos_in_csc_form = cos(arg).rewrite(csc, **kwargs)
        sin_in_csc_form = sin(arg).rewrite(csc, **kwargs)
        return cos_in_csc_form/sin_in_csc_form

    # 将表达式重写为另一种形式，首先重写为 cos(arg) 形式，再重写为 pow 形式
    def _eval_rewrite_as_pow(self, arg, **kwargs):
        y = self.rewrite(cos, **kwargs).rewrite(pow, **kwargs)
        if y.has(cos):
            return None
        return y

    # 将表达式重写为另一种形式，首先重写为 cos(arg) 形式，再重写为 sqrt 形式
    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        y = self.rewrite(cos, **kwargs).rewrite(sqrt, **kwargs)
        if y.has(cos):
            return None
        return y

    # 将表达式重写为 Bessel 函数的形式
    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return besselj(-S.Half, arg)/besselj(S.Half, arg)

    # 将表达式视为主导项，在某些条件下返回合适的结果
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = 2*x0/pi
        if n.is_integer:
            lt = (arg - n*pi/2).as_leading_term(x)
            return 1/lt if n.is_even else -lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self

    # 判断表达式是否是扩展实数
    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    # 将三角函数表达式展开，处理加法形式和乘法形式
    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        x = None
        if arg.is_Add:
            n = len(arg.args)
            CX = []
            for x in arg.args:
                cx = cot(x, evaluate=False)._eval_expand_trig()
                CX.append(cx)

            Yg = numbered_symbols('Y')
            Y = [ next(Yg) for i in range(n) ]

            p = [0, 0]
            for i in range(n, -1, -1):
                p[(n - i) % 2] += symmetric_poly(i, Y)*(-1)**(((n - i) % 4)//2)
            return (p[0]/p[1]).subs(list(zip(Y, CX)))
        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer and coeff > 1:
                I = S.ImaginaryUnit
                z = Symbol('dummy', real=True)
                P = ((z + I)**coeff).expand()
                return (re(P)/im(P)).subs([(z, cot(terms))])
        return cot(arg)  # XXX sec and csc return 1/cos and 1/sin
    # 判断表达式是否为有限数
    def _eval_is_finite(self):
        # 获取参数
        arg = self.args[0]
        # 如果参数是实数并且不是 pi 的整数倍
        if arg.is_real and (arg/pi).is_integer is False:
            return True
        # 如果参数是虚数，则认为是有限数
        if arg.is_imaginary:
            return True

    # 判断表达式是否为实数
    def _eval_is_real(self):
        # 获取参数
        arg = self.args[0]
        # 如果参数是实数并且不是 pi 的整数倍
        if arg.is_real and (arg/pi).is_integer is False:
            return True

    # 判断表达式是否为复数
    def _eval_is_complex(self):
        # 获取参数
        arg = self.args[0]
        # 如果参数是实数并且不是 pi 的整数倍
        if arg.is_real and (arg/pi).is_integer is False:
            return True

    # 判断表达式是否为零
    def _eval_is_zero(self):
        # 使用 _peeloff_pi 函数分离 pi 倍数和余下部分
        rest, pimult = _peeloff_pi(self.args[0])
        # 如果有 pi 倍数并且余下部分为零
        if pimult and rest.is_zero:
            return (pimult - S.Half).is_integer

    # 替换表达式中的旧变量为新变量
    def _eval_subs(self, old, new):
        # 获取参数
        arg = self.args[0]
        # 使用新变量替换旧变量后的新参数
        argnew = arg.subs(old, new)
        # 如果新旧参数不同并且新参数是 pi 的整数倍
        if arg != argnew and (argnew/pi).is_integer:
            # 返回复数无穷大
            return S.ComplexInfinity
        # 返回余切函数应用于新参数
        return cot(argnew)
class ReciprocalTrigonometricFunction(TrigonometricFunction):
    """Base class for reciprocal functions of trigonometric functions."""

    _reciprocal_of = None       # 要在子类中定义的必需属性：表示被倒数的三角函数
    _singularities = (S.ComplexInfinity,)   # 奇点集合，包括复无穷

    # _is_even and _is_odd are used for correct evaluation of csc(-x), sec(-x)
    # TODO refactor into TrigonometricFunction common parts of
    # trigonometric functions eval() like even/odd, func(x+2*k*pi), etc.
    # _is_even 和 _is_odd 用于正确评估 csc(-x), sec(-x)。
    # TODO：重构为 TrigonometricFunction 的公共部分，如 eval() 中的偶数/奇数，func(x+2*k*pi)等。

    # optional, to be defined in subclasses:
    _is_even: FuzzyBool = None   # 可选属性，在子类中定义：表示是否为偶函数
    _is_odd: FuzzyBool = None    # 可选属性，在子类中定义：表示是否为奇函数

    @classmethod
    def eval(cls, arg):
        # 如果能够提取负号
        if arg.could_extract_minus_sign():
            # 如果是偶函数，则返回其相反数
            if cls._is_even:
                return cls(-arg)
            # 如果是奇函数，则返回其相反数的相反数，即自身
            if cls._is_odd:
                return -cls(-arg)

        # 计算参数中的 pi 系数
        pi_coeff = _pi_coeff(arg)
        if (pi_coeff is not None
            and not (2*pi_coeff).is_integer
            and pi_coeff.is_Rational):
                q = pi_coeff.q
                p = pi_coeff.p % (2*q)
                if p > q:
                    narg = (pi_coeff - 1)*pi
                    return -cls(narg)
                if 2*p > q:
                    narg = (1 - pi_coeff)*pi
                    if cls._is_odd:
                        return cls(narg)
                    elif cls._is_even:
                        return -cls(narg)

        # 如果参数具有 'inverse' 属性，并且其反函数是 cls，则返回参数本身
        if hasattr(arg, 'inverse') and arg.inverse() == cls:
            return arg.args[0]

        # 计算 _reciprocal_of 对参数 arg 的求值
        t = cls._reciprocal_of.eval(arg)
        if t is None:
            return t
        # 如果 t 是 cos 函数或者其相反数，则返回 sec 函数的重写
        elif any(isinstance(i, cos) for i in (t, -t)):
            return (1/t).rewrite(sec)
        # 如果 t 是 sin 函数或者其相反数，则返回 csc 函数的重写
        elif any(isinstance(i, sin) for i in (t, -t)):
            return (1/t).rewrite(csc)
        else:
            return 1/t

    def _call_reciprocal(self, method_name, *args, **kwargs):
        # 在 _reciprocal_of 上调用 method_name 方法
        o = self._reciprocal_of(self.args[0])
        return getattr(o, method_name)(*args, **kwargs)

    def _calculate_reciprocal(self, method_name, *args, **kwargs):
        # 如果在 _reciprocal_of 上调用 method_name 方法返回非 None 值，则返回其倒数
        t = self._call_reciprocal(method_name, *args, **kwargs)
        return 1/t if t is not None else t

    def _rewrite_reciprocal(self, method_name, arg):
        # 对重写函数进行特殊处理。如果倒数的重写返回未修改的表达式，则返回 None
        t = self._call_reciprocal(method_name, arg)
        if t is not None and t != self._reciprocal_of(arg):
            return 1/t

    def _period(self, symbol):
        # 对参数 symbol 进行周期处理
        f = expand_mul(self.args[0])
        return self._reciprocal_of(f).period(symbol)

    def fdiff(self, argindex=1):
        # 返回关于参数的偏导数
        return -self._calculate_reciprocal("fdiff", argindex)/self**2

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        # 将重写为 exp 函数的操作应用于参数 arg
        return self._rewrite_reciprocal("_eval_rewrite_as_exp", arg)
    # 重写当前对象作为幂函数的方法
    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_Pow", arg)

    # 重写当前对象作为正弦函数的方法
    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_sin", arg)

    # 重写当前对象作为余弦函数的方法
    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_cos", arg)

    # 重写当前对象作为正切函数的方法
    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_tan", arg)

    # 重写当前对象作为幂函数的方法
    def _eval_rewrite_as_pow(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_pow", arg)

    # 重写当前对象作为平方根函数的方法
    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_sqrt", arg)

    # 返回当前对象的共轭复数
    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    # 将当前对象的倒数转换为实部和虚部
    def as_real_imag(self, deep=True, **hints):
        return (1/self._reciprocal_of(self.args[0])).as_real_imag(deep,
                                                                  **hints)

    # 对三角函数的展开求逆
    def _eval_expand_trig(self, **hints):
        return self._calculate_reciprocal("_eval_expand_trig", **hints)

    # 判断当前对象的倒数是否为扩展实数
    def _eval_is_extended_real(self):
        return self._reciprocal_of(self.args[0])._eval_is_extended_real()

    # 返回当前对象的倒数作为主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        return (1/self._reciprocal_of(self.args[0]))._eval_as_leading_term(x)

    # 判断当前对象的倒数是否有限
    def _eval_is_finite(self):
        return (1/self._reciprocal_of(self.args[0])).is_finite

    # 对当前对象的倒数进行 n 次数列展开
    def _eval_nseries(self, x, n, logx, cdir=0):
        return (1/self._reciprocal_of(self.args[0]))._eval_nseries(x, n, logx)
class sec(ReciprocalTrigonometricFunction):
    """
    The secant function.

    Returns the secant of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import sec
    >>> from sympy.abc import x
    >>> sec(x**2).diff(x)
    2*x*tan(x**2)*sec(x**2)
    >>> sec(1).diff(x)
    0

    See Also
    ========

    sin, csc, cos, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Sec

    """

    _reciprocal_of = cos  # 设置逆余弦函数作为其倒数
    _is_even = True  # 表示函数是偶函数

    def period(self, symbol=None):
        """
        Return the period of the secant function.

        Parameters
        ==========

        symbol : Symbol, optional
            The symbol with respect to which the period is calculated.

        Returns
        =======

        period : Expression
            The period of the secant function.

        """
        return self._period(symbol)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        """
        Rewrite secant function in terms of cotangent.

        Parameters
        ==========

        arg : Expression
            Argument of the secant function.

        Returns
        =======

        rewritten : Expression
            Secant function rewritten in terms of cotangent.

        """
        cot_half_sq = cot(arg/2)**2
        return (cot_half_sq + 1)/(cot_half_sq - 1)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        """
        Rewrite secant function in terms of cosine.

        Parameters
        ==========

        arg : Expression
            Argument of the secant function.

        Returns
        =======

        rewritten : Expression
            Secant function rewritten in terms of cosine.

        """
        return (1/cos(arg))

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        """
        Rewrite secant function in terms of sine and cosine.

        Parameters
        ==========

        arg : Expression
            Argument of the secant function.

        Returns
        =======

        rewritten : Expression
            Secant function rewritten in terms of sine, cosine, and their product.

        """
        return sin(arg)/(cos(arg)*sin(arg))

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        """
        Rewrite secant function in terms of sine.

        Parameters
        ==========

        arg : Expression
            Argument of the secant function.

        Returns
        =======

        rewritten : Expression
            Secant function rewritten in terms of sine and cosine.

        """
        return (1/cos(arg).rewrite(sin, **kwargs))

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        """
        Rewrite secant function in terms of tangent.

        Parameters
        ==========

        arg : Expression
            Argument of the secant function.

        Returns
        =======

        rewritten : Expression
            Secant function rewritten in terms of tangent.

        """
        return (1/cos(arg).rewrite(tan, **kwargs))

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        """
        Rewrite secant function in terms of cosecant.

        Parameters
        ==========

        arg : Expression
            Argument of the secant function.

        Returns
        =======

        rewritten : Expression
            Secant function rewritten in terms of cosecant.

        """
        return csc(pi/2 - arg, evaluate=False)

    def fdiff(self, argindex=1):
        """
        Differentiate secant function with respect to its argument.

        Parameters
        ==========

        argindex : int, optional
            Index of the argument with respect to which differentiation is done.

        Returns
        =======

        diff_result : Expression
            The result of differentiation.

        Raises
        ======

        ArgumentIndexError
            If the argument index is not valid.

        """
        if argindex == 1:
            return tan(self.args[0])*sec(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        """
        Rewrite secant function in terms of Bessel function.

        Parameters
        ==========

        arg : Expression
            Argument of the secant function.

        Returns
        =======

        rewritten : Piecewise
            Secant function rewritten in terms of Bessel function.

        """
        from sympy.functions.special.bessel import besselj
        return Piecewise(
                (1/(sqrt(pi*arg)/(sqrt(2))*besselj(-S.Half, arg)), Ne(arg, 0)),
                (1, True)
            )

    def _eval_is_complex(self):
        """
        Check if the argument of secant function is complex.

        Returns
        =======

        is_complex : bool
            True if the argument is complex and not an integer multiple of pi/2, False otherwise.

        """
        arg = self.args[0]

        if arg.is_complex and (arg/pi - S.Half).is_integer is False:
            return True

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Return the nth Taylor expansion term for the secant function.

        Parameters
        ==========

        n : int
            Index of the Taylor series term.

        x : Symbol or Expression
            Point around which the Taylor series is expanded.

        previous_terms : tuple of Expressions
            Previous terms in the Taylor series.

        Returns
        =======

        term : Expression
            The nth term of the Taylor series expansion of the secant function.

        """
        # Reference Formula:
        # https://functions.wolfram.com/ElementaryFunctions/Sec/06/01/02/01/
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            k = n//2
            return S.NegativeOne**k*euler(2*k)/factorial(2*k)*x**(2*k)
    # 定义一个方法用于计算表达式在 x 趋近零时的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 导入积分累积边界的计算模块
        from sympy.calculus.accumulationbounds import AccumBounds
        # 导入复数的实部函数
        from sympy.functions.elementary.complexes import re
        # 获取当前表达式的第一个参数
        arg = self.args[0]
        # 计算在 x 等于零时的极限，并化简结果
        x0 = arg.subs(x, 0).cancel()
        # 计算 n 的值，用于后续的处理
        n = (x0 + pi/2)/pi
        # 如果 n 是整数
        if n.is_integer:
            # 计算表达式减去 n*pi + pi/2 的主导项
            lt = (arg - n*pi + pi/2).as_leading_term(x)
            # 返回 (-1)^n 除以 lt 的结果
            return (S.NegativeOne**n)/lt
        # 如果 x0 是复无穷
        if x0 is S.ComplexInfinity:
            # 在 x 趋近零的情况下，计算 arg 的极限
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        # 如果 x0 是正无穷或负无穷
        if x0 in (S.Infinity, S.NegativeInfinity):
            # 返回负无穷到正无穷的累积边界
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        # 如果 x0 是有限的，则返回 self 在 x0 处的值；否则返回 self 本身
        return self.func(x0) if x0.is_finite else self
class csc(ReciprocalTrigonometricFunction):
    """
    The cosecant function.

    Returns the cosecant of x (measured in radians).

    Explanation
    ===========

    See :func:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import csc
    >>> from sympy.abc import x
    >>> csc(x**2).diff(x)
    -2*x*cot(x**2)*csc(x**2)
    >>> csc(1).diff(x)
    0

    See Also
    ========

    sin, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Csc

    """

    # 逆三角函数的倒数是正弦函数
    _reciprocal_of = sin
    # 是奇函数
    _is_odd = True

    # 计算周期
    def period(self, symbol=None):
        return self._period(symbol)

    # 以正弦函数重写
    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return (1/sin(arg))

    # 以正弦余弦函数重写
    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return cos(arg)/(sin(arg)*cos(arg))

    # 以余切函数重写
    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half = cot(arg/2)
        return (1 + cot_half**2)/(2*cot_half)

    # 以余弦函数重写
    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return 1/sin(arg).rewrite(cos, **kwargs)

    # 以正割函数重写
    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return sec(pi/2 - arg, evaluate=False)

    # 以正切函数重写
    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return (1/sin(arg).rewrite(tan, **kwargs))

    # 以贝塞尔函数重写
    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return sqrt(2/pi)*(1/(sqrt(arg)*besselj(S.Half, arg)))

    # 对参数求导数
    def fdiff(self, argindex=1):
        if argindex == 1:
            return -cot(self.args[0])*csc(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    # 判断是否为复数
    def _eval_is_complex(self):
        arg = self.args[0]
        if arg.is_real and (arg/pi).is_integer is False:
            return True

    # 静态方法，使用缓存计算泰勒级数项
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1/sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = n//2 + 1
            return (S.NegativeOne**(k - 1)*2*(2**(2*k - 1) - 1)*
                    bernoulli(2*k)*x**(2*k - 1)/factorial(2*k))

    # 计算主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = x0/pi
        if n.is_integer:
            lt = (arg - n*pi).as_leading_term(x)
            return (S.NegativeOne**n)/lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self
class sinc(Function):
    r"""
    Represents an unnormalized sinc function:

    .. math::

        \operatorname{sinc}(x) =
        \begin{cases}
          \frac{\sin x}{x} & \qquad x \neq 0 \\
          1 & \qquad x = 0
        \end{cases}

    Examples
    ========

    >>> from sympy import sinc, oo, jn
    >>> from sympy.abc import x
    >>> sinc(x)
    sinc(x)

    * Automated Evaluation

    >>> sinc(0)
    1
    >>> sinc(oo)
    0

    * Differentiation

    >>> sinc(x).diff()
    cos(x)/x - sin(x)/x**2

    * Series Expansion

    >>> sinc(x).series()
    1 - x**2/6 + x**4/120 + O(x**6)

    * As zero'th order spherical Bessel Function

    >>> sinc(x).rewrite(jn)
    jn(0, x)

    See also
    ========

    sin

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Sinc_function

    """
    _singularities = (S.ComplexInfinity,)  # 定义函数的奇点为复无穷

    def fdiff(self, argindex=1):
        x = self.args[0]
        if argindex == 1:
            # 在这里我们想返回 Piecewise，但 Piecewise.diff
            # 目前无法处理可移除奇点，这意味着像 sinc(x).diff(x, 2) 在 x = 0 处给出了错误的答案。参见
            # https://github.com/sympy/sympy/issues/11402.
            #
            # return Piecewise(((x*cos(x) - sin(x))/x**2, Ne(x, S.Zero)), (S.Zero, S.true))
            return cos(x)/x - sin(x)/x**2  # 返回 sinc 函数的导数表达式
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.One  # 如果参数是零，返回 1
        if arg.is_Number:
            if arg in [S.Infinity, S.NegativeInfinity]:
                return S.Zero  # 如果参数是正无穷或负无穷，返回 0
            elif arg is S.NaN:
                return S.NaN  # 如果参数是 NaN，返回 NaN

        if arg is S.ComplexInfinity:
            return S.NaN  # 如果参数是复无穷，返回 NaN

        if arg.could_extract_minus_sign():
            return cls(-arg)  # 如果参数可能具有负号，返回其相反数的 sinc 函数

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                if fuzzy_not(arg.is_zero):
                    return S.Zero  # 如果参数是整数倍数π且不为零，则返回 0
            elif (2*pi_coeff).is_integer:
                return S.NegativeOne**(pi_coeff - S.Half)/arg  # 如果参数是半整数倍数π，则返回对应的表达式

    def _eval_nseries(self, x, n, logx, cdir=0):
        x = self.args[0]
        return (sin(x)/x)._eval_nseries(x, n, logx)  # 返回 sinc 函数在 x 处的 n 阶级数展开结果

    def _eval_rewrite_as_jn(self, arg, **kwargs):
        from sympy.functions.special.bessel import jn
        return jn(0, arg)  # 返回 sinc 函数的重写，使用零阶球贝塞尔函数 jn(0, arg)

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return Piecewise((sin(arg)/arg, Ne(arg, S.Zero)), (S.One, S.true))  # 返回 sinc 函数的重写，使用 sin 函数的形式

    def _eval_is_zero(self):
        if self.args[0].is_infinite:
            return True  # 如果参数是无穷大，则 sinc 函数为零
        rest, pi_mult = _peeloff_pi(self.args[0])
        if rest.is_zero:
            return fuzzy_and([pi_mult.is_integer, pi_mult.is_nonzero])  # 如果参数是π的整数倍且非零，则 sinc 函数为零
        if rest.is_Number and pi_mult.is_integer:
            return False

    def _eval_is_real(self):
        if self.args[0].is_extended_real or self.args[0].is_imaginary:
            return True  # 如果参数是扩展实数或虚数，则 sinc 函数为实数
    # 将 _eval_is_finite 的引用指向 _eval_is_real 的函数或方法
    _eval_is_finite = _eval_is_real
###############################################################################
########################### TRIGONOMETRIC INVERSES ############################
###############################################################################


class InverseTrigonometricFunction(Function):
    """Base class for inverse trigonometric functions."""
    # 定义特定数学表达式的元组，表示反三角函数的奇点
    _singularities = (S.One, S.NegativeOne, S.Zero, S.ComplexInfinity)  # type: tTuple[Expr, ...]

    @staticmethod
    @cacheit
    def _asin_table():
        # 返回一个字典，包含特定角度对应的反正弦值
        # 只有对应的键的 could_extract_minus_sign() == False 才是必需的
        return {
            sqrt(3)/2: pi/3,
            sqrt(2)/2: pi/4,
            1/sqrt(2): pi/4,
            sqrt((5 - sqrt(5))/8): pi/5,
            sqrt(2)*sqrt(5 - sqrt(5))/4: pi/5,
            sqrt((5 + sqrt(5))/8): pi*Rational(2, 5),
            sqrt(2)*sqrt(5 + sqrt(5))/4: pi*Rational(2, 5),
            S.Half: pi/6,
            sqrt(2 - sqrt(2))/2: pi/8,
            sqrt(S.Half - sqrt(2)/4): pi/8,
            sqrt(2 + sqrt(2))/2: pi*Rational(3, 8),
            sqrt(S.Half + sqrt(2)/4): pi*Rational(3, 8),
            (sqrt(5) - 1)/4: pi/10,
            (1 - sqrt(5))/4: -pi/10,
            (sqrt(5) + 1)/4: pi*Rational(3, 10),
            sqrt(6)/4 - sqrt(2)/4: pi/12,
            -sqrt(6)/4 + sqrt(2)/4: -pi/12,
            (sqrt(3) - 1)/sqrt(8): pi/12,
            (1 - sqrt(3))/sqrt(8): -pi/12,
            sqrt(6)/4 + sqrt(2)/4: pi*Rational(5, 12),
            (1 + sqrt(3))/sqrt(8): pi*Rational(5, 12)
        }


    @staticmethod
    @cacheit
    def _atan_table():
        # 返回一个字典，包含特定角度对应的反正切值
        # 只有对应的键的 could_extract_minus_sign() == False 才是必需的
        return {
            sqrt(3)/3: pi/6,
            1/sqrt(3): pi/6,
            sqrt(3): pi/3,
            sqrt(2) - 1: pi/8,
            1 - sqrt(2): -pi/8,
            1 + sqrt(2): pi*Rational(3, 8),
            sqrt(5 - 2*sqrt(5)): pi/5,
            sqrt(5 + 2*sqrt(5)): pi*Rational(2, 5),
            sqrt(1 - 2*sqrt(5)/5): pi/10,
            sqrt(1 + 2*sqrt(5)/5): pi*Rational(3, 10),
            2 - sqrt(3): pi/12,
            -2 + sqrt(3): -pi/12,
            2 + sqrt(3): pi*Rational(5, 12)
        }

    @staticmethod
    @cacheit
    # 缓存装饰器的应用，用于优化函数调用性能
    # 定义一个函数 _acsc_table()，返回一个字典，该字典包含一些特定的键值对
    def _acsc_table():
        # 返回一个字典，其中包含一些特定的键值对，这些键对应的值是角度的弧度表示
        return {
            2*sqrt(3)/3: pi/3,                           # 键: 2 * sqrt(3) / 3，值: pi / 3
            sqrt(2): pi/4,                               # 键: sqrt(2)，值: pi / 4
            sqrt(2 + 2*sqrt(5)/5): pi/5,                  # 键: sqrt(2 + 2 * sqrt(5) / 5)，值: pi / 5
            1/sqrt(Rational(5, 8) - sqrt(5)/8): pi/5,     # 键: 1 / sqrt(5/8 - sqrt(5)/8)，值: pi / 5
            sqrt(2 - 2*sqrt(5)/5): pi*Rational(2, 5),     # 键: sqrt(2 - 2 * sqrt(5) / 5)，值: pi * 2 / 5
            1/sqrt(Rational(5, 8) + sqrt(5)/8): pi*Rational(2, 5),  # 键: 1 / sqrt(5/8 + sqrt(5)/8)，值: pi * 2 / 5
            2: pi/6,                                      # 键: 2，值: pi / 6
            sqrt(4 + 2*sqrt(2)): pi/8,                    # 键: sqrt(4 + 2 * sqrt(2))，值: pi / 8
            2/sqrt(2 - sqrt(2)): pi/8,                    # 键: 2 / sqrt(2 - sqrt(2))，值: pi / 8
            sqrt(4 - 2*sqrt(2)): pi*Rational(3, 8),       # 键: sqrt(4 - 2 * sqrt(2))，值: pi * 3 / 8
            2/sqrt(2 + sqrt(2)): pi*Rational(3, 8),       # 键: 2 / sqrt(2 + sqrt(2))，值: pi * 3 / 8
            1 + sqrt(5): pi/10,                           # 键: 1 + sqrt(5)，值: pi / 10
            sqrt(5) - 1: pi*Rational(3, 10),              # 键: sqrt(5) - 1，值: pi * 3 / 10
            -(sqrt(5) - 1): pi*Rational(-3, 10),          # 键: -(sqrt(5) - 1)，值: pi * (-3) / 10
            sqrt(6) + sqrt(2): pi/12,                     # 键: sqrt(6) + sqrt(2)，值: pi / 12
            sqrt(6) - sqrt(2): pi*Rational(5, 12),        # 键: sqrt(6) - sqrt(2)，值: pi * 5 / 12
            -(sqrt(6) - sqrt(2)): pi*Rational(-5, 12)    # 键: -(sqrt(6) - sqrt(2))，值: pi * (-5) / 12
        }
class asin(InverseTrigonometricFunction):
    r"""
    The inverse sine function.

    Returns the arcsine of x in radians.

    Explanation
    ===========

    ``asin(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$ and for some instances when the
    result is a rational multiple of $\pi$ (see the ``eval`` class method).

    A purely imaginary argument will lead to an asinh expression.

    Examples
    ========

    >>> from sympy import asin, oo
    >>> asin(1)
    pi/2
    >>> asin(-1)
    -pi/2
    >>> asin(-oo)
    oo*I
    >>> asin(oo)
    -oo*I

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSin

    """

    def fdiff(self, argindex=1):
        # 如果求导的参数索引是1，返回 arcsin 函数的导数公式
        if argindex == 1:
            return 1/sqrt(1 - self.args[0]**2)
        else:
            # 如果参数索引不是1，抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    def _eval_is_rational(self):
        # 计算当前函数表达式
        s = self.func(*self.args)
        # 检查表达式是否与当前函数相同
        if s.func == self.func:
            # 如果表达式的第一个参数是有理数，返回 False
            if s.args[0].is_rational:
                return False
        else:
            # 否则返回表达式是否为有理数的结果
            return s.is_rational

    def _eval_is_positive(self):
        # 检查函数值是否为扩展实数，并且参数是否为正数
        return self._eval_is_extended_real() and self.args[0].is_positive

    def _eval_is_negative(self):
        # 检查函数值是否为扩展实数，并且参数是否为负数
        return self._eval_is_extended_real() and self.args[0].is_negative

    @classmethod
    # 定义一个类方法 eval，用于计算给定参数 arg 的特定数学表达式的值
    def eval(cls, arg):
        # 如果 arg 是一个数值对象
        if arg.is_Number:
            # 检查是否为 NaN
            if arg is S.NaN:
                return S.NaN  # 返回 NaN
            # 检查是否为正无穷
            elif arg is S.Infinity:
                return S.NegativeInfinity * S.ImaginaryUnit  # 返回负无穷乘虚数单位
            # 检查是否为负无穷
            elif arg is S.NegativeInfinity:
                return S.Infinity * S.ImaginaryUnit  # 返回正无穷乘虚数单位
            # 检查是否为零
            elif arg.is_zero:
                return S.Zero  # 返回零
            # 检查是否为一
            elif arg is S.One:
                return pi / 2  # 返回 pi/2
            # 检查是否为负一
            elif arg is S.NegativeOne:
                return -pi / 2  # 返回 -pi/2

        # 如果 arg 是复数无穷
        if arg is S.ComplexInfinity:
            return S.ComplexInfinity  # 返回复数无穷

        # 如果可以提取 arg 的负号
        if arg.could_extract_minus_sign():
            return -cls(-arg)  # 返回负参数的相反数

        # 如果 arg 是一个数值
        if arg.is_number:
            # 获取 cls 类的 asin_table 静态方法返回的表格
            asin_table = cls._asin_table()
            # 如果 arg 在 asin_table 中
            if arg in asin_table:
                return asin_table[arg]  # 返回 asin_table[arg]

        # 检查 arg 是否可以表示为虚数单位乘以某个值
        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            # 导入 asinh 函数并返回虚数单位乘以 asinh(i_coeff) 的结果
            from sympy.functions.elementary.hyperbolic import asinh
            return S.ImaginaryUnit * asinh(i_coeff)

        # 如果 arg 是零
        if arg.is_zero:
            return S.Zero  # 返回零

        # 如果 arg 是 sin 函数的实例
        if isinstance(arg, sin):
            # 获取 sin 函数的参数 ang
            ang = arg.args[0]
            # 如果 ang 是可比较的
            if ang.is_comparable:
                ang %= 2 * pi  # 取模运算限制在 [0, 2*pi) 内
                if ang > pi:  # 将 ang 限制在 (-pi, pi] 内
                    ang = pi - ang

                # 将 ang 限制在 [-pi/2, pi/2] 内
                if ang > pi / 2:
                    ang = pi - ang
                if ang < -pi / 2:
                    ang = -pi - ang

                return ang  # 返回计算后的 ang

        # 如果 arg 是 cos 函数的实例
        if isinstance(arg, cos):
            # 获取 cos 函数的参数 ang
            ang = arg.args[0]
            # 如果 ang 是可比较的
            if ang.is_comparable:
                return pi / 2 - acos(arg)  # 返回 pi/2 减去 ang 的反余弦值
    # 定义一个函数用于计算该函数在 x 处的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):  # asin
        # 获取函数的参数
        arg = self.args[0]
        # 计算参数在 x=0 处的主导项，并消去可能的无效结果
        x0 = arg.subs(x, 0).cancel()
        # 如果 x0 是 NaN，则返回函数自身在 arg 的主导项
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        # 如果 x0 是零，则返回参数 arg 的主导项
        if x0.is_zero:
            return arg.as_leading_term(x)

        # 处理分支点
        if x0 in (-S.One, S.One, S.ComplexInfinity):
            # 将函数重写为对数形式，然后计算其在 x 处的主导项并展开
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # 处理位于分支切割线 (-oo, -1) U (1, oo) 上的点
        if (1 - x0**2).is_negative:
            # 计算参数 arg 在 x 处的方向，并检查其虚部是否为负
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_negative:
                    return -pi - self.func(x0)
            elif im(ndir).is_positive:
                if x0.is_positive:
                    return pi - self.func(x0)
            else:
                # 将函数重写为对数形式，然后计算其在 x 处的主导项并展开
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # 返回函数在 x0 处的值
        return self.func(x0)

    # 定义一个函数用于计算该函数在 x 处的 n 级数展开
    def _eval_nseries(self, x, n, logx, cdir=0):  # asin
        from sympy.series.order import O
        # 计算参数在 x=0 处的值
        arg0 = self.args[0].subs(x, 0)

        # 处理分支点 arg0 = 1 的情况
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            # 对 asin(1 - t**2) 进行对数重写并进行 t 的 n 级数展开
            ser = asin(S.One - t**2).rewrite(log).nseries(t, 0, 2*n)
            # 计算参数 arg1 = 1 - self.args[0] 的主导项
            arg1 = S.One - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            # 如果 g 不是整函数，则返回适当的阶数 O(1) 或者 pi/2 + O(sqrt(x))
            if not g.is_meromorphic(x, 0):   # 无法展开
                return O(1) if n == 0 else pi/2 + O(sqrt(x))
            # 计算 sqrt(1 + g) 的 n 级数展开
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            # 计算结果并展开
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        # 处理分支点 arg0 = -1 的情况
        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            # 对 asin(-1 + t**2) 进行对数重写并进行 t 的 n 级数展开
            ser = asin(S.NegativeOne + t**2).rewrite(log).nseries(t, 0, 2*n)
            # 计算参数 arg1 = 1 + self.args[0] 的主导项
            arg1 = S.One + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            # 如果 g 不是整函数，则返回适当的阶数 O(1) 或者 -pi/2 + O(sqrt(x))
            if not g.is_meromorphic(x, 0):   # 无法展开
                return O(1) if n == 0 else -pi/2 + O(sqrt(x))
            # 计算 sqrt(1 + g) 的 n 级数展开
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            # 计算结果并展开
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        # 计算函数在 x 处的 n 级数展开
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        # 处理分支点 arg0 = oo 的情况
        if arg0 is S.ComplexInfinity:
            return res
        # 处理位于分支切割线 (-oo, -1) U (1, oo) 上的点
        if (1 - arg0**2).is_negative:
            # 计算参数 self.args[0] 在 x 处的方向，并检查其虚部是否为负
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_negative:
                    return -pi - res
            elif im(ndir).is_positive:
                if arg0.is_positive:
                    return pi - res
            else:
                # 将函数重写为对数形式，然后计算其在 x 处的 n 级数展开
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        # 返回函数在 x 处的 n 级数展开结果
        return res
    # 将反余弦重写为反余弦函数
    def _eval_rewrite_as_acos(self, x, **kwargs):
        return pi/2 - acos(x)

    # 将反正切重写为反正切函数
    def _eval_rewrite_as_atan(self, x, **kwargs):
        return 2*atan(x/(1 + sqrt(1 - x**2)))

    # 将对数重写为对数函数
    def _eval_rewrite_as_log(self, x, **kwargs):
        return -S.ImaginaryUnit*log(S.ImaginaryUnit*x + sqrt(1 - x**2))

    # 将函数重写为对数函数
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    # 将反余切重写为反余切函数
    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return 2*acot((1 + sqrt(1 - arg**2))/arg)

    # 将反正割重写为反正割函数
    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return pi/2 - asec(1/arg)

    # 将反余割重写为反余割函数
    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return acsc(1/arg)

    # 检查函数的自变量是否是扩展实数
    def _eval_is_extended_real(self):
        x = self.args[0]
        return x.is_extended_real and (1 - abs(x)).is_nonnegative

    # 返回此函数的逆函数
    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sin
class acos(InverseTrigonometricFunction):
    r"""
    The inverse cosine function.

    Explanation
    ===========

    Returns the arc cosine of x (measured in radians).

    ``acos(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$ and for some instances when
    the result is a rational multiple of $\pi$ (see the eval class method).

    ``acos(zoo)`` evaluates to ``zoo``
    (see note in :class:`sympy.functions.elementary.trigonometric.asec`)

    A purely imaginary argument will be rewritten to asinh.

    Examples
    ========

    >>> from sympy import acos, oo
    >>> acos(1)
    0
    >>> acos(0)
    pi/2
    >>> acos(oo)
    oo*I

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCos

    """

    def fdiff(self, argindex=1):
        # 返回反函数对第一个参数的偏导数
        if argindex == 1:
            return -1/sqrt(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_rational(self):
        # 检查函数是否具有有理数结果
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational:
                return False
        else:
            return s.is_rational

    @classmethod
    def eval(cls, arg):
        # 对给定参数进行特定情况的求值
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity*S.ImaginaryUnit
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity*S.ImaginaryUnit
            elif arg.is_zero:
                return pi/2
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi

        if arg is S.ComplexInfinity:
            return S.ComplexInfinity

        if arg.is_number:
            asin_table = cls._asin_table()
            if arg in asin_table:
                return pi/2 - asin_table[arg]
            elif -arg in asin_table:
                return pi/2 + asin_table[-arg]

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            return pi/2 - asin(arg)

        if isinstance(arg, cos):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= 2*pi  # 将角度限制在 [0, 2*pi)
                if ang > pi:  # 将角度限制在 [0, pi]
                    ang = 2*pi - ang

                return ang

        if isinstance(arg, sin):  # acos(x) + asin(x) = pi/2
            ang = arg.args[0]
            if ang.is_comparable:
                return pi/2 - asin(arg)

    @staticmethod
    @cacheit
    # 计算泰勒级数的第n项
    def taylor_term(n, x, *previous_terms):
        # 如果 n 等于 0，返回 π/2
        if n == 0:
            return pi/2
        # 如果 n 小于 0 或者 n 是偶数，返回零
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            # 将 x 转换为符号表达式
            x = sympify(x)
            # 如果传入的前几项个数大于等于 2 并且 n 大于 2，则计算使用前两项的关系式
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return p*(n - 2)**2/(n*(n - 1))*x**2
            else:
                # 否则，计算特定的泰勒级数项
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return -R/F*x**n/n

    # 计算表达式的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):  # acos
        # 获取函数的参数
        arg = self.args[0]
        # 计算在 x=0 处的值，并化简
        x0 = arg.subs(x, 0).cancel()
        # 如果 x0 是 NaN，则返回函数参数的主导项
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        # 处理分支点
        if x0 == 1:
            return sqrt(2)*sqrt((S.One - arg).as_leading_term(x))
        if x0 in (-S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # 处理位于分支切口 (-oo, -1) U (1, oo) 上的点
        if (1 - x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_negative:
                    return 2*pi - self.func(x0)
            elif im(ndir).is_positive:
                if x0.is_positive:
                    return -self.func(x0)
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    # 检查表达式是否是扩展实数
    def _eval_is_extended_real(self):
        x = self.args[0]
        return x.is_extended_real and (1 - abs(x)).is_nonnegative

    # 检查表达式是否是非负数
    def _eval_is_nonnegative(self):
        return self._eval_is_extended_real()
    # 定义一个方法 _eval_nseries，用于计算 acos 函数的 n 阶级数展开
    def _eval_nseries(self, x, n, logx, cdir=0):  # acos
        # 导入 sympy.series.order 模块中的 O 对象
        from sympy.series.order import O
        # 计算 self.args[0] 在 x=0 处的值
        arg0 = self.args[0].subs(x, 0)
        
        # 处理分支点
        if arg0 is S.One:
            # 定义一个正的虚拟变量 t
            t = Dummy('t', positive=True)
            # 对 acos(1 - t**2) 应用对数重写并进行 t 的 n 阶级数展开
            ser = acos(S.One - t**2).rewrite(log).nseries(t, 0, 2*n)
            # 计算 S.One - self.args[0] 的主导项
            arg1 = S.One - self.args[0]
            # 计算 g = (arg1 - f) / f，其中 f 是 arg1 的主导项
            f = arg1.as_leading_term(x)
            g = (arg1 - f) / f
            # 如果 g 在 x=0 处不是亚纯的，无法展开
            if not g.is_meromorphic(x, 0):
                return O(1) if n == 0 else O(sqrt(x))
            # 对 sqrt(1 + g) 在 x 上进行 n 阶级数展开
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            # 计算最终结果并展开
            res = (res1.removeO() * sqrt(f)).expand()
            # 返回展开结果加上 x**n 阶无穷小量 O(x**n)
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        # 处理 arg0 = -1 的情况
        if arg0 is S.NegativeOne:
            # 定义一个正的虚拟变量 t
            t = Dummy('t', positive=True)
            # 对 acos(-1 + t**2) 应用对数重写并进行 t 的 n 阶级数展开
            ser = acos(S.NegativeOne + t**2).rewrite(log).nseries(t, 0, 2*n)
            # 计算 S.One + self.args[0] 的主导项
            arg1 = S.One + self.args[0]
            # 计算 g = (arg1 - f) / f，其中 f 是 arg1 的主导项
            f = arg1.as_leading_term(x)
            g = (arg1 - f) / f
            # 如果 g 在 x=0 处不是亚纯的，无法展开
            if not g.is_meromorphic(x, 0):
                return O(1) if n == 0 else pi + O(sqrt(x))
            # 对 sqrt(1 + g) 在 x 上进行 n 阶级数展开
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            # 计算最终结果并展开
            res = (res1.removeO() * sqrt(f)).expand()
            # 返回展开结果加上 x**n 阶无穷小量 O(x**n)
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        # 如果 arg0 是 S.ComplexInfinity，则直接返回基类 Function 的 _eval_nseries 计算结果
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        
        # 处理位于分支切割线上的点 (-oo, -1) U (1, oo)
        if (1 - arg0**2).is_negative:
            # 计算 self.args[0] 在 x 上的方向 ndir
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            # 如果 im(ndir) 是负数
            if im(ndir).is_negative:
                # 如果 arg0 是负数，则返回 2*pi - res
                if arg0.is_negative:
                    return 2*pi - res
            # 如果 im(ndir) 是正数
            elif im(ndir).is_positive:
                # 如果 arg0 是正数，则返回 -res
                if arg0.is_positive:
                    return -res
            else:
                # 其他情况，重新应用对数重写并进行 n 阶级数展开
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        
        # 默认情况下，返回基类 Function 的 _eval_nseries 计算结果
        return res

    # 将 acos 函数重写为对数形式的表达式
    def _eval_rewrite_as_log(self, x, **kwargs):
        return pi/2 + S.ImaginaryUnit * log(S.ImaginaryUnit * x + sqrt(1 - x**2))

    # 将 acos 函数重写为更容易处理的对数形式的表达式
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    # 将 acos 函数重写为 asin 函数的表达式
    def _eval_rewrite_as_asin(self, x, **kwargs):
        return pi/2 - asin(x)

    # 将 acos 函数重写为 atan 函数的表达式
    def _eval_rewrite_as_atan(self, x, **kwargs):
        return atan(sqrt(1 - x**2) / x) + (pi/2) * (1 - x * sqrt(1 / x**2))

    # 返回该函数的逆函数，这里是 cos 函数
    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return cos

    # 将 acos 函数重写为 acot 函数的表达式
    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return pi/2 - 2 * acot((1 + sqrt(1 - arg**2)) / arg)

    # 将 acos 函数重写为 asec 函数的表达式
    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return asec(1 / arg)

    # 将 acos 函数重写为 acsc 函数的表达式
    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return pi/2 - acsc(1 / arg)
    # 定义一个方法 `_eval_conjugate`，用于计算给定表达式的共轭
    def _eval_conjugate(self):
        # 从参数列表中获取第一个参数 `z`
        z = self.args[0]
        # 计算参数 `z` 的共轭，并调用类的 `func` 方法进行处理
        r = self.func(self.args[0].conjugate())
        # 如果参数 `z` 不是扩展实数，则直接返回计算结果 `r`
        if z.is_extended_real is False:
            return r
        # 如果参数 `z` 是扩展实数，并且 `z + 1` 非负且 `z - 1` 非正，则也返回计算结果 `r`
        elif z.is_extended_real and (z + 1).is_nonnegative and (z - 1).is_nonpositive:
            return r
# 定义一个类 atan，继承自 InverseTrigonometricFunction 类
class atan(InverseTrigonometricFunction):
    """
    The inverse tangent function.

    Returns the arc tangent of x (measured in radians).

    Explanation
    ===========

    ``atan(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$ and for some instances when the
    result is a rational multiple of $\pi$ (see the eval class method).

    Examples
    ========

    >>> from sympy import atan, oo
    >>> atan(0)
    0
    >>> atan(1)
    pi/4
    >>> atan(oo)
    pi/2

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcTan

    """

    # 定义 args 属性为一个元组，包含一个 Expr 类型的元素
    args: tTuple[Expr]

    # 定义一个类属性 _singularities，表示函数的奇点，包括虚数单位和负虚数单位
    _singularities = (S.ImaginaryUnit, -S.ImaginaryUnit)

    # 定义 fdiff 方法，用于计算对自变量的偏导数
    def fdiff(self, argindex=1):
        # 如果 argindex 为 1，则返回对第一个自变量的偏导数表达式
        if argindex == 1:
            return 1/(1 + self.args[0]**2)
        else:
            # 如果 argindex 不为 1，则引发 ArgumentIndexError 异常
            raise ArgumentIndexError(self, argindex)

    # 定义 _eval_is_rational 方法，用于判断函数值是否为有理数
    def _eval_is_rational(self):
        # 计算函数自身，判断其是否为 atan 函数
        s = self.func(*self.args)
        if s.func == self.func:
            # 如果 s 是 atan 函数且其自变量是有理数，则返回 False
            if s.args[0].is_rational:
                return False
        else:
            # 否则，返回 s 是否为有理数的判断结果
            return s.is_rational

    # 定义 _eval_is_positive 方法，用于判断函数值是否为正数
    def _eval_is_positive(self):
        # 返回函数自变量是否为扩展正数的判断结果
        return self.args[0].is_extended_positive

    # 定义 _eval_is_nonnegative 方法，用于判断函数值是否为非负数
    def _eval_is_nonnegative(self):
        # 返回函数自变量是否为扩展非负数的判断结果
        return self.args[0].is_extended_nonnegative

    # 定义 _eval_is_zero 方法，用于判断函数值是否为零
    def _eval_is_zero(self):
        # 返回函数自变量是否为零的判断结果
        return self.args[0].is_zero

    # 定义 _eval_is_real 方法，用于判断函数值是否为实数
    def _eval_is_real(self):
        # 返回函数自变量是否为扩展实数的判断结果
        return self.args[0].is_extended_real

    # 定义一个类方法，表示未完的代码块
    @classmethod
    # 定义一个类方法 `eval`，用于计算给定参数 `arg` 的反正切值
    def eval(cls, arg):
        # 检查参数是否为 SymPy 的数值类型
        if arg.is_Number:
            # 如果参数为 NaN，则返回 NaN
            if arg is S.NaN:
                return S.NaN
            # 如果参数为正无穷大，则返回 π/2
            elif arg is S.Infinity:
                return pi/2
            # 如果参数为负无穷大，则返回 -π/2
            elif arg is S.NegativeInfinity:
                return -pi/2
            # 如果参数为零，则返回 0
            elif arg.is_zero:
                return S.Zero
            # 如果参数为 1，则返回 π/4
            elif arg is S.One:
                return pi/4
            # 如果参数为 -1，则返回 -π/4
            elif arg is S.NegativeOne:
                return -pi/4

        # 如果参数为复无穷大
        if arg is S.ComplexInfinity:
            # 导入 AccumBounds 类并返回其范围为 [-π/2, π/2]
            from sympy.calculus.accumulationbounds import AccumBounds
            return AccumBounds(-pi/2, pi/2)

        # 如果可以提取参数的负号，返回其相反数的反正切值
        if arg.could_extract_minus_sign():
            return -cls(-arg)

        # 如果参数为数值类型
        if arg.is_number:
            # 获取反正切表并查找参数值，返回相应的结果
            atan_table = cls._atan_table()
            if arg in atan_table:
                return atan_table[arg]

        # 如果参数是虚数系数
        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            # 导入双曲正切反函数并返回结果
            from sympy.functions.elementary.hyperbolic import atanh
            return S.ImaginaryUnit * atanh(i_coeff)

        # 如果参数为零，则返回 0
        if arg.is_zero:
            return S.Zero

        # 如果参数是 tan 类型的实例
        if isinstance(arg, tan):
            # 获取角度参数
            ang = arg.args[0]
            # 如果角度可以比较
            if ang.is_comparable:
                ang %= pi  # 将角度限制在 [0, pi)
                if ang > pi/2:  # 将角度限制在 [-pi/2, pi/2]
                    ang -= pi

                return ang

        # 如果参数是 cot 类型的实例
        if isinstance(arg, cot):
            # 获取角度参数
            ang = arg.args[0]
            # 如果角度可以比较
            if ang.is_comparable:
                # 计算 pi/2 - acot(arg)，将角度限制在 [-pi/2, pi/2]
                ang = pi/2 - acot(arg)
                if ang > pi/2:
                    ang -= pi
                return ang

    @staticmethod
    @cacheit
    # 定义静态方法 `taylor_term`，用于计算泰勒级数的项
    def taylor_term(n, x, *previous_terms):
        # 如果 n 小于 0 或者 n 是偶数，则返回 0
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            # 将 x 转换为 SymPy 表达式
            x = sympify(x)
            # 返回泰勒级数的当前项
            return S.NegativeOne**((n - 1)//2) * x**n / n

    # 定义方法 `_eval_as_leading_term`，用于计算作为主导项的反正切函数
    def _eval_as_leading_term(self, x, logx=None, cdir=0):  # atan
        # 获取函数的参数
        arg = self.args[0]
        # 计算在 x=0 处的极限值，并化简
        x0 = arg.subs(x, 0).cancel()
        # 如果极限值为 NaN，则返回函数的原始参数
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        # 如果极限值为零，则返回参数的主导项
        if x0.is_zero:
            return arg.as_leading_term(x)
        # 处理分支点
        # 如果极限值为虚数单位或者复无穷大，则重写为对数并计算主导项
        if x0 in (-S.ImaginaryUnit, S.ImaginaryUnit, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # 处理位于分支切割线上的点 (-I*oo, -I) U (I, I*oo)
        if (1 + x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_negative:
                if im(x0).is_positive:
                    return self.func(x0) - pi
            elif re(ndir).is_positive:
                if im(x0).is_negative:
                    return self.func(x0) + pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)
    def _eval_nseries(self, x, n, logx, cdir=0):  # atan
        arg0 = self.args[0].subs(x, 0)
        # 计算函数在 x=0 处的表达式
        # 处理分支点
        if arg0 in (S.ImaginaryUnit, S.NegativeOne*S.ImaginaryUnit):
            # 如果 arg0 是虚数单位或负虚数单位，转换为 log 函数的级数求值
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        # 调用父类 Function 的 _eval_nseries 方法，求解级数展开
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        # 获取自变量的方向导数
        ndir = self.args[0].dir(x, cdir if cdir else 1)
        # 处理 arg0 为正无穷的情况
        if arg0 is S.ComplexInfinity:
            if re(ndir) > 0:
                return res - pi
            return res
        # 处理位于分支切割线上的点 (-I*oo, -I) U (I, I*oo)
        if (1 + arg0**2).is_negative:
            if re(ndir).is_negative:
                if im(arg0).is_positive:
                    return res - pi
            elif re(ndir).is_positive:
                if im(arg0).is_negative:
                    return res + pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        # 返回计算结果
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        # 使用对数函数重写当前函数
        return S.ImaginaryUnit/2*(log(S.One - S.ImaginaryUnit*x)
            - log(S.One + S.ImaginaryUnit*x))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_aseries(self, n, args0, x, logx):
        # 当参数为无穷大或负无穷大时，返回对应的级数展开结果
        if args0[0] in [S.Infinity, S.NegativeInfinity]:
            return (pi/2 - atan(1/self.args[0]))._eval_nseries(x, n, logx)
        else:
            return super()._eval_aseries(n, args0, x, logx)

    def inverse(self, argindex=1):
        """
        返回该函数的反函数。
        """
        return tan

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        # 使用反正弦函数重写当前函数
        return sqrt(arg**2)/arg*(pi/2 - asin(1/sqrt(1 + arg**2)))

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        # 使用反余弦函数重写当前函数
        return sqrt(arg**2)/arg*acos(1/sqrt(1 + arg**2))

    def _eval_rewrite_as_acot(self, arg, **kwargs):
        # 使用反余切函数重写当前函数
        return acot(1/arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        # 使用反正割函数重写当前函数
        return sqrt(arg**2)/arg*asec(sqrt(1 + arg**2))

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        # 使用反余割函数重写当前函数
        return sqrt(arg**2)/arg*(pi/2 - acsc(sqrt(1 + arg**2)))
class acot(InverseTrigonometricFunction):
    r"""
    The inverse cotangent function.

    Returns the arc cotangent of x (measured in radians).

    Explanation
    ===========

    ``acot(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, \tilde{\infty}, 0, 1, -1\}$
    and for some instances when the result is a rational multiple of $\pi$
    (see the eval class method).

    A purely imaginary argument will lead to an ``acoth`` expression.

    ``acot(x)`` has a branch cut along $(-i, i)$, hence it is discontinuous
    at 0. Its range for real $x$ is $(-\frac{\pi}{2}, \frac{\pi}{2}]$.

    Examples
    ========

    >>> from sympy import acot, sqrt
    >>> acot(0)
    pi/2
    >>> acot(1)
    pi/4
    >>> acot(sqrt(3) - 2)
    -5*pi/12

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, atan2

    References
    ==========

    .. [1] https://dlmf.nist.gov/4.23
    .. [2] https://functions.wolfram.com/ElementaryFunctions/ArcCot

    """
    _singularities = (S.ImaginaryUnit, -S.ImaginaryUnit)

    # 计算函数关于其参数的导数
    def fdiff(self, argindex=1):
        # 对于唯一的参数进行导数计算
        if argindex == 1:
            return -1/(1 + self.args[0]**2)
        else:
            # 如果参数索引不是1，则引发参数索引错误
            raise ArgumentIndexError(self, argindex)

    # 检查函数是否是有理数
    def _eval_is_rational(self):
        # 调用函数的实例化对象，并检查其是否是有理数
        s = self.func(*self.args)
        if s.func == self.func:
            # 如果是函数的实例，则检查其第一个参数是否是有理数
            if s.args[0].is_rational:
                return False
        else:
            # 否则直接返回函数的有理数属性
            return s.is_rational

    # 检查函数是否为正数
    def _eval_is_positive(self):
        # 检查函数的参数是否为非负数
        return self.args[0].is_nonnegative

    # 检查函数是否为负数
    def _eval_is_negative(self):
        # 检查函数的参数是否为负数
        return self.args[0].is_negative

    # 检查函数是否为扩展实数
    def _eval_is_extended_real(self):
        # 检查函数的参数是否为扩展实数
        return self.args[0].is_extended_real

    @classmethod
    # 定义类方法 `eval`，用于处理给定参数 `arg` 的求值
    def eval(cls, arg):
        # 检查参数是否为数值
        if arg.is_Number:
            # 如果参数是 NaN，返回 NaN
            if arg is S.NaN:
                return S.NaN
            # 如果参数是正无穷或负无穷，返回零
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            # 如果参数为零，返回 pi/2
            elif arg.is_zero:
                return pi/ 2
            # 如果参数为 1，返回 pi/4
            elif arg is S.One:
                return pi/4
            # 如果参数为 -1，返回 -pi/4
            elif arg is S.NegativeOne:
                return -pi/4

        # 如果参数为复无穷，返回零
        if arg is S.ComplexInfinity:
            return S.Zero

        # 如果参数可以提取负号，返回其相反数的求值
        if arg.could_extract_minus_sign():
            return -cls(-arg)

        # 如果参数为数值，处理其反正切
        if arg.is_number:
            # 获取反正切表
            atan_table = cls._atan_table()
            # 如果参数在反正切表中，计算角度
            if arg in atan_table:
                ang = pi/2 - atan_table[arg]
                # 角度限制在 (-pi/2, pi/2] 范围内
                if ang > pi/2:
                    ang -= pi
                return ang

        # 检查是否可以将参数视为虚数系数
        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            # 计算反双曲余切乘以虚数单位
            from sympy.functions.elementary.hyperbolic import acoth
            return -S.ImaginaryUnit*acoth(i_coeff)

        # 如果参数为零，返回 pi/2
        if arg.is_zero:
            return pi*S.Half

        # 如果参数是 cot 函数的实例，处理其角度
        if isinstance(arg, cot):
            ang = arg.args[0]
            # 如果角度可以比较，限制在 [0, pi) 范围内
            if ang.is_comparable:
                ang %= pi
                # 角度限制在 (-pi/2, pi/2] 范围内
                if ang > pi/2:
                    ang -= pi
                return ang

        # 如果参数是 tan 函数的实例，处理其角度
        if isinstance(arg, tan):
            ang = arg.args[0]
            # 如果角度可以比较，计算反正切和反余切之和
            if ang.is_comparable:
                ang = pi/2 - atan(arg)
                # 角度限制在 (-pi/2, pi/2] 范围内
                if ang > pi/2:
                    ang -= pi
                return ang

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # 计算泰勒级数的项
        if n == 0:
            return pi/2  # 返回 pi/2，这里可能需要修复
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            # 返回泰勒级数的具体项
            return S.NegativeOne**((n + 1)//2)*x**n/n

    # 定义内部方法 `_eval_as_leading_term`，处理作为主导项的求值
    def _eval_as_leading_term(self, x, logx=None, cdir=0):  # acot
        # 获取参数
        arg = self.args[0]
        # 计算在 x=0 处的极限
        x0 = arg.subs(x, 0).cancel()
        # 如果极限为 NaN，返回参数的主导项
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        # 如果极限为复无穷，返回其倒数的主导项
        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        # 处理分支点
        if x0 in (-S.ImaginaryUnit, S.ImaginaryUnit, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # 处理位于分支切割 [-I, I] 上的点
        if x0.is_imaginary and (1 + x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_positive:
                    return self.func(x0) + pi
            elif re(ndir).is_negative:
                if im(x0).is_negative:
                    return self.func(x0) - pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)
    def _eval_nseries(self, x, n, logx, cdir=0):  # acot
        arg0 = self.args[0].subs(x, 0)
        # 计算自变量为 x 的函数中，参数列表第一个参数在 x=0 处的表达式值

        # 处理分支点
        if arg0 in (S.ImaginaryUnit, S.NegativeOne*S.ImaginaryUnit):
            # 如果参数为虚数单位或负虚数单位，则将表达式重写为对数形式，并重新进行 n 级数展开
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        # 计算函数在 x 点处的 n 级数展开结果
        res = Function._eval_nseries(self, x, n=n, logx=logx)

        # 如果参数为复无穷，则直接返回展开结果
        if arg0 is S.ComplexInfinity:
            return res

        # 计算自变量 x 方向的导数，如果未指定方向则默认为 1
        ndir = self.args[0].dir(x, cdir if cdir else 1)

        # 如果参数为零
        if arg0.is_zero:
            # 根据导数的实部判断方向，如果实部小于零，则结果减去 π，否则返回结果
            if re(ndir) < 0:
                return res - pi
            return res

        # 处理位于分支割线 [-I, I] 上的点
        if arg0.is_imaginary and (1 + arg0**2).is_positive:
            # 如果实部导数为正
            if re(ndir).is_positive:
                # 如果参数的虚部为正，则结果加上 π
                if im(arg0).is_positive:
                    return res + pi
            # 如果实部导数为负
            elif re(ndir).is_negative:
                # 如果参数的虚部为负，则结果减去 π
                if im(arg0).is_negative:
                    return res - pi
            else:
                # 其他情况将表达式重写为对数形式，并重新进行 n 级数展开
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        # 返回 n 级数展开的结果
        return res

    def _eval_aseries(self, n, args0, x, logx):
        # 如果参数列表中第一个参数为无穷大或负无穷大，则返回反正切函数的 n 级数展开结果
        if args0[0] in [S.Infinity, S.NegativeInfinity]:
            return atan(1/self.args[0])._eval_nseries(x, n, logx)
        else:
            # 否则调用父类的方法进行 n 级数展开
            return super()._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_log(self, x, **kwargs):
        # 返回一个表达式，其中包含对数函数的重写形式
        return S.ImaginaryUnit/2*(log(1 - S.ImaginaryUnit/x)
            - log(1 + S.ImaginaryUnit/x))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        # 返回此函数的反函数 cot
        return cot

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        # 返回一个表达式，其中包含 arcsin 函数的重写形式
        return (arg*sqrt(1/arg**2)*
                (pi/2 - asin(sqrt(-arg**2)/sqrt(-arg**2 - 1))))

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        # 返回一个表达式，其中包含 arccos 函数的重写形式
        return arg*sqrt(1/arg**2)*acos(sqrt(-arg**2)/sqrt(-arg**2 - 1))

    def _eval_rewrite_as_atan(self, arg, **kwargs):
        # 返回一个表达式，其中包含 arctan 函数的重写形式
        return atan(1/arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        # 返回一个表达式，其中包含 asec 函数的重写形式
        return arg*sqrt(1/arg**2)*asec(sqrt((1 + arg**2)/arg**2))

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        # 返回一个表达式，其中包含 acsc 函数的重写形式
        return arg*sqrt(1/arg**2)*(pi/2 - acsc(sqrt((1 + arg**2)/arg**2)))
class asec(InverseTrigonometricFunction):
    r"""
    The inverse secant function.

    Returns the arc secant of x (measured in radians).

    Explanation
    ===========

    ``asec(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$ and for some instances when the
    result is a rational multiple of $\pi$ (see the eval class method).

    ``asec(x)`` has branch cut in the interval $[-1, 1]$. For complex arguments,
    it can be defined [4]_ as

    .. math::
        \operatorname{sec^{-1}}(z) = -i\frac{\log\left(\sqrt{1 - z^2} + 1\right)}{z}

    At ``x = 0``, for positive branch cut, the limit evaluates to ``zoo``. For
    negative branch cut, the limit

    .. math::
        \lim_{z \to 0}-i\frac{\log\left(-\sqrt{1 - z^2} + 1\right)}{z}

    simplifies to :math:`-i\log\left(z/2 + O\left(z^3\right)\right)` which
    ultimately evaluates to ``zoo``.

    As ``acos(x) = asec(1/x)``, a similar argument can be given for
    ``acos(x)``.

    Examples
    ========

    >>> from sympy import asec, oo
    >>> asec(1)
    0
    >>> asec(-1)
    pi
    >>> asec(0)
    zoo
    >>> asec(-oo)
    pi/2

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSec
    .. [4] https://reference.wolfram.com/language/ref/ArcSec.html

    """

    @classmethod
    # eval 方法用于计算 asec(x) 的值
    def eval(cls, arg):
        # 如果参数 arg 是零，则返回复无穷
        if arg.is_zero:
            return S.ComplexInfinity
        # 如果参数 arg 是一个数值
        if arg.is_Number:
            # 如果 arg 是 NaN，则返回 NaN
            if arg is S.NaN:
                return S.NaN
            # 如果 arg 是 1，则返回 0
            elif arg is S.One:
                return S.Zero
            # 如果 arg 是 -1，则返回 π
            elif arg is S.NegativeOne:
                return pi
        # 如果 arg 是无穷大、负无穷大或复无穷大，则返回 π/2
        if arg in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
            return pi/2

        # 如果 arg 是一个数值
        if arg.is_number:
            # 获取 acsc 表
            acsc_table = cls._acsc_table()
            # 如果 arg 在 acsc 表中
            if arg in acsc_table:
                return pi/2 - acsc_table[arg]
            # 如果 -arg 在 acsc 表中
            elif -arg in acsc_table:
                return pi/2 + acsc_table[-arg]

        # 如果 arg 是无穷的
        if arg.is_infinite:
            return pi/2

        # 如果 arg 是 sec 的实例
        if isinstance(arg, sec):
            # 获取角度参数
            ang = arg.args[0]
            # 如果角度可以比较大小
            if ang.is_comparable:
                ang %= 2*pi  # 将角度限制在 [0, 2π) 范围内
                if ang > pi:  # 将角度限制在 [0, π] 范围内
                    ang = 2*pi - ang

                return ang

        # 如果 arg 是 csc 的实例，则返回 pi/2 减去 acsc(arg) 的值
        if isinstance(arg, csc):  # asec(x) + acsc(x) = pi/2
            ang = arg.args[0]
            if ang.is_comparable:
                return pi/2 - acsc(arg)

    # fdiff 方法用于计算导数
    def fdiff(self, argindex=1):
        # 如果 argindex 等于 1，则返回对应导数表达式
        if argindex == 1:
            return 1/(self.args[0]**2*sqrt(1 - 1/self.args[0]**2))
        else:
            raise ArgumentIndexError(self, argindex)

    # inverse 方法返回该函数的逆函数
    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sec

    @staticmethod
    # 静态方法或静态属性的定义位置
    # 使用装饰器 @cacheit 来缓存函数结果，提高性能
    @cacheit
    # 定义 Taylor 级数的单个项的计算函数，接收 n、x 和之前的若干项作为参数
    def taylor_term(n, x, *previous_terms):
        # 当 n 等于 0 时，返回复数单位乘以 log(2/x) 的值作为结果
        if n == 0:
            return S.ImaginaryUnit*log(2 / x)
        # 当 n 小于 0 或者 n 是奇数时，返回零作为结果
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            # 将 x 转换为符号表达式
            x = sympify(x)
            # 当 previous_terms 中的项数大于 2 且 n 大于 2 时，根据前两个项计算当前项的值
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                # 计算 k 值为 n 除以 2 的整数部分
                k = n // 2
                # 计算 R 为升幂阶乘 (1/2)_k * n
                R = RisingFactorial(S.Half, k) *  n
                # 计算 F 为 k! * (n//2)! * (n//2)!
                F = factorial(k) * n // 2 * n // 2
                # 返回结果，其中包括复数单位、R 和 F 的运算结果以及 x 的 n 次方，除以 4
                return -S.ImaginaryUnit * R / F * x**n / 4

    # 定义函数 _eval_as_leading_term，用于计算对象的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):  # asec
        # 获取对象的第一个参数
        arg = self.args[0]
        # 计算在 x 等于 0 时 arg 的值，并化简
        x0 = arg.subs(x, 0).cancel()
        # 如果 x0 是 NaN，则返回对象关于 arg 的主导项
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        # 处理分支点
        if x0 == 1:
            return sqrt(2)*sqrt((arg - S.One).as_leading_term(x))
        if x0 in (-S.One, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # 处理位于分支切口 (-1, 1) 上的点
        if x0.is_real and (1 - x0**2).is_positive:
            # 计算 arg 在 x 方向上的方向导数 ndir
            ndir = arg.dir(x, cdir if cdir else 1)
            # 如果 ndir 的虚部为负，根据 x0 的正负返回结果
            if im(ndir).is_negative:
                if x0.is_positive:
                    return -self.func(x0)
            # 如果 ndir 的虚部为正，根据 x0 的正负返回结果
            elif im(ndir).is_positive:
                if x0.is_negative:
                    return 2*pi - self.func(x0)
            else:
                # 对于其他情况，返回对象关于 log 的重写后的主导项
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # 返回对象关于 x0 的主导项
        return self.func(x0)
    def _eval_nseries(self, x, n, logx, cdir=0):  # asec
        # 导入必要的模块以进行级数展开
        from sympy.series.order import O
        # 计算表达式的零阶项
        arg0 = self.args[0].subs(x, 0)
        
        # 处理分支点
        if arg0 is S.One:
            # 定义一个正的虚拟变量 t
            t = Dummy('t', positive=True)
            # 对 asec(S.One + t**2) 应用对数重写并进行 t 的级数展开
            ser = asec(S.One + t**2).rewrite(log).nseries(t, 0, 2*n)
            # 计算 S.NegativeOne + self.args[0] 的主导项
            arg1 = S.NegativeOne + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f) / f
            # 对 sqrt(S.One + g) 应用级数展开
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO() * sqrt(f)).expand()
            # 返回级数展开结果，考虑高阶项 O(x**n, x)
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            # 定义一个正的虚拟变量 t
            t = Dummy('t', positive=True)
            # 对 asec(S.NegativeOne - t**2) 应用对数重写并进行 t 的级数展开
            ser = asec(S.NegativeOne - t**2).rewrite(log).nseries(t, 0, 2*n)
            # 计算 S.NegativeOne - self.args[0] 的主导项
            arg1 = S.NegativeOne - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f) / f
            # 对 sqrt(S.One + g) 应用级数展开
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO() * sqrt(f)).expand()
            # 返回级数展开结果，考虑高阶项 O(x**n, x)
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        # 对于其他情况，调用父类的级数展开方法
        res = Function._eval_nseries(self, x, n=n, logx=logx)

        # 处理无穷远点的情况
        if arg0 is S.ComplexInfinity:
            return res

        # 处理位于分支切割线 (-1, 1) 上的点
        if arg0.is_real and (1 - arg0**2).is_positive:
            # 计算 self.args[0] 在 x 方向上的导数
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            # 如果导数的虚部为负
            if im(ndir).is_negative:
                # 如果 arg0 是正数，则返回 -res
                if arg0.is_positive:
                    return -res
            # 如果导数的虚部为正
            elif im(ndir).is_positive:
                # 如果 arg0 是负数，则返回 2*pi - res
                if arg0.is_negative:
                    return 2*pi - res
            else:
                # 其他情况下，重新用对数重写后进行级数展开
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        
        # 默认情况，返回父类的级数展开结果
        return res

    def _eval_is_extended_real(self):
        # 获取自变量 x
        x = self.args[0]
        # 如果 x 不是扩展实数，则返回 False
        if x.is_extended_real is False:
            return False
        # 否则，返回 ((x - 1).is_nonnegative 或者 (-x - 1).is_nonnegative) 的模糊逻辑或
        return fuzzy_or(((x - 1).is_nonnegative, (-x - 1).is_nonnegative))

    def _eval_rewrite_as_log(self, arg, **kwargs):
        # 返回 asec(arg) 重写为对数的形式
        return pi/2 + S.ImaginaryUnit*log(S.ImaginaryUnit/arg + sqrt(1 - 1/arg**2))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        # 返回 asec(arg) 重写为 arcsin 的形式
        return pi/2 - asin(1/arg)

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        # 返回 asec(arg) 重写为 arccos 的形式
        return acos(1/arg)

    def _eval_rewrite_as_atan(self, x, **kwargs):
        # 返回 asec(arg) 重写为 arctan 的形式
        sx2x = sqrt(x**2)/x
        return pi/2*(1 - sx2x) + sx2x*atan(sqrt(x**2 - 1))

    def _eval_rewrite_as_acot(self, x, **kwargs):
        # 返回 asec(arg) 重写为 arccot 的形式
        sx2x = sqrt(x**2)/x
        return pi/2*(1 - sx2x) + sx2x*acot(1/sqrt(x**2 - 1))

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        # 返回 asec(arg) 重写为 arccsc 的形式
        return pi/2 - acsc(arg)
class acsc(InverseTrigonometricFunction):
    r"""
    The inverse cosecant function.

    Returns the arc cosecant of x (measured in radians).

    Explanation
    ===========

    ``acsc(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$` and for some instances when the
    result is a rational multiple of $\pi$ (see the ``eval`` class method).

    Examples
    ========

    >>> from sympy import acsc, oo
    >>> acsc(1)
    pi/2
    >>> acsc(-1)
    -pi/2
    >>> acsc(oo)
    0
    >>> acsc(-oo) == acsc(oo)
    True
    >>> acsc(0)
    zoo

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCsc

    """

    @classmethod
    def eval(cls, arg):
        # 如果参数是零，则返回复无穷大
        if arg.is_zero:
            return S.ComplexInfinity
        # 如果参数是数字
        if arg.is_Number:
            # 分别处理特殊的数字情况
            if arg is S.NaN:
                return S.NaN
            elif arg is S.One:
                return pi/2
            elif arg is S.NegativeOne:
                return -pi/2
        # 如果参数是正无穷、负无穷或复无穷，则返回零
        if arg in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
            return S.Zero

        # 如果参数能够提取负号，则返回负参数的相反数的结果
        if arg.could_extract_minus_sign():
            return -cls(-arg)

        # 如果参数是无限大，则返回零
        if arg.is_infinite:
            return S.Zero

        # 如果参数是一个数字
        if arg.is_number:
            # 获取acsc表并查找参数对应的值
            acsc_table = cls._acsc_table()
            if arg in acsc_table:
                return acsc_table[arg]

        # 如果参数是csc函数的实例
        if isinstance(arg, csc):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= 2*pi  # 将角度限制在[0, 2*pi)范围内
                if ang > pi:  # 将角度限制在(-pi, pi]范围内
                    ang = pi - ang

                # 将角度限制在[-pi/2, pi/2]范围内
                if ang > pi/2:
                    ang = pi - ang
                if ang < -pi/2:
                    ang = -pi - ang

                return ang

        # 如果参数是sec函数的实例，则使用asec(x) + acsc(x) = pi/2关系返回结果
        if isinstance(arg, sec):
            ang = arg.args[0]
            if ang.is_comparable:
                return pi/2 - asec(arg)

    def fdiff(self, argindex=1):
        # 当argindex为1时，返回关于参数的导数
        if argindex == 1:
            return -1/(self.args[0]**2*sqrt(1 - 1/self.args[0]**2))
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return csc

    @staticmethod
    @cacheit
    # 计算泰勒级数的第 n 项，基于给定的 x 值和之前的项
    def taylor_term(n, x, *previous_terms):
        # 如果 n 为 0，计算特定的初项
        if n == 0:
            return pi/2 - S.ImaginaryUnit*log(2) + S.ImaginaryUnit*log(x)
        # 如果 n 小于 0 或者 n 是奇数，则返回零
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            # 将 x 转换为符号表示
            x = sympify(x)
            # 如果之前的项数量大于 2 并且 n 大于 2，则使用递推关系计算
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                # 计算 k，并且计算 R 和 F
                k = n // 2
                R = RisingFactorial(S.Half, k) *  n
                F = factorial(k) * n // 2 * n // 2
                return S.ImaginaryUnit * R / F * x**n / 4

    # 评估函数作为主导项，处理特定的 x 值和对数项，以及指定的方向
    def _eval_as_leading_term(self, x, logx=None, cdir=0):  # acsc
        # 获取函数的参数
        arg = self.args[0]
        # 求解 x=0 时的极限并化简
        x0 = arg.subs(x, 0).cancel()
        # 如果极限为非数值，返回函数参数的主导项
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        # 处理分支点
        if x0 in (-S.One, S.One, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # 处理无穷远点
        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        # 处理位于分支切割线 (-1, 1) 上的点
        if x0.is_real and (1 - x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            # 如果虚部为负，根据 x0 的符号返回对应的值
            if im(ndir).is_negative:
                if x0.is_positive:
                    return pi - self.func(x0)
            # 如果虚部为正，根据 x0 的符号返回对应的值
            elif im(ndir).is_positive:
                if x0.is_negative:
                    return -pi - self.func(x0)
            # 否则，返回函数参数的主导项
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # 返回函数参数的主导项
        return self.func(x0)
    # 定义一个方法用于计算 nseries，即多项式级数展开，包括处理分支点
    def _eval_nseries(self, x, n, logx, cdir=0):  # acsc
        # 导入必要的模块和类
        from sympy.series.order import O
        # 计算参数表达式关于 x 在 x=0 处的代数式
        arg0 = self.args[0].subs(x, 0)
        
        # 处理分支点
        if arg0 is S.One:
            # 声明一个正的虚拟变量 t
            t = Dummy('t', positive=True)
            # 计算 acsc(1 + t**2) 关于 t 在 t=0 处的 nseries 展开，重写为对数形式
            ser = acsc(S.One + t**2).rewrite(log).nseries(t, 0, 2*n)
            # 计算参数表达式减去领先项的结果
            arg1 = S.NegativeOne + self.args[0]
            # 计算 g 的表达式
            g = (arg1 - f)/ f
            # 计算 sqrt(1 + g) 关于 x 的 nseries 展开，带对数因子
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            # 计算 res 的结果
            res = (res1.removeO()*sqrt(f)).expand()
            # 返回 nseries 展开的结果
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            # 声明一个正的虚拟变量 t
            t = Dummy('t', positive=True)
            # 计算 acsc(-1 - t**2) 关于 t 在 t=0 处的 nseries 展开，重写为对数形式
            ser = acsc(S.NegativeOne - t**2).rewrite(log).nseries(t, 0, 2*n)
            # 计算参数表达式减去领先项的结果
            arg1 = S.NegativeOne - self.args[0]
            # 计算 g 的表达式
            g = (arg1 - f)/ f
            # 计算 sqrt(1 + g) 关于 x 的 nseries 展开，带对数因子
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            # 计算 res 的结果
            res = (res1.removeO()*sqrt(f)).expand()
            # 返回 nseries 展开的结果
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        # 对于非特殊情况，调用父类方法进行 nseries 展开计算
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        
        # 处理无穷远点
        if arg0 is S.ComplexInfinity:
            return res
        
        # 处理落在分支切割线上的点 (-1, 1)
        if arg0.is_real and (1 - arg0**2).is_positive:
            # 计算参数表达式相对于 x 的方向导数
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            # 判断虚部是否为负
            if im(ndir).is_negative:
                # 如果参数表达式是正的，返回 pi 减去结果
                if arg0.is_positive:
                    return pi - res
            elif im(ndir).is_positive:
                # 如果参数表达式是负的，返回负的 pi 减去结果
                if arg0.is_negative:
                    return -pi - res
            else:
                # 其它情况，重写为对数形式，并进行 nseries 展开计算
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        
        # 返回 nseries 展开的结果
        return res

    # 将对象重写为对数的形式
    def _eval_rewrite_as_log(self, arg, **kwargs):
        return -S.ImaginaryUnit*log(S.ImaginaryUnit/arg + sqrt(1 - 1/arg**2))

    # 将对象重写为可处理的形式，这里与对数形式相同
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    # 将对象重写为 arcsin 的形式
    def _eval_rewrite_as_asin(self, arg, **kwargs):
        return asin(1/arg)

    # 将对象重写为 arccos 的形式
    def _eval_rewrite_as_acos(self, arg, **kwargs):
        return pi/2 - acos(1/arg)

    # 将对象重写为 arctan 的形式
    def _eval_rewrite_as_atan(self, x, **kwargs):
        return sqrt(x**2)/x*(pi/2 - atan(sqrt(x**2 - 1)))

    # 将对象重写为 arccot 的形式
    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return sqrt(arg**2)/arg*(pi/2 - acot(1/sqrt(arg**2 - 1)))

    # 将对象重写为 arcsec 的形式
    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return pi/2 - asec(arg)
# atan2 继承自 InverseTrigonometricFunction 类
class atan2(InverseTrigonometricFunction):
    """
    The function ``atan2(y, x)`` computes `\operatorname{atan}(y/x)` taking
    two arguments `y` and `x`.  Signs of both `y` and `x` are considered to
    determine the appropriate quadrant of `\operatorname{atan}(y/x)`.
    The range is `(-\pi, \pi]`. The complete definition reads as follows:

    .. math::

        \operatorname{atan2}(y, x) =
        \begin{cases}
          \arctan\left(\frac y x\right) & \qquad x > 0 \\
          \arctan\left(\frac y x\right) + \pi& \qquad y \ge 0, x < 0 \\
          \arctan\left(\frac y x\right) - \pi& \qquad y < 0, x < 0 \\
          +\frac{\pi}{2} & \qquad y > 0, x = 0 \\
          -\frac{\pi}{2} & \qquad y < 0, x = 0 \\
          \text{undefined} & \qquad y = 0, x = 0
        \end{cases}

    Attention: Note the role reversal of both arguments. The `y`-coordinate
    is the first argument and the `x`-coordinate the second.

    If either `x` or `y` is complex:

    .. math::

        \operatorname{atan2}(y, x) =
            -i\log\left(\frac{x + iy}{\sqrt{x^2 + y^2}}\right)

    Examples
    ========

    Going counter-clock wise around the origin we find the
    following angles:

    >>> from sympy import atan2
    >>> atan2(0, 1)
    0
    >>> atan2(1, 1)
    pi/4
    >>> atan2(1, 0)
    pi/2
    >>> atan2(1, -1)
    3*pi/4
    >>> atan2(0, -1)
    pi
    >>> atan2(-1, -1)
    -3*pi/4
    >>> atan2(-1, 0)
    -pi/2
    >>> atan2(-1, 1)
    -pi/4

    which are all correct. Compare this to the results of the ordinary
    `\operatorname{atan}` function for the point `(x, y) = (-1, 1)`

    >>> from sympy import atan, S
    >>> atan(S(1)/-1)
    -pi/4
    >>> atan2(1, -1)
    3*pi/4

    where only the `\operatorname{atan2}` function reurns what we expect.
    We can differentiate the function with respect to both arguments:

    >>> from sympy import diff
    >>> from sympy.abc import x, y
    >>> diff(atan2(y, x), x)
    -y/(x**2 + y**2)

    >>> diff(atan2(y, x), y)
    x/(x**2 + y**2)

    We can express the `\operatorname{atan2}` function in terms of
    complex logarithms:

    >>> from sympy import log
    >>> atan2(y, x).rewrite(log)
    -I*log((x + I*y)/sqrt(x**2 + y**2))

    and in terms of `\operatorname(atan)`:

    >>> from sympy import atan
    >>> atan2(y, x).rewrite(atan)
    Piecewise((2*atan(y/(x + sqrt(x**2 + y**2))), Ne(y, 0)), (pi, re(x) < 0), (0, Ne(x, 0)), (nan, True))

    but note that this form is undefined on the negative real axis.

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://en.wikipedia.org/wiki/Atan2
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcTan2

    """

    @classmethod
    def eval(cls, y, x):
        # 导入符号计算库中的 Heaviside 函数
        from sympy.functions.special.delta_functions import Heaviside
        # 处理 x 为负无穷大的情况
        if x is S.NegativeInfinity:
            if y.is_zero:
                # 特殊情况：当 y = 0 时，Heaviside(0) = 1/2
                return pi
            # 对于其他情况返回计算结果
            return 2*pi*(Heaviside(re(y))) - pi
        # 处理 x 为正无穷大的情况
        elif x is S.Infinity:
            return S.Zero
        # 处理 x 和 y 均为虚数且为数值的情况
        elif x.is_imaginary and y.is_imaginary and x.is_number and y.is_number:
            x = im(x)
            y = im(y)

        # 处理 x 和 y 均为扩展实数的情况
        if x.is_extended_real and y.is_extended_real:
            if x.is_positive:
                return atan(y/x)
            elif x.is_negative:
                if y.is_negative:
                    return atan(y/x) - pi
                elif y.is_nonnegative:
                    return atan(y/x) + pi
            elif x.is_zero:
                if y.is_positive:
                    return pi/2
                elif y.is_negative:
                    return -pi/2
                elif y.is_zero:
                    return S.NaN
        # 处理 y 为零的情况
        if y.is_zero:
            if x.is_extended_nonzero:
                return pi*(S.One - Heaviside(x))
            if x.is_number:
                return Piecewise((pi, re(x) < 0),
                                 (0, Ne(x, 0)),
                                 (S.NaN, True))
        # 处理 x 和 y 均为数值的情况
        if x.is_number and y.is_number:
            return -S.ImaginaryUnit*log(
                (x + S.ImaginaryUnit*y)/sqrt(x**2 + y**2))

    def _eval_rewrite_as_log(self, y, x, **kwargs):
        # 重写为对数形式
        return -S.ImaginaryUnit*log((x + S.ImaginaryUnit*y)/sqrt(x**2 + y**2))

    def _eval_rewrite_as_atan(self, y, x, **kwargs):
        # 重写为反正切形式
        return Piecewise((2*atan(y/(x + sqrt(x**2 + y**2))), Ne(y, 0)),
                         (pi, re(x) < 0),
                         (0, Ne(x, 0)),
                         (S.NaN, True))

    def _eval_rewrite_as_arg(self, y, x, **kwargs):
        # 重写为参数形式
        if x.is_extended_real and y.is_extended_real:
            return arg_f(x + y*S.ImaginaryUnit)
        n = x + S.ImaginaryUnit*y
        d = x**2 + y**2
        return arg_f(n/sqrt(d)) - S.ImaginaryUnit*log(abs(n)/sqrt(abs(d)))

    def _eval_is_extended_real(self):
        # 判断是否为扩展实数
        return self.args[0].is_extended_real and self.args[1].is_extended_real

    def _eval_conjugate(self):
        # 计算共轭复数
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def fdiff(self, argindex):
        # 计算偏导数
        y, x = self.args
        if argindex == 1:
            # 对 y 求偏导数
            return x/(x**2 + y**2)
        elif argindex == 2:
            # 对 x 求偏导数
            return -y/(x**2 + y**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        # 数值求解
        y, x = self.args
        if x.is_extended_real and y.is_extended_real:
            return super()._eval_evalf(prec)
```