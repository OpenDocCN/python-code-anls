# `D:\src\scipysrc\sympy\sympy\functions\elementary\hyperbolic.py`

```
# 从 sympy 核心模块导入指定对象
from sympy.core import S, sympify, cacheit
# 从 sympy 核心模块中导入加法操作对象 Add
from sympy.core.add import Add
# 从 sympy 核心模块中导入函数对象和相关异常处理
from sympy.core.function import Function, ArgumentIndexError
# 从 sympy 核心模块中导入逻辑运算函数 fuzzy_or, fuzzy_and, fuzzy_not, FuzzyBool
from sympy.core.logic import fuzzy_or, fuzzy_and, fuzzy_not, FuzzyBool
# 从 sympy 核心模块中导入常见数值对象 I, pi, Rational
from sympy.core.numbers import I, pi, Rational
# 从 sympy 核心模块中导入虚拟符号对象 Dummy
from sympy.core.symbol import Dummy
# 从 sympy 组合数学模块中导入阶乘和二项式系数计算函数
from sympy.functions.combinatorial.factorials import (binomial, factorial,
                                                      RisingFactorial)
# 从 sympy 组合数学模块中导入伯努利数、欧拉数和组合数计算函数
from sympy.functions.combinatorial.numbers import bernoulli, euler, nC
# 从 sympy 复数函数模块中导入绝对值、虚部和实部函数
from sympy.functions.elementary.complexes import Abs, im, re
# 从 sympy 指数函数模块中导入指数、对数和实部虚部匹配函数
from sympy.functions.elementary.exponential import exp, log, match_real_imag
# 从 sympy 整数函数模块中导入向下取整函数 floor
from sympy.functions.elementary.integers import floor
# 从 sympy 杂项函数模块中导入平方根函数 sqrt
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy 三角函数模块中导入各种三角函数和虚数单位作为系数函数
from sympy.functions.elementary.trigonometric import (
    acos, acot, asin, atan, cos, cot, csc, sec, sin, tan,
    _imaginary_unit_as_coefficient)
# 从 sympy 多项式特殊多项式模块中导入对称多项式函数
from sympy.polys.specialpolys import symmetric_poly


# 定义一个函数，用于将超越函数转换为指数形式的函数
def _rewrite_hyperbolics_as_exp(expr):
    return expr.xreplace({h: h.rewrite(exp)
        for h in expr.atoms(HyperbolicFunction)})


# 定义一个缓存函数，返回反双曲线余弦函数的值表
@cacheit
def _acosh_table():
    return {
        I: log(I*(1 + sqrt(2))),
        -I: log(-I*(1 + sqrt(2))),
        S.Half: pi/3,
        Rational(-1, 2): pi*Rational(2, 3),
        sqrt(2)/2: pi/4,
        -sqrt(2)/2: pi*Rational(3, 4),
        1/sqrt(2): pi/4,
        -1/sqrt(2): pi*Rational(3, 4),
        sqrt(3)/2: pi/6,
        -sqrt(3)/2: pi*Rational(5, 6),
        (sqrt(3) - 1)/sqrt(2**3): pi*Rational(5, 12),
        -(sqrt(3) - 1)/sqrt(2**3): pi*Rational(7, 12),
        sqrt(2 + sqrt(2))/2: pi/8,
        -sqrt(2 + sqrt(2))/2: pi*Rational(7, 8),
        sqrt(2 - sqrt(2))/2: pi*Rational(3, 8),
        -sqrt(2 - sqrt(2))/2: pi*Rational(5, 8),
        (1 + sqrt(3))/(2*sqrt(2)): pi/12,
        -(1 + sqrt(3))/(2*sqrt(2)): pi*Rational(11, 12),
        (sqrt(5) + 1)/4: pi/5,
        -(sqrt(5) + 1)/4: pi*Rational(4, 5)
    }


# 定义一个缓存函数，返回反双曲线余割函数的值表
@cacheit
def _acsch_table():
    return {
        I: -pi / 2,
        I*(sqrt(2) + sqrt(6)): -pi / 12,
        I*(1 + sqrt(5)): -pi / 10,
        I*2 / sqrt(2 - sqrt(2)): -pi / 8,
        I*2: -pi / 6,
        I*sqrt(2 + 2/sqrt(5)): -pi / 5,
        I*sqrt(2): -pi / 4,
        I*(sqrt(5)-1): -3*pi / 10,
        I*2 / sqrt(3): -pi / 3,
        I*2 / sqrt(2 + sqrt(2)): -3*pi / 8,
        I*sqrt(2 - 2/sqrt(5)): -2*pi / 5,
        I*(sqrt(6) - sqrt(2)): -5*pi / 12,
        S(2): -I*log((1+sqrt(5))/2),
    }
# 定义一个私有函数 `_asech_table`，返回一个包含各种反双曲余弦函数的值的字典
def _asech_table():
    return {
        I: - (pi*I / 2) + log(1 + sqrt(2)),  # 对于复数 I，返回其对应的反双曲余弦函数值
        -I: (pi*I / 2) + log(1 + sqrt(2)),  # 对于复数 -I，返回其对应的反双曲余弦函数值
        (sqrt(6) - sqrt(2)): pi / 12,  # 对于表达式 sqrt(6) - sqrt(2)，返回其对应的反双曲余弦函数值
        (sqrt(2) - sqrt(6)): 11*pi / 12,  # 对于表达式 sqrt(2) - sqrt(6)，返回其对应的反双曲余弦函数值
        sqrt(2 - 2/sqrt(5)): pi / 10,  # 对于表达式 sqrt(2 - 2/sqrt(5))，返回其对应的反双曲余弦函数值
        -sqrt(2 - 2/sqrt(5)): 9*pi / 10,  # 对于表达式 -sqrt(2 - 2/sqrt(5))，返回其对应的反双曲余弦函数值
        2 / sqrt(2 + sqrt(2)): pi / 8,  # 对于表达式 2 / sqrt(2 + sqrt(2))，返回其对应的反双曲余弦函数值
        -2 / sqrt(2 + sqrt(2)): 7*pi / 8,  # 对于表达式 -2 / sqrt(2 + sqrt(2))，返回其对应的反双曲余弦函数值
        2 / sqrt(3): pi / 6,  # 对于表达式 2 / sqrt(3)，返回其对应的反双曲余弦函数值
        -2 / sqrt(3): 5*pi / 6,  # 对于表达式 -2 / sqrt(3)，返回其对应的反双曲余弦函数值
        (sqrt(5) - 1): pi / 5,  # 对于表达式 sqrt(5) - 1，返回其对应的反双曲余弦函数值
        (1 - sqrt(5)): 4*pi / 5,  # 对于表达式 1 - sqrt(5)，返回其对应的反双曲余弦函数值
        sqrt(2): pi / 4,  # 对于表达式 sqrt(2)，返回其对应的反双曲余弦函数值
        -sqrt(2): 3*pi / 4,  # 对于表达式 -sqrt(2)，返回其对应的反双曲余弦函数值
        sqrt(2 + 2/sqrt(5)): 3*pi / 10,  # 对于表达式 sqrt(2 + 2/sqrt(5))，返回其对应的反双曲余弦函数值
        -sqrt(2 + 2/sqrt(5)): 7*pi / 10,  # 对于表达式 -sqrt(2 + 2/sqrt(5))，返回其对应的反双曲余弦函数值
        S(2): pi / 3,  # 对于符号 S(2)，返回其对应的反双曲余弦函数值
        -S(2): 2*pi / 3,  # 对于符号 -S(2)，返回其对应的反双曲余弦函数值
        sqrt(2*(2 + sqrt(2))): 3*pi / 8,  # 对于表达式 sqrt(2*(2 + sqrt(2)))，返回其对应的反双曲余弦函数值
        -sqrt(2*(2 + sqrt(2))): 5*pi / 8,  # 对于表达式 -sqrt(2*(2 + sqrt(2)))，返回其对应的反双曲余弦函数值
        (1 + sqrt(5)): 2*pi / 5,  # 对于表达式 1 + sqrt(5)，返回其对应的反双曲余弦函数值
        (-1 - sqrt(5)): 3*pi / 5,  # 对于表达式 -1 - sqrt(5)，返回其对应的反双曲余弦函数值
        (sqrt(6) + sqrt(2)): 5*pi / 12,  # 对于表达式 sqrt(6) + sqrt(2)，返回其对应的反双曲余弦函数值
        (-sqrt(6) - sqrt(2)): 7*pi / 12,  # 对于表达式 -sqrt(6) - sqrt(2)，返回其对应的反双曲余弦函数值
        I*S.Infinity: -pi*I / 2,  # 对于复数 I 乘以正无穷，返回其对应的反双曲余弦函数值
        I*S.NegativeInfinity: pi*I / 2,  # 对于复数 I 乘以负无穷，返回其对应的反双曲余弦函数值
    }

###############################################################################
########################### HYPERBOLIC FUNCTIONS ##############################
###############################################################################


class HyperbolicFunction(Function):
    """
    Base class for hyperbolic functions.

    See Also
    ========

    sinh, cosh, tanh, coth
    """

    unbranched = True


def _peeloff_ipi(arg):
    r"""
    Split ARG into two parts, a "rest" and a multiple of $I\pi$.
    This assumes ARG to be an ``Add``.
    The multiple of $I\pi$ returned in the second position is always a ``Rational``.

    Examples
    ========

    >>> from sympy.functions.elementary.hyperbolic import _peeloff_ipi as peel
    >>> from sympy import pi, I
    >>> from sympy.abc import x, y
    >>> peel(x + I*pi/2)
    (x, 1/2)
    >>> peel(x + I*2*pi/3 + I*pi*y)
    (x + I*pi*y + I*pi/6, 1/2)
    """
    ipi = pi*I
    # 对参数 arg 进行解析，将其分解为 "剩余部分" 和 $I\pi$ 的倍数
    for a in Add.make_args(arg):
        if a == ipi:
            K = S.One
            break
        elif a.is_Mul:
            K, p = a.as_two_terms()
            if p == ipi and K.is_Rational:
                break
    else:
        return arg, S.Zero

    m1 = (K % S.Half)
    m2 = K - m1
    return arg - m2*ipi, m2


class sinh(HyperbolicFunction):
    r"""
    ``sinh(x)`` is the hyperbolic sine of ``x``.

    The hyperbolic sine function is $\frac{e^x - e^{-x}}{2}$.

    Examples
    ========

    >>> from sympy import sinh
    >>> from sympy.abc import x
    >>> sinh(x)
    sinh(x)

    See Also
    ========

    cosh, tanh, asinh
    """

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return cosh(self.args[0])  # 返回 sinh 函数对应参数的导数，即 cosh 函数
        else:
            raise ArgumentIndexError(self, argindex)
    # 返回该函数的反函数，这里默认为返回双曲反正弦函数 asinh
    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return asinh

    # 类方法，用于评估给定参数的函数值
    @classmethod
    def eval(cls, arg):
        # 检查参数是否为数值类型
        if arg.is_Number:
            # 处理特殊数值情况
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity
            elif arg.is_zero:
                return S.Zero
            elif arg.is_negative:
                # 如果参数为负数，返回其相反数的函数值
                return -cls(-arg)
        else:
            # 处理复数无穷大的情况
            if arg is S.ComplexInfinity:
                return S.NaN

            # 尝试将参数转换为虚部系数
            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                # 如果成功转换，返回对应的复数正弦函数值
                return I * sin(i_coeff)
            else:
                # 如果参数可能提取负号，则返回其相反数的函数值
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

            # 处理参数为加法表达式的情况
            if arg.is_Add:
                # 分离参数中的虚数系数和主要部分
                x, m = _peeloff_ipi(arg)
                if m:
                    # 如果存在虚数系数，返回双曲正弦和余双曲余弦的组合
                    m = m*pi*I
                    return sinh(m)*cosh(x) + cosh(m)*sinh(x)

            # 处理参数为零的情况
            if arg.is_zero:
                return S.Zero

            # 处理参数为反双曲正弦函数的情况
            if arg.func == asinh:
                return arg.args[0]

            # 处理参数为反双曲余弦函数的情况
            if arg.func == acosh:
                x = arg.args[0]
                return sqrt(x - 1) * sqrt(x + 1)

            # 处理参数为反双曲正切函数的情况
            if arg.func == atanh:
                x = arg.args[0]
                return x/sqrt(1 - x**2)

            # 处理参数为反双曲余切函数的情况
            if arg.func == acoth:
                x = arg.args[0]
                return 1/(sqrt(x - 1) * sqrt(x + 1))

    # 静态方法，用于计算泰勒级数展开中的下一个项
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Returns the next term in the Taylor series expansion.
        """
        # 如果 n 小于零或者 n 是偶数，返回零
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            # 如果已有的前几项大于两项，使用前两项计算下一项
            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return p * x**2 / (n*(n - 1))
            else:
                # 否则，返回 x 的 n 次幂除以 n 的阶乘
                return x**(n) / factorial(n)

    # 内部方法，返回该函数的共轭值
    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    # 将该函数表示为复坐标系下的函数
    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a complex coordinate.
        """
        # 如果参数是扩展实数，则返回其实部和零
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        # 如果参数不是扩展实数，则返回其实部和虚部
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (sinh(re)*cos(im), cosh(re)*sin(im))

    # 内部方法，展开该函数的复数形式
    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*I
    # 根据深度参数选择是否对参数进行展开处理
    def _eval_expand_trig(self, deep=True, **hints):
        if deep:
            # 如果深度展开，递归调用第一个参数的展开方法
            arg = self.args[0].expand(deep, **hints)
        else:
            # 否则直接使用第一个参数
            arg = self.args[0]
        x = None
        if arg.is_Add:
            # 如果参数是加法类型，尝试将其分解为两个项
            x, y = arg.as_two_terms()
        else:
            # 否则尝试将参数分解为乘法系数和项
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff is not S.One and coeff.is_Integer and terms is not S.One:
                # 如果乘法系数不是1且是整数，且项不是1，则确定 x 和 y 的值
                x = terms
                y = (coeff - 1) * x
        if x is not None:
            # 如果 x 不为 None，则返回三角函数的展开表达式
            return (sinh(x) * cosh(y) + sinh(y) * cosh(x)).expand(trig=True)
        # 否则返回参数的双曲正弦函数
        return sinh(arg)

    # 将表达式重写为更易处理的形式，即双曲函数的展开形式
    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return (exp(arg) - exp(-arg)) / 2

    # 将表达式重写为指数函数的形式
    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return (exp(arg) - exp(-arg)) / 2

    # 将表达式重写为正弦函数的形式
    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return -I * sin(I * arg)

    # 将表达式重写为余割函数的形式
    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return -I / csc(I * arg)

    # 将表达式重写为双曲余弦函数的形式
    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return -I * cosh(arg + pi*I/2)

    # 将表达式重写为双曲正切函数的形式
    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        tanh_half = tanh(S.Half * arg)
        return 2 * tanh_half / (1 - tanh_half**2)

    # 将表达式重写为双曲余切函数的形式
    def _eval_rewrite_as_coth(self, arg, **kwargs):
        coth_half = coth(S.Half * arg)
        return 2 * coth_half / (coth_half**2 - 1)

    # 将表达式重写为双曲余割函数的形式
    def _eval_rewrite_as_csch(self, arg, **kwargs):
        return 1 / csch(arg)

    # 返回参数的主导项（leading term）
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 获取参数的主导项并计算在 x = 0 时的值
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            # 如果主导项在 x = 0 处为 NaN，则在左或右方向求极限
            arg0 = arg.limit(x, 0, dir='-' if cdir.is_negative else '+')
        if arg0.is_zero:
            # 如果主导项在 x = 0 处为 0，则返回原始表达式
            return arg
        elif arg0.is_finite:
            # 如果主导项在 x = 0 处有限，则返回计算结果
            return self.func(arg0)
        else:
            # 否则返回自身
            return self

    # 判断参数是否为实数
    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real:
            return True

        # 如果虚部 `im` 是 n*pi 的形式
        # 否则，检查是否为数值
        re, im = arg.as_real_imag()
        return (im % pi).is_zero

    # 判断参数是否为扩展实数
    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    # 判断参数是否为正数
    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    # 判断参数是否为负数
    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    # 判断参数是否为有限数
    def _eval_is_finite(self):
        arg = self.args[0]
        return arg.is_finite

    # 判断参数是否为零
    def _eval_is_zero(self):
        # 尝试将参数分解为主要部分和 `n*pi` 的倍数
        rest, ipi_mult = _peeloff_ipi(self.args[0])
        if rest.is_zero:
            # 如果主要部分为零，则检查是否 `ipi_mult` 为整数
            return ipi_mult.is_integer
class cosh(HyperbolicFunction):
    r"""
    ``cosh(x)`` is the hyperbolic cosine of ``x``.

    The hyperbolic cosine function is $\frac{e^x + e^{-x}}{2}$.

    Examples
    ========

    >>> from sympy import cosh
    >>> from sympy.abc import x
    >>> cosh(x)
    cosh(x)

    See Also
    ========

    sinh, tanh, acosh
    """

    # 计算多元函数的偏导数
    def fdiff(self, argindex=1):
        if argindex == 1:
            # 返回对第一个参数的偏导数，即 sinh(self.args[0])
            return sinh(self.args[0])
        else:
            # 若参数索引不为1，则抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    # 类方法，用于求解特定参数的函数值
    @classmethod
    def eval(cls, arg):
        from sympy.functions.elementary.trigonometric import cos
        # 若参数是数值
        if arg.is_Number:
            # 若参数为 NaN，则返回 NaN
            if arg is S.NaN:
                return S.NaN
            # 若参数为正无穷，则返回正无穷
            elif arg is S.Infinity:
                return S.Infinity
            # 若参数为负无穷，则返回正无穷
            elif arg is S.NegativeInfinity:
                return S.Infinity
            # 若参数为零，则返回 1
            elif arg.is_zero:
                return S.One
            # 若参数为负数，则返回其相反数的双曲余弦函数值
            elif arg.is_negative:
                return cls(-arg)
        else:
            # 若参数为复无穷，则返回 NaN
            if arg is S.ComplexInfinity:
                return S.NaN

            # 提取虚数单位作为系数
            i_coeff = _imaginary_unit_as_coefficient(arg)

            # 若虚数系数不为空，则返回其余弦函数值
            if i_coeff is not None:
                return cos(i_coeff)
            else:
                # 若能提取负号，则返回参数的相反数的双曲余弦函数值
                if arg.could_extract_minus_sign():
                    return cls(-arg)

                # 若参数为加法形式，则尝试分解成特定形式，计算其双曲余弦函数值
                if arg.is_Add:
                    x, m = _peeloff_ipi(arg)
                    if m:
                        m = m*pi*I
                        return cosh(m)*cosh(x) + sinh(m)*sinh(x)

                # 若参数为零，则返回 1
                if arg.is_zero:
                    return S.One

                # 若参数的函数为反双曲正弦函数，则返回其对应的值
                if arg.func == asinh:
                    return sqrt(1 + arg.args[0]**2)

                # 若参数的函数为反双曲余弦函数，则返回其参数值
                if arg.func == acosh:
                    return arg.args[0]

                # 若参数的函数为反双曲正切函数，则返回其对应的值
                if arg.func == atanh:
                    return 1/sqrt(1 - arg.args[0]**2)

                # 若参数的函数为反双曲余切函数，则返回其对应的值
                if arg.func == acoth:
                    x = arg.args[0]
                    return x/(sqrt(x - 1) * sqrt(x + 1))

    # 静态方法，使用缓存机制计算泰勒级数的项
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)

            # 若前项数大于2，则利用前两项计算泰勒级数的项
            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return p * x**2 / (n*(n - 1))
            else:
                # 若前项数不足2，则直接计算泰勒级数的项
                return x**(n)/factorial(n)

    # 计算函数的共轭复数
    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    # 将函数表示为实部和虚部的形式
    def as_real_imag(self, deep=True, **hints):
        # 若参数的实部为扩展实数，则返回实部和零的元组
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        # 若参数的实部非扩展实数，则将其展开为实部和虚部并返回
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()

        return (cosh(re)*cos(im), sinh(re)*sin(im))

    # 扩展函数的复数形式
    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*I
    # 对三角函数表达式进行展开和转化，可以选择是否进行深度展开
    def _eval_expand_trig(self, deep=True, **hints):
        # 如果选择深度展开，则对第一个参数进行深度展开
        if deep:
            arg = self.args[0].expand(deep, **hints)
        else:
            # 否则，直接使用原始参数
            arg = self.args[0]
        
        x = None
        # 如果参数是加法表达式
        if arg.is_Add: # TODO, implement more if deep stuff here
            # 尝试将参数分解为两个项
            x, y = arg.as_two_terms()
        else:
            # 否则，将参数分解为系数和乘法项
            coeff, terms = arg.as_coeff_Mul(rational=True)
            # 如果系数不是 1 且是整数，并且乘法项不是 1
            if coeff is not S.One and coeff.is_Integer and terms is not S.One:
                # 则将乘法项作为 x
                x = terms
                # y 为 (系数 - 1) * x
                y = (coeff - 1) * x
        
        # 如果成功分解 x，则返回展开后的双曲函数表达式
        if x is not None:
            return (cosh(x)*cosh(y) + sinh(x)*sinh(y)).expand(trig=True)
        # 否则，返回原始参数的双曲余弦函数
        return cosh(arg)

    # 将表达式重写为可处理的形式，使用指数函数的表达式
    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return (exp(arg) + exp(-arg)) / 2

    # 将表达式重写为使用指数函数的形式
    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return (exp(arg) + exp(-arg)) / 2

    # 将表达式重写为使用余弦函数的形式
    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return cos(I * arg, evaluate=False)

    # 将表达式重写为使用正割函数的形式
    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return 1 / sec(I * arg, evaluate=False)

    # 将表达式重写为使用双曲正弦函数的形式
    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return -I*sinh(arg + pi*I/2, evaluate=False)

    # 将表达式重写为使用双曲正切函数的形式
    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        tanh_half = tanh(S.Half*arg)**2
        return (1 + tanh_half)/(1 - tanh_half)

    # 将表达式重写为使用双曲余切函数的形式
    def _eval_rewrite_as_coth(self, arg, **kwargs):
        coth_half = coth(S.Half*arg)**2
        return (coth_half + 1)/(coth_half - 1)

    # 将表达式重写为使用双曲正割函数的形式
    def _eval_rewrite_as_sech(self, arg, **kwargs):
        return 1 / sech(arg)

    # 返回表达式的主导项，根据参数 x，可以指定对数项和方向
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 获取参数的主导项
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        # 将主导项代入 x=0 的情况
        arg0 = arg.subs(x, 0)

        # 如果主导项为 NaN，则计算极限
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if cdir.is_negative else '+')
        # 如果主导项为零，返回常数 1
        if arg0.is_zero:
            return S.One
        # 如果主导项为有限数值，返回函数自身在该值处的计算结果
        elif arg0.is_finite:
            return self.func(arg0)
        # 否则，返回自身
        else:
            return self

    # 判断函数是否为实数
    def _eval_is_real(self):
        arg = self.args[0]

        # 如果参数是实数或纯虚数，则函数为实数
        if arg.is_real or arg.is_imaginary:
            return True

        # 对于复数 arg = a + ib，其中 a 和 b 是实数部分和虚数部分
        re, im = arg.as_real_imag()
        # 如果虚数部分模 pi 为零，则返回 True
        return (im % pi).is_zero
    def _eval_is_positive(self):
        # 获取参数 z，这里假设 z 是复数
        z = self.args[0]

        # 获取 z 的实部和虚部
        x, y = z.as_real_imag()

        # 对 y 取模 2*pi，确保在 [0, 2*pi) 范围内
        ymod = y % (2*pi)

        # 检查 ymod 是否为零
        yzero = ymod.is_zero
        # 如果 ymod 是零，快速返回 True
        if yzero:
            return True

        # 检查 x 是否为零
        xzero = x.is_zero
        # 如果 x 不为零，根据 xzero 返回结果
        if xzero is False:
            return yzero

        # 处理复杂情况，使用模糊逻辑处理复数的正判定
        return fuzzy_or([
                # Case 1:
                yzero,
                # Case 2:
                fuzzy_and([
                    xzero,
                    fuzzy_or([ymod < pi/2, ymod > 3*pi/2])
                ])
            ])


    def _eval_is_nonnegative(self):
        # 获取参数 z，假设 z 是复数
        z = self.args[0]

        # 获取 z 的实部和虚部
        x, y = z.as_real_imag()

        # 对 y 取模 2*pi，确保在 [0, 2*pi) 范围内
        ymod = y % (2*pi)

        # 检查 ymod 是否为零
        yzero = ymod.is_zero
        # 如果 ymod 是零，快速返回 True
        if yzero:
            return True

        # 检查 x 是否为零
        xzero = x.is_zero
        # 如果 x 不为零，根据 xzero 返回结果
        if xzero is False:
            return yzero

        # 处理复杂情况，使用模糊逻辑处理复数的非负判定
        return fuzzy_or([
                # Case 1:
                yzero,
                # Case 2:
                fuzzy_and([
                    xzero,
                    fuzzy_or([ymod <= pi/2, ymod >= 3*pi/2])
                ])
            ])

    def _eval_is_finite(self):
        # 获取参数 arg
        arg = self.args[0]
        # 检查参数是否有限
        return arg.is_finite

    def _eval_is_zero(self):
        # 将复数 self.args[0] 拆解为整数倍的π和余数 rest
        rest, ipi_mult = _peeloff_ipi(self.args[0])
        # 如果 ipi_mult 存在且 rest 为零，返回 ipi_mult 是否为整数减半
        if ipi_mult and rest.is_zero:
            return (ipi_mult - S.Half).is_integer
class tanh(HyperbolicFunction):
    r"""
    ``tanh(x)`` is the hyperbolic tangent of ``x``.

    The hyperbolic tangent function is $\frac{\sinh(x)}{\cosh(x)}.

    Examples
    ========

    >>> from sympy import tanh
    >>> from sympy.abc import x
    >>> tanh(x)
    tanh(x)

    See Also
    ========

    sinh, cosh, atanh
    """

    def fdiff(self, argindex=1):
        # 导数函数，返回对指定参数的偏导数
        if argindex == 1:
            return S.One - tanh(self.args[0])**2
        else:
            # 抛出参数索引错误异常
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        # 返回此函数的反函数
        return atanh

    @classmethod
    def eval(cls, arg):
        # 对给定参数进行求值
        if arg.is_Number:
            # 如果参数是数值
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg.is_zero:
                return S.Zero
            elif arg.is_negative:
                return -cls(-arg)
        else:
            # 如果参数不是数值
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                if i_coeff.could_extract_minus_sign():
                    return -I * tan(-i_coeff)
                return I * tan(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    tanhm = tanh(m*pi*I)
                    if tanhm is S.ComplexInfinity:
                        return coth(x)
                    else: # tanhm == 0
                        return tanh(x)

            if arg.is_zero:
                return S.Zero

            if arg.func == asinh:
                x = arg.args[0]
                return x/sqrt(1 + x**2)

            if arg.func == acosh:
                x = arg.args[0]
                return sqrt(x - 1) * sqrt(x + 1) / x

            if arg.func == atanh:
                return arg.args[0]

            if arg.func == acoth:
                return 1/arg.args[0]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # 计算泰勒展开的项
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            a = 2**(n + 1)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return a*(a - 1) * B/F * x**n

    def _eval_conjugate(self):
        # 返回此函数的共轭
        return self.func(self.args[0].conjugate())
    # 将对象转换为其实部和虚部的形式
    def as_real_imag(self, deep=True, **hints):
        # 检查对象的第一个参数是否是扩展实数
        if self.args[0].is_extended_real:
            # 如果需要深度展开
            if deep:
                # 更新提示字典，指明不是复数
                hints['complex'] = False
                # 对对象进行深度展开并返回其实部为扩展后的对象，虚部为零
                return (self.expand(deep, **hints), S.Zero)
            else:
                # 返回对象本身作为实部，虚部为零
                return (self, S.Zero)
        # 如果需要深度展开
        if deep:
            # 获取第一个参数的实部和虚部
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            # 获取第一个参数的实部和虚部
            re, im = self.args[0].as_real_imag()
        # 计算双曲正弦和余弦的平方和
        denom = sinh(re)**2 + cos(im)**2
        # 返回双曲正弦和双曲余弦的比值作为实部，正弦和余弦的比值作为虚部
        return (sinh(re)*cosh(re)/denom, sin(im)*cos(im)/denom)

    # 将对象展开为三角函数的形式
    def _eval_expand_trig(self, **hints):
        # 获取对象的第一个参数
        arg = self.args[0]
        # 如果参数是加法表达式
        if arg.is_Add:
            # 获取加法表达式的项数
            n = len(arg.args)
            # 对每个项应用双曲正切的展开，并将结果存储在TX列表中
            TX = [tanh(x, evaluate=False)._eval_expand_trig()
                for x in arg.args]
            # 初始化分子和分母的列表
            p = [0, 0]  # [den, num]
            # 对每个指数应用对称多项式，并将结果加到相应的分子或分母中
            for i in range(n + 1):
                p[i % 2] += symmetric_poly(i, TX)
            # 返回展开后的表达式
            return p[1]/p[0]
        # 如果参数是乘法表达式
        elif arg.is_Mul:
            # 将参数拆分为系数和项
            coeff, terms = arg.as_coeff_Mul()
            # 如果系数是整数且大于1
            if coeff.is_Integer and coeff > 1:
                # 对乘积应用双曲正切的展开，并分别计算分子和分母
                T = tanh(terms)
                n = [nC(range(coeff), k)*T**k for k in range(1, coeff + 1, 2)]
                d = [nC(range(coeff), k)*T**k for k in range(0, coeff + 1, 2)]
                # 返回展开后的表达式
                return Add(*n)/Add(*d)
        # 如果参数不是加法或乘法表达式，直接返回双曲正切函数应用到参数上
        return tanh(arg)

    # 将对象重写为可处理的形式
    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        # 计算参数的负指数和正指数
        neg_exp, pos_exp = exp(-arg), exp(arg)
        # 返回重写后的表达式
        return (pos_exp - neg_exp)/(pos_exp + neg_exp)

    # 将对象重写为指数函数的形式
    def _eval_rewrite_as_exp(self, arg, **kwargs):
        # 计算参数的负指数和正指数
        neg_exp, pos_exp = exp(-arg), exp(arg)
        # 返回重写后的表达式
        return (pos_exp - neg_exp)/(pos_exp + neg_exp)

    # 将对象重写为正切函数的形式
    def _eval_rewrite_as_tan(self, arg, **kwargs):
        # 返回重写后的表达式
        return -I * tan(I * arg, evaluate=False)

    # 将对象重写为余切函数的形式
    def _eval_rewrite_as_cot(self, arg, **kwargs):
        # 返回重写后的表达式
        return -I / cot(I * arg, evaluate=False)

    # 将对象重写为双曲正弦函数的形式
    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        # 返回重写后的表达式
        return I*sinh(arg)/sinh(pi*I/2 - arg, evaluate=False)

    # 将对象重写为双曲余弦函数的形式
    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        # 返回重写后的表达式
        return I*cosh(pi*I/2 - arg, evaluate=False)/cosh(arg)

    # 将对象重写为双曲余切函数的形式
    def _eval_rewrite_as_coth(self, arg, **kwargs):
        # 返回重写后的表达式
        return 1/coth(arg)

    # 计算对象的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 导入计算级数的Order类
        from sympy.series.order import Order
        # 获取对象的第一个参数的主导项
        arg = self.args[0].as_leading_term(x)

        # 如果主导项中包含变量x，并且是一阶项
        if x in arg.free_symbols and Order(1, x).contains(arg):
            # 返回主导项
            return arg
        else:
            # 否则返回原始对象
            return self.func(arg)

    # 判断对象是否为实数
    def _eval_is_real(self):
        # 获取对象的第一个参数
        arg = self.args[0]
        # 如果参数是实数
        if arg.is_real:
            return True

        # 获取参数的实部和虚部
        re, im = arg.as_real_imag()

        # 如果实部为0且虚部模π为π/2
        if re == 0 and im % pi == pi/2:
            return None

        # 检查虚部是否为n*pi/2的形式以确定sin(2*im)是否为0
        # 如果不是，虚部可能是一个数字，此时返回False
        return (im % (pi/2)).is_zero

    # 判断对象是否为扩展实数
    def _eval_is_extended_real(self):
        # 如果对象的第一个参数是扩展实数
        if self.args[0].is_extended_real:
            return True
    # 判断表达式是否为正数的求值函数
    def _eval_is_positive(self):
        # 检查参数是否为扩展实数
        if self.args[0].is_extended_real:
            # 返回参数是否为正数的结果
            return self.args[0].is_positive

    # 判断表达式是否为负数的求值函数
    def _eval_is_negative(self):
        # 检查参数是否为扩展实数
        if self.args[0].is_extended_real:
            # 返回参数是否为负数的结果
            return self.args[0].is_negative

    # 判断表达式是否为有限数的求值函数
    def _eval_is_finite(self):
        # 获取函数的参数
        arg = self.args[0]

        # 将参数拆分为实部和虚部
        re, im = arg.as_real_imag()
        # 计算分母
        denom = cos(im)**2 + sinh(re)**2
        # 如果分母为零，则返回 False
        if denom == 0:
            return False
        # 如果分母为数值类型，则返回 True
        elif denom.is_number:
            return True
        # 如果参数是扩展实数，则返回 True
        if arg.is_extended_real:
            return True

    # 判断表达式是否为零的求值函数
    def _eval_is_zero(self):
        # 获取函数的参数
        arg = self.args[0]
        # 检查参数是否为零
        if arg.is_zero:
            return True
class coth(HyperbolicFunction):
    r"""
    ``coth(x)`` is the hyperbolic cotangent of ``x``.

    The hyperbolic cotangent function is $\frac{\cosh(x)}{\sinh(x)}.

    Examples
    ========

    >>> from sympy import coth
    >>> from sympy.abc import x
    >>> coth(x)
    coth(x)

    See Also
    ========

    sinh, cosh, acoth
    """

    def fdiff(self, argindex=1):
        # 计算该函数对第一个参数的导数
        if argindex == 1:
            return -1/sinh(self.args[0])**2
        else:
            # 抛出参数索引错误异常
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        # 返回该函数的逆函数 acoth
        return acoth

    @classmethod
    def eval(cls, arg):
        # 如果参数是数值类型
        if arg.is_Number:
            # 处理特殊数值情况
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg.is_zero:
                return S.ComplexInfinity
            elif arg.is_negative:
                # 处理负数情况
                return -cls(-arg)
        else:
            # 如果参数是复数无穷
            if arg is S.ComplexInfinity:
                return S.NaN

            # 提取虚数单位作为系数
            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                # 如果能提取出负号
                if i_coeff.could_extract_minus_sign():
                    return I * cot(-i_coeff)
                return -I * cot(i_coeff)
            else:
                # 如果参数能提取负号
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

            # 如果参数是加法运算
            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    cothm = coth(m*pi*I)
                    # 如果 cothm 为复数无穷
                    if cothm is S.ComplexInfinity:
                        return coth(x)
                    else: # cothm == 0
                        return tanh(x)

            # 如果参数是零
            if arg.is_zero:
                return S.ComplexInfinity

            # 如果参数是 arcsinh 函数
            if arg.func == asinh:
                x = arg.args[0]
                return sqrt(1 + x**2)/x

            # 如果参数是 arccosh 函数
            if arg.func == acosh:
                x = arg.args[0]
                return x/(sqrt(x - 1) * sqrt(x + 1))

            # 如果参数是 arctanh 函数
            if arg.func == atanh:
                return 1/arg.args[0]

            # 如果参数是 arccoth 函数
            if arg.func == acoth:
                return arg.args[0]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # 计算泰勒级数的第 n 项
        if n == 0:
            return 1 / sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            # 计算伯努利数
            B = bernoulli(n + 1)
            F = factorial(n + 1)

            # 返回泰勒级数的第 n 项
            return 2**(n + 1) * B/F * x**n

    def _eval_conjugate(self):
        # 返回该函数的共轭函数
        return self.func(self.args[0].conjugate())
    # 将当前对象转换为实部和虚部的形式，返回一个元组
    def as_real_imag(self, deep=True, **hints):
        # 导入必要的三角函数（余弦和正弦）用于后续计算
        from sympy.functions.elementary.trigonometric import (cos, sin)
        
        # 如果自身的第一个参数是实数
        if self.args[0].is_extended_real:
            # 如果需要深度展开
            if deep:
                # 设置提示参数中的复数为 False
                hints['complex'] = False
                # 对当前对象进行深度展开并返回展开后的实部和零虚部的元组
                return (self.expand(deep, **hints), S.Zero)
            else:
                # 否则直接返回自身和零虚部的元组
                return (self, S.Zero)
        
        # 如果需要深度展开
        if deep:
            # 对第一个参数进行深度展开，然后获取其实部和虚部
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            # 否则直接获取第一个参数的实部和虚部
            re, im = self.args[0].as_real_imag()
        
        # 计算双曲正弦和正弦的平方和
        denom = sinh(re)**2 + sin(im)**2
        
        # 返回实部和虚部转换后的结果
        return (sinh(re)*cosh(re)/denom, -sin(im)*cos(im)/denom)

    # 将对象重写为与双曲函数相关的表达式
    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        # 计算指数函数的负和正值
        neg_exp, pos_exp = exp(-arg), exp(arg)
        # 返回重写后的表达式
        return (pos_exp + neg_exp)/(pos_exp - neg_exp)

    # 将对象重写为指数函数的表达式
    def _eval_rewrite_as_exp(self, arg, **kwargs):
        # 计算指数函数的负和正值
        neg_exp, pos_exp = exp(-arg), exp(arg)
        # 返回重写后的表达式
        return (pos_exp + neg_exp)/(pos_exp - neg_exp)

    # 将对象重写为双曲正弦的表达式
    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        # 返回按照双曲正弦重写的表达式
        return -I*sinh(pi*I/2 - arg, evaluate=False)/sinh(arg)

    # 将对象重写为双曲余弦的表达式
    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        # 返回按照双曲余弦重写的表达式
        return -I*cosh(arg)/cosh(pi*I/2 - arg, evaluate=False)

    # 将对象重写为双曲正切的表达式
    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        # 返回按照双曲正切重写的表达式
        return 1/tanh(arg)

    # 判断对象是否为正数
    def _eval_is_positive(self):
        # 如果第一个参数是扩展实数，则返回该参数是否为正数的判断
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    # 判断对象是否为负数
    def _eval_is_negative(self):
        # 如果第一个参数是扩展实数，则返回该参数是否为负数的判断
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    # 将对象展开为主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 导入必要的 Order 类用于后续比较
        from sympy.series.order import Order
        # 获取第一个参数关于 x 的主导项
        arg = self.args[0].as_leading_term(x)

        # 如果参数中包含 x 并且是 Order(1, x) 的一部分
        if x in arg.free_symbols and Order(1, x).contains(arg):
            # 返回 1/arg
            return 1/arg
        else:
            # 否则返回原始函数应用于 arg 的结果
            return self.func(arg)

    # 展开对象的三角函数
    def _eval_expand_trig(self, **hints):
        # 获取第一个参数
        arg = self.args[0]
        
        # 如果参数是 Add 类型
        if arg.is_Add:
            # 对参数中的每个元素递归地应用 coth 函数的展开
            CX = [coth(x, evaluate=False)._eval_expand_trig() for x in arg.args]
            p = [[], []]
            n = len(arg.args)
            # 构建对称多项式
            for i in range(n, -1, -1):
                p[(n - i) % 2].append(symmetric_poly(i, CX))
            # 返回对称多项式中奇数项和偶数项的比值
            return Add(*p[0])/Add(*p[1])
        
        # 如果参数是 Mul 类型
        elif arg.is_Mul:
            # 获取参数中的系数和 x
            coeff, x = arg.as_coeff_Mul(rational=True)
            # 如果系数是整数且大于1
            if coeff.is_Integer and coeff > 1:
                # 计算 coth(x) 的展开
                c = coth(x, evaluate=False)
                p = [[], []]
                # 构建二项式展开
                for i in range(coeff, -1, -1):
                    p[(coeff - i) % 2].append(binomial(coeff, i)*c**i)
                # 返回二项式展开中奇数项和偶数项的比值
                return Add(*p[0])/Add(*p[1])
        
        # 对于其他情况，直接返回 coth(arg) 的结果
        return coth(arg)
class ReciprocalHyperbolicFunction(HyperbolicFunction):
    """Base class for reciprocal functions of hyperbolic functions. """

    # 以下三个类属性分别用于存储被倒数的函数、是否为偶函数、是否为奇函数的信息
    _reciprocal_of = None
    _is_even: FuzzyBool = None
    _is_odd: FuzzyBool = None

    @classmethod
    def eval(cls, arg):
        # 如果可以提取参数的负号
        if arg.could_extract_minus_sign():
            # 如果当前类是偶函数
            if cls._is_even:
                # 返回参数的相反数
                return cls(-arg)
            # 如果当前类是奇函数
            if cls._is_odd:
                # 返回参数的相反数，并取负值
                return -cls(-arg)

        # 计算被倒数函数在参数上的值
        t = cls._reciprocal_of.eval(arg)
        # 如果参数具有逆函数且其逆函数是当前类，则返回参数的第一个元素
        if hasattr(arg, 'inverse') and arg.inverse() == cls:
            return arg.args[0]
        # 返回被倒数函数的倒数，如果t不为None的话
        return 1/t if t is not None else t

    def _call_reciprocal(self, method_name, *args, **kwargs):
        # 在被倒数函数上调用指定的方法名
        o = self._reciprocal_of(self.args[0])
        return getattr(o, method_name)(*args, **kwargs)

    def _calculate_reciprocal(self, method_name, *args, **kwargs):
        # 如果在被倒数函数上调用指定的方法名返回不为None的值，则返回其倒数
        t = self._call_reciprocal(method_name, *args, **kwargs)
        return 1/t if t is not None else t

    def _rewrite_reciprocal(self, method_name, arg):
        # 特殊处理重写函数。如果被倒数重写返回未修改的表达式，则返回None
        t = self._call_reciprocal(method_name, arg)
        if t is not None and t != self._reciprocal_of(arg):
            return 1/t

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        # 重写为指数函数的形式
        return self._rewrite_reciprocal("_eval_rewrite_as_exp", arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        # 重写为可处理函数的形式
        return self._rewrite_reciprocal("_eval_rewrite_as_tractable", arg)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        # 重写为双曲正切函数的形式
        return self._rewrite_reciprocal("_eval_rewrite_as_tanh", arg)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        # 重写为双曲余切函数的形式
        return self._rewrite_reciprocal("_eval_rewrite_as_coth", arg)

    def as_real_imag(self, deep=True, **hints):
        # 返回被倒数函数的倒数的实部和虚部
        return (1 / self._reciprocal_of(self.args[0])).as_real_imag(deep, **hints)

    def _eval_conjugate(self):
        # 返回参数的共轭函数
        return self.func(self.args[0].conjugate())

    def _eval_expand_complex(self, deep=True, **hints):
        # 将函数展开为实部和虚部的和
        re_part, im_part = self.as_real_imag(deep=True, **hints)
        return re_part + I*im_part

    def _eval_expand_trig(self, **hints):
        # 计算被倒数函数的扩展三角形式
        return self._calculate_reciprocal("_eval_expand_trig", **hints)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 返回被倒数函数的主导项作为参数
        return (1/self._reciprocal_of(self.args[0]))._eval_as_leading_term(x)

    def _eval_is_extended_real(self):
        # 检查被倒数函数是否是扩展实数
        return self._reciprocal_of(self.args[0]).is_extended_real

    def _eval_is_finite(self):
        # 检查被倒数函数的倒数是否是有限数
        return (1/self._reciprocal_of(self.args[0])).is_finite


class csch(ReciprocalHyperbolicFunction):
    r"""
    ``csch(x)`` is the hyperbolic cosecant of ``x``.

    The hyperbolic cosecant function is $\frac{2}{e^x - e^{-x}}$

    Examples
    _reciprocal_of = sinh
    _is_odd = True
    
    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function
        """
        # 如果参数索引为1，返回函数关于第一个参数的导数
        if argindex == 1:
            return -coth(self.args[0]) * csch(self.args[0])
        else:
            # 如果参数索引不为1，抛出参数索引错误异常
            raise ArgumentIndexError(self, argindex)
    
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Returns the next term in the Taylor series expansion
        """
        # 如果 n 为0，返回 Taylor 级数展开的下一项
        if n == 0:
            return 1/sympify(x)
        # 如果 n 小于0或者是偶数，返回零
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
    
            # 计算伯努利数和阶乘
            B = bernoulli(n + 1)
            F = factorial(n + 1)
    
            # 返回 Taylor 级数展开的下一项
            return 2 * (1 - 2**n) * B/F * x**n
    
    def _eval_rewrite_as_sin(self, arg, **kwargs):
        """
        Rewrites the function in terms of sine
        """
        return I / sin(I * arg, evaluate=False)
    
    def _eval_rewrite_as_csc(self, arg, **kwargs):
        """
        Rewrites the function in terms of cosecant
        """
        return I * csc(I * arg, evaluate=False)
    
    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        """
        Rewrites the function in terms of hyperbolic cosine
        """
        return I / cosh(arg + I * pi / 2, evaluate=False)
    
    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        """
        Rewrites the function in terms of hyperbolic sine
        """
        return 1 / sinh(arg)
    
    def _eval_is_positive(self):
        """
        Checks if the function is positive
        """
        # 如果参数是扩展实数，返回参数是否为正
        if self.args[0].is_extended_real:
            return self.args[0].is_positive
    
    def _eval_is_negative(self):
        """
        Checks if the function is negative
        """
        # 如果参数是扩展实数，返回参数是否为负
        if self.args[0].is_extended_real:
            return self.args[0].is_negative
class sech(ReciprocalHyperbolicFunction):
    r"""
    ``sech(x)`` is the hyperbolic secant of ``x``.

    The hyperbolic secant function is $\frac{2}{e^x + e^{-x}}$

    Examples
    ========

    >>> from sympy import sech
    >>> from sympy.abc import x
    >>> sech(x)
    sech(x)

    See Also
    ========

    sinh, cosh, tanh, coth, csch, asinh, acosh
    """

    # 设置逆函数为双曲余弦函数
    _reciprocal_of = cosh
    # 标记为偶函数
    _is_even = True

    # 计算函数的导数
    def fdiff(self, argindex=1):
        if argindex == 1:
            # 返回双曲正切函数乘以双曲余弦函数的负数作为导数
            return - tanh(self.args[0])*sech(self.args[0])
        else:
            # 报告参数索引错误
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    # 计算泰勒级数的项
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            # 如果指数为负数或奇数，返回零
            return S.Zero
        else:
            # 否则，根据欧拉数和阶乘计算泰勒级数的项
            x = sympify(x)
            return euler(n) / factorial(n) * x**(n)

    # 使用余弦函数重写函数
    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return 1 / cos(I * arg, evaluate=False)

    # 使用正割函数重写函数
    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return sec(I * arg, evaluate=False)

    # 使用双曲正弦函数重写函数
    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return I / sinh(arg + I * pi /2, evaluate=False)

    # 使用双曲余弦函数重写函数
    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return 1 / cosh(arg)

    # 检查函数是否为正数
    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return True


###############################################################################
############################# HYPERBOLIC INVERSES #############################
###############################################################################

class InverseHyperbolicFunction(Function):
    """Base class for inverse hyperbolic functions."""

    pass


class asinh(InverseHyperbolicFunction):
    """
    ``asinh(x)`` is the inverse hyperbolic sine of ``x``.

    The inverse hyperbolic sine function.

    Examples
    ========

    >>> from sympy import asinh
    >>> from sympy.abc import x
    >>> asinh(x).diff(x)
    1/sqrt(x**2 + 1)
    >>> asinh(1)
    log(1 + sqrt(2))

    See Also
    ========

    acosh, atanh, sinh
    """

    # 计算函数的导数
    def fdiff(self, argindex=1):
        if argindex == 1:
            # 返回反双曲正弦函数的导数
            return 1/sqrt(self.args[0]**2 + 1)
        else:
            # 报告参数索引错误
            raise ArgumentIndexError(self, argindex)

    @classmethod
    # 定义一个类方法 eval，用于计算给定参数 arg 的数学表达式的值
    def eval(cls, arg):
        # 检查参数是否为数值类型
        if arg.is_Number:
            # 如果参数是数值类型，进一步判断其具体类型
            if arg is S.NaN:
                return S.NaN  # 如果是 NaN，则返回 NaN
            elif arg is S.Infinity:
                return S.Infinity  # 如果是正无穷，则返回正无穷
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity  # 如果是负无穷，则返回负无穷
            elif arg.is_zero:
                return S.Zero  # 如果是零，则返回零
            elif arg is S.One:
                return log(sqrt(2) + 1)  # 如果是 1，则返回 log(sqrt(2) + 1)
            elif arg is S.NegativeOne:
                return log(sqrt(2) - 1)  # 如果是 -1，则返回 log(sqrt(2) - 1)
            elif arg.is_negative:
                return -cls(-arg)  # 如果是负数，则返回其相反数的递归调用结果
        else:
            # 如果参数不是数值类型，根据不同情况返回特定的数学表达式的值
            if arg is S.ComplexInfinity:
                return S.ComplexInfinity  # 如果是复数无穷，则返回复数无穷

            if arg.is_zero:
                return S.Zero  # 如果是零，则返回零

            # 将虚数单位作为系数提取出来
            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return I * asin(i_coeff)  # 如果存在虚数单位系数，则返回其与 asin 函数的乘积
            else:
                # 如果可以提取负号，则返回其相反数的递归调用结果
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

        # 处理特定情况：如果参数是 sinh 函数的实例且其参数是数值类型
        if isinstance(arg, sinh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return z  # 如果参数是实数，则直接返回该参数
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor((i + pi/2) / pi)
                m = z - I * pi * f
                even = f.is_even
                if even is True:
                    return m  # 如果 f 是偶数，则返回 m
                elif even is False:
                    return -m  # 如果 f 是奇数，则返回 -m

    # 定义一个静态方法 taylor_term，用于计算泰勒级数的第 n 项
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # 如果 n 小于 0 或者 n 是偶数，则返回零
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)  # 将 x 转换为符号表达式
            # 如果提供的前几项数大于等于 2 并且 n 大于 2，则使用前两项计算当前项
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return -p * (n - 2)**2 / (n * (n - 1)) * x**2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return S.NegativeOne**k * R / F * x**n / n
    # 定义函数 _eval_as_leading_term，计算函数的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):  # asinh
        # 获取函数的参数
        arg = self.args[0]
        # 计算参数在 x=0 处的极限并化简
        x0 = arg.subs(x, 0).cancel()
        # 如果 x0 是零，返回参数的主导项
        if x0.is_zero:
            return arg.as_leading_term(x)

        # 如果 x0 是 NaN，将参数的主导项作为新表达式，如果该表达式是有限的，则返回它，否则返回自身
        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # 处理分支点
        if x0 in (-I, I, S.ComplexInfinity):
            # 重写为对数形式，并计算主导项
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        
        # 处理位于分支切线上的点 (-I*oo, -I) U (I, I*oo)
        if (1 + x0**2).is_negative:
            # 确定参数在 x 方向的导数方向
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_negative:
                    # 返回结果，考虑虚部为负的情况
                    return -self.func(x0) - I*pi
            elif re(ndir).is_negative:
                if im(x0).is_positive:
                    # 返回结果，考虑虚部为正的情况
                    return -self.func(x0) + I*pi
            else:
                # 重写为对数形式，并计算主导项
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        
        # 返回参数 x0 的函数值作为结果
        return self.func(x0)

    # 定义函数 _eval_nseries，计算函数的 n 级数展开
    def _eval_nseries(self, x, n, logx, cdir=0):  # asinh
        # 获取函数的参数
        arg = self.args[0]
        # 计算参数在 x=0 处的值
        arg0 = arg.subs(x, 0)

        # 处理分支点
        if arg0 in (I, -I):
            # 重写为对数形式，并计算 n 级数展开
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        # 计算函数的 n 级数展开
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # 处理位于分支切线上的点 (-I*oo, -I) U (I, I*oo)
        if (1 + arg0**2).is_negative:
            # 确定参数在 x 方向的导数方向
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(arg0).is_negative:
                    # 返回结果，考虑虚部为负的情况
                    return -res - I*pi
            elif re(ndir).is_negative:
                if im(arg0).is_positive:
                    # 返回结果，考虑虚部为正的情况
                    return -res + I*pi
            else:
                # 重写为对数形式，并计算 n 级数展开
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        
        # 返回计算的 n 级数展开结果
        return res

    # 定义函数 _eval_rewrite_as_log，将函数重写为对数形式
    def _eval_rewrite_as_log(self, x, **kwargs):
        return log(x + sqrt(x**2 + 1))

    # 将 _eval_rewrite_as_tractable 定义为与 _eval_rewrite_as_log 相同的函数
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    # 定义函数 _eval_rewrite_as_atanh，将函数重写为双曲反正切形式
    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return atanh(x/sqrt(1 + x**2))

    # 定义函数 _eval_rewrite_as_acosh，将函数重写为反双曲余弦形式
    def _eval_rewrite_as_acosh(self, x, **kwargs):
        ix = I*x
        return I*(sqrt(1 - ix)/sqrt(ix - 1) * acosh(ix) - pi/2)

    # 定义函数 _eval_rewrite_as_asin，将函数重写为反正弦形式
    def _eval_rewrite_as_asin(self, x, **kwargs):
        return -I * asin(I * x, evaluate=False)

    # 定义函数 _eval_rewrite_as_acos，将函数重写为反余弦形式
    def _eval_rewrite_as_acos(self, x, **kwargs):
        return I * acos(I * x, evaluate=False) - I*pi/2

    # 定义函数 inverse，返回该函数的逆函数
    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sinh

    # 定义函数 _eval_is_zero，判断函数是否在其参数为零时为零
    def _eval_is_zero(self):
        return self.args[0].is_zero

    # 定义函数 _eval_is_extended_real，判断函数是否在其参数为实数时为实数
    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    # 定义函数 _eval_is_finite，判断函数是否在其参数为有限数时为有限数
    def _eval_is_finite(self):
        return self.args[0].is_finite
class acosh(InverseHyperbolicFunction):
    """
    ``acosh(x)`` is the inverse hyperbolic cosine of ``x``.

    The inverse hyperbolic cosine function.

    Examples
    ========

    >>> from sympy import acosh
    >>> from sympy.abc import x
    >>> acosh(x).diff(x)
    1/(sqrt(x - 1)*sqrt(x + 1))
    >>> acosh(1)
    0

    See Also
    ========

    asinh, atanh, cosh
    """

    # 定义求导函数 fdiff，用于计算导数
    def fdiff(self, argindex=1):
        if argindex == 1:
            # 获取函数参数
            arg = self.args[0]
            # 返回导数表达式
            return 1/(sqrt(arg - 1)*sqrt(arg + 1))
        else:
            # 抛出参数索引错误异常
            raise ArgumentIndexError(self, argindex)

    @classmethod
    # 类方法 eval，用于对特定参数值进行求值
    def eval(cls, arg):
        if arg.is_Number:
            # 处理特定的数值情况
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg.is_zero:
                return pi*I / 2
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi*I

        if arg.is_number:
            # 使用 _acosh_table 函数获取常数表
            cst_table = _acosh_table()

            if arg in cst_table:
                # 返回表中对应的值
                if arg.is_extended_real:
                    return cst_table[arg]*I
                return cst_table[arg]

        # 处理特定的复数情况
        if arg is S.ComplexInfinity:
            return S.ComplexInfinity
        if arg == I*S.Infinity:
            return S.Infinity + I*pi/2
        if arg == -I*S.Infinity:
            return S.Infinity - I*pi/2

        # 处理特定的零情况
        if arg.is_zero:
            return pi*I*S.Half

        # 处理 cosh 类实例及其参数的情况
        if isinstance(arg, cosh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return Abs(z)
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor(i/pi)
                m = z - I*pi*f
                even = f.is_even
                if even is True:
                    if r.is_nonnegative:
                        return m
                    elif r.is_negative:
                        return -m
                elif even is False:
                    m -= I*pi
                    if r.is_nonpositive:
                        return -m
                    elif r.is_positive:
                        return m

    @staticmethod
    @cacheit
    # 静态方法 taylor_term，用于计算泰勒级数的项
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return I*pi/2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return p * (n - 2)**2/(n*(n - 1)) * x**2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return -R / F * I * x**n / n
    def _eval_as_leading_term(self, x, logx=None, cdir=0):  # acosh
        # 获取函数的参数表达式
        arg = self.args[0]
        # 计算参数在 x=0 处的主导项
        x0 = arg.subs(x, 0).cancel()
        
        # 处理分支点
        if x0 in (-S.One, S.Zero, S.One, S.ComplexInfinity):
            # 若 x0 是特定分支点，则重写为对数形式后再计算主导项
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        if x0 is S.NaN:
            # 若 x0 是 NaN，则将参数的主导项作为表达式返回，如果是有限值则返回表达式本身，否则返回自身
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # 处理落在分支切割线 (-oo, 1) 上的点
        if (x0 - 1).is_negative:
            # 确定参数在 x 方向上的导数方向
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                # 如果虚部为负，则根据具体情况调整返回值
                if (x0 + 1).is_negative:
                    return self.func(x0) - 2*I*pi
                return -self.func(x0)
            elif not im(ndir).is_positive:
                # 如果虚部不为正，则重写为对数形式后再计算主导项
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # 返回参数 x0 的函数值
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acosh
        # 获取函数的参数表达式
        arg = self.args[0]
        # 计算参数在 x=0 处的主导项
        arg0 = arg.subs(x, 0)

        # 处理分支点
        if arg0 in (S.One, S.NegativeOne):
            # 若参数在特定分支点上，则重写为对数形式后再计算 n 级数展开
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        # 计算函数在 x 处的 n 级数展开
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            # 如果参数为无穷大，则直接返回 n 级数展开的结果
            return res

        # 处理落在分支切割线 (-oo, 1) 上的点
        if (arg0 - 1).is_negative:
            # 确定参数在 x 方向上的导数方向
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                # 如果虚部为负，则根据具体情况调整 n 级数展开的结果
                if (arg0 + 1).is_negative:
                    return res - 2*I*pi
                return -res
            elif not im(ndir).is_positive:
                # 如果虚部不为正，则重写为对数形式后再计算 n 级数展开
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        # 返回 n 级数展开的结果
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        # 返回函数的对数形式表达式
        return log(x + sqrt(x + 1) * sqrt(x - 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_acos(self, x, **kwargs):
        # 返回函数的反余弦形式表达式
        return sqrt(x - 1)/sqrt(1 - x) * acos(x)

    def _eval_rewrite_as_asin(self, x, **kwargs):
        # 返回函数的反正弦形式表达式
        return sqrt(x - 1)/sqrt(1 - x) * (pi/2 - asin(x))

    def _eval_rewrite_as_asinh(self, x, **kwargs):
        # 返回函数的反双曲正弦形式表达式
        return sqrt(x - 1)/sqrt(1 - x) * (pi/2 + I*asinh(I*x, evaluate=False))

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        # 返回函数的反双曲正切形式表达式
        sxm1 = sqrt(x - 1)
        s1mx = sqrt(1 - x)
        sx2m1 = sqrt(x**2 - 1)
        return (pi/2*sxm1/s1mx*(1 - x * sqrt(1/x**2)) +
                sxm1*sqrt(x + 1)/sx2m1 * atanh(sx2m1/x))

    def inverse(self, argindex=1):
        """
        返回该函数的反函数。
        """
        return cosh

    def _eval_is_zero(self):
        # 判断函数是否为零的特殊情况
        if (self.args[0] - 1).is_zero:
            return True

    def _eval_is_extended_real(self):
        # 判断函数是否为扩展实数的特殊情况
        return fuzzy_and([self.args[0].is_extended_real, (self.args[0] - 1).is_extended_nonnegative])

    def _eval_is_finite(self):
        # 判断函数是否为有限的特殊情况
        return self.args[0].is_finite
# 定义 atanh 类，继承自 InverseHyperbolicFunction 类
class atanh(InverseHyperbolicFunction):
    """
    ``atanh(x)`` is the inverse hyperbolic tangent of ``x``.

    The inverse hyperbolic tangent function.

    Examples
    ========

    >>> from sympy import atanh
    >>> from sympy.abc import x
    >>> atanh(x).diff(x)
    1/(1 - x**2)

    See Also
    ========

    asinh, acosh, tanh
    """

    # fdiff 方法：计算函数的偏导数
    def fdiff(self, argindex=1):
        # 如果参数索引为1
        if argindex == 1:
            # 返回偏导数表达式
            return 1/(1 - self.args[0]**2)
        else:
            # 抛出参数索引错误异常
            raise ArgumentIndexError(self, argindex)

    # eval 类方法：对给定的参数进行求值
    @classmethod
    def eval(cls, arg):
        # 如果参数是一个数字
        if arg.is_Number:
            # 处理特殊情况
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return S.Infinity
            elif arg is S.NegativeOne:
                return S.NegativeInfinity
            elif arg is S.Infinity:
                return -I * atan(arg)
            elif arg is S.NegativeInfinity:
                return I * atan(-arg)
            elif arg.is_negative:
                return -cls(-arg)
        else:
            # 如果参数不是一个数字
            if arg is S.ComplexInfinity:
                # 处理复无穷情况
                from sympy.calculus.accumulationbounds import AccumBounds
                return I * AccumBounds(-pi/2, pi/2)

            # 尝试将参数视为虚数单位的系数
            i_coeff = _imaginary_unit_as_coefficient(arg)

            # 如果成功提取到虚数单位系数
            if i_coeff is not None:
                return I * atan(i_coeff)
            else:
                # 如果无法提取到虚数单位系数
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

        # 处理特殊情况：参数为零
        if arg.is_zero:
            return S.Zero

        # 如果参数是 tanh 类的实例且参数为数字
        if isinstance(arg, tanh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return z
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor(2*i/pi)
                even = f.is_even
                m = z - I*f*pi/2
                if even is True:
                    return m
                elif even is False:
                    return m - I*pi/2

    # taylor_term 静态方法：计算泰勒级数中的第 n 项
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # 如果 n 小于零或者 n 是偶数
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            # 返回泰勒级数的第 n 项
            return x**n / n
    # 对于 self.args 中的第一个参数进行计算，代表函数的参数
    def _eval_as_leading_term(self, x, logx=None, cdir=0):  # atanh
        # 获取函数的参数
        arg = self.args[0]
        # 计算在 x=0 处的表达式的极限
        x0 = arg.subs(x, 0).cancel()
        # 如果 x0 等于零，则返回参数的主导项
        if x0.is_zero:
            return arg.as_leading_term(x)
        # 如果 x0 是 NaN，则将参数的主导项作为函数的表达式，如果表达式是有限的则返回，否则返回自身
        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # 处理分支点
        if x0 in (-S.One, S.One, S.ComplexInfinity):
            # 将函数重写为对数的形式，并计算主导项
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # 处理位于分支切割线上的点 (-oo, -1] U [1, oo)
        if (1 - x0**2).is_negative:
            # 确定参数 x 的方向
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_negative:
                    return self.func(x0) - I*pi
            elif im(ndir).is_positive:
                if x0.is_positive:
                    return self.func(x0) + I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # 返回参数的主导项
        return self.func(x0)

    # 对于参数 x 计算 n 阶级数展开
    def _eval_nseries(self, x, n, logx, cdir=0):  # atanh
        # 获取函数的参数
        arg = self.args[0]
        # 计算在 x=0 处的参数值
        arg0 = arg.subs(x, 0)

        # 处理分支点
        if arg0 in (S.One, S.NegativeOne):
            # 将函数重写为对数的形式，并计算 n 阶级数展开
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        # 计算函数的 n 阶级数展开
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # 处理位于分支切割线上的点 (-oo, -1] U [1, oo)
        if (1 - arg0**2).is_negative:
            # 确定参数 x 的方向
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_negative:
                    return res - I*pi
            elif im(ndir).is_positive:
                if arg0.is_positive:
                    return res + I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        # 返回计算结果
        return res

    # 将函数重写为对数的形式
    def _eval_rewrite_as_log(self, x, **kwargs):
        return (log(1 + x) - log(1 - x)) / 2

    # 将函数重写为与 log 相同的形式
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    # 将函数重写为反双曲正弦函数的形式
    def _eval_rewrite_as_asinh(self, x, **kwargs):
        f = sqrt(1/(x**2 - 1))
        return (pi*x/(2*sqrt(-x**2)) -
                sqrt(-x)*sqrt(1 - x**2)/sqrt(x)*f*asinh(f))

    # 判断函数是否为零
    def _eval_is_zero(self):
        if self.args[0].is_zero:
            return True

    # 判断函数是否为扩展实数
    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, (1 - self.args[0]).is_nonnegative, (self.args[0] + 1).is_nonnegative])

    # 判断函数是否为有限数
    def _eval_is_finite(self):
        return fuzzy_not(fuzzy_or([(self.args[0] - 1).is_zero, (self.args[0] + 1).is_zero]))

    # 判断函数是否为虚数
    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    # 返回函数的反函数
    def inverse(self, argindex=1):
        """
        返回此函数的反函数。
        """
        return tanh
class acoth(InverseHyperbolicFunction):
    """
    ``acoth(x)`` is the inverse hyperbolic cotangent of ``x``.

    The inverse hyperbolic cotangent function.

    Examples
    ========

    >>> from sympy import acoth
    >>> from sympy.abc import x
    >>> acoth(x).diff(x)
    1/(1 - x**2)

    See Also
    ========

    asinh, acosh, coth
    """

    def fdiff(self, argindex=1):
        # 如果参数索引为1，返回对参数的导数值
        if argindex == 1:
            return 1/(1 - self.args[0]**2)
        else:
            # 否则抛出参数索引错误异常
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        # 如果参数是一个数字
        if arg.is_Number:
            # 根据不同的参数值返回对应的特定结果
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return pi*I / 2
            elif arg is S.One:
                return S.Infinity
            elif arg is S.NegativeOne:
                return S.NegativeInfinity
            elif arg.is_negative:
                return -cls(-arg)
        else:
            # 如果参数是复无穷
            if arg is S.ComplexInfinity:
                return S.Zero
            
            # 尝试将参数视作虚数系数
            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return -I * acot(i_coeff)
            else:
                # 如果参数可能包含负号，返回对其负值的结果
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

        # 处理特殊情况：参数为零时的处理
        if arg.is_zero:
            return pi*I*S.Half

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # 根据泰勒级数展开返回对应的项
        if n == 0:
            return -I*pi/2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return x**n / n

    def _eval_as_leading_term(self, x, logx=None, cdir=0):  # acoth
        # 获取参数并计算其在 x=0 处的值
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        # 处理参数为复无穷的情况
        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        # 处理参数为 NaN 的情况
        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # 处理分支点
        if x0 in (-S.One, S.One, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # 处理位于分支割线 [-1, 1] 上的点
        if x0.is_real and (1 - x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_positive:
                    return self.func(x0) + I*pi
            elif im(ndir).is_positive:
                if x0.is_negative:
                    return self.func(x0) - I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)
    # 定义一个方法 `_eval_nseries`，计算函数在 x 处的 n 级数展开，支持处理分支点
    def _eval_nseries(self, x, n, logx, cdir=0):  # acoth
        # 获取函数的参数表达式
        arg = self.args[0]
        # 计算参数在 x=0 处的代入值
        arg0 = arg.subs(x, 0)

        # 处理分支点
        if arg0 in (S.One, S.NegativeOne):
            # 若参数在分支点上，则利用对数函数重写并重新计算 n 级数展开
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        # 计算函数在 x 处的 n 级数展开
        res = Function._eval_nseries(self, x, n=n, logx=logx)

        # 处理参数为复无穷的情况
        if arg0 is S.ComplexInfinity:
            return res

        # 处理落在分支割线 [-1, 1] 上的点
        if arg0.is_real and (1 - arg0**2).is_positive:
            # 获取参数 x 在方向 cdir 上的导数
            ndir = arg.dir(x, cdir if cdir else 1)
            # 判断导数的虚部是否为负
            if im(ndir).is_negative:
                # 如果参数为正，则返回结果加上虚数单位乘以 π
                if arg0.is_positive:
                    return res + I*pi
            elif im(ndir).is_positive:
                # 如果参数为负，则返回结果减去虚数单位乘以 π
                if arg0.is_negative:
                    return res - I*pi
            else:
                # 对于其他情况，利用对数函数重写并重新计算 n 级数展开
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        
        # 返回计算得到的 n 级数展开结果
        return res

    # 将函数重写为基于对数函数的形式
    def _eval_rewrite_as_log(self, x, **kwargs):
        return (log(1 + 1/x) - log(1 - 1/x)) / 2

    # 将函数重写为基于双曲反正切函数的形式
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    # 将函数重写为基于反双曲正切函数的形式
    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return atanh(1/x)

    # 将函数重写为基于反双曲正弦函数的形式
    def _eval_rewrite_as_asinh(self, x, **kwargs):
        return (pi*I/2*(sqrt((x - 1)/x)*sqrt(x/(x - 1)) - sqrt(1 + 1/x)*sqrt(x/(x + 1))) +
                x*sqrt(1/x**2)*asinh(sqrt(1/(x**2 - 1))))

    # 返回函数的逆函数，即反双曲余切函数
    def inverse(self, argindex=1):
        """
        返回此函数的逆函数。
        """
        return coth

    # 判断函数是否是扩展实数
    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, fuzzy_or([(self.args[0] - 1).is_extended_nonnegative, (self.args[0] + 1).is_extended_nonpositive])])

    # 判断函数是否是有限的
    def _eval_is_finite(self):
        return fuzzy_not(fuzzy_or([(self.args[0] - 1).is_zero, (self.args[0] + 1).is_zero]))
class asech(InverseHyperbolicFunction):
    """
    ``asech(x)`` is the inverse hyperbolic secant of ``x``.

    The inverse hyperbolic secant function.

    Examples
    ========

    >>> from sympy import asech, sqrt, S
    >>> from sympy.abc import x
    >>> asech(x).diff(x)
    -1/(x*sqrt(1 - x**2))
    >>> asech(1).diff(x)
    0
    >>> asech(1)
    0
    >>> asech(S(2))
    I*pi/3
    >>> asech(-sqrt(2))
    3*I*pi/4
    >>> asech((sqrt(6) - sqrt(2)))
    I*pi/12

    See Also
    ========

    asinh, atanh, cosh, acoth

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    .. [2] https://dlmf.nist.gov/4.37
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSech/

    """

    # 定义导数方法，返回对于第一个参数的偏导数
    def fdiff(self, argindex=1):
        if argindex == 1:
            # 获取函数的参数
            z = self.args[0]
            # 返回函数的导数值
            return -1/(z*sqrt(1 - z**2))
        else:
            # 如果参数索引不是1，则引发参数索引错误
            raise ArgumentIndexError(self, argindex)

    # 类方法，用于计算函数在给定参数值时的值
    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            # 处理特殊情况：参数为NaN
            if arg is S.NaN:
                return S.NaN
            # 处理特殊情况：参数为正无穷或负无穷
            elif arg is S.Infinity:
                return pi*I / 2
            elif arg is S.NegativeInfinity:
                return pi*I / 2
            # 处理特殊情况：参数为零或正一
            elif arg.is_zero:
                return S.Infinity
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi*I

        # 如果参数是数字，则检查是否在常数表中
        if arg.is_number:
            cst_table = _asech_table()

            if arg in cst_table:
                # 如果参数在常数表中，则返回对应的值
                if arg.is_extended_real:
                    return cst_table[arg]*I
                return cst_table[arg]

        # 处理特殊情况：参数为复无穷
        if arg is S.ComplexInfinity:
            from sympy.calculus.accumulationbounds import AccumBounds
            return I*AccumBounds(-pi/2, pi/2)

        # 处理特殊情况：参数为零
        if arg.is_zero:
            return S.Infinity

    # 静态方法，使用缓存机制计算泰勒级数的第n项
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # 如果n为0，返回特定的对数项
        if n == 0:
            return log(2 / x)
        # 如果n为负数或奇数，返回零
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            # 处理大于2的n和之前项大于2的情况
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                # 计算泰勒级数的一般情况
                k = n // 2
                R = RisingFactorial(S.Half, k) * n
                F = factorial(k) * n // 2 * n // 2
                return -1 * R / F * x**n / 4
    def _eval_as_leading_term(self, x, logx=None, cdir=0):  # asech
        # 获取函数的参数
        arg = self.args[0]
        # 计算在 x=0 处的主导项
        x0 = arg.subs(x, 0).cancel()
        # 处理分支点
        if x0 in (-S.One, S.Zero, S.One, S.ComplexInfinity):
            # 重写表达式并计算在 x=0 处的主导项
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        if x0 is S.NaN:
            # 若 x0 为 NaN，则重新计算主导项
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # 处理位于分支切割 (-oo, 0] U (1, oo) 上的点
        if x0.is_negative or (1 - x0).is_negative:
            # 确定方向
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_positive:
                if x0.is_positive or (x0 + 1).is_negative:
                    return -self.func(x0)
                return self.func(x0) - 2*I*pi
            elif not im(ndir).is_negative:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # asech
        from sympy.series.order import O
        # 获取函数的参数
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # 处理分支点
        if arg0 is S.One:
            # 使用变量 t 进行展开
            t = Dummy('t', positive=True)
            ser = asech(S.One - t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # 无法展开
                return O(1) if n == 0 else O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            # 使用变量 t 进行展开
            t = Dummy('t', positive=True)
            ser = asech(S.NegativeOne + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # 无法展开
                return O(1) if n == 0 else I*pi + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = Function._eval_nseries(self, x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # 处理位于分支切割 (-oo, 0] U (1, oo) 上的点
        if arg0.is_negative or (1 - arg0).is_negative:
            # 确定方向
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_positive:
                if arg0.is_positive or (arg0 + 1).is_negative:
                    return -res
                return res - 2*I*pi
            elif not im(ndir).is_negative:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res
    # 返回此函数的反函数，这里直接返回 sech 函数
    def inverse(self, argindex=1):
        return sech

    # 重写函数，将其重写为以对数形式表示
    def _eval_rewrite_as_log(self, arg, **kwargs):
        return log(1/arg + sqrt(1/arg - 1) * sqrt(1/arg + 1))

    # 将 _eval_rewrite_as_log 函数作为 _eval_rewrite_as_tractable 的重写版本
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    # 将函数重写为以反双曲余弦形式表示
    def _eval_rewrite_as_acosh(self, arg, **kwargs):
        return acosh(1/arg)

    # 将函数重写为以反双曲正弦形式表示
    def _eval_rewrite_as_asinh(self, arg, **kwargs):
        return sqrt(1/arg - 1)/sqrt(1 - 1/arg)*(I*asinh(I/arg, evaluate=False)
                                                + pi*S.Half)

    # 将函数重写为以反双曲正切形式表示
    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return (I*pi*(1 - sqrt(x)*sqrt(1/x) - I/2*sqrt(-x)/sqrt(x) - I/2*sqrt(x**2)/sqrt(-x**2))
                + sqrt(1/(x + 1))*sqrt(x + 1)*atanh(sqrt(1 - x**2)))

    # 将函数重写为以反双曲余切形式表示
    def _eval_rewrite_as_acsch(self, x, **kwargs):
        return sqrt(1/x - 1)/sqrt(1 - 1/x)*(pi/2 - I*acsch(I*x, evaluate=False))

    # 检查函数是否是扩展实数
    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, self.args[0].is_nonnegative, (1 - self.args[0]).is_nonnegative])

    # 检查函数是否是有限的
    def _eval_is_finite(self):
        return fuzzy_not(self.args[0].is_zero)
class acsch(InverseHyperbolicFunction):
    """
    ``acsch(x)`` is the inverse hyperbolic cosecant of ``x``.

    The inverse hyperbolic cosecant function.

    Examples
    ========

    >>> from sympy import acsch, sqrt, I
    >>> from sympy.abc import x
    >>> acsch(x).diff(x)
    -1/(x**2*sqrt(1 + x**(-2)))
    >>> acsch(1).diff(x)
    0
    >>> acsch(1)
    log(1 + sqrt(2))
    >>> acsch(I)
    -I*pi/2
    >>> acsch(-2*I)
    I*pi/6
    >>> acsch(I*(sqrt(6) - sqrt(2)))
    -5*I*pi/12

    See Also
    ========

    asinh

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    .. [2] https://dlmf.nist.gov/4.37
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCsch/

    """

    def fdiff(self, argindex=1):
        # 求解 acsch 函数关于其参数的偏导数
        if argindex == 1:
            z = self.args[0]
            return -1/(z**2*sqrt(1 + 1/z**2))
        else:
            # 如果参数索引不为1，抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        # 对给定参数进行求值
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return S.ComplexInfinity
            elif arg is S.One:
                return log(1 + sqrt(2))
            elif arg is S.NegativeOne:
                return - log(1 + sqrt(2))

        # 如果参数是数值，则查找常数表进行计算
        if arg.is_number:
            cst_table = _acsch_table()

            if arg in cst_table:
                return cst_table[arg]*I

        # 对特定无穷大、零、负数等情况返回特定值
        if arg is S.ComplexInfinity:
            return S.Zero

        if arg.is_infinite:
            return S.Zero

        if arg.is_zero:
            return S.ComplexInfinity

        # 如果参数可能提取负号，则返回相反数的计算结果
        if arg.could_extract_minus_sign():
            return -cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # 计算泰勒级数中的指定项
        if n == 0:
            return log(2 / x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return -p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) *  n
                F = factorial(k) * n // 2 * n // 2
                return S.NegativeOne**(k +1) * R / F * x**n / 4
    # 定义一个方法 _eval_as_leading_term，用于计算表达式在给定点 x 处的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):  # acsch
        # 获取表达式的第一个参数
        arg = self.args[0]
        # 计算在 x=0 处的极限值，并化简
        x0 = arg.subs(x, 0).cancel()

        # 处理分支点
        if x0 in (-I, I, S.Zero):
            # 如果 x0 是分支点 (-i, i, 0)，则调用 rewrite 方法处理，并递归计算主导项
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        # 处理 x0 为 NaN 的情况
        if x0 is S.NaN:
            # 将表达式重新构造为其主导项
            expr = self.func(arg.as_leading_term(x))
            # 如果重新构造的表达式是有限的，则返回它，否则返回原表达式
            if expr.is_finite:
                return expr
            else:
                return self

        # 处理 x0 为 ComplexInfinity 的情况
        if x0 is S.ComplexInfinity:
            # 返回表达式的倒数的主导项
            return (1/arg).as_leading_term(x)

        # 处理落在分支切割线上的点 (-i, i)
        if x0.is_imaginary and (1 + x0**2).is_positive:
            # 计算参数 x 的方向导数
            ndir = arg.dir(x, cdir if cdir else 1)
            # 如果实部方向导数是正的
            if re(ndir).is_positive:
                # 如果虚部是正的，则返回特定的表达式
                if im(x0).is_positive:
                    return -self.func(x0) - I*pi
            # 如果实部方向导数是负的
            elif re(ndir).is_negative:
                # 如果虚部是负的，则返回特定的表达式
                if im(x0).is_negative:
                    return -self.func(x0) + I*pi
            # 如果以上条件均不满足，则调用 rewrite 方法处理，并递归计算主导项
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        # 默认情况下返回 x0 的函数形式
        return self.func(x0)
    def _eval_nseries(self, x, n, logx, cdir=0):  # acsch
        # 导入符号数列的 O 对象
        from sympy.series.order import O
        # 获取函数的参数表达式
        arg = self.args[0]
        # 计算在 x=0 处的参数表达式值
        arg0 = arg.subs(x, 0)

        # 处理分支点
        if arg0 is I:
            # 创建一个正数的虚数 Dummy 变量 t
            t = Dummy('t', positive=True)
            # 对 acsch(I + t**2) 使用对数重写并计算其 t 的 nseries 展开，精确到 2*n 阶
            ser = acsch(I + t**2).rewrite(log).nseries(t, 0, 2*n)
            # 重写参数表达式 arg1，并获取其主导项
            arg1 = -I + self.args[0]
            f = arg1.as_leading_term(x)
            # 计算 g = (arg1 - f) / f
            g = (arg1 - f) / f
            # 如果 g 在 x=0 处不是全纯的，则无法展开
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else -I*pi/2 + O(sqrt(x))
            # 计算 sqrt(S.One + g) 的 nseries 展开，再乘以 sqrt(f)，并展开结果
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO() * sqrt(f)).expand()
            # 替换 ser 中的 t 为 res，并展开并化简结果
            res = ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)
            return res

        # 处理 arg0 = -I 的情况
        if arg0 == S.NegativeOne*I:
            # 创建一个正数的虚数 Dummy 变量 t
            t = Dummy('t', positive=True)
            # 对 acsch(-I + t**2) 使用对数重写并计算其 t 的 nseries 展开，精确到 2*n 阶
            ser = acsch(-I + t**2).rewrite(log).nseries(t, 0, 2*n)
            # 重写参数表达式 arg1，并获取其主导项
            arg1 = I + self.args[0]
            f = arg1.as_leading_term(x)
            # 计算 g = (arg1 - f) / f
            g = (arg1 - f) / f
            # 如果 g 在 x=0 处不是全纯的，则无法展开
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else I*pi/2 + O(sqrt(x))
            # 计算 sqrt(S.One + g) 的 nseries 展开，再乘以 sqrt(f)，并展开结果
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO() * sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        # 如果以上条件不满足，调用父类 Function 的 _eval_nseries 方法计算结果
        res = Function._eval_nseries(self, x, n=n, logx=logx)

        # 处理 arg0 = ComplexInfinity 的情况
        if arg0 is S.ComplexInfinity:
            return res

        # 处理落在分支割线 (-I, I) 上的点
        if arg0.is_imaginary and (1 + arg0**2).is_positive:
            # 计算参数表达式 arg0 在 x, cdir 方向上的导数
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            # 如果 re(ndir) 是正数
            if re(ndir).is_positive:
                # 如果 im(arg0) 是正数，返回 -res - I*pi
                if im(arg0).is_positive:
                    return -res - I*pi
            # 如果 re(ndir) 是负数
            elif re(ndir).is_negative:
                # 如果 im(arg0) 是负数，返回 -res + I*pi
                if im(arg0).is_negative:
                    return -res + I*pi
            else:
                # 否则，重写为 log 形式，并计算其 nseries 展开结果
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        
        # 默认返回计算结果 res
        return res

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        # 返回 csch 函数作为本函数的逆函数
        return csch

    def _eval_rewrite_as_log(self, arg, **kwargs):
        # 返回该函数的对数形式重写
        return log(1/arg + sqrt(1/arg**2 + 1))

    def _eval_rewrite_as_tractable(self, arg, **kwargs):
        # 返回该函数的 log 形式重写
        return log(1/arg + sqrt(1/arg**2 + 1))

    def _eval_rewrite_as_asinh(self, arg, **kwargs):
        # 返回该函数的 asinh 形式重写
        return asinh(1/arg)

    def _eval_rewrite_as_acosh(self, arg, **kwargs):
        # 返回该函数的 acosh 形式重写
        return I*(sqrt(1 - I/arg)/sqrt(I/arg - 1)*acosh(I/arg, evaluate=False) - pi*S.Half)

    def _eval_rewrite_as_atanh(self, arg, **kwargs):
        # 计算参数的平方和加一
        arg2 = arg**2
        arg2p1 = arg2 + 1
        # 返回该函数的 atanh 形式重写
        return sqrt(-arg2)/arg*(pi*S.Half - sqrt(-arg2p1**2)/arg2p1*atanh(sqrt(arg2p1)))

    def _eval_is_zero(self):
        # 判断函数参数是否为无穷
        return self.args[0].is_infinite

    def _eval_is_extended_real(self):
        # 判断函数参数是否为扩展实数
        return self.args[0].is_extended_real
    # 定义一个方法 `_eval_is_finite`，用于判断对象是否有限
    def _eval_is_finite(self):
        # 返回调用 `fuzzy_not` 函数，对 `self.args[0].is_zero` 的结果取反
        return fuzzy_not(self.args[0].is_zero)
```