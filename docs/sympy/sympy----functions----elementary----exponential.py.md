# `D:\src\scipysrc\sympy\sympy\functions\elementary\exponential.py`

```
# 导入模块 itertools 中的 product 函数，用于生成迭代器的笛卡尔积
from itertools import product
# 导入 typing 模块中的 Tuple 别名 tTuple，用于类型提示
from typing import Tuple as tTuple

# 导入 sympy 库中的各种符号、函数和异常类
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (Function, ArgumentIndexError, expand_log,
    expand_mul, FunctionClass, PoleError, expand_multinomial, expand_complex)
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, pi, I
from sympy.core.parameters import global_parameters
from sympy.core.power import Pow
from sympy.core.relational import Ge
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory import multiplicity, perfect_power
from sympy.ntheory.factor_ import factorint

# NOTE IMPORTANT
# 这个文件中的级数展开代码是 Gruntz 算法计算极限的重要部分。
# _eval_nseries 必须返回一个系数在 C(log(x), log) 中的广义幂级数。
# 更详细地说，_eval_nseries(self, x, n) 的结果必须是：
#   c_0*x**e_0 + ... （有限多项）
# 其中 e_i 是数（不一定是整数），c_i 只涉及数值、log 和 log(x) 函数。
# 这也意味着它不能包含 log(x(1+p))，如果 x 是正数且 p 是正数，应将其展开为 log(x)+log(1+p)。

# 定义一个名为 ExpBase 的类，继承自 sympy.core.function.Function
class ExpBase(Function):

    # 标记函数为无分支的
    unbranched = True
    # 设置奇点为正无穷
    _singularities = (S.ComplexInfinity,)

    @property
    def kind(self):
        # 返回函数的类型
        return self.exp.kind

    def inverse(self, argindex=1):
        """
        返回 ``exp(x)`` 的逆函数 log。
        """
        return log

    def as_numer_denom(self):
        """
        将函数表示为正指数的分数形式，返回一个二元组 (分子, 分母)。

        Examples
        ========

        >>> from sympy import exp
        >>> from sympy.abc import x
        >>> exp(-x).as_numer_denom()
        (1, exp(x))
        >>> exp(x).as_numer_denom()
        (exp(x), 1)
        """
        # 这与 Pow.as_numer_denom 在处理指数时应该是相同的
        if not self.is_commutative:
            return self, S.One
        exp = self.exp
        neg_exp = exp.is_negative
        if not neg_exp and not (-exp).is_negative:
            neg_exp = exp.could_extract_minus_sign()
        if neg_exp:
            return S.One, self.func(-exp)
        return self, S.One

    @property
    def exp(self):
        """
        返回函数的指数部分。
        """
        return self.args[0]

    def as_base_exp(self):
        """
        返回一个二元组 (底数, 指数)。
        """
        return self.func(1), Mul(*self.args)

    def _eval_adjoint(self):
        """
        返回函数的共轭转置。
        """
        return self.func(self.exp.adjoint())
    # 返回 self.exp 的共轭
    def _eval_conjugate(self):
        return self.func(self.exp.conjugate())

    # 返回 self.exp 的转置
    def _eval_transpose(self):
        return self.func(self.exp.transpose())

    # 判断 self.exp 是否是有限的
    def _eval_is_finite(self):
        arg = self.exp
        if arg.is_infinite:
            # 如果 self.exp 是无穷大
            if arg.is_extended_negative:
                return True  # 返回 True
            if arg.is_extended_positive:
                return False  # 返回 False
        if arg.is_finite:
            return True  # 返回 True

    # 判断 self 是否是有理数
    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            z = s.exp.is_zero
            if z:
                return True  # 如果 s.exp 是零，返回 True
            elif s.exp.is_rational and fuzzy_not(z):
                return False  # 如果 s.exp 是有理数且不是零，返回 False
        else:
            return s.is_rational  # 否则，返回 s 是否是有理数

    # 判断 self.exp 是否是零
    def _eval_is_zero(self):
        return self.exp is S.NegativeInfinity  # 返回 self.exp 是否为负无穷大

    # 计算 self.exp 的指数操作
    def _eval_power(self, other):
        """exp(arg)**e -> exp(arg*e) if assumptions allow it.
        """
        b, e = self.as_base_exp()
        return Pow._eval_power(Pow(b, e, evaluate=False), other)

    # 展开 self.args[0] 的幂次表达式
    def _eval_expand_power_exp(self, **hints):
        from sympy.concrete.products import Product
        from sympy.concrete.summations import Sum
        arg = self.args[0]
        if arg.is_Add and arg.is_commutative:
            return Mul.fromiter(self.func(x) for x in arg.args)
        elif isinstance(arg, Sum) and arg.is_commutative:
            return Product(self.func(arg.function), *arg.limits)
        return self.func(arg)
# 定义一个名为 exp_polar 的类，继承自 ExpBase
class exp_polar(ExpBase):
    """
    Represent a *polar number* (see g-function Sphinx documentation).

    Explanation
    ===========

    ``exp_polar`` represents the function
    `Exp: \mathbb{C} \rightarrow \mathcal{S}`, sending the complex number
    `z = a + bi` to the polar number `r = exp(a), \theta = b`. It is one of
    the main functions to construct polar numbers.

    Examples
    ========

    >>> from sympy import exp_polar, pi, I, exp

    The main difference is that polar numbers do not "wrap around" at `2 \pi`:

    >>> exp(2*pi*I)
    1
    >>> exp_polar(2*pi*I)
    exp_polar(2*I*pi)

    apart from that they behave mostly like classical complex numbers:

    >>> exp_polar(2)*exp_polar(3)
    exp_polar(5)

    See Also
    ========

    sympy.simplify.powsimp.powsimp
    polar_lift
    periodic_argument
    principal_branch
    """

    is_polar = True  # 设置为极坐标数
    is_comparable = False  # 无法进行 evalf() 运算

    # 返回 Abs(this) 的结果，Abs 在极坐标数中不适用
    def _eval_Abs(self):
        return exp(re(self.args[0]))

    # 对极坐标数进行 evalf() 运算是不稳定的
    def _eval_evalf(self, prec):
        """ Careful! any evalf of polar numbers is flaky """
        i = im(self.args[0])
        try:
            bad = (i <= -pi or i > pi)
        except TypeError:
            bad = True
        if bad:
            return self  # 无法对此参数进行 evalf 运算
        res = exp(self.args[0])._eval_evalf(prec)
        if i > 0 and im(res) < 0:
            # i ~ pi, but exp(I*i) evaluated to argument slightly bigger than pi
            return re(res)
        return res

    # 对极坐标数进行幂运算
    def _eval_power(self, other):
        return self.func(self.args[0]*other)

    # 判断极坐标数是否是扩展实数
    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    # 返回基数和指数对
    def as_base_exp(self):
        # XXX exp_polar(0) is special!
        if self.args[0] == 0:
            return self, S.One
        return ExpBase.as_base_exp(self)


class ExpMeta(FunctionClass):
    # 检查实例是否为 exp 类或其子类的实例
    def __instancecheck__(cls, instance):
        if exp in instance.__class__.__mro__:
            return True
        return isinstance(instance, Pow) and instance.base is S.Exp1


class exp(ExpBase, metaclass=ExpMeta):
    """
    The exponential function, :math:`e^x`.

    Examples
    ========

    >>> from sympy import exp, I, pi
    >>> from sympy.abc import x
    >>> exp(x)
    exp(x)
    >>> exp(x).diff(x)
    exp(x)
    >>> exp(I*pi)
    -1

    Parameters
    ==========

    arg : Expr

    See Also
    ========

    log
    """

    # 计算该函数的第一个导数
    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return self
        else:
            raise ArgumentIndexError(self, argindex)
    # 定义一个用于评估和细化表达式的方法，基于给定的假设
    def _eval_refine(self, assumptions):
        # 导入 sympy 中的询问模块和符号 Q
        from sympy.assumptions import ask, Q
        # 获取表达式的第一个参数
        arg = self.args[0]
        # 如果参数是乘法类型
        if arg.is_Mul:
            # 设置无穷大乘以虚数单位的值
            Ioo = I * S.Infinity
            # 如果参数等于无穷大乘以虚数单位或负无穷大乘以虚数单位，返回非数值 S.NaN
            if arg in [Ioo, -Ioo]:
                return S.NaN

            # 提取参数中 pi*I 的系数
            coeff = arg.as_coefficient(pi*I)
            # 如果找到了系数
            if coeff:
                # 如果系数是 2 的整数倍
                if ask(Q.integer(2*coeff)):
                    # 如果系数是偶数
                    if ask(Q.even(coeff)):
                        return S.One
                    # 如果系数是奇数
                    elif ask(Q.odd(coeff)):
                        return S.NegativeOne
                    # 如果系数加上 1/2 是偶数
                    elif ask(Q.even(coeff + S.Half)):
                        return -I
                    # 如果系数加上 1/2 是奇数
                    elif ask(Q.odd(coeff + S.Half)):
                        return I

    @classmethod
    @property
    # 返回指数函数的基数，即常数 e
    def base(self):
        """
        Returns the base of the exponential function.
        """
        return S.Exp1

    @staticmethod
    @cacheit
    # 计算泰勒级数展开中的下一项
    def taylor_term(n, x, *previous_terms):
        """
        Calculates the next term in the Taylor series expansion.
        """
        # 如果 n 小于 0，返回零
        if n < 0:
            return S.Zero
        # 如果 n 等于 0，返回 1
        if n == 0:
            return S.One
        # 将 x 转换为 sympy 表达式
        x = sympify(x)
        # 如果有前一项，则使用前一项计算下一项
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return p * x / n
        # 否则，返回 x 的 n 次方除以 n 的阶乘
        return x**n / factorial(n)

    # 将函数表示为复数的实部和虚部
    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a 2-tuple representing a complex number.

        Examples
        ========

        >>> from sympy import exp, I
        >>> from sympy.abc import x
        >>> exp(x).as_real_imag()
        (exp(re(x))*cos(im(x)), exp(re(x))*sin(im(x)))
        >>> exp(1).as_real_imag()
        (E, 0)
        >>> exp(I).as_real_imag()
        (cos(1), sin(1))
        >>> exp(1+I).as_real_imag()
        (E*cos(1), E*sin(1))

        See Also
        ========

        sympy.functions.elementary.complexes.re
        sympy.functions.elementary.complexes.im
        """
        # 导入三角函数模块中的余弦和正弦函数
        from sympy.functions.elementary.trigonometric import cos, sin
        # 获取参数的实部和虚部
        re, im = self.args[0].as_real_imag()
        # 如果 deep 参数为 True，展开实部和虚部
        if deep:
            re = re.expand(deep, **hints)
            im = im.expand(deep, **hints)
        # 计算指数函数的实部和虚部表示
        cos, sin = cos(im), sin(im)
        return (exp(re)*cos, exp(re)*sin)

    # 对函数进行替换操作
    def _eval_subs(self, old, new):
        # 如果 old 是 Pow 类型，处理类似 (exp(3*log(x))).subs(x**2, z) -> z**(3/2) 的情况
        if old.is_Pow:
            old = exp(old.exp * log(old.base))
        # 如果 old 是 S.Exp1 并且 new 是函数类型，将 old 替换为 exp
        elif old is S.Exp1 and new.is_Function:
            old = exp
        # 如果 old 是 exp 类型或者 S.Exp1
        if isinstance(old, exp) or old is S.Exp1:
            # 定义一个 lambda 函数 f 处理基于 Pow 的参数替换
            f = lambda a: Pow(*a.as_base_exp(), evaluate=False) if (
                a.is_Pow or isinstance(a, exp)) else a
            return Pow._eval_subs(f(self), f(old), new)

        # 如果 old 是 exp 并且 new 不是函数类型，将返回 new 的 self.exp 次方替换
        if old is exp and not new.is_Function:
            return new**self.exp._subs(old, new)
        # 否则调用父类 Function 的替换方法
        return Function._eval_subs(self, old, new)
    # 检查参数是否是扩展实数
    def _eval_is_extended_real(self):
        # 如果第一个参数是扩展实数，则返回True
        if self.args[0].is_extended_real:
            return True
        # 如果第一个参数是虚数，则进行下一步处理
        elif self.args[0].is_imaginary:
            # 计算 arg2，并检查其是否为偶数
            arg2 = -S(2) * I * self.args[0] / pi
            return arg2.is_even

    # 检查表达式是否是复数
    def _eval_is_complex(self):
        # 定义一个内部函数，检查参数是否是复数或扩展负数
        def complex_extended_negative(arg):
            yield arg.is_complex
            yield arg.is_extended_negative
        # 返回对参数应用复杂扩展负数检查后的结果
        return fuzzy_or(complex_extended_negative(self.args[0]))

    # 检查表达式是否是代数的
    def _eval_is_algebraic(self):
        # 如果表达式除以 pi*I 是有理数，则返回True
        if (self.exp / pi / I).is_rational:
            return True
        # 如果表达式不是零，则进行下一步检查
        if fuzzy_not(self.exp.is_zero):
            # 如果表达式是代数的，则返回False
            if self.exp.is_algebraic:
                return False
            # 如果表达式除以 pi 是有理数，则返回False
            elif (self.exp / pi).is_rational:
                return False

    # 检查表达式是否是扩展正数
    def _eval_is_extended_positive(self):
        # 如果指数部分是扩展实数，则检查第一个参数是否不是负无穷
        if self.exp.is_extended_real:
            return self.args[0] is not S.NegativeInfinity
        # 如果指数部分是虚数，则计算 arg2，并检查其是否为偶数
        elif self.exp.is_imaginary:
            arg2 = -I * self.args[0] / pi
            return arg2.is_even

    # 对表达式进行 n-级数展开
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 请查看文件开头的重要注释
        from sympy.functions.elementary.complexes import sign
        from sympy.functions.elementary.integers import ceiling
        from sympy.series.limits import limit
        from sympy.series.order import Order
        from sympy.simplify.powsimp import powsimp
        # 获取表达式的指数部分
        arg = self.exp
        # 对指数部分进行 n-级数展开，获取展开结果
        arg_series = arg._eval_nseries(x, n=n, logx=logx)
        # 如果展开结果是 Order 类型，则返回 1 加上展开结果
        if arg_series.is_Order:
            return 1 + arg_series
        # 计算指数部分在 x 趋向于 0 时的极限
        arg0 = limit(arg_series.removeO(), x, 0)
        # 如果极限为负无穷，则返回 x 的 n 次方的 Order
        if arg0 is S.NegativeInfinity:
            return Order(x**n, x)
        # 如果极限为正无穷，则直接返回原表达式
        if arg0 is S.Infinity:
            return self
        # 如果极限为无穷，则引发 PoleError 异常
        if arg0.is_infinite:
            raise PoleError("Cannot expand %s around 0" % (self))
        # 检查 arg0 是否包含不确定性或符号项
        if any(isinstance(arg, sign) for arg in arg0.args):
            return self
        # 定义虚拟变量 t，并初始化 nterms
        t = Dummy("t")
        nterms = n
        try:
            # 获取 arg 的主导项，并计算其 Order
            cf = Order(arg.as_leading_term(x, logx=logx), x).getn()
        except (NotImplementedError, PoleError):
            cf = 0
        # 如果 cf 大于 0，则更新 nterms
        if cf and cf > 0:
            nterms = ceiling(n/cf)
        # 计算指数函数在 t 点的 nterms 级数展开
        exp_series = exp(t)._taylor(t, nterms)
        # 计算最终展开结果 r
        r = exp(arg0)*exp_series.subs(t, arg_series - arg0)
        # 替换 logx，如果存在的话
        rep = {logx: log(x)} if logx is not None else {}
        # 如果 r 经过替换后等于原表达式，则返回 r
        if r.subs(rep) == self:
            return r
        # 如果 cf 大于 1，则添加 Order 修正项；否则，添加标准 Order 修正项
        if cf and cf > 1:
            r += Order((arg_series - arg0)**n, x)/x**((cf-1)*n)
        else:
            r += Order((arg_series - arg0)**n, x)
        # 展开 r 并简化
        r = r.expand()
        r = powsimp(r, deep=True, combine='exp')
        # powsimp 可能会引入未展开的 (-1)**Rational；参见 PR #17201
        simplerat = lambda x: x.is_Rational and x.q in [3, 4, 6]
        w = Wild('w', properties=[simplerat])
        r = r.replace(S.NegativeOne**w, expand_complex(S.NegativeOne**w))
        return r
    # 定义一个方法 `_taylor`，计算对象在给定点 x 处的泰勒展开的前 n 项之和
    def _taylor(self, x, n):
        l = []  # 初始化一个空列表 l，用于存放泰勒展开的各项
        g = None  # 初始化 g 为 None
        for i in range(n):
            g = self.taylor_term(i, self.args[0], g)  # 计算第 i 项的泰勒展开，并更新 g
            g = g.nseries(x, n=n)  # 在 x 点处对 g 进行 n 阶近似
            l.append(g.removeO())  # 将 g 中的高阶无关项去除后加入列表 l
        return Add(*l)  # 返回列表 l 中所有元素的和作为泰勒展开的结果

    # 定义一个方法 `_eval_as_leading_term`，返回对象在 x 点的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.calculus.util import AccumBounds  # 导入 AccumBounds 类
        arg = self.args[0].cancel().as_leading_term(x, logx=logx)  # 获取对象参数的主导项
        arg0 = arg.subs(x, 0)  # 计算主导项在 x=0 处的值
        if arg is S.NaN:  # 如果主导项是 NaN，则返回 NaN
            return S.NaN
        if isinstance(arg0, AccumBounds):  # 如果主导项是 AccumBounds 类型
            # 处理 AccumBounds 类型的特殊情况
            # 如果 cdir 是负数，则返回 exp(-arg0)，否则返回 exp(arg0)
            if re(cdir) < S.Zero:
                return exp(-arg0)
            return exp(arg0)
        if arg0 is S.NaN:  # 如果主导项在 x=0 处的值是 NaN
            arg0 = arg.limit(x, 0)  # 计算主导项在 x=0 处的极限
        if arg0.is_infinite is False:  # 如果主导项在 x=0 处不是无穷大
            return exp(arg0)  # 返回 exp(arg0)
        # 如果以上条件都不满足，则抛出 PoleError 异常
        raise PoleError("Cannot expand %s around 0" % (self))

    # 定义一个方法 `_eval_rewrite_as_sin`，返回对象的一个 sin 函数的重写形式
    def _eval_rewrite_as_sin(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import sin  # 导入 sin 函数
        return sin(I*arg + pi/2) - I*sin(I*arg)  # 返回 sin(I*arg + pi/2) - I*sin(I*arg)

    # 定义一个方法 `_eval_rewrite_as_cos`，返回对象的一个 cos 函数的重写形式
    def _eval_rewrite_as_cos(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import cos  # 导入 cos 函数
        return cos(I*arg) + I*cos(I*arg + pi/2)  # 返回 cos(I*arg) + I*cos(I*arg + pi/2)

    # 定义一个方法 `_eval_rewrite_as_tanh`，返回对象的一个 tanh 函数的重写形式
    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        from sympy.functions.elementary.hyperbolic import tanh  # 导入 tanh 函数
        return (1 + tanh(arg/2))/(1 - tanh(arg/2))  # 返回 (1 + tanh(arg/2))/(1 - tanh(arg/2))

    # 定义一个方法 `_eval_rewrite_as_sqrt`，返回对象的一个 sqrt 函数的重写形式
    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import sin, cos  # 导入 sin 和 cos 函数
        if arg.is_Mul:  # 如果参数是乘法类型
            coeff = arg.coeff(pi*I)  # 获取参数中 pi*I 的系数
            if coeff and coeff.is_number:  # 如果系数存在且是数字类型
                cosine, sine = cos(pi*coeff), sin(pi*coeff)  # 计算 cos(pi*coeff) 和 sin(pi*coeff)
                if not isinstance(cosine, cos) and not isinstance(sine, sin):
                    return cosine + I*sine  # 返回 cos(pi*coeff) + I*sin(pi*coeff)

    # 定义一个方法 `_eval_rewrite_as_Pow`，返回对象的一个 Pow 函数的重写形式
    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if arg.is_Mul:  # 如果参数是乘法类型
            logs = [a for a in arg.args if isinstance(a, log) and len(a.args) == 1]  # 找出参数中的对数项
            if logs:  # 如果存在对数项
                return Pow(logs[0].args[0], arg.coeff(logs[0]))  # 返回对数项的幂
def match_real_imag(expr):
    r"""
    尝试将表达式与 $a + Ib$ 形式匹配，其中 $a$ 和 $b$ 是实数。

    ``match_real_imag`` 返回一个包含表达式的实部和虚部的元组，如果无法直接匹配，则返回 ``(None, None)``。与 :func:`~.re()`、:func:`~.im()` 和 ``as_real_imag()`` 不同，这个辅助函数不会通过返回包含 ``re()`` 或 ``im()`` 的表达式来强制处理，并且也不会展开其参数。

    """
    # 将表达式分解为独立的实部和虚部
    r_, i_ = expr.as_independent(I, as_Add=True)
    # 如果虚部为零且实部是实数
    if i_ == 0 and r_.is_real:
        return (r_, i_)
    # 尝试将虚部系数提取为虚数单位 I
    i_ = i_.as_coefficient(I)
    # 如果虚部不为零且是实数，同时实部也是实数
    if i_ and i_.is_real and r_.is_real:
        return (r_, i_)
    else:
        return (None, None)  # 比检查是否为 None 更简单


class log(Function):
    r"""
    自然对数函数 `\ln(x)` 或 `\log(x)`。

    Explanation
    ===========

    对数函数使用自然底数 `e`。要得到以不同底数 ``b`` 的对数，使用 ``log(x, b)``，这相当于 ``log(x)/log(b)``。

    ``log`` 表示自然对数的主支。因此，在负实轴上有一个分支切割，并返回复数参数在 `(-\pi, \pi]` 内的值。

    Examples
    ========

    >>> from sympy import log, sqrt, S, I
    >>> log(8, 2)
    3
    >>> log(S(8)/3, 2)
    -log(3)/log(2) + 3
    >>> log(-1 + I*sqrt(3))
    log(2) + 2*I*pi/3

    See Also
    ========

    exp

    """

    args: tTuple[Expr]

    _singularities = (S.Zero, S.ComplexInfinity)

    def fdiff(self, argindex=1):
        """
        返回函数的一阶导数。
        """
        if argindex == 1:
            return 1/self.args[0]
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        r"""
        返回 `\log(x)` 的反函数 `e^x`。
        """
        return exp

    @classmethod
    def as_base_exp(self):
        """
        将此函数返回为 (底数, 指数) 的形式。
        """
        return self, S.One

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):  # of log(1+x)
        r"""
        返回 `\log(1+x)` 泰勒级数展开的下一个项。
        """
        from sympy.simplify.powsimp import powsimp
        if n < 0:
            return S.Zero
        x = sympify(x)
        if n == 0:
            return x
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return powsimp((-n) * p * x / (n + 1), deep=True, combine='exp')
        return (1 - 2*(n % 2)) * x**(n + 1)/(n + 1)
    # 对对数函数表达式进行展开评估，可选是否深度展开和其他提示参数
    def _eval_expand_log(self, deep=True, **hints):
        # 导入 Sum 和 Product 类，用于处理展开中的求和和乘积
        from sympy.concrete import Sum, Product
        # 从提示参数中获取 force 和 factor，默认为 False
        force = hints.get('force', False)
        factor = hints.get('factor', False)
        
        # 如果表达式中有两个参数
        if (len(self.args) == 2):
            # 返回展开后的对数表达式，根据 deep 和 force 参数决定是否深度展开
            return expand_log(self.func(*self.args), deep=deep, force=force)
        
        # 取第一个参数
        arg = self.args[0]
        
        # 如果参数是整数
        if arg.is_Integer:
            # 移除完全幂
            p = perfect_power(arg)
            logarg = None
            coeff = 1
            if p is not False:
                arg, coeff = p
                logarg = self.func(arg)
            
            # 如果 factor=True，将参数展开为其质因数的乘积
            if factor:
                p = factorint(arg)
                if arg not in p.keys():
                    logarg = sum(n*log(val) for val, n in p.items())
            
            # 如果 logarg 不为空，返回系数乘以对数参数
            if logarg is not None:
                return coeff*logarg
        
        # 如果参数是有理数
        elif arg.is_Rational:
            # 返回有理数参数的对数
            return log(arg.p) - log(arg.q)
        
        # 如果参数是乘积表达式
        elif arg.is_Mul:
            expr = []
            nonpos = []
            for x in arg.args:
                # 如果强制展开或者 x 是正数或极坐标
                if force or x.is_positive or x.is_polar:
                    a = self.func(x)
                    # 如果是对数函数，将其展开
                    if isinstance(a, log):
                        expr.append(self.func(x)._eval_expand_log(**hints))
                    else:
                        expr.append(a)
                # 如果 x 是负数
                elif x.is_negative:
                    a = self.func(-x)
                    expr.append(a)
                    nonpos.append(S.NegativeOne)
                else:
                    nonpos.append(x)
            # 返回乘积表达式的对数展开结果
            return Add(*expr) + log(Mul(*nonpos))
        
        # 如果参数是幂函数或指数函数
        elif arg.is_Pow or isinstance(arg, exp):
            # 如果强制展开或者底数是正数或极坐标，或者指数+1是正数且指数-1是非正数
            if force or (arg.exp.is_extended_real and (arg.base.is_positive or ((arg.exp+1)
                .is_positive and (arg.exp-1).is_nonpositive))) or arg.base.is_polar:
                b = arg.base
                e = arg.exp
                a = self.func(b)
                # 如果是对数函数，将指数展开后与底数对数相乘
                if isinstance(a, log):
                    return unpolarify(e) * a._eval_expand_log(**hints)
                else:
                    return unpolarify(e) * a
        
        # 如果参数是乘积表达式
        elif isinstance(arg, Product):
            # 如果强制展开或者乘积函数是正数
            if force or arg.function.is_positive:
                # 返回乘积函数的对数和
                return Sum(log(arg.function), *arg.limits)
        
        # 返回原始的对数表达式
        return self.func(arg)

    # 对对数表达式进行简化评估
    def _eval_simplify(self, **kwargs):
        # 导入简化相关的函数：expand_log, simplify, inversecombine
        from sympy.simplify.simplify import expand_log, simplify, inversecombine
        
        # 如果表达式有两个参数，说明它还未被评估
        if len(self.args) == 2:
            # 返回简化后的表达式
            return simplify(self.func(*self.args), **kwargs)
        
        # 对第一个参数进行简化
        expr = self.func(simplify(self.args[0], **kwargs))
        
        # 如果 kwargs 中指定需要反向操作
        if kwargs['inverse']:
            # 对表达式进行反向组合操作
            expr = inversecombine(expr)
        
        # 对简化后的表达式进行对数展开，深度为 True
        expr = expand_log(expr, deep=True)
        
        # 返回简化后的表达式和原始表达式中更小的那个，依据 kwargs 中的度量方法
        return min([expr, self], key=kwargs['measure'])
    # 返回该函数的实部和虚部作为复坐标表示
    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a complex coordinate.

        Examples
        ========

        >>> from sympy import I, log
        >>> from sympy.abc import x
        >>> log(x).as_real_imag()
        (log(Abs(x)), arg(x))
        >>> log(I).as_real_imag()
        (0, pi/2)
        >>> log(1 + I).as_real_imag()
        (log(sqrt(2)), pi/4)
        >>> log(I*x).as_real_imag()
        (log(Abs(x)), arg(I*x))

        """
        # 获取函数的第一个参数
        sarg = self.args[0]
        # 如果 deep 参数为 True，则展开函数的第一个参数
        if deep:
            sarg = self.args[0].expand(deep, **hints)
        # 计算函数第一个参数的绝对值
        sarg_abs = Abs(sarg)
        # 如果函数的第一个参数等于其绝对值，则返回函数本身和零
        if sarg_abs == sarg:
            return self, S.Zero
        # 否则计算函数第一个参数的辐角
        sarg_arg = arg(sarg)
        # 如果 hints 中有 'log' 键且其值为 True，则展开对数函数
        if hints.get('log', False):  # Expand the log
            hints['complex'] = False
            return (log(sarg_abs).expand(deep, **hints), sarg_arg)
        else:
            return log(sarg_abs), sarg_arg

    # 判断函数是否为有理数
    def _eval_is_rational(self):
        # 创建函数的实例
        s = self.func(*self.args)
        # 如果函数的类型与自身相同
        if s.func == self.func:
            # 如果函数的第一个参数减去1为零，则返回 True
            if (self.args[0] - 1).is_zero:
                return True
            # 如果函数的第一个参数为有理数且不等于1，则返回 False
            if s.args[0].is_rational and fuzzy_not((self.args[0] - 1).is_zero):
                return False
        else:
            return s.is_rational

    # 判断函数是否为代数数
    def _eval_is_algebraic(self):
        # 创建函数的实例
        s = self.func(*self.args)
        # 如果函数的类型与自身相同
        if s.func == self.func:
            # 如果函数的第一个参数减去1为零，则返回 True
            if (self.args[0] - 1).is_zero:
                return True
            # 如果函数的第一个参数不等于1且不为代数数，则返回 False
            elif fuzzy_not((self.args[0] - 1).is_zero):
                if self.args[0].is_algebraic:
                    return False
        else:
            return s.is_algebraic

    # 判断函数的第一个参数是否为扩展实数
    def _eval_is_extended_real(self):
        return self.args[0].is_extended_positive

    # 判断函数的第一个参数是否为复数
    def _eval_is_complex(self):
        z = self.args[0]
        return fuzzy_and([z.is_complex, fuzzy_not(z.is_zero)])

    # 判断函数的第一个参数是否为有限数
    def _eval_is_finite(self):
        arg = self.args[0]
        # 如果函数的第一个参数为零，则返回 False
        if arg.is_zero:
            return False
        # 否则返回函数的第一个参数是否为有限数
        return arg.is_finite

    # 判断函数的第一个参数是否为扩展正数
    def _eval_is_extended_positive(self):
        return (self.args[0] - 1).is_extended_positive

    # 判断函数的第一个参数是否为零
    def _eval_is_zero(self):
        return (self.args[0] - 1).is_zero

    # 判断函数的第一个参数是否为扩展非负数
    def _eval_is_extended_nonnegative(self):
        return (self.args[0] - 1).is_extended_nonnegative
    # NOTE
    # 有关此方法中涉及的每个步骤的更多信息，请参阅 https://github.com/sympy/sympy/pull/23592

    # STEP 1
    # 将参数列表中的第一个表达式合并到一个整体表达式中
    arg0 = self.args[0].together()

    # STEP 2
    # 创建一个正数虚拟变量 t
    t = Dummy('t', positive=True)
    # 如果方向 cdir 为 0，则设为 1
    if cdir == 0:
        cdir = 1
    # 将参数表达式 arg0 中的 x 替换为 cdir*t，得到 z
    z = arg0.subs(x, cdir*t)

    # STEP 3
    try:
        # 获取 z 相对于 t 的主导项系数和指数
        c, e = z.leadterm(t, logx=logx, cdir=1)
    except ValueError:
        # 如果无法计算主导项，则返回参数表达式 arg0 的对数
        arg = arg0.as_leading_term(x, logx=logx, cdir=cdir)
        return log(arg)
    # 如果主导项系数 c 中包含 t，则将 t 替换为 x/cdir
    if c.has(t):
        c = c.subs(t, x/cdir)
        # 如果指数 e 不为零，则抛出极点错误
        if e != 0:
            raise PoleError("Cannot expand %s around 0" % (self))
        # 返回 c 的对数
        return log(c)

    # STEP 4
    # 如果 c 是 1，且 e 是 0
    if c == S.One and e == S.Zero:
        # 返回 arg0 - 1 的主导项的对数
        return (arg0 - S.One).as_leading_term(x, logx=logx)

    # STEP 5
    # 计算 res = log(c) - e*log(cdir)
    res = log(c) - e*log(cdir)
    # 如果 logx 未指定，则设为 log(x)
    logx = log(x) if logx is None else logx
    # 加上 e*logx
    res += e*logx

    # 如果 c 是负数且 z 的虚部不为 0
    if c.is_negative and im(z) != 0:
        # 导入 Heaviside 函数
        from sympy.functions.special.delta_functions import Heaviside
        # 对 z 的级数展开进行迭代，找到前五个非实数项
        for i, term in enumerate(z.lseries(t)):
            if not term.is_real or i == 5:
                break
        # 如果找到的非实数项个数小于 5
        if i < 5:
            # 获取该项的系数和指数
            coeff, _ = term.as_coeff_exponent(t)
            # 根据 coeff 的虚部是否小于 0，添加额外的 -2*I*pi*Heaviside(-im(coeff), 0)
            res += -2*I*pi*Heaviside(-im(coeff), 0)

    # 返回最终结果 res
    return res
class LambertW(Function):
    r"""
    The Lambert W function $W(z)$ is defined as the inverse
    function of $w \exp(w)$ [1]_.

    Explanation
    ===========

    In other words, the value of $W(z)$ is such that $z = W(z) \exp(W(z))$
    for any complex number $z$.  The Lambert W function is a multivalued
    function with infinitely many branches $W_k(z)$, indexed by
    $k \in \mathbb{Z}$.  Each branch gives a different solution $w$
    of the equation $z = w \exp(w)$.

    The Lambert W function has two partially real branches: the
    principal branch ($k = 0$) is real for real $z > -1/e$, and the
    $k = -1$ branch is real for $-1/e < z < 0$. All branches except
    $k = 0$ have a logarithmic singularity at $z = 0$.

    Examples
    ========

    >>> from sympy import LambertW
    >>> LambertW(1.2)
    0.635564016364870
    >>> LambertW(1.2, -1).n()
    -1.34747534407696 - 4.41624341514535*I
    >>> LambertW(-1).is_real
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lambert_W_function
    """
    _singularities = (-Pow(S.Exp1, -1, evaluate=False), S.ComplexInfinity)

    @classmethod
    # 定义类方法 eval，用于计算 Lambert W 函数的值
    def eval(cls, x, k=None):
        if k == S.Zero:
            # 如果 k 为主分支 (k = 0)，返回主分支的 Lambert W 函数计算结果
            return cls(x)
        elif k is None:
            k = S.Zero

        if k.is_zero:
            if x.is_zero:
                # 当 k = 0 且 x = 0 时，返回 0
                return S.Zero
            if x is S.Exp1:
                # 当 k = 0 且 x = e 时，返回 1
                return S.One
            w = Wild('w')
            # 对于 x = W(x*log(x)) 的情况，当 x >= 1/e 时，返回 log(x)
            # 例如，W(-1/e) = -1, W(2*log(2)) = log(2)
            result = x.match(w*log(w))
            if result is not None and Ge(result[w]*S.Exp1, S.One) is S.true:
                return log(result[w])
            if x == -log(2)/2:
                # 当 x = -log(2)/2 时，返回 -log(2)
                return -log(2)
            # 对于 x = W(x**(x+1)*log(x)) 的情况，当 x > 0 时，返回 x*log(x)
            # 例如，W(81*log(3)) = 3*log(3)
            result = x.match(w**(w+1)*log(w))
            if result is not None and result[w].is_positive is True:
                return result[w]*log(result[w])
            # 对于 x = W(e**(1/n)/n) 的情况，返回 1/n
            # 例如，W(sqrt(e)/2) = 1/2
            result = x.match(S.Exp1**(1/w)/w)
            if result is not None:
                return 1 / result[w]
            if x == -pi/2:
                # 当 x = -pi/2 时，返回 -I*pi/2
                return I*pi/2
            if x == exp(1 + S.Exp1):
                # 当 x = e^(1+e) 时，返回 e
                return S.Exp1
            if x is S.Infinity:
                # 当 x = 无穷大 时，返回 无穷大
                return S.Infinity

        if fuzzy_not(k.is_zero):
            if x.is_zero:
                # 当 k 不为 0 且 x = 0 时，返回 负无穷大
                return S.NegativeInfinity
        if k is S.NegativeOne:
            if x == -pi/2:
                # 当 k = -1 且 x = -pi/2 时，返回 -I*pi/2
                return -I*pi/2
            elif x == -1/S.Exp1:
                # 当 k = -1 且 x = -1/e 时，返回 -1
                return S.NegativeOne
            elif x == -2*exp(-2):
                # 当 k = -1 且 x = -2*exp(-2) 时，返回 -2
                return -Integer(2)
    # 计算函数的第一阶导数
    def fdiff(self, argindex=1):
        # 获取函数的第一个参数
        x = self.args[0]

        # 如果函数只有一个参数
        if len(self.args) == 1:
            # 如果要求第一个参数的导数
            if argindex == 1:
                return LambertW(x)/(x*(1 + LambertW(x)))
        else:
            # 否则获取第二个参数
            k = self.args[1]
            # 如果要求第一个参数的导数
            if argindex == 1:
                return LambertW(x, k)/(x*(1 + LambertW(x, k)))

        # 如果参数索引不合法，则引发异常
        raise ArgumentIndexError(self, argindex)

    # 判断函数是否是扩展实数
    def _eval_is_extended_real(self):
        # 获取函数的第一个参数
        x = self.args[0]
        # 如果函数只有一个参数，则第二个参数默认为零
        if len(self.args) == 1:
            k = S.Zero
        else:
            k = self.args[1]
        # 如果第二个参数为零
        if k.is_zero:
            # 判断 x + 1/e 是否为正数或非正数
            if (x + 1/S.Exp1).is_positive:
                return True
            elif (x + 1/S.Exp1).is_nonpositive:
                return False
        # 如果第二个参数加一为零
        elif (k + 1).is_zero:
            # 如果 x 为负数且 x + 1/e 为正数，则函数为扩展实数
            if x.is_negative and (x + 1/S.Exp1).is_positive:
                return True
            # 如果 x 非正数或者 x + 1/e 非负数，则函数不是扩展实数
            elif x.is_nonpositive or (x + 1/S.Exp1).is_nonnegative:
                return False
        # 如果第二个参数不为零且不接近零
        elif fuzzy_not(k.is_zero) and fuzzy_not((k + 1).is_zero):
            # 如果 x 是扩展实数，则函数不是扩展实数
            if x.is_extended_real:
                return False

    # 判断函数是否有限
    def _eval_is_finite(self):
        # 判断函数的第一个参数是否是有限的
        return self.args[0].is_finite

    # 判断函数是否是代数的
    def _eval_is_algebraic(self):
        # 构建函数的实例
        s = self.func(*self.args)
        # 如果构建的实例和原函数相同
        if s.func == self.func:
            # 如果第一个参数不为零且是代数的，则函数不是代数的
            if fuzzy_not(self.args[0].is_zero) and self.args[0].is_algebraic:
                return False
        else:
            # 否则判断构建的实例是否是代数的
            return s.is_algebraic

    # 计算函数在 leading term 下的表达式
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 如果函数只有一个参数
        if len(self.args) == 1:
            # 获取参数关于 x 的极限
            arg = self.args[0]
            arg0 = arg.subs(x, 0).cancel()
            # 如果极限不为零，则返回其 leading term
            if not arg0.is_zero:
                return self.func(arg0)
            # 否则返回参数关于 x 的 leading term
            return arg.as_leading_term(x)

    # 计算函数在 nseries 展开下的表达式
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 如果函数只有一个参数
        if len(self.args) == 1:
            # 导入所需的模块
            from sympy.functions.elementary.integers import ceiling
            from sympy.series.order import Order
            # 获取参数的 nseries 展开
            arg = self.args[0].nseries(x, n=n, logx=logx)
            # 获取 leading term
            lt = arg.as_leading_term(x, logx=logx)
            lte = 1
            if lt.is_Pow:
                lte = lt.exp
            # 计算级数展开
            if ceiling(n/lte) >= 1:
                s = Add(*[(-S.One)**(k - 1)*Integer(k)**(k - 2)/
                          factorial(k - 1)*arg**k for k in range(1, ceiling(n/lte))])
                s = expand_multinomial(s)
            else:
                s = S.Zero

            return s + Order(x**n, x)
        # 否则调用父类的 _eval_nseries 方法
        return super()._eval_nseries(x, n, logx)

    # 判断函数是否为零
    def _eval_is_zero(self):
        # 获取函数的第一个参数
        x = self.args[0]
        # 如果函数只有一个参数，则判断这个参数是否为零
        if len(self.args) == 1:
            return x.is_zero
        else:
            # 否则判断第一个参数和第二个参数是否同时为零
            return fuzzy_and([x.is_zero, self.args[1].is_zero])
# 使用装饰器 @cacheit 对 _log_atan_table 函数进行缓存，以提高性能
@cacheit
# 返回一个字典，其中包含一些特定角度的正切值的对数的近似值
def _log_atan_table():
    return {
        # 第一象限的角度对应的近似值
        sqrt(3): pi / 3,
        1: pi / 4,
        sqrt(5 - 2 * sqrt(5)): pi / 5,
        sqrt(2) * sqrt(5 - sqrt(5)) / (1 + sqrt(5)): pi / 5,
        sqrt(5 + 2 * sqrt(5)): pi * Rational(2, 5),
        sqrt(2) * sqrt(sqrt(5) + 5) / (-1 + sqrt(5)): pi * Rational(2, 5),
        sqrt(3) / 3: pi / 6,
        sqrt(2) - 1: pi / 8,
        sqrt(2 - sqrt(2)) / sqrt(sqrt(2) + 2): pi / 8,
        sqrt(2) + 1: pi * Rational(3, 8),
        sqrt(sqrt(2) + 2) / sqrt(2 - sqrt(2)): pi * Rational(3, 8),
        sqrt(1 - 2 * sqrt(5) / 5): pi / 10,
        (-sqrt(2) + sqrt(10)) / (2 * sqrt(sqrt(5) + 5)): pi / 10,
        sqrt(1 + 2 * sqrt(5) / 5): pi * Rational(3, 10),
        (sqrt(2) + sqrt(10)) / (2 * sqrt(5 - sqrt(5))): pi * Rational(3, 10),
        2 - sqrt(3): pi / 12,
        (-1 + sqrt(3)) / (1 + sqrt(3)): pi / 12,
        2 + sqrt(3): pi * Rational(5, 12),
        (1 + sqrt(3)) / (-1 + sqrt(3)): pi * Rational(5, 12)
    }
```