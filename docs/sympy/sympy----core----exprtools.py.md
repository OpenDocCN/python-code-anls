# `D:\src\scipysrc\sympy\sympy\core\exprtools.py`

```
"""Tools for manipulating of large commutative expressions. """

# 导入需要的模块和类
from .add import Add
from .mul import Mul, _keep_coeff
from .power import Pow
from .basic import Basic
from .expr import Expr
from .function import expand_power_exp
from .sympify import sympify
from .numbers import Rational, Integer, Number, I, equal_valued
from .singleton import S
from .sorting import default_sort_key, ordered
from .symbol import Dummy
from .traversal import preorder_traversal
from .coreerrors import NonCommutativeExpression
from .containers import Tuple, Dict
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import (common_prefix, common_suffix,
        variations, iterable, is_sequence)

# 导入标准库中的模块
from collections import defaultdict
from typing import Tuple as tTuple

# 创建一个名为_eps的符号变量，用于表示正数
_eps = Dummy(positive=True)


def _isnumber(i):
    """判断输入是否为数值类型（整数、浮点数或者SymPy中定义的数值类型）"""
    return isinstance(i, (SYMPY_INTS, float)) or i.is_Number


def _monotonic_sign(self):
    """返回 self 可能具有的最接近 0 的值，假设所有符号都有符号，
    结果对所有符号值都是同一符号。如果符号只有符号但未知整数或结果为0，则返回符号的代表值self。
    如果自变量有可能为正数或者负数，或者self不属于以下几种情况，返回None：

    - L(x, y, ...) + A：对所有符号 x, y, ... 线性函数与一个常数 A；如果 A 为零，则函数可以是变量范围内单调的单项式，例如 (x + 1)**3 如果 x 是非负数。
    - A/L(x, y, ...) + B：所有符号 x, y, ... 线性函数的倒数，不会因变量集的任何值而从正数变为负数。
    - M(x, y, ...) + A：所有因子都有符号和常数 A 的单项式 M。
    - A/M(x, y, ...) + B：单项式的倒数和常数 A 和 B。
    - P(x)：单变量多项式

    返回示例
    ========

    >>> from sympy.core.exprtools import _monotonic_sign as F
    >>> from sympy import Dummy
    >>> nn = Dummy(integer=True, nonnegative=True)
    >>> p = Dummy(integer=True, positive=True)
    >>> p2 = Dummy(integer=True, positive=True)
    >>> F(nn + 1)
    1
    >>> F(p - 1)
    _nneg
    >>> F(nn*p + 1)
    1
    >>> F(p2*p + 1)
    2
    >>> F(nn - 1)  # 可能是负数、零或正数
    """
    # 如果 self 不是扩展实数，则返回空
    if not self.is_extended_real:
        return

    # 如果 -self 是符号，则递归调用 _monotonic_sign(-self)
    if (-self).is_Symbol:
        rv = _monotonic_sign(-self)
        return rv if rv is None else -rv
    # 如果不是添加操作并且分母为数字
    if not self.is_Add and self.as_numer_denom()[1].is_number:
        # 将当前对象赋给变量 s
        s = self
        # 如果 s 是素数
        if s.is_prime:
            # 如果 s 是奇数，返回整数 3
            if s.is_odd:
                return Integer(3)
            else:
                # 如果 s 是偶数，返回整数 2
                return Integer(2)
        # 如果 s 是合数
        elif s.is_composite:
            # 如果 s 是奇数，返回整数 9
            if s.is_odd:
                return Integer(9)
            else:
                # 如果 s 是偶数，返回整数 4
                return Integer(4)
        # 如果 s 是正数
        elif s.is_positive:
            # 如果 s 是偶数
            if s.is_even:
                # 如果 s 不是素数，返回整数 4
                if s.is_prime is False:
                    return Integer(4)
                else:
                    # 如果 s 是素数，返回整数 2
                    return Integer(2)
            # 如果 s 是整数，返回 1
            elif s.is_integer:
                return S.One
            else:
                # 否则返回 _eps
                return _eps
        # 如果 s 是扩展负数
        elif s.is_extended_negative:
            # 如果 s 是偶数，返回整数 -2
            if s.is_even:
                return Integer(-2)
            elif s.is_integer:
                # 如果 s 是整数，返回 -1
                return S.NegativeOne
            else:
                # 否则返回 -_eps
                return -_eps
        # 如果 s 是零或者扩展非正或非负数
        if s.is_zero or s.is_extended_nonpositive or s.is_extended_nonnegative:
            return S.Zero
        # 如果以上条件都不满足，返回 None
        return None

    # 对于一元多项式，获取其中的自由符号
    free = self.free_symbols
    # 如果自由变量集合只有一个变量
    if len(free) == 1:
        # 如果表达式是多项式
        if self.is_polynomial():
            # 导入必要的库函数
            from sympy.polys.polytools import real_roots
            from sympy.polys.polyroots import roots
            from sympy.polys.polyerrors import PolynomialError
            
            # 从自由变量集合中取出唯一的变量，并计算其单调性的符号
            x = free.pop()
            x0 = _monotonic_sign(x)
            # 如果单调性符号接近零，则设置为零
            if x0 in (_eps, -_eps):
                x0 = S.Zero
            
            # 如果能够计算导数
            if x0 is not None:
                d = self.diff(x)
                # 如果导数是一个数值
                if d.is_number:
                    currentroots = []
                else:
                    try:
                        # 尝试计算导数的实根
                        currentroots = real_roots(d)
                    except (PolynomialError, NotImplementedError):
                        # 如果计算失败，则仅选择扩展实数根
                        currentroots = [r for r in roots(d, x) if r.is_extended_real]
                
                # 计算在 x=x0 处的表达式值
                y = self.subs(x, x0)
                
                # 如果 x 是非负数，并且当前所有根与 x0 的差都是非正数
                if x.is_nonnegative and all(
                        (r - x0).is_nonpositive for r in currentroots):
                    # 如果 y 是非负数，并且导数是正数
                    if y.is_nonnegative and d.is_positive:
                        # 如果 y 存在，则返回 y 如果为正数，否则返回一个标记为正的 Dummy 对象
                        if y:
                            return y if y.is_positive else Dummy('pos', positive=True)
                        else:
                            return Dummy('nneg', nonnegative=True)
                    # 如果 y 是非正数，并且导数是负数
                    if y.is_nonpositive and d.is_negative:
                        # 如果 y 存在，则返回 y 如果为负数，否则返回一个标记为负的 Dummy 对象
                        if y:
                            return y if y.is_negative else Dummy('neg', negative=True)
                        else:
                            return Dummy('npos', nonpositive=True)
                
                # 如果 x 是非正数，并且当前所有根与 x0 的差都是非负数
                elif x.is_nonpositive and all(
                        (r - x0).is_nonnegative for r in currentroots):
                    # 如果 y 是非负数，并且导数是负数
                    if y.is_nonnegative and d.is_negative:
                        # 如果 y 存在，则返回一个标记为正的 Dummy 对象
                        if y:
                            return Dummy('pos', positive=True)
                        else:
                            return Dummy('nneg', nonnegative=True)
                    # 如果 y 是非正数，并且导数是正数
                    if y.is_nonpositive and d.is_positive:
                        # 如果 y 存在，则返回一个标记为负的 Dummy 对象
                        if y:
                            return Dummy('neg', negative=True)
                        else:
                            return Dummy('npos', nonpositive=True)
        
        # 如果表达式不是多项式
        else:
            # 将表达式分解为分子和分母
            n, d = self.as_numer_denom()
            den = None
            # 如果分子是一个数值，则尝试计算分母的单调性符号
            if n.is_number:
                den = _monotonic_sign(d)
            # 如果分子不是一个数值，并且分子的单调性符号不为 None
            elif not d.is_number:
                if _monotonic_sign(n) is not None:
                    den = _monotonic_sign(d)
            
            # 如果成功计算出分母的单调性符号，并且其为正数或负数
            if den is not None and (den.is_positive or den.is_negative):
                # 计算表达式 n * den 的值
                v = n * den
                # 根据不同的值返回对应的 Dummy 对象
                if v.is_positive:
                    return Dummy('pos', positive=True)
                elif v.is_nonnegative:
                    return Dummy('nneg', nonnegative=True)
                elif v.is_negative:
                    return Dummy('neg', negative=True)
                elif v.is_nonpositive:
                    return Dummy('npos', nonpositive=True)
        
        # 如果以上条件都不满足，则返回 None
        return None
    
    # 如果自由变量集合中不止一个变量，则处理多变量情况
    # 以及下一行代码的处理
    c, a = self.as_coeff_Add()
    v = None
    # 如果表达式 a 不是多项式，则按以下规则处理：
    if not a.is_polynomial():
        # 分解 a 为分子 n 和分母 d
        n, d = a.as_numer_denom()
        # 如果 n 和 d 都不是数字，则直接返回
        if not (n.is_number or d.is_number):
            return
        # 如果 a 是乘积或者幂，并且是有理数，并且所有幂的指数都是整数，并且 a 是正数或负数
        if (
                a.is_Mul or a.is_Pow) and \
                a.is_rational and \
                all(p.exp.is_Integer for p in a.atoms(Pow) if p.is_Pow) and \
                (a.is_positive or a.is_negative):
            # 初始化 v 为单位元素
            v = S.One
            # 遍历 a 的每个乘积项或幂项
            for ai in Mul.make_args(a):
                # 如果 ai 是数字，则累乘到 v 中
                if ai.is_number:
                    v *= ai
                    continue
                # 初始化替换字典 reps
                reps = {}
                # 对 ai 中的每个自由符号 x，计算其单调性符号
                for x in ai.free_symbols:
                    reps[x] = _monotonic_sign(x)
                    # 如果单调性未知，则返回
                    if reps[x] is None:
                        return
                # 将 ai 中的自由符号替换为计算得到的单调性符号后，累乘到 v 中
                v *= ai.subs(reps)
    # 如果表达式 a 是多项式，并且 c 存在
    elif c:
        # 如果 a 中不存在非数字的幂，并且 a 是非正数或非负数
        if not any(p for p in a.atoms(Pow) if not p.is_number) and (a.is_nonpositive or a.is_nonnegative):
            # 获取 a 中的自由符号列表
            free = list(a.free_symbols)
            # 初始化替换字典 p
            p = {}
            # 对于每个自由符号 i
            for i in free:
                # 计算 i 的单调性符号
                v = _monotonic_sign(i)
                # 如果单调性未知，则返回
                if v is None:
                    return
                # 如果 v 为 None，则设为 _eps（如果 i 是非负数），否则设为 -_eps
                p[i] = v or (_eps if i.is_nonnegative else -_eps)
            # 将 a 中的自由符号用 p 中的值替换
            v = a.xreplace(p)
    # 如果 v 不为 None，则计算 rv，并根据条件返回结果
    if v is not None:
        rv = v + c
        # 如果 v 是非负数并且 rv 是正数，则将 rv 中的 _eps 替换为 0 后返回
        if v.is_nonnegative and rv.is_positive:
            return rv.subs(_eps, 0)
        # 如果 v 是非正数并且 rv 是负数，则将 rv 中的 _eps 替换为 0 后返回
        if v.is_nonpositive and rv.is_negative:
            return rv.subs(_eps, 0)
def decompose_power(expr: Expr) -> tTuple[Expr, int]:
    """
    Decompose power into symbolic base and integer exponent.

    Examples
    ========

    >>> from sympy.core.exprtools import decompose_power
    >>> from sympy.abc import x, y
    >>> from sympy import exp

    >>> decompose_power(x)
    (x, 1)
    >>> decompose_power(x**2)
    (x, 2)
    >>> decompose_power(exp(2*y/3))
    (exp(y/3), 2)

    """
    # 将表达式分解为基数和指数
    base, exp = expr.as_base_exp()

    # 检查指数是否为数值
    if exp.is_Number:
        # 如果指数为有理数
        if exp.is_Rational:
            # 如果指数不是整数，则将基数用 Pow 函数重新包装
            if not exp.is_Integer:
                base = Pow(base, Rational(1, exp.q))  # type: ignore
            # 取指数的分子部分
            e = exp.p  # type: ignore
        else:
            # 如果指数不是有理数，则将基数保持不变，指数设为 1
            base, e = expr, 1
    else:
        # 如果指数不是数值，则将指数分解为乘积的系数部分
        exp, tail = exp.as_coeff_Mul(rational=True)

        # 如果指数是 -1，则基数为 Pow(base, tail)，指数为 -1
        if exp is S.NegativeOne:
            base, e = Pow(base, tail), -1
        elif exp is not S.One:
            # todo: after dropping python 3.7 support, use overload and Literal
            #  in as_coeff_Mul to make exp Rational, and remove these 2 ignores
            # 如果指数不是 1，则将尾部系数保持为 Rational，基数为 Pow(base, tail)，指数为 exp.p
            tail = _keep_coeff(Rational(1, exp.q), tail)  # type: ignore
            base, e = Pow(base, tail), exp.p  # type: ignore
        else:
            # 如果指数为 1，则基数保持不变，指数为 1
            base, e = expr, 1

    return base, e


def decompose_power_rat(expr: Expr) -> tTuple[Expr, Rational]:
    """
    Decompose power into symbolic base and rational exponent;
    if the exponent is not a Rational, then separate only the
    integer coefficient.

    Examples
    ========

    >>> from sympy.core.exprtools import decompose_power_rat
    >>> from sympy.abc import x
    >>> from sympy import sqrt, exp

    >>> decompose_power_rat(sqrt(x))
    (x, 1/2)
    >>> decompose_power_rat(exp(-3*x/2))
    (exp(x/2), -3)

    """
    # 将表达式分解为基数和指数
    _ = base, exp = expr.as_base_exp()
    # 如果指数是有理数，则返回基数和指数；否则调用 decompose_power 函数进行处理
    return _ if exp.is_Rational else decompose_power(expr)


class Factors:
    """Efficient representation of ``f_1*f_2*...*f_n``."""

    __slots__ = ('factors', 'gens')

    def __hash__(self):  # Factors
        # 计算对象的哈希值
        keys = tuple(ordered(self.factors.keys()))
        values = [self.factors[k] for k in keys]
        return hash((keys, values))

    def __repr__(self):  # Factors
        # 返回对象的字符串表示，按照因子顺序输出
        return "Factors({%s})" % ', '.join(
            ['%s: %s' % (k, v) for k, v in ordered(self.factors.items())])

    @property
    def is_zero(self):  # Factors
        """
        >>> from sympy.core.exprtools import Factors
        >>> Factors(0).is_zero
        True
        """
        # 检查因子对象是否表示为零
        f = self.factors
        return len(f) == 1 and S.Zero in f

    @property
    def is_one(self):  # Factors
        """
        >>> from sympy.core.exprtools import Factors
        >>> Factors(1).is_one
        True
        """
        # 检查因子对象是否表示为一
        return not self.factors
    def as_expr(self):  # Factors
        """Return the underlying expression.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y
        >>> Factors((x*y**2).as_powers_dict()).as_expr()
        x*y**2

        """

        # 初始化一个空列表用于存放表达式的各个因子
        args = []
        # 遍历 factors 字典的每一项，factor 是因子，exp 是指数
        for factor, exp in self.factors.items():
            # 如果指数不为 1
            if exp != 1:
                # 如果指数是整数，将因子分解为底数和指数，并根据 exp 调整指数
                if isinstance(exp, Integer):
                    b, e = factor.as_base_exp()
                    e = _keep_coeff(exp, e)
                    args.append(b**e)
                else:
                    # 如果指数不是整数，直接将因子的 exp 次方加入到参数列表中
                    args.append(factor**exp)
            else:
                # 如果指数为 1，直接将因子加入到参数列表中
                args.append(factor)
        # 返回乘积表达式
        return Mul(*args)

    def mul(self, other):  # Factors
        """Return Factors of ``self * other``.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.mul(b)
        Factors({x: 2, y: 3, z: -1})
        >>> a*b
        Factors({x: 2, y: 3, z: -1})
        """
        # 如果 other 不是 Factors 类型，则转换为 Factors 类型
        if not isinstance(other, Factors):
            other = Factors(other)
        # 如果 self 或 other 中有任何一个因子为零，直接返回包含零因子的 Factors 对象
        if any(f.is_zero for f in (self, other)):
            return Factors(S.Zero)
        # 复制 self 的 factors 字典
        factors = dict(self.factors)

        # 遍历 other 的 factors 字典
        for factor, exp in other.factors.items():
            # 如果 factor 已经存在于 factors 中，则将其指数相加
            if factor in factors:
                exp = factors[factor] + exp

                # 如果新的指数为零，则从 factors 中删除该因子
                if not exp:
                    del factors[factor]
                    continue

            # 更新 factors 中的 factor 和 exp
            factors[factor] = exp

        # 返回合并后的 Factors 对象
        return Factors(factors)
    def normal(self, other):
        """Return ``self`` and ``other`` with ``gcd`` removed from each.
        The only differences between this and method ``div`` is that this
        is 1) optimized for the case when there are few factors in common and
        2) this does not raise an error if ``other`` is zero.

        See Also
        ========
        div

        """
        # 检查是否 `other` 是 Factors 类型，若不是则转换为 Factors 对象
        if not isinstance(other, Factors):
            other = Factors(other)
            # 若 `other` 是零，则返回空因数对象和代表零的因数对象
            if other.is_zero:
                return (Factors(), Factors(S.Zero))
            # 若 `self` 是零，则返回代表零的因数对象和空因数对象
            if self.is_zero:
                return (Factors(S.Zero), Factors())

        # 复制 `self` 和 `other` 的因数字典
        self_factors = dict(self.factors)
        other_factors = dict(other.factors)

        # 遍历 `self` 的因数字典
        for factor, self_exp in self.factors.items():
            try:
                # 尝试从 `other` 的因数字典中获取相同因数的指数
                other_exp = other.factors[factor]
            except KeyError:
                # 若未找到相同因数，则继续下一个因数
                continue

            # 计算因数的指数差异
            exp = self_exp - other_exp

            # 如果指数差为零，从因数字典中删除这个因数
            if not exp:
                del self_factors[factor]
                del other_factors[factor]
            # 如果指数差是数字类型，并且大于零，则更新 `self` 的因数指数
            elif _isnumber(exp):
                if exp > 0:
                    self_factors[factor] = exp
                    del other_factors[factor]
                else:
                    del self_factors[factor]
                    other_factors[factor] = -exp
            else:
                # 否则尝试通过 `extract_additively` 处理指数的复杂情况
                r = self_exp.extract_additively(other_exp)
                if r is not None:
                    if r:
                        self_factors[factor] = r
                        del other_factors[factor]
                    else:  # 应该已经处理过
                        del self_factors[factor]
                        del other_factors[factor]
                else:
                    # 如果不能通过 `extract_additively` 解决，则尝试分解为加法组合
                    sc, sa = self_exp.as_coeff_Add()
                    if sc:
                        oc, oa = other_exp.as_coeff_Add()
                        diff = sc - oc
                        if diff > 0:
                            self_factors[factor] -= oc
                            other_exp = oa
                        elif diff < 0:
                            self_factors[factor] -= sc
                            other_factors[factor] -= sc
                            other_exp = oa - diff
                        else:
                            self_factors[factor] = sa
                            other_exp = oa
                    if other_exp:
                        other_factors[factor] = other_exp
                    else:
                        del other_factors[factor]

        # 返回处理后的 `self_factors` 和 `other_factors` 作为 Factors 对象
        return Factors(self_factors), Factors(other_factors)
    def quo(self, other):  # Factors
        """Return numerator Factor of ``self / other``.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.quo(b)  # same as a/b
        Factors({y: 1})
        """
        # 使用除法运算得到 self 除以 other 的商，然后返回商的因子
        return self.div(other)[0]

    def rem(self, other):  # Factors
        """Return denominator Factors of ``self / other``.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.rem(b)
        Factors({z: -1})
        >>> a.rem(a)
        Factors({})
        """
        # 使用除法运算得到 self 除以 other 的商，然后返回商的余数部分的因子
        return self.div(other)[1]

    def pow(self, other):  # Factors
        """Return self raised to a non-negative integer power.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> a**2
        Factors({x: 2, y: 4})

        """
        # 检查 other 是否为 Factors 类型，如果是则转换为表达式
        if isinstance(other, Factors):
            other = other.as_expr()
            if other.is_Integer:
                other = int(other)
        # 检查 other 是否为非负整数
        if isinstance(other, SYMPY_INTS) and other >= 0:
            factors = {}

            # 如果 other 不为零，则计算 self 的因子与 other 的乘积
            if other:
                for factor, exp in self.factors.items():
                    factors[factor] = exp * other

            return Factors(factors)
        else:
            # 如果 other 不是非负整数则引发 ValueError 异常
            raise ValueError("expected non-negative integer, got %s" % other)

    def gcd(self, other):  # Factors
        """Return Factors of ``gcd(self, other)``. The keys are
        the intersection of factors with the minimum exponent for
        each factor.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.gcd(b)
        Factors({x: 1, y: 1})
        """
        # 如果 other 不是 Factors 类型，则将其转换为 Factors 类型
        if not isinstance(other, Factors):
            other = Factors(other)
            # 如果 other 是零则返回 self 的因子
            if other.is_zero:
                return Factors(self.factors)

        factors = {}

        # 遍历 self 的因子及其指数
        for factor, exp in self.factors.items():
            factor, exp = sympify(factor), sympify(exp)
            # 如果 other 也有相同的因子，则比较它们的指数
            if factor in other.factors:
                lt = (exp - other.factors[factor]).is_negative
                if lt == True:
                    factors[factor] = exp
                elif lt == False:
                    factors[factor] = other.factors[factor]

        return Factors(factors)
    def lcm(self, other):  # Factors
        """Return Factors of ``lcm(self, other)`` which are
        the union of factors with the maximum exponent for
        each factor.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.lcm(b)
        Factors({x: 1, y: 2, z: -1})
        """
        # 如果other不是Factors类型，则将其转换为Factors对象
        if not isinstance(other, Factors):
            other = Factors(other)
        # 如果self或other中有任何一个是零因子，则返回零因子Factors对象
        if any(f.is_zero for f in (self, other)):
            return Factors(S.Zero)

        # 复制当前对象的因子字典
        factors = dict(self.factors)

        # 遍历other对象的因子字典
        for factor, exp in other.factors.items():
            # 如果当前对象已经包含该因子，则取当前对象和other对象中的最大指数
            if factor in factors:
                exp = max(exp, factors[factor])

            # 更新因子字典
            factors[factor] = exp

        # 返回一个新的Factors对象，包含合并后的因子字典
        return Factors(factors)

    def __mul__(self, other):  # Factors
        # 调用mul方法处理乘法操作
        return self.mul(other)

    def __divmod__(self, other):  # Factors
        # 调用div方法处理除法取模操作
        return self.div(other)

    def __truediv__(self, other):  # Factors
        # 调用quo方法处理真除法操作
        return self.quo(other)

    def __mod__(self, other):  # Factors
        # 调用rem方法处理取模操作
        return self.rem(other)

    def __pow__(self, other):  # Factors
        # 调用pow方法处理幂运算操作
        return self.pow(other)

    def __eq__(self, other):  # Factors
        # 如果other不是Factors类型，则将其转换为Factors对象
        if not isinstance(other, Factors):
            other = Factors(other)
        # 比较当前对象的因子字典与other对象的因子字典是否相等
        return self.factors == other.factors

    def __ne__(self, other):  # Factors
        # 判断当前对象与other对象的因子字典是否不相等
        return not self == other
class Term:
    """Efficient representation of ``coeff*(numer/denom)``. """

    __slots__ = ('coeff', 'numer', 'denom')

    def __init__(self, term, numer=None, denom=None):  # Term
        # 如果未提供分子和分母，则从给定的表达式中提取系数和因子
        if numer is None and denom is None:
            # 检查表达式是否可交换，若不可交换则引发异常
            if not term.is_commutative:
                raise NonCommutativeExpression(
                    'commutative expression expected')

            # 提取表达式的系数和因子
            coeff, factors = term.as_coeff_mul()
            numer, denom = defaultdict(int), defaultdict(int)

            # 遍历因子列表
            for factor in factors:
                # 分解因子为底数和指数
                base, exp = decompose_power(factor)

                # 如果底数是加法表达式，则将其拆解为最简形式
                if base.is_Add:
                    cont, base = base.primitive()
                    coeff *= cont**exp

                # 根据指数正负，更新分子或分母的对应因子及其指数
                if exp > 0:
                    numer[base] += exp
                else:
                    denom[base] += -exp

            # 将分子和分母转换为 Factors 对象
            numer = Factors(numer)
            denom = Factors(denom)
        else:
            # 如果提供了分子和分母，则直接使用给定的系数和因子
            coeff = term

            if numer is None:
                numer = Factors()

            if denom is None:
                denom = Factors()

        # 初始化 Term 对象的属性
        self.coeff = coeff
        self.numer = numer
        self.denom = denom

    def __hash__(self):  # Term
        # 计算 Term 对象的哈希值
        return hash((self.coeff, self.numer, self.denom))

    def __repr__(self):  # Term
        # 返回 Term 对象的字符串表示形式
        return "Term(%s, %s, %s)" % (self.coeff, self.numer, self.denom)

    def as_expr(self):  # Term
        # 将 Term 对象转换为表达式形式
        return self.coeff*(self.numer.as_expr()/self.denom.as_expr())

    def mul(self, other):  # Term
        # 两个 Term 对象相乘，返回新的 Term 对象
        coeff = self.coeff*other.coeff
        numer = self.numer.mul(other.numer)
        denom = self.denom.mul(other.denom)

        # 规范化乘积的分子和分母
        numer, denom = numer.normal(denom)

        return Term(coeff, numer, denom)

    def inv(self):  # Term
        # 计算 Term 对象的倒数
        return Term(1/self.coeff, self.denom, self.numer)

    def quo(self, other):  # Term
        # 两个 Term 对象相除，返回新的 Term 对象
        return self.mul(other.inv())

    def pow(self, other):  # Term
        # 计算 Term 对象的指数幂
        if other < 0:
            return self.inv().pow(-other)
        else:
            return Term(self.coeff ** other,
                        self.numer.pow(other),
                        self.denom.pow(other))

    def gcd(self, other):  # Term
        # 计算两个 Term 对象的最大公因数
        return Term(self.coeff.gcd(other.coeff),
                    self.numer.gcd(other.numer),
                    self.denom.gcd(other.denom))

    def lcm(self, other):  # Term
        # 计算两个 Term 对象的最小公倍数
        return Term(self.coeff.lcm(other.coeff),
                    self.numer.lcm(other.numer),
                    self.denom.lcm(other.denom))

    def __mul__(self, other):  # Term
        # 定义 Term 对象的乘法操作
        if isinstance(other, Term):
            return self.mul(other)
        else:
            return NotImplemented

    def __truediv__(self, other):  # Term
        # 定义 Term 对象的除法操作
        if isinstance(other, Term):
            return self.quo(other)
        else:
            return NotImplemented

    def __pow__(self, other):  # Term
        # 定义 Term 对象的指数操作
        if isinstance(other, SYMPY_INTS):
            return self.pow(other)
        else:
            return NotImplemented
    def __eq__(self, other):  # Term
        # 检查当前对象的系数、分子和分母是否与另一个对象相等
        return (self.coeff == other.coeff and
                self.numer == other.numer and
                self.denom == other.denom)

    def __ne__(self, other):  # Term
        # 使用 __eq__ 方法的相反结果来判断两个对象是否不相等
        return not self == other
# 定义函数 `_gcd_terms`，用于辅助 `gcd_terms` 函数计算最大公因数并整合结果
def _gcd_terms(terms, isprimitive=False, fraction=True):
    """Helper function for :func:`gcd_terms`.

    Parameters
    ==========

    isprimitive : boolean, optional
        If ``isprimitive`` is True then the call to primitive
        for an Add will be skipped. This is useful when the
        content has already been extracted.

    fraction : boolean, optional
        If ``fraction`` is True then the expression will appear over a common
        denominator, the lcm of all term denominators.
    """

    # 如果 terms 是 Basic 类型但不是 Tuple 类型，则将其转换为 Add.make_args 处理后的列表
    if isinstance(terms, Basic) and not isinstance(terms, Tuple):
        terms = Add.make_args(terms)

    # 将 terms 转换为 Term 类型的列表，滤除空值
    terms = list(map(Term, [t for t in terms if t]))

    # 如果 terms 列表为空，则返回三个零
    if len(terms) == 0:
        return S.Zero, S.Zero, S.One

    # 如果 terms 只有一个元素，将其分别作为 cont（系数）、numer（分子）、denom（分母）
    if len(terms) == 1:
        cont = terms[0].coeff
        numer = terms[0].numer.as_expr()
        denom = terms[0].denom.as_expr()

    else:
        # 计算 terms 中所有项的最大公因数 cont
        cont = terms[0]
        for term in terms[1:]:
            cont = cont.gcd(term)

        # 将每个 term 除以 cont，即得到归一化的 terms
        for i, term in enumerate(terms):
            terms[i] = term.quo(cont)

        # 如果 fraction=True，则计算 terms 的公共分母 denom
        if fraction:
            denom = terms[0].denom

            for term in terms[1:]:
                denom = denom.lcm(term.denom)

            numers = []
            for term in terms:
                numer = term.numer.mul(denom.quo(term.denom))
                numers.append(term.coeff*numer.as_expr())
        else:
            # 如果 fraction=False，则直接将 terms 的数值表达式作为分子
            numers = [t.as_expr() for t in terms]
            denom = Term(S.One).numer

        # 将 cont、numer、denom 转换为数值表达式
        cont = cont.as_expr()
        numer = Add(*numers)
        denom = denom.as_expr()

    # 如果 isprimitive=False 且 numer 是 Add 类型，则计算 numer 的原始表达式
    if not isprimitive and numer.is_Add:
        _cont, numer = numer.primitive()
        cont *= _cont

    # 返回计算结果 cont（系数）、numer（分子）、denom（分母）
    return cont, numer, denom


# 定义函数 gcd_terms，计算给定 terms 的最大公因数并整合结果
def gcd_terms(terms, isprimitive=False, clear=True, fraction=True):
    """Compute the GCD of ``terms`` and put them together.

    Parameters
    ==========

    terms : Expr
        Can be an expression or a non-Basic sequence of expressions
        which will be handled as though they are terms from a sum.

    isprimitive : bool, optional
        If ``isprimitive`` is True the _gcd_terms will not run the primitive
        method on the terms.

    clear : bool, optional
        It controls the removal of integers from the denominator of an Add
        expression. When True (default), all numerical denominator will be cleared;
        when False the denominators will be cleared only if all terms had numerical
        denominators other than 1.

    fraction : bool, optional
        When True (default), will put the expression over a common
        denominator.

    Examples
    ========

    >>> from sympy import gcd_terms
    >>> from sympy.abc import x, y

    >>> gcd_terms((x + 1)**2*y + (x + 1)*y**2)
    y*(x + 1)*(x + y + 1)
    >>> gcd_terms(x/2 + 1)
    (x + 2)/2
    >>> gcd_terms(x/2 + 1, clear=False)
    x/2 + 1
    """
    def gcd_terms(x, clear=False, fraction=True):
        """
        Simplify an expression by factoring out the greatest common divisor (GCD) of its terms.

        Parameters
        ----------
        x : Expr
            The expression to simplify.
        clear : bool, optional
            If True, attempts to clear the GCD from the result; defaults to False.
        fraction : bool, optional
            If True, returns the result as a fraction if possible; defaults to True.

        Returns
        -------
        Expr
            The simplified expression.

        Examples
        --------
        >>> gcd_terms(x/2 + y/2, clear=False)
        (x + y)/2
        >>> gcd_terms(x/2 + 1/x)
        (x**2 + 2)/(2*x)
        >>> gcd_terms(x/2 + 1/x, fraction=False)
        (x + 2/x)/2
        >>> gcd_terms(x/2 + 1/x, fraction=False, clear=False)
        x/2 + 1/x

        >>> gcd_terms(x/2/y + 1/x/y)
        (x**2 + 2)/(2*x*y)
        >>> gcd_terms(x/2/y + 1/x/y, clear=False)
        (x**2/2 + 1)/(x*y)
        >>> gcd_terms(x/2/y + 1/x/y, clear=False, fraction=False)
        (x/2 + 1/x)/y

        Notes
        -----
        The `clear` flag may be ignored if the returned expression is a rational expression,
        not a simple sum.

        See Also
        --------
        factor_terms, sympy.polys.polytools.terms_gcd
        """

        def mask(terms):
            """
            Replace non-commutative portions of each term with a unique Dummy symbol
            and return the replacements to restore them.
            """
            args = [(a, []) if a.is_commutative else a.args_cnc() for a in terms]
            reps = []
            for i, (c, nc) in enumerate(args):
                if nc:
                    nc = Mul(*nc)
                    d = Dummy()
                    reps.append((d, nc))
                    c.append(d)
                    args[i] = Mul(*c)
                else:
                    args[i] = c
            return args, dict(reps)

        isadd = isinstance(terms, Add)
        addlike = isadd or (not isinstance(terms, Basic) and
                             is_sequence(terms, include=set) and
                             not isinstance(terms, Dict))

        if addlike:
            if isadd:
                terms = list(terms.args)
            else:
                terms = sympify(terms)
            terms, reps = mask(terms)
            cont, numer, denom = _gcd_terms(terms, isprimitive, fraction)
            numer = numer.xreplace(reps)
            coeff, factors = cont.as_coeff_Mul()
            if not clear:
                c, _coeff = coeff.as_coeff_Mul()
                if not c.is_Integer and not clear and numer.is_Add:
                    n, d = c.as_numer_denom()
                    _numer = numer/d
                    if any(a.as_coeff_Mul()[0].is_Integer for a in _numer.args):
                        numer = _numer
                        coeff = n*_coeff
            return _keep_coeff(coeff, factors*numer/denom, clear=clear)

        if not isinstance(terms, Basic):
            return terms

        if terms.is_Atom:
            return terms

        if terms.is_Mul:
            c, args = terms.as_coeff_mul()
            return _keep_coeff(c, Mul(*[gcd_terms(i, isprimitive, clear, fraction)
                                         for i in args]), clear=clear)

        def handle(a):
            """
            Recursively apply gcd_terms to handle nested expressions.
            """
            if not isinstance(a, Expr):
                if isinstance(a, Basic):
                    if not a.args:
                        return a
                    return a.func(*[handle(i) for i in a.args])
                return type(a)([handle(i) for i in a])
            return gcd_terms(a, isprimitive, clear, fraction)

        if isinstance(terms, Dict):
            return Dict(*[(k, handle(v)) for k, v in terms.args])
        return terms.func(*[handle(i) for i in terms.args])
def _factor_sum_int(expr, **kwargs):
    """Return Sum or Integral object with factors that are not
    in the wrt variables removed. In cases where there are additive
    terms in the function of the object that are independent, the
    object will be separated into two objects.

    Examples
    ========

    >>> from sympy import Sum, factor_terms
    >>> from sympy.abc import x, y
    >>> factor_terms(Sum(x + y, (x, 1, 3)))
    y*Sum(1, (x, 1, 3)) + Sum(x, (x, 1, 3))
    >>> factor_terms(Sum(x*y, (x, 1, 3)))
    y*Sum(x, (x, 1, 3))

    Notes
    =====

    If a function in the summand or integrand is replaced
    with a symbol, then this simplification should not be
    done or else an incorrect result will be obtained when
    the symbol is replaced with an expression that depends
    on the variables of summation/integration:

    >>> eq = Sum(y, (x, 1, 3))
    >>> factor_terms(eq).subs(y, x).doit()
    3*x
    >>> eq.subs(y, x).doit()
    6
    """
    # Get the main function or expression from the input 'expr'
    result = expr.function
    # If the main function is zero, return the zero value from the sympy library
    if result == 0:
        return S.Zero
    # Extract the limits of summation or integration from 'expr'
    limits = expr.limits

    # Get the variables with respect to which factors should be considered
    wrt = {i.args[0] for i in limits}

    # Factor out common terms that are independent of the wrt variables
    f = factor_terms(result, **kwargs)
    # Split into independent and dependent components
    i, d = f.as_independent(*wrt)
    # Check if the result is an Add expression
    if isinstance(f, Add):
        # Return the separated components as specified
        return i * expr.func(1, *limits) + expr.func(d, *limits)
    else:
        # Return the modified expression
        return i * expr.func(d, *limits)


def factor_terms(expr, radical=False, clear=False, fraction=False, sign=True):
    """Remove common factors from terms in all arguments without
    changing the underlying structure of the expr. No expansion or
    simplification (and no processing of non-commutatives) is performed.

    Parameters
    ==========

    radical: bool, optional
        If radical=True then a radical common to all terms will be factored
        out of any Add sub-expressions of the expr.

    clear : bool, optional
        If clear=False (default) then coefficients will not be separated
        from a single Add if they can be distributed to leave one or more
        terms with integer coefficients.

    fraction : bool, optional
        If fraction=True (default is False) then a common denominator will be
        constructed for the expression.

    sign : bool, optional
        If sign=True (default) then even if the only factor in common is a -1,
        it will be factored out of the expression.

    Examples
    ========

    >>> from sympy import factor_terms, Symbol
    >>> from sympy.abc import x, y
    >>> factor_terms(x + x*(2 + 4*y)**3)
    x*(8*(2*y + 1)**3 + 1)
    >>> A = Symbol('A', commutative=False)
    >>> factor_terms(x*A + x*A + x*y*A)
    x*(y*A + 2*A)

    When ``clear`` is False, a rational will only be factored out of an
    Add expression if all terms of the Add have coefficients that are
    fractions:

    >>> factor_terms(x/2 + 1, clear=False)
    x/2 + 1
    >>> factor_terms(x/2 + 1, clear=True)
    (x + 2)/2
    """
    # Implementation of factor_terms function which removes common factors
    # from terms in the input expression 'expr' based on the specified options
    # 定义一个函数，用于对表达式进行因式分解和处理
    def factor_terms(expr, radical=None, clear=None, fraction=None, sign=True):
        # 导入需要使用的模块和函数
        from sympy.concrete.summations import Sum
        from sympy.integrals.integrals import Integral
        # 检查表达式是否可迭代
        is_iterable = iterable(expr)

        # 如果表达式不是 Basic 类型或者是原子表达式，则直接返回表达式本身或处理后的可迭代对象
        if not isinstance(expr, Basic) or expr.is_Atom:
            if is_iterable:
                return type(expr)([do(i) for i in expr])
            return expr

        # 如果表达式是幂函数、一般函数、可迭代对象或者没有 args_cnc 属性，则对其参数逐一处理
        if expr.is_Pow or expr.is_Function or \
                is_iterable or not hasattr(expr, 'args_cnc'):
            args = expr.args
            newargs = tuple([do(i) for i in args])
            # 如果处理后的参数和原参数相同，则返回原表达式，否则返回处理后的表达式
            if newargs == args:
                return expr
            return expr.func(*newargs)

        # 如果表达式是求和或积分表达式，则调用 _factor_sum_int 函数处理
        if isinstance(expr, (Sum, Integral)):
            return _factor_sum_int(expr,
                radical=radical, clear=clear,
                fraction=fraction, sign=sign)

        # 将表达式拆分为内容部分和原子部分，并进行处理
        cont, p = expr.as_content_primitive(radical=radical, clear=clear)
        
        # 如果原子部分是加法表达式，则逐一处理每个加法项
        if p.is_Add:
            list_args = [do(a) for a in Add.make_args(p)]
            # 检查是否有公共的负数因子，如果有，则进行特殊处理
            if not any(a.as_coeff_Mul()[0].extract_multiplicatively(-1) is None
                       for a in list_args):
                cont = -cont
                list_args = [-a for a in list_args]
            # 处理特殊的幂函数表达式，避免 gcd_terms 函数改变其顺序
            special = {}
            for i, a in enumerate(list_args):
                b, e = a.as_base_exp()
                if e.is_Mul and e != Mul(*e.args):
                    list_args[i] = Dummy()
                    special[list_args[i]] = a
            # 重新构建加法表达式 p，不关心顺序，因为 gcd_terms 函数会修正顺序
            p = Add._from_args(list_args)
            p = gcd_terms(p,
                isprimitive=True,
                clear=clear,
                fraction=fraction).xreplace(special)
        # 如果原子部分有参数，则对参数逐一处理
        elif p.args:
            p = p.func(
                *[do(a) for a in p.args])
        # 使用 _keep_coeff 函数保持系数，根据参数设置保持正负号
        rv = _keep_coeff(cont, p, clear=clear, sign=sign)
        return rv
    
    # 将输入的表达式转换为 Sympy 的表达式对象，并调用 factor_terms 函数处理后返回结果
    expr = sympify(expr)
    return do(expr)
# 定义一个函数 _mask_nc，用于处理非交换对象，将它们替换为虚拟符号（Dummy symbols）
def _mask_nc(eq, name=None):
    """
    Return ``eq`` with non-commutative objects replaced with Dummy
    symbols. A dictionary that can be used to restore the original
    values is returned: if it is None, the expression is noncommutative
    and cannot be made commutative. The third value returned is a list
    of any non-commutative symbols that appear in the returned equation.

    Explanation
    ===========

    All non-commutative objects other than Symbols are replaced with
    a non-commutative Symbol. Identical objects will be identified
    by identical symbols.

    If there is only 1 non-commutative object in an expression it will
    be replaced with a commutative symbol. Otherwise, the non-commutative
    entities are retained and the calling routine should handle
    replacements in this case since some care must be taken to keep
    track of the ordering of symbols when they occur within Muls.

    Parameters
    ==========

    name : str
        ``name``, if given, is the name that will be used with numbered Dummy
        variables that will replace the non-commutative objects and is mainly
        used for doctesting purposes.

    Examples
    ========

    >>> from sympy.physics.secondquant import Commutator, NO, F, Fd
    >>> from sympy import symbols
    >>> from sympy.core.exprtools import _mask_nc
    >>> from sympy.abc import x, y
    >>> A, B, C = symbols('A,B,C', commutative=False)

    One nc-symbol:

    >>> _mask_nc(A**2 - x**2, 'd')
    (_d0**2 - x**2, {_d0: A}, [])

    Multiple nc-symbols:

    >>> _mask_nc(A**2 - B**2, 'd')
    (A**2 - B**2, {}, [A, B])

    An nc-object with nc-symbols but no others outside of it:

    >>> _mask_nc(1 + x*Commutator(A, B), 'd')
    (_d0*x + 1, {_d0: Commutator(A, B)}, [])
    >>> _mask_nc(NO(Fd(x)*F(y)), 'd')
    (_d0, {_d0: NO(CreateFermion(x)*AnnihilateFermion(y))}, [])

    Multiple nc-objects:

    >>> eq = x*Commutator(A, B) + x*Commutator(A, C)*Commutator(A, B)
    >>> _mask_nc(eq, 'd')
    (x*_d0 + x*_d1*_d0, {_d0: Commutator(A, B), _d1: Commutator(A, C)}, [_d0, _d1])

    Multiple nc-objects and nc-symbols:

    >>> eq = A*Commutator(A, B) + B*Commutator(A, C)
    >>> _mask_nc(eq, 'd')
    (A*_d0 + B*_d1, {_d0: Commutator(A, B), _d1: Commutator(A, C)}, [_d0, _d1, A, B])

    """
    # 默认名称为 'mask'
    name = name or 'mask'
    # 生成带有序号的 Dummy 符号名称的生成器
    def numbered_names():
        i = 0
        while True:
            yield name + str(i)
            i += 1

    # 使用生成器创建名称生成器
    names = numbered_names()

    # 定义一个 Dummy 函数，返回一个 Dummy 符号
    def Dummy(*args, **kwargs):
        from .symbol import Dummy
        return Dummy(next(names), *args, **kwargs)

    # 将输入表达式保存到变量 expr 中
    expr = eq
    # 如果表达式是可交换的，则直接返回原始表达式、空字典和空列表
    if expr.is_commutative:
        return eq, {}, []

    # 标识非交换对象，并对符号和其他对象进行替换
    rep = []
    nc_obj = set()
    nc_syms = set()
    # 使用 preorder_traversal 函数遍历表达式，使用 default_sort_key 进行键排序
    pot = preorder_traversal(expr, keys=default_sort_key)
    # 对于列表 `pot` 中的每个元素 `a`，使用 `enumerate` 获取其索引 `i`
    for i, a in enumerate(pot):
        # 检查是否有任何与 `a` 的首元素相同的项在 `rep` 中，如果是，则跳过当前项
        if any(a == r[0] for r in rep):
            pot.skip()
        # 如果 `a` 不是可交换的
        elif not a.is_commutative:
            # 如果 `a` 是符号
            if a.is_symbol:
                # 将 `a` 添加到非交换符号集合 `nc_syms` 中，并跳过当前项
                nc_syms.add(a)
                pot.skip()
            # 如果 `a` 不是加法、乘法或幂运算
            elif not (a.is_Add or a.is_Mul or a.is_Pow):
                # 将 `a` 添加到非交换对象集合 `nc_obj` 中，并跳过当前项
                nc_obj.add(a)
                pot.skip()

    # 如果只有一个非交换对象 `nc_obj`，且没有非交换符号 `nc_syms`
    # 将其替换为一个虚拟变量 Dummy，以便进行正常因式分解，避免 polys 报错
    if len(nc_obj) == 1 and not nc_syms:
        rep.append((nc_obj.pop(), Dummy()))
    # 如果只有一个非交换符号 `nc_syms`，且没有非交换对象 `nc_obj`
    # 将其替换为一个虚拟变量 Dummy
    elif len(nc_syms) == 1 and not nc_obj:
        rep.append((nc_syms.pop(), Dummy()))

    # 对剩余的非交换对象 `nc_obj`，替换为一个非交换的虚拟变量 Dummy，并将其添加到 `rep` 中
    nc_obj = sorted(nc_obj, key=default_sort_key)
    for n in nc_obj:
        nc = Dummy(commutative=False)
        rep.append((n, nc))
        nc_syms.add(nc)
    
    # 使用替换规则 `rep` 替换表达式 `expr` 中的所有匹配项
    expr = expr.subs(rep)

    # 将非交换符号集合 `nc_syms` 转换为列表，并按默认排序方式排序
    nc_syms = list(nc_syms)
    nc_syms.sort(key=default_sort_key)
    
    # 返回替换后的表达式 `expr`、用于反向映射的字典 `{v: k for k, v in rep}` 和非交换符号列表 `nc_syms`
    return expr, {v: k for k, v in rep}, nc_syms
# 定义函数 `factor_nc`，用于处理非交换表达式的因式分解
def factor_nc(expr):
    """Return the factored form of ``expr`` while handling non-commutative
    expressions.

    Examples
    ========

    >>> from sympy import factor_nc, Symbol
    >>> from sympy.abc import x
    >>> A = Symbol('A', commutative=False)
    >>> B = Symbol('B', commutative=False)
    >>> factor_nc((x**2 + 2*A*x + A**2).expand())
    (x + A)**2
    >>> factor_nc(((x + A)*(x + B)).expand())
    (x + A)*(x + B)
    """
    # 将输入的表达式转换为 SymPy 的表达式对象
    expr = sympify(expr)
    # 如果表达式不是 SymPy 的表达式对象，或者没有子表达式，则直接返回原表达式
    if not isinstance(expr, Expr) or not expr.args:
        return expr
    # 如果表达式不是加法表达式，则递归地对表达式的每个子表达式调用 `factor_nc` 函数
    if not expr.is_Add:
        return expr.func(*[factor_nc(a) for a in expr.args])
    # 展开表达式中的幂乘法项
    expr = expr.func(*[expand_power_exp(i) for i in expr.args])

    # 导入 SymPy 中的多项式工具函数 gcd 和 factor
    from sympy.polys.polytools import gcd, factor
    # 调用 `_mask_nc` 函数对表达式进行处理，获取处理后的表达式、替换字典和非交换符号集合
    expr, rep, nc_symbols = _mask_nc(expr)

    # 如果存在替换字典，则对处理后的表达式进行因式分解并进行替换
    if rep:
        return factor(expr).subs(rep)
```