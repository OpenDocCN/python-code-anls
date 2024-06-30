# `D:\src\scipysrc\sympy\sympy\core\power.py`

```
from __future__ import annotations
from typing import Callable
from itertools import product

from .sympify import _sympify       # 导入符号化函数_sympify
from .cache import cacheit           # 导入缓存函数cacheit
from .singleton import S             # 导入单例对象S
from .expr import Expr              # 导入表达式基类Expr
from .evalf import PrecisionExhausted  # 导入精度耗尽异常类PrecisionExhausted
from .function import (expand_complex, expand_multinomial,
    expand_mul, _mexpand, PoleError)  # 导入数学函数扩展相关的函数和异常类
from .logic import fuzzy_bool, fuzzy_not, fuzzy_and, fuzzy_or  # 导入模糊逻辑函数
from .parameters import global_parameters  # 导入全局参数对象global_parameters
from .relational import is_gt, is_lt    # 导入关系运算函数is_gt和is_lt
from .kind import NumberKind, UndefinedKind  # 导入数值类型相关的类
from sympy.utilities.iterables import sift  # 导入可迭代对象操作函数sift
from sympy.utilities.exceptions import sympy_deprecation_warning  # 导入SymPy的警告异常类sympy_deprecation_warning
from sympy.utilities.misc import as_int   # 导入类型转换函数as_int
from sympy.multipledispatch import Dispatcher  # 导入多重分派对象Dispatcher


class Pow(Expr):
    """
    Defines the expression x**y as "x raised to a power y"

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Singleton definitions involving (0, 1, -1, oo, -oo, I, -I):

    +--------------+---------+-----------------------------------------------+
    | expr         | value   | reason                                        |
    +==============+=========+===============================================+
    | z**0         | 1       | Although arguments over 0**0 exist, see [2].  |
    +--------------+---------+-----------------------------------------------+
    | z**1         | z       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**(-1)  | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-1)**-1     | -1      |                                               |
    +--------------+---------+-----------------------------------------------+
    | S.Zero**-1   | zoo     | This is not strictly true, as 0**-1 may be    |
    |              |         | undefined, but is convenient in some contexts |
    |              |         | where the base is assumed to be positive.     |
    +--------------+---------+-----------------------------------------------+
    | 1**-1        | 1       |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-1       | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | 0**oo        | 0       | Because for all complex numbers z near        |
    |              |         | 0, z**oo -> 0.                                |
    +--------------+---------+-----------------------------------------------+
    | 0**-oo       | zoo     | This is not strictly true, as 0**oo may be    |
    |              |         | oscillating between positive and negative     |
    """
    # 以下是对不同数值和复数幂运算结果的说明和解释。
    # 由于复数幂运算中涉及到各种情况，可能会产生特定的结果或未定义的情况。
    # 这些结果可以影响符号计算的灵活性，避免返回不正确的答案。
    
    +--------------+---------+-----------------------------------------------+
    | 1**oo        | nan     | 因为存在多种情况，其中 x(t) -> 1, y(t) -> oo (或 -oo)，但 lim( x(t)**y(t), t) != 1。参见 [3]。
    | 1**-oo       |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | b**zoo       | nan     | 因为 b**z 在 z -> zoo 时没有极限。                     |
    +--------------+---------+-----------------------------------------------+
    | (-1)**oo     | nan     | 因为在极限中存在振荡。                               |
    | (-1)**(-oo)  |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**oo       | oo      | 结果是无穷大。                                     |
    +--------------+---------+-----------------------------------------------+
    | oo**-oo      | 0       | 结果是 0。                                        |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**oo    | nan     | 结果未定义。                                      |
    | (-oo)**-oo   |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**I        | nan     | 当底数为无穷大，指数为复数 I 时，其极限不存在，用 nan 表示。   |
    | (-oo)**I     |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**(1+I)    | zoo     | 当复数指数的实部为正时，底数为无穷大时，abs(x**e) 的极限是无穷大。 |
    | (-oo)**(1+I) |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**(-1+I)   | 0       | 当复数指数的实部为负时，底数为无穷大时，极限为 0。            |
    | -oo**(-1+I)  |         |                                               |
    +--------------+---------+-----------------------------------------------+
    
    # 因为符号计算比浮点数计算更加灵活，并且我们希望永远不返回错误的答案，
    # 所以我们选择不完全遵循所有 IEEE 754 的规范。这有助于避免在极限计算中额外添加测试用例代码。
    """
    is_Pow = True
        # 设置变量is_Pow为True，用于标识这是一个Pow类
    
    __slots__ = ('is_commutative',)
        # 定义只允许存在一个属性is_commutative的槽（slot）
    
    args: tuple[Expr, Expr]
        # 声明args变量为一个包含两个Expr类型元素的元组
    
    _args: tuple[Expr, Expr]
        # 声明_args变量为一个包含两个Expr类型元素的元组
    
    @cacheit
        # 修饰器：缓存方法的结果以提高性能
    def __new__(cls, b, e, evaluate=None):
        # Pow类的构造方法，接受参数b和e，可选的evaluate参数
        if evaluate is None:
            evaluate = global_parameters.evaluate
            # 如果evaluate参数未指定，则使用全局参数global_parameters.evaluate的值
    
        b = _sympify(b)
            # 将b参数转换为符号表达式（sympify）
    
        e = _sympify(e)
            # 将e参数转换为符号表达式（sympify）
    
        # XXX: This can be removed when non-Expr args are disallowed rather
        # than deprecated.
        from .relational import Relational
            # 导入Relational类用于类型检查
    
        if isinstance(b, Relational) or isinstance(e, Relational):
            raise TypeError('Relational cannot be used in Pow')
            # 如果b或e是Relational类型，则抛出TypeError异常
    
        # XXX: This should raise TypeError once deprecation period is over:
        for arg in [b, e]:
            if not isinstance(arg, Expr):
                sympy_deprecation_warning(
                    f"""
        Using non-Expr arguments in Pow is deprecated (in this case, one of the
        arguments is of type {type(arg).__name__!r}).
    
        If you really did intend to construct a power with this base, use the **
                # 如果b或e不是Expr类型，则发出警告信息
    
    def inverse(self, argindex=1):
        # 定义方法inverse，返回指定索引的逆操作函数
        if self.base == S.Exp1:
            from sympy.functions.elementary.exponential import log
            return log
            # 如果底数是自然指数S.Exp1，则返回对数函数log
    
        return None
            # 否则返回None
    
    @property
    def base(self) -> Expr:
        # 定义属性base，返回底数（_args的第一个元素）
        return self._args[0]
    
    @property
    def exp(self) -> Expr:
        # 定义属性exp，返回指数（_args的第二个元素）
        return self._args[1]
    
    @property
    def kind(self):
        # 定义属性kind，返回幂运算的类型
        if self.exp.kind is NumberKind:
            return self.base.kind
        else:
            return UndefinedKind
        # 如果指数是NumberKind类型，则返回底数的类型，否则返回UndefinedKind
    
    @classmethod
    def class_key(cls):
        # 定义类方法class_key，返回类的标识元组
        return 3, 2, cls.__name__
        # 返回元组(3, 2, 类名)，用于类的标识
    
    def _eval_refine(self, assumptions):
        # 定义方法_eval_refine，用于细化运算
        from sympy.assumptions.ask import ask, Q
            # 导入ask函数和Q对象
    
        b, e = self.as_base_exp()
            # 将当前Pow对象拆分为底数b和指数e
    
        if ask(Q.integer(e), assumptions) and b.could_extract_minus_sign():
            # 如果指数e是整数并且底数b可以提取负号
            if ask(Q.even(e), assumptions):
                return Pow(-b, e)
                # 如果指数e是偶数，则返回Pow(-b, e)
            elif ask(Q.odd(e), assumptions):
                return -Pow(-b, e)
                # 如果指数e是奇数，则返回-Pow(-b, e)
    # 定义 `_eval_Mod` 方法，用于计算 `b^e \bmod q` 的结果，由 `Mod` 调用
    def _eval_Mod(self, q):
        r"""A dispatched function to compute `b^e \bmod q`, dispatched
        by ``Mod``.
        
        Notes
        =====
        
        Algorithms:
        
        1. For unevaluated integer power, use built-in ``pow`` function
        with 3 arguments, if powers are not too large wrt base.
        
        2. For very large powers, use totient reduction if $e \ge \log(m)$.
        Bound on m, is for safe factorization memory wise i.e. $m^{1/4}$.
        For pollard-rho to be faster than built-in pow $\log(e) > m^{1/4}$
        check is added.
        
        3. For any unevaluated power found in `b` or `e`, the step 2
        will be recursed down to the base and the exponent
        such that the $b \bmod q$ becomes the new base and
        $\phi(q) + e \bmod \phi(q)$ becomes the new exponent, and then
        the computation for the reduced expression can be done.
        """

        # 从对象中获取基数和指数
        base, exp = self.base, self.exp

        # 如果指数是整数且为正数
        if exp.is_integer and exp.is_positive:
            # 若 q 是整数且基数对 q 取模为 0，则返回零
            if q.is_integer and base % q == 0:
                return S.Zero

            # 导入欧拉函数 totient
            from sympy.functions.combinatorial.numbers import totient

            # 如果基数、指数和 q 都是整数
            if base.is_Integer and exp.is_Integer and q.is_Integer:
                b, e, m = int(base), int(exp), int(q)
                mb = m.bit_length()
                # 如果 m 的比特位长度小于等于 80，且 e 大于等于 mb，且 e 的比特位的 4 次方大于等于 m
                if mb <= 80 and e >= mb and e.bit_length()**4 >= m:
                    # 计算欧拉函数值
                    phi = int(totient(m))
                    # 返回计算结果 b^(phi + e%phi) % m 的整数值
                    return Integer(pow(b, phi + e % phi, m))
                # 返回计算结果 b^e % m 的整数值
                return Integer(pow(b, e, m))

            # 导入 Mod 类
            from .mod import Mod

            # 如果基数是 Pow 类的实例且是整数和数字
            if isinstance(base, Pow) and base.is_integer and base.is_number:
                # 将基数转换为 Mod 对象，以 q 为模
                base = Mod(base, q)
                # 返回基数和指数的 Mod 运算结果，评估设为 False
                return Mod(Pow(base, exp, evaluate=False), q)

            # 如果指数是 Pow 类的实例且是整数和数字
            if isinstance(exp, Pow) and exp.is_integer and exp.is_number:
                # 计算 q 的比特位长度
                bit_length = int(q).bit_length()
                # 如果比特位长度小于等于 80
                if bit_length <= 80:
                    # 计算 q 的欧拉函数值
                    phi = totient(q)
                    # 将指数转换为 Mod 对象，以 phi 为模
                    exp = phi + Mod(exp, phi)
                    # 返回基数和指数的 Mod 运算结果，评估设为 False
                    return Mod(Pow(base, exp, evaluate=False), q)

    # 定义 `_eval_is_even` 方法，用于判断指数是否为偶数
    def _eval_is_even(self):
        # 如果指数是整数且为正数
        if self.exp.is_integer and self.exp.is_positive:
            # 返回基数是否为偶数的判断结果
            return self.base.is_even

    # 定义 `_eval_is_negative` 方法，用于判断表达式是否为负数
    def _eval_is_negative(self):
        # 调用 Pow 类的 `_eval_is_extended_negative` 方法
        ext_neg = Pow._eval_is_extended_negative(self)
        # 如果 `ext_neg` 为 True，则返回表达式是否有限的判断结果
        if ext_neg is True:
            return self.is_finite
        # 否则返回 `ext_neg` 的值
        return ext_neg
    # 判断表达式是否为扩展正数
    def _eval_is_extended_positive(self):
        # 如果底数等于指数
        if self.base == self.exp:
            # 并且底数是扩展非负数，则返回True
            if self.base.is_extended_nonnegative:
                return True
        # 如果底数是正数
        elif self.base.is_positive:
            # 并且指数是实数，则返回True
            if self.exp.is_real:
                return True
        # 如果底数是扩展负数
        elif self.base.is_extended_negative:
            # 并且指数是偶数，则返回True
            if self.exp.is_even:
                return True
            # 如果指数是奇数，则返回False
            if self.exp.is_odd:
                return False
        # 如果底数是零
        elif self.base.is_zero:
            # 并且指数是扩展实数，则返回指数是否为零
            if self.exp.is_extended_real:
                return self.exp.is_zero
        # 如果底数是扩展非正数
        elif self.base.is_extended_nonpositive:
            # 并且指数是奇数，则返回False
            if self.exp.is_odd:
                return False
        # 如果底数是虚数
        elif self.base.is_imaginary:
            # 并且指数是整数
            if self.exp.is_integer:
                # 对指数取模4
                m = self.exp % 4
                # 如果余数为零，则返回True
                if m.is_zero:
                    return True
                # 如果余数是整数且不为零，则返回False
                if m.is_integer and m.is_zero is False:
                    return False
            # 如果指数也是虚数
            if self.exp.is_imaginary:
                # 导入对数函数
                from sympy.functions.elementary.exponential import log
                # 返回底数的对数是否为虚数
                return log(self.base).is_imaginary

    # 判断表达式是否为扩展负数
    def _eval_is_extended_negative(self):
        # 如果指数是1/2
        if self.exp is S.Half:
            # 如果底数是复数或者扩展实数，则返回False
            if self.base.is_complex or self.base.is_extended_real:
                return False
        # 如果底数是扩展负数
        if self.base.is_extended_negative:
            # 并且指数是奇数且底数是有限的，则返回True
            if self.exp.is_odd and self.base.is_finite:
                return True
            # 如果指数是偶数，则返回False
            if self.exp.is_even:
                return False
        # 如果底数是扩展正数
        elif self.base.is_extended_positive:
            # 并且指数是扩展实数，则返回False
            if self.exp.is_extended_real:
                return False
        # 如果底数是零
        elif self.base.is_zero:
            # 并且指数是扩展实数，则返回False
            if self.exp.is_extended_real:
                return False
        # 如果底数是扩展非负数
        elif self.base.is_extended_nonnegative:
            # 并且指数是扩展非负数，则返回False
            if self.exp.is_extended_nonnegative:
                return False
        # 如果底数是扩展非正数
        elif self.base.is_extended_nonpositive:
            # 并且指数是偶数，则返回False
            if self.exp.is_even:
                return False
        # 如果底数是扩展实数
        elif self.base.is_extended_real:
            # 并且指数是偶数，则返回False
            if self.exp.is_even:
                return False

    # 判断表达式是否为零
    def _eval_is_zero(self):
        # 如果底数是零
        if self.base.is_zero:
            # 并且指数是扩展正数，则返回True
            if self.exp.is_extended_positive:
                return True
            # 如果指数是扩展非正数，则返回False
            elif self.exp.is_extended_nonpositive:
                return False
        # 如果底数等于Euler数
        elif self.base == S.Exp1:
            # 返回指数是否为负无穷
            return self.exp is S.NegativeInfinity
        # 如果底数不是零
        elif self.base.is_zero is False:
            # 并且底数是有限的且指数也是有限的
            if self.base.is_finite and self.exp.is_finite:
                return False
            # 如果指数是负数，则返回底数是否为无穷大
            elif self.exp.is_negative:
                return self.base.is_infinite
            # 如果指数是非负数
            elif self.exp.is_nonnegative:
                # 并且指数是无穷大且扩展实数
                if self.exp.is_infinite and self.exp.is_extended_real:
                    # 如果（1 - 底数的绝对值）是扩展正数，则返回指数是否为扩展正数
                    if (1 - abs(self.base)).is_extended_positive:
                        return self.exp.is_extended_positive
                    # 如果（1 - 底数的绝对值）是扩展负数，则返回指数是否为扩展负数
                    elif (1 - abs(self.base)).is_extended_negative:
                        return self.exp.is_extended_negative
        # 如果底数是有限的且指数是负数
        elif self.base.is_finite and self.exp.is_negative:
            # 当底数是否为零时
            return False
    # 检查指数是否为整数的私有方法
    def _eval_is_integer(self):
        # 将参数拆分为底数和指数
        b, e = self.args
        # 如果底数是有理数
        if b.is_rational:
            # 如果底数不是整数且指数是正数，则返回 False，表示有理数的非负幂
            if b.is_integer is False and e.is_positive:
                return False  # rat**nonneg
        # 如果底数和指数都是整数
        if b.is_integer and e.is_integer:
            # 如果底数是 -1，则返回 True
            if b is S.NegativeOne:
                return True
            # 如果指数是非负数或正数，则返回 True
            if e.is_nonnegative or e.is_positive:
                return True
        # 如果底数是整数且指数是负数且指数为有限或整数
        if b.is_integer and e.is_negative and (e.is_finite or e.is_integer):
            # 如果底数减去 1 不是零且底数加上 1 不是零，则返回 False
            if fuzzy_not((b - 1).is_zero) and fuzzy_not((b + 1).is_zero):
                return False
        # 如果底数和指数都是数字
        if b.is_Number and e.is_Number:
            # 对当前对象应用函数，并检查结果是否为整数
            check = self.func(*self.args)
            return check.is_Integer
        # 如果指数为负且底数为正且底数减去 1 为正，则返回 False
        if e.is_negative and b.is_positive and (b - 1).is_positive:
            return False
        # 如果指数为负且底数为负且底数加上 1 为负，则返回 False
        if e.is_negative and b.is_negative and (b + 1).is_negative:
            return False

    # 检查表达式是否为复数的私有方法
    def _eval_is_complex(self):

        # 如果底数是自然对数 e
        if self.base == S.Exp1:
            return fuzzy_or([self.exp.is_complex, self.exp.is_extended_negative])

        # 如果所有参数都是复数并且表达式的有限性评估结果为真
        if all(a.is_complex for a in self.args) and self._eval_is_finite():
            return True

    # 检查表达式是否为虚数的私有方法
    def _eval_is_imaginary(self):
        # 如果底数不是可交换的，则返回 False
        if self.base.is_commutative is False:
            return False

        # 如果底数是虚数且指数是整数
        if self.base.is_imaginary:
            if self.exp.is_integer:
                odd = self.exp.is_odd
                if odd is not None:
                    return odd
                return

        # 如果底数是 e （自然对数）
        if self.base == S.Exp1:
            f = 2 * self.exp / (S.Pi*S.ImaginaryUnit)
            # 如果 f 是偶数，则返回 False
            if f.is_even:
                return False
            # 如果 f 是奇数，则返回 True
            if f.is_odd:
                return True
            return None

        # 如果指数是虚数
        if self.exp.is_imaginary:
            from sympy.functions.elementary.exponential import log
            imlog = log(self.base).is_imaginary
            if imlog is not None:
                return False  # I**i -> real; (2*I)**i -> complex ==> not imaginary

        # 如果底数和指数都是扩展实数且底数为正数，则返回 False
        if self.base.is_extended_real and self.exp.is_extended_real:
            if self.base.is_positive:
                return False
            else:
                rat = self.exp.is_rational
                if not rat:
                    return rat
                if self.exp.is_integer:
                    return False
                else:
                    half = (2*self.exp).is_integer
                    if half:
                        return self.base.is_negative
                    return half

        # 如果底数不是扩展实数，则我们已经知道它不是虚数
        if self.base.is_extended_real is False:  # we already know it's not imag
            from sympy.functions.elementary.complexes import arg
            i = arg(self.base)*self.exp/S.Pi
            isodd = (2*i).is_odd
            if isodd is not None:
                return isodd
    # 检查指数是否为整数
    def _eval_is_odd(self):
        if self.exp.is_integer:
            # 如果指数是正整数，返回基数是否为奇数
            if self.exp.is_positive:
                return self.base.is_odd
            # 如果指数是非负整数且基数为奇数，返回True
            elif self.exp.is_nonnegative and self.base.is_odd:
                return True
            # 如果基数为-1，返回True
            elif self.base is S.NegativeOne:
                return True

    # 检查表达式是否有限
    def _eval_is_finite(self):
        if self.exp.is_negative:
            # 如果指数为负数且基数为零，返回False
            if self.base.is_zero:
                return False
            # 如果基数为无穷大或非零，返回True
            if self.base.is_infinite or self.base.is_nonzero:
                return True
        # 检查基数和指数是否都是有限的
        c1 = self.base.is_finite
        if c1 is None:
            return
        c2 = self.exp.is_finite
        if c2 is None:
            return
        # 如果基数和指数都是有限的，且指数是非负数或基数非零，则返回True
        if c1 and c2:
            if self.exp.is_nonnegative or fuzzy_not(self.base.is_zero):
                return True

    # 检查数是否为质数
    def _eval_is_prime(self):
        '''
        如果基数和指数都是整数且指数大于等于2，则不可能是质数。
        '''
        if self.base.is_integer and self.exp.is_integer and (self.exp - 1).is_positive:
            return False

    # 检查数是否为合数
    def _eval_is_composite(self):
        """
        如果基数和指数都大于1，则幂是合数。
        """
        if (self.base.is_integer and self.exp.is_integer and
            ((self.base - 1).is_positive and (self.exp - 1).is_positive or
            (self.base + 1).is_negative and self.exp.is_positive and self.exp.is_even)):
            return True

    # 检查数是否为极坐标形式
    def _eval_is_polar(self):
        return self.base.is_polar

    # 返回幂的基数和指数
    def as_base_exp(self):
        """Return base and exp of self.

        Explanation
        ===========

        如果基数是小于1的有理数，则返回1/基数, -指数。
        如果不需要额外处理，可以直接使用基数和指数属性提供的原始参数。

        Examples
        ========

        >>> from sympy import Pow, S
        >>> p = Pow(S.Half, 2, evaluate=False)
        >>> p.as_base_exp()
        (2, -2)
        >>> p.args
        (1/2, 2)
        >>> p.base, p.exp
        (1/2, 2)

        """

        b, e = self.args
        if b.is_Rational and b.p < b.q and b.p > 0:
            return 1/b, -e
        return b, e

    # 返回幂的伴随
    def _eval_adjoint(self):
        from sympy.functions.elementary.complexes import adjoint
        i, p = self.exp.is_integer, self.base.is_positive
        if i:
            return adjoint(self.base)**self.exp
        if p:
            return self.base**adjoint(self.exp)
        if i is False and p is False:
            expanded = expand_complex(self)
            if expanded != self:
                return adjoint(expanded)
    # 计算复数的共轭
    def _eval_conjugate(self):
        # 导入共轭函数
        from sympy.functions.elementary.complexes import conjugate as c
        # 检查指数是否整数和底数是否正数
        i, p = self.exp.is_integer, self.base.is_positive
        # 如果指数是整数，应用共轭到底数的指数次方
        if i:
            return c(self.base)**self.exp
        # 如果底数是正数，将指数应用到共轭后的底数
        if p:
            return self.base**c(self.exp)
        # 如果指数不是整数且底数不是正数
        if i is False and p is False:
            # 扩展复数表达式并共轭
            expanded = expand_complex(self)
            # 如果扩展后与原来不同，则返回其共轭
            if expanded != self:
                return c(expanded)
        # 如果表达式是扩展实数，则返回其本身
        if self.is_extended_real:
            return self

    # 计算复数的转置
    def _eval_transpose(self):
        # 导入转置函数
        from sympy.functions.elementary.complexes import transpose
        # 如果底数是自然常数e，将转置应用到指数上
        if self.base == S.Exp1:
            return self.func(S.Exp1, self.exp.transpose())
        # 检查指数是否整数和底数是否复数或无穷
        i, p = self.exp.is_integer, (self.base.is_complex or self.base.is_infinite)
        # 如果底数是复数，将指数应用到底数上
        if p:
            return self.base**self.exp
        # 如果指数是整数，将转置应用到底数上的指数次方
        if i:
            return transpose(self.base)**self.exp
        # 如果指数不是整数且底数不是复数或无穷
        if i is False and p is False:
            # 扩展复数表达式并转置
            expanded = expand_complex(self)
            # 如果扩展后与原来不同，则返回其转置
            if expanded != self:
                return transpose(expanded)

    # 展开指数和的幂
    def _eval_expand_power_exp(self, **hints):
        """a**(n + m) -> a**n*a**m"""
        # 获取底数和指数
        b = self.base
        e = self.exp
        # 如果底数是自然常数e
        if b == S.Exp1:
            # 导入求和和积分的类
            from sympy.concrete.summations import Sum
            # 如果指数是求和并且是可交换的
            if isinstance(e, Sum) and e.is_commutative:
                # 导入乘积类并返回乘积形式
                from sympy.concrete.products import Product
                return Product(self.func(b, e.function), *e.limits)
        # 如果指数是加法，并且满足强制要求或者底数不是零或者指数都是非负数
        if e.is_Add and (hints.get('force', False) or
                b.is_zero is False or e._all_nonneg_or_nonppos()):
            # 如果指数是可交换的
            if e.is_commutative:
                # 返回乘积表达式
                return Mul(*[self.func(b, x) for x in e.args])
            # 如果底数是可交换的
            if b.is_commutative:
                # 将元组分为可交换和非可交换
                c, nc = sift(e.args, lambda x: x.is_commutative, binary=True)
                # 如果是可交换的
                if c:
                    # 返回乘积表达式
                    return Mul(*[self.func(b, x) for x in c]
                        )*b**Add._from_args(nc)
        # 返回自身
        return self

    # 计算导数
    def _eval_derivative(self, s):
        # 导入对数函数
        from sympy.functions.elementary.exponential import log
        # 计算底数和指数的导数
        dbase = self.base.diff(s)
        dexp = self.exp.diff(s)
        # 返回导数计算结果
        return self * (dexp * log(self.base) + dbase * self.exp/self.base)

    # 计算浮点数值
    def _eval_evalf(self, prec):
        # 获取底数和指数
        base, exp = self.as_base_exp()
        # 如果底数是自然常数e
        if base == S.Exp1:
            # 使用与类"exp"相关的mpmath函数
            from sympy.functions.elementary.exponential import exp as exp_function
            return exp_function(self.exp, evaluate=False)._eval_evalf(prec)
        # 计算底数的浮点数值
        base = base._evalf(prec)
        # 如果指数不是整数
        if not exp.is_Integer:
            exp = exp._evalf(prec)
        # 如果指数是负数且底数是数字且不是扩展实数
        if exp.is_negative and base.is_number and base.is_extended_real is False:
            # 返回共轭的底数除以底数的共轭的浮点数值
            base = base.conjugate() / (base * base.conjugate())._evalf(prec)
            exp = -exp
            return self.func(base, exp).expand()
        # 返回自身
        return self
    # 检查指数表达式中是否包含给定的符号，如果包含则不是多项式
    def _eval_is_polynomial(self, syms):
        if self.exp.has(*syms):
            return False
        
        # 如果底数表达式中包含给定的符号，则判断为多项式的条件更严格
        if self.base.has(*syms):
            return bool(self.base._eval_is_polynomial(syms) and
                self.exp.is_Integer and (self.exp >= 0))
        else:
            return True

    # 判断幂函数是否为有理数
    def _eval_is_rational(self):
        # 对于整数**整数的情况，如果指数很大，评估 self.func 可能非常昂贵。
        # 在必要之前应尽早退出：
        if (self.exp.is_integer and self.base.is_rational
                and fuzzy_not(fuzzy_and([self.exp.is_negative, self.base.is_zero]))):
            return True
        
        p = self.func(*self.as_base_exp())  # 如果尚未评估，则评估为幂函数对象
        if not p.is_Pow:
            return p.is_rational
        
        b, e = p.as_base_exp()
        if e.is_Rational and b.is_Rational:
            # 没有检查 e 不是整数，因为 Rational**Integer 会自动简化
            return False
        
        if e.is_integer:
            if b.is_rational:
                if fuzzy_not(b.is_zero) or e.is_nonnegative:
                    return True
                if b == e:  # 即使对于 0**0，也始终是有理数
                    return True
            elif b.is_irrational:
                return e.is_zero
        
        if b is S.Exp1:
            if e.is_rational and e.is_nonzero:
                return False

    # 判断幂函数是否为代数数
    def _eval_is_algebraic(self):
        def _is_one(expr):
            try:
                return (expr - 1).is_zero
            except ValueError:
                # 当操作不被允许时
                return False

        if self.base.is_zero or _is_one(self.base):
            return True
        elif self.base is S.Exp1:
            s = self.func(*self.args)
            if s.func == self.func:
                if self.exp.is_nonzero:
                    if self.exp.is_algebraic:
                        return False
                    elif (self.exp/S.Pi).is_rational:
                        return False
                    elif (self.exp/(S.ImaginaryUnit*S.Pi)).is_rational:
                        return True
            else:
                return s.is_algebraic
        elif self.exp.is_rational:
            if self.base.is_algebraic is False:
                return self.exp.is_zero
            if self.base.is_zero is False:
                if self.exp.is_nonzero:
                    return self.base.is_algebraic
                elif self.base.is_algebraic:
                    return True
            if self.exp.is_positive:
                return self.base.is_algebraic
        elif self.base.is_algebraic and self.exp.is_algebraic:
            if ((fuzzy_not(self.base.is_zero)
                and fuzzy_not(_is_one(self.base)))
                or self.base.is_integer is False
                or self.base.is_irrational):
                return self.exp.is_rational
    def _eval_is_rational_function(self, syms):
        # 检查表达式中是否包含给定符号，如果有，则不是有理函数
        if self.exp.has(*syms):
            return False

        # 如果底数中包含给定符号，则需要进一步判断底数和指数是否符合有理函数的定义
        if self.base.has(*syms):
            return self.base._eval_is_rational_function(syms) and \
                self.exp.is_Integer
        else:
            return True

    def _eval_is_meromorphic(self, x, a):
        # 如果指数是整数且底数是亚黎曼函数，则整体是亚黎曼函数
        base_merom = self.base._eval_is_meromorphic(x, a)
        exp_integer = self.exp.is_Integer
        if exp_integer:
            return base_merom

        # 如果指数是整数，但底数不是亚黎曼函数，则根据指数的亚黎曼性质判断
        exp_merom = self.exp._eval_is_meromorphic(x, a)
        if base_merom is False:
            # 如果底数不是亚黎曼函数，且指数也不是亚黎曼函数，则返回 False
            return False if exp_merom else None
        elif base_merom is None:
            return None

        b = self.base.subs(x, a)
        # 当底数为亚黎曼函数时，基本上底数的对数是有限和亚黎曼的
        b_zero = b.is_zero
        if b_zero:
            log_defined = False
        else:
            log_defined = fuzzy_and((b.is_finite, fuzzy_not(b_zero)))

        if log_defined is False: # 底数为零或极点
            return exp_integer  # 返回 False 或 None
        elif log_defined is None:
            return None

        if not exp_merom:
            return exp_merom  # 返回 False 或 None

        return self.exp.subs(x, a).is_finite

    def _eval_is_algebraic_expr(self, syms):
        # 检查表达式中是否包含给定符号，如果有，则不是代数表达式
        if self.exp.has(*syms):
            return False

        # 如果底数中包含给定符号，则需要进一步判断底数和指数是否符合代数表达式的定义
        if self.base.has(*syms):
            return self.base._eval_is_algebraic_expr(syms) and \
                self.exp.is_Rational
        else:
            return True

    def _eval_rewrite_as_exp(self, base, expo, **kwargs):
        from sympy.functions.elementary.exponential import exp, log

        # 如果底数为零或包含指数函数，则直接返回底数的指数幂
        if base.is_zero or base.has(exp) or expo.has(exp):
            return base**expo

        evaluate = expo.has(Symbol)

        if base.has(Symbol):
            # 如果底数包含符号，根据全局参数决定是否延迟评估
            if global_parameters.exp_is_pow:
                return Pow(S.Exp1, log(base)*expo, evaluate=evaluate)
            else:
                return exp(log(base)*expo, evaluate=evaluate)

        else:
            from sympy.functions.elementary.complexes import arg, Abs
            return exp((log(Abs(base)) + S.ImaginaryUnit*arg(base))*expo)
    # 将表达式转换为分子和分母形式的元组
    def as_numer_denom(self):
        # 如果表达式不可交换，则返回自身和单位元1
        if not self.is_commutative:
            return self, S.One
        # 将表达式分解为底数和指数
        base, exp = self.as_base_exp()
        # 递归调用底数的as_numer_denom方法，获取其分子和分母
        n, d = base.as_numer_denom()
        
        # 处理指数的正负情况，确保处理与ExpBase.as_numer_denom相同
        neg_exp = exp.is_negative
        if exp.is_Mul and not neg_exp and not exp.is_positive:
            neg_exp = exp.could_extract_minus_sign()
        int_exp = exp.is_integer
        
        # 如果分母d不是扩展实数或者指数exp不是整数，分子n直接为base，分母d设为1
        if not (d.is_extended_real or int_exp):
            n = base
            d = S.One
        
        # 检查分母的非正特性
        dnonpos = d.is_nonpositive
        if dnonpos:
            n, d = -n, -d
        # 如果分母的非正特性未知且指数exp不是整数，重置n为base，d为1
        elif dnonpos is None and not int_exp:
            n = base
            d = S.One
        
        # 处理指数为负数的情况，交换分子和分母，并取指数的负值
        if neg_exp:
            n, d = d, n
            exp = -exp
        
        # 如果指数为无穷大
        if exp.is_infinite:
            # 如果分子n是1且分母d不是1，则返回分子为n，分母为self.func(d, exp)构造的结果
            if n is S.One and d is not S.One:
                return n, self.func(d, exp)
            # 如果分子n不是1且分母d是1，则返回分子为self.func(n, exp)，分母为d
            if n is not S.One and d is S.One:
                return self.func(n, exp), d
        
        # 返回结果，分子为self.func(n, exp)，分母为self.func(d, exp)
        return self.func(n, exp), self.func(d, exp)

    # 判断当前表达式与给定表达式expr是否匹配
    def matches(self, expr, repl_dict=None, old=False):
        # 将expr转换为符号表达式
        expr = _sympify(expr)
        # 如果repl_dict为None，则初始化为空字典
        if repl_dict is None:
            repl_dict = {}

        # 特殊情况，当pattern为1且expr的指数可以匹配为0时
        if expr is S.One:
            d = self.exp.matches(S.Zero, repl_dict)
            if d is not None:
                return d
        
        # 确保要匹配的表达式是Expr类型
        if not isinstance(expr, Expr):
            return None
        
        # 将表达式expr分解为底数和指数
        b, e = expr.as_base_exp()
        
        # 处理特殊情况，当sb为Symbol且se为Integer，且expr不为空
        sb, se = self.as_base_exp()
        if sb.is_Symbol and se.is_Integer and expr:
            # 如果e是有理数，使用sb.matches(b**(e/se), repl_dict)进行匹配
            if e.is_rational:
                return sb.matches(b**(e/se), repl_dict)
            # 否则使用sb.matches(expr**(1/se), repl_dict)进行匹配
            return sb.matches(expr**(1/se), repl_dict)
        
        # 复制repl_dict为d
        d = repl_dict.copy()
        # 对基数进行匹配，并更新d
        d = self.base.matches(b, d)
        if d is None:
            return None
        
        # 对指数进行替换并匹配更新后的表达式e，并更新d
        d = self.exp.xreplace(d).matches(e, d)
        if d is None:
            return Expr.matches(self, expr, repl_dict)
        # 返回匹配结果d
        return d
    # 根据给定的自变量 x，计算作为主导项的表达式值
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 导入必要的函数库
        from sympy.functions.elementary.exponential import exp, log
        e = self.exp  # 提取指数部分
        b = self.base  # 提取底数部分
        if self.base is S.Exp1:  # 如果底数是自然指数 e
            arg = e.as_leading_term(x, logx=logx)  # 计算指数部分的主导项
            arg0 = arg.subs(x, 0)  # 在 x=0 处求主导项的值
            if arg0 is S.NaN:  # 如果结果为 NaN
                arg0 = arg.limit(x, 0)  # 尝试在 x=0 处求极限
            if arg0.is_infinite is False:  # 如果不是无穷大
                return S.Exp1**arg0  # 返回 e 的主导项的 x 次幂
            raise PoleError("Cannot expand %s around 0" % (self))  # 若无法展开，则抛出异常
        elif e.has(x):  # 如果指数部分包含自变量 x
            lt = exp(e * log(b))  # 计算指数函数形式的主导项
            return lt.as_leading_term(x, logx=logx, cdir=cdir)  # 返回指数函数的主导项
        else:  # 否则
            from sympy.functions.elementary.complexes import im
            try:
                f = b.as_leading_term(x, logx=logx, cdir=cdir)  # 计算底数的主导项
            except PoleError:
                return self
            if not e.is_integer and f.is_negative and not f.has(x):  # 如果指数非整数且底数为负数且不包含自变量 x
                ndir = (b - f).dir(x, cdir)  # 计算方向导数
                if im(ndir).is_negative:
                    # 在分支切割处，f**e 通常会被计算为 exp(e*log(f))，但是在分支切割处，期望通过以下计算得到另一个值
                    # exp(e*(log(f) - 2*pi*I)) == f**e*exp(-2*e*pi*I) == f**e*(-1)**(-2*e)。
                    return self.func(f, e) * (-1)**(-2*e)  # 返回修正的表达式值
                elif im(ndir).is_zero:
                    log_leadterm = log(b)._eval_as_leading_term(x, logx=logx, cdir=cdir)  # 计算对数函数的主导项
                    if log_leadterm.is_infinite is False:  # 如果不是无穷大
                        return exp(e*log_leadterm)  # 返回指数函数的主导项
            return self.func(f, e)  # 返回原始的表达式值

    @cacheit
    # 返回泰勒级数的第 n 项表达式，对于 (1 + x)**e
    def _taylor_term(self, n, x, *previous_terms):
        # 导入必要的函数库
        from sympy.functions.combinatorial.factorials import binomial
        return binomial(self.exp, n) * self.func(x, n)  # 计算泰勒级数的第 n 项表达式

    # 返回泰勒级数的第 n 项表达式，对于底数为自然指数的情况
    def taylor_term(self, n, x, *previous_terms):
        if self.base is not S.Exp1:  # 如果底数不是自然指数 e
            return super().taylor_term(n, x, *previous_terms)  # 调用父类的方法计算泰勒级数项
        if n < 0:  # 如果 n 小于 0
            return S.Zero  # 返回 0
        if n == 0:  # 如果 n 等于 0
            return S.One  # 返回 1
        from .sympify import sympify
        x = sympify(x)  # 将 x 转换为符号变量
        if previous_terms:  # 如果有之前的项
            p = previous_terms[-1]  # 取出最后一项
            if p is not None:  # 如果最后一项不为 None
                return p * x / n  # 计算泰勒级数项
        from sympy.functions.combinatorial.factorials import factorial
        return x**n/factorial(n)  # 计算泰勒级数项

    # 将表达式重写为以 sin 函数形式表示
    def _eval_rewrite_as_sin(self, base, exp, **hints):
        if self.base is S.Exp1:  # 如果底数是自然指数 e
            from sympy.functions.elementary.trigonometric import sin
            return sin(S.ImaginaryUnit*self.exp + S.Pi/2) - S.ImaginaryUnit*sin(S.ImaginaryUnit*self.exp)

    # 将表达式重写为以 cos 函数形式表示
    def _eval_rewrite_as_cos(self, base, exp, **hints):
        if self.base is S.Exp1:  # 如果底数是自然指数 e
            from sympy.functions.elementary.trigonometric import cos
            return cos(S.ImaginaryUnit*self.exp) + S.ImaginaryUnit*cos(S.ImaginaryUnit*self.exp + S.Pi/2)
    # 将指数函数重写为双曲正切的形式
    def _eval_rewrite_as_tanh(self, base, exp, **hints):
        # 如果底数是自然常数 e
        if self.base is S.Exp1:
            # 导入双曲正切函数
            from sympy.functions.elementary.hyperbolic import tanh
            # 根据公式重写表达式
            return (1 + tanh(self.exp/2))/(1 - tanh(self.exp/2))

    # 将指数函数重写为正弦和余弦函数的形式
    def _eval_rewrite_as_sqrt(self, base, exp, **kwargs):
        # 如果底数不是自然常数 e，则返回 None
        if base is not S.Exp1:
            return None
        # 如果指数是乘积
        if exp.is_Mul:
            # 提取乘积中的系数
            coeff = exp.coeff(S.Pi * S.ImaginaryUnit)
            # 如果系数是数字
            if coeff and coeff.is_number:
                # 导入正弦和余弦函数
                from sympy.functions.elementary.trigonometric import sin, cos
                # 计算正弦和余弦值
                cosine, sine = cos(S.Pi*coeff), sin(S.Pi*coeff)
                # 如果正弦和余弦值不是其函数类型
                if not isinstance(cosine, cos) and not isinstance(sine, sin):
                    # 返回计算结果
                    return cosine + S.ImaginaryUnit*sine

    # 判断表达式是否是常数
    def is_constant(self, *wrt, **flags):
        # 将表达式赋给局部变量 expr
        expr = self
        # 如果需要简化表达式
        if flags.get('simplify', True):
            # 简化表达式
            expr = expr.simplify()
        # 将表达式转换为底数和指数的形式
        b, e = expr.as_base_exp()
        # 检查底数是否为零
        bz = b.equals(0)
        # 如果底数为零
        if bz:
            # 在假设条件下重新计算以确保表达式已评估
            new = b**e
            # 如果新计算的结果不等于原表达式
            if new != expr:
                # 递归检查新表达式是否是常数
                return new.is_constant()
        # 检查指数是否是常数
        econ = e.is_constant(*wrt)
        # 检查底数是否是常数
        bcon = b.is_constant(*wrt)
        # 如果底数是常数
        if bcon:
            # 如果指数也是常数，则整个表达式是常数
            if econ:
                return True
            # 否则，再次检查底数是否为零
            bz = b.equals(0)
            # 如果底数不为零
            if bz is False:
                return False
        # 如果底数可能是常数，但无法确定
        elif bcon is None:
            return None

        # 如果以上条件都不满足，则表达式不是常数
        return e.equals(0)

    # 计算表达式的差分值
    def _eval_difference_delta(self, n, step):
        # 将表达式分解为底数和指数
        b, e = self.args
        # 如果指数包含变量 n，而底数不包含变量 n
        if e.has(n) and not b.has(n):
            # 计算新的指数值
            new_e = e.subs(n, n + step)
            # 返回差分值的计算结果
            return (b**(new_e - e) - 1) * self
# 创建一个名为 power 的调度器对象，用于管理函数的多态性和分发
power = Dispatcher('power')

# 将 (object, object) 类型的参数对应到 Pow 函数上，这是 power 调度器的一条规则
power.add((object, object), Pow)

# 从当前包的 add 模块导入 Add 类
from .add import Add

# 从当前包的 numbers 模块导入 Integer 和 Rational 类
from .numbers import Integer, Rational

# 从当前包的 mul 模块导入 Mul 和 _keep_coeff 函数
from .mul import Mul, _keep_coeff

# 从当前包的 symbol 模块导入 Symbol, Dummy 和 symbols 函数
from .symbol import Symbol, Dummy, symbols
```