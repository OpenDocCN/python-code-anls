# `D:\src\scipysrc\sympy\sympy\functions\combinatorial\factorials.py`

```
# 导入将来的注释，允许使用类型提示
from __future__ import annotations
# 导入 functools 库中的 reduce 函数
from functools import reduce

# 导入 sympy 库中的核心模块和函数
from sympy.core import S, sympify, Dummy, Mod
# 导入 sympy 库中的缓存功能
from sympy.core.cache import cacheit
# 导入 sympy 库中的函数类、异常和逻辑操作
from sympy.core.function import Function, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and
# 导入 sympy 库中的数字类型和常数
from sympy.core.numbers import Integer, pi, I
# 导入 sympy 库中的关系运算
from sympy.core.relational import Eq
# 导入 sympy.external.gmpy 中的 gmpy 库
from sympy.external.gmpy import gmpy as _gmpy
# 导入 sympy.ntheory 中的筛法和模重数理论
from sympy.ntheory import sieve
from sympy.ntheory.residue_ntheory import binomial_mod
# 导入 sympy.polys.polytools 中的多项式工具
from sympy.polys.polytools import Poly

# 导入 math 库中的 factorial 函数并别名为 _factorial，以及 prod 和 sqrt 函数别名
from math import factorial as _factorial, prod, sqrt as _sqrt


class CombinatorialFunction(Function):
    """组合函数的基类。"""

    def _eval_simplify(self, **kwargs):
        # 导入 combsimp 函数并使用它来简化组合函数表达式
        from sympy.simplify.combsimp import combsimp
        # 当组合函数的参数不是整数时，自动传递给 gammasimp 函数处理
        expr = combsimp(self)
        # 获取参数中的测量函数
        measure = kwargs['measure']
        # 如果简化后的表达式小于等于原始表达式的指定比率乘以测量值，则返回简化后的表达式
        if measure(expr) <= kwargs['ratio'] * measure(self):
            return expr
        # 否则返回原始表达式
        return self


###############################################################################
######################## FACTORIAL and MULTI-FACTORIAL ########################
###############################################################################


class factorial(CombinatorialFunction):
    r"""非负整数阶乘函数的实现。
       根据约定（与伽玛函数和二项式系数一致），负整数的阶乘被定义为复无穷。

       阶乘在组合学中非常重要，它表示将 n 个对象进行排列的方式数。
       它也出现在微积分、概率、数论等领域。

       阶乘与伽玛函数有严格的关系。事实上，对于非负整数，n! = gamma(n+1)。
       这种重写在组合简化中非常有用。

       阶乘的计算使用两种算法。对于小的参数，使用预先计算的查找表。
       对于较大的输入，使用 Prime-Swing 算法。它是已知的最快算法，
       通过对特定类别（称为“Swing Numbers”）的数的素因数分解来计算 n!。

       示例
       ========

       >>> from sympy import Symbol, factorial, S
       >>> n = Symbol('n', integer=True)

       >>> factorial(0)
       1

       >>> factorial(7)
       5040

       >>> factorial(-2)
       zoo

       >>> factorial(n)
       factorial(n)

       >>> factorial(2*n)
       factorial(2*n)

       >>> factorial(S(1)/2)
       factorial(1/2)

       参见
       ========

       factorial2, RisingFactorial, FallingFactorial
    """
    def fdiff(self, argindex=1):
        # 导入需要的数学函数
        from sympy.functions.special.gamma_functions import (gamma, polygamma)
        # 如果参数索引为1，计算并返回 gamma 函数和 polygamma 函数的乘积
        if argindex == 1:
            return gamma(self.args[0] + 1)*polygamma(0, self.args[0] + 1)
        else:
            # 抛出参数索引错误异常
            raise ArgumentIndexError(self, argindex)

    _small_swing = [
        1, 1, 1, 3, 3, 15, 5, 35, 35, 315, 63, 693, 231, 3003, 429, 6435, 6435, 109395,
        12155, 230945, 46189, 969969, 88179, 2028117, 676039, 16900975, 1300075,
        35102025, 5014575, 145422675, 9694845, 300540195, 300540195
    ]

    _small_factorials: list[int] = []

    @classmethod
    def _swing(cls, n):
        # 如果 n 小于 33，直接返回预定义的小幅度振荡数组中的值
        if n < 33:
            return cls._small_swing[n]
        else:
            # 计算需要的素数和产品
            N, primes = int(_sqrt(n)), []

            # 使用筛法找出小于 N+1 的所有素数
            for prime in sieve.primerange(3, N + 1):
                p, q = 1, n

                # 计算 p 的值，直到 q 为奇数
                while True:
                    q //= prime

                    if q > 0:
                        if q & 1 == 1:
                            p *= prime
                    else:
                        break

                # 如果 p 大于 1，将其加入素数列表
                if p > 1:
                    primes.append(p)

            # 继续找出大于 N+1 小于等于 n//3 的所有素数
            for prime in sieve.primerange(N + 1, n//3 + 1):
                # 如果 n // prime 为奇数，将其加入素数列表
                if (n // prime) & 1 == 1:
                    primes.append(prime)

            # 计算左半部分和右半部分的乘积并返回结果
            L_product = prod(sieve.primerange(n//2 + 1, n + 1))
            R_product = prod(primes)

            return L_product*R_product

    @classmethod
    def _recursive(cls, n):
        # 如果 n 小于 2，直接返回 1
        if n < 2:
            return 1
        else:
            # 否则，递归计算结果并返回
            return (cls._recursive(n//2)**2)*cls._swing(n)

    @classmethod
    # 定义一个类方法 `eval`，用于计算给定表达式 `n` 的值
    def eval(cls, n):
        # 将输入的表达式 `n` 转换为 SymPy 表达式
        n = sympify(n)

        # 检查 `n` 是否为数值类型
        if n.is_Number:
            # 如果 `n` 是零
            if n.is_zero:
                return S.One
            # 如果 `n` 是正无穷
            elif n is S.Infinity:
                return S.Infinity
            # 如果 `n` 是整数
            elif n.is_Integer:
                # 如果 `n` 是负整数
                if n.is_negative:
                    return S.ComplexInfinity
                else:
                    # 获取 `n` 的整数部分
                    n = n.p

                    # 如果 `n` 小于 20
                    if n < 20:
                        # 如果 `_small_factorials` 尚未初始化
                        if not cls._small_factorials:
                            result = 1
                            # 计算小于 20 的阶乘并缓存
                            for i in range(1, 20):
                                result *= i
                                cls._small_factorials.append(result)
                        # 返回预计算的小于 20 的阶乘值
                        result = cls._small_factorials[n-1]

                    # 如果有 GMPY 库，则使用其阶乘计算（更快）
                    #
                    # XXX: 在 sympy.external.gmpy.factorial 函数中提供 gmpy.fac（如果可用），
                    # 或者使用 flint 版本。可以在这里使用它，以避免条件逻辑，但需要检查
                    # 纯 Python 回退是否与此处使用的回退一样快（也许这里的回退应该移动到 sympy.external.ntheory）。
                    elif _gmpy is not None:
                        result = _gmpy.fac(n)

                    else:
                        # 计算 `n` 的二进制表示中 '1' 的个数
                        bits = bin(n).count('1')
                        # 使用递归算法计算阶乘，结合二进制位移优化
                        result = cls._recursive(n)*2**(n - bits)

                    # 返回结果的整数表示
                    return Integer(result)

    # 定义一个实例方法 `_facmod`，用于计算 `n` 阶乘模 `q` 的结果
    def _facmod(self, n, q):
        # 初始化结果和 n 的平方根的整数部分 N
        res, N = 1, int(_sqrt(n))

        # 计算 n! 中每个素数 p 的指数 e_p(n) = [n/p] + [n/p**2] + ...
        # 对于大于 sqrt(n) 的素数 p，e_p(n) < sqrt(n)，具有 [n/p] = m，
        # 出现连续的 m，并在稍后的阶段同时指数化到 pw[m] 中
        pw = [1]*N

        m = 2 # 初始化以下 if 条件
        # 遍历素数范围为 [2, n + 1]
        for prime in sieve.primerange(2, n + 1):
            if m > 1:
                m, y = 0, n // prime
                while y:
                    m += y
                    y //= prime
            # 如果 m 小于 N，则将 prime 的幂模 q 存储在 pw[m] 中
            if m < N:
                pw[m] = pw[m]*prime % q
            else:
                # 否则，使用快速幂求解并模 q
                res = res*pow(prime, m, q) % q

        # 对 pw 中每个指数和其对应的底数进行最终的模 q 运算
        for ex, bs in enumerate(pw):
            if ex == 0 or bs == 1:
                continue
            if bs == 0:
                return 0
            res = res*pow(bs, ex, q) % q

        # 返回最终的结果
        return res
    # 定义一个函数 `_eval_Mod`，用于计算模运算结果
    def _eval_Mod(self, q):
        # 从参数中获取第一个参数并赋值给变量 n
        n = self.args[0]
        # 检查 n 是否为整数且非负，q 是否为整数
        if n.is_integer and n.is_nonnegative and q.is_integer:
            # 计算 q 的绝对值
            aq = abs(q)
            # 计算 aq - n 的结果
            d = aq - n
            # 如果 d 非正，则返回零
            if d.is_nonpositive:
                return S.Zero
            else:
                # 检查 aq 是否为素数
                isprime = aq.is_prime
                # 如果 d 等于 1
                if d == 1:
                    # 应用威尔逊定理（如果自然数 n > 1 是素数，则 (n-1)! = -1 mod n）
                    # 及其逆定理（如果 n > 4 是合数，则 (n-1)! = 0 mod n）
                    if isprime:
                        # 返回 -1 对 q 取模的结果
                        return -1 % q
                    # 如果不是素数且 aq - 6 非负
                    elif isprime is False and (aq - 6).is_nonnegative:
                        # 返回零
                        return S.Zero
                # 如果 n 和 q 都是整数
                elif n.is_Integer and q.is_Integer:
                    # 将 n, d, aq 转换为整数类型
                    n, d, aq = map(int, (n, d, aq))
                    # 如果是素数且 (d - 1) < n
                    if isprime and (d - 1 < n):
                        # 计算 (d - 1)! 对 aq 取模的结果
                        fc = self._facmod(d - 1, aq)
                        # 计算 fc 对 aq - 2 幂运算的结果
                        fc = pow(fc, aq - 2, aq)
                        # 如果 d 是奇数，将 fc 取反
                        if d % 2:
                            fc = -fc
                    else:
                        # 否则，计算 n! 对 aq 取模的结果
                        fc = self._facmod(n, aq)

                    return fc % q

    # 将函数 `_eval_rewrite_as_gamma` 重写为 gamma 函数
    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        # 导入 gamma 函数
        from sympy.functions.special.gamma_functions import gamma
        # 返回 gamma(n + 1) 的结果
        return gamma(n + 1)

    # 将函数 `_eval_rewrite_as_Product` 重写为乘积形式
    def _eval_rewrite_as_Product(self, n, **kwargs):
        # 导入乘积函数
        from sympy.concrete.products import Product
        # 如果 n 非负且为整数
        if n.is_nonnegative and n.is_integer:
            # 创建一个整数类型的虚拟变量 i
            i = Dummy('i', integer=True)
            # 返回从 1 到 n 的乘积形式
            return Product(i, (i, 1, n))

    # 判断对象是否为整数
    def _eval_is_integer(self):
        # 如果对象的第一个参数是整数且非负
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    # 判断对象是否为正数
    def _eval_is_positive(self):
        # 如果对象的第一个参数是整数且非负
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    # 判断对象是否为偶数
    def _eval_is_even(self):
        # 获取对象的第一个参数
        x = self.args[0]
        # 如果 x 是整数且非负，返回 (x - 2) 是否非负的结果
        if x.is_integer and x.is_nonnegative:
            return (x - 2).is_nonnegative

    # 判断对象是否为合数
    def _eval_is_composite(self):
        # 获取对象的第一个参数
        x = self.args[0]
        # 如果 x 是整数且非负，返回 (x - 3) 是否非负的结果
        if x.is_integer and x.is_nonnegative:
            return (x - 3).is_nonnegative

    # 判断对象是否为实数
    def _eval_is_real(self):
        # 获取对象的第一个参数
        x = self.args[0]
        # 如果 x 非负或者 x 非整数，返回 True
        if x.is_nonnegative or x.is_noninteger:
            return True

    # 返回对象的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 获取对象的第一个参数，并以 x 为基础求其主导项
        arg = self.args[0].as_leading_term(x)
        # 计算 arg 在 x = 0 处的值
        arg0 = arg.subs(x, 0)
        # 如果 arg0 是零，返回 1
        if arg0.is_zero:
            return S.One
        # 如果 arg0 不是无穷大
        elif not arg0.is_infinite:
            # 返回以 arg 为参数的函数结果
            return self.func(arg)
        # 否则，抛出极点错误
        raise PoleError("Cannot expand %s around 0" % (self))
class MultiFactorial(CombinatorialFunction):
    pass



# 多重阶乘类，继承自组合函数类
class MultiFactorial(CombinatorialFunction):
    pass



class subfactorial(CombinatorialFunction):
    r"""The subfactorial counts the derangements of $n$ items and is
    defined for non-negative integers as:

    .. math:: !n = \begin{cases} 1 & n = 0 \\ 0 & n = 1 \\
                    (n-1)(!(n-1) + !(n-2)) & n > 1 \end{cases}

    It can also be written as ``int(round(n!/exp(1)))`` but the
    recursive definition with caching is implemented for this function.

    An interesting analytic expression is the following [2]_

    .. math:: !x = \Gamma(x + 1, -1)/e

    which is valid for non-negative integers `x`. The above formula
    is not very useful in case of non-integers. `\Gamma(x + 1, -1)` is
    single-valued only for integral arguments `x`, elsewhere on the positive
    real axis it has an infinite number of branches none of which are real.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Subfactorial
    .. [2] https://mathworld.wolfram.com/Subfactorial.html

    Examples
    ========

    >>> from sympy import subfactorial
    >>> from sympy.abc import n
    >>> subfactorial(n + 1)
    subfactorial(n + 1)
    >>> subfactorial(5)
    44

    See Also
    ========

    factorial, uppergamma,
    sympy.utilities.iterables.generate_derangements
    """



    @classmethod
    @cacheit
    def _eval(self, n):
        # 如果 n 是 0，则返回 1
        if not n:
            return S.One
        # 如果 n 是 1，则返回 0
        elif n == 1:
            return S.Zero
        # 对于 n 大于 1 的情况，使用递推关系计算阶乘
        else:
            z1, z2 = 1, 0
            for i in range(2, n + 1):
                z1, z2 = z2, (i - 1)*(z2 + z1)
            return z2



    @classmethod
    def eval(cls, arg):
        # 如果参数是一个数值类型
        if arg.is_Number:
            # 如果参数是整数且非负，调用 _eval 方法计算并返回结果
            if arg.is_Integer and arg.is_nonnegative:
                return cls._eval(arg)
            # 如果参数是 NaN，则返回 NaN
            elif arg is S.NaN:
                return S.NaN
            # 如果参数是 Infinity，则返回 Infinity
            elif arg is S.Infinity:
                return S.Infinity



    def _eval_is_even(self):
        # 如果参数是奇数且非负，返回 True
        if self.args[0].is_odd and self.args[0].is_nonnegative:
            return True



    def _eval_is_integer(self):
        # 如果参数是整数且非负，返回 True
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True



    def _eval_rewrite_as_factorial(self, arg, **kwargs):
        # 导入必要的函数和符号
        from sympy.concrete.summations import summation
        i = Dummy('i')
        f = S.NegativeOne**i / factorial(i)
        # 返回以阶乘和求和表示的表达式
        return factorial(arg) * summation(f, (i, 0, arg))



    def _eval_rewrite_as_gamma(self, arg, piecewise=True, **kwargs):
        # 导入必要的函数和符号
        from sympy.functions.elementary.exponential import exp
        from sympy.functions.special.gamma_functions import (gamma, lowergamma)
        # 返回以伽玛函数和下不完全伽玛函数表示的表达式
        return (S.NegativeOne**(arg + 1)*exp(-I*pi*arg)*lowergamma(arg + 1, -1)
                + gamma(arg + 1))*exp(-1)



    def _eval_rewrite_as_uppergamma(self, arg, **kwargs):
        # 导入必要的函数
        from sympy.functions.special.gamma_functions import uppergamma
        # 返回以上不完全伽玛函数表示的表达式
        return uppergamma(arg + 1, -1)/S.Exp1



# subfactorial 类的定义结束



# 代码块结束
    # 判断对象是否为非负数
    def _eval_is_nonnegative(self):
        # 如果第一个参数是整数并且是非负数，则返回True
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True
    
    # 判断对象是否为奇数
    def _eval_is_odd(self):
        # 如果第一个参数是偶数并且是非负数，则返回True
        if self.args[0].is_even and self.args[0].is_nonnegative:
            return True
class factorial2(CombinatorialFunction):
    r"""The double factorial `n!!`, not to be confused with `(n!)!`

    The double factorial is defined for nonnegative integers and for odd
    negative integers as:

    .. math:: n!! = \begin{cases} 1 & n = 0 \\
                    n(n-2)(n-4) \cdots 1 & n\ \text{positive odd} \\
                    n(n-2)(n-4) \cdots 2 & n\ \text{positive even} \\
                    (n+2)!!/(n+2) & n\ \text{negative odd} \end{cases}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Double_factorial

    Examples
    ========

    >>> from sympy import factorial2, var
    >>> n = var('n')
    >>> n
    n
    >>> factorial2(n + 1)
    factorial2(n + 1)
    >>> factorial2(5)
    15
    >>> factorial2(-1)
    1
    >>> factorial2(-5)
    1/3

    See Also
    ========

    factorial, RisingFactorial, FallingFactorial
    """

    @classmethod
    def eval(cls, arg):
        # TODO: extend this to complex numbers?
        
        # 检查参数是否为数值类型
        if arg.is_Number:
            # 如果参数不是整数，抛出错误
            if not arg.is_Integer:
                raise ValueError("argument must be nonnegative integer "
                                    "or negative odd integer")

            # 这个实现比递归实现更快
            # 同时避免了“最大递归深度超过”运行时错误
            # 如果参数是非负偶数，计算其双阶乘
            if arg.is_nonnegative:
                if arg.is_even:
                    k = arg / 2
                    return 2**k * factorial(k)
                # 如果参数是正奇数，计算其阶乘与前一个阶乘的乘积
                return factorial(arg) / factorial2(arg - 1)

            # 如果参数是负奇数，计算其双阶乘
            if arg.is_odd:
                return arg*(S.NegativeOne)**((1 - arg)/2) / factorial2(-arg)
            # 如果参数既不是非负整数也不是负奇数，抛出错误
            raise ValueError("argument must be nonnegative integer "
                                "or negative odd integer")


    def _eval_is_even(self):
        # 对于每个正偶数输入，双阶乘都是偶数
        n = self.args[0]
        if n.is_integer:
            if n.is_odd:
                return False
            if n.is_even:
                if n.is_positive:
                    return True
                if n.is_zero:
                    return False

    def _eval_is_integer(self):
        # 对于每个非负输入以及-1和-3，双阶乘都是整数
        n = self.args[0]
        if n.is_integer:
            if (n + 1).is_nonnegative:
                return True
            if n.is_odd:
                return (n + 3).is_nonnegative

    def _eval_is_odd(self):
        # 对于每个不小于-3的奇数输入以及0，双阶乘都是奇数
        n = self.args[0]
        if n.is_odd:
            return (n + 3).is_nonnegative
        if n.is_even:
            if n.is_positive:
                return False
            if n.is_zero:
                return True
    # 定义一个方法来评估双阶乘函数是否为正数
    def _eval_is_positive(self):
        # 双阶乘对于每个非负的输入都是正数，
        # 对于形如 -1-4k 的奇数负数输入也是正数，其中 k 是非负整数
        n = self.args[0]
        # 如果 n 是整数
        if n.is_integer:
            # 如果 n+1 是非负数
            if (n + 1).is_nonnegative:
                return True
            # 如果 n 是奇数
            if n.is_odd:
                # 返回 (n+1)/2 是否为偶数
                return ((n + 1) / 2).is_even

    # 将当前对象重写为 gamma 函数的形式
    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        # 导入所需的函数和类
        from sympy.functions.elementary.miscellaneous import sqrt
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.special.gamma_functions import gamma
        # 返回表达式 2^(n/2) * gamma(n/2 + 1) * Piecewise((1, Eq(Mod(n, 2), 0)),
        # (sqrt(2/pi), Eq(Mod(n, 2), 1)))
        return 2**(n/2) * gamma(n/2 + 1) * Piecewise((1, Eq(Mod(n, 2), 0)),
                (sqrt(2/pi), Eq(Mod(n, 2), 1)))
# 定义 RisingFactorial 类，继承自 CombinatorialFunction 类
class RisingFactorial(CombinatorialFunction):
    """
    Rising factorial (also called Pochhammer symbol [1]_) is a double valued
    function arising in concrete mathematics, hypergeometric functions
    and series expansions. It is defined by:
    
    .. math:: \texttt{rf(y, k)} = (x)^k = x \cdot (x+1) \cdots (x+k-1)
    
    where `x` can be arbitrary expression and `k` is an integer. For
    more information check "Concrete mathematics" by Graham, pp. 66
    or visit https://mathworld.wolfram.com/RisingFactorial.html page.
    
    When `x` is a `~.Poly` instance of degree $\ge 1$ with a single variable,
    `(x)^k = x(y) \cdot x(y+1) \cdots x(y+k-1)`, where `y` is the
    variable of `x`. This is as described in [2]_.
    
    Examples
    ========
    
    >>> from sympy import rf, Poly
    >>> from sympy.abc import x
    >>> rf(x, 0)
    1
    >>> rf(1, 5)
    120
    >>> rf(x, 5) == x*(1 + x)*(2 + x)*(3 + x)*(4 + x)
    True
    >>> rf(Poly(x**3, x), 2)
    Poly(x**6 + 3*x**5 + 3*x**4 + x**3, x, domain='ZZ')
    
    Rewriting is complicated unless the relationship between
    the arguments is known, but rising factorial can
    be rewritten in terms of gamma, factorial, binomial,
    and falling factorial.
    
    >>> from sympy import Symbol, factorial, ff, binomial, gamma
    >>> n = Symbol('n', integer=True, positive=True)
    >>> R = rf(n, n + 2)
    >>> for i in (rf, ff, factorial, binomial, gamma):
    ...  R.rewrite(i)
    ...
    RisingFactorial(n, n + 2)
    FallingFactorial(2*n + 1, n + 2)
    factorial(2*n + 1)/factorial(n - 1)
    binomial(2*n + 1, n + 2)*factorial(n + 2)
    gamma(2*n + 2)/gamma(n)
    
    See Also
    ========
    
    factorial, factorial2, FallingFactorial
    
    References
    ==========
    
    .. [1] https://en.wikipedia.org/wiki/Pochhammer_symbol
    .. [2] Peter Paule, "Greatest Factorial Factorization and Symbolic
           Summation", Journal of Symbolic Computation, vol. 20, pp. 235-268,
           1995.
    
    """
    
    # 类方法，用于处理 RisingFactorial 类的特定行为和操作
    @classmethod
    # 定义静态方法 `eval`，用于计算超几何函数的值
    def eval(cls, x, k):
        # 使用 sympify 将 x 和 k 转换为 SymPy 的表达式对象
        x = sympify(x)
        k = sympify(k)

        # 检查 x 或 k 是否为 NaN，若是则返回 NaN
        if x is S.NaN or k is S.NaN:
            return S.NaN
        # 若 k 为 1，则返回 k 的阶乘
        elif x is S.One:
            return factorial(k)
        # 若 k 为整数
        elif k.is_Integer:
            # 若 k 为 0，则返回 1
            if k.is_zero:
                return S.One
            else:
                # 若 k 为正整数
                if k.is_positive:
                    # 若 x 为正无穷，则返回正无穷
                    if x is S.Infinity:
                        return S.Infinity
                    # 若 x 为负无穷
                    elif x is S.NegativeInfinity:
                        # 若 k 为奇数，则返回负无穷；否则返回正无穷
                        if k.is_odd:
                            return S.NegativeInfinity
                        else:
                            return S.Infinity
                    else:
                        # 若 x 是多项式对象
                        if isinstance(x, Poly):
                            gens = x.gens
                            # 若多项式有多个生成元，则抛出异常
                            if len(gens) != 1:
                                raise ValueError("rf only defined for "
                                                 "polynomials on one generator")
                            else:
                                # 计算超几何函数的值
                                return reduce(lambda r, i:
                                              r * (x.shift(i)),
                                              range(int(k)), 1)
                        else:
                            # 计算超几何函数的值
                            return reduce(lambda r, i: r * (x + i),
                                          range(int(k)), 1)

                # 若 k 为负整数或零
                else:
                    # 若 x 为正无穷，则返回正无穷
                    if x is S.Infinity:
                        return S.Infinity
                    # 若 x 为负无穷，则返回正无穷
                    elif x is S.NegativeInfinity:
                        return S.Infinity
                    else:
                        # 若 x 是多项式对象
                        if isinstance(x, Poly):
                            gens = x.gens
                            # 若多项式有多个生成元，则抛出异常
                            if len(gens) != 1:
                                raise ValueError("rf only defined for "
                                                 "polynomials on one generator")
                            else:
                                # 计算超几何函数的值
                                return 1 / reduce(lambda r, i:
                                                  r * (x.shift(-i)),
                                                  range(1, abs(int(k)) + 1), 1)
                        else:
                            # 计算超几何函数的值
                            return 1 / reduce(lambda r, i:
                                              r * (x - i),
                                              range(1, abs(int(k)) + 1), 1)

        # 若 k 不是整数
        if k.is_integer == False:
            # 若 x 是整数且为负数，则返回 0
            if x.is_integer and x.is_negative:
                return S.Zero

    # 定义内部方法 `_eval_rewrite_as_gamma`，将超几何函数重写为 Gamma 函数形式
    def _eval_rewrite_as_gamma(self, x, k, piecewise=True, **kwargs):
        # 导入需要的库
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.special.gamma_functions import gamma
        
        # 若不使用分段函数表示
        if not piecewise:
            # 若 x 小于等于 0，则返回负一的 k 次方乘以 Gamma 函数的比值
            if (x <= 0) == True:
                return S.NegativeOne**k * gamma(1 - x) / gamma(-k - x + 1)
            # 否则返回 Gamma 函数的比值
            return gamma(x + k) / gamma(x)
        
        # 使用分段函数表示
        return Piecewise(
            (gamma(x + k) / gamma(x), x > 0),
            (S.NegativeOne**k * gamma(1 - x) / gamma(-k - x + 1), True))
    # 将当前对象表示的二项式表达式重写为下降阶乘的形式
    def _eval_rewrite_as_FallingFactorial(self, x, k, **kwargs):
        # 返回下降阶乘的结果
        return FallingFactorial(x + k - 1, k)

    # 将当前对象表示的二项式表达式重写为阶乘的形式
    def _eval_rewrite_as_factorial(self, x, k, **kwargs):
        # 导入分段函数模块
        from sympy.functions.elementary.piecewise import Piecewise
        # 如果 x 和 k 都是整数
        if x.is_integer and k.is_integer:
            # 返回一个分段函数，根据条件返回不同的表达式
            return Piecewise(
                (factorial(k + x - 1)/factorial(x - 1), x > 0),  # x > 0 时的表达式
                (S.NegativeOne**k*factorial(-x)/factorial(-k - x), True))  # 其他情况下的表达式

    # 将当前对象表示的二项式表达式重写为二项式系数的形式
    def _eval_rewrite_as_binomial(self, x, k, **kwargs):
        # 如果 k 是整数
        if k.is_integer:
            # 返回二项式系数的表达式
            return factorial(k) * binomial(x + k - 1, k)

    # 将当前对象表示的二项式表达式重写为可处理形式的表达式
    def _eval_rewrite_as_tractable(self, x, k, limitvar=None, **kwargs):
        # 导入 gamma 函数模块
        from sympy.functions.special.gamma_functions import gamma
        # 如果有指定的极限变量
        if limitvar:
            # 对 k 进行极限替换
            k_lim = k.subs(limitvar, S.Infinity)
            # 如果 k 的极限是正无穷
            if k_lim is S.Infinity:
                # 返回 gamma 函数的可处理形式重写结果
                return (gamma(x + k).rewrite('tractable', deep=True) / gamma(x))
            # 如果 k 的极限是负无穷
            elif k_lim is S.NegativeInfinity:
                # 返回 gamma 函数的可处理形式重写结果
                return (S.NegativeOne**k*gamma(1 - x) / gamma(-k - x + 1).rewrite('tractable', deep=True))
        # 否则，调用自身的 gamma 函数重写并返回可处理形式的结果
        return self.rewrite(gamma).rewrite('tractable', deep=True)

    # 判断当前对象表示的二项式表达式是否为整数
    def _eval_is_integer(self):
        # 返回判断结果，所有参数都是整数并且第二个参数是非负数
        return fuzzy_and((self.args[0].is_integer, self.args[1].is_integer,
                          self.args[1].is_nonnegative))
# 定义一个名为 FallingFactorial 的类，它继承自 CombinatorialFunction 类
class FallingFactorial(CombinatorialFunction):
    """
    Falling factorial (related to rising factorial) is a double valued
    function arising in concrete mathematics, hypergeometric functions
    and series expansions. It is defined by

    .. math:: \texttt{ff(x, k)} = (x)_k = x \cdot (x-1) \cdots (x-k+1)

    where `x` can be arbitrary expression and `k` is an integer. For
    more information check "Concrete mathematics" by Graham, pp. 66
    or [1]_.

    When `x` is a `~.Poly` instance of degree $\ge 1$ with single variable,
    `(x)_k = x(y) \cdot x(y-1) \cdots x(y-k+1)`, where `y` is the
    variable of `x`. This is as described in

    >>> from sympy import ff, Poly, Symbol
    >>> from sympy.abc import x
    >>> n = Symbol('n', integer=True)

    >>> ff(x, 0)
    1
    >>> ff(5, 5)
    120
    >>> ff(x, 5) == x*(x - 1)*(x - 2)*(x - 3)*(x - 4)
    True
    >>> ff(Poly(x**2, x), 2)
    Poly(x**4 - 2*x**3 + x**2, x, domain='ZZ')
    >>> ff(n, n)
    factorial(n)

    Rewriting is complicated unless the relationship between
    the arguments is known, but falling factorial can
    be rewritten in terms of gamma, factorial and binomial
    and rising factorial.

    >>> from sympy import factorial, rf, gamma, binomial, Symbol
    >>> n = Symbol('n', integer=True, positive=True)
    >>> F = ff(n, n - 2)
    >>> for i in (rf, ff, factorial, binomial, gamma):
    ...  F.rewrite(i)
    ...
    RisingFactorial(3, n - 2)
    FallingFactorial(n, n - 2)
    factorial(n)/2
    binomial(n, n - 2)*factorial(n - 2)
    gamma(n + 1)/2

    See Also
    ========

    factorial, factorial2, RisingFactorial

    References
    ==========

    .. [1] https://mathworld.wolfram.com/FallingFactorial.html
    .. [2] Peter Paule, "Greatest Factorial Factorization and Symbolic
           Summation", Journal of Symbolic Computation, vol. 20, pp. 235-268,
           1995.

    """

    @classmethod
    # 定义类方法 eval，用于计算特定的符号表达式
    def eval(cls, x, k):
        # 将输入的 x 和 k 转换为 SymPy 的符号表达式
        x = sympify(x)
        k = sympify(k)

        # 检查是否输入的 x 或 k 是 NaN（非数值），若是则返回 NaN
        if x is S.NaN or k is S.NaN:
            return S.NaN
        # 若 k 是整数且 x 等于 k，则返回 x 的阶乘
        elif k.is_integer and x == k:
            return factorial(x)
        # 若 k 是整数且非负，根据不同情况返回不同的计算结果
        elif k.is_Integer:
            if k.is_zero:
                return S.One
            else:
                if k.is_positive:
                    # 当 x 为正无穷时返回正无穷
                    if x is S.Infinity:
                        return S.Infinity
                    # 当 x 为负无穷时根据 k 的奇偶性返回对应的结果
                    elif x is S.NegativeInfinity:
                        if k.is_odd:
                            return S.NegativeInfinity
                        else:
                            return S.Infinity
                    else:
                        # 若 x 是多项式对象，则根据定义进行计算
                        if isinstance(x, Poly):
                            gens = x.gens
                            if len(gens) != 1:
                                raise ValueError("ff only defined for "
                                                 "polynomials on one generator")
                            else:
                                return reduce(lambda r, i:
                                              r * (x.shift(-i)),
                                              range(int(k)), 1)
                        else:
                            # 对一般情况下的 x 进行计算
                            return reduce(lambda r, i: r * (x - i),
                                          range(int(k)), 1)
                else:
                    # 当 k 为负数时，根据不同情况返回不同的计算结果
                    if x is S.Infinity:
                        return S.Infinity
                    elif x is S.NegativeInfinity:
                        return S.Infinity
                    else:
                        # 若 x 是多项式对象，则根据定义进行计算
                        if isinstance(x, Poly):
                            gens = x.gens
                            if len(gens) != 1:
                                raise ValueError("rf only defined for "
                                                 "polynomials on one generator")
                            else:
                                return 1 / reduce(lambda r, i:
                                                  r * (x.shift(i)),
                                                  range(1, abs(int(k)) + 1), 1)
                        else:
                            # 对一般情况下的 x 进行计算
                            return 1 / reduce(lambda r, i: r * (x + i),
                                              range(1, abs(int(k)) + 1), 1)

    # 定义方法 _eval_rewrite_as_gamma，将函数重写为 gamma 函数的表达式
    def _eval_rewrite_as_gamma(self, x, k, piecewise=True, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.special.gamma_functions import gamma
        # 若 piecewise 为 False，根据条件返回 gamma 函数的计算结果
        if not piecewise:
            if (x < 0) == True:
                return S.NegativeOne**k * gamma(k - x) / gamma(-x)
            return gamma(x + 1) / gamma(x - k + 1)
        # 若 piecewise 为 True，使用 Piecewise 函数进行条件判断并返回结果
        return Piecewise(
            (gamma(x + 1) / gamma(x - k + 1), x >= 0),
            (S.NegativeOne**k * gamma(k - x) / gamma(-x), True))

    # 定义方法 _eval_rewrite_as_RisingFactorial，将函数重写为 RisingFactorial 函数的表达式
    def _eval_rewrite_as_RisingFactorial(self, x, k, **kwargs):
        # 调用 rf 函数进行计算
        return rf(x - k + 1, k)

    # 定义方法 _eval_rewrite_as_binomial，将函数重写为二项式系数的表达式
    def _eval_rewrite_as_binomial(self, x, k, **kwargs):
        # 若 k 是整数，返回 x 的阶乘乘以二项式系数的结果
        if k.is_integer:
            return factorial(k) * binomial(x, k)
    # 将表达式重写为阶乘形式
    def _eval_rewrite_as_factorial(self, x, k, **kwargs):
        # 导入需要的函数库
        from sympy.functions.elementary.piecewise import Piecewise
        # 如果 x 和 k 都是整数
        if x.is_integer and k.is_integer:
            # 返回一个分段函数，根据条件 x >= 0 选择合适的阶乘比值
            return Piecewise(
                (factorial(x)/factorial(-k + x), x >= 0),
                (S.NegativeOne**k*factorial(k - x - 1)/factorial(-x - 1), True))

    # 将表达式重写为可处理形式
    def _eval_rewrite_as_tractable(self, x, k, limitvar=None, **kwargs):
        # 导入需要的函数库
        from sympy.functions.special.gamma_functions import gamma
        # 如果有限制变量
        if limitvar:
            # 计算在限制变量处 k 的极限
            k_lim = k.subs(limitvar, S.Infinity)
            # 如果极限是正无穷
            if k_lim is S.Infinity:
                # 返回 gamma 函数重写为可处理形式的表达式
                return (S.NegativeOne**k*gamma(k - x).rewrite('tractable', deep=True) / gamma(-x))
            # 如果极限是负无穷
            elif k_lim is S.NegativeInfinity:
                # 返回 gamma 函数的重写形式，处理深度为 True
                return (gamma(x + 1) / gamma(x - k + 1).rewrite('tractable', deep=True))
        # 否则，调用对象本身的重写方法，将 gamma 函数重写为可处理形式
        return self.rewrite(gamma).rewrite('tractable', deep=True)

    # 判断对象是否为整数
    def _eval_is_integer(self):
        # 返回参数中所有参数都是整数且第二个参数是非负数的模糊逻辑与结果
        return fuzzy_and((self.args[0].is_integer, self.args[1].is_integer,
                          self.args[1].is_nonnegative))
# 导入 RisingFactorial 和 FallingFactorial 函数
rf = RisingFactorial
ff = FallingFactorial

# 定义 binomial 类，继承自 CombinatorialFunction 类
class binomial(CombinatorialFunction):
    # binomial 系数的实现，支持两种定义方式
    r"""Implementation of the binomial coefficient. It can be defined
    in two ways depending on its desired interpretation:

    .. math:: \binom{n}{k} = \frac{n!}{k!(n-k)!}\ \text{or}\
                \binom{n}{k} = \frac{(n)_k}{k!}

    First, in a strict combinatorial sense it defines the
    number of ways we can choose `k` elements from a set of
    `n` elements. In this case both arguments are nonnegative
    integers and binomial is computed using an efficient
    algorithm based on prime factorization.

    The other definition is generalization for arbitrary `n`,
    however `k` must also be nonnegative. This case is very
    useful when evaluating summations.

    For the sake of convenience, for negative integer `k` this function
    will return zero no matter the other argument.

    To expand the binomial when `n` is a symbol, use either
    ``expand_func()`` or ``expand(func=True)``. The former will keep
    the polynomial in factored form while the latter will expand the
    polynomial itself. See examples for details.

    Examples
    ========

    >>> from sympy import Symbol, Rational, binomial, expand_func
    >>> n = Symbol('n', integer=True, positive=True)

    >>> binomial(15, 8)
    6435

    >>> binomial(n, -1)
    0

    Rows of Pascal's triangle can be generated with the binomial function:

    >>> for N in range(8):
    ...     print([binomial(N, i) for i in range(N + 1)])
    ...
    [1]
    [1, 1]
    [1, 2, 1]
    [1, 3, 3, 1]
    [1, 4, 6, 4, 1]
    [1, 5, 10, 10, 5, 1]
    [1, 6, 15, 20, 15, 6, 1]
    [1, 7, 21, 35, 35, 21, 7, 1]

    As can a given diagonal, e.g. the 4th diagonal:

    >>> N = -4
    >>> [binomial(N, i) for i in range(1 - N)]
    [1, -4, 10, -20, 35]

    >>> binomial(Rational(5, 4), 3)
    -5/128
    >>> binomial(Rational(-5, 4), 3)
    -195/128

    >>> binomial(n, 3)
    binomial(n, 3)

    >>> binomial(n, 3).expand(func=True)
    n**3/6 - n**2/2 + n/3

    >>> expand_func(binomial(n, 3))
    n*(n - 2)*(n - 1)/6

    In many cases, we can also compute binomial coefficients modulo a
    prime p quickly using Lucas' Theorem [2]_, though we need to include
    `evaluate=False` to postpone evaluation:

    >>> from sympy import Mod
    >>> Mod(binomial(156675, 4433, evaluate=False), 10**5 + 3)
    28625

    Using a generalisation of Lucas's Theorem given by Granville [3]_,
    we can extend this to arbitrary n:

    >>> Mod(binomial(10**18, 10**12, evaluate=False), (10**5 + 3)**2)
    3744312326

    References
    ==========

    .. [1] https://www.johndcook.com/blog/binomial_coefficients/
    .. [2] https://en.wikipedia.org/wiki/Lucas%27s_theorem
    """
    pass  # binomial 类的定义结束
    """
    Implement binomial coefficient calculation and differentiation methods.

    References:
    [1] Binomial coefficients modulo prime powers, Andrew Granville,
        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf
    """

    # 实现对自变量的求导方法
    def fdiff(self, argindex=1):
        # 导入 polygamma 函数用于求多重对数函数的导数
        from sympy.functions.special.gamma_functions import polygamma
        if argindex == 1:
            # 当自变量索引为1时，根据 Gamma 函数多重对数的差分计算二项式系数
            n, k = self.args
            return binomial(n, k)*(polygamma(0, n + 1) - \
                polygamma(0, n - k + 1))
        elif argindex == 2:
            # 当自变量索引为2时，根据 Gamma 函数多重对数的差分计算二项式系数
            n, k = self.args
            return binomial(n, k)*(polygamma(0, n - k + 1) - \
                polygamma(0, k + 1))
        else:
            # 如果自变量索引不是1或2，抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def _eval(self, n, k):
        # 检查参数是否为整数并且满足条件，然后计算二项式系数

        if k.is_Integer:
            if n.is_Integer and n >= 0:
                n, k = int(n), int(k)

                if k > n:
                    return S.Zero
                elif k > n // 2:
                    k = n - k

                # XXX: 这个条件逻辑应该移至 sympy.external.gmpy，并且
                # 纯 Python 版本的 bincoef 应该移至 sympy.external.ntheory。
                if _gmpy is not None:
                    return Integer(_gmpy.bincoef(n, k))

                d, result = n - k, 1
                for i in range(1, k + 1):
                    d += 1
                    result = result * d // i
                return Integer(result)
            else:
                d, result = n - k, 1
                for i in range(1, k + 1):
                    d += 1
                    result *= d
                return result / _factorial(k)

    @classmethod
    def eval(cls, n, k):
        # 对输入参数 n 和 k 进行符号化处理
        n, k = map(sympify, (n, k))
        d = n - k
        n_nonneg, n_isint = n.is_nonnegative, n.is_integer
        if k.is_zero or ((n_nonneg or n_isint is False)
                and d.is_zero):
            return S.One
        if (k - 1).is_zero or ((n_nonneg or n_isint is False)
                and (d - 1).is_zero):
            return n
        if k.is_integer:
            if k.is_negative or (n_nonneg and n_isint and d.is_negative):
                return S.Zero
            elif n.is_number:
                res = cls._eval(n, k)
                return res.expand(basic=True) if res else res
        elif n_nonneg is False and n_isint:
            # 特殊情况，当二项式系数评估为复数无穷大时
            return S.ComplexInfinity
        elif k.is_number:
            from sympy.functions.special.gamma_functions import gamma
            return gamma(n + 1)/(gamma(k + 1)*gamma(n - k + 1))
    # 定义一个方法 _eval_Mod，用于计算二项式系数模 q 的结果
    def _eval_Mod(self, q):
        # 将参数解构为 n 和 k
        n, k = self.args

        # 检查 n、k、q 中是否有非整数，若有则抛出异常
        if any(x.is_integer is False for x in (n, k, q)):
            raise ValueError("Integers expected for binomial Mod")

        # 如果 n、k、q 都是整数
        if all(x.is_Integer for x in (n, k, q)):
            # 将 n 和 k 转换为整数类型
            n, k = map(int, (n, k))
            aq, res = abs(q), 1

            # 处理负数情况下的 k 或 n
            if k < 0:
                return S.Zero
            if n < 0:
                n = -n + k - 1
                res = -1 if k%2 else 1

            # 处理非负整数的 k 和 n
            if k > n:
                return S.Zero

            # 判断 q 是否为质数
            isprime = aq.is_prime
            aq = int(aq)
            if isprime:
                if aq < n:
                    # 使用 Lucas 定理进行计算
                    N, K = n, k
                    while N or K:
                        res = res*binomial(N % aq, K % aq) % aq
                        N, K = N // aq, K // aq

                else:
                    # 使用阶乘模进行计算
                    d = n - k
                    if k > d:
                        k, d = d, k
                    kf = 1
                    for i in range(2, k + 1):
                        kf = kf*i % aq
                    df = kf
                    for i in range(k + 1, d + 1):
                        df = df*i % aq
                    res *= df
                    for i in range(d + 1, n + 1):
                        res = res*i % aq

                    res *= pow(kf*df % aq, aq - 2, aq)
                    res %= aq

            elif _sqrt(q) < k and q != 1:
                # 当 sqrt(q) < k 且 q 不等于 1 时，使用二项式模计算函数 binomial_mod
                res = binomial_mod(n, k, q)

            else:
                # 对于其它情况，使用二项式分解计算方法
                # 在计算 n!/(k!(n-k)!) 中 <= n 的质数的指数时
                M = int(_sqrt(n))
                for prime in sieve.primerange(2, n + 1):
                    if prime > n - k:
                        res = res*prime % aq
                    elif prime > n // 2:
                        continue
                    elif prime > M:
                        if n % prime < k % prime:
                            res = res*prime % aq
                    else:
                        N, K = n, k
                        exp = a = 0

                        while N > 0:
                            a = int((N % prime) < (K % prime + a))
                            N, K = N // prime, K // prime
                            exp += a

                        if exp > 0:
                            res *= pow(prime, exp, aq)
                            res %= aq

            # 返回结果 S(res % q)，其中 res 为最终的计算结果
            return S(res % q)
    # 定义一个函数，用于处理二项式展开，前提是 m 是正整数
    def _eval_expand_func(self, **hints):
        """
        Function to expand binomial(n, k) when m is positive integer
        Also,
        n is self.args[0] and k is self.args[1] while using binomial(n, k)
        """
        # 提取 self.args 中的第一个参数作为 n
        n = self.args[0]
        # 如果 n 是一个数值
        if n.is_Number:
            # 调用 sympy 中的 binomial 函数，计算二项式系数
            return binomial(*self.args)

        # 提取 self.args 中的第二个参数作为 k
        k = self.args[1]
        # 如果 n - k 是一个整数
        if (n - k).is_Integer:
            # 更新 k 为 n - k，以简化计算
            k = n - k

        # 如果 k 是一个整数
        if k.is_Integer:
            # 如果 k 等于零，返回 1
            if k.is_zero:
                return S.One
            # 如果 k 是负数，返回 0
            elif k.is_negative:
                return S.Zero
            else:
                # 初始化 n 和结果为 1
                n, result = self.args[0], 1
                # 计算阶乘的除法，使用循环计算二项式展开
                for i in range(1, k + 1):
                    result *= n - k + i
                return result / _factorial(k)
        else:
            # 如果 k 不是整数，调用 sympy 中的 binomial 函数计算二项式系数
            return binomial(*self.args)

    # 将二项式展开重写为阶乘的比值
    def _eval_rewrite_as_factorial(self, n, k, **kwargs):
        return factorial(n) / (factorial(k) * factorial(n - k))

    # 将二项式展开重写为 gamma 函数的比值
    def _eval_rewrite_as_gamma(self, n, k, piecewise=True, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        return gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))

    # 将二项式展开重写为 tractable 形式
    def _eval_rewrite_as_tractable(self, n, k, limitvar=None, **kwargs):
        # 调用 _eval_rewrite_as_gamma 方法，然后再重写为 'tractable' 形式
        return self._eval_rewrite_as_gamma(n, k).rewrite('tractable')

    # 将二项式展开重写为 FallingFactorial 形式的比值
    def _eval_rewrite_as_FallingFactorial(self, n, k, **kwargs):
        # 如果 k 是整数，返回 FallingFactorial(n, k) / k 的阶乘
        if k.is_integer:
            return ff(n, k) / factorial(k)

    # 判断二项式展开是否得到整数结果
    def _eval_is_integer(self):
        n, k = self.args
        # 如果 n 和 k 都是整数，返回 True
        if n.is_integer and k.is_integer:
            return True
        # 如果 k 不是整数，返回 False
        elif k.is_integer is False:
            return False

    # 判断二项式展开结果是否为非负数
    def _eval_is_nonnegative(self):
        n, k = self.args
        # 如果 n 和 k 都是整数
        if n.is_integer and k.is_integer:
            # 如果 n 是非负数或者 k 是负数或者 k 是偶数，返回 True
            if n.is_nonnegative or k.is_negative or k.is_even:
                return True
            # 如果 k 是奇数，返回 False
            elif k.is_even is False:
                return False

    # 计算二项式展开的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.functions.special.gamma_functions import gamma
        # 调用 rewrite 方法，然后再调用 _eval_as_leading_term 方法计算主导项
        return self.rewrite(gamma)._eval_as_leading_term(x, logx=logx, cdir=cdir)
```