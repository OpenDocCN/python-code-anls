# `D:\src\scipysrc\sympy\sympy\integrals\trigonometry.py`

```
# 从 sympy 库中导入所需的模块和函数
from sympy.core import cacheit, Dummy, Ne, Integer, Rational, S, Wild
from sympy.functions import binomial, sin, cos, Piecewise, Abs
from .integrals import integrate
# 导入本地的 integrate 函数，用于积分计算

# TODO sin(a*x)*cos(b*x) -> sin((a+b)x) + sin((a-b)x) ?

# 每次创建 Wild's 和 sin/cos/Mul 都很昂贵。此外，如果不缓存，我们的匹配和替换非常慢，
# 如果每次创建 Wild，实际上会阻止缓存。
#
# 因此，我们缓存模式

# 需要使用函数而不是 lambda，因为 lambda 的哈希在每次调用 _pat_sincos 时都会改变
def _integer_instance(n):
    return isinstance(n, Integer)

@cacheit
def _pat_sincos(x):
    # 创建 Wild 对象，排除 x 在外，并且 n 和 m 需要是整数
    a = Wild('a', exclude=[x])
    n, m = [Wild(s, exclude=[x], properties=[_integer_instance])
                for s in 'nm']
    # 定义匹配模式，匹配 sin(a*x)**n * cos(a*x)**m 的形式
    pat = sin(a*x)**n * cos(a*x)**m
    return pat, a, n, m

# 创建虚拟符号 u，用于代表未知的函数
_u = Dummy('u')


def trigintegrate(f, x, conds='piecewise'):
    """
    对 f = Mul(trig) 在变量 x 上进行积分。

    Examples
    ========

    >>> from sympy import sin, cos, tan, sec
    >>> from sympy.integrals.trigonometry import trigintegrate
    >>> from sympy.abc import x

    >>> trigintegrate(sin(x)*cos(x), x)
    sin(x)**2/2

    >>> trigintegrate(sin(x)**2, x)
    x/2 - sin(x)*cos(x)/2

    >>> trigintegrate(tan(x)*sec(x), x)
    1/cos(x)

    >>> trigintegrate(sin(x)*tan(x), x)
    -log(sin(x) - 1)/2 + log(sin(x) + 1)/2 - sin(x)

    References
    ==========

    .. [1] https://en.wikibooks.org/wiki/Calculus/Integration_techniques

    See Also
    ========

    sympy.integrals.integrals.Integral.doit
    sympy.integrals.integrals.Integral
    """
    # 获得模式 pat 和相关的 Wild 对象 a, n, m
    pat, a, n, m = _pat_sincos(x)

    # 重新将 f 重写为 'sincos' 形式的表达式
    f = f.rewrite('sincos')
    # 尝试将 f 匹配到模式 pat 上
    M = f.match(pat)

    if M is None:
        return

    # 从匹配 M 中提取出 n 和 m 的值
    n, m = M[n], M[m]
    # 如果 n 和 m 都是零，返回 x
    if n.is_zero and m.is_zero:
        return x
    # 如果 n 是零，则 zz 为 x，否则为 S.Zero
    zz = x if n.is_zero else S.Zero

    # 从匹配 M 中获取 Wild 对象 a 的值
    a = M[a]
    # 如果 n 或者 m 是奇数
    if n.is_odd or m.is_odd:
        u = _u
        n_, m_ = n.is_odd, m.is_odd

        # 选择最小的 n 或者 m，以便选择最简单的替代
        if n_ and m_:

            # 确保选择正数，否则可能导致错误的积分
            if n < 0 and m > 0:
                m_ = True
                n_ = False
            elif m < 0 and n > 0:
                n_ = True
                m_ = False
            # 如果 n 和 m 都是负数，选择绝对值最小的 n 或者 m 进行最简单的替代
            elif (n < 0 and m < 0):
                n_ = n > m
                m_ = not (n > m)

            # 如果 n 和 m 都是奇数且为正数
            else:
                n_ = (n < m)      # 注意：这里要小心，其中一个条件 *必须* 为真
                m_ = not (n < m)

        #  n      m       u=C        (n-1)/2    m
        # S(x) * C(x) dx  --> -(1-u^2)       * u  du
        if n_:
            ff = -(1 - u**2)**((n - 1)/2) * u**m
            uu = cos(a*x)

        #  n      m       u=S   n         (m-1)/2
        # S(x) * C(x) dx  -->  u  * (1-u^2)       du
        elif m_:
            ff = u**n * (1 - u**2)**((m - 1)/2)
            uu = sin(a*x)

        fi = integrate(ff, u)  # XXX cyclic deps
        fx = fi.subs(u, uu)
        if conds == 'piecewise':
            return Piecewise((fx / a, Ne(a, 0)), (zz, True))
        return fx / a

    # n 和 m 都是偶数
    #
    #               2k      2m                         2l       2l
    # 将 S (x) * C (x) 转换为只含有 S (x) 或者 C (x) 的项
    #
    # 例如:
    #  100     4       100        2    2    100          4         2
    # S (x) * C (x) = S (x) * (1-S (x))  = S (x) * (1 + S (x) - 2*S (x))
    #
    #                  104       102     100
    #               = S (x) - 2*S (x) + S (x)
    #       2k
    # 然后使用递归公式对 S 进行积分

    # 选择绝对值较大的 n 或者 m，以便选择最简单的替代
    n_ = (Abs(n) > Abs(m))
    m_ = (Abs(m) > Abs(n))
    res = S.Zero
    # 如果 n_ 不为空（即 n_ 非零），执行以下代码块
    if n_:
        # 当 m 大于 0 时，执行以下循环
        for i in range(0, m//2 + 1):
            # 计算 res 的值，根据公式 res += (-1)^i * binomial(m//2, i) * _sin_pow_integrate(n + 2*i, x)
            res += (S.NegativeOne**i * binomial(m//2, i) *
                    _sin_pow_integrate(n + 2*i, x))

        # 当 m 等于 0 时，将 res 设为 _sin_pow_integrate(n, x) 的返回值
        elif m == 0:
            res = _sin_pow_integrate(n, x)
        else:
            # 当 m 小于 0 且 |n| > |m| 时，根据以下复合积分公式计算 res 的值
            res = (Rational(-1, m + 1) * cos(x)**(m + 1) * sin(x)**(n - 1) +
                   Rational(n - 1, m + 1) *
                   trigintegrate(cos(x)**(m + 2)*sin(x)**(n - 2), x))
    elif m_:
        # 如果 m_ 为真，执行以下代码块

        # 计算 S 的表达式
        # S   = (1 - C ) = sum(i, (-1)^i * B(k, i) * C^(2i))
        if n > 0:
            # 如果 n 大于 0，执行以下代码块

            # 对于 i 从 0 到 n//2，计算累加结果
            for i in range(0, n//2 + 1):
                # res 累加 (S.NegativeOne^i * binomial(n//2, i) * _cos_pow_integrate(m + 2*i, x))
                res += (S.NegativeOne**i * binomial(n//2, i) *
                        _cos_pow_integrate(m + 2*i, x))

        elif n == 0:
            # 如果 n 等于 0，执行以下代码块

            # res 赋值为 _cos_pow_integrate(m, x) 的结果
            res = _cos_pow_integrate(m, x)
        else:
            # 如果 n 小于 0，执行以下代码块

            # res 赋值为 (Rational(1, n + 1) * cos(x)**(m - 1)*sin(x)**(n + 1) +
            #            Rational(m - 1, n + 1) * trigintegrate(cos(x)**(m - 2)*sin(x)**(n + 2), x))
            res = (Rational(1, n + 1) * cos(x)**(m - 1)*sin(x)**(n + 1) +
                   Rational(m - 1, n + 1) *
                   trigintegrate(cos(x)**(m - 2)*sin(x)**(n + 2), x))

    else:
        # 如果 m_ 不为真，执行以下代码块

        if m == n:
            # 如果 m 等于 n，执行以下代码块

            # 使用 sin(2x)/2 替换 sin(x)*cos(x)，然后进行积分
            res = integrate((sin(2*x)*S.Half)**m, x)
        elif (m == -n):
            # 如果 m 等于 -n，执行以下代码块

            if n < 0:
                # 如果 n 小于 0，执行以下代码块

                # res 赋值为 (Rational(1, n + 1) * cos(x)**(m - 1) * sin(x)**(n + 1) +
                #            Rational(m - 1, n + 1) * integrate(cos(x)**(m - 2) * sin(x)**(n + 2), x))
                res = (Rational(1, n + 1) * cos(x)**(m - 1) * sin(x)**(n + 1) +
                       Rational(m - 1, n + 1) *
                       integrate(cos(x)**(m - 2) * sin(x)**(n + 2), x))
            else:
                # 如果 n 大于等于 0，执行以下代码块

                # res 赋值为 (Rational(-1, m + 1) * cos(x)**(m + 1) * sin(x)**(n - 1) +
                #            Rational(n - 1, m + 1) * integrate(cos(x)**(m + 2)*sin(x)**(n - 2), x))
                res = (Rational(-1, m + 1) * cos(x)**(m + 1) * sin(x)**(n - 1) +
                       Rational(n - 1, m + 1) *
                       integrate(cos(x)**(m + 2)*sin(x)**(n - 2), x))

    if conds == 'piecewise':
        # 如果 conds 等于 'piecewise'，执行以下代码块

        # 返回 Piecewise((res.subs(x, a*x) / a, Ne(a, 0)), (zz, True))
        return Piecewise((res.subs(x, a*x) / a, Ne(a, 0)), (zz, True))
    
    # 返回 res.subs(x, a*x) / a
    return res.subs(x, a*x) / a
# 定义一个函数，用于计算正弦的幂次乘积积分
def _sin_pow_integrate(n, x):
    # 如果 n 大于 0
    if n > 0:
        # 如果 n 等于 1，递归终止条件，返回-cos(x)
        if n == 1:
            return -cos(x)

        # 如果 n > 0
        # 使用递推公式计算正弦的幂次乘积积分
        #   /                                                 /
        #  |                                                 |
        #  |    n           -1               n-1     n - 1   |     n-2
        #  | sin (x) dx =  ______ cos (x) sin (x) + _______  |  sin (x) dx
        #  |                                                 |
        #  |                 n                         n     |
        # /                                                 /
        return (Rational(-1, n) * cos(x) * sin(x)**(n - 1) +
                Rational(n - 1, n) * _sin_pow_integrate(n - 2, x))

    # 如果 n 小于 0
    if n < 0:
        # 如果 n 等于 -1，递归终止条件，使用三角函数积分处理 1/sin(x)
        if n == -1:
            return trigintegrate(1/sin(x), x)

        # 如果 n < 0
        # 使用递推公式计算正弦的幂次乘积积分
        #   /                                                 /
        #  |                                                 |
        #  |    n            1               n+1     n + 2   |     n+2
        #  | sin (x) dx = _______ cos (x) sin (x) + _______  |  sin (x) dx
        #  |                                                 |
        #  |               n + 1                     n + 1   |
        # /                                                 /
        return (Rational(1, n + 1) * cos(x) * sin(x)**(n + 1) +
                Rational(n + 2, n + 1) * _sin_pow_integrate(n + 2, x))

    else:
        # 如果 n 等于 0，递归终止条件，返回 x
        return x


# 定义一个函数，用于计算余弦的幂次乘积积分
def _cos_pow_integrate(n, x):
    # 如果 n 大于 0
    if n > 0:
        # 如果 n 等于 1，递归终止条件，返回 sin(x)
        if n == 1:
            return sin(x)

        # 如果 n > 0
        # 使用递推公式计算余弦的幂次乘积积分
        #   /                                                 /
        #  |                                                 |
        #  |    n            1               n-1     n - 1   |     n-2
        #  | sin (x) dx =  ______ sin (x) cos (x) + _______  |  cos (x) dx
        #  |                                                 |
        #  |                 n                         n     |
        # /                                                 /
        return (Rational(1, n) * sin(x) * cos(x)**(n - 1) +
                Rational(n - 1, n) * _cos_pow_integrate(n - 2, x))
    # 如果 n 小于 0，则处理负指数情况
    if n < 0:
        # 如果 n 等于 -1，说明是递归中断条件
        if n == -1:
            # 递归中断，返回对 1/cos(x) 的积分结果
            return trigintegrate(1/cos(x), x)

        # 当 n 小于 0 且不等于 -1 时，使用递推公式计算余弦函数的负整数幂积分结果
        #   /                                                 /
        #  |                                                 |
        #  |    n            -1              n+1     n + 2   |     n+2
        #  | cos (x) dx = _______ sin (x) cos (x) + _______  |  cos (x) dx
        #  |                                                 |
        #  |               n + 1                     n + 1   |
        # /                                                 /
        #
        return (Rational(-1, n + 1) * sin(x) * cos(x)**(n + 1) +
                Rational(n + 2, n + 1) * _cos_pow_integrate(n + 2, x))
    else:
        # 当 n 等于 0 时，递归中断条件
        # Recursion Break.
        return x
```