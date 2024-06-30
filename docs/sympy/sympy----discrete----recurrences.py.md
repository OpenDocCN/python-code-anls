# `D:\src\scipysrc\sympy\sympy\discrete\recurrences.py`

```
"""
Recurrences
"""

# 导入 SymPy 库中的相关模块和函数
from sympy.core import S, sympify
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int


# 定义函数 linrec，用于求解线性递推关系的初始值问题
def linrec(coeffs, init, n):
    r"""
    Evaluation of univariate linear recurrences of homogeneous type
    having coefficients independent of the recurrence variable.

    Parameters
    ==========

    coeffs : iterable
        Coefficients of the recurrence
    init : iterable
        Initial values of the recurrence
    n : Integer
        Point of evaluation for the recurrence

    Notes
    =====

    Let `y(n)` be the recurrence of given type, ``c`` be the sequence
    of coefficients, ``b`` be the sequence of initial/base values of the
    recurrence and ``k`` (equal to ``len(c)``) be the order of recurrence.
    Then,

    .. math :: y(n) = \begin{cases} b_n & 0 \le n < k \\
        c_0 y(n-1) + c_1 y(n-2) + \cdots + c_{k-1} y(n-k) & n \ge k
        \end{cases}

    Let `x_0, x_1, \ldots, x_n` be a sequence and consider the transformation
    that maps each polynomial `f(x)` to `T(f(x))` where each power `x^i` is
    replaced by the corresponding value `x_i`. The sequence is then a solution
    of the recurrence if and only if `T(x^i p(x)) = 0` for each `i \ge 0` where
    `p(x) = x^k - c_0 x^(k-1) - \cdots - c_{k-1}` is the characteristic
    polynomial.

    Then `T(f(x)p(x)) = 0` for each polynomial `f(x)` (as it is a linear
    combination of powers `x^i`). Now, if `x^n` is congruent to
    `g(x) = a_0 x^0 + a_1 x^1 + \cdots + a_{k-1} x^{k-1}` modulo `p(x)`, then
    `T(x^n) = x_n` is equal to
    `T(g(x)) = a_0 x_0 + a_1 x_1 + \cdots + a_{k-1} x_{k-1}`.

    Computation of `x^n`,
    given `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \cdots + c_{k-1}`
    is performed using exponentiation by squaring (refer to [1_]) with
    an additional reduction step performed to retain only first `k` powers
    of `x` in the representation of `x^n`.

    Examples
    ========

    >>> from sympy.discrete.recurrences import linrec
    >>> from sympy.abc import x, y, z

    >>> linrec(coeffs=[1, 1], init=[0, 1], n=10)
    55

    >>> linrec(coeffs=[1, 1], init=[x, y], n=10)
    34*x + 55*y

    >>> linrec(coeffs=[x, y], init=[0, 1], n=5)
    x**2*y + x*(x**3 + 2*x*y) + y**2

    >>> linrec(coeffs=[1, 2, 3, 0, 0, 4], init=[x, y, z], n=16)
    13576*x + 5676*y + 2356*z

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponentiation_by_squaring
    .. [2] https://en.wikipedia.org/w/index.php?title=Modular_exponentiation&section=6#Matrices

    See Also
    ========

    sympy.polys.agca.extensions.ExtensionElement.__pow__

    """

    # 若没有给定系数，则返回零
    if not coeffs:
        return S.Zero

    # 如果给定的系数不是可迭代对象，抛出类型错误异常
    if not iterable(coeffs):
        raise TypeError("Expected a sequence of coefficients for"
                        " the recurrence")

    # 如果给定的初始值不是可迭代对象，抛出类型错误异常
    if not iterable(init):
        raise TypeError("Expected a sequence of values for the initialization"
                        " of the recurrence")

    # 将 n 转换为整数类型
    n = as_int(n)
    # 如果 n 小于 0，则抛出值错误异常，说明递归计算的点必须是非负整数
    if n < 0:
        raise ValueError("Point of evaluation of recurrence must be a "
                        "non-negative integer")

    # 将 coeffs 列表中的每个元素转换为 sympy 对象，并存储在 c 列表中
    c = [sympify(arg) for arg in coeffs]
    
    # 将 init 列表中的每个元素转换为 sympy 对象，并存储在 b 列表中
    b = [sympify(arg) for arg in init]
    
    # 获取递归方程的阶数
    k = len(c)

    # 如果初始值列表 init 的长度大于递归方程的阶数 k，则抛出类型错误异常，
    # 说明初始值的数量不应该超过递归方程的阶数
    if len(b) > k:
        raise TypeError("Count of initial values should not exceed the "
                        "order of the recurrence")
    else:
        # 如果初始值列表 init 的长度小于等于递归方程的阶数 k，
        # 则在列表末尾补充长度为 k - len(b) 的 S.Zero 对象，即默认值为零
        b += [S.Zero]*(k - len(b)) # remaining initial values default to zero

    # 如果 n 小于递归方程的阶数 k，则直接返回初始值列表中的第 n 个值
    if n < k:
        return b[n]
    
    # 调用 linrec_coeffs 函数计算长度为 n 的线性递归系数列表，
    # 然后将每个系数与初始值列表 b 对应元素相乘，形成 terms 列表
    terms = [u*v for u, v in zip(linrec_coeffs(c, n), b)]
    
    # 将 terms 列表中除最后一个元素外的所有元素相加，再加上最后一个元素，返回结果
    return sum(terms[:-1], terms[-1])
# 计算线性递归序列中第 n 项的系数。
def linrec_coeffs(c, n):
    """
    Compute the coefficients of n'th term in linear recursion
    sequence defined by c.

    `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \cdots + c_{k-1}`.

    It computes the coefficients by using binary exponentiation.
    This function is used by `linrec` and `_eval_pow_by_cayley`.

    Parameters
    ==========

    c = coefficients of the divisor polynomial
    n = exponent of x, so dividend is x^n

    """

    k = len(c)

    def _square_and_reduce(u, offset):
        # squares `(u_0 + u_1 x + u_2 x^2 + \cdots + u_{k-1} x^k)` (and
        # multiplies by `x` if offset is 1) and reduces the above result of
        # length upto `2k` to `k` using the characteristic equation of the
        # recurrence given by, `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \cdots + c_{k-1}`

        # 初始化一个长度为 2*len(u) - 1 + offset 的零列表
        w = [S.Zero]*(2*len(u) - 1 + offset)
        # 计算 u 和 u 的乘积，并根据偏移量加到 w 对应位置上
        for i, p in enumerate(u):
            for j, q in enumerate(u):
                w[offset + i + j] += p*q

        # 使用递推关系的特征方程将长度为 2k 的结果减少到 k
        for j in range(len(w) - 1, k - 1, -1):
            for i in range(k):
                w[j - i - 1] += w[j]*c[i]

        return w[:k]

    def _final_coeffs(n):
        # 计算最终的系数列表 `cf`，对应于递归评估的点 `n`
        # 使得 `y(n) = cf_0 y(k-1) + cf_1 y(k-2) + \cdots + cf_{k-1} y(0)`

        if n < k:
            return [S.Zero]*n + [S.One] + [S.Zero]*(k - n - 1)
        else:
            return _square_and_reduce(_final_coeffs(n // 2), n % 2)

    return _final_coeffs(n)
```