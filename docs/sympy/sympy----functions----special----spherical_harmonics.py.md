# `D:\src\scipysrc\sympy\sympy\functions\special\spherical_harmonics.py`

```
# 从 sympy.core.expr 模块导入 Expr 类，表示表达式
# 从 sympy.core.function 模块导入 Function 类和 ArgumentIndexError 异常类
# 从 sympy.core.numbers 模块导入 I 和 pi 常数
# 从 sympy.core.singleton 模块导入 S 单例对象
# 从 sympy.core.symbol 模块导入 Dummy 类，用于创建符号变量
# 从 sympy.functions 模块导入 assoc_legendre 函数，表示关联勒让德多项式
# 从 sympy.functions.combinatorial.factorials 模块导入 factorial 函数，表示阶乘
# 从 sympy.functions.elementary.complexes 模块导入 Abs 和 conjugate 函数，表示复数的绝对值和共轭
# 从 sympy.functions.elementary.exponential 模块导入 exp 函数，表示指数函数
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt 函数，表示平方根函数
# 从 sympy.functions.elementary.trigonometric 模块导入 sin, cos, cot 函数，表示正弦、余弦和余切函数

# 创建一个虚拟变量 _x，用于代表符号变量
_x = Dummy("x")

# 定义一个 Ynm 类，继承自 Function 类，表示球谐函数
class Ynm(Function):
    r"""
    Spherical harmonics defined as

    .. math::
        Y_n^m(\theta, \varphi) := \sqrt{\frac{(2n+1)(n-m)!}{4\pi(n+m)!}}
                                  \exp(i m \varphi)
                                  \mathrm{P}_n^m\left(\cos(\theta)\right)

    Explanation
    ===========

    ``Ynm()`` gives the spherical harmonic function of order $n$ and $m$
    in $\theta$ and $\varphi$, $Y_n^m(\theta, \varphi)$. The four
    parameters are as follows: $n \geq 0$ an integer and $m$ an integer
    such that $-n \leq m \leq n$ holds. The two angles are real-valued
    with $\theta \in [0, \pi]$ and $\varphi \in [0, 2\pi]$.

    Examples
    ========

    >>> from sympy import Ynm, Symbol, simplify
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")

    >>> Ynm(n, m, theta, phi)
    Ynm(n, m, theta, phi)

    Several symmetries are known, for the order:

    >>> Ynm(n, -m, theta, phi)
    (-1)**m*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)

    As well as for the angles:

    >>> Ynm(n, m, -theta, phi)
    Ynm(n, m, theta, phi)

    >>> Ynm(n, m, theta, -phi)
    exp(-2*I*m*phi)*Ynm(n, m, theta, phi)

    For specific integers $n$ and $m$ we can evaluate the harmonics
    to more useful expressions:

    >>> simplify(Ynm(0, 0, theta, phi).expand(func=True))
    1/(2*sqrt(pi))

    >>> simplify(Ynm(1, -1, theta, phi).expand(func=True))
    sqrt(6)*exp(-I*phi)*sin(theta)/(4*sqrt(pi))

    >>> simplify(Ynm(1, 0, theta, phi).expand(func=True))
    sqrt(3)*cos(theta)/(2*sqrt(pi))

    >>> simplify(Ynm(1, 1, theta, phi).expand(func=True))
    -sqrt(6)*exp(I*phi)*sin(theta)/(4*sqrt(pi))

    >>> simplify(Ynm(2, -2, theta, phi).expand(func=True))
    sqrt(30)*exp(-2*I*phi)*sin(theta)**2/(8*sqrt(pi))

    >>> simplify(Ynm(2, -1, theta, phi).expand(func=True))
    sqrt(30)*exp(-I*phi)*sin(2*theta)/(8*sqrt(pi))

    >>> simplify(Ynm(2, 0, theta, phi).expand(func=True))
    sqrt(5)*(3*cos(theta)**2 - 1)/(4*sqrt(pi))

    >>> simplify(Ynm(2, 1, theta, phi).expand(func=True))
    -sqrt(30)*exp(I*phi)*sin(2*theta)/(8*sqrt(pi))

    >>> simplify(Ynm(2, 2, theta, phi).expand(func=True))
    sqrt(30)*exp(2*I*phi)*sin(theta)**2/(8*sqrt(pi))

    We can differentiate the functions with respect
    to both angles:

    >>> from sympy import Ynm, Symbol, diff
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")


# 定义符号变量 phi
phi = Symbol("phi")



    >>> diff(Ynm(n, m, theta, phi), theta)
    m*cot(theta)*Ynm(n, m, theta, phi) + sqrt((-m + n)*(m + n + 1))*exp(-I*phi)*Ynm(n, m + 1, theta, phi)


# 对 Ynm(n, m, theta, phi) 求关于 theta 的偏导数
result = m*cot(theta)*Ynm(n, m, theta, phi) + sqrt((-m + n)*(m + n + 1))*exp(-I*phi)*Ynm(n, m + 1, theta, phi)



    >>> diff(Ynm(n, m, theta, phi), phi)
    I*m*Ynm(n, m, theta, phi)


# 对 Ynm(n, m, theta, phi) 求关于 phi 的偏导数
result = I*m*Ynm(n, m, theta, phi)



    Further we can compute the complex conjugation:


# 进一步计算复共轭

    >>> from sympy import Ynm, Symbol, conjugate
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")



    >>> conjugate(Ynm(n, m, theta, phi))
    (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)


# 计算 Ynm(n, m, theta, phi) 的复共轭
result = (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)



    To get back the well known expressions in spherical
    coordinates, we use full expansion:


# 为了得到球坐标中众所周知的表达式，我们使用完全展开

    >>> from sympy import Ynm, Symbol, expand_func
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")



    >>> expand_func(Ynm(n, m, theta, phi))
    sqrt((2*n + 1)*factorial(-m + n)/factorial(m + n))*exp(I*m*phi)*assoc_legendre(n, m, cos(theta))/(2*sqrt(pi))


# 对 Ynm(n, m, theta, phi) 进行展开
result = sqrt((2*n + 1)*factorial(-m + n)/factorial(m + n))*exp(I*m*phi)*assoc_legendre(n, m, cos(theta))/(2*sqrt(pi))



    See Also
    ========

    Ynm_c, Znm

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics
    .. [2] https://mathworld.wolfram.com/SphericalHarmonic.html
    .. [3] https://functions.wolfram.com/Polynomials/SphericalHarmonicY/
    .. [4] https://dlmf.nist.gov/14.30


# 参见

    Ynm_c, Znm

# 参考文献

    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics
    .. [2] https://mathworld.wolfram.com/SphericalHarmonic.html
    .. [3] https://functions.wolfram.com/Polynomials/SphericalHarmonicY/
    .. [4] https://dlmf.nist.gov/14.30



    """

    @classmethod
    def eval(cls, n, m, theta, phi):
        # 处理负指数 m 和参数 theta, phi
        if m.could_extract_minus_sign():
            m = -m
            return S.NegativeOne**m * exp(-2*I*m*phi) * Ynm(n, m, theta, phi)
        if theta.could_extract_minus_sign():
            theta = -theta
            return Ynm(n, m, theta, phi)
        if phi.could_extract_minus_sign():
            phi = -phi
            return exp(-2*I*m*phi) * Ynm(n, m, theta, phi)

        # TODO Add more simplififcation here


    # 类方法，用于计算 Ynm(n, m, theta, phi) 的特定表达式
    def eval(cls, n, m, theta, phi):
        # 处理负指数 m 和参数 theta, phi 的情况
        if m.could_extract_minus_sign():
            m = -m
            return S.NegativeOne**m * exp(-2*I*m*phi) * Ynm(n, m, theta, phi)
        if theta.could_extract_minus_sign():
            theta = -theta
            return Ynm(n, m, theta, phi)
        if phi.could_extract_minus_sign():
            phi = -phi
            return exp(-2*I*m*phi) * Ynm(n, m, theta, phi)

        # TODO 在这里添加更多的简化操作

    def _eval_expand_func(self, **hints):
        n, m, theta, phi = self.args
        # 展开 Ynm(n, m, theta, phi) 的特定函数
        rv = (sqrt((2*n + 1)/(4*pi) * factorial(n - m)/factorial(n + m)) *
                exp(I*m*phi) * assoc_legendre(n, m, cos(theta)))
        # 因为 theta 的范围，可以做如下替换
        return rv.subs(sqrt(-cos(theta)**2 + 1), sin(theta))

    def fdiff(self, argindex=4):
        if argindex == 1:
            # 对 n 求偏导数
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # 对 m 求偏导数
            raise ArgumentIndexError(self, argindex)
        elif argindex == 3:
            # 对 theta 求偏导数
            n, m, theta, phi = self.args
            return (m * cot(theta) * Ynm(n, m, theta, phi) +
                    sqrt((n - m)*(n + m + 1)) * exp(-I*phi) * Ynm(n, m + 1, theta, phi))
        elif argindex == 4:
            # 对 phi 求偏导数
            n, m, theta, phi = self.args
            return I * m * Ynm(n, m, theta, phi)
        else:
            raise ArgumentIndexError(self, argindex)
    def _eval_rewrite_as_polynomial(self, n, m, theta, phi, **kwargs):
        # TODO: Make sure n \in N (确保 n 是自然数)
        # TODO: Assert |m| <= n ortherwise we should return 0 (断言 |m| <= n，否则返回 0)
        # 调用父类方法展开表达式
        return self.expand(func=True)

    def _eval_rewrite_as_sin(self, n, m, theta, phi, **kwargs):
        # 调用实例方法使用 sin 重写
        return self.rewrite(cos)

    def _eval_rewrite_as_cos(self, n, m, theta, phi, **kwargs):
        # This method can be expensive due to extensive use of simplification! (这个方法可能因为大量简化运算而显得耗时)
        from sympy.simplify import simplify, trigsimp
        # TODO: Make sure n \in N (确保 n 是自然数)
        # TODO: Assert |m| <= n ortherwise we should return 0 (断言 |m| <= n，否则返回 0)
        # 将展开后的表达式简化
        term = simplify(self.expand(func=True))
        # 因为 theta 的范围允许，可以进行下面这个替换
        term = term.xreplace({Abs(sin(theta)): sin(theta)})
        # 对整体表达式进行三角函数简化
        return simplify(trigsimp(term))

    def _eval_conjugate(self):
        # TODO: Make sure theta \in R and phi \in R (确保 theta 和 phi 是实数)
        n, m, theta, phi = self.args
        # 返回共轭的球谐函数表达式
        return S.NegativeOne**m * self.func(n, -m, theta, phi)

    def as_real_imag(self, deep=True, **hints):
        # TODO: Handle deep and hints (处理 deep 和 hints 参数)
        n, m, theta, phi = self.args
        # 计算实部和虚部
        re = (sqrt((2*n + 1)/(4*pi) * factorial(n - m)/factorial(n + m)) *
              cos(m*phi) * assoc_legendre(n, m, cos(theta)))
        im = (sqrt((2*n + 1)/(4*pi) * factorial(n - m)/factorial(n + m)) *
              sin(m*phi) * assoc_legendre(n, m, cos(theta)))
        # 返回实部和虚部的元组
        return (re, im)

    def _eval_evalf(self, prec):
        # Note: works without this function by just calling
        #       mpmath for Legendre polynomials. But using
        #       the dedicated function directly is cleaner.
        from mpmath import mp, workprec
        n = self.args[0]._to_mpmath(prec)
        m = self.args[1]._to_mpmath(prec)
        theta = self.args[2]._to_mpmath(prec)
        phi = self.args[3]._to_mpmath(prec)
        # 在指定精度下计算球谐函数的数值近似
        with workprec(prec):
            res = mp.spherharm(n, m, theta, phi)
        return Expr._from_mpmath(res, prec)
# 定义函数 Ynm_c，计算共轭球谐函数
def Ynm_c(n, m, theta, phi):
    """
    Conjugate spherical harmonics defined as

    .. math::
        \overline{Y_n^m(\theta, \varphi)} := (-1)^m Y_n^{-m}(\theta, \varphi).

    Examples
    ========

    >>> from sympy import Ynm_c, Symbol, simplify
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")
    >>> Ynm_c(n, m, theta, phi)
    (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)
    >>> Ynm_c(n, m, -theta, phi)
    (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)

    For specific integers $n$ and $m$ we can evaluate the harmonics
    to more useful expressions:

    >>> simplify(Ynm_c(0, 0, theta, phi).expand(func=True))
    1/(2*sqrt(pi))
    >>> simplify(Ynm_c(1, -1, theta, phi).expand(func=True))
    sqrt(6)*exp(I*(-phi + 2*conjugate(phi)))*sin(theta)/(4*sqrt(pi))

    See Also
    ========

    Ynm, Znm

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics
    .. [2] https://mathworld.wolfram.com/SphericalHarmonic.html
    .. [3] https://functions.wolfram.com/Polynomials/SphericalHarmonicY/

    """
    # 返回球谐函数 Ynm 的共轭
    return conjugate(Ynm(n, m, theta, phi))


# 定义类 Znm，表示实数球谐函数
class Znm(Function):
    """
    Real spherical harmonics defined as

    .. math::

        Z_n^m(\theta, \varphi) :=
        \begin{cases}
          \frac{Y_n^m(\theta, \varphi) + \overline{Y_n^m(\theta, \varphi)}}{\sqrt{2}} &\quad m > 0 \\
          Y_n^m(\theta, \varphi) &\quad m = 0 \\
          \frac{Y_n^m(\theta, \varphi) - \overline{Y_n^m(\theta, \varphi)}}{i \sqrt{2}} &\quad m < 0 \\
        \end{cases}

    which gives in simplified form

    .. math::

        Z_n^m(\theta, \varphi) =
        \begin{cases}
          \frac{Y_n^m(\theta, \varphi) + (-1)^m Y_n^{-m}(\theta, \varphi)}{\sqrt{2}} &\quad m > 0 \\
          Y_n^m(\theta, \varphi) &\quad m = 0 \\
          \frac{Y_n^m(\theta, \varphi) - (-1)^m Y_n^{-m}(\theta, \varphi)}{i \sqrt{2}} &\quad m < 0 \\
        \end{cases}

    Examples
    ========

    >>> from sympy import Znm, Symbol, simplify
    >>> from sympy.abc import n, m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")
    >>> Znm(n, m, theta, phi)
    Znm(n, m, theta, phi)

    For specific integers n and m we can evaluate the harmonics
    to more useful expressions:

    >>> simplify(Znm(0, 0, theta, phi).expand(func=True))
    1/(2*sqrt(pi))
    >>> simplify(Znm(1, 1, theta, phi).expand(func=True))
    -sqrt(3)*sin(theta)*cos(phi)/(2*sqrt(pi))
    >>> simplify(Znm(2, 1, theta, phi).expand(func=True))
    -sqrt(15)*sin(2*theta)*cos(phi)/(4*sqrt(pi))

    See Also
    ========

    Ynm, Ynm_c

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics
    .. [2] https://mathworld.wolfram.com/SphericalHarmonic.html
    .. [3] https://functions.wolfram.com/Polynomials/SphericalHarmonicY/

    """

    @classmethod
    # 定义一个类方法 `eval`，用于计算特定的球谐函数值
    def eval(cls, n, m, theta, phi):
        # 如果 m 是正数，则计算球谐函数 Ynm 和其共轭 Ynm_c 的平均值除以根号2
        if m.is_positive:
            zz = (Ynm(n, m, theta, phi) + Ynm_c(n, m, theta, phi)) / sqrt(2)
            return zz
        # 如果 m 是零，则直接返回球谐函数 Ynm 的值
        elif m.is_zero:
            return Ynm(n, m, theta, phi)
        # 如果 m 是负数，则计算球谐函数 Ynm 和其共轭 Ynm_c 的差值除以（根号2 * 虚数单位 i）
        elif m.is_negative:
            zz = (Ynm(n, m, theta, phi) - Ynm_c(n, m, theta, phi)) / (sqrt(2)*I)
            return zz
```