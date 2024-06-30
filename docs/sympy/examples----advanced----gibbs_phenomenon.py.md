# `D:\src\scipysrc\sympy\examples\advanced\gibbs_phenomenon.py`

```
#!/usr/bin/env python

"""
This example illustrates the Gibbs phenomenon.

It also calculates the Wilbraham-Gibbs constant by two approaches:

1) calculating the Fourier series of the step function and determining the
first maximum.
2) evaluating the integral for si(pi).

See:
 * https://en.wikipedia.org/wiki/Gibbs_phenomena
"""

from sympy import var, sqrt, integrate, conjugate, seterr, Abs, pprint, I, pi,\
    sin, cos, sign, lambdify, Integral, S

x = var("x", real=True)


def l2_norm(f, lim):
    """
    Calculates L2 norm of the function "f", over the domain lim=(x, a, b).

    x ...... the independent variable in f over which to integrate
    a, b ... the limits of the interval

    Examples
    ========

    >>> from sympy import Symbol
    >>> from gibbs_phenomenon import l2_norm
    >>> x = Symbol('x', real=True)
    >>> l2_norm(1, (x, -1, 1))
    sqrt(2)
    >>> l2_norm(x, (x, -1, 1))
    sqrt(6)/3

    """
    return sqrt(integrate(Abs(f)**2, lim))


def l2_inner_product(a, b, lim):
    """
    Calculates the L2 inner product (a, b) over the domain lim.
    """
    return integrate(conjugate(a)*b, lim)


def l2_projection(f, basis, lim):
    """
    L2 projects the function f on the basis over the domain lim.
    """
    r = 0
    for b in basis:
        r += l2_inner_product(f, b, lim) * b
    return r


def l2_gram_schmidt(list, lim):
    """
    Orthonormalizes the "list" of functions using the Gram-Schmidt process.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from gibbs_phenomenon import l2_gram_schmidt

    >>> x = Symbol('x', real=True)    # perform computations over reals to save time
    >>> l2_gram_schmidt([1, x, x**2], (x, -1, 1))
    [sqrt(2)/2, sqrt(6)*x/2, 3*sqrt(10)*(x**2 - 1/3)/4]

    """
    r = []
    for a in list:
        if r == []:
            v = a
        else:
            v = a - l2_projection(a, r, lim)
        v_norm = l2_norm(v, lim)
        if v_norm == 0:
            raise ValueError("The sequence is not linearly independent.")
        r.append(v/v_norm)
    return r


def integ(f):
    """
    Integrates the function f over the domain (-pi, pi).

    Examples
    ========

    >>> from gibbs_phenomenon import integ
    >>> integ(sin(x))
    0

    """
    return integrate(f, (x, -pi, pi))


def series(L):
    """
    Normalizes the series.

    """
    r = 0
    for b in L:
        r += integ(b)*b
    return r


def msolve(f, x):
    """
    Finds the first root of f(x) to the left of 0.

    The x0 and dx below are tailored to get the correct result for our
    particular function --- the general solver often overshoots the first
    solution.
    """
    f = lambdify(x, f)
    x0 = -0.001
    dx = 0.001
    while f(x0 - dx) * f(x0) > 0:
        x0 = x0 - dx
    x_max = x0 - dx
    x_min = x0
    assert f(x_max) > 0
    assert f(x_min) < 0
    for n in range(100):
        x0 = (x_max + x_min)/2
        if f(x0) > 0:
            x_max = x0
        else:
            x_min = x0
    return x0


def main():
    """
    Initializes and computes a list L containing Fourier series components.

    """
    L = [1]
    for i in range(1, 100):
        L.append(cos(i*x))
        L.append(sin(i*x))
    # 将 L 列表中的第一个元素除以 sqrt(2)，以完成归一化操作
    L[0] /= sqrt(2)
    # 使用列表推导式对 L 中的每个元素进行操作，使其除以 sqrt(pi)，得到归一化的 Fourier 系数列表
    L = [f/sqrt(pi) for f in L]

    # 计算 Fourier 级数
    f = series(L)
    # 打印输出 Fourier 级数的结果
    print("Fourier series of the step function")
    # 使用 pprint 函数美观地输出 Fourier 级数 f
    pprint(f)
    # 解方程 f.diff(x) = 0，找到其解 x0
    x0 = msolve(f.diff(x), x)

    # 打印输出最大值所对应的 x 值
    print("x-value of the maximum:", x0)
    # 计算并打印最大值 max，并将 x 替换为 x0 后进行数值化
    max = f.subs(x, x0).evalf()
    # 打印输出最大值所对应的 y 值
    print("y-value of the maximum:", max)
    # 计算并打印 Wilbraham-Gibbs 常数 g，其为最大值 max 乘以 pi/2
    g = max*pi/2
    print("Wilbraham-Gibbs constant        :", g.evalf())
    # 打印输出精确的 Wilbraham-Gibbs 常数，即 sin(x)/x 在 [0, pi] 区间上的积分结果
    print("Wilbraham-Gibbs constant (exact):", \
        Integral(sin(x)/x, (x, 0, pi)).evalf())
if __name__ == "__main__":
    # 程序的入口点，判断当前模块是否作为主程序执行
    main()
```