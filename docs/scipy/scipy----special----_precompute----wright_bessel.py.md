# `D:\src\scipysrc\scipy\scipy\special\_precompute\wright_bessel.py`

```
"""Precompute coefficients of several series expansions
of Wright's generalized Bessel function Phi(a, b, x).

See https://dlmf.nist.gov/10.46.E1 with rho=a, beta=b, z=x.
"""
# 导入必要的模块
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, curve_fit
from time import time

try:
    import sympy
    from sympy import EulerGamma, Rational, S, Sum, \
        factorial, gamma, gammasimp, pi, polygamma, symbols, zeta
    from sympy.polys.polyfuncs import horner
except ImportError:
    pass


def series_small_a():
    """Tylor series expansion of Phi(a, b, x) in a=0 up to order 5.
    """
    order = 5
    a, b, x, k = symbols("a b x k")
    A = []  # terms with a
    X = []  # terms with x
    B = []  # terms with b (polygammas)
    # Phi(a, b, x) = exp(x)/gamma(b) * sum(A[i] * X[i] * B[i])
    expression = Sum(x**k/factorial(k)/gamma(a*k+b), (k, 0, S.Infinity))
    expression = gamma(b)/sympy.exp(x) * expression

    # nth term of taylor series in a=0: a^n/n! * (d^n Phi(a, b, x)/da^n at a=0)
    for n in range(0, order+1):
        term = expression.diff(a, n).subs(a, 0).simplify().doit()
        # set the whole bracket involving polygammas to 1
        x_part = (term.subs(polygamma(0, b), 1)
                  .replace(polygamma, lambda *args: 0))
        # sign convention: x part always positive
        x_part *= (-1)**n

        A.append(a**n/factorial(n))
        X.append(horner(x_part))
        B.append(horner((term/x_part).simplify()))

    s = "Tylor series expansion of Phi(a, b, x) in a=0 up to order 5.\n"
    s += "Phi(a, b, x) = exp(x)/gamma(b) * sum(A[i] * X[i] * B[i], i=0..5)\n"
    for name, c in zip(['A', 'X', 'B'], [A, X, B]):
        for i in range(len(c)):
            s += f"\n{name}[{i}] = " + str(c[i])
    return s


# expansion of digamma
def dg_series(z, n):
    """Symbolic expansion of digamma(z) in z=0 to order n.

    See https://dlmf.nist.gov/5.7.E4 and with https://dlmf.nist.gov/5.5.E2
    """
    k = symbols("k")
    return -1/z - EulerGamma + \
        sympy.summation((-1)**k * zeta(k) * z**(k-1), (k, 2, n+1))


def pg_series(k, z, n):
    """Symbolic expansion of polygamma(k, z) in z=0 to order n."""
    return sympy.diff(dg_series(z, n+k), z, k)


def series_small_a_small_b():
    """Tylor series expansion of Phi(a, b, x) in a=0 and b=0 up to order 5.

    Be aware of cancellation of poles in b=0 of digamma(b)/Gamma(b) and
    polygamma functions.

    digamma(b)/Gamma(b) = -1 - 2*M_EG*b + O(b^2)
    digamma(b)^2/Gamma(b) = 1/b + 3*M_EG + b*(-5/12*PI^2+7/2*M_EG^2) + O(b^2)
    polygamma(1, b)/Gamma(b) = 1/b + M_EG + b*(1/12*PI^2 + 1/2*M_EG^2) + O(b^2)
    and so on.
    """
    order = 5
    a, b, x, k = symbols("a b x k")
    M_PI, M_EG, M_Z3 = symbols("M_PI M_EG M_Z3")
    c_subs = {pi: M_PI, EulerGamma: M_EG, zeta(3): M_Z3}
    A = []  # terms with a
    X = []  # terms with x
    B = []  # terms with b (polygammas expanded)
    C = []  # terms that generate B
    # Phi(a, b, x) = exp(x) * sum(A[i] * X[i] * B[i])
    # B[0] = 1
    # B[k] = sum(C[k] * b**k/k!, k=0..)
    # Note: C[k] can be obtained from a series expansion of 1/gamma(b).
    expression = gamma(b)/sympy.exp(x) * \
        Sum(x**k/factorial(k)/gamma(a*k+b), (k, 0, S.Infinity))

    # nth term of taylor series in a=0: a^n/n! * (d^n Phi(a, b, x)/da^n at a=0)
    for n in range(0, order+1):
        term = expression.diff(a, n).subs(a, 0).simplify().doit()
        # set the whole bracket involving polygammas to 1
        x_part = (term.subs(polygamma(0, b), 1)
                  .replace(polygamma, lambda *args: 0))
        # sign convention: x part always positive
        x_part *= (-1)**n
        # expansion of polygamma part with 1/gamma(b)
        pg_part = term/x_part/gamma(b)
        if n >= 1:
            # Note: highest term is digamma^n
            pg_part = pg_part.replace(polygamma,
                                      lambda k, x: pg_series(k, x, order+1+n))
            pg_part = (pg_part.series(b, 0, n=order+1-n)
                       .removeO()
                       .subs(polygamma(2, 1), -2*zeta(3))
                       .simplify()
                       )

        A.append(a**n/factorial(n))
        X.append(horner(x_part))
        B.append(pg_part)

    # Calculate C and put in the k!
    C = sympy.Poly(B[1].subs(c_subs), b).coeffs()
    C.reverse()
    for i in range(len(C)):
        C[i] = (C[i] * factorial(i)).simplify()

    s = "Tylor series expansion of Phi(a, b, x) in a=0 and b=0 up to order 5."
    s += "\nPhi(a, b, x) = exp(x) * sum(A[i] * X[i] * B[i], i=0..5)\n"
    s += "B[0] = 1\n"
    s += "B[i] = sum(C[k+i-1] * b**k/k!, k=0..)\n"
    s += "\nM_PI = pi"
    s += "\nM_EG = EulerGamma"
    s += "\nM_Z3 = zeta(3)"
    for name, c in zip(['A', 'X'], [A, X]):
        for i in range(len(c)):
            s += f"\n{name}[{i}] = "
            s += str(c[i])
    # For C, do also compute the values numerically
    for i in range(len(C)):
        s += f"\n# C[{i}] = "
        s += str(C[i])
        s += f"\nC[{i}] = "
        s += str(C[i].subs({M_EG: EulerGamma, M_PI: pi, M_Z3: zeta(3)})
                 .evalf(17))

    # Does B have the assumed structure?
    s += "\n\nTest if B[i] does have the assumed structure."
    s += "\nC[i] are derived from B[1] alone."
    s += "\nTest B[2] == C[1] + b*C[2] + b^2/2*C[3] + b^3/6*C[4] + .."
    test = sum([b**k/factorial(k) * C[k+1] for k in range(order-1)])
    test = (test - B[2].subs(c_subs)).simplify()
    s += f"\ntest successful = {test==S(0)}"
    s += "\nTest B[3] == C[2] + b*C[3] + b^2/2*C[4] + .."
    test = sum([b**k/factorial(k) * C[k+2] for k in range(order-2)])
    test = (test - B[3].subs(c_subs)).simplify()
    s += f"\ntest successful = {test==S(0)}"
    return s


注释：
- `C = []`: 初始化一个空列表，用于存储生成 B 的项。
- `expression = ...`: 定义了 Phi(a, b, x) 的表达式，是一个数学公式，用于后续的数值计算。
- `for n in range(0, order+1):`: 循环计算 Taylor 级数的每一项。
- `A.append(a**n/factorial(n))`, `X.append(horner(x_part))`, `B.append(pg_part)`: 分别向列表 A、X、B 添加计算得到的数值。
- `C = sympy.Poly(B[1].subs(c_subs), b).coeffs()`: 计算 C 列表的值，并进行逆序处理。
- 循环和字符串操作部分：生成描述数学表达式和计算结果的文本描述。
- 最后的测试部分：用于验证 B 的结构是否满足预期，并输出测试结果。
def asymptotic_series():
    """Asymptotic expansion for large x.

    Phi(a, b, x) ~ Z^(1/2-b) * exp((1+a)/a * Z) * sum_k (-1)^k * C_k / Z^k
    Z = (a*x)^(1/(1+a))

    Wright (1935) lists the coefficients C_0 and C_1 (he calls them a_0 and
    a_1). With slightly different notation, Paris (2017) lists coefficients
    c_k up to order k=3.
    Paris (2017) uses ZP = (1+a)/a * Z  (ZP = Z of Paris) and
    C_k = C_0 * (-a/(1+a))^k * c_k
    """
    order = 8  # 定义展开的阶数为8

    class g(sympy.Function):
        """Helper function g according to Wright (1935)

        g(n, rho, v) = (1 + (rho+2)/3 * v + (rho+2)*(rho+3)/(2*3) * v^2 + ...)

        Note: Wright (1935) uses square root of above definition.
        """
        nargs = 3  # 定义函数 g 的参数数量为3

        @classmethod
        def eval(cls, n, rho, v):
            if not n >= 0:
                raise ValueError("must have n >= 0")
            elif n == 0:
                return 1  # 当 n 为 0 时，返回 1
            else:
                return g(n-1, rho, v) \
                    + gammasimp(gamma(rho+2+n)/gamma(rho+2)) \
                    / gammasimp(gamma(3+n)/gamma(3))*v**n  # 递归计算 g 函数的值

    class coef_C(sympy.Function):
        """Calculate coefficients C_m for integer m.

        C_m is the coefficient of v^(2*m) in the Taylor expansion in v=0 of
        Gamma(m+1/2)/(2*pi) * (2/(rho+1))^(m+1/2) * (1-v)^(-b)
            * g(rho, v)^(-m-1/2)
        """
        nargs = 3  # 定义函数 coef_C 的参数数量为3

        @classmethod
        def eval(cls, m, rho, beta):
            if not m >= 0:
                raise ValueError("must have m >= 0")

            v = symbols("v")
            expression = (1-v)**(-beta) * g(2*m, rho, v)**(-m-Rational(1, 2))
            res = expression.diff(v, 2*m).subs(v, 0) / factorial(2*m)  # 计算表达式的二阶导数在 v=0 处的值
            res = res * (gamma(m + Rational(1, 2)) / (2*pi)
                         * (2/(rho+1))**(m + Rational(1, 2)))  # 计算最终的系数 res
            return res

    # in order to have nice ordering/sorting of expressions, we set a = xa.
    xa, b, xap1 = symbols("xa b xap1")
    C0 = coef_C(0, xa, b)  # 计算 C_0 的值
    s = "Asymptotic expansion for large x\n"
    s += "Phi(a, b, x) = Z**(1/2-b) * exp((1+a)/a * Z) \n"
    s += "               * sum((-1)**k * C[k]/Z**k, k=0..6)\n\n"
    s += "Z      = pow(a * x, 1/(1+a))\n"
    s += "A[k]   = pow(a, k)\n"
    s += "B[k]   = pow(b, k)\n"
    s += "Ap1[k] = pow(1+a, k)\n\n"
    s += "C[0] = 1./sqrt(2. * M_PI * Ap1[1])\n"
    for i in range(1, order+1):
        expr = (coef_C(i, xa, b) / (C0/(1+xa)**i)).simplify()  # 计算 C[i] 的表达式，并简化
        factor = [x.denominator() for x in sympy.Poly(expr).coeffs()]
        factor = sympy.lcm(factor)  # 计算表达式的最小公倍数
        expr = (expr * factor).simplify().collect(b, sympy.factor)  # 收集表达式中的 b，并进行因式分解
        expr = expr.xreplace({xa+1: xap1})  # 替换表达式中的 xa+1 为 xap1
        s += f"C[{i}] = C[0] / ({factor} * Ap1[{i}])\n"
        s += f"C[{i}] *= {str(expr)}\n\n"
    import re
    re_a = re.compile(r'xa\*\*(\d+)')
    s = re_a.sub(r'A[\1]', s)  # 将 s 中的 xa**n 替换为 A[n]
    re_b = re.compile(r'b\*\*(\d+)')
    s = re_b.sub(r'B[\1]', s)  # 将 s 中的 b**n 替换为 B[n]
    s = s.replace('xap1', 'Ap1[1]')  # 将 s 中的 xap1 替换为 Ap1[1]
    s = s.replace('xa', 'a')  # 将 s 中的 xa 替换为 a
    # 使用正则表达式找出字符串 s 中长度为 10 或更多的连续数字
    re_digits = re.compile(r'(\d{10,})')
    # 将匹配到的连续数字替换为该数字后跟一个点号，用于处理超过 10 位的整数
    s = re_digits.sub(r'\1.', s)
    # 返回处理后的字符串 s
    return s
def main():
    # 记录程序开始时间
    t0 = time()
    # 创建参数解析器，使用文档字符串作为描述信息，使用RawTextHelpFormatter格式化输出
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawTextHelpFormatter)
    # 添加一个位置参数 'action'，类型为整数，只能是 [1, 2, 3, 4] 中的一个，用于选择预计算的操作类型
    parser.add_argument('action', type=int, choices=[1, 2, 3, 4],
                        help='chose what expansion to precompute\n'
                             '1 : Series for small a\n'
                             '2 : Series for small a and small b\n'
                             '3 : Asymptotic series for large x\n'
                             '    This may take some time (>4h).\n'
                             '4 : Fit optimal eps for integral representation.'
                        )
    # 解析命令行参数
    args = parser.parse_args()

    # 创建一个字典，将每个操作映射到相应的函数调用
    switch = {1: lambda: print(series_small_a()),
              2: lambda: print(series_small_a_small_b()),
              3: lambda: print(asymptotic_series()),
              4: lambda: print(optimal_epsilon_integral())
              }
    # 根据用户输入的操作选择对应的函数，并执行
    switch.get(args.action, lambda: print("Invalid input."))()
    # 输出运行时间（从开始到结束的分钟数）
    print(f"\n{(time() - t0)/60:.1f} minutes elapsed.\n")
# 如果当前脚本作为主程序运行（而不是被导入为模块），则执行下面的代码块
if __name__ == '__main__':
    # 调用主函数 main()
    main()
```