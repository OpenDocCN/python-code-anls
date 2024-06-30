# `D:\src\scipysrc\sympy\sympy\solvers\tests\test_recurr.py`

```
from sympy.core.function import (Function, Lambda, expand)  # 导入函数类和扩展函数
from sympy.core.numbers import (I, Rational)  # 导入虚数和有理数
from sympy.core.relational import Eq  # 导入等式类
from sympy.core.singleton import S  # 导入单例类
from sympy.core.symbol import (Symbol, symbols)  # 导入符号类和符号列表
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)  # 导入阶乘相关函数
from sympy.functions.elementary.complexes import Abs  # 导入复数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入余弦和正弦函数
from sympy.polys.polytools import factor  # 导入因式分解函数
from sympy.solvers.recurr import rsolve, rsolve_hyper, rsolve_poly, rsolve_ratio  # 导入递推解函数
from sympy.testing.pytest import raises, slow, XFAIL  # 导入测试相关的装饰器
from sympy.abc import a, b  # 导入符号 a 和 b

y = Function('y')  # 创建函数对象 y
n, k = symbols('n,k', integer=True)  # 创建整数符号 n 和 k
C0, C1, C2 = symbols('C0,C1,C2')  # 创建符号 C0, C1, C2


def test_rsolve_poly():
    # 测试多项式类型的递推解
    assert rsolve_poly([-1, -1, 1], 0, n) == 0
    assert rsolve_poly([-1, -1, 1], 1, n) == -1

    assert rsolve_poly([-1, n + 1], n, n) == 1
    assert rsolve_poly([-1, 1], n, n) == C0 + (n**2 - n)/2
    assert rsolve_poly([-n - 1, n], 1, n) == C0*n - 1
    assert rsolve_poly([-4*n - 2, 1], 4*n + 1, n) == -1

    assert rsolve_poly([-1, 1], n**5 + n**3, n) == \
        C0 - n**3 / 2 - n**5 / 2 + n**2 / 6 + n**6 / 6 + 2*n**4 / 3


def test_rsolve_ratio():
    # 测试有理函数类型的递推解
    solution = rsolve_ratio([-2*n**3 + n**2 + 2*n - 1, 2*n**3 + n**2 - 6*n,
        -2*n**3 - 11*n**2 - 18*n - 9, 2*n**3 + 13*n**2 + 22*n + 8], 0, n)
    assert solution == C0*(2*n - 3)/(n**2 - 1)/2


def test_rsolve_hyper():
    # 测试超几何类型的递推解
    assert rsolve_hyper([-1, -1, 1], 0, n) in [
        C0*(S.Half - S.Half*sqrt(5))**n + C1*(S.Half + S.Half*sqrt(5))**n,
        C1*(S.Half - S.Half*sqrt(5))**n + C0*(S.Half + S.Half*sqrt(5))**n,
    ]

    assert rsolve_hyper([n**2 - 2, -2*n - 1, 1], 0, n) in [
        C0*rf(sqrt(2), n) + C1*rf(-sqrt(2), n),
        C1*rf(sqrt(2), n) + C0*rf(-sqrt(2), n),
    ]

    assert rsolve_hyper([n**2 - k, -2*n - 1, 1], 0, n) in [
        C0*rf(sqrt(k), n) + C1*rf(-sqrt(k), n),
        C1*rf(sqrt(k), n) + C0*rf(-sqrt(k), n),
    ]

    assert rsolve_hyper(
        [2*n*(n + 1), -n**2 - 3*n + 2, n - 1], 0, n) == C1*factorial(n) + C0*2**n

    assert rsolve_hyper(
        [n + 2, -(2*n + 3)*(17*n**2 + 51*n + 39), n + 1], 0, n) == 0

    assert rsolve_hyper([-n - 1, -1, 1], 0, n) == 0

    assert rsolve_hyper([-1, 1], n, n).expand() == C0 + n**2/2 - n/2

    assert rsolve_hyper([-1, 1], 1 + n, n).expand() == C0 + n**2/2 + n/2

    assert rsolve_hyper([-1, 1], 3*(n + n**2), n).expand() == C0 + n**3 - n

    assert rsolve_hyper([-a, 1],0,n).expand() == C0*a**n

    assert rsolve_hyper([-a, 0, 1], 0, n).expand() == (-1)**n*C1*a**(n/2) + C0*a**(n/2)

    assert rsolve_hyper([1, 1, 1], 0, n).expand() == \
        C0*(Rational(-1, 2) - sqrt(3)*I/2)**n + C1*(Rational(-1, 2) + sqrt(3)*I/2)**n

    assert rsolve_hyper([1, -2*n/a - 2/a, 1], 0, n) == 0


@XFAIL
def test_rsolve_ratio_missed():
    # this arises during computation
    # assert rsolve_hyper([-1, 1], 3*(n + n**2), n).expand() == C0 + n**3 - n
    # 调用 rsolve_ratio 函数，断言其返回值不为 None
    assert rsolve_ratio([-n, n + 2], n, n) is not None
def recurrence_term(c, f):
    """Compute RHS of recurrence in f(n) with coefficients in c."""
    # 计算具有系数 c 的递推关系的右侧值
    return sum(c[i]*f.subs(n, n + i) for i in range(len(c)))


def test_rsolve_bulk():
    """Some bulk-generated tests."""
    funcs = [ n, n + 1, n**2, n**3, n**4, n + n**2, 27*n + 52*n**2 - 3*
        n**3 + 12*n**4 - 52*n**5 ]
    coeffs = [ [-2, 1], [-2, -1, 1], [-1, 1, 1, -1, 1], [-n, 1], [n**2 -
        n + 12, 1] ]
    for p in funcs:
        # compute difference
        for c in coeffs:
            q = recurrence_term(c, p)
            if p.is_polynomial(n):
                # 检查是否为多项式，若是则验证递推方程解是否为 p
                assert rsolve_poly(c, q, n) == p
            # See issue 3956:
            # 查看问题编号 3956
            if p.is_hypergeometric(n) and len(c) <= 3:
                # 如果是超几何函数且系数长度小于等于3，则验证超几何函数的递推方程解是否为 p
                assert rsolve_hyper(c, q, n).subs(zip(symbols('C:3'), [0, 0, 0])).expand() == p


def test_rsolve_0_sol_homogeneous():
    # fixed by cherry-pick from
    # https://github.com/diofant/diofant/commit/e1d2e52125199eb3df59f12e8944f8a5f24b00a5
    # 从指定的 GitHub 提交中修复
    assert rsolve_hyper([n**2 - n + 12, 1], n*(n**2 - n + 12) + n + 1, n) == n


def test_rsolve():
    f = y(n + 2) - y(n + 1) - y(n)
    h = sqrt(5)*(S.Half + S.Half*sqrt(5))**n \
        - sqrt(5)*(S.Half - S.Half*sqrt(5))**n

    assert rsolve(f, y(n)) in [
        C0*(S.Half - S.Half*sqrt(5))**n + C1*(S.Half + S.Half*sqrt(5))**n,
        C1*(S.Half - S.Half*sqrt(5))**n + C0*(S.Half + S.Half*sqrt(5))**n,
    ]

    assert rsolve(f, y(n), [0, 5]) == h
    assert rsolve(f, y(n), {0: 0, 1: 5}) == h
    assert rsolve(f, y(n), {y(0): 0, y(1): 5}) == h
    assert rsolve(y(n) - y(n - 1) - y(n - 2), y(n), [0, 5]) == h
    assert rsolve(Eq(y(n), y(n - 1) + y(n - 2)), y(n), [0, 5]) == h

    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    f = (n - 1)*y(n + 2) - (n**2 + 3*n - 2)*y(n + 1) + 2*n*(n + 1)*y(n)
    g = C1*factorial(n) + C0*2**n
    h = -3*factorial(n) + 3*2**n

    assert rsolve(f, y(n)) == g
    assert rsolve(f, y(n), []) == g
    assert rsolve(f, y(n), {}) == g

    assert rsolve(f, y(n), [0, 3]) == h
    assert rsolve(f, y(n), {0: 0, 1: 3}) == h
    assert rsolve(f, y(n), {y(0): 0, y(1): 3}) == h

    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    f = y(n) - y(n - 1) - 2

    assert rsolve(f, y(n), {y(0): 0}) == 2*n
    assert rsolve(f, y(n), {y(0): 1}) == 2*n + 1
    assert rsolve(f, y(n), {y(0): 0, y(1): 1}) is None

    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    f = 3*y(n - 1) - y(n) - 1

    assert rsolve(f, y(n), {y(0): 0}) == -3**n/2 + S.Half
    assert rsolve(f, y(n), {y(0): 1}) == 3**n/2 + S.Half
    assert rsolve(f, y(n), {y(0): 2}) == 3*3**n/2 + S.Half

    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    f = y(n) - 1/n*y(n - 1)
    assert rsolve(f, y(n)) == C0/factorial(n)
    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    f = y(n) - 1/n*y(n - 1) - 1
    assert rsolve(f, y(n)) is None

    f = 2*y(n - 1) + (1 - n)*y(n)/n
    # 断言，验证 rsolve 函数计算得出的结果是否等于 2**(n - 1)*n
    assert rsolve(f, y(n), {y(1): 1}) == 2**(n - 1)*n
    # 断言，验证 rsolve 函数计算得出的结果是否等于 2**(n - 1)*n*2
    assert rsolve(f, y(n), {y(1): 2}) == 2**(n - 1)*n*2
    # 断言，验证 rsolve 函数计算得出的结果是否等于 2**(n - 1)*n*3
    assert rsolve(f, y(n), {y(1): 3}) == 2**(n - 1)*n*3

    # 计算 Lambda 函数，并对其结果进行化简，验证其是否等于 0
    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    # 定义 f 表达式
    f = (n - 1)*(n - 2)*y(n + 2) - (n + 1)*(n + 2)*y(n)

    # 断言，验证 rsolve 函数计算得出的结果是否等于 n*(n - 1)*(n - 2)，并给定初始条件
    assert rsolve(f, y(n), {y(3): 6, y(4): 24}) == n*(n - 1)*(n - 2)
    # 断言，验证 rsolve 函数计算得出的结果是否等于 -n*(n - 1)*(n - 2)*(-1)**(n)，并给定不同的初始条件
    assert rsolve(f, y(n), {y(3): 6, y(4): -24}) == -n*(n - 1)*(n - 2)*(-1)**(n)

    # 计算 Lambda 函数，并对其结果进行化简，验证其是否等于 0
    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    # 断言，验证 rsolve 函数计算得出的结果是否等于 a**n，并给定初始条件
    assert rsolve(Eq(y(n + 1), a*y(n)), y(n), {y(1): a}).simplify() == a**n

    # 断言，验证 rsolve 函数计算得出的结果是否等于 a**(n/2 + 1) - b*(-sqrt(a))**n，并给定初始条件
    assert rsolve(y(n) - a*y(n-2),y(n), \
            {y(1): sqrt(a)*(a + b), y(2): a*(a - b)}).simplify() == \
            a**(n/2 + 1) - b*(-sqrt(a))**n

    # 定义 f 表达式
    f = (-16*n**2 + 32*n - 12)*y(n - 1) + (4*n**2 - 12*n + 9)*y(n)

    # 计算 rsolve 函数得到 y(n) 的解，并验证其是否等于预期的 sol
    yn = rsolve(f, y(n), {y(1): binomial(2*n + 1, 3)})
    sol = 2**(2*n)*n*(2*n - 1)**2*(2*n + 1)/12
    assert factor(expand(yn, func=True)) == sol

    # 计算 rsolve 函数得到 y(n) 的通解
    sol = rsolve(y(n) + a*(y(n + 1) + y(n - 1))/2, y(n))
    # 断言，验证通解是否符合预期的字符串形式
    assert str(sol) == 'C0*((-sqrt(1 - a**2) - 1)/a)**n + C1*((sqrt(1 - a**2) - 1)/a)**n'

    # 断言，验证 rsolve 函数计算得出的结果是否为 None
    assert rsolve((k + 1)*y(k), y(k)) is None
    # 断言，验证 rsolve 函数计算得出的结果是否为 None
    assert (rsolve((k + 1)*y(k) + (k + 3)*y(k + 1) + (k + 5)*y(k + 2), y(k))
            is None)

    # 断言，验证 rsolve 函数计算得出的结果是否等于 (-1)**n*C0 - 2**n/3 - 3**n/4
    assert rsolve(y(n) + y(n + 1) + 2**n + 3**n, y(n)) == (-1)**n*C0 - 2**n/3 - 3**n/4
# 测试rsolve函数在抛出异常的情况下是否正常工作
def test_rsolve_raises():
    # 定义符号函数x
    x = Function('x')
    # 断言在解决递推关系时，当给定y(n) - y(k + 1)时会引发值错误异常
    raises(ValueError, lambda: rsolve(y(n) - y(k + 1), y(n)))
    # 断言在解决递推关系时，当给定y(n) - y(n + 1)且y(n)为x(n)时会引发值错误异常
    raises(ValueError, lambda: rsolve(y(n) - y(n + 1), x(n)))
    # 断言在解决递推关系时，当给定y(n) - x(n + 1)时会引发值错误异常
    raises(ValueError, lambda: rsolve(y(n) - x(n + 1), y(n)))
    # 断言在解决递推关系时，当给定y(n) - sqrt(n)*y(n + 1)时会引发值错误异常
    raises(ValueError, lambda: rsolve(y(n) - sqrt(n)*y(n + 1), y(n)))
    # 断言在解决递推关系时，当给定y(n) - y(n + 1)且提供初始条件x(0): 0时会引发值错误异常
    raises(ValueError, lambda: rsolve(y(n) - y(n + 1), y(n), {x(0): 0}))
    # 断言在解决递推关系时，当给定y(n) + y(n + 1) + 2**n + cos(n)时会引发值错误异常
    raises(ValueError, lambda: rsolve(y(n) + y(n + 1) + 2**n + cos(n), y(n)))


# 测试问题6844的解决方案
def test_issue_6844():
    # 定义表达式f
    f = y(n + 2) - y(n + 1) + y(n)/4
    # 断言解决递推关系f对y(n)的解为2**(-n + 1)*C1*n + 2**(-n)*C0
    assert rsolve(f, y(n)) == 2**(-n + 1)*C1*n + 2**(-n)*C0
    # 断言解决递推关系f对y(n)的解为2**(1 - n)*n，并提供初始条件y(0): 0, y(1): 1
    assert rsolve(f, y(n), {y(0): 0, y(1): 1}) == 2**(1 - n)*n


# 测试问题18751的解决方案
def test_issue_18751():
    # 定义符号r和theta
    r = Symbol('r', positive=True)
    theta = Symbol('theta', real=True)
    # 定义表达式f
    f = y(n) - 2 * r * cos(theta) * y(n - 1) + r**2 * y(n - 2)
    # 断言解决递推关系f对y(n)的解为C0*(r*(cos(theta) - I*Abs(sin(theta))))**n + C1*(r*(cos(theta) + I*Abs(sin(theta))))**n
    assert rsolve(f, y(n)) == \
        C0*(r*(cos(theta) - I*Abs(sin(theta))))**n + C1*(r*(cos(theta) + I*Abs(sin(theta))))**n


# 测试命名常数的问题
def test_constant_naming():
    #issue 8697
    # 断言解决递推关系y(n+3) - y(n+2) - y(n+1) + y(n)对y(n)的解为(-1)**n*C1 + C0 + C2*n
    assert rsolve(y(n+3) - y(n+2) - y(n+1) + y(n), y(n)) == (-1)**n*C1 + C0 + C2*n
    # 断言展开解决递推关系y(n+3) + 3*y(n+2) + 3*y(n+1) + y(n)对y(n)的解为(-1)**n*C0 - (-1)**n*C1*n - (-1)**n*C2*n**2
    assert rsolve(y(n+3)+3*y(n+2)+3*y(n+1)+y(n), y(n)).expand() == (-1)**n*C0 - (-1)**n*C1*n - (-1)**n*C2*n**2
    # 断言解决递推关系y(n) - 2*y(n - 3) + 5*y(n - 2) - 4*y(n - 1)，并提供初始条件a(0): 1, a(1): 3, a(2): 8时对y(n)的解为3*2**n - n - 2
    assert rsolve(y(n) - 2*y(n - 3) + 5*y(n - 2) - 4*y(n - 1), y(n), [1, 3, 8]) == 3*2**n - n - 2

    #issue 19630
    # 断言解决递推关系y(n+3) - 3*y(n+1) + 2*y(n)，并提供初始条件y(1):0, y(2):8, y(3):-2时对y(n)的解为(-2)**n + 2*n
    assert rsolve(y(n+3) - 3*y(n+1) + 2*y(n), y(n), {y(1):0, y(2):8, y(3):-2}) == (-2)**n + 2*n


@slow
# 测试问题15751的解决方案
def test_issue_15751():
    # 定义表达式f
    f = y(n) + 21*y(n + 1) - 273*y(n + 2) - 1092*y(n + 3) + 1820*y(n + 4) + 1092*y(n + 5) - 273*y(n + 6) - 21*y(n + 7) + y(n + 8)
    # 断言解决递推关系f对y(n)的解不为空
    assert rsolve(f, y(n)) is not None


# 测试问题17990的解决方案
def test_issue_17990():
    # 定义表达式f
    f = -10*y(n) + 4*y(n + 1) + 6*y(n + 2) + 46*y(n + 3)
    # 求解递推关系f对y(n)的解
    sol = rsolve(f, y(n))
    # 期望的解表达式
    expected = C0*((86*18**(S(1)/3)/69 + (-12 + (-1 + sqrt(3)*I)*(290412 +
        3036*sqrt(9165))**(S(1)/3))*(1 - sqrt(3)*I)*(24201 + 253*sqrt(9165))**
        (S(1)/3)/276)/((1 - sqrt(3)*I)*(24201 + 253*sqrt(9165))**(S(1)/3))
        )**n + C1*((86*18**(S(1)/3)/69 + (-12 + (-1 - sqrt(3)*I)*(290412 + 3036
        *sqrt(9165))**(S(1)/3))*(1 + sqrt(3)*I)*(24201 + 253*sqrt(9165))**
        (S(1)/3)/276)/((1 + sqrt(3)*I)*(24201 + 253*sqrt(9165))**(S(1)/3))
        )**n + C2*(-43*18**(S(1)/3)/(69*(24201 + 253*sqrt(9165))**(S(1)/3)) -
        S(1)/23 + (290412 + 3036*sqrt(9165))**(S(1)/3)/138)**n
    # 断言解与期望相等
    assert sol == expected
    # 对解求特定条件下的数值，并验证精度
    e = sol.subs({C0: 1, C1: 1, C2: 1, n: 1}).evalf()
    assert abs(e + 0.130434782608696) < 1e-13


# 测试问题8697的解决方案
def test_issue_8697():
    # 定义符号函数a
    # 使用 assert 语句进行断言，验证 rsolve 函数的输出是否符合预期值
    assert rsolve(a(n) - 2*a(n - 1) - n, a(n), {a(0): 1}) == 3*2**n - n - 2
# 定义测试函数 test_diofantissue_294，用于测试解决常微分方程问题
def test_diofantissue_294():
    # 定义方程 f = y(n) - y(n - 1) - 2*y(n - 2) - 2*n
    f = y(n) - y(n - 1) - 2*y(n - 2) - 2*n
    # 断言解决方程 rsolve(f, y(n)) 的结果
    assert rsolve(f, y(n)) == (-1)**n*C0 + 2**n*C1 - n - Rational(5, 2)
    # 断言在特定初始条件下解决方程 rsolve(f, y(n), {y(0): -1, y(1): 1}) 的结果
    # issue sympy/sympy#11261
    assert rsolve(f, y(n), {y(0): -1, y(1): 1}) == (-(-1)**n/2 + 2*2**n -
                                                    n - Rational(5, 2))
    # 断言解决方程 rsolve(-2*y(n) + y(n + 1) + n - 1, y(n)) 的结果
    # issue sympy/sympy#7055
    assert rsolve(-2*y(n) + y(n + 1) + n - 1, y(n)) == 2**n*C0 + n


# 定义测试函数 test_issue_15553，用于测试解决常微分方程问题
def test_issue_15553():
    # 定义函数 f(n)
    f = Function("f")
    # 断言解决方程 rsolve(Eq(f(n), 2*f(n - 1) + n), f(n)) 的结果
    assert rsolve(Eq(f(n), 2*f(n - 1) + n), f(n)) == 2**n*C0 - n - 2
    # 断言解决方程 rsolve(Eq(f(n + 1), 2*f(n) + n**2 + 1), f(n)) 的结果
    assert rsolve(Eq(f(n + 1), 2*f(n) + n**2 + 1), f(n)) == 2**n*C0 - n**2 - 2*n - 4
    # 断言在特定初始条件下解决方程 rsolve(Eq(f(n + 1), 2*f(n) + n**2 + 1), f(n), {f(1): 0}) 的结果
    assert rsolve(Eq(f(n + 1), 2*f(n) + n**2 + 1), f(n), {f(1): 0}) == 7*2**n/2 - n**2 - 2*n - 4
    # 断言解决方程 rsolve(Eq(f(n), 2*f(n - 1) + 3*n**2), f(n)) 的结果
    assert rsolve(Eq(f(n), 2*f(n - 1) + 3*n**2), f(n)) == 2**n*C0 - 3*n**2 - 12*n - 18
    # 断言解决方程 rsolve(Eq(f(n), 2*f(n - 1) + n**2), f(n)) 的结果
    assert rsolve(Eq(f(n), 2*f(n - 1) + n**2), f(n)) == 2**n*C0 - n**2 - 4*n - 6
    # 断言在特定初始条件下解决方程 rsolve(Eq(f(n), 2*f(n - 1) + n), f(n), {f(0): 1}) 的结果
    assert rsolve(Eq(f(n), 2*f(n - 1) + n), f(n), {f(0): 1}) == 3*2**n - n - 2
```