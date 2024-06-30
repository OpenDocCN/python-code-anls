# `D:\src\scipysrc\sympy\sympy\solvers\tests\test_numeric.py`

```
# 导入所需的模块和函数
from sympy.core.function import nfloat
from sympy.core.numbers import (Float, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from mpmath import mnorm, mpf
from sympy.solvers import nsolve
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises, XFAIL
from sympy.utilities.decorator import conserve_mpmath_dps

# 标记为预期失败的测试函数
@XFAIL
def test_nsolve_fail():
    x = symbols('x')
    # 有时候使用分子（问题编号 4829）更好，
    # 但有时候不是（问题编号 11768），所以这由用户决定
    ans = nsolve(x**2/(1 - x)/(1 - 2*x)**2 - 100, x, 0)
    assert ans > 0.46 and ans < 0.47

# 测试 nsolve 函数的分母情况
def test_nsolve_denominator():
    x = symbols('x')
    # 测试 nsolve 是否使用完整表达式（包括分子和分母）
    ans = nsolve((x**2 + 3*x + 2)/(x + 2), -2.1)
    # -2 被除去了，所以确保我们找不到它
    assert ans == -1.0

# 测试 nsolve 函数的基本功能
def test_nsolve():
    # 一维情况
    x = Symbol('x')
    assert nsolve(sin(x), 2) - pi.evalf() < 1e-15
    assert nsolve(Eq(2*x, 2), x, -10) == nsolve(2*x - 2, -10)
    # 测试输入数量检查
    raises(TypeError, lambda: nsolve(Eq(2*x, 2)))
    raises(TypeError, lambda: nsolve(Eq(2*x, 2), x, 1, 2))
    
    # 多维情况
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    f1 = 3 * x1**2 - 2 * x2**2 - 1
    f2 = x1**2 - 2 * x1 + x2**2 + 2 * x2 - 8
    f = Matrix((f1, f2)).T
    F = lambdify((x1, x2), f.T, modules='mpmath')
    for x0 in [(-1, 1), (1, -2), (4, 4), (-4, -4)]:
        x = nsolve(f, (x1, x2), x0, tol=1.e-8)
        assert mnorm(F(*x), 1) <= 1.e-10
    
    # 700年前由中国数学家朱世杰解决的非线性系统测试
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    f1 = -x + 2*y
    f2 = (x**2 + x*(y**2 - 2) - 4*y) / (x + 4)
    f3 = sqrt(x**2 + y**2)*z
    f = Matrix((f1, f2, f3)).T
    F = lambdify((x, y, z), f.T, modules='mpmath')

    def getroot(x0):
        root = nsolve(f, (x, y, z), x0)
        assert mnorm(F(*root), 1) <= 1.e-8
        return root
    assert list(map(round, getroot((1, 1, 1)))) == [2, 1, 0]
    assert nsolve([Eq(
        f1, 0), Eq(f2, 0), Eq(f3, 0)], [x, y, z], (1, 1, 1))  # just see that it works
    
    a = Symbol('a')
    assert abs(nsolve(1/(0.001 + a)**3 - 6/(0.9 - a)**3, a, 0.3) -
               mpf('0.31883011387318591')) < 1e-15

# 测试问题 #6408
def test_issue_6408():
    x = Symbol('x')
    assert nsolve(Piecewise((x, x < 1), (x**2, True)), x, 2) == 0.0

# 测试问题 #6408 的积分版本
def test_issue_6408_integral():
    x, y = symbols('x y')
    assert nsolve(Integral(x*y, (x, 0, 5)), y, 2) == 0.0

# 应用 mpmath 的精度保护装饰器
@conserve_mpmath_dps
# 测试增加的 double precision floating point (dps) 精度
def test_increased_dps():
    # 导入 mpmath 库
    import mpmath
    # 设置 mpmath 库的精度为 128 位
    mpmath.mp.dps = 128
    # 创建符号变量 x
    x = Symbol('x')
    # 定义方程 e1 = x^2 - pi
    e1 = x**2 - pi
    # 使用 nsolve 函数求解方程 e1 关于 x 在 x = 3.0 附近的数值解
    q = nsolve(e1, x, 3.0)

    # 断言语句，验证计算出的 sqrt(pi) 的数值近似与 q 的差异小于 1e-128
    assert abs(sqrt(pi).evalf(128) - q) < 1e-128

# 测试 nsolve 函数的数值解精度
def test_nsolve_precision():
    # 创建符号变量 x, y
    x, y = symbols('x y')
    # 使用 nsolve 函数求解 x^2 - pi = 0 关于 x 在 x = 3 处的数值解，精度为 128 位
    sol = nsolve(x**2 - pi, x, 3, prec=128)
    # 断言语句，验证计算出的 sqrt(pi) 的数值近似与 sol 的差异小于 1e-128
    assert abs(sqrt(pi).evalf(128) - sol) < 1e-128
    # 断言语句，验证 sol 的类型为 Float
    assert isinstance(sol, Float)

    # 使用 nsolve 函数求解方程组 [y^2 - x = 0, x^2 - pi = 0] 关于 (x, y) 在 (3, 3) 处的数值解，精度为 128 位
    sols = nsolve((y**2 - x, x**2 - pi), (x, y), (3, 3), prec=128)
    # 断言语句，验证 sols 的类型为 Matrix
    assert isinstance(sols, Matrix)
    # 断言语句，验证 sols 的形状为 (2, 1)
    assert sols.shape == (2, 1)
    # 断言语句，验证计算出的 sqrt(pi) 的数值近似与 sols[0] 的差异小于 1e-128
    assert abs(sqrt(pi).evalf(128) - sols[0]) < 1e-128
    # 断言语句，验证计算出的 sqrt(sqrt(pi)) 的数值近似与 sols[1] 的差异小于 1e-128
    assert abs(sqrt(sqrt(pi)).evalf(128) - sols[1]) < 1e-128
    # 断言语句，验证 sols 中所有元素的类型为 Float
    assert all(isinstance(i, Float) for i in sols)

# 测试 nsolve 函数处理复数解
def test_nsolve_complex():
    # 创建符号变量 x, y
    x, y = symbols('x y')

    # 断言语句，验证 nsolve 函数对于 x^2 + 2 = 0 在 x = 1j 处的解为 sqrt(2)*I
    assert nsolve(x**2 + 2, 1j) == sqrt(2.)*I
    # 断言语句，验证 nsolve 函数对于 x^2 + 2 = 0 在 x = I 处的解为 sqrt(2)*I
    assert nsolve(x**2 + 2, I) == sqrt(2.)*I

    # 断言语句，验证 nsolve 函数对于方程组 [x^2 + 2 = 0, y^2 + 2 = 0] 在 (x, y) = (I, I) 处的解为 [sqrt(2)*I, sqrt(2)*I]
    assert nsolve([x**2 + 2, y**2 + 2], [x, y], [I, I]) == Matrix([sqrt(2.)*I, sqrt(2.)*I])
    # 断言语句，验证 nsolve 函数对于方程组 [x^2 + 2 = 0, y^2 + 2 = 0] 在 (x, y) = (I, I) 处的解为 [sqrt(2)*I, sqrt(2)*I]
    assert nsolve([x**2 + 2, y**2 + 2], [x, y], [I, I]) == Matrix([sqrt(2.)*I, sqrt(2.)*I])

# 测试 nsolve 函数的 dict 参数
def test_nsolve_dict_kwarg():
    # 创建符号变量 x, y
    x, y = symbols('x y')

    # 单变量情况下，验证 nsolve 函数对于 x^2 - 2 = 0 在 x = 1 处的解为 [{x: sqrt(2.)}]
    assert nsolve(x**2 - 2, 1, dict=True) == [{x: sqrt(2.)}]
    # 单变量复数解情况下，验证 nsolve 函数对于 x^2 + 2 = 0 在 x = I 处的解为 [{x: sqrt(2.)*I}]
    assert nsolve(x**2 + 2, I, dict=True) == [{x: sqrt(2.)*I}]
    # 双变量情况下，验证 nsolve 函数对于方程组 [x^2 + y^2 - 5 = 0, x^2 - y^2 + 1 = 0] 在 (x, y) = (1, 1) 处的解为 [{x: sqrt(2.), y: sqrt(3.)}]
    assert nsolve([x**2 + y**2 - 5, x**2 - y**2 + 1], [x, y], [1, 1], dict=True) == [{x: sqrt(2.), y: sqrt(3.)}]

# 测试 nsolve 函数处理有理数解
def test_nsolve_rational():
    # 创建符号变量 x
    x = symbols('x')
    # 验证 nsolve 函数对于 x - 1/3 = 0 在 x = 0 处的解为 Rational(1, 3).evalf(100)
    assert nsolve(x - Rational(1, 3), 0, prec=100) == Rational(1, 3).evalf(100)

# 测试问题 14950
def test_issue_14950():
    # 创建符号向量 x
    x = Matrix(symbols('t s'))
    # 创建初始向量 x0
    x0 = Matrix([17, 23])
    # 定义方程 eqn = x + x0
    eqn = x + x0
    # 断言语句，验证 nsolve 函数对于方程 eqn = 0 在 x = x0 处的解为 -x0
    assert nsolve(eqn, x, x0) == nfloat(-x0)
    # 断言语句，验证 nsolve 函数对于方程 eqn^T = 0 在 x^T = x0^T 处的解为 -x0
    assert nsolve(eqn.T, x.T, x0.T) == nfloat(-x0)
```