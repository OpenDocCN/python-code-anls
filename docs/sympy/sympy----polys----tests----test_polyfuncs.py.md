# `D:\src\scipysrc\sympy\sympy\polys\tests\test_polyfuncs.py`

```
"""Tests for high-level polynomials manipulation functions. """

# 从 sympy.polys.polyfuncs 模块中导入所需函数
from sympy.polys.polyfuncs import (
    symmetrize, horner, interpolate, rational_interpolate, viete,
)

# 导入多变量多项式错误类
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
)

# 导入 sympy.core.singleton 中的 S 对象，用于表示符号常量
from sympy.core.singleton import S

# 导入 sympy.core.symbol 中的 symbols 函数，用于定义符号变量
from sympy.core.symbol import symbols

# 导入 sympy.testing.pytest 中的 raises 函数，用于测试异常情况
from sympy.testing.pytest import raises

# 从 sympy.abc 中导入常用的符号变量
from sympy.abc import a, b, c, d, e, x, y, z


def test_symmetrize():
    # 检查对于常数 0 的符号对称化
    assert symmetrize(0, x, y, z) == (0, 0)
    # 检查对于常数 1 的符号对称化
    assert symmetrize(1, x, y, z) == (1, 0)

    # 定义一些常用的符号表达式
    s1 = x + y + z
    s2 = x*y + x*z + y*z

    # 检查对于常数 1 的符号对称化，不指定形式参数
    assert symmetrize(1) == (1, 0)
    # 检查对于常数 1 的符号对称化，指定形式参数为 True
    assert symmetrize(1, formal=True) == (1, 0, [])

    # 检查对于单个符号变量 x 的符号对称化
    assert symmetrize(x) == (x, 0)
    # 检查对于 x + 1 的符号对称化
    assert symmetrize(x + 1) == (x + 1, 0)

    # 检查对于 x 和 y 的符号对称化
    assert symmetrize(x, x, y) == (x + y, -y)
    # 检查对于 x + 1、x 和 y 的符号对称化
    assert symmetrize(x + 1, x, y) == (x + y + 1, -y)

    # 检查对于 x、x、y 和 z 的符号对称化
    assert symmetrize(x, x, y, z) == (s1, -y - z)
    # 检查对于 x + 1、x、y 和 z 的符号对称化
    assert symmetrize(x + 1, x, y, z) == (s1 + 1, -y - z)

    # 检查对于 x**2、x、y 和 z 的符号对称化
    assert symmetrize(x**2, x, y, z) == (s1**2 - 2*s2, -y**2 - z**2)

    # 检查对于 x**2 + y**2 的符号对称化
    assert symmetrize(x**2 + y**2) == (-2*x*y + (x + y)**2, 0)
    # 检查对于 x**2 - y**2 的符号对称化
    assert symmetrize(x**2 - y**2) == (-2*x*y + (x + y)**2, -2*y**2)

    # 检查对于 x**3 + y**2 + a*x**2 + b*y**3 的符号对称化，指定 x 和 y 为符号变量
    assert symmetrize(x**3 + y**2 + a*x**2 + b*y**3, x, y) == \
        (-3*x*y*(x + y) - 2*a*x*y + a*(x + y)**2 + (x + y)**3,
         y**2*(1 - a) + y**3*(b - 1))

    # 定义符号列表 U，并进行对称化测试
    U = [u0, u1, u2] = symbols('u:3')
    assert symmetrize(x + 1, x, y, z, formal=True, symbols=U) == \
        (u0 + 1, -y - z, [(u0, x + y + z), (u1, x*y + x*z + y*z), (u2, x*y*z)])

    # 检查对于列表 [1, 2, 3] 的符号对称化
    assert symmetrize([1, 2, 3]) == [(1, 0), (2, 0), (3, 0)]
    # 检查对于列表 [1, 2, 3] 的符号对称化，指定形式参数为 True
    assert symmetrize([1, 2, 3], formal=True) == ([(1, 0), (2, 0), (3, 0)], [])

    # 检查对于列表 [x + y, x - y] 的符号对称化
    assert symmetrize([x + y, x - y]) == [(x + y, 0), (x + y, -2*y)]


def test_horner():
    # 检查对于常数 0 的霍纳方案展开
    assert horner(0) == 0
    # 检查对于常数 1 的霍纳方案展开
    assert horner(1) == 1
    # 检查对于单个符号变量 x 的霍纳方案展开
    assert horner(x) == x

    # 检查对于 x + 1 的霍纳方案展开
    assert horner(x + 1) == x + 1
    # 检查对于 x**2 + 1 的霍纳方案展开
    assert horner(x**2 + 1) == x**2 + 1
    # 检查对于 x**2 + x 的霍纳方案展开
    assert horner(x**2 + x) == (x + 1)*x
    # 检查对于 x**2 + x + 1 的霍纳方案展开
    assert horner(x**2 + x + 1) == (x + 1)*x + 1

    # 检查对于 9*x**4 + 8*x**3 + 7*x**2 + 6*x + 5 的霍纳方案展开
    assert horner(
        9*x**4 + 8*x**3 + 7*x**2 + 6*x + 5) == (((9*x + 8)*x + 7)*x + 6)*x + 5
    # 检查对于 a*x**4 + b*x**3 + c*x**2 + d*x + e 的霍纳方案展开
    assert horner(
        a*x**4 + b*x**3 + c*x**2 + d*x + e) == (((a*x + b)*x + c)*x + d)*x + e

    # 检查对于 4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y 的霍纳方案展开，指定 wrt=x
    assert horner(4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y, wrt=x) == ((
        4*y + 2)*x*y + (2*y + 1)*y)*x
    # 检查对于 4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y 的霍纳方案展开，指定 wrt=y
    assert horner(4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y, wrt=y) == ((
        4*x + 2)*y*x + (2*x + 1)*x)*y


def test_interpolate():
    # 检查对于列表 [1, 4, 9, 16] 的插值，结果为 x**2
    assert interpolate([1, 4, 9, 16], x) == x**2
    # 检查对于列表 [1, 4, 9,
    # 断言：对于给定的插值数据和目标值，验证插值函数的返回结果是否符合预期
    assert interpolate((9, 4, 9), 3) == 9
    
    # 断言：对于给定的插值数据和目标值，验证插值函数的返回结果是否为 SymPy 中的常量 S.One
    assert interpolate((1, 9, 16), 1) is S.One
    
    # 断言：对于给定的插值数据和目标值，验证插值函数的返回结果是否为 SymPy 中的常量 S.One
    assert interpolate(((x, 1), (2, 3)), x) is S.One
    
    # 断言：对于给定的插值数据和目标值，验证插值函数的返回结果是否为 SymPy 中的常量 S.One
    assert interpolate({x: 1, 2: 3}, x) is S.One
    
    # 断言：对于给定的插值数据和目标值，验证插值函数的返回结果是否符合预期
    assert interpolate(((2, x), (1, 3)), x) == x**2 - 4*x + 6
# 定义一个函数用于测试有理插值算法
def test_rational_interpolate():
    # 定义符号变量 x, y
    x, y = symbols('x,y')
    # 设置 x 轴数据
    xdata = [1, 2, 3, 4, 5, 6]
    # 设置 y1 轴数据
    ydata1 = [120, 150, 200, 255, 312, 370]
    # 设置 y2 轴数据
    ydata2 = [-210, -35, 105, 231, 350, 465]
    # 第一个断言：验证有理插值算法在 (xdata, ydata1) 数据上插值结果是否为 (60*x**2 + 60)/x
    assert rational_interpolate(list(zip(xdata, ydata1)), 2) == (
      (60*x**2 + 60)/x )
    # 第二个断言：验证有理插值算法在 (xdata, ydata1) 数据上插值结果是否为 (60*x**2 + 60)/x
    assert rational_interpolate(list(zip(xdata, ydata1)), 3) == (
      (60*x**2 + 60)/x )
    # 第三个断言：验证有理插值算法在 (xdata, ydata2) 数据上插值结果是否为 (105*y**2 - 525)/(y + 1)
    assert rational_interpolate(list(zip(xdata, ydata2)), 2, X=y) == (
      (105*y**2 - 525)/(y + 1) )
    # 重新设置 x 轴数据
    xdata = list(range(1,11))
    # 设置 y 轴数据
    ydata = [-1923885361858460, -5212158811973685, -9838050145867125,
      -15662936261217245, -22469424125057910, -30073793365223685,
      -38332297297028735, -47132954289530109, -56387719094026320,
      -66026548943876885]
    # 第四个断言：验证有理插值算法在 (xdata, ydata) 数据上插值结果是否为 (-12986226192544605*x**4 +
    # 8657484128363070*x**3 - 30301194449270745*x**2 + 4328742064181535*x
    # - 4328742064181535)/(x**3 + 9*x**2 - 3*x + 11)
    assert rational_interpolate(list(zip(xdata, ydata)), 5) == (
      (-12986226192544605*x**4 +
      8657484128363070*x**3 - 30301194449270745*x**2 + 4328742064181535*x
      - 4328742064181535)/(x**3 + 9*x**2 - 3*x + 11))


# 定义一个函数用于测试维特公式
def test_viete():
    # 定义符号变量 r1, r2
    r1, r2 = symbols('r1, r2')

    # 第一个断言：验证维特公式在给定多项式 a*x**2 + b*x + c 和根 [r1, r2] 下的输出是否为 [(r1 + r2, -b/a), (r1*r2, c/a)]
    assert viete(
        a*x**2 + b*x + c, [r1, r2], x) == [(r1 + r2, -b/a), (r1*r2, c/a)]

    # 第二个断言：验证当根列表为空时，维特公式是否会引发 ValueError 异常
    raises(ValueError, lambda: viete(1, [], x))
    # 第三个断言：验证当给定的多项式不包含 x 的项时，维特公式是否会引发 ValueError 异常
    raises(ValueError, lambda: viete(x**2 + 1, [r1]))

    # 第四个断言：验证当给定的多项式为多变量多项式时，维特公式是否会引发 MultivariatePolynomialError 异常
    raises(MultivariatePolynomialError, lambda: viete(x + y, [r1]))
```