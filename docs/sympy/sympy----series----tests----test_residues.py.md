# `D:\src\scipysrc\sympy\sympy\series\tests\test_residues.py`

```
# 导入 sympy 库中特定模块和函数

from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cot, sin, tan)
from sympy.series.residues import residue
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, z, a, s, k

# 定义测试函数 test_basic1，用于测试 residue 函数在 x=0 时的值
def test_basic1():
    assert residue(1/x, x, 0) == 1  # 计算 1/x 在 x=0 处的留数应为 1
    assert residue(-2/x, x, 0) == -2  # 计算 -2/x 在 x=0 处的留数应为 -2
    assert residue(81/x, x, 0) == 81  # 计算 81/x 在 x=0 处的留数应为 81
    assert residue(1/x**2, x, 0) == 0  # 计算 1/x**2 在 x=0 处的留数应为 0
    assert residue(0, x, 0) == 0  # 计算常数函数 0 在 x=0 处的留数应为 0
    assert residue(5, x, 0) == 0  # 计算常数函数 5 在 x=0 处的留数应为 0
    assert residue(x, x, 0) == 0  # 计算一次函数 x 在 x=0 处的留数应为 0
    assert residue(x**2, x, 0) == 0  # 计算二次函数 x**2 在 x=0 处的留数应为 0

# 定义测试函数 test_basic2，用于测试 residue 函数在不同 x 值时的行为
def test_basic2():
    assert residue(1/x, x, 1) == 0  # 计算 1/x 在 x=1 处的留数应为 0
    assert residue(-2/x, x, 1) == 0  # 计算 -2/x 在 x=1 处的留数应为 0
    assert residue(81/x, x, -1) == 0  # 计算 81/x 在 x=-1 处的留数应为 0
    assert residue(1/x**2, x, 1) == 0  # 计算 1/x**2 在 x=1 处的留数应为 0
    assert residue(0, x, 1) == 0  # 计算常数函数 0 在 x=1 处的留数应为 0
    assert residue(5, x, 1) == 0  # 计算常数函数 5 在 x=1 处的留数应为 0
    assert residue(x, x, 1) == 0  # 计算一次函数 x 在 x=1 处的留数应为 0
    assert residue(x**2, x, 5) == 0  # 计算二次函数 x**2 在 x=5 处的留数应为 0

# 定义测试函数 test_f，用于测试 residue 函数在 f(x)/x**5 形式下的留数计算
def test_f():
    f = Function("f")
    assert residue(f(x)/x**5, x, 0) == f(x).diff(x, 4).subs(x, 0)/24  # 计算 f(x)/x**5 在 x=0 处的留数应为 f(x) 在 x=0 处四阶导数除以 24

# 定义测试函数 test_functions，用于测试余函数的留数计算
def test_functions():
    assert residue(1/sin(x), x, 0) == 1  # 计算 1/sin(x) 在 x=0 处的留数应为 1
    assert residue(2/sin(x), x, 0) == 2  # 计算 2/sin(x) 在 x=0 处的留数应为 2
    assert residue(1/sin(x)**2, x, 0) == 0  # 计算 1/sin(x)**2 在 x=0 处的留数应为 0
    assert residue(1/sin(x)**5, x, 0) == Rational(3, 8)  # 计算 1/sin(x)**5 在 x=0 处的留数应为 3/8

# 定义测试函数 test_expressions，用于测试复杂表达式的留数计算
def test_expressions():
    assert residue(1/(x + 1), x, 0) == 0  # 计算 1/(x + 1) 在 x=0 处的留数应为 0
    assert residue(1/(x + 1), x, -1) == 1  # 计算 1/(x + 1) 在 x=-1 处的留数应为 1
    assert residue(1/(x**2 + 1), x, -1) == 0  # 计算 1/(x**2 + 1) 在 x=-1 处的留数应为 0
    assert residue(1/(x**2 + 1), x, I) == -I/2  # 计算 1/(x**2 + 1) 在 x=I 处的留数应为 -I/2
    assert residue(1/(x**2 + 1), x, -I) == I/2  # 计算 1/(x**2 + 1) 在 x=-I 处的留数应为 I/2
    assert residue(1/(x**4 + 1), x, 0) == 0  # 计算 1/(x**4 + 1) 在 x=0 处的留数应为 0
    assert residue(1/(x**4 + 1), x, exp(I*pi/4)).equals(-(Rational(1, 4) + I/4)/sqrt(2))  # 计算 1/(x**4 + 1) 在 x=exp(I*pi/4) 处的留数应为 -(1/4 + I/4)/sqrt(2)
    assert residue(1/(x**2 + a**2)**2, x, a*I) == -I/4/a**3  # 计算 1/(x**2 + a**2)**2 在 x=a*I 处的留数应为 -I/(4*a**3)

# 定义测试函数 test_issue_5654，用于测试已知的问题 5654 的情况
def test_issue_5654():
    assert residue(1/(x**2 + a**2)**2, x, a*I) == -I/(4*a**3)  # 计算 1/(x**2 + a**2)**2 在 x=a*I 处的留数应为 -I/(4*a**3)
    assert residue(1/s*1/(z - exp(s)), s, 0) == 1/(z - 1)  # 计算 1/s*1/(z - exp(s)) 在 s=0 处的留数应为 1/(z - 1)
    assert residue((1 + k)/s*1/(z - exp(s)), s, 0) == k/(z - 1) + 1/(z - 1)  # 计算 (1 + k)/s*1/(z - exp(s)) 在 s=0 处的留数应为 k/(z - 1) + 1/(z - 1)

# 定义测试函数 test_issue_6499，用于测试已知的问题 6499 的情况
def test_issue_6499():
    assert residue(1/(exp(z) - 1), z, 0) == 1  # 计算 1/(exp(z) - 1) 在 z=0 处的留数应为 1

# 定义测试函数 test_issue_14037，用于测试已知的问题 14037 的情况
def test_issue_14037():
    assert residue(sin(x**50)/x**51, x,
    # 计算在复平面上 x = 3/2 - sqrt(3)*i/2 处的函数 cot(pi*x)/((x - 1)*(x - 2) + 1) 的残余
    a = residue(cot(pi*x)/((x - 1)*(x - 2) + 1), x, S(3)/2 - sqrt(3)*I/2)
    
    # 计算在复平面上 x = 3/2 - sqrt(3)*i/2 处的函数 cot(pi*x)/(x**2 - 3*x + 3) 的残余
    b = residue(cot(pi*x)/(x**2 - 3*x + 3), x, S(3)/2 - sqrt(3)*I/2)
    
    # 断言：验证前面计算的 a 是否等于 r
    assert a == r
    
    # 断言：验证 b - a 是否化简为 0
    assert (b - a).cancel() == 0
```