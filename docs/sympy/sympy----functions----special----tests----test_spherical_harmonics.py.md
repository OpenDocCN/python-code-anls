# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_spherical_harmonics.py`

```
# 导入必要的函数和符号
from sympy.core.function import diff
from sympy.core.numbers import (I, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, cot, sin)
from sympy.functions.special.spherical_harmonics import Ynm, Znm, Ynm_c

# 定义测试函数 test_Ynm，用于测试球谐函数 Ynm 的性质
def test_Ynm():
    # 设置球坐标的角度符号 theta 和 phi，要求为实数
    th, ph = Symbol("theta", real=True), Symbol("phi", real=True)
    # 导入 sympy.abc 中的符号 n 和 m

    # 检查 Ynm(0, 0, th, ph) 的展开结果是否等于 1/(2*sqrt(pi))
    assert Ynm(0, 0, th, ph).expand(func=True) == 1/(2*sqrt(pi))

    # 检查 Ynm(1, -1, th, ph) 是否等于 -exp(-2*I*ph)*Ynm(1, 1, th, ph)
    assert Ynm(1, -1, th, ph) == -exp(-2*I*ph)*Ynm(1, 1, th, ph)

    # 检查 Ynm(1, -1, th, ph) 的展开结果是否等于 sqrt(6)*sin(th)*exp(-I*ph)/(4*sqrt(pi))
    assert Ynm(1, -1, th, ph).expand(func=True) == sqrt(6)*sin(th)*exp(-I*ph)/(4*sqrt(pi))

    # 检查 Ynm(1, 0, th, ph) 的展开结果是否等于 sqrt(3)*cos(th)/(2*sqrt(pi))
    assert Ynm(1, 0, th, ph).expand(func=True) == sqrt(3)*cos(th)/(2*sqrt(pi))

    # 检查 Ynm(1, 1, th, ph) 的展开结果是否等于 -sqrt(6)*sin(th)*exp(I*ph)/(4*sqrt(pi))
    assert Ynm(1, 1, th, ph).expand(func=True) == -sqrt(6)*sin(th)*exp(I*ph)/(4*sqrt(pi))

    # 检查 Ynm(2, 0, th, ph) 的展开结果是否等于 3*sqrt(5)*cos(th)**2/(4*sqrt(pi)) - sqrt(5)/(4*sqrt(pi))
    assert Ynm(2, 0, th, ph).expand(func=True) == 3*sqrt(5)*cos(th)**2/(4*sqrt(pi)) - sqrt(5)/(4*sqrt(pi))

    # 检查 Ynm(2, 1, th, ph) 的展开结果是否等于 -sqrt(30)*sin(th)*exp(I*ph)*cos(th)/(4*sqrt(pi))
    assert Ynm(2, 1, th, ph).expand(func=True) == -sqrt(30)*sin(th)*exp(I*ph)*cos(th)/(4*sqrt(pi))

    # 检查 Ynm(2, -2, th, ph) 的展开结果是否等于 (-sqrt(30)*exp(-2*I*ph)*cos(th)**2/(8*sqrt(pi))
    #                                    + sqrt(30)*exp(-2*I*ph)/(8*sqrt(pi)))
    assert Ynm(2, -2, th, ph).expand(func=True) == (-sqrt(30)*exp(-2*I*ph)*cos(th)**2/(8*sqrt(pi))
                                                    + sqrt(30)*exp(-2*I*ph)/(8*sqrt(pi)))

    # 检查 Ynm(2, 2, th, ph) 的展开结果是否等于 (-sqrt(30)*exp(2*I*ph)*cos(th)**2/(8*sqrt(pi))
    #                                    + sqrt(30)*exp(2*I*ph)/(8*sqrt(pi)))
    assert Ynm(2, 2, th, ph).expand(func=True) == (-sqrt(30)*exp(2*I*ph)*cos(th)**2/(8*sqrt(pi))
                                                   + sqrt(30)*exp(2*I*ph)/(8*sqrt(pi)))

    # 检查 Ynm(n, m, th, ph) 对 theta 的偏导数是否符合预期
    assert diff(Ynm(n, m, th, ph), th) == (m*cot(th)*Ynm(n, m, th, ph)
                                           + sqrt((-m + n)*(m + n + 1))*exp(-I*ph)*Ynm(n, m + 1, th, ph))

    # 检查 Ynm(n, m, th, ph) 对 phi 的偏导数是否符合预期
    assert diff(Ynm(n, m, th, ph), ph) == I*m*Ynm(n, m, th, ph)

    # 检查 Ynm(n, m, th, ph) 的共轭是否符合预期
    assert conjugate(Ynm(n, m, th, ph)) == (-1)**(2*m)*exp(-2*I*m*ph)*Ynm(n, m, th, ph)

    # 检查 Ynm(n, m, -th, ph) 是否等于 Ynm(n, m, th, ph)
    assert Ynm(n, m, -th, ph) == Ynm(n, m, th, ph)

    # 检查 Ynm(n, m, th, -ph) 是否等于 exp(-2*I*m*ph)*Ynm(n, m, th, ph)
    assert Ynm(n, m, th, -ph) == exp(-2*I*m*ph)*Ynm(n, m, th, ph)

    # 检查 Ynm(n, -m, th, ph) 是否等于 (-1)**m*exp(-2*I*m*ph)*Ynm(n, m, th, ph)
    assert Ynm(n, -m, th, ph) == (-1)**m*exp(-2*I*m*ph)*Ynm(n, m, th, ph)


# 定义测试函数 test_Ynm_c，用于测试归一化球谐函数 Ynm_c 的性质
def test_Ynm_c():
    # 设置球坐标的角度符号 theta 和 phi，要求为实数
    th, ph = Symbol("theta", real=True), Symbol("phi", real=True)
    # 导入 sympy.abc 中的符号 n 和 m

    # 检查 Ynm_c(n, m, th, ph) 是否等于 (-1)**(2*m)*exp(-2*I*m*ph)*Ynm(n, m, th, ph)
    assert Ynm_c(n, m, th, ph) == (-1)**(2*m)*exp(-2*I*m*ph)*Ynm(n, m, th, ph)


# 定义测试函数 test_Znm，用于测试固体球谐函数 Znm 的性质
def test_Znm():
    # 设置球坐标的角度符号 theta 和 phi，要求为实数
    th, ph = Symbol("theta", real=True), Symbol("phi", real=True)

    # 检查 Znm(0, 0, th, ph) 是否等于 Ynm(0, 0, th, ph)
    assert Znm(0, 0, th, ph) == Ynm(0, 0, th, ph)

    # 检查 Znm(1, -1, th, ph) 是否等于 (-sqrt(2)*I*(Ynm(1, 1, th, ph) - exp(-2*I*ph)*Ynm(1, 1, th, ph))/2)
    assert Znm(1, -1, th, ph) == (-sqrt(2)*I*(Ynm(1, 1, th, ph) - exp(-2*I*ph)*Ynm(1, 1, th, ph))/2)

    # 检查 Znm(1, 0, th, ph) 是否等于 Ynm(1, 0, th, ph)
    assert Znm(1, 0, th, ph) == Ynm(1, 0, th, ph)

    # 检查 Znm(1, 1, th, ph) 是否等于 (sqrt(2)*(Ynm(1, 1, th, ph) + exp(-2*I*ph)*Ynm(1, 1, th, ph))/2)
    assert Znm(1, 1, th, ph) == (sqrt(2)*(Ynm(1, 1, th, ph) + exp(-2*I*ph)*Ynm(1, 1, th, ph))/2)

    # 检查 Znm(0, 0, th, ph)
    # 断言：验证球谐函数 Znm 的展开式是否正确
    assert Znm(1, -1, th, ph).expand(func=True) == (sqrt(3)*I*sin(th)*exp(I*ph)/(4*sqrt(pi))
                                                    - sqrt(3)*I*sin(th)*exp(-I*ph)/(4*sqrt(pi)))
    # 断言：验证球谐函数 Znm 的展开式是否正确
    assert Znm(1, 0, th, ph).expand(func=True) == sqrt(3)*cos(th)/(2*sqrt(pi))
    # 断言：验证球谐函数 Znm 的展开式是否正确
    assert Znm(1, 1, th, ph).expand(func=True) == (-sqrt(3)*sin(th)*exp(I*ph)/(4*sqrt(pi))
                                                   - sqrt(3)*sin(th)*exp(-I*ph)/(4*sqrt(pi)))
    # 断言：验证球谐函数 Znm 的展开式是否正确
    assert Znm(2, -1, th, ph).expand(func=True) == (sqrt(15)*I*sin(th)*exp(I*ph)*cos(th)/(4*sqrt(pi))
                                                    - sqrt(15)*I*sin(th)*exp(-I*ph)*cos(th)/(4*sqrt(pi)))
    # 断言：验证球谐函数 Znm 的展开式是否正确
    assert Znm(2, 0, th, ph).expand(func=True) == 3*sqrt(5)*cos(th)**2/(4*sqrt(pi)) - sqrt(5)/(4*sqrt(pi))
    # 断言：验证球谐函数 Znm 的展开式是否正确
    assert Znm(2, 1, th, ph).expand(func=True) == (-sqrt(15)*sin(th)*exp(I*ph)*cos(th)/(4*sqrt(pi))
                                                   - sqrt(15)*sin(th)*exp(-I*ph)*cos(th)/(4*sqrt(pi)))
```