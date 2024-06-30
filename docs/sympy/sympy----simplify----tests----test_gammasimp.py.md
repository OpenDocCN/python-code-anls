# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_gammasimp.py`

```
# 从 sympy.core.function 模块导入 Function 类
from sympy.core.function import Function
# 从 sympy.core.numbers 模块导入 Rational 和 pi
from sympy.core.numbers import (Rational, pi)
# 从 sympy.core.singleton 模块导入 S
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 symbols 函数
from sympy.core.symbol import symbols
# 从 sympy.functions.combinatorial.factorials 模块导入 rf, binomial, factorial 函数
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)
# 从 sympy.functions.elementary.exponential 模块导入 exp 函数
from sympy.functions.elementary.exponential import exp
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt 函数
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.functions.elementary.piecewise 模块导入 Piecewise 函数
from sympy.functions.elementary.piecewise import Piecewise
# 从 sympy.functions.elementary.trigonometric 模块导入 cos, sin 函数
from sympy.functions.elementary.trigonometric import (cos, sin)
# 从 sympy.functions.special.gamma_functions 模块导入 gamma 函数
from sympy.functions.special.gamma_functions import gamma
# 从 sympy.simplify.gammasimp 模块导入 gammasimp 函数
from sympy.simplify.gammasimp import gammasimp
# 从 sympy.simplify.powsimp 模块导入 powsimp 函数
from sympy.simplify.powsimp import powsimp
# 从 sympy.simplify.simplify 模块导入 simplify 函数
from sympy.simplify.simplify import simplify

# 从 sympy.abc 模块导入 x, y, n, k 符号
from sympy.abc import x, y, n, k


# 定义测试函数 test_gammasimp
def test_gammasimp():
    # 将 Rational 类赋值给 R
    R = Rational

    # 测试 gammasimp 函数的几个断言语句
    assert gammasimp(gamma(x)) == gamma(x)
    assert gammasimp(gamma(x + 1)/x) == gamma(x)
    assert gammasimp(gamma(x)/(x - 1)) == gamma(x - 1)
    assert gammasimp(x*gamma(x)) == gamma(x + 1)
    assert gammasimp((x + 1)*gamma(x + 1)) == gamma(x + 2)
    assert gammasimp(gamma(x + y)*(x + y)) == gamma(x + y + 1)
    assert gammasimp(x/gamma(x + 1)) == 1/gamma(x)
    assert gammasimp((x + 1)**2/gamma(x + 2)) == (x + 1)/gamma(x + 1)
    assert gammasimp(x*gamma(x) + gamma(x + 3)/(x + 2)) == \
        (x + 2)*gamma(x + 1)

    assert gammasimp(gamma(2*x)*x) == gamma(2*x + 1)/2
    assert gammasimp(gamma(2*x)/(x - S.Half)) == 2*gamma(2*x - 1)

    assert gammasimp(gamma(x)*gamma(1 - x)) == pi/sin(pi*x)
    assert gammasimp(gamma(x)*gamma(-x)) == -pi/(x*sin(pi*x))
    assert gammasimp(1/gamma(x + 3)/gamma(1 - x)) == \
        sin(pi*x)/(pi*x*(x + 1)*(x + 2))

    assert gammasimp(factorial(n + 2)) == gamma(n + 3)
    assert gammasimp(binomial(n, k)) == \
        gamma(n + 1)/(gamma(k + 1)*gamma(-k + n + 1))

    assert powsimp(gammasimp(
        gamma(x)*gamma(x + S.Half)*gamma(y)/gamma(x + y))) == \
        2**(-2*x + 1)*sqrt(pi)*gamma(2*x)*gamma(y)/gamma(x + y)
    assert gammasimp(1/gamma(x)/gamma(x - Rational(1, 3))/gamma(x + Rational(1, 3))) == \
        3**(3*x - Rational(3, 2))/(2*pi*gamma(3*x - 1))
    assert simplify(
        gamma(S.Half + x/2)*gamma(1 + x/2)/gamma(1 + x)/sqrt(pi)*2**x) == 1
    assert gammasimp(gamma(Rational(-1, 4))*gamma(Rational(-3, 4))) == 16*sqrt(2)*pi/3

    assert powsimp(gammasimp(gamma(2*x)/gamma(x))) == \
        2**(2*x - 1)*gamma(x + S.Half)/sqrt(pi)

    # issue 6792
    # 定义一个表达式 e
    e = (-gamma(k)*gamma(k + 2) + gamma(k + 1)**2)/gamma(k)**2
    assert gammasimp(e) == -k
    assert gammasimp(1/e) == -1/k
    e = (gamma(x) + gamma(x + 1))/gamma(x)
    assert gammasimp(e) == x + 1
    assert gammasimp(1/e) == 1/(x + 1)
    e = (gamma(x) + gamma(x + 2))*(gamma(x - 1) + gamma(x))/gamma(x)
    assert gammasimp(e) == (x**2 + x + 1)*gamma(x + 1)/(x - 1)
    e = (-gamma(k)*gamma(k + 2) + gamma(k + 1)**2)/gamma(k)**2
    assert gammasimp(e**2) == k**2
    assert gammasimp(e**2/gamma(k + 1)) == k/gamma(k)
    a = R(1, 2) + R(1, 3)
    # 计算变量 b 的值，等于变量 a 加上函数 R 的返回值在区间 [1, 3] 内的随机数
    b = a + R(1, 3)
    
    # 断言：对于给定的表达式，使用 gammasimp 函数简化后应该等于指定的值
    assert gammasimp(gamma(2*k)/gamma(k)*gamma(k + a)*gamma(k + b)
        ) == 3*2**(2*k + 1)*3**(-3*k - 2)*sqrt(pi)*gamma(3*k + R(3, 2))/2

    # issue 9699 的问题验证
    assert gammasimp((x + 1)*factorial(x)/gamma(y)) == gamma(x + 2)/gamma(y)
    
    # 断言：对于给定的表达式，使用 gammasimp 函数简化后应该等于 Piecewise 对象
    assert gammasimp(rf(x + n, k)*binomial(n, k)).simplify() == Piecewise(
        (gamma(n + 1)*gamma(k + n + x)/(gamma(k + 1)*gamma(n + x)*gamma(-k + n + 1)), n > -x),
        ((-1)**k*gamma(n + 1)*gamma(-n - x + 1)/(gamma(k + 1)*gamma(-k + n + 1)*gamma(-k - n - x + 1)), True))

    # 定义符号 A 和 B，这些符号不满足交换律
    A, B = symbols('A B', commutative=False)
    assert gammasimp(e*B*A) == gammasimp(e)*B*A

    # 检查迭代结果
    assert gammasimp(gamma(2*k)/gamma(k)*gamma(-k - R(1, 2))) == (
        -2**(2*k + 1)*sqrt(pi)/(2*((2*k + 1)*cos(pi*k))))
    assert gammasimp(
        gamma(k)*gamma(k + R(1, 3))*gamma(k + R(2, 3))/gamma(k*R(3, 2))) == (
        3*2**(3*k + 1)*3**(-3*k - S.Half)*sqrt(pi)*gamma(k*R(3, 2) + S.Half)/2)

    # issue 6153 的问题验证
    assert gammasimp(gamma(Rational(1, 4))/gamma(Rational(5, 4))) == 4

    # 测试函数 test_combsimp() 中的一部分
    assert gammasimp(binomial(n + 2, k + S.Half)) == gamma(n + 3)/ \
        (gamma(k + R(3, 2))*gamma(-k + n + R(5, 2)))
    assert gammasimp(binomial(n + 2, k + 2.0)) == \
        gamma(n + 3)/(gamma(k + 3.0)*gamma(-k + n + 1))

    # issue 11548 的问题验证
    assert gammasimp(binomial(0, x)) == sin(pi*x)/(pi*x)

    # 对于变量 e 的 gamma 函数应用，预期结果应该等于 e 本身
    e = gamma(n + Rational(1, 3))*gamma(n + R(2, 3))
    assert gammasimp(e) == e
    assert gammasimp(gamma(4*n + S.Half)/gamma(2*n - R(3, 4))) == \
        2**(4*n - R(5, 2))*(8*n - 3)*gamma(2*n + R(3, 4))/sqrt(pi)

    # 定义整数符号 i 和 m
    i, m = symbols('i m', integer = True)
    
    # 对于 gamma 函数的应用，预期结果应该等于函数本身
    e = gamma(exp(i))
    assert gammasimp(e) == e
    e = gamma(m + 3)
    assert gammasimp(e) == e
    
    # 对于复杂的 gamma 函数应用，预期结果应该等于函数本身
    e = gamma(m + 1)/(gamma(i + 1)*gamma(-i + m + 1))
    assert gammasimp(e) == e

    # 定义正整数符号 p
    p = symbols("p", integer=True, positive=True)
    
    # issue 11548 的问题验证
    assert gammasimp(gamma(-p + 4)) == gamma(-p + 4)
# 定义测试函数 test_issue_22606，用于测试特定问题
def test_issue_22606():
    # 创建一个函数表达式 fx = f(x)
    fx = Function('f')(x)
    # 创建一个表达式 eq = x + gamma(y)，其中 gamma 是伽马函数
    eq = x + gamma(y)
    # 对表达式 eq 进行伽马函数化简，得到结果 ans
    ans = gammasimp(eq)
    # 断言：将表达式 eq 中的 x 替换为 fx，再对 fx 替换为 x，应该得到 ans
    assert gammasimp(eq.subs(x, fx)).subs(fx, x) == ans
    # 断言：将表达式 eq 中的 x 替换为 cos(x)，再对 cos(x) 替换为 x，应该得到 ans
    assert gammasimp(eq.subs(x, cos(x))).subs(cos(x), x) == ans
    # 断言：对 1/eq 进行伽马函数化简，应该得到 1/ans
    assert 1/gammasimp(1/eq) == ans
    # 断言：对 fx 替换为 eq 后，取其第一个参数应该得到 ans
    assert gammasimp(fx.subs(x, eq)).args[0] == ans
```