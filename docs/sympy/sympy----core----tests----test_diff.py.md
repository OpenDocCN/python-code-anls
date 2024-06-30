# `D:\src\scipysrc\sympy\sympy\core\tests\test_diff.py`

```
# 导入 SymPy 模块中的具体类和函数

from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, diff, Subs)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.tensor.array.ndim_array import NDimArray
from sympy.testing.pytest import raises
from sympy.abc import a, b, c, x, y, z

# 定义一个测试函数，用于测试 SymPy 中的微分功能
def test_diff():
    # 断言：有理数 1/3 对 x 求偏导数为 0
    assert Rational(1, 3).diff(x) is S.Zero
    # 断言：虚数单位 I 对 x 求偏导数为 0
    assert I.diff(x) is S.Zero
    # 断言：π 对 x 求偏导数为 0
    assert pi.diff(x) is S.Zero
    # 断言：x 对 x 求 0 阶偏导数得到 x 本身
    assert x.diff(x, 0) == x
    # 断言：x^2 对 x 连续求 2 次偏导数再求一次 x 的偏导数为 0
    assert (x**2).diff(x, 2, x) == 0
    # 断言：x^2 对 x 连续求 2 次偏导数再求一次 (x, 1) 的偏导数为 0
    assert (x**2).diff((x, 2), x) == 0
    # 断言：x^2 对 x 连续求 1 次偏导数再求 (x, 1) 的偏导数为 2
    assert (x**2).diff((x, 1), x) == 2
    # 断言：x^2 对 x 连续求 1 次偏导数再求 (x, 1) 的偏导数为 2
    assert (x**2).diff((x, 1), (x, 1)) == 2
    # 断言：x^2 对 x 连续求 2 次偏导数为 2
    assert (x**2).diff((x, 2)) == 2
    # 断言：x^2 对 x, y, 0 求偏导数得到 2x
    assert (x**2).diff(x, y, 0) == 2*x
    # 断言：x^2 对 x, (y, 0) 求偏导数得到 2x
    assert (x**2).diff(x, (y, 0)) == 2*x
    # 断言：x^2 对 x, y 求偏导数为 0
    assert (x**2).diff(x, y) == 0
    # 断言：尝试使用 x 对 1 求偏导数引发 ValueError
    raises(ValueError, lambda: x.diff(1, x))

    # 定义有理数 p，表达式 e
    p = Rational(5)
    e = a*b + b**p
    # 断言：e 对 a 求偏导数得到 b
    assert e.diff(a) == b
    # 断言：e 对 b 求偏导数得到 a + 5*b^4
    assert e.diff(b) == a + 5*b**4
    # 断言：e 对 b 求偏导数再对 a 求偏导数得到 1
    assert e.diff(b).diff(a) == Rational(1)

    # 重新定义表达式 e
    e = a*(b + c)
    # 断言：e 对 a 求偏导数得到 b + c
    assert e.diff(a) == b + c
    # 断言：e 对 b 求偏导数得到 a
    assert e.diff(b) == a
    # 断言：e 对 b 求偏导数再对 a 求偏导数得到 1
    assert e.diff(b).diff(a) == Rational(1)

    # 重新定义表达式 e
    e = c**p
    # 断言：e 对 c 连续求 6 次偏导数得到 0
    assert e.diff(c, 6) == Rational(0)
    # 断言：e 对 c 连续求 5 次偏导数得到 120
    assert e.diff(c, 5) == Rational(120)

    # 重新定义表达式 e
    e = c**Rational(2)
    # 断言：e 对 c 求偏导数得到 2c
    assert e.diff(c) == 2*c

    # 重新定义表达式 e
    e = a*b*c
    # 断言：e 对 c 求偏导数得到 ab
    assert e.diff(c) == a*b


def test_diff2():
    # 定义有理数常量
    n3 = Rational(3)
    n2 = Rational(2)
    n6 = Rational(6)

    # 定义复杂表达式 e
    e = n3*(-n2 + x**n2)*cos(x) + x*(-n6 + x**n2)*sin(x)
    # 断言：验证复杂表达式 e 的结果
    assert e == 3*(-2 + x**2)*cos(x) + x*(-6 + x**2)*sin(x)
    # 断言：e 对 x 求偏导并展开得到 x^3*cos(x)
    assert e.diff(x).expand() == x**3*cos(x)

    # 重新定义表达式 e
    e = (x + 1)**3
    # 断言：e 对 x 求偏导得到 3*(x + 1)^2
    assert e.diff(x) == 3*(x + 1)**2

    # 重新定义表达式 e
    e = x*(x + 1)**3
    # 断言：e 对 x 求偏导得到 (x + 1)^3 + 3*x*(x + 1)^2
    assert e.diff(x) == (x + 1)**3 + 3*x*(x + 1)**2

    # 重新定义表达式 e
    e = 2*exp(x*x)*x
    # 断言：e 对 x 求偏导得到 2*exp(x^2) + 4*x^2*exp(x^2)
    assert e.diff(x) == 2*exp(x**2) + 4*x**2*exp(x**2)


def test_diff3():
    # 定义有理数常量 p
    p = Rational(5)

    # 重新定义表达式 e
    e = a*b + sin(b**p)
    # 断言：验证表达式 e 的结果
    assert e == a*b + sin(b**5)
    # 断言：e 对 a 求偏导得到 b
    assert e.diff(a) == b
    # 断言：e 对 b 求偏导得到 a + 5*b^4*cos(b^5)
    assert e.diff(b) == a + 5*b**4*cos(b**5)

    # 重新定义表达式 e
    e = tan(c)
    # 断言：验证表达式 e 的结果
    assert e == tan(c)
    # 断言：e 对 c 求偏导结果应在 [cos(c)^(-2), 1 + sin(c)^2/cos(c)^2, 1 + tan(c)^2] 中
    assert e.diff(c) in [cos(c)**(-2), 1 + sin(c)**2/cos(c)**2, 1 + tan(c)**2]

    # 重新定义表达式 e
    e = c*log(c) - c
    # 断言：验证表达式 e 的结果
    assert e ==
    # 断言：使用 My 类创建对象 x，对其进行一阶导数操作，并检查结果是否为 Derivative 类型
    assert My(x).diff(x).func is Derivative
    
    # 断言：使用 My 类创建对象 x，对其进行三阶导数操作，并检查结果是否为 Derivative 类型
    assert My(x).diff(x, 3).func is Derivative
    
    # 断言：对 re(x) 对象进行二阶导数操作，检查结果是否等于 Derivative(re(x), (x, 2))，用于解决问题 15518
    assert re(x).diff(x, 2) == Derivative(re(x), (x, 2))  # issue 15518
    
    # 断言：对包含 [re(x), im(x)] 的 NDimArray 对象进行二阶导数操作，期望结果是包含导数结果的 NDimArray
    assert diff(NDimArray([re(x), im(x)]), (x, 2)) == NDimArray(
        [Derivative(re(x), (x, 2)), Derivative(im(x), (x, 2))])
    
    # 断言：针对 My 类创建的对象 x，进行对 y 的导数操作，预期结果为 0
    # 如果 y 不是对象的属性，则该操作应不需要特定的方法
    assert My(x).diff(y) == 0
def test_speed():
    # 测试函数运行速度是否合理，应当在 0.0 秒内返回。如果运行时间过长，可能存在问题。
    assert x.diff(x, 10**8) == 0


def test_deriv_noncommutative():
    # 定义一个非交换符号 A
    A = Symbol("A", commutative=False)
    # 定义一个函数 f(x)
    f = Function("f")
    # 验证非交换乘法的性质
    assert A*f(x)*A == f(x)*A**2
    # 验证函数导数在非交换乘法下的性质
    assert A*f(x).diff(x)*A == f(x).diff(x) * A**2


def test_diff_nth_derivative():
    # 定义一个函数 f(x)
    f = Function("f")
    # 定义一个整数符号 n
    n = Symbol("n", integer=True)

    # 计算 sin(x) 的 n 阶导数
    expr = diff(sin(x), (x, n))
    # 计算 f(x) 的二阶导数
    expr2 = diff(f(x), (x, 2))
    # 计算 f(x) 的 n 阶导数
    expr3 = diff(f(x), (x, n))

    # 验证在特定替换下 sin(x) 的 n 阶导数
    assert expr.subs(sin(x), cos(-x)) == Derivative(cos(-x), (x, n))
    # 验证 n=1 时的 sin(x) 的导数
    assert expr.subs(n, 1).doit() == cos(x)
    # 验证 n=2 时的 sin(x) 的导数
    assert expr.subs(n, 2).doit() == -sin(x)

    # 验证二阶导数的性质
    assert expr2.subs(Derivative(f(x), x), y) == Derivative(y, x)
    # 当前不支持的断言（无法确定 `n > 1`）:
    # assert expr3.subs(Derivative(f(x), x), y) == Derivative(y, (x, n-1))
    # 验证 f(x) 的 n 阶导数的表达式
    assert expr3 == Derivative(f(x), (x, n))

    # 计算 x 的 n 阶导数
    assert diff(x, (x, n)) == Piecewise((x, Eq(n, 0)), (1, Eq(n, 1)), (0, True))
    # 计算 2*x 的 n 阶导数
    assert diff(2*x, (x, n)).dummy_eq(
        Sum(Piecewise((2*x*factorial(n)/(factorial(y)*factorial(-y + n)),
        Eq(y, 0) & Eq(Max(0, -y + n), 0)),
        (2*factorial(n)/(factorial(y)*factorial(-y + n)), Eq(y, 0) & Eq(Max(0,
        -y + n), 1)), (0, True)), (y, 0, n)))
    # TODO: assert diff(x**2, (x, n)) == x**(2-n)*ff(2, n)

    # 计算复合函数 x*sin(x) 的 n 阶导数
    exprm = x*sin(x)
    mul_diff = diff(exprm, (x, n))
    assert isinstance(mul_diff, Sum)
    # 验证前五阶导数的展开结果
    for i in range(5):
        assert mul_diff.subs(n, i).doit() == exprm.diff((x, i)).expand()

    # 计算复合函数 2*y*x*sin(x)*cos(x)*log(x)*exp(x) 的 n 阶导数
    exprm2 = 2*y*x*sin(x)*cos(x)*log(x)*exp(x)
    dex = exprm2.diff((x, n))
    assert isinstance(dex, Sum)
    # 验证前七阶导数的展开结果
    for i in range(7):
        assert dex.subs(n, i).doit().expand() == \
        exprm2.diff((x, i)).expand()

    # 计算 (cos(x)*sin(y)) 对 [x, y, z] 的偏导数
    assert (cos(x)*sin(y)).diff([[x, y, z]]) == NDimArray([
        -sin(x)*sin(y), cos(x)*cos(y), 0])


def test_issue_16160():
    # 验证 x**3 对 x 的二阶导数在 x=2 处的值
    assert Derivative(x**3, (x, x)).subs(x, 2) == Subs(
        Derivative(x**3, (x, 2)), x, 2)
    # 验证 1 + x**3 对 x 的二阶导数在 x=0 处的值
    assert Derivative(1 + x**3, (x, x)).subs(x, 0
        ) == Derivative(1 + y**3, (y, 0)).subs(y, 0)
```