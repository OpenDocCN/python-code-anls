# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_cfunctions.py`

```
from sympy.core.numbers import (Rational, pi)  # 导入 Rational 和 pi 从 sympy.core.numbers 模块
from sympy.core.singleton import S  # 导入 S 从 sympy.core.singleton 模块
from sympy.core.symbol import (Symbol, symbols)  # 导入 Symbol 和 symbols 从 sympy.core.symbol 模块
from sympy.functions.elementary.exponential import (exp, log)  # 导入 exp 和 log 从 sympy.functions.elementary.exponential 模块
from sympy.codegen.cfunctions import (  # 导入 expm1, log1p, exp2, log2, fma, log10, Sqrt, Cbrt, hypot 从 sympy.codegen.cfunctions 模块
    expm1, log1p, exp2, log2, fma, log10, Sqrt, Cbrt, hypot
)
from sympy.core.function import expand_log  # 导入 expand_log 从 sympy.core.function 模块


def test_expm1():
    # Eval
    assert expm1(0) == 0  # 检查 expm1 在输入为 0 时的值是否为 0

    x = Symbol('x', real=True)

    # Expand and rewrite
    assert expm1(x).expand(func=True) - exp(x) == -1  # 展开 expm1(x)，并检查是否等于 exp(x) - 1
    assert expm1(x).rewrite('tractable') - exp(x) == -1  # 使用 'tractable' 方式重写 expm1(x)，检查是否等于 exp(x) - 1
    assert expm1(x).rewrite('exp') - exp(x) == -1  # 使用 'exp' 方式重写 expm1(x)，检查是否等于 exp(x) - 1

    # Precision
    assert not ((exp(1e-10).evalf() - 1) - 1e-10 - 5e-21) < 1e-22  # 精度比较，确保 exp(1e-10) 的值与期望值之间的差距在可接受范围内
    assert abs(expm1(1e-10).evalf() - 1e-10 - 5e-21) < 1e-22  # 精度比较，确保 expm1(1e-10) 的值与期望值之间的差距在可接受范围内

    # Properties
    assert expm1(x).is_real  # 检查 expm1(x) 是否为实数
    assert expm1(x).is_finite  # 检查 expm1(x) 是否有限

    # Diff
    assert expm1(42*x).diff(x) - 42*exp(42*x) == 0  # 检查对 expm1(42*x) 求导后的结果是否与预期相等
    assert expm1(42*x).diff(x) - expm1(42*x).expand(func=True).diff(x) == 0  # 检查对 expm1(42*x) 展开后再求导的结果是否与直接求导的结果相等


def test_log1p():
    # Eval
    assert log1p(0) == 0  # 检查 log1p 在输入为 0 时的值是否为 0
    d = S(10)
    assert expand_log(log1p(d**-1000) - log(d**1000 + 1) + log(d**1000)) == 0  # 展开 log1p(d**-1000) 并验证是否等于 0

    x = Symbol('x', real=True)

    # Expand and rewrite
    assert log1p(x).expand(func=True) - log(x + 1) == 0  # 展开 log1p(x) 并检查是否等于 log(x + 1)
    assert log1p(x).rewrite('tractable') - log(x + 1) == 0  # 使用 'tractable' 方式重写 log1p(x)，检查是否等于 log(x + 1)
    assert log1p(x).rewrite('log') - log(x + 1) == 0  # 使用 'log' 方式重写 log1p(x)，检查是否等于 log(x + 1)

    # Precision
    assert not abs(log(1e-99 + 1).evalf() - 1e-99) < 1e-100  # 精度比较，确保 log(1e-99 + 1) 的值与期望值之间的差距在可接受范围内
    assert abs(expand_log(log1p(1e-99)).evalf() - 1e-99) < 1e-100  # 精度比较，确保 expand_log(log1p(1e-99)) 的值与期望值之间的差距在可接受范围内

    # Properties
    assert log1p(-2**Rational(-1, 2)).is_real  # 检查 log1p(-2**Rational(-1, 2)) 是否为实数

    assert not log1p(-1).is_finite  # 检查 log1p(-1) 是否为有限数
    assert log1p(pi).is_finite  # 检查 log1p(pi) 是否为有限数

    assert not log1p(x).is_positive  # 检查 log1p(x) 是否为正数
    assert log1p(Symbol('y', positive=True)).is_positive  # 检查 log1p(Symbol('y', positive=True)) 是否为正数

    assert not log1p(x).is_zero  # 检查 log1p(x) 是否为零
    assert log1p(Symbol('z', zero=True)).is_zero  # 检查 log1p(Symbol('z', zero=True)) 是否为零

    assert not log1p(x).is_nonnegative  # 检查 log1p(x) 是否为非负数
    assert log1p(Symbol('o', nonnegative=True)).is_nonnegative  # 检查 log1p(Symbol('o', nonnegative=True)) 是否为非负数

    # Diff
    assert log1p(42*x).diff(x) - 42/(42*x + 1) == 0  # 检查对 log1p(42*x) 求导后的结果是否与预期相等
    assert log1p(42*x).diff(x) - log1p(42*x).expand(func=True).diff(x) == 0  # 检查对 log1p(42*x) 展开后再求导的结果是否与直接求导的结果相等


def test_exp2():
    # Eval
    assert exp2(2) == 4  # 检查 exp2(2) 的值是否为 4

    x = Symbol('x', real=True)

    # Expand
    assert exp2(x).expand(func=True) - 2**x == 0  # 展开 exp2(x) 并检查是否等于 2**x

    # Diff
    assert exp2(42*x).diff(x) - 42*exp2(42*x)*log(2) == 0  # 检查对 exp2(42*x) 求导后的结果是否与预期相等
    assert exp2(42*x).diff(x) - exp2(42*x).diff(x) == 0  # 检查对 exp2(42*x) 求导后的结果是否与自身求导结果相等


def test_log2():
    # Eval
    assert log2(8) == 3  # 检查 log2(8) 的值是否为 3
    assert log2(pi) != log(pi)/log(2)  # 检查 log2(pi) 是否等于 log(pi)/log(2)，以节省 CPU 指令

    x = Symbol('x', real=True)
    assert log2(x) != log(x)/log(2)  # 检查 log2(x) 是否等于 log(x)/log(2)
    assert log2(2**x) == x  # 检查 log2(2**x) 是否等于 x

    # Expand
    assert log2(x).expand(func=True) - log(x)/log(2) == 0  # 展开 log2(x) 并检查是否等于 log(x)/log(2)

    # Diff
    assert log2(42*x).diff() - 1/(log(2)*x) == 0  #
    # 断言：验证表达式在对变量 x 求导后，经过展开后对 x 求导的结果是否为零
    assert expr.diff(x) - expr.expand(func=True).diff(x) == 0
    
    # 断言：验证表达式在对变量 y 求导后，经过展开后对 y 求导的结果是否为零
    assert expr.diff(y) - expr.expand(func=True).diff(y) == 0
    
    # 断言：验证表达式在对变量 z 求导后，经过展开后对 z 求导的结果是否为零
    assert expr.diff(z) - expr.expand(func=True).diff(z) == 0
    
    # 断言：验证表达式在对变量 x 求导后的结果是否等于 17*42*y
    assert expr.diff(x) - 17*42*y == 0
    
    # 断言：验证表达式在对变量 y 求导后的结果是否等于 17*42*x
    assert expr.diff(y) - 17*42*x == 0
    
    # 断言：验证表达式在对变量 z 求导后的结果是否等于 101
    assert expr.diff(z) - 101 == 0
def test_log10():
    x = Symbol('x')

    # Expand
    assert log10(x).expand(func=True) - log(x)/log(10) == 0

    # Diff
    assert log10(42*x).diff(x) - 1/(log(10)*x) == 0
    assert log10(42*x).diff(x) - log10(42*x).expand(func=True).diff(x) == 0


def test_Cbrt():
    x = Symbol('x')

    # Expand
    assert Cbrt(x).expand(func=True) - x**Rational(1, 3) == 0

    # Diff
    assert Cbrt(42*x).diff(x) - 42*(42*x)**(Rational(1, 3) - 1)/3 == 0
    assert Cbrt(42*x).diff(x) - Cbrt(42*x).expand(func=True).diff(x) == 0


def test_Sqrt():
    x = Symbol('x')

    # Expand
    assert Sqrt(x).expand(func=True) - x**S.Half == 0

    # Diff
    assert Sqrt(42*x).diff(x) - 42*(42*x)**(S.Half - 1)/2 == 0
    assert Sqrt(42*x).diff(x) - Sqrt(42*x).expand(func=True).diff(x) == 0


def test_hypot():
    x, y = symbols('x y')

    # Expand
    assert hypot(x, y).expand(func=True) - (x**2 + y**2)**S.Half == 0

    # Diff
    assert hypot(17*x, 42*y).diff(x).expand(func=True) - hypot(17*x, 42*y).expand(func=True).diff(x) == 0
    assert hypot(17*x, 42*y).diff(y).expand(func=True) - hypot(17*x, 42*y).expand(func=True).diff(y) == 0

    assert hypot(17*x, 42*y).diff(x).expand(func=True) - 2*17*17*x*((17*x)**2 + (42*y)**2)**Rational(-1, 2)/2 == 0
    assert hypot(17*x, 42*y).diff(y).expand(func=True) - 2*42*42*y*((17*x)**2 + (42*y)**2)**Rational(-1, 2)/2 == 0
```