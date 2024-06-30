# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_rewrite.py`

```
# 导入 sympy 库中所需的模块和函数
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (cos, cot, sin)
from sympy.testing.pytest import _both_exp_pow

# 定义符号变量 x, y, z, n
x, y, z, n = symbols('x,y,z,n')

# 使用修饰器 @_both_exp_pow 标记的测试函数，用于测试表达式中的函数特性
@_both_exp_pow
def test_has():
    # 断言 cot(x) 是否包含变量 x
    assert cot(x).has(x)
    # 断言 cot(x) 是否包含 cot 函数
    assert cot(x).has(cot)
    # 断言 cot(x) 是否不包含 sin 函数
    assert not cot(x).has(sin)
    # 断言 sin(x) 是否包含变量 x
    assert sin(x).has(x)
    # 断言 sin(x) 是否包含 sin 函数
    assert sin(x).has(sin)
    # 断言 sin(x) 是否不包含 cot 函数
    assert not sin(x).has(cot)
    # 断言 exp(x) 是否包含 exp 函数
    assert exp(x).has(exp)

# 使用修饰器 @_both_exp_pow 标记的测试函数，用于测试 sin 函数与 exp 函数的重写
@_both_exp_pow
def test_sin_exp_rewrite():
    # 断言 sin(x) 重写为 exp 函数表达式的结果是否符合预期
    assert sin(x).rewrite(sin, exp) == -I/2*(exp(I*x) - exp(-I*x))
    # 断言连续两次重写后结果是否恢复为原始 sin(x) 表达式
    assert sin(x).rewrite(sin, exp).rewrite(exp, sin) == sin(x)
    # 断言 cos(x) 重写为 exp 函数表达式的结果是否与原始 cos(x) 相等
    assert cos(x).rewrite(cos, exp).rewrite(exp, cos) == cos(x)
    # 断言 sin(5*y) - sin(2*x) 的重写结果是否保持不变
    assert (sin(5*y) - sin(2*x)).rewrite(sin, exp).rewrite(exp, sin) == sin(5*y) - sin(2*x)
    # 断言 sin(x + y) 的重写结果是否保持不变
    assert sin(x + y).rewrite(sin, exp).rewrite(exp, sin) == sin(x + y)
    # 断言 cos(x + y) 的重写结果是否保持不变
    assert cos(x + y).rewrite(cos, exp).rewrite(exp, cos) == cos(x + y)
    # 断言 cos(x) 重写为 exp 函数表达式后再重写为 sin 函数表达式的结果是否与原始 cos(x) 相等
    assert cos(x).rewrite(cos, exp).rewrite(exp, sin) == cos(x)
```