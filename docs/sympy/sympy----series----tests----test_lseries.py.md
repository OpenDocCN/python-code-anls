# `D:\src\scipysrc\sympy\sympy\series\tests\test_lseries.py`

```
# 从 sympy.core.numbers 模块导入常数 E
from sympy.core.numbers import E
# 从 sympy.core.singleton 模块导入常数 S
from sympy.core.singleton import S
# 从 sympy.functions.elementary.exponential 模块导入 exp 函数
from sympy.functions.elementary.exponential import exp
# 从 sympy.functions.elementary.hyperbolic 模块导入 tanh 函数
from sympy.functions.elementary.hyperbolic import tanh
# 从 sympy.functions.elementary.trigonometric 模块导入 cos, sin 函数
from sympy.functions.elementary.trigonometric import (cos, sin)
# 从 sympy.series.order 模块导入 Order 类
from sympy.series.order import Order
# 从 sympy.abc 模块导入 x, y 符号

# 定义测试 sin 函数的函数
def test_sin():
    # 获取 sin(x) 的 x 级数展开生成器
    e = sin(x).lseries(x)
    # 断言下一个生成器的值为 x
    assert next(e) == x
    # 断言下一个生成器的值为 -x**3/6
    assert next(e) == -x**3/6
    # 断言下一个生成器的值为 x**5/120
    assert next(e) == x**5/120

# 定义测试 cos 函数的函数
def test_cos():
    # 获取 cos(x) 的 x 级数展开生成器
    e = cos(x).lseries(x)
    # 断言下一个生成器的值为 1
    assert next(e) == 1
    # 断言下一个生成器的值为 -x**2/2
    assert next(e) == -x**2/2
    # 断言下一个生成器的值为 x**4/24
    assert next(e) == x**4/24

# 定义测试 exp 函数的函数
def test_exp():
    # 获取 exp(x) 的 x 级数展开生成器
    e = exp(x).lseries(x)
    # 断言下一个生成器的值为 1
    assert next(e) == 1
    # 断言下一个生成器的值为 x
    assert next(e) == x
    # 断言下一个生成器的值为 x**2/2
    assert next(e) == x**2/2
    # 断言下一个生成器的值为 x**3/6
    assert next(e) == x**3/6

# 定义测试 exp(cos(x)) 函数的函数
def test_exp2():
    # 获取 exp(cos(x)) 的 x 级数展开生成器
    e = exp(cos(x)).lseries(x)
    # 断言下一个生成器的值为 E
    assert next(e) == E
    # 断言下一个生成器的值为 -E*x**2/2
    assert next(e) == -E*x**2/2
    # 断言下一个生成器的值为 E*x**4/6
    assert next(e) == E*x**4/6
    # 断言下一个生成器的值为 -31*E*x**6/720
    assert next(e) == -31*E*x**6/720

# 定义测试简单表达式的函数
def test_simple():
    # 断言 x 的级数展开结果为 [x]
    assert list(x.lseries()) == [x]
    # 断言 S.One 在 x 上的级数展开结果为 [1]
    assert list(S.One.lseries(x)) == [1]
    # 断言 x/(x + y) 在 y 上的级数展开不含 Order
    assert not next((x/(x + y)).lseries(y)).has(Order)

# 定义测试 issue 5183 的函数
def test_issue_5183():
    # 计算 (x + 1/x) 的级数展开
    s = (x + 1/x).lseries()
    # 断言展开结果为 [1/x, x]
    assert list(s) == [1/x, x]
    # 断言 (x + x**2) 的级数展开的下一个值为 x
    assert next((x + x**2).lseries()) == x
    # 断言 ((1 + x)**7) 在 x 上的级数展开的下一个值为 1
    assert next(((1 + x)**7).lseries(x)) == 1
    # 断言 sin(x + y) 在 x 展开到 n=3 后再在 y 上的级数展开的下一个值为 x
    assert next((sin(x + y)).series(x, n=3).lseries(y)) == x
    # 在以下情况下，如果所有项都被分组，那将是很好的，但实际上每个项都有常数，
    # 因此必须知道每个项。
    s = ((1 + x)**7).series(x, 1, n=None)
    # 断言 s 的前两个值为 [128, -448 + 448*x]
    assert [next(s) for i in range(2)] == [128, -448 + 448*x]

# 定义测试 issue 6999 的函数
def test_issue_6999():
    # 计算 tanh(x) 在 x=1 上的级数展开
    s = tanh(x).lseries(x, 1)
    # 断言展开结果的下一个值为 tanh(1)
    assert next(s) == tanh(1)
    # 断言展开结果的下一个值为 x - (x - 1)*tanh(1)**2 - 1
    assert next(s) == x - (x - 1)*tanh(1)**2 - 1
    # 断言展开结果的下一个值为 -(x - 1)**2*tanh(1) + (x - 1)**2*tanh(1)**3
    assert next(s) == -(x - 1)**2*tanh(1) + (x - 1)**2*tanh(1)**3
    # 断言展开结果的下一个值为 -(x - 1)**3*tanh(1)**4 - (x - 1)**3/3 +
    # 4*(x - 1)**3*tanh(1)**2/3
    assert next(s) == -(x - 1)**3*tanh(1)**4 - (x - 1)**3/3 + \
        4*(x - 1)**3*tanh(1)**2/3
```