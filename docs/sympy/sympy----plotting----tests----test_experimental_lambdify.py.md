# `D:\src\scipysrc\sympy\sympy\plotting\tests\test_experimental_lambdify.py`

```
# 导入需要的模块和函数
from sympy.core.symbol import symbols, Symbol
from sympy.functions import Max
from sympy.plotting.experimental_lambdify import experimental_lambdify
from sympy.plotting.intervalmath.interval_arithmetic import \
    interval, intervalMembership

# 测试 experimental_lambdify 中的异常处理
def test_experimental_lambify():
    # 定义符号变量 x
    x = Symbol('x')
    # 创建一个 Lambda 函数 f，计算 Max(x, 5)
    f = experimental_lambdify([x], Max(x, 5))
    # 断言 Max(2, 5) 等于 5
    assert Max(2, 5) == 5
    # 断言 Max(5, 7) 等于 7
    assert Max(5, 7) == 7

    # 定义符号变量 x-3（x 减 3）
    x = Symbol('x-3')
    # 创建一个 Lambda 函数 f，计算 x + 1
    f = experimental_lambdify([x], x + 1)
    # 断言 f(1) 等于 2
    assert f(1) == 2


# 测试复合布尔区域
def test_composite_boolean_region():
    # 定义符号变量 x 和 y
    x, y = symbols('x y')

    # 定义布尔区域 r1 和 r2
    r1 = (x - 1)**2 + y**2 < 2
    r2 = (x + 1)**2 + y**2 < 2

    # 创建 Lambda 函数 f，计算 r1 & r2
    f = experimental_lambdify((x, y), r1 & r2)
    # 定义不同的区间并断言结果
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)

    # 创建 Lambda 函数 f，计算 r1 | r2
    f = experimental_lambdify((x, y), r1 | r2)
    # 定义不同的区间并断言结果
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)

    # 创建 Lambda 函数 f，计算 r1 & ~r2
    f = experimental_lambdify((x, y), r1 & ~r2)
    # 定义不同的区间并断言结果
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)

    # 创建 Lambda 函数 f，计算 ~r1 & r2
    f = experimental_lambdify((x, y), ~r1 & r2)
    # 定义不同的区间并断言结果
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)

    # 创建 Lambda 函数 f，计算 ~r1 & ~r2
    f = experimental_lambdify((x, y), ~r1 & ~r2)
    # 定义不同的区间并断言结果
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    # 断言函数 f(*a) 的返回结果与 intervalMembership(False, True) 相等
    assert f(*a) == intervalMembership(False, True)
    # 更新变量 a，设置为包含两个 interval 对象的元组
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    # 断言函数 f(*a) 的返回结果与 intervalMembership(True, True) 相等
    assert f(*a) == intervalMembership(True, True)
```