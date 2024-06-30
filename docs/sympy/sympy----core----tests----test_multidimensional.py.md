# `D:\src\scipysrc\sympy\sympy\core\tests\test_multidimensional.py`

```
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import sin
from sympy.core.multidimensional import vectorize

# 定义符号变量 x, y, z
x, y, z = symbols('x y z')
# 将字符串 'fgh' 转换为对应的 Function 对象列表 f, g, h
f, g, h = list(map(Function, 'fgh'))

# 定义测试函数 test_vectorize
def test_vectorize():
    # 使用 vectorize 装饰器，将 vsin 函数向量化，指定轴为 0
    @vectorize(0)
    def vsin(x):
        return sin(x)

    # 断言 vsin 函数对向量 [1, x, y] 的输出
    assert vsin([1, x, y]) == [sin(1), sin(x), sin(y)]

    # 使用 vectorize 装饰器，将 vdiff 函数向量化，指定轴为 0 和 1
    @vectorize(0, 1)
    def vdiff(f, y):
        return diff(f, y)

    # 断言 vdiff 函数对向量 [f(x, y, z), g(x, y, z), h(x, y, z)] 和 [x, y, z] 的输出
    assert vdiff([f(x, y, z), g(x, y, z), h(x, y, z)], [x, y, z]) == \
         [[Derivative(f(x, y, z), x), Derivative(f(x, y, z), y),
           Derivative(f(x, y, z), z)], [Derivative(g(x, y, z), x),
                     Derivative(g(x, y, z), y), Derivative(g(x, y, z), z)],
         [Derivative(h(x, y, z), x), Derivative(h(x, y, z), y), Derivative(h(x, y, z), z)]]
```