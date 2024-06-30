# `D:\src\scipysrc\sympy\sympy\series\tests\test_approximants.py`

```
# 导入 sympy 库中的 approximants 函数
from sympy.series import approximants
# 导入 sympy 库中的 symbols 函数
from sympy.core.symbol import symbols
# 导入 sympy 库中的 binomial 函数
from sympy.functions.combinatorial.factorials import binomial
# 导入 sympy 库中的 fibonacci 和 lucas 函数
from sympy.functions.combinatorial.numbers import (fibonacci, lucas)

# 定义测试函数 test_approximants
def test_approximants():
    # 定义符号变量 x 和 t
    x, t = symbols("x,t")
    # 构建 lucas 数列 g
    g = [lucas(k) for k in range(16)]
    # 断言使用 approximants 函数生成的近似值列表
    assert list(approximants(g)) == (
        [2, -4/(x - 2), (5*x - 2)/(3*x - 1), (x - 2)/(x**2 + x - 1)] )
    
    # 修改 lucas 数列 g，加上 fibonacci 数列
    g = [lucas(k)+fibonacci(k+2) for k in range(16)]
    # 断言使用 approximants 函数生成的近似值列表
    assert list(approximants(g)) == (
        [3, -3/(x - 1), (3*x - 3)/(2*x - 1), -3/(x**2 + x - 1)] )
    
    # 修改 lucas 数列 g，取其平方
    g = [lucas(k)**2 for k in range(16)]
    # 断言使用 approximants 函数生成的近似值列表
    assert list(approximants(g)) == (
        [4, -16/(x - 4), (35*x - 4)/(9*x - 1), (37*x - 28)/(13*x**2 + 11*x - 7),
        (50*x**2 + 63*x - 52)/(37*x**2 + 19*x - 13),
        (-x**2 - 7*x + 4)/(x**3 - 2*x**2 - 2*x + 1)] )
    
    # 构建多项式 p，每个项为 binomial(k, i) * x^i 的和
    p = [sum(binomial(k,i)*x**i for i in range(k+1)) for k in range(16)]
    # 使用 t 变量进行近似值计算，简化结果
    y = approximants(p, t, simplify=True)
    # 断言第一个生成的近似值为 1
    assert next(y) == 1
    # 断言下一个生成的近似值
    assert next(y) == -1/(t*(x + 1) - 1)
```