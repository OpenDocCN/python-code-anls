# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_singularityfunctions.py`

```
# 导入 sympy 库中的 singularityintegrate 函数
# 从 sympy.integrals.singularityfunctions 模块导入 singularityintegrate 函数
from sympy.integrals.singularityfunctions import singularityintegrate

# 导入 sympy 库中的 Function 类
# 从 sympy.core.function 模块导入 Function 类
from sympy.core.function import Function

# 导入 sympy 库中的 symbols 函数
# 从 sympy.core.symbol 模块导入 symbols 函数
from sympy.core.symbol import symbols

# 导入 sympy 库中的 SingularityFunction 类
# 从 sympy.functions.special.singularity_functions 模块导入 SingularityFunction 类
from sympy.functions.special.singularity_functions import SingularityFunction

# 定义符号变量 x, a, n, y
x, a, n, y = symbols('x a n y')

# 定义函数 f
f = Function('f')

# 定义测试函数 test_singularityintegrate
def test_singularityintegrate():
    # 断言 singularityintegrate(x, x) 返回 None
    assert singularityintegrate(x, x) is None
    # 断言 singularityintegrate(x + SingularityFunction(x, 9, 1), x) 返回 None
    assert singularityintegrate(x + SingularityFunction(x, 9, 1), x) is None

    # 断言 4*singularityintegrate(SingularityFunction(x, a, 3), x) 等于 4*SingularityFunction(x, a, 4)/4
    assert 4*singularityintegrate(SingularityFunction(x, a, 3), x) == 4*SingularityFunction(x, a, 4)/4
    # 断言 singularityintegrate(5*SingularityFunction(x, 5, -2), x) 等于 5*SingularityFunction(x, 5, -1)
    assert singularityintegrate(5*SingularityFunction(x, 5, -2), x) == 5*SingularityFunction(x, 5, -1)
    # 断言 singularityintegrate(6*SingularityFunction(x, 5, -1), x) 等于 6*SingularityFunction(x, 5, 0)
    assert singularityintegrate(6*SingularityFunction(x, 5, -1), x) == 6*SingularityFunction(x, 5, 0)
    # 断言 singularityintegrate(x*SingularityFunction(x, 0, -1), x) 等于 0
    assert singularityintegrate(x*SingularityFunction(x, 0, -1), x) == 0
    # 断言 singularityintegrate((x - 5)*SingularityFunction(x, 5, -1), x) 等于 0
    assert singularityintegrate((x - 5)*SingularityFunction(x, 5, -1), x) == 0
    # 断言 singularityintegrate(SingularityFunction(x, 0, -1) * f(x), x) 等于 f(0) * SingularityFunction(x, 0, 0)
    assert singularityintegrate(SingularityFunction(x, 0, -1) * f(x), x) == f(0) * SingularityFunction(x, 0, 0)
    # 断言 singularityintegrate(SingularityFunction(x, 1, -1) * f(x), x) 等于 f(1) * SingularityFunction(x, 1, 0)
    assert singularityintegrate(SingularityFunction(x, 1, -1) * f(x), x) == f(1) * SingularityFunction(x, 1, 0)
    # 断言 singularityintegrate(y*SingularityFunction(x, 0, -1)**2, x) 等于 y*SingularityFunction(0, 0, -1)*SingularityFunction(x, 0, 0)
    assert singularityintegrate(y*SingularityFunction(x, 0, -1)**2, x) == \
        y*SingularityFunction(0, 0, -1)*SingularityFunction(x, 0, 0)
```