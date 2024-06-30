# `D:\src\scipysrc\sympy\sympy\functions\special\benchmarks\bench_special.py`

```
# 从 sympy 库中导入符号变量 symbols 和球谐函数 Ynm
from sympy.core.symbol import symbols
from sympy.functions.special.spherical_harmonics import Ynm

# 创建符号变量 x 和 y
x, y = symbols('x,y')

# 定义函数 timeit_Ynm_xy，用于测试计算球谐函数 Ynm(1, 1, x, y) 的执行时间
def timeit_Ynm_xy():
    # 调用 sympy 库中的球谐函数 Ynm，计算 Ynm(1, 1, x, y)
    Ynm(1, 1, x, y)
```