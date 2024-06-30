# `D:\src\scipysrc\sympy\sympy\functions\elementary\benchmarks\bench_exp.py`

```
# 从 sympy.core.symbol 模块中导入 symbols 符号函数，用于创建符号变量
# 从 sympy.functions.elementary.exponential 模块中导入 exp 指数函数
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp

# 使用 symbols 函数创建符号变量 x 和 y
x, y = symbols('x,y')

# 计算 exp(2*x)，得到指数函数 2^x 的值
e = exp(2*x)

# 计算 exp(3*x)，得到指数函数 3^x 的值
q = exp(3*x)

# 定义一个函数 timeit_exp_subs，用于执行 e 对 q 的符号替换，并且没有返回值
def timeit_exp_subs():
    e.subs(q, y)
```