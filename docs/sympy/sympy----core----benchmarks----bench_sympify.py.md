# `D:\src\scipysrc\sympy\sympy\core\benchmarks\bench_sympify.py`

```
# 从 sympy.core 模块中导入 sympify 和 Symbol 函数
from sympy.core import sympify, Symbol

# 创建一个符号变量 x
x = Symbol('x')

# 定义函数 timeit_sympify_1，用于测试 sympify 函数对整数 1 的性能
def timeit_sympify_1():
    # 使用 sympify 函数将整数 1 转换为 SymPy 的表达式对象
    sympify(1)

# 定义函数 timeit_sympify_x，用于测试 sympify 函数对符号变量 x 的性能
def timeit_sympify_x():
    # 使用 sympify 函数将符号变量 x 转换为 SymPy 的表达式对象
    sympify(x)
```