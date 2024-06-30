# `D:\src\scipysrc\sympy\sympy\core\benchmarks\bench_basic.py`

```
# 从 sympy.core 模块中导入 symbols 和 S
from sympy.core import symbols, S

# 使用 symbols 函数创建符号变量 x 和 y
x, y = symbols('x,y')

# 定义函数 timeit_Symbol_meth_lookup，用于演示符号变量的方法查找
def timeit_Symbol_meth_lookup():
    x.diff  # 仅进行方法查找，没有调用

# 定义函数 timeit_S_lookup，用于演示符号常量 S 的使用
def timeit_S_lookup():
    S.Exp1  # 引用 sympy 中的常量 Exp1

# 定义函数 timeit_Symbol_eq_xy，用于演示符号变量之间的相等性比较
def timeit_Symbol_eq_xy():
    x == y  # 检查符号变量 x 和 y 是否相等
```