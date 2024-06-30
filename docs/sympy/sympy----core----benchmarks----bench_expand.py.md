# `D:\src\scipysrc\sympy\sympy\core\benchmarks\bench_expand.py`

```
# 导入sympy模块中的symbols和I函数
from sympy.core import symbols, I

# 定义符号变量x, y, z
x, y, z = symbols('x,y,z')

# 定义多项式p
p = 3*x**2*y*z**7 + 7*x*y*z**2 + 4*x + x*y**4

# 定义表达式e
e = (x + y + z + 1)**32

# 定义一个函数，用于对多项式p进行展开，但不做任何事情
def timeit_expand_nothing_todo():
    p.expand()

# 定义一个函数，用于对表达式e进行展开为多项式，输出展开结果
def bench_expand_32():
    """(x+y+z+1)**32  -> expand"""
    e.expand()

# 定义一个函数，用于计算复数(2 + 3*I)的1000次幂展开结果，包含虚数计算
def timeit_expand_complex_number_1():
    ((2 + 3*I)**1000).expand(complex=True)

# 定义一个函数，用于计算复数(2 + 3*I/4)的1000次幂展开结果，包含虚数计算
def timeit_expand_complex_number_2():
    ((2 + 3*I/4)**1000).expand(complex=True)
```