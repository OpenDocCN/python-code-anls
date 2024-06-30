# `D:\src\scipysrc\sympy\sympy\core\benchmarks\bench_arit.py`

```
# 导入 sympy 库中的 Add、Mul 和 symbols 函数
from sympy.core import Add, Mul, symbols

# 定义符号变量 x, y, z
x, y, z = symbols('x,y,z')

# 定义一个函数，计时负数操作 -x
def timeit_neg():
    -x

# 定义一个函数，计时加法操作 x + 1
def timeit_Add_x1():
    x + 1

# 定义一个函数，计时加法操作 1 + x
def timeit_Add_1x():
    1 + x

# 定义一个函数，计时加法操作 x + 0.5
def timeit_Add_x05():
    x + 0.5

# 定义一个函数，计时加法操作 x + y
def timeit_Add_xy():
    x + y

# 定义一个函数，计时加法操作 Add(x, y, z)
def timeit_Add_xyz():
    Add(*[x, y, z])

# 定义一个函数，计时乘法操作 x * y
def timeit_Mul_xy():
    x*y

# 定义一个函数，计时乘法操作 Mul(x, y, z)
def timeit_Mul_xyz():
    Mul(*[x, y, z])

# 定义一个函数，计时除法操作 x / y
def timeit_Div_xy():
    x/y

# 定义一个函数，计时除法操作 2 / y
def timeit_Div_2y():
    2/y
```