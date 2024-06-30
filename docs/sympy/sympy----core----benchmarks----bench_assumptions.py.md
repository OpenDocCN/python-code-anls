# `D:\src\scipysrc\sympy\sympy\core\benchmarks\bench_assumptions.py`

```
# 从 sympy.core 模块中导入 Symbol 和 Integer 类
from sympy.core import Symbol, Integer

# 创建一个符号变量 x
x = Symbol('x')

# 创建一个整数对象 i3，其值为 3
i3 = Integer(3)

# 定义一个函数 timeit_x_is_integer，用于测试 x 是否为整数
def timeit_x_is_integer():
    # 访问 x 对象的 is_integer 方法，但此处未调用该方法
    x.is_integer

# 定义一个函数 timeit_Integer_is_irrational，用于测试 i3 是否为无理数
def timeit_Integer_is_irrational():
    # 访问 i3 对象的 is_irrational 方法，但此处未调用该方法
    i3.is_irrational
```