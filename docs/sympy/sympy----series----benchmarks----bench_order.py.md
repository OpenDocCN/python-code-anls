# `D:\src\scipysrc\sympy\sympy\series\benchmarks\bench_order.py`

```
# 从 sympy.core.add 模块中导入 Add 类
# 从 sympy.core.symbol 模块中导入 Symbol 类
# 从 sympy.series.order 模块中导入 O 函数

from sympy.core.add import Add
from sympy.core.symbol import Symbol
from sympy.series.order import O

# 创建一个符号 x
x = Symbol('x')

# 使用列表推导式生成一个包含 x 的各次幂的列表，范围是 0 到 999
l = [x**i for i in range(1000)]

# 将 O(x**1001) 添加到列表 l 中
l.append(O(x**1001))

# 定义一个函数 timeit_order_1x，其目的是对列表 l 中的所有表达式进行加法操作
def timeit_order_1x():
    Add(*l)
```