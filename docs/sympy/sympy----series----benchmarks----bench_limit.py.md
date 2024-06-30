# `D:\src\scipysrc\sympy\sympy\series\benchmarks\bench_limit.py`

```
# 从 sympy.core.numbers 模块中导入 oo（无穷大）常量
# 从 sympy.core.symbol 模块中导入 Symbol 类，用于创建符号变量
# 从 sympy.series.limits 模块中导入 limit 函数，用于计算极限

from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.series.limits import limit

# 创建一个符号变量 x
x = Symbol('x')

# 定义一个函数 timeit_limit_1x，用于计算表达式 1/x 的极限当 x 趋向无穷大时
def timeit_limit_1x():
    limit(1/x, x, oo)
```