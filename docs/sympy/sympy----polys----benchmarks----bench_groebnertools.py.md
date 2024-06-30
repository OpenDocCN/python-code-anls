# `D:\src\scipysrc\sympy\sympy\polys\benchmarks\bench_groebnertools.py`

```
"""Benchmark of the Groebner bases algorithms. """

# 导入所需模块和函数
from sympy.polys.rings import ring
from sympy.polys.domains import QQ
from sympy.polys.groebnertools import groebner

# 创建一个多项式环 R，包括 12 个变量 x1 到 x12，并使用有理数域 QQ
R, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = ring("x1:13", QQ)

# 将变量列表 V 设置为 R 的生成元
V = R.gens

# 定义边集 E，包含多个有序对，表示图的边
E = [(x1, x2), (x2, x3), (x1, x4), (x1, x6), (x1, x12), (x2, x5), (x2, x7), (x3, x8),
     (x3, x10), (x4, x11), (x4, x9), (x5, x6), (x6, x7), (x7, x8), (x8, x9), (x9, x10),
     (x10, x11), (x11, x12), (x5, x12), (x5, x9), (x6, x10), (x7, x11), (x8, x12)]

# 定义 F3，包含 x1 到 x12 的立方减一的多项式列表
F3 = [ x**3 - 1 for x in V ]

# 定义 Fg，包含 x1 到 x12 之间的二次多项式列表
Fg = [ x**2 + x*y + y**2 for x, y in E ]

# F_1 是 F3 和 Fg 的联合列表
F_1 = F3 + Fg

# F_2 是 F3、Fg 和额外的 x3**2 + x3*x4 + x4**2 多项式的联合列表
F_2 = F3 + Fg + [x3**2 + x3*x4 + x4**2]

# 定义性能测试函数 time_vertex_color_12_vertices_23_edges
def time_vertex_color_12_vertices_23_edges():
    # 断言使用 F_1 和环 R 的 Groebner 基不等于 [1]
    assert groebner(F_1, R) != [1]

# 定义性能测试函数 time_vertex_color_12_vertices_24_edges
def time_vertex_color_12_vertices_24_edges():
    # 断言使用 F_2 和环 R 的 Groebner 基等于 [1]
    assert groebner(F_2, R) == [1]
```