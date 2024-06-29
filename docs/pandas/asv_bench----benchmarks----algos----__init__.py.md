# `D:\src\scipysrc\pandas\asv_bench\benchmarks\algos\__init__.py`

```
"""
algos/ directory is intended for individual functions from core.algorithms

In many cases these algorithms are reachable in multiple ways:
   algos.foo(x, y)
   Series(x).foo(y)
   Index(x).foo(y)
   pd.array(x).foo(y)

In most cases we profile the Series variant directly, trusting the performance
of the others to be highly correlated.
"""



"""
algos/ 目录用于存放来自 core.algorithms 的单个函数

在许多情况下，这些算法可以通过多种方式访问：
   algos.foo(x, y)
   Series(x).foo(y)
   Index(x).foo(y)
   pd.array(x).foo(y)

在大多数情况下，我们直接对 Series 变体进行性能分析，相信其他方式的性能高度相关。
"""
```