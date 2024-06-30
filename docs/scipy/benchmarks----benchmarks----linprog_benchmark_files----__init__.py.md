# `D:\src\scipysrc\scipy\benchmarks\benchmarks\linprog_benchmark_files\__init__.py`

```
"""
==============================================================================
`` --  Problems for testing linear programming routines
==============================================================================

This module provides a comprehensive set of problems for benchmarking linear 
programming routines, that is, scipy.optimize.linprog with method =
'interior-point' or 'simplex'.

"""

"""
All problems are from the Netlib LP Test Problem Set, courtesy of CUTEr
ftp://ftp.numerical.rl.ac.uk/pub/cutest/netlib/netlib.html

Converted from SIF (MPS) format by Matt Haberland
"""

# 导出所有非下划线开头的模块成员名称列表
__all__ = [s for s in dir() if not s.startswith('_')]
```