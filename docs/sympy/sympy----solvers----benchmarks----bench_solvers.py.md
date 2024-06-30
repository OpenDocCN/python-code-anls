# `D:\src\scipysrc\sympy\sympy\solvers\benchmarks\bench_solvers.py`

```
# 从 sympy 库中导入符号和矩阵相关模块
from sympy.core.symbol import Symbol
from sympy.matrices.dense import (eye, zeros)
# 从 sympy 中导入线性求解器
from sympy.solvers.solvers import solve_linear_system

# 定义一个常数 N，表示矩阵的维度
N = 8
# 创建一个 N 行 (N+1) 列的零矩阵 M
M = zeros(N, N + 1)
# 将 M 的前 N 列设为单位矩阵
M[:, :N] = eye(N)
# 创建一个包含 N 个符号变量的列表 S，变量名形如 'A0', 'A1', ..., 'A7'
S = [Symbol('A%i' % i) for i in range(N)]

# 定义一个函数 timeit_linsolve_trivial，用于解线性方程组
def timeit_linsolve_trivial():
    # 调用 sympy 中的线性求解器，解决由 M 和 S 表示的线性方程组
    solve_linear_system(M, *S)
```