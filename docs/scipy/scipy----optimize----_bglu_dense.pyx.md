# `D:\src\scipysrc\scipy\scipy\optimize\_bglu_dense.pyx`

```
# Author: Matt Haberland

# 引入Cython模块声明
cimport cython
# 引入NumPy模块
import numpy as np
# 引入Cython化的NumPy模块声明
cimport numpy as np
# 从SciPy线性代数模块中引入相关函数和异常类
from scipy.linalg import (solve, lu_solve, lu_factor, solve_triangular,
                          LinAlgError)
# 从Cython加速的BLAS模块中引入特定函数
from scipy.linalg.cython_blas cimport daxpy, dswap
# 尝试引入进程时间，若不支持则使用时钟时间
try:
    from time import process_time as timer
except ImportError:
    from time import clock as timer

# 初始化NumPy C-API
np.import_array()

# 定义公开接口
__all__ = ['LU', 'BGLU']

# 禁止数组边界检查和负索引检查
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void swap_rows(self, double[:, ::1] H, int i) noexcept:
    """
    交换矩阵H的第i行和下一行；代表矩阵5.10后描述的PI_i矩阵的乘积
    """
    # Python
    # H[[i, i+1]] = H[[i+1, i]]
    # Cython，使用BLAS
    cdef int n = H.shape[1]-i
    cdef int one = 1
    dswap(&n, &H[i, i], &one, &H[i+1, i], &one)

# 禁止数组边界检查和负索引检查
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)  # 不是非常重要
cdef double row_subtract(self, double[:, ::1] H, int i) noexcept:
    """
    将H的第i+1行的第一个非零元素置零，通过减去合适倍数的第i行；代表矩阵5.10的乘积。返回用于存储的因子g。
    """
    cdef double g = H[i+1, i] / H[i, i]

    # Python
    # H[i+1, i:] -= g*H[i, i:]
    # Cython，使用BLAS
    cdef int n = H.shape[1]-i
    cdef double ng = -g
    cdef int one = 1
    daxpy(&n, &ng, &H[i, i], &one, &H[i+1, i], &one)

    return g

# 禁止数组边界检查和负索引检查
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void hess_lu(self, double[:, ::1] H, int i, double[:,::1] ops) noexcept:
    """
    将具有第i列中第一个非零对角元素的Hessenberg矩阵H转换为上三角形，记录初等行操作。即在等式5.9中执行和记录操作。
    """
    cdef int m = H.shape[1]
    cdef int j, k
    cdef double piv1, piv2
    cdef double g
    cdef double swap

    for k in range(i, m-1):
        j = k-i
        piv1, piv2 = abs(H[k, k]), abs(H[k+1, k])  # np.abs(H[k:k+2,k])
        swap = float(piv1 < piv2)
        # 交换行以确保|g| <= 1
        if swap:
            swap_rows(self, H, k)
        g = row_subtract(self, H, k)

        ops[j, 0] = swap
        ops[j, 1] = g

# 禁止数组边界检查和负索引检查
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void perform_ops(self, double[::1] y, const double[:,::1] ops, bint rev=False) noexcept:
    """
    回放将Hessenberg矩阵转换为上三角形形式所需的操作到向量y上。相当于矩阵乘以5.12的逆矩阵。
    """
    cdef int i, j, k, m
    cdef double g, swap

    m = y.shape[0]
    i = m - ops.shape[0] - 1
    if not rev:
        for k in range(i, m-1):
            j = k - i
            swap = ops[j,0]
            g = ops[j, 1]
            if swap:
                y[k], y[k+1] = y[k+1], y[k]

            y[k+1] -= g * y[k]
    else:
        # 从 m-2 开始递减直到 i-1（不包括 i-1），遍历索引 k
        for k in range(m-2, i-1, -1):
            # 计算 j，表示 k 和 i 之间的偏移
            j = k - i
            # 获取操作数组 ops 中对应位置的 swap 和 g 值
            swap = ops[j, 0]
            g = ops[j, 1]
            # 对 y 数组中的元素进行更新操作
            y[k] -= g * y[k+1]
            # 如果 swap 标志为真，则交换 y[k] 和 y[k+1] 的值
            if swap:
                y[k], y[k+1] = y[k+1], y[k]
def _consider_refactor(method):
    """
    This decorator records the time spent in the major BGLU
    routines - refactor, update, and solve - in order to
    calculate the average time required to solve a system.
    It also forces PLU factorization of the basis matrix from
    scratch to minimize the average solve time and to
    accumulation of roundoff error.

    Immediately after PLU factorization, the average solve time
    will be rather high because PLU factorization is slow. For
    some number of factor updates, the average solve time is
    expected to decrease because the updates and solves are fast.
    However, updates increase the complexity of the factorization,
    so solve times are expected to increase with each update.
    When the average solve time stops decreasing and begins
    increasing, we perform PLU factorization from scratch rather
    than updating. PLU factorization is also performed after the
    maximum permitted number of updates is reached to prevent
    further accumulation of roundoff error.
    """
    def f(self, *args, **kwargs):
        # Initialize variables
        refactor_now = False
        out = None

        # Check if method is "update" and increment update counter
        if method.__name__ == "update":
            self.updates += 1

            # Check if average solve time is increasing
            slowing_down = (self.average_solve_times[1] >
                            self.average_solve_times[0])

            # Check if update limit is reached
            too_many_updates = self.updates >= self.max_updates

            # Determine if refactorization is needed based on conditions
            if self.mast:
                refactor_now = (slowing_down or too_many_updates)
            else:
                refactor_now = too_many_updates

            # Perform refactor if necessary
            if refactor_now:
                # Update basis indices and factor from scratch
                self.update_basis(*args, **kwargs)
                out = self.refactor()  # time will be recorded

        # If refactor_now is False, execute the original method
        if not refactor_now:
            # Record the time taken to execute the method
            t0 = timer()
            out = method(self, *args, **kwargs)
            # Check for NaNs in output array
            if isinstance(out, np.ndarray) and np.any(np.isnan(out)):
                raise LinAlgError("Nans in output")
            t1 = timer()
            self.bglu_time += (t1-t0)

        # Calculate average solve time after calling "solve" method
        if method.__name__ == "solve":
            self.solves += 1
            avg = self.bglu_time/self.solves
            self.average_solve_times = [
                self.average_solve_times[1], avg]

        return out
    return f


cdef class LU:
    """
    Represents PLU factorization of a basis matrix with naive rank-one updates
    """

    cdef public np.ndarray A
    cdef public np.ndarray b
    # 声明一个公共的 NumPy 数组 B，用来表示基础矩阵
    cdef public np.ndarray B
    # 声明两个公共整数 m 和 n，分别表示矩阵的行数和列数
    cdef public int m
    cdef public int n

    def __init__(self, A, b):
        """ 给定矩阵 A 和基础索引 b，构建基础矩阵 B """
        # 将输入的矩阵 A 和基础索引 b 分别赋给对象的属性 self.A 和 self.b
        self.A = A
        self.b = b
        # 从矩阵 A 中选取指定列索引 b，形成基础矩阵 B
        self.B = A[:, b]
        # 获取矩阵 A 的形状信息，分别赋给 self.m 和 self.n
        self.m, self.n = A.shape

    def update(self, i, j):
        """ 对基础矩阵进行秩-1更新 """
        # 将基础索引 b 中从 i 开始到 m-1 的元素向前移动一个位置
        self.b[i:self.m-1] = self.b[i+1:self.m]
        # 将基础索引 b 的最后一个位置更新为 j
        self.b[-1] = j
        self.B = self.A[:, self.b]  # 获取基础矩阵 B，从 A 中选择指定列的子集


    def solve(self, q, transposed = False):
        """
        解方程 B @ v = q
        """
        v = solve(self.B, q, transposed=transposed)  # 使用 solve 函数解 B @ v = q 的方程
        return v


cdef class BGLU(LU):
    """
    表示带有 Golub rank-one 更新的 PLU 分解
    """
    cdef public tuple plu
    cdef public np.ndarray L
    cdef public np.ndarray U
    cdef public np.ndarray pi
    cdef public np.ndarray pit
    cdef public list ops_list
    cdef public double bglu_time
    cdef public int solves
    cdef public int updates
    cdef public int max_updates
    cdef public list average_solve_times
    cdef public bint mast

    def __init__(self, A, b, max_updates=10, mast=False):
        """
        给定矩阵 A 和基础索引 b，执行基础矩阵 B 的 PLU 分解
        """
        self.A = A  # 初始化矩阵 A
        self.b = b  # 初始化基础索引 b
        self.m, self.n = A.shape  # 获取矩阵 A 的形状信息
        self.max_updates = max_updates  # 最大重构次数
        self.refactor()  # 执行重构操作
        self.mast = mast  # 是否使用 mast 标志

    @_consider_refactor
    def refactor(self):
        """
        按照方程 5.1 进行分解
        """
        self.B = self.A[:, self.b]  # 获取基础矩阵 B
        self.plu = lu_factor(self.B)  # 使用 lu_factor 执行 PLU 分解
        self.L = self.plu[0]  # 存储 L 矩阵
        self.U = self.plu[0].copy()  # 需要修改但不改变 L 的 U 矩阵
        self.pi = self.perform_perm(self.plu[1])  # 计算排列索引 pi
        self.pit = np.zeros(self.m, dtype=int)  # 初始化置换转置矩阵 pit
        self.pit[self.pi] = np.arange(self.m)  # 构建置换转置矩阵 pit
        self.ops_list = []  # 存储按顺序的初等行操作

        self.bglu_time = 0  # 更新和求解累计耗时
        self.solves = 0     # 重构后解的数量
        self.updates = 0    # 重构后更新的次数
        self.average_solve_times = [np.inf, np.inf]  # 当前和上一个平均求解时间

    def update_basis(self, i, j):
        """
        更新基础矩阵 B 的基础索引
        """
        self.b[i:self.m-1] = self.b[i+1:self.m]  # 从基础索引中删除 i
        self.b[-1] = j  # 将 j 添加到基础索引的末尾

    @_consider_refactor
    def update(self, i, j):
        """ Perform rank-one update to basis and factorization """
        # 调用 update_basis 方法更新基和因式分解

        self.update_basis(i, j)

        # 计算 Hessenberg 矩阵的最后一列
        # TODO: 将此计算与 simplex 方法共享
        pla = self.A[self.pi, j]
        um = solve_triangular(self.L, pla, lower=True,
                              check_finite=False, unit_diagonal=True)
        for ops in self.ops_list:
            perform_ops(self, um, ops)  # 就地修改 um

        # 形成 Hessenberg 矩阵
        H = self.U
        H[:, i:self.m-1] = self.U[:, i+1:self.m]  # 消除第 i 列
        H[:, -1] = um  # 添加与 j 对应的列

        # 将 H 转换为上三角矩阵，并记录初等行操作
        ops = np.zeros((self.m-1-i, 2))
        hess_lu(self, H, i, ops)  # hess_lu 就地修改 ops
        self.ops_list.append(ops)

        self.U = H

    @_consider_refactor
    def solve(self, q, transposed=False):
        """
        Solve B @ v = q efficiently using factorization
        """
        if not self.ops_list:
            # 在任何更新之前，根据方程式 5.2 解决问题
            v = lu_solve(self.plu, q, trans=transposed)
        else:
            if not transposed:
                q = q[self.pi]  # 论文通过假设“非必要假设”跳过此步骤

                # 方程式 5.16
                t = solve_triangular(self.L, q, lower=True,
                                     check_finite=False, unit_diagonal=True)

                # 方程式 5.17
                temp = t
                for ops in self.ops_list:
                    perform_ops(self, temp, ops)  # 就地修改 temp
                w = temp

                # 方程式 5.18
                # 由于数组顺序，使用 U.T 并设置 trans=True 更快
                v = solve_triangular(self.U.T, w, lower=True,
                                     trans=True, check_finite=False)

            else:  # 执行转置并按相反顺序进行所有操作
                t = solve_triangular(self.U.T, q, lower=True,
                                     trans=False, check_finite=False)
                temp = t
                for ops in reversed(self.ops_list):
                    perform_ops(self, temp, ops, rev=True)  # 就地修改 temp
                w = temp
                v = solve_triangular(self.L, w, lower=True, trans=True,
                                     check_finite=False, unit_diagonal=True)
                v = v[self.pit]

        return v

    def perform_perm(self, p):
        """
        Perform individual row swaps defined in p returned by factor_lu to
        generate final permutation indices pi
        """
        pi = np.arange(len(p))
        for i, row in enumerate(p):
            pi[i], pi[row] = pi[row], pi[i]
        return pi
```