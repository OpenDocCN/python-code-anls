# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_pairwise_fast.pyx`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 从 Cython 模块中导入需要的类型和函数
from cython cimport floating
from cython.parallel cimport prange
from libc.math cimport fabs

# 从自定义的 utils 模块中导入特定类型定义
from ..utils._typedefs cimport intp_t

# 从自定义的 utils 模块中导入 OpenMP 相关函数
from ..utils._openmp_helpers import _openmp_effective_n_threads

# 定义一个快速的 Chi-squared 核函数，用于计算两个矩阵之间的相似度
def _chi2_kernel_fast(floating[:, :] X,
                      floating[:, :] Y,
                      floating[:, :] result):
    # 定义循环变量
    cdef intp_t i, j, k
    # 获取样本数量和特征数量
    cdef intp_t n_samples_X = X.shape[0]
    cdef intp_t n_samples_Y = Y.shape[0]
    cdef intp_t n_features = X.shape[1]
    # 定义结果、分子和分母变量
    cdef double res, nom, denom

    # 使用 nogil 上下文以释放全局解释器锁（GIL），提高并行性能
    with nogil:
        # 循环遍历样本 X 和 Y
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                # 初始化结果
                res = 0
                # 遍历特征维度
                for k in range(n_features):
                    # 计算分母和分子
                    denom = (X[i, k] - Y[j, k])
                    nom = (X[i, k] + Y[j, k])
                    # 避免除以零，计算 Chi-squared 核函数值
                    if nom != 0:
                        res += denom * denom / nom
                # 存储结果到 result 矩阵中
                result[i, j] = -res


# 定义稀疏矩阵之间的曼哈顿距离函数
def _sparse_manhattan(
    const floating[::1] X_data,
    const int[:] X_indices,
    const int[:] X_indptr,
    const floating[::1] Y_data,
    const int[:] Y_indices,
    const int[:] Y_indptr,
    double[:, ::1] D,
):
    """Pairwise L1 distances for CSR matrices.

    Usage:
    >>> D = np.zeros(X.shape[0], Y.shape[0])
    >>> _sparse_manhattan(X.data, X.indices, X.indptr,
    ...                   Y.data, Y.indices, Y.indptr,
    ...                   D)
    """
    # 定义循环变量和临时距离变量
    cdef intp_t px, py, i, j, ix, iy
    cdef double d = 0.0

    # 获取矩阵 D 的维度信息
    cdef int m = D.shape[0]
    cdef int n = D.shape[1]

    # 初始化 X 和 Y 矩阵的索引结束位置
    cdef int X_indptr_end = 0
    cdef int Y_indptr_end = 0

    # 获取 OpenMP 的有效线程数
    cdef int num_threads = _openmp_effective_n_threads()

    # 我们逐行扫描矩阵。
    # 对于 X 中的行 px 和 Y 中的行 py，我们找到它们在 .indices 中开始的位置（i 和 j）。
    # 如果索引（ix 和 iy）相同，则处理对应的数据值，并且递增游标 i 和 j。
    # 如果不相同，则考虑最低的索引。处理它相关的数据值，并递增相应的游标。
    # 我们持续进行此操作，直到其中一个游标达到其行的末尾。
    # 然后，我们处理另一行中所有剩余的数据值。

    # 下面的避免使用原地操作符是有意的。
    # 在使用 prange 时，原地操作符具有特殊含义，即它表示一个“reduction”。
    # 并行循环，使用prange函数并指定无GIL（全局解释器锁）模式和线程数
    for px in prange(m, nogil=True, num_threads=num_threads):
        # 获取X_indptr数组中当前行(px)和下一行(px+1)的索引范围
        X_indptr_end = X_indptr[px + 1]
        # 遍历Y的每一列(py)
        for py in range(n):
            # 获取Y_indptr数组中当前列(py)和下一列(py+1)的索引范围
            Y_indptr_end = Y_indptr[py + 1]
            # 初始化i和j为当前行和列的起始索引
            i = X_indptr[px]
            j = Y_indptr[py]
            # 初始化距离度量为0
            d = 0.0
            # 当i和j都在其索引范围内时，执行循环
            while i < X_indptr_end and j < Y_indptr_end:
                # 获取当前行和列的索引值
                ix = X_indices[i]
                iy = Y_indices[j]

                # 如果索引相等，计算数据差的绝对值并累加到距离度量
                if ix == iy:
                    d = d + fabs(X_data[i] - Y_data[j])
                    i = i + 1
                    j = j + 1
                # 如果当前行索引小于当前列索引，计算当前行数据的绝对值并累加
                elif ix < iy:
                    d = d + fabs(X_data[i])
                    i = i + 1
                # 如果当前列索引小于当前行索引，计算当前列数据的绝对值并累加
                else:
                    d = d + fabs(Y_data[j])
                    j = j + 1

            # 如果当前行索引达到了行的结束索引，继续累加当前列剩余数据的绝对值
            if i == X_indptr_end:
                while j < Y_indptr_end:
                    d = d + fabs(Y_data[j])
                    j = j + 1
            # 否则，累加当前行剩余数据的绝对值
            else:
                while i < X_indptr_end:
                    d = d + fabs(X_data[i])
                    i = i + 1

            # 将计算得到的距离度量赋值给D矩阵的对应位置(px, py)
            D[px, py] = d
```