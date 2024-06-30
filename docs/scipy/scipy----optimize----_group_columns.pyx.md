# `D:\src\scipysrc\scipy\scipy\optimize\_group_columns.pyx`

```
"""
Cython implementation of columns grouping for finite difference Jacobian
estimation. Used by ._numdiff.group_columns.
"""

# 导入必要的Cython模块
cimport cython

# 导入NumPy模块
import numpy as np

# 导入NumPy C API，用于底层操作
cimport numpy as np

# 允许使用NumPy数组的C API
np.import_array()

# 禁用边界检查以及负索引的包装
@cython.boundscheck(False)
@cython.wraparound(False)
def group_dense(int m, int n, const int[:, :] A):
    cdef const int [:, :] B = A.T  # 为方便起见，创建A的转置视图B

    # 初始化一个大小为n的整数数组，用于存储每列的分组信息，默认为-1
    cdef int [:] groups = np.full(n, -1, dtype=np.int32)
    # 当前分组编号
    cdef int current_group = 0

    cdef int i, j, k

    # 创建一个大小为m的整数数组，用于存储并集
    union = np.empty(m, dtype=np.int32)
    cdef int [:] union_v = union

    # 遍历所有列
    for i in range(n):
        if groups[i] >= 0:  # 如果已分配了分组
            continue

        groups[i] = current_group
        all_grouped = True

        # 存储已分组列的并集
        union_v[:] = B[i]

        # 遍历剩余的列
        for j in range(groups.shape[0]):
            if groups[j] < 0:
                all_grouped = False
            else:
                continue

            # 确定第j列是否与并集有交集
            intersect = False
            for k in range(m):
                if union_v[k] > 0 and B[j, k] > 0:
                    intersect = True
                    break

            # 如果没有交集，则将第j列添加到并集并分配相同的分组
            if not intersect:
                union += B[j]
                groups[j] = current_group

        # 如果所有列均已分组，提前退出循环
        if all_grouped:
            break

        current_group += 1

    return groups.base


# 禁用负索引的包装
@cython.wraparound(False)
def group_sparse(int m, int n, const int[:] indices, const int[:] indptr):
    # 初始化一个大小为n的整数数组，用于存储每列的分组信息，默认为-1
    cdef int [:] groups = np.full(n, -1, dtype=np.int32)
    # 当前分组编号
    cdef int current_group = 0

    cdef int i, j, k

    # 创建一个大小为m的整数数组，用于存储并集
    union = np.empty(m, dtype=np.int32)
    cdef int [:] union_v = union

    # 遍历所有列
    for i in range(n):
        if groups[i] >= 0:
            continue

        groups[i] = current_group
        all_grouped = True

        # 清空并集数组
        union.fill(0)
        # 将稀疏矩阵中指定行的列索引加入并集
        for k in range(indptr[i], indptr[i + 1]):
            union_v[indices[k]] = 1

        # 再次遍历所有列
        for j in range(groups.shape[0]):
            if groups[j] < 0:
                all_grouped = False
            else:
                continue

            intersect = False
            # 检查第j列是否与并集有交集
            for k in range(indptr[j], indptr[j + 1]):
                if union_v[indices[k]] == 1:
                    intersect = True
                    break
            # 如果没有交集，则将第j列添加到并集并分配相同的分组
            if not intersect:
                for k in range(indptr[j], indptr[j + 1]):
                    union_v[indices[k]] = 1
                groups[j] = current_group

        # 如果所有列均已分组，提前退出循环
        if all_grouped:
            break

        current_group += 1

    return groups.base
```