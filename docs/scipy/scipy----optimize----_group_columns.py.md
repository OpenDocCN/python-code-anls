# `D:\src\scipysrc\scipy\scipy\optimize\_group_columns.py`

```
"""
Pythran implementation of columns grouping for finite difference Jacobian
estimation. Used by ._numdiff.group_columns and based on the Cython version.
"""

import numpy as np

# 定义函数，导出到 Pythran
# 输入参数：m - 行数, n - 列数, A - 稠密矩阵
# 返回：分组结果，每列对应的分组编号
#pythran export group_dense(int, int, intc[:,:])
#pythran export group_dense(int, int, int[:,:])
def group_dense(m, n, A):
    # 转置矩阵方便处理
    B = A.T  # Transposed view for convenience.

    # 用于存储每列的分组编号，初始值为 -1 表示未分组
    groups = -np.ones(n, dtype=np.intp)
    current_group = 0

    # 用于存储并集的向量
    union = np.empty(m, dtype=np.intp)

    # 遍历所有列
    for i in range(n):
        if groups[i] >= 0:  # 如果列已经分组
            continue

        groups[i] = current_group
        all_grouped = True

        # 存储并集到 union 中
        union[:] = B[i]  # Here we store the union of grouped columns.

        # 检查其余列是否与当前列有交集
        for j in range(groups.shape[0]):
            if groups[j] < 0:
                all_grouped = False
            else:
                continue

            # 确定第 j 列是否与并集有交集
            intersect = False
            for k in range(m):
                if union[k] > 0 and B[j, k] > 0:
                    intersect = True
                    break

            # 如果没有交集，则将第 j 列添加到并集中，并分配相同的分组编号
            if not intersect:
                union += B[j]
                groups[j] = current_group

        if all_grouped:
            break

        current_group += 1

    return groups


# 定义函数，导出到 Pythran
# 输入参数：m - 行数, n - 列数, indices - 稀疏矩阵的索引数组, indptr - 稀疏矩阵的指针数组
# 返回：分组结果，每列对应的分组编号
#pythran export group_sparse(int, int, int32[], int32[])
#pythran export group_sparse(int, int, int64[], int64[])
#pythran export group_sparse(int, int, int32[::], int32[::])
#pythran export group_sparse(int, int, int64[::], int64[::])
def group_sparse(m, n, indices, indptr):
    # 用于存储每列的分组编号，初始值为 -1 表示未分组
    groups = -np.ones(n, dtype=np.intp)
    current_group = 0

    # 用于存储并集的向量
    union = np.empty(m, dtype=np.intp)

    # 遍历所有列
    for i in range(n):
        if groups[i] >= 0:
            continue

        groups[i] = current_group
        all_grouped = True

        # 清空并集向量
        union.fill(0)
        
        # 将当前列涉及的行索引添加到并集中
        for k in range(indptr[i], indptr[i + 1]):
            union[indices[k]] = 1

        # 检查其余列是否与当前列有交集
        for j in range(groups.shape[0]):
            if groups[j] < 0:
                all_grouped = False
            else:
                continue

            intersect = False
            for k in range(indptr[j], indptr[j + 1]):
                if union[indices[k]] == 1:
                    intersect = True
                    break
            
            # 如果没有交集，则将第 j 列涉及的行索引添加到并集中，并分配相同的分组编号
            if not intersect:
                for k in range(indptr[j], indptr[j + 1]):
                    union[indices[k]] = 1
                groups[j] = current_group

        if all_grouped:
            break

        current_group += 1

    return groups
```