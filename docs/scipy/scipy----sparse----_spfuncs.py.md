# `D:\src\scipysrc\scipy\scipy\sparse\_spfuncs.py`

```
""" Functions that operate on sparse matrices
"""

__all__ = ['count_blocks','estimate_blocksize']

from ._base import issparse
from ._csr import csr_array
from ._sparsetools import csr_count_blocks


def estimate_blocksize(A,efficiency=0.7):
    """Attempt to determine the blocksize of a sparse matrix

    Returns a blocksize=(r,c) such that
        - A.nnz / A.tobsr( (r,c) ).nnz > efficiency
    """
    # 如果输入的 A 不是稀疏矩阵或者格式不是 "csc" 或 "csr"，则转换为 csr_array
    if not (issparse(A) and A.format in ("csc", "csr")):
        A = csr_array(A)

    # 如果 A 的非零元素个数为 0，则返回最小的块大小 (1,1)
    if A.nnz == 0:
        return (1,1)

    # 如果 efficiency 不在 (0.0, 1.0) 范围内，抛出 ValueError 异常
    if not 0 < efficiency < 1.0:
        raise ValueError('efficiency must satisfy 0.0 < efficiency < 1.0')

    # 计算高效性的阈值
    high_efficiency = (1.0 + efficiency) / 2.0
    nnz = float(A.nnz)
    M,N = A.shape

    # 根据 M 和 N 的奇偶性判断是否尝试使用 (2,2) 块大小
    if M % 2 == 0 and N % 2 == 0:
        e22 = nnz / (4 * count_blocks(A,(2,2)))
    else:
        e22 = 0.0

    # 根据 M 和 N 的奇偶性判断是否尝试使用 (3,3) 块大小
    if M % 3 == 0 and N % 3 == 0:
        e33 = nnz / (9 * count_blocks(A,(3,3)))
    else:
        e33 = 0.0

    # 如果 e22 和 e33 都大于高效性阈值，则尝试使用 (6,6) 块大小
    if e22 > high_efficiency and e33 > high_efficiency:
        e66 = nnz / (36 * count_blocks(A,(6,6)))
        if e66 > efficiency:
            return (6,6)
        else:
            return (3,3)
    else:
        # 如果不满足以上条件，则根据 M 和 N 的奇偶性尝试使用 (4,4) 块大小
        if M % 4 == 0 and N % 4 == 0:
            e44 = nnz / (16 * count_blocks(A,(4,4)))
        else:
            e44 = 0.0

        # 根据效率判断返回合适的块大小
        if e44 > efficiency:
            return (4,4)
        elif e33 > efficiency:
            return (3,3)
        elif e22 > efficiency:
            return (2,2)
        else:
            return (1,1)


def count_blocks(A,blocksize):
    """For a given blocksize=(r,c) count the number of occupied
    blocks in a sparse matrix A
    """
    # 解析块大小
    r,c = blocksize
    # 如果 r 或 c 小于等于 0，则抛出 ValueError 异常
    if r < 1 or c < 1:
        raise ValueError('r and c must be positive')

    # 如果 A 是稀疏矩阵
    if issparse(A):
        # 如果 A 的格式为 "csr"
        if A.format == "csr":
            # 获取 A 的形状
            M,N = A.shape
            # 调用 csr_count_blocks 函数计算使用给定块大小 (r,c) 的占用块数
            return csr_count_blocks(M,N,r,c,A.indptr,A.indices)
        # 如果 A 的格式为 "csc"
        elif A.format == "csc":
            # 递归调用 count_blocks 函数，转置 A 并交换块大小的行列顺序
            return count_blocks(A.T,(c,r))
    # 如果 A 不是稀疏矩阵，则将其转换为 csr_array 后再次调用 count_blocks 函数
    return count_blocks(csr_array(A),blocksize)
```