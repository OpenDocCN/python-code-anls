# `D:\src\scipysrc\scipy\scipy\linalg\_matfuncs_sqrtm.py`

```
"""
Matrix square root for general matrices and for upper triangular matrices.

This module exists to avoid cyclic imports.

"""
__all__ = ['sqrtm']  # 声明模块中公开的接口只有 sqrtm 函数

import numpy as np  # 导入 NumPy 库

from scipy._lib._util import _asarray_validated  # 导入辅助函数

# Local imports
from ._misc import norm  # 导入本地模块中的 norm 函数
from .lapack import ztrsyl, dtrsyl  # 导入 lapack 模块中的 ztrsyl 和 dtrsyl 函数
from ._decomp_schur import schur, rsf2csf  # 导入本地模块中的 schur 和 rsf2csf 函数


class SqrtmError(np.linalg.LinAlgError):
    pass


from ._matfuncs_sqrtm_triu import within_block_loop  # 导入本地模块中的 within_block_loop 函数 (禁止 E402 错误)


def _sqrtm_triu(T, blocksize=64):
    """
    Matrix square root of an upper triangular matrix.

    This is a helper function for `sqrtm` and `logm`.

    Parameters
    ----------
    T : (N, N) array_like upper triangular
        Matrix whose square root to evaluate
    blocksize : int, optional
        If the blocksize is not degenerate with respect to the
        size of the input array, then use a blocked algorithm. (Default: 64)

    Returns
    -------
    sqrtm : (N, N) ndarray
        Value of the sqrt function at `T`

    References
    ----------
    .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)
           "Blocked Schur Algorithms for Computing the Matrix Square Root,
           Lecture Notes in Computer Science, 7782. pp. 171-182.

    """
    T_diag = np.diag(T)  # 提取 T 的对角线元素
    keep_it_real = np.isrealobj(T) and np.min(T_diag, initial=0.) >= 0  # 检查 T 是否为实数且其对角元素非负

    # Cast to complex as necessary + ensure double precision
    if not keep_it_real:
        T = np.asarray(T, dtype=np.complex128, order="C")  # 将 T 转换为复数类型（complex128）
        T_diag = np.asarray(T_diag, dtype=np.complex128)  # 将对角元素转换为复数类型
    else:
        T = np.asarray(T, dtype=np.float64, order="C")  # 将 T 转换为双精度浮点数类型（float64）
        T_diag = np.asarray(T_diag, dtype=np.float64)  # 将对角元素转换为双精度浮点数类型

    R = np.diag(np.sqrt(T_diag))  # 计算对角线元素的平方根并构造对角矩阵 R

    # Compute the number of blocks to use; use at least one block.
    n, n = T.shape  # 获取矩阵 T 的维度
    nblocks = max(n // blocksize, 1)  # 计算需要使用的块数，至少使用一个块

    # Compute the smaller of the two sizes of blocks that
    # we will actually use, and compute the number of large blocks.
    bsmall, nlarge = divmod(n, nblocks)  # 计算块的大小及其数量
    blarge = bsmall + 1
    nsmall = nblocks - nlarge
    if nsmall * bsmall + nlarge * blarge != n:
        raise Exception('internal inconsistency')  # 如果块的划分存在内部不一致性则抛出异常

    # Define the index range covered by each block.
    start_stop_pairs = []  # 存储每个块所覆盖的索引范围
    start = 0
    for count, size in ((nsmall, bsmall), (nlarge, blarge)):
        for i in range(count):
            start_stop_pairs.append((start, start + size))
            start += size

    # Within-block interactions (Cythonized)
    try:
        within_block_loop(R, T, start_stop_pairs, nblocks)  # 调用 Cython 加速的 within_block_loop 函数处理块内交互
    except RuntimeError as e:
        raise SqrtmError(*e.args) from e  # 捕获运行时异常并抛出自定义的 SqrtmError 异常

    # Between-block interactions (Cython would give no significant speedup)
    # 对每个块进行循环，j 从 0 到 nblocks-1
    for j in range(nblocks):
        # 获取当前块 j 的起始和结束索引
        jstart, jstop = start_stop_pairs[j]
        
        # 对当前块 j 之前的每个块进行逆序循环，i 从 j-1 到 0
        for i in range(j-1, -1, -1):
            # 获取块 i 的起始和结束索引
            istart, istop = start_stop_pairs[i]
            
            # 提取子矩阵 S，从 T 中切片取出
            S = T[istart:istop, jstart:jstop]
            
            # 如果 j - i 大于 1，则执行以下操作
            if j - i > 1:
                # 计算更新 S，减去 R 的乘积结果
                S = S - R[istart:istop, istop:jstart].dot(R[istop:jstart, jstart:jstop])

            # 调用 LAPACK 函数进行求解
            # 如果 keep_it_real 为 True，使用 dtrsyl 函数；否则使用 ztrsyl 函数
            Rii = R[istart:istop, istart:istop]
            Rjj = R[jstart:jstop, jstart:jstop]
            if keep_it_real:
                x, scale, info = dtrsyl(Rii, Rjj, S)
            else:
                x, scale, info = ztrsyl(Rii, Rjj, S)
            
            # 更新 R 矩阵的子块 R[istart:istop, jstart:jstop]
            R[istart:istop, jstart:jstop] = x * scale

    # 返回矩阵的平方根 R
    return R
# 计算输入矩阵的字节大小
byte_size = np.asarray(A).dtype.itemsize
# 将输入矩阵 A 转换为确保其是一个浮点数类型的数组，并检查是否有无穷大或 NaN 值
A = _asarray_validated(A, check_finite=True, as_inexact=True)
# 如果输入不是二维矩阵，则抛出异常
if len(A.shape) != 2:
    raise ValueError("Non-matrix input to matrix function.")
# 如果块大小小于1，则抛出异常
if blocksize < 1:
    raise ValueError("The blocksize should be at least 1.")
# 检查输入矩阵 A 是否是实数类型
keep_it_real = np.isrealobj(A)
# 如果是实数类型，则对 A 进行 Schur 分解
if keep_it_real:
    T, Z = schur(A)
    # 获取 Schur 分解后的主对角线元素和次对角线元素
    d0 = np.diagonal(T)
    d1 = np.diagonal(T, -1)
    # 计算浮点数的机器精度
    eps = np.finfo(T.dtype).eps
    # 判断是否需要将实数 Schur 形式转换为复数 Schur 形式
    needs_conversion = abs(d1) > eps * (abs(d0[1:]) + abs(d0[:-1]))
    if needs_conversion.any():
        T, Z = rsf2csf(T, Z)
# 如果输入是复数类型，则直接对 A 进行复数 Schur 分解
else:
    T, Z = schur(A, output='complex')
# 初始化失败标志位为 False
failflag = False
try:
    # 计算上三角部分的平方根矩阵 R
    R = _sqrtm_triu(T, blocksize=blocksize)
    # 计算 Z 的共轭转置
    ZH = np.conjugate(Z).T
    # 计算矩阵平方根近似值 X = Z * R * ZH
    X = Z.dot(R).dot(ZH)
    # 如果 X 不是复数类型，则根据字节大小范围选择浮点数类型
    if not np.iscomplexobj(X):
        X = X.astype(f"f{np.clip(byte_size, 2, 16)}", copy=False)
    else:
        # 如果 X 是复数类型，则根据字节大小范围选择复数类型
        if hasattr(np, 'complex256'):
            X = X.astype(f"c{np.clip(byte_size*2, 8, 32)}", copy=False)
        else:
            X = X.astype(f"c{np.clip(byte_size*2, 8, 16)}", copy=False)
# 如果计算失败，则捕获 SqrtmError 异常，设置失败标志位为 True，并创建一个与 A 类型相同的空数组 X
except SqrtmError:
    failflag = True
    X = np.empty_like(A)
    X.fill(np.nan)
    # 如果 disp 参数为真，则执行以下代码块
    if disp:
        # 如果 failflag 标志为真，则打印消息表明未找到平方根
        if failflag:
            print("Failed to find a square root.")
        # 返回变量 X 的值并结束函数
        return X
    # 如果 disp 参数为假，则执行以下代码块
    else:
        try:
            # 计算参数 arg2，它是 X 矩阵的 Frobenius 范数平方与 A 矩阵 Frobenius 范数之比
            arg2 = norm(X.dot(X) - A, 'fro')**2 / norm(A, 'fro')
        except ValueError:
            # 如果出现 ValueError 异常，说明矩阵中有 NaN 值
            # 将 arg2 设置为无穷大（np.inf）
            arg2 = np.inf

        # 返回变量 X 和计算得到的 arg2 的值作为元组，并结束函数
        return X, arg2
```