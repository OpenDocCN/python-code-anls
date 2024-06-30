# `D:\src\scipysrc\scipy\scipy\linalg\_decomp_lu.py`

```
# 导入警告模块中的 warn 函数
from warnings import warn
# 导入 numpy 库，并将其别名为 np
from numpy import asarray, asarray_chkfinite
# 导入 numpy 库，并将其别名为 np
import numpy as np
# 导入 itertools 库中的 product 函数
from itertools import product

# 导入本地模块
# 从 _misc 模块中导入 _datacopied 和 LinAlgWarning
from ._misc import _datacopied, LinAlgWarning
# 从 lapack 模块中导入 get_lapack_funcs 函数
from .lapack import get_lapack_funcs
# 从 _decomp_lu_cython 模块中导入 lu_dispatcher 函数
from ._decomp_lu_cython import lu_dispatcher

# 创建一个字典，用于将 numpy 类型码映射到能够在 LAPACK 中使用的类型
lapack_cast_dict = {x: ''.join([y for y in 'fdFD' if np.can_cast(x, y)])
                    for x in np.typecodes['All']}

# 定义模块的公开接口，即外部可以访问的函数名列表
__all__ = ['lu', 'lu_solve', 'lu_factor']

# 定义 lu_factor 函数，用于计算矩阵的 LU 分解
def lu_factor(a, overwrite_a=False, check_finite=True):
    """
    Compute pivoted LU decomposition of a matrix.

    The decomposition is::

        A = P L U

    where P is a permutation matrix, L lower triangular with unit
    diagonal elements, and U upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to decompose
    overwrite_a : bool, optional
        Whether to overwrite data in A (may increase performance)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    lu : (M, N) ndarray
        Matrix containing U in its upper triangle, and L in its lower triangle.
        The unit diagonal elements of L are not stored.
    piv : (K,) ndarray
        Pivot indices representing the permutation matrix P:
        row i of matrix was interchanged with row piv[i].
        Of shape ``(K,)``, with ``K = min(M, N)``.

    See Also
    --------
    lu : gives lu factorization in more user-friendly format
    lu_solve : solve an equation system using the LU factorization of a matrix

    Notes
    -----
    This is a wrapper to the ``*GETRF`` routines from LAPACK. Unlike
    :func:`lu`, it outputs the L and U factors into a single array
    and returns pivot indices instead of a permutation matrix.

    While the underlying ``*GETRF`` routines return 1-based pivot indices, the
    ``piv`` array returned by ``lu_factor`` contains 0-based indices.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import lu_factor
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> lu, piv = lu_factor(A)
    >>> piv
    array([2, 2, 3, 3], dtype=int32)

    Convert LAPACK's ``piv`` array to NumPy index and test the permutation

    >>> def pivot_to_permutation(piv):
    ...     perm = np.arange(len(piv))
    ...     for i in range(len(piv)):
    ...         perm[i], perm[piv[i]] = perm[piv[i]], perm[i]
    ...     return perm
    ...
    >>> p_inv = pivot_to_permutation(piv)
    >>> p_inv
    array([2, 0, 3, 1])
    >>> L, U = np.tril(lu, k=-1) + np.eye(4), np.triu(lu)
    >>> np.allclose(A[p_inv] - L @ U, np.zeros((4, 4)))
    True

    The P matrix in P L U is defined by the inverse permutation and
    can be recovered using argsort:

    >>> p = np.argsort(p_inv)

    """
    # 函数文档字符串，描述了该函数的作用及其输入输出
    # 调用 LAPACK 的 *GETRF 子程序进行 LU 分解
    # 返回 LU 分解结果 lu 和置换矩阵的索引 piv
    pass
    >>> p
    array([1, 3, 0, 2])

这行代码展示了一个示例，变量 `p` 是一个NumPy数组，存储了一个排列。


    >>> np.allclose(A - L[p] @ U, np.zeros((4, 4)))
    True

这行代码演示了一种通过 LU 分解来检验矩阵相等性的方法。`A` 是一个矩阵，`L` 和 `U` 是分解得到的下三角矩阵和上三角矩阵，`p` 是一个排列数组，`@` 是矩阵乘法运算符，`np.allclose` 函数用来检查 `A - L[p] @ U` 是否接近零矩阵。


    or alternatively:

    >>> P = np.eye(4)[p]
    >>> np.allclose(A - P @ L @ U, np.zeros((4, 4)))
    True
    """

这段代码展示了另一种方法来检验矩阵相等性。`P` 是一个置换矩阵，通过单位矩阵 `np.eye(4)` 根据排列 `p` 得到。然后使用 `np.allclose` 函数检查 `A - P @ L @ U` 是否接近零矩阵。


    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)

根据 `check_finite` 的值，选择性地使用 `asarray_chkfinite` 或 `asarray` 函数来将输入 `a` 转换为NumPy数组 `a1`，确保数组中没有无穷大或NaN。


    # accommodate empty arrays
    if a1.size == 0:
        lu = np.empty_like(a1)
        piv = np.arange(0, dtype=np.int32)
        return lu, piv

如果数组 `a1` 是空数组（即大小为0），则创建一个形状与 `a1` 相同的空数组 `lu`，并返回空的置换数组 `piv`。


    overwrite_a = overwrite_a or (_datacopied(a1, a))

根据 `_datacopied` 函数的返回值或输入参数 `overwrite_a` 的值来确定是否覆盖输入数组 `a1`。


    getrf, = get_lapack_funcs(('getrf',), (a1,))
    lu, piv, info = getrf(a1, overwrite_a=overwrite_a)

通过 `get_lapack_funcs` 获取 LAPACK 库中的 LU 分解函数 `getrf`，并使用 `a1` 进行 LU 分解，返回 LU 分解后的结果 `lu`、置换向量 `piv` 以及信息 `info`。


    if info < 0:
        raise ValueError('illegal value in %dth argument of '
                         'internal getrf (lu_factor)' % -info)

如果 `info` 小于0，说明在调用 LAPACK 的 `getrf` 函数时出现非法参数，抛出 `ValueError` 异常。


    if info > 0:
        warn("Diagonal number %d is exactly zero. Singular matrix." % info,
             LinAlgWarning, stacklevel=2)

如果 `info` 大于0，表示 LU 分解中发现矩阵的某个对角元素为零，警告可能是奇异矩阵。


    return lu, piv

返回 LU 分解后的 `lu` 和置换向量 `piv`。这个函数的目的是进行矩阵的 LU 分解，并返回分解结果及相关信息。
def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """解决方程组 a x = b，给定矩阵 a 的 LU 分解

    Parameters
    ----------
    (lu, piv)
        系数矩阵 a 的 LU 分解，由 lu_factor 函数给出。其中 piv 是从 0 开始的主元索引。
    b : array
        方程组的右侧向量
    trans : {0, 1, 2}, optional
        解决方程的类型:

        =====  =========
        trans  系统
        =====  =========
        0      a x   = b
        1      a^T x = b
        2      a^H x = b
        =====  =========
    overwrite_b : bool, optional
        是否覆盖 b 中的数据（可能提高性能）
    check_finite : bool, optional
        是否检查输入矩阵是否仅包含有限数。禁用此选项可能提高性能，但如果输入包含无穷大或 NaN，可能会导致问题（崩溃或不收敛）。

    Returns
    -------
    x : array
        方程组的解

    See Also
    --------
    lu_factor : 对矩阵进行 LU 分解

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import lu_factor, lu_solve
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> b = np.array([1, 1, 1, 1])
    >>> lu, piv = lu_factor(A)
    >>> x = lu_solve((lu, piv), b)
    >>> np.allclose(A @ x - b, np.zeros((4,)))
    True

    """
    (lu, piv) = lu_and_piv

    # 根据 check_finite 的值选择合适的数组检查方法
    if check_finite:
        b1 = asarray_chkfinite(b)
    else:
        b1 = asarray(b)

    # 是否需要覆盖 b1 中的数据，以及如何判断是否需要
    overwrite_b = overwrite_b or _datacopied(b1, b)

    # 检查 LU 分解和 b1 的形状是否兼容
    if lu.shape[0] != b1.shape[0]:
        raise ValueError(f"Shapes of lu {lu.shape} and b {b1.shape} are incompatible")

    # 处理空数组的情况
    if b1.size == 0:
        m = lu_solve((np.eye(2, dtype=lu.dtype), [0, 1]), np.ones(2, dtype=b.dtype))
        return np.empty_like(b1, dtype=m.dtype)

    # 获取 LAPACK 中解方程的函数
    getrs, = get_lapack_funcs(('getrs',), (lu, b1))
    
    # 调用 LAPACK 中的 getrs 函数解方程
    x, info = getrs(lu, piv, b1, trans=trans, overwrite_b=overwrite_b)
    
    # 检查返回的 info 值，若不为 0 则抛出异常
    if info == 0:
        return x
    raise ValueError('illegal value in %dth argument of internal gesv|posv'
                     % -info)
    # 检查输入矩阵是否包含仅有有限数字，默认开启，如果输入包含无穷大或NaN，则禁用可能导致性能提升但可能引发问题（崩溃、非终止）。
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    # 如果 `True`，返回置换信息作为行索引，为了向后兼容，默认为 `False`。
    p_indices : bool, optional
        If ``True`` the permutation information is returned as row indices.
        The default is ``False`` for backwards-compatibility reasons.

    Returns
    -------
    **(If `permute_l` is ``False``)**

    # 如果 `permute_l` 为 `False`，返回以下结果：

    p : (..., M, M) ndarray
        Permutation arrays or vectors depending on `p_indices`
        # 根据 `p_indices` 返回置换数组或向量
    l : (..., M, K) ndarray
        Lower triangular or trapezoidal array with unit diagonal.
        ``K = min(M, N)``
        # 带有单位对角线的下三角或梯形数组，`K = min(M, N)`
    u : (..., K, N) ndarray
        Upper triangular or trapezoidal array
        # 上三角或梯形数组

    **(If `permute_l` is ``True``)**

    # 如果 `permute_l` 为 `True`，返回以下结果：

    pl : (..., M, K) ndarray
        Permuted L matrix.
        ``K = min(M, N)``
        # 置换后的 L 矩阵，`K = min(M, N)`
    u : (..., K, N) ndarray
        Upper triangular or trapezoidal array
        # 上三角或梯形数组

    Notes
    -----
    # 注意事项：

    Permutation matrices are costly since they are nothing but row reorder of
    ``L`` and hence indices are strongly recommended to be used instead if the
    permutation is required. The relation in the 2D case then becomes simply
    ``A = L[P, :] @ U``. In higher dimensions, it is better to use `permute_l`
    to avoid complicated indexing tricks.
    # 置换矩阵代价高昂，因为它们只是对 `L` 的行重新排序，因此强烈建议使用索引代替，如果需要置换。在二维情况下，关系简单化为 `A = L[P, :] @ U`。在更高维度中，最好使用 `permute_l` 避免复杂的索引技巧。

    In 2D case, if one has the indices however, for some reason, the
    permutation matrix is still needed then it can be constructed by
    ``np.eye(M)[P, :]``.
    # 在二维情况下，如果由于某种原因需要索引，但仍然需要置换矩阵，则可以通过 ``np.eye(M)[P, :]`` 构建。

    Examples
    --------

    # 示例：

    >>> import numpy as np
    >>> from scipy.linalg import lu
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> p, l, u = lu(A)
    >>> np.allclose(A, p @ l @ u)
    True
    >>> p  # Permutation matrix
    array([[0., 1., 0., 0.],  # Row index 1
           [0., 0., 0., 1.],  # Row index 3
           [1., 0., 0., 0.],  # Row index 0
           [0., 0., 1., 0.]]) # Row index 2
    >>> p, _, _ = lu(A, p_indices=True)
    >>> p
    array([1, 3, 0, 2])  # as given by row indices above
    >>> np.allclose(A, l[p, :] @ u)
    True

    We can also use nd-arrays, for example, a demonstration with 4D array:

    # 我们也可以使用多维数组，例如，使用 4 维数组进行演示：

    >>> rng = np.random.default_rng()
    >>> A = rng.uniform(low=-4, high=4, size=[3, 2, 4, 8])
    >>> p, l, u = lu(A)
    >>> p.shape, l.shape, u.shape
    ((3, 2, 4, 4), (3, 2, 4, 4), (3, 2, 4, 8))
    >>> np.allclose(A, p @ l @ u)
    True
    >>> PL, U = lu(A, permute_l=True)
    >>> np.allclose(A, PL @ U)
    True

    """
    a1 = np.asarray_chkfinite(a) if check_finite else np.asarray(a)
    # 检查是否为 LAPACK 兼容的数据类型

    if a1.ndim < 2:
        raise ValueError('The input array must be at least two-dimensional.')

    # Also check if dtype is LAPACK compatible
    # 如果数组 a1 的数据类型不是 'f', 'd', 'F', 'D' 中的一种，则进行类型转换
    if a1.dtype.char not in 'fdFD':
        # 根据 a1.dtype.char 在 lapack_cast_dict 中查找对应的类型转换方式
        dtype_char = lapack_cast_dict[a1.dtype.char]
        # 如果没有找到对应的类型转换方式，则抛出类型错误异常
        if not dtype_char:  # 没有可用的类型转换方式
            raise TypeError(f'The dtype {a1.dtype} cannot be cast '
                            'to float(32, 64) or complex(64, 128).')

        # 将 a1 转换为对应的数据类型，这会生成一个副本
        a1 = a1.astype(dtype_char[0])  # makes a copy, free to scratch
        # 设置 overwrite_a 为 True，表示允许对 a1 进行覆盖操作
        overwrite_a = True

    # 解构 a1.shape，获取维度信息
    *nd, m, n = a1.shape
    # 计算最小的维度长度，用于后续的计算
    k = min(m, n)
    # 根据 a1 的数据类型判断实部的数据类型字符
    real_dchar = 'f' if a1.dtype.char in 'fF' else 'd'

    # 如果 a1 是空输入
    if min(*a1.shape) == 0:
        if permute_l:
            # 创建空的 PL 和 U 数组，根据 a1 的数据类型
            PL = np.empty(shape=[*nd, m, k], dtype=a1.dtype)
            U = np.empty(shape=[*nd, k, n], dtype=a1.dtype)
            return PL, U  # 返回 PL 和 U
        else:
            # 根据是否需要 p_indices 来创建不同类型的 P 数组
            P = (np.empty([*nd, 0], dtype=np.int32) if p_indices else
                 np.empty([*nd, 0, 0], dtype=real_dchar))
            L = np.empty(shape=[*nd, m, k], dtype=a1.dtype)
            U = np.empty(shape=[*nd, k, n], dtype=a1.dtype)
            return P, L, U  # 返回 P, L, U

    # 如果 a1 是标量
    if a1.shape[-2:] == (1, 1):
        if permute_l:
            # 返回形状与 a1 相同的全 1 数组，以及 a1 或其副本
            return np.ones_like(a1), (a1 if overwrite_a else a1.copy())
        else:
            # 根据是否需要 p_indices 来创建不同类型的 P 数组
            P = (np.zeros(shape=[*nd, m], dtype=int) if p_indices
                 else np.ones_like(a1))
            # 返回 P, 形状与 a1 相同的全 1 数组，以及 a1 或其副本
            return P, np.ones_like(a1), (a1 if overwrite_a else a1.copy())

    # 检查是否需要复制数据到 a1
    if not _datacopied(a1, a):  # "a"  still alive through "a1"
        if not overwrite_a:
            # 如果数据属于 "a"，则进行复制
            a1 = a1.copy(order='C')
        #  else: 不做任何操作，将使用 "a" 的数据

    # 检查数组布局，可能会出现允许覆盖但原始数组是只读或非连续的情况
    if not (a1.flags['C_CONTIGUOUS'] and a1.flags['WRITEABLE']):
        # 如果 a1 不是 C 连续或可写，则复制到 C 连续的数组
        a1 = a1.copy(order='C')

    # 如果 nd 为空，说明是二维数组
    if not nd:  # 2D array
        # 创建空的置换向量 p 和全零的 kxk 数组 u
        p = np.empty(m, dtype=np.int32)
        u = np.zeros([k, k], dtype=a1.dtype)
        # 调用 lu_dispatcher 函数计算 LU 分解
        lu_dispatcher(a1, u, p, permute_l)
        # 根据 m 和 n 的大小确定返回的顺序
        P, L, U = (p, a1, u) if m > n else (p, u, a1)

    else:  # Stacked array
        # 准备连续数据持有者 P
        P = np.empty([*nd, m], dtype=np.int32)  # perm vecs

        # 如果 m 大于 n，表示高瘦数组，将创建 U
        if m > n:
            # 创建全零的 kxk 数组 U
            U = np.zeros([*nd, k, k], dtype=a1.dtype)
            # 对每个索引的数组调用 lu_dispatcher 进行 LU 分解
            for ind in product(*[range(x) for x in a1.shape[:-2]]):
                lu_dispatcher(a1[ind], U[ind], P[ind], permute_l)
            L = a1  # L 是原始数组 a1

        else:  # 如果 m 小于等于 n，表示胖数组，将创建 L
            # 创建全零的 kxk 数组 L
            L = np.zeros([*nd, k, k], dtype=a1.dtype)
            # 对每个索引的数组调用 lu_dispatcher 进行 LU 分解
            for ind in product(*[range(x) for x in a1.shape[:-2]]):
                lu_dispatcher(a1[ind], L[ind], P[ind], permute_l)
            U = a1  # U 是原始数组 a1

    # 将置换向量 P 转换为置换数组（仅当 permute_l=False 时才需要进入此部分避免浪费）
    `
    # 如果没有置换索引和不需要置换下三角矩阵
    if (not p_indices) and (not permute_l):
        # 如果存在维度信息
        if nd:
            # 创建全零数组 Pa，形状为 [*nd, m, m]，使用实数复数数据类型
            Pa = np.zeros([*nd, m, m], dtype=real_dchar)
            # 创建索引数组 nd_ix，用于不可读索引技巧 - 为置换矩阵创建独热编码
            nd_ix = np.ix_(*([np.arange(x) for x in nd] + [np.arange(m)]))
            # 将矩阵 P 的位置设为 1
            Pa[(*nd_ix, P)] = 1
            # 更新 P 为 Pa
            P = Pa
        else:  # 对于二维情况
            # 创建全零数组 Pa，形状为 [m, m]，使用实数复数数据类型
            Pa = np.zeros([m, m], dtype=real_dchar)
            # 将 Pa 的主对角线上对应位置置为 1
            Pa[np.arange(m), P] = 1
            # 更新 P 为 Pa
            P = Pa
    
    # 根据 permute_l 的布尔值选择返回值
    return (L, U) if permute_l else (P, L, U)
```