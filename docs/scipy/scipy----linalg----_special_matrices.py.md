# `D:\src\scipysrc\scipy\scipy\linalg\_special_matrices.py`

```
# 导入math库，用于数学运算
import math

# 导入numpy库，并从numpy.lib.stride_tricks中导入as_strided函数
import numpy as np
from numpy.lib.stride_tricks import as_strided

# 定义一个公共变量列表，包含多个函数名，用于模块导入时指定可导出的内容
__all__ = ['toeplitz', 'circulant', 'hankel',
           'hadamard', 'leslie', 'kron', 'block_diag', 'companion',
           'helmert', 'hilbert', 'invhilbert', 'pascal', 'invpascal', 'dft',
           'fiedler', 'fiedler_companion', 'convolution_matrix']


# -----------------------------------------------------------------------------
#  matrix construction functions
# -----------------------------------------------------------------------------


def toeplitz(c, r=None):
    """
    Construct a Toeplitz matrix.

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row. If r is not given, ``r == conjugate(c)`` is
    assumed.

    Parameters
    ----------
    c : array_like
        First column of the matrix.  Whatever the actual shape of `c`, it
        will be converted to a 1-D array.
    r : array_like, optional
        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
        in this case, if c[0] is real, the result is a Hermitian matrix.
        r[0] is ignored; the first row of the returned matrix is
        ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
        converted to a 1-D array.

    Returns
    -------
    A : (len(c), len(r)) ndarray
        The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.

    See Also
    --------
    circulant : circulant matrix
    hankel : Hankel matrix
    solve_toeplitz : Solve a Toeplitz system.

    Notes
    -----
    The behavior when `c` or `r` is a scalar, or when `c` is complex and
    `r` is None, was changed in version 0.8.0. The behavior in previous
    versions was undocumented and is no longer supported.

    Examples
    --------
    >>> from scipy.linalg import toeplitz
    >>> toeplitz([1,2,3], [1,4,5,6])
    array([[1, 4, 5, 6],
           [2, 1, 4, 5],
           [3, 2, 1, 4]])
    >>> toeplitz([1.0, 2+3j, 4-1j])
    array([[ 1.+0.j,  2.-3.j,  4.+1.j],
           [ 2.+3.j,  1.+0.j,  2.-3.j],
           [ 4.-1.j,  2.+3.j,  1.+0.j]])

    """
    # 将输入的c转换为一维数组
    c = np.asarray(c).ravel()
    # 如果r未给定，则假设r等于c的共轭
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r).ravel()
    # 构建一个由倒转的c和r[1:]组成的一维数组，用于生成Toeplitz矩阵
    vals = np.concatenate((c[::-1], r[1:]))
    out_shp = len(c), len(r)
    n = vals.strides[0]
    # 使用as_strided函数根据给定的strides生成Toeplitz矩阵的视图，并进行深拷贝返回
    return as_strided(vals[len(c)-1:], shape=out_shp, strides=(-n, n)).copy()


def circulant(c):
    """
    Construct a circulant matrix.

    Parameters
    ----------
    c : (N,) array_like
        1-D array, the first column of the matrix.

    Returns
    -------
    A : (N, N) ndarray
        A circulant matrix whose first column is `c`.

    See Also
    --------
    toeplitz : Toeplitz matrix
    hankel : Hankel matrix
    solve_circulant : Solve a circulant system.

    Notes
    -----
    .. versionadded:: 0.8.0

    Examples
    --------
    """
    # 将输入的c转换为一维数组，作为矩阵的第一列
    c = np.asarray(c).ravel()
    # 构建并返回一个以c为第一列的循环矩阵
    return toeplitz(c)
    # 将输入的数组 `c` 转换为 NumPy 数组并展开为一维数组
    c = np.asarray(c).ravel()
    # 创建一个扩展的数组，用于生成循环矩阵的版本
    c_ext = np.concatenate((c[::-1], c[:0:-1]))
    # 获取原始数组 `c` 的长度
    L = len(c)
    # 获取扩展数组 `c_ext` 中元素的步幅（stride）
    n = c_ext.strides[0]
    # 使用 `as_strided` 函数生成一个新的视图，形成循环矩阵，并确保返回一个拷贝
    return as_strided(c_ext[L-1:], shape=(L, L), strides=(-n, n)).copy()
# 构建一个莱斯利矩阵，用于描述人口或生物种群的年龄结构与繁殖率之间的关系。

def leslie(f, s):
    """
    构建一个莱斯利矩阵。

    莱斯利矩阵描述了人口或生物种群的年龄结构与繁殖率之间的关系。
    参数：
    f : array_like
        长度为 n 的一维数组，表示年龄段的生存率。
    s : array_like
        长度为 n-1 的一维数组，表示相邻年龄段的生育率。

    返回：
    -------
    L : (n, n) ndarray
        莱斯利矩阵，第 i 行表示年龄段 i 的生育率，对角线以下的元素表示
        年龄段 i+1 的生育率。

    注意：
    -------
    此函数假设 f 和 s 为数组，并且长度合适以构建一个有效的莱斯利矩阵。
    """

    n = len(f)
    # 创建一个全零的 n × n 矩阵
    L = np.zeros((n, n))
    # 将生存率数组 f 设置为矩阵的第一行
    L[0, :] = f
    # 将生育率数组 s 设置为矩阵的对角线以下的元素
    np.fill_diagonal(L[1:, :-1], s)

    return L
    # 创建一个 Leslie 矩阵。
    
    # 给定长度为 n 的“繁殖系数”数组 `f` 和长度为 n-1 的“存活系数”数组 `s`，返回相应的 Leslie 矩阵。
    
    # Parameters
    # ----------
    # f : (N,) array_like
    #     “繁殖系数”数组。
    # s : (N-1,) array_like
    #     “存活系数”数组，必须是一维数组。数组 `s` 的长度必须比数组 `f` 的长度小 1，并且至少为 1。
    
    # Returns
    # -------
    # L : (N, N) ndarray
    #     数组除了第一行为 `f` 和第一个次对角线为 `s` 外均为零。数组的数据类型将是 `f[0]+s[0]` 的数据类型。
    
    # Notes
    # -----
    # .. versionadded:: 0.8.0
    
    # Leslie 矩阵用于模拟离散时间的年龄结构人口增长 [1]_ [2]_。在具有 `n` 个年龄类别的人口中，两组参数定义了 Leslie 矩阵：
    # `n` 个“繁殖系数”，它们表示每个年龄类别每人口单位产生的后代数量，以及 `n-1` 个“存活系数”，它们表示每个年龄类别每人口单位的存活率。
    
    # References
    # ----------
    # .. [1] P. H. Leslie, On the use of matrices in certain population
    #        mathematics, Biometrika, Vol. 33, No. 3, 183--212 (Nov. 1945)
    # .. [2] P. H. Leslie, Some further notes on the use of matrices in
    #        population mathematics, Biometrika, Vol. 35, No. 3/4, 213--245
    #        (Dec. 1948)
    
    # Examples
    # --------
    # >>> from scipy.linalg import leslie
    # >>> leslie([0.1, 2.0, 1.0, 0.1], [0.2, 0.8, 0.7])
    # array([[ 0.1,  2. ,  1. ,  0.1],
    #        [ 0.2,  0. ,  0. ,  0. ],
    #        [ 0. ,  0.8,  0. ,  0. ],
    #        [ 0. ,  0. ,  0.7,  0. ]])
    
    """
    # 将 `f` 和 `s` 转换为至少是一维的 numpy 数组
    f = np.atleast_1d(f)
    s = np.atleast_1d(s)
    
    # 如果 `f` 不是一维数组，抛出 ValueError 异常
    if f.ndim != 1:
        raise ValueError("Incorrect shape for f.  f must be 1D")
    
    # 如果 `s` 不是一维数组，抛出 ValueError 异常
    if s.ndim != 1:
        raise ValueError("Incorrect shape for s.  s must be 1D")
    
    # 如果 `s` 的长度不等于 `f` 的长度减 1，抛出 ValueError 异常
    if f.size != s.size + 1:
        raise ValueError("Incorrect lengths for f and s.  The length"
                         " of s must be one less than the length of f.")
    
    # 如果 `s` 的长度为 0，抛出 ValueError 异常
    if s.size == 0:
        raise ValueError("The length of s must be at least 1.")
    
    # 计算临时变量 `tmp` 的值为 `f[0] + s[0]`
    tmp = f[0] + s[0]
    
    # 获取数组 `f` 的长度，即矩阵的维度 `n`
    n = f.size
    
    # 创建一个数据类型为 `tmp.dtype`，形状为 `(n, n)` 的全零 numpy 数组 `a`
    a = np.zeros((n, n), dtype=tmp.dtype)
    
    # 将矩阵 `a` 的第一行设为 `f`
    a[0] = f
    
    # 将矩阵 `a` 的第一个次对角线设为 `s`
    a[list(range(1, n)), list(range(0, n - 1))] = s
    
    # 返回生成的 Leslie 矩阵 `a`
    return a
def block_diag(*arrs):
    """
    Create a block diagonal matrix from provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    Parameters
    ----------
    A, B, C, ... : array_like, up to 2-D
        Input arrays.  A 1-D array or array_like sequence of length `n` is
        treated as a 2-D array with shape ``(1,n)``.

    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal. `D` has the
        same dtype as `A`.

    Notes
    -----
    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Empty sequences (i.e., array-likes of zero size) will not be ignored.
    Noteworthy, both [] and [[]] are treated as matrices with shape ``(1,0)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import block_diag
    >>> A = [[1, 0],
    ...      [0, 1]]
    >>> B = [[3, 4, 5],
    ...      [6, 7, 8]]
    >>> C = [[7]]
    >>> P = np.zeros((2, 0), dtype='int32')
    >>> block_diag(A, B, C)
    array([[1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 3, 4, 5, 0],
           [0, 0, 6, 7, 8, 0],
           [0, 0, 0, 0, 0, 7]])
    >>> block_diag(A, P, B, C)
    array([[1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 3, 4, 5, 0],
           [0, 0, 6, 7, 8, 0],
           [0, 0, 0, 0, 0, 7]])
    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])

    """
    # 如果没有输入参数，则将 arrs 设置为包含一个空数组的元组
    if arrs == ():
        arrs = ([],)
    # 将所有输入数组转换为至少二维数组
    arrs = [np.atleast_2d(a) for a in arrs]
    # 找出维度大于2的数组在参数列表中的索引
    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    # 如果存在维度大于2的数组，抛出数值错误异常，指明出错的位置
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                         "greater than 2: %s" % bad_args)

    # 获取所有数组的形状并转换为 NumPy 数组
    shapes = np.array([a.shape for a in arrs])
    # 确定输出数组的数据类型，使用所有输入数组中的最广泛的数据类型
    out_dtype = np.result_type(*[arr.dtype for arr in arrs])
    # 创建一个全零数组作为输出数组，形状为所有输入数组形状的总和，数据类型为确定的输出数据类型
    out = np.zeros(np.sum(shapes, axis=0), dtype=out_dtype)

    # 初始化输出数组的行和列的起始位置
    r, c = 0, 0
    # 遍历每个输入数组及其形状
    for i, (rr, cc) in enumerate(shapes):
        # 将当前输入数组的数据复制到输出数组的对应位置
        out[r:r + rr, c:c + cc] = arrs[i]
        # 更新输出数组的行和列的起始位置，以便处理下一个输入数组
        r += rr
        c += cc
    # 返回最终的拼接结果数组
    return out
def helmert(n, full=False):
    """
    Create an Helmert matrix of order `n`.

    This has applications in statistics, compositional or simplicial analysis,
    and in Aitchison geometry.

    Parameters
    ----------
    n : int
        The size of the array to create.
    full : bool, optional
        If True the (n, n) ndarray will be returned.
        Otherwise the submatrix that does not include the first
        row will be returned.
        Default: False.

    Returns
    -------
    M : ndarray
        The Helmert matrix.
        The shape is (n, n) or (n-1, n) depending on the `full` argument.

    Examples
    --------
    >>> from scipy.linalg import helmert
    >>> helmert(5, full=True)
    array([[ 0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ],
           [ 0.70710678, -0.70710678,  0.        ,  0.        ,  0.        ],
           [ 0.40824829,  0.40824829, -0.81649658,  0.        ,  0.        ],
           [ 0.28867513,  0.28867513,  0.28867513, -0.8660254 ,  0.        ],
           [ 0.2236068 ,  0.2236068 ,  0.2236068 ,  0.2236068 , -0.89442719]])

    """
    # 创建一个下三角矩阵，对角线以下为全1，对角线元素为0到n-1
    H = np.tril(np.ones((n, n)), -1) - np.diag(np.arange(n))
    # 计算系数向量，元素为 0, 1, ..., n-1 的乘积
    d = np.arange(n) * np.arange(1, n+1)
    # 将第一行设置为全1
    H[0] = 1
    # 将 d 的第一个元素设置为 n
    d[0] = n
    # 计算 Helmert 矩阵
    H_full = H / np.sqrt(d)[:, np.newaxis]
    # 根据 full 参数返回完整的 Helmert 矩阵或者不包括第一行的子矩阵
    if full:
        return H_full
    else:
        # 如果条件不满足，返回H_full列表的第二个元素到末尾的切片
        return H_full[1:]
def invhilbert(n, exact=False):
    """
    Compute the inverse of the Hilbert matrix of order `n`.

    The entries in the inverse of a Hilbert matrix are integers. When `n`
    is greater than 14, some entries in the inverse exceed the upper limit
    of 64 bit integers. The `exact` argument provides two options for
    dealing with these large integers.

    Parameters
    ----------
    n : int
        The order of the Hilbert matrix.
    exact : bool, optional
        If False, the data type of the array that is returned is np.float64,
        and the array is an approximation of the inverse.
        If True, the array is the exact integer inverse array. To represent
        the exact inverse when n > 14, the returned array is an object array
        of long integers. For n <= 14, the exact inverse is returned as an
        array with data type np.int64.

    Returns
    -------
    invh : (n, n) ndarray
        The data type of the array is np.float64 if `exact` is False.
        If `exact` is True, the data type is either np.int64 (for n <= 14)
        or object (for n > 14). In the latter case, the objects in the
        array will be long integers.

    See Also
    --------
    hilbert : Create a Hilbert matrix.

    Notes
    -----
    .. versionadded:: 0.10.0

    Examples
    --------
    >>> from scipy.linalg import invhilbert
    >>> invhilbert(4)
    array([[   16.,  -120.,   240.,  -140.],
           [ -120.,  1200., -2700.,  1680.],
           [  240., -2700.,  6480., -4200.],
           [ -140.,  1680., -4200.,  2800.]])
    >>> invhilbert(4, exact=True)
    array([[   16,  -120,   240,  -140],
           [ -120,  1200, -2700,  1680],
           [  240, -2700,  6480, -4200],
           [ -140,  1680, -4200,  2800]], dtype=int64)
    >>> invhilbert(16)[7,7]
    4.2475099528537506e+19
    >>> invhilbert(16, exact=True)[7,7]
    42475099528537378560

    """
    from scipy.special import comb  # 导入组合数函数 comb
    if exact:
        if n > 14:
            dtype = object  # 如果 exact 为 True 且 n 大于 14，则数据类型为对象数组（长整数）
        else:
            dtype = np.int64  # 如果 exact 为 True 且 n 不大于 14，则数据类型为 np.int64
    else:
        dtype = np.float64  # 如果 exact 为 False，则数据类型为 np.float64
    invh = np.empty((n, n), dtype=dtype)  # 创建一个空的 n x n 的数组，指定数据类型为 dtype
    # 循环遍历范围为 n 的整数序列，i 从 0 到 n-1
    for i in range(n):
        # 内部循环遍历范围为 0 到 i+1 的整数序列，j 从 0 到 i
        for j in range(0, i + 1):
            # 计算 s 的值为 i + j
            s = i + j
            # 使用以下公式计算 invh[i, j] 的值
            # (-1) 的 s 次方乘以 (s + 1)、
            # comb(n + i, n - j - 1, exact=exact)、
            # comb(n + j, n - i - 1, exact=exact)、
            # comb(s, i, exact=exact) 的平方
            invh[i, j] = ((-1) ** s * (s + 1) *
                          comb(n + i, n - j - 1, exact=exact) *
                          comb(n + j, n - i - 1, exact=exact) *
                          comb(s, i, exact=exact) ** 2)
            # 如果 i 不等于 j，则设置 invh[j, i] 的值为 invh[i, j]
            if i != j:
                invh[j, i] = invh[i, j]
    # 返回计算后的 invh 字典
    return invh
# 定义一个函数 pascal，用于生成 n x n 的帕斯卡矩阵。
def pascal(n, kind='symmetric', exact=True):
    """
    Returns the n x n Pascal matrix.

    The Pascal matrix is a matrix containing the binomial coefficients as
    its elements.

    Parameters
    ----------
    n : int
        The size of the matrix to create; that is, the result is an n x n
        matrix.
    kind : str, optional
        Must be one of 'symmetric', 'lower', or 'upper'.
        Default is 'symmetric'.
    exact : bool, optional
        If `exact` is True, the result is either an array of type
        numpy.uint64 (if n < 35) or an object array of Python long integers.
        If `exact` is False, the coefficients in the matrix are computed using
        `scipy.special.comb` with ``exact=False``. The result will be a floating
        point array, and the values in the array will not be the exact
        coefficients, but this version is much faster than ``exact=True``.

    Returns
    -------
    p : (n, n) ndarray
        The Pascal matrix.

    See Also
    --------
    invpascal

    Notes
    -----
    See https://en.wikipedia.org/wiki/Pascal_matrix for more information
    about Pascal matrices.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy.linalg import pascal
    >>> pascal(4)
    array([[ 1,  1,  1,  1],
           [ 1,  2,  3,  4],
           [ 1,  3,  6, 10],
           [ 1,  4, 10, 20]], dtype=uint64)
    >>> pascal(4, kind='lower')
    array([[1, 0, 0, 0],
           [1, 1, 0, 0],
           [1, 2, 1, 0],
           [1, 3, 3, 1]], dtype=uint64)
    >>> pascal(50)[-1, -1]
    25477612258980856902730428600
    >>> from scipy.special import comb
    >>> comb(98, 49, exact=True)
    25477612258980856902730428600

    """

    # 导入 comb 函数用于计算组合数
    from scipy.special import comb
    # 检查 kind 参数是否合法
    if kind not in ['symmetric', 'lower', 'upper']:
        raise ValueError("kind must be 'symmetric', 'lower', or 'upper'")

    # 如果 exact 参数为 True，根据 n 的大小选择不同类型的数组来存储结果
    if exact:
        if n >= 35:
            # 创建一个对象数组用于存储精确整数
            L_n = np.empty((n, n), dtype=object)
            L_n.fill(0)
        else:
            # 创建一个无符号整数数组
            L_n = np.zeros((n, n), dtype=np.uint64)
        # 使用 comb 函数填充帕斯卡矩阵的元素
        for i in range(n):
            for j in range(i + 1):
                L_n[i, j] = comb(i, j, exact=True)
    else:
        # 使用 comb 函数计算帕斯卡矩阵的元素，结果为浮点数数组
        L_n = comb(*np.ogrid[:n, :n])

    # 根据 kind 参数选择返回的矩阵类型
    if kind == 'lower':
        p = L_n
    elif kind == 'upper':
        # 如果 kind 为 'upper'，返回 L_n 的转置
        p = L_n.T
    else:
        # 如果 kind 为 'symmetric'，返回 L_n 与其转置的乘积
        p = np.dot(L_n, L_n.T)

    # 返回生成的帕斯卡矩阵
    return p
    # 导入 comb 函数用于计算组合数
    from scipy.special import comb

    # 检查 kind 参数是否为有效取值：'symmetric', 'lower', 'upper'
    if kind not in ['symmetric', 'lower', 'upper']:
        # 如果 kind 参数无效，则抛出 ValueError 异常
        raise ValueError("'kind' must be 'symmetric', 'lower' or 'upper'.")

    # 根据 exact 参数的取值确定结果矩阵的数据类型
    if kind == 'symmetric':
        if exact:
            # 如果 exact=True 且 n <= 34，则数据类型为 np.int64
            # 否则为 object 类型
            if n > 34:
                dt = object
            else:
                dt = np.int64
        else:
            # 如果 exact=False，则数据类型为 np.float64
            dt = np.float64
        
        # 创建一个空的 n x n 的数组，数据类型为上述确定的 dt
        invp = np.empty((n, n), dtype=dt)
        
        # 嵌套循环计算 Pascal 矩阵的逆矩阵元素
        for i in range(n):
            for j in range(0, i + 1):
                v = 0
                # 计算每个元素的值，通过 comb 函数计算组合数
                for k in range(n - i):
                    v += comb(i + k, k, exact=exact) * comb(i + k, i + k - j, exact=exact)
                
                # 根据定义填充对称位置的值
                invp[i, j] = (-1)**(i - j) * v
                if i != j:
                    invp[j, i] = invp[i, j]
    else:
        # 对于 'lower' 和 'upper' 情况，通过改变帕斯卡矩阵中每条对角线的符号来计算逆矩阵。
        invp = pascal(n, kind=kind, exact=exact)
        # 如果 invp 的数据类型是 np.uint64，则进行类型转换为 np.int64。
        if invp.dtype == np.uint64:
            # 这里从 np.uint64 转换为 int64 是安全的，因为如果 `kind` 不是 "symmetric"，
            # 那么 invp 中的值远小于 2**63。
            invp = invp.view(np.int64)

        # toeplitz 矩阵的特点是交替的 1 和 -1 带。
        invp *= toeplitz((-1)**np.arange(n)).astype(invp.dtype)

    # 返回计算得到的逆矩阵 invp
    return invp
def fiedler(a):
    """
    Returns a symmetric Fiedler matrix

    Given an sequence of numbers `a`, Fiedler matrices have the structure
    ``F[i, j] = np.abs(a[i] - a[j])``, and hence zero diagonals and nonnegative
    entries. A Fiedler matrix has a dominant positive eigenvalue and other
    eigenvalues are negative. Although not valid generally, for certain inputs,
    the inverse and the determinant can be derived explicitly as given in [1]_.

    Parameters
    ----------
    a : (n,) array_like
        coefficient array

    Returns
    -------
    m : (n, n) ndarray
        The Fiedler matrix corresponding to input `a`.

    Notes
    -----
    Fiedler matrices are used in various applications, including graph theory,
    where they represent the Laplacian matrix of certain graphs.

    References
    ----------
    .. [1] H. J. Ryser, "Combinatorial Mathematics", The Mathematical
           Association of America, 1963.
    """
    # 创建一个大小为 n 的零矩阵
    n = len(a)
    m = np.zeros((n, n))
    
    # 填充 Fiedler 矩阵的非对角线元素
    for i in range(n):
        for j in range(n):
            m[i, j] = np.abs(a[i] - a[j])
    
    # 返回生成的 Fiedler 矩阵
    return m
    # 定义函数文档字符串，描述了函数的输入、输出及示例用法
        F : (n, n) ndarray
    
        See Also
        --------
        circulant, toeplitz
    
        Notes
        -----
        函数版本说明，标明自版本1.3.0起添加了该函数
    
        References
        ----------
        参考文献引用，引用了J. Todd在1977年出版的书籍《Basic Numerical Mathematics: Vol.2 : Numerical Algebra》，DOI为10.1007/978-3-0348-7286-7
    
        Examples
        --------
        >>> import numpy as np
        >>> from scipy.linalg import det, inv, fiedler
        >>> a = [1, 4, 12, 45, 77]
        >>> n = len(a)
        >>> A = fiedler(a)
        >>> A
        array([[ 0,  3, 11, 44, 76],
               [ 3,  0,  8, 41, 73],
               [11,  8,  0, 33, 65],
               [44, 41, 33,  0, 32],
               [76, 73, 65, 32,  0]])
    
        显式指出行列式和逆矩阵仅对单调递增/递减的数组有效。注意三对角结构及其角落元素。
    
        >>> Ai = inv(A)
        >>> Ai[np.abs(Ai) < 1e-12] = 0.  # 清除显示中的数值噪音
        >>> Ai
        array([[-0.16008772,  0.16666667,  0.        ,  0.        ,  0.00657895],
               [ 0.16666667, -0.22916667,  0.0625    ,  0.        ,  0.        ],
               [ 0.        ,  0.0625    , -0.07765152,  0.01515152,  0.        ],
               [ 0.        ,  0.        ,  0.01515152, -0.03077652,  0.015625  ],
               [ 0.00657895,  0.        ,  0.        ,  0.015625  , -0.00904605]])
        >>> det(A)
        15409151.999999998
        >>> (-1)**(n-1) * 2**(n-2) * np.diff(a).prod() * (a[-1] - a[0])
        15409152
    
        """
        # 将输入数组a至少转换为1维数组
        a = np.atleast_1d(a)
    
        # 如果输入数组a的维度不为1，则引发值错误
        if a.ndim != 1:
            raise ValueError("Input 'a' must be a 1D array.")
    
        # 如果输入数组a为空，则返回空的浮点数组
        if a.size == 0:
            return np.array([], dtype=float)
        # 如果输入数组a只包含一个元素，则返回一个包含0的2D数组
        elif a.size == 1:
            return np.array([[0.]])
        else:
            # 返回由数组a中元素两两之差的绝对值构成的2D数组
            return np.abs(a[:, None] - a)
def convolution_matrix(a, n, mode='full'):
    """
    Construct a convolution matrix.

    Constructs the Toeplitz matrix representing one-dimensional
    convolution [1]_.  See the notes below for details.

    Parameters
    ----------
    a : (m,) array_like
        The 1-D array to convolve.
    n : int
        The number of columns in the resulting matrix.  It gives the length
        of the input to be convolved with `a`.  This is analogous to the
        length of `v` in ``numpy.convolve(a, v)``.
    mode : str
        This is analogous to `mode` in ``numpy.convolve(v, a, mode)``.
        It must be one of ('full', 'valid', 'same').
        See below for how `mode` determines the shape of the result.

    Returns
    -------
    c : (n + m - 1, n) ndarray
        The convolution matrix.

    Notes
    -----
    The resulting matrix `c` has dimensions `(n + m - 1) x n`.
    The mode determines how the Toeplitz structure of the matrix is filled.

    .. versionadded:: 1.3.0

    References
    ----------
    .. [1] Wikipedia, "Toeplitz matrix", https://en.wikipedia.org/wiki/Toeplitz_matrix

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import convolution_matrix
    >>> a = np.array([1, 2, 3])
    >>> n = 4
    >>> c = convolution_matrix(a, n, mode='full')
    >>> c
    array([[1., 0., 0., 0.],
           [2., 1., 0., 0.],
           [3., 2., 1., 0.],
           [0., 3., 2., 1.]])
    """
    m = len(a)
    v = np.zeros(n + m - 1)
    v[:m] = a
    return toeplitz(v[:n], v[:m], mode=mode)
    A : (k, n) ndarray
        The convolution matrix whose row count `k` depends on `mode`::
            =======  =========================
             mode    k
            =======  =========================
            'full'   m + n -1
            'same'   max(m, n)
            'valid'  max(m, n) - min(m, n) + 1
            =======  =========================
        
        这是一个文档字符串，描述了数组 `A` 的形状和其含义，表示为一个二维数组（ndarray），行数 `k` 取决于参数 `mode` 的值。

    See Also
    --------
    toeplitz : Toeplitz matrix

    这段注释提供了一个参考链接，指向一个相关的函数或概念，供读者进一步了解相关内容。

    Notes
    -----
    The code::

        A = convolution_matrix(a, n, mode)

    creates a Toeplitz matrix `A` such that ``A @ v`` is equivalent to
    using ``convolve(a, v, mode)``.  The returned array always has `n`
    columns.  The number of rows depends on the specified `mode`, as
    explained above.

    这段注释说明了如何使用函数 `convolution_matrix` 创建一个 Toeplitz 矩阵 `A`，使得 `A @ v` 等价于使用 `convolve(a, v, mode)`。返回的数组总是具有 `n` 列，行数取决于指定的 `mode` 参数，如上文所述。

    In the default 'full' mode, the entries of `A` are given by::

        A[i, j] == (a[i-j] if (0 <= (i-j) < m) else 0)

    where ``m = len(a)``.  Suppose, for example, the input array is
    ``[x, y, z]``.  The convolution matrix has the form::

        [x, 0, 0, ..., 0, 0]
        [y, x, 0, ..., 0, 0]
        [z, y, x, ..., 0, 0]
        ...
        [0, 0, 0, ..., x, 0]
        [0, 0, 0, ..., y, x]
        [0, 0, 0, ..., z, y]
        [0, 0, 0, ..., 0, z]

    这段注释解释了在 'full' 模式下，矩阵 `A` 的元素是如何计算的，给出了一个示例用于说明其形式。

    In 'valid' mode, the entries of `A` are given by::

        A[i, j] == (a[i-j+m-1] if (0 <= (i-j+m-1) < m) else 0)

    This corresponds to a matrix whose rows are the subset of those from
    the 'full' case where all the coefficients in `a` are contained in the
    row.  For input ``[x, y, z]``, this array looks like::

        [z, y, x, 0, 0, ..., 0, 0, 0]
        [0, z, y, x, 0, ..., 0, 0, 0]
        [0, 0, z, y, x, ..., 0, 0, 0]
        ...
        [0, 0, 0, 0, 0, ..., x, 0, 0]
        [0, 0, 0, 0, 0, ..., y, x, 0]
        [0, 0, 0, 0, 0, ..., z, y, x]

    这段注释解释了在 'valid' 模式下，矩阵 `A` 的元素是如何计算的，给出了一个示例用于说明其形式。

    In the 'same' mode, the entries of `A` are given by::

        d = (m - 1) // 2
        A[i, j] == (a[i-j+d] if (0 <= (i-j+d) < m) else 0)

    The typical application of the 'same' mode is when one has a signal of
    length `n` (with `n` greater than ``len(a)``), and the desired output
    is a filtered signal that is still of length `n`.

    For input ``[x, y, z]``, this array looks like::

        [y, x, 0, 0, ..., 0, 0, 0]
        [z, y, x, 0, ..., 0, 0, 0]
        [0, z, y, x, ..., 0, 0, 0]
        [0, 0, z, y, ..., 0, 0, 0]
        ...
        [0, 0, 0, 0, ..., y, x, 0]
        [0, 0, 0, 0, ..., z, y, x]
        [0, 0, 0, 0, ..., 0, z, y]

    这段注释解释了在 'same' 模式下，矩阵 `A` 的元素是如何计算的，给出了一个示例用于说明其形式。

    .. versionadded:: 1.5.0

    这行注释表示从特定版本开始引入了这个功能，这对于了解函数的演变和兼容性很有帮助。

    References
    ----------
    .. [1] "Convolution", https://en.wikipedia.org/wiki/Convolution

    这段注释提供了一个参考文献的链接，帮助读者深入了解相关的数学背景和理论基础。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import convolution_matrix
    >>> A = convolution_matrix([-1, 4, -2], 5, mode='same')
    >>> A
    array([[ 4, -1,  0,  0,  0],
           [-2,  4, -1,  0,  0],
           [ 0, -2,  4, -1,  0],
           [ 0,  0, -2,  4, -1],
           [ 0,  0,  0, -2,  4]])

    Compare multiplication by `A` with the use of `numpy.convolve`.

    这段注释展示了函数的使用示例，并提供了一个比较说明，说明了矩阵 `A` 如何应用于信号处理问题中。
    if n <= 0:
        raise ValueError('n must be a positive integer.')

    # 将 a 转换为 NumPy 数组，确保是一维数组
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError('convolution_matrix expects a one-dimensional '
                         'array as input')
    if a.size == 0:
        raise ValueError('len(a) must be at least 1.')

    if mode not in ('full', 'valid', 'same'):
        raise ValueError(
            "'mode' argument must be one of ('full', 'valid', 'same')")

    # 创建长度为 n 的零填充版本的数组 a 和其反转版本的零填充数组
    az = np.pad(a, (0, n-1), 'constant')
    raz = np.pad(a[::-1], (0, n-1), 'constant')

    if mode == 'same':
        # 计算在 'same' 模式下需要修剪的长度，并计算起始和结束位置
        trim = min(n, len(a)) - 1
        tb = trim // 2
        te = trim - tb
        # 提取用于构造 Toeplitz 矩阵的列和行
        col0 = az[tb:len(az)-te]
        row0 = raz[-n-tb:len(raz)-tb]
    elif mode == 'valid':
        # 计算在 'valid' 模式下需要修剪的长度，并计算起始和结束位置
        tb = min(n, len(a)) - 1
        te = tb
        # 提取用于构造 Toeplitz 矩阵的列和行
        col0 = az[tb:len(az)-te]
        row0 = raz[-n-tb:len(raz)-tb]
    else:  # 'full'
        # 对于 'full' 模式，使用整个填充后的数组作为列和反转数组作为行
        col0 = az
        row0 = raz[-n:]

    # 返回由列和行构成的 Toeplitz 矩阵
    return toeplitz(col0, row0)
```