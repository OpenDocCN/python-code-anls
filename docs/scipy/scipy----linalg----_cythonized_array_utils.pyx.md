# `D:\src\scipysrc\scipy\scipy\linalg\_cythonized_array_utils.pyx`

```
# cython: language_level=3
# 引入 Cython 模块，设置语言级别为 3
cimport cython
# 引入 NumPy 模块，重命名为 np
import numpy as np
# 从 scipy.linalg._cythonized_array_utils 中引入特定的 Cython 定义
from scipy.linalg._cythonized_array_utils cimport (
    lapack_t,  # LAPACK 类型
    np_complex_numeric_t,  # NumPy 复杂数类型
    np_numeric_t  # NumPy 数值类型
    )
# 从 scipy.linalg.cython_lapack 中引入 LAPACK 函数
from scipy.linalg.cython_lapack cimport sgetrf, dgetrf, cgetrf, zgetrf
# 从 libc.stdlib 中引入 malloc 和 free 函数
from libc.stdlib cimport malloc, free

# 暴露的模块成员列表
__all__ = ['bandwidth', 'issymmetric', 'ishermitian']


# =========================== find_det_from_lu : s, d, c, z ==================
# 定义函数 find_det_from_lu，用于从 LU 分解中计算行列式
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
def find_det_from_lu(lapack_t[:, ::1] a):
    # 参数 a 的维度和大小
    cdef int n = a.shape[0], k, perm = 0, info = 0
    # 行列式值的初始化
    cdef double det = 1.
    cdef double complex detj = 1.+0.j
    # 分配并初始化 LU 分解中使用的整型数组
    cdef int *piv = <int *>malloc(<int>n * sizeof(int))
    try:
        # 内存分配失败处理
        if not piv:
            raise MemoryError('Internal memory allocation request for LU '
                              'factorization failed in "find_det_from_lu".')

        # 根据 lapack_t 类型调用相应的 LAPACK LU 分解函数
        if lapack_t is float:
            sgetrf(&n, &n, &a[0,0], &n, &piv[0], &info)
            if info > 0:
                return 0.
        elif lapack_t is double:
            dgetrf(&n, &n, &a[0,0], &n, &piv[0], &info)
            if info > 0:
                return 0.
        elif lapack_t is floatcomplex:
            cgetrf(&n, &n, &a[0,0], &n, &piv[0], &info)
            if info > 0:
                return 0.+0.j
        else:
            zgetrf(&n, &n, &a[0,0], &n, &piv[0], &info)
            if info > 0:
                return 0.+0.j

        # 处理 LAPACK LU 分解的返回信息
        if info < 0:
            raise ValueError('find_det_from_lu has encountered an internal'
                             ' error in ?getrf routine with invalid'
                             f' value at {-info}-th parameter.'
                             )

        # 根据 lapack_t 类型计算行列式的值
        if lapack_t is float or lapack_t is double:
            for k in range(n):
                if piv[k] != (k + 1):
                    perm += 1
                det *= a[k, k]
            return -det if perm % 2 else det
        else:
            for k in range(n):
                if piv[k] != (k + 1):
                    perm += 1
                detj *= a[k, k]
            return -detj if perm % 2 else detj
    finally:
        # 释放动态分配的内存
        free(piv)
# ============================================================================


# ====================== swap_c_and_f_layout : s, d, c, z ====================
# 定义内联函数 swap_c_and_f_layout，用于交换 C 和 Fortran 内存布局
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef inline void swap_c_and_f_layout(lapack_t *a, lapack_t *b, int r, int c) noexcept nogil:
    """
    Swap+copy the memory layout of same sized buffers mainly
    for Cython LAPACK interfaces.
    """
    # 定义变量
    cdef int row, col, ith_row
    cdef lapack_t *bb = b
    cdef lapack_t *aa = a

    # 循环交换内存布局
    for col in range(c):
        ith_row = 0
        for row in range(r):
            bb[row] = aa[ith_row]
            ith_row += c
        aa += 1
        bb += r
# ============================================================================


@cython.embedsignature(True)
    # 导入 Cython 模块并禁用初始化检查
    @cython.initializedcheck(False)
    # 定义函数 bandwidth_c，参数为一个 C 排序的二维 NumPy 数组
    def bandwidth_c(const np_numeric_t[:, ::1]A):
        # 声明并初始化带宽变量 l 和 u
        cdef int l, u
        # 使用 nogil 块，调用内部 C 函数进行带宽检查
        with nogil:
            l, u = band_check_internal_c(A)
        # 返回带宽结果 l 和 u
        return l, u


    # 导入 Cython 模块并禁用初始化检查
    @cython.initializedcheck(False)
    # 定义函数 bandwidth_noncontig，参数为一个非 C 排序的二维 NumPy 数组
    def bandwidth_noncontig(const np_numeric_t[:, :]A):
        # 声明并初始化带宽变量 l 和 u
        cdef int l, u
        # 使用 nogil 块，调用内部 C 函数进行带宽检查
        with nogil:
            l, u = band_check_internal_noncontig(A)
        # 返回带宽结果 l 和 u
        return l, u


    # 定义内联 C 函数 band_check_internal_c，参数为 C 排序的二维 NumPy 数组
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline (int, int) band_check_internal_c(const np_numeric_t[:, ::1]A) noexcept nogil:
        # 获取数组的行数和列数
        cdef Py_ssize_t n = A.shape[0], m = A.shape[1]
        # 初始化下三角带宽和上三角带宽为 0
        cdef Py_ssize_t lower_band = 0, upper_band = 0, r, c
        # 设置零值用于比较
        cdef np_numeric_t zero = 0
    # 从最后一行开始向上遍历，直到第1行（不包括第1行），步长为-1
    for r in range(n-1, 0, -1):
        # 只处理超出现有带宽范围之外的情况
        for c in range(min(r - lower_band, m - 1)):
            # 如果矩阵 A 中位置 (r, c) 不为零
            if A[r, c] != zero:
                # 更新 lower_band 为 r - c，表示当前找到的下带宽
                lower_band = r - c
                # 跳出内层循环
                break
        # 如果第一个元素是满的，不再继续向上，结束外层循环
        if lower_band == r:
            break

    # 处理上三角部分
    for r in range(n-1):
        # 从最后一列开始向左遍历，直到 r + upper_band，步长为-1
        for c in range(m - 1, r + upper_band, -1):
            # 如果矩阵 A 中位置 (r, c) 不为零
            if A[r, c] != zero:
                # 更新 upper_band 为 c - r，表示当前找到的上带宽
                upper_band = c - r
                # 跳出内层循环
                break
        # 如果现有带宽超出矩阵范围，结束外层循环
        if r + 1 + upper_band > m - 1:
            break

    # 返回计算得到的 lower_band 和 upper_band
    return lower_band, upper_band
# 使用 Cython 的初始化检查装饰器，禁用初始化检查
@cython.initializedcheck(False)
# 使用 Cython 的边界检查装饰器，禁用边界检查
@cython.boundscheck(False)
# 使用 Cython 的循环边界检查装饰器，禁用循环边界检查
@cython.wraparound(False)
# 定义一个 Cython 内联函数，返回两个整数元组
cdef inline (int, int) band_check_internal_noncontig(const np_numeric_t[:, :]A) noexcept nogil:
    # 获取矩阵 A 的行数和列数
    cdef Py_ssize_t n = A.shape[0], m = A.shape[1]
    # 初始化下三角带宽和上三角带宽为 0
    cdef Py_ssize_t lower_band = 0, upper_band = 0, r, c
    # 设置数值类型 np_numeric_t 的零值
    cdef np_numeric_t zero = 0

    # 检查下三角部分
    for r in range(n-1, 0, -1):
        # 只有在超出现有带宽时才进行检查
        for c in range(min(r-lower_band, m - 1)):
            # 如果 A[r, c] 不等于零，则更新下三角带宽并中断内层循环
            if A[r, c] != zero:
                lower_band = r - c
                break
        # 如果第一个元素已满，则不再向上检查；完成下三角部分的检查
        if lower_band == r:
            break

    # 检查上三角部分
    for r in range(n-1):
        for c in range(m - 1, r + upper_band, -1):
            # 如果 A[r, c] 不等于零，则更新上三角带宽并中断内层循环
            if A[r, c] != zero:
                upper_band = c - r
                break
        # 如果现有带宽超出矩阵边界，则完成上三角部分的检查
        if r + 1 + upper_band > m - 1:
            break

    # 返回下三角和上三角的带宽
    return lower_band, upper_band


# 使用 Cython 的签名嵌入装饰器，启用函数签名嵌入
@cython.embedsignature(True)
# 定义 Python 函数，检查一个方阵的二维数组是否对称
def issymmetric(a, atol=None, rtol=None):
    """Check if a square 2D array is symmetric.

    Parameters
    ----------
    a : ndarray
        Input array of size (N, N).

    atol : float, optional
        Absolute error bound

    rtol : float, optional
        Relative error bound

    Returns
    -------
    sym : bool
        Returns True if the array symmetric.

    Raises
    ------
    TypeError
        If the dtype of the array is not supported, in particular, NumPy
        float16, float128 and complex256 dtypes for exact comparisons.

    See Also
    --------
    ishermitian : Check if a square 2D array is Hermitian

    Notes
    -----
    For square empty arrays the result is returned True by convention. Complex
    valued arrays are tested for symmetricity and not for being Hermitian (see
    examples)

    The diagonal of the array is not scanned. Thus if there are infs, NaNs or
    similar problematic entries on the diagonal, they will be ignored. However,
    `numpy.inf` will be treated as a number, that is to say ``[[1, inf],
    [inf, 2]]`` will return ``True``. On the other hand `numpy.nan` is never
    symmetric, say, ``[[1, nan], [nan, 2]]`` will return ``False``.

    When ``atol`` and/or ``rtol`` are set to , then the comparison is performed
    by `numpy.allclose` and the tolerance values are passed to it. Otherwise an
    exact comparison against zero is performed by internal functions. Hence
    performance can improve or degrade depending on the size and dtype of the
    array. If one of ``atol`` or ``rtol`` given the other one is automatically
    set to zero.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import issymmetric
    >>> A = np.arange(9).reshape(3, 3)
    >>> A = A + A.T
    >>> issymmetric(A)
    True
    >>> Ac = np.array([[1. + 1.j, 3.j], [3.j, 2.]])
    >>> issymmetric(Ac)  # not Hermitian but symmetric
    True

    """
    # 检查输入数组是否为二维数组，如果不是则引发值错误异常
    if a.ndim != 2:
        raise ValueError('Input array must be a 2D NumPy array.')
    
    # 检查输入数组的行数和列数是否相等（即是否为方阵），如果不相等则引发值错误异常
    if not np.equal(*a.shape):
        raise ValueError('Input array must be square.')
    
    # 如果输入数组的大小为0（即空数组），直接返回True
    if a.size == 0:
        return True

    # 如果指定了atol或rtol且数组元素不是整数类型，则进行浮点数比较
    if (atol or rtol) and not np.issubdtype(a.dtype, np.integer):
        # 如果其中的atol或rtol有值，则使用np.allclose进行数组对称性检查
        # 如果atol或rtol为None，则将其替换为0
        return np.allclose(a, a.T,
                           atol=atol if atol else 0.,
                           rtol=rtol if rtol else 0.)
    
    # 如果数组以C顺序存储，则调用is_sym_her_real_c函数进行检查
    if a.flags['C_CONTIGUOUS']:
        s = is_sym_her_real_c(a)
    # 如果数组以Fortran顺序存储，则调用is_sym_her_real_c函数检查其转置
    elif a.flags['F_CONTIGUOUS']:
        s = is_sym_her_real_c(a.T)
    # 如果数组不是以连续的C或Fortran顺序存储，则调用is_sym_her_real_noncontig函数进行检查
    else:
        s = is_sym_her_real_noncontig(a)
    
    # 返回最终的对称性检查结果
    return s
# 检查是否已初始化 Cython，如果未初始化则报错
@cython.initializedcheck(False)
# 定义一个函数，检查输入的二维数组是否为对称的实数矩阵
def is_sym_her_real_c(const np_numeric_t[:, ::1] A):
    # 声明一个布尔变量s，用来存储检查结果
    cdef bint s
    # 使用 nogil 上下文，调用内部函数进行检查
    with nogil:
        s = is_sym_her_real_c_internal(A)
    # 返回检查结果
    return s


# 检查是否已初始化 Cython，如果未初始化则报错
@cython.initializedcheck(False)
# 定义一个函数，检查输入的二维数组（非连续内存布局）是否为对称的实数矩阵
def is_sym_her_real_noncontig(const np_numeric_t[:, :] A):
    # 声明一个布尔变量s，用来存储检查结果
    cdef bint s
    # 使用 nogil 上下文，调用内部函数进行检查
    with nogil:
        s = is_sym_her_real_noncontig_internal(A)
    # 返回检查结果
    return s


# 声明一个内联函数，检查输入的二维数组是否为对称的实数矩阵
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_sym_her_real_c_internal(const np_numeric_t[:, ::1] A) noexcept nogil:
    # 获取矩阵的行数
    cdef Py_ssize_t n = A.shape[0], r, c

    # 遍历矩阵的上三角部分（不包括对角线），检查是否对称
    for r in xrange(n):
        for c in xrange(r):
            if A[r, c] != A[c, r]:
                return False
    # 若全部检查通过，则返回True
    return True


# 声明一个内联函数，检查输入的二维数组（非连续内存布局）是否为对称的实数矩阵
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_sym_her_real_noncontig_internal(const np_numeric_t[:, :] A) noexcept nogil:
    # 获取矩阵的行数
    cdef Py_ssize_t n = A.shape[0], r, c

    # 遍历矩阵的上三角部分（不包括对角线），检查是否对称
    for r in xrange(n):
        for c in xrange(r):
            if A[r, c] != A[c, r]:
                return False
    # 若全部检查通过，则返回True
    return True


# 使用 Cython 嵌入签名的方式定义 Python 函数，检查输入的方阵是否为 Hermitian 矩阵
@cython.embedsignature(True)
def ishermitian(a, atol=None, rtol=None):
    """Check if a square 2D array is Hermitian.

    Parameters
    ----------
    a : ndarray
        Input array of size (N, N)

    atol : float, optional
        Absolute error bound

    rtol : float, optional
        Relative error bound

    Returns
    -------
    her : bool
        Returns True if the array Hermitian.

    Raises
    ------
    TypeError
        If the dtype of the array is not supported, in particular, NumPy
        float16, float128 and complex256 dtypes.

    See Also
    --------
    issymmetric : Check if a square 2D array is symmetric

    Notes
    -----
    For square empty arrays the result is returned True by convention.

    `numpy.inf` will be treated as a number, that is to say ``[[1, inf],
    [inf, 2]]`` will return ``True``. On the other hand `numpy.nan` is never
    symmetric, say, ``[[1, nan], [nan, 2]]`` will return ``False``.

    When ``atol`` and/or ``rtol`` are set to , then the comparison is performed
    by `numpy.allclose` and the tolerance values are passed to it. Otherwise an
    exact comparison against zero is performed by internal functions. Hence
    performance can improve or degrade depending on the size and dtype of the
    array. If one of ``atol`` or ``rtol`` given the other one is automatically
    set to zero.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import ishermitian
    >>> A = np.arange(9).reshape(3, 3)
    >>> A = A + A.T
    >>> ishermitian(A)
    True
    >>> A = np.array([[1., 2. + 3.j], [2. - 3.j, 4.]])
    >>> ishermitian(A)
    True
    >>> Ac = np.array([[1. + 1.j, 3.j], [3.j, 2.]])
    >>> ishermitian(Ac)  # not Hermitian but symmetric
    False
    >>> Af = np.array([[0, 1 + 1j], [1 - (1+1e-12)*1j, 0]])
    >>> ishermitian(Af)
    False
    >>> ishermitian(Af, atol=5e-11) # almost hermitian with atol
    True
    """
    # 函数主体已经在文档字符串中进行了详细解释，这里不再重复注释
    pass
    """
    # 如果输入数组不是二维的，抛出数值错误异常
    if a.ndim != 2:
        raise ValueError('Input array must be a 2D NumPy array.')
    
    # 如果输入数组的形状不是正方形，抛出数值错误异常
    if not np.equal(*a.shape):
        raise ValueError('Input array must be square.')
    
    # 如果输入数组是空数组，直接返回True
    if a.size == 0:
        return True

    # 如果需要进行近似比较 (atol 或 rtol 非零)，且数组元素类型不是整数
    if (atol or rtol) and not np.issubdtype(a.dtype, np.integer):
        # 当需要至少一个 atol 或 rtol 时才执行此处代码
        # 调用 np.allclose 对数组 a 和其共轭转置 a.conj().T 进行比较
        # 若 atol 或 rtol 为 None，则替换为 0.
        return np.allclose(a, a.conj().T,
                           atol=atol if atol else 0.,
                           rtol=rtol if rtol else 0.)

    # 如果数组包含复数元素
    if np.iscomplexobj(a):
        # 复数元素在对角线上
        if a.flags['C_CONTIGUOUS']:
            s = is_sym_her_complex_c(a)
        elif a.flags['F_CONTIGUOUS']:
            s = is_sym_her_complex_c(a.T)
        else:
            s = is_sym_her_complex_noncontig(a)

    else:  # 处理实数分支，委托给 issymmetric 函数
        if a.flags['C_CONTIGUOUS']:
            s = is_sym_her_real_c(a)
        elif a.flags['F_CONTIGUOUS']:
            s = is_sym_her_real_c(a.T)
        else:
            s = is_sym_her_real_noncontig(a)

    # 返回计算结果 s
    return s
# 使用 Cython 声明初始化检查为 False 的函数装饰器
@cython.initializedcheck(False)
# 定义接受常量二维复数数值类型数组 A 的函数
def is_sym_her_complex_c(const np_complex_numeric_t[:, ::1]A):
    # 声明布尔类型变量 s
    cdef bint s
    # 使用 nogil 上下文，调用内部函数 is_sym_her_complex_c_internal 处理数组 A
    with nogil:
        s = is_sym_her_complex_c_internal(A)
    # 返回处理结果 s
    return s

# 使用 Cython 声明初始化检查为 False 的函数装饰器
@cython.initializedcheck(False)
# 定义接受非连续二维复数数值类型数组 A 的函数
def is_sym_her_complex_noncontig(const np_complex_numeric_t[:, :]A):
    # 声明布尔类型变量 s
    cdef bint s
    # 使用 nogil 上下文，调用内部函数 is_sym_her_complex_noncontig_internal 处理数组 A
    with nogil:
        s = is_sym_her_complex_noncontig_internal(A)
    # 返回处理结果 s
    return s

# 使用 Cython 声明初始化检查为 False，并关闭边界检查和 wraparound 的内联函数装饰器
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
# 声明内联函数 is_sym_her_complex_c_internal 处理连续二维复数数值类型数组 A
cdef inline bint is_sym_her_complex_c_internal(const np_complex_numeric_t[:, ::1]A) noexcept nogil:
    # 声明并初始化变量 n 为 A 的行数
    cdef Py_ssize_t n = A.shape[0], r, c

    # 遍历数组 A 的行 r
    for r in xrange(n):
        # 遍历行 r 中从起始到当前行数的列 c
        for c in xrange(r+1):
            # 如果 A[r, c] 不等于 A[c, r] 的共轭，则返回 False
            if A[r, c] != A[c, r].conjugate():
                return False
    # 若所有元素满足对称条件，则返回 True
    return True

# 使用 Cython 声明初始化检查为 False，并关闭边界检查和 wraparound 的内联函数装饰器
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
# 声明内联函数 is_sym_her_complex_noncontig_internal 处理非连续二维复数数值类型数组 A
cdef inline bint is_sym_her_complex_noncontig_internal(const np_complex_numeric_t[:, :]A) noexcept nogil:
    # 声明并初始化变量 n 为 A 的行数
    cdef Py_ssize_t n = A.shape[0], r, c

    # 遍历数组 A 的行 r
    for r in xrange(n):
        # 遍历行 r 中从起始到当前行数的列 c
        for c in xrange(r+1):
            # 如果 A[r, c] 不等于 A[c, r] 的共轭，则返回 False
            if A[r, c] != A[c, r].conjugate():
                return False
    # 若所有元素满足对称条件，则返回 True
    return True
```