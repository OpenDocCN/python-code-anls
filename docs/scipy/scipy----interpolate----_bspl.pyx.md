# `D:\src\scipysrc\scipy\scipy\interpolate\_bspl.pyx`

```
"""
Routines for evaluating and manipulating B-splines.

"""

# 导入必要的库
import numpy as np  # 导入NumPy库
cimport numpy as cnp  # 使用Cython导入NumPy库

from numpy cimport npy_intp, npy_int64, npy_int32  # 导入NumPy整数类型

cimport cython  # 使用Cython导入cython库
from libc.math cimport NAN  # 导入数学库中的NaN常量

cnp.import_array()  # 导入NumPy数组模块

cdef extern from "src/__fitpack.h":
    void _deBoor_D(const double *t, double x, int k, int ell, int m, double *result) nogil

ctypedef double complex double_complex  # 定义复数类型

ctypedef fused double_or_complex:  # 定义复合类型，包括双精度浮点数和复数
    double
    double complex

ctypedef fused int32_or_int64:  # 定义复合类型，包括32位整数和64位整数
    cnp.npy_int32
    cnp.npy_int64

#------------------------------------------------------------------------------
# B-splines
#------------------------------------------------------------------------------

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int find_interval(const double[::1] t,
                       int k,
                       double xval,
                       int prev_l,
                       bint extrapolate) noexcept nogil:
    """
    Find an interval such that t[interval] <= xval < t[interval+1].

    Uses a linear search with locality, see fitpack's splev.

    Parameters
    ----------
    t : ndarray, shape (nt,)
        Knots
    k : int
        B-spline degree
    xval : double
        value to find the interval for
    prev_l : int
        interval where the previous value was located.
        if unknown, use any value < k to start the search.
    extrapolate : int
        whether to return the last or the first interval if xval
        is out of bounds.

    Returns
    -------
    interval : int
        Suitable interval or -1 if xval was nan.

    """
    cdef:
        int l  # 定义整数变量l
        int n = t.shape[0] - k - 1  # 计算节点数n
        double tb = t[k]  # 获取起始节点
        double te = t[n]  # 获取结束节点

    if xval != xval:
        # 如果xval为NaN，则返回-1
        return -1

    if ((xval < tb) or (xval > te)) and not extrapolate:
        # 如果xval超出节点范围且不允许外推，则返回-1
        return -1

    l = prev_l if k < prev_l < n else k  # 设置初始搜索区间

    # 在支持区间内搜索，使得t[l] <= xval < t[l+1]
    while(xval < t[l] and l != k):
        l -= 1

    l += 1
    while(xval >= t[l] and l != n):
        l += 1

    return l-1  # 返回找到的区间索引

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def evaluate_spline(const double[::1] t,
             const double_or_complex[:, ::1] c,
             int k,
             const double[::1] xp,
             int nu,
             bint extrapolate,
             double_or_complex[:, ::1] out):
    """
    Evaluate a spline in the B-spline basis.

    Parameters
    ----------
    t : ndarray, shape (n+k+1)
        knots
    c : ndarray, shape (n, m)
        B-spline coefficients
    xp : ndarray, shape (s,)
        Points to evaluate the spline at.
    nu : int
        Order of derivative to evaluate.
    extrapolate : int, optional
        Whether to extrapolate to ouf-of-bounds points, or to return NaNs.
    out : ndarray, shape (s, m)
        Computed values of the spline at each of the input points.
        This argument is modified in-place.

    """
    # 声明变量 ip, jp, a，类型为整数
    cdef int ip, jp, a
    # 声明变量 interval，类型为整数
    cdef int interval
    # 声明变量 xval，类型为双精度浮点数
    cdef double xval

    # 进行形状检查
    if out.shape[0] != xp.shape[0]:
        # 如果 out 和 xp 的第一个维度不相等，则引发数值错误异常
        raise ValueError("out and xp have incompatible shapes")
    if out.shape[1] != c.shape[1]:
        # 如果 out 和 c 的第二个维度不相等，则引发数值错误异常
        raise ValueError("out and c have incompatible shapes")

    # 检查导数阶数
    if nu < 0:
        # 如果 nu 小于零，则引发未实现错误异常，并显示该阶数
        raise NotImplementedError("Cannot do derivative order %s." % nu)

    # 创建大小为 2*k+2 的双精度浮点数数组 work
    cdef double[::1] work = np.empty(2*k+2, dtype=np.float64)

    # 进行求值
    with nogil:
        # 初始化 interval 为 k
        interval = k
        # 遍历 xp 的第一个维度
        for ip in range(xp.shape[0]):
            # 获取当前 x 值
            xval = xp[ip]

            # 查找正确的区间
            interval = find_interval(t, k, xval, interval, extrapolate)

            if interval < 0:
                # 如果 interval 小于零，说明 xval 是 NaN 或其他异常值
                # 将 out[ip, jp] 的所有元素设置为 NaN
                for jp in range(c.shape[1]):
                    out[ip, jp] = NAN
                continue

            # 求解在该区间上非零的 (k+1) 个 B-样条基函数
            # 返回时，work 的前 k+1 个元素是 B_{m-k},..., B_{m}
            _deBoor_D(&t[0], xval, k, interval, nu, &work[0])

            # 计算线性组合
            for jp in range(c.shape[1]):
                out[ip, jp] = 0.
                for a in range(k+1):
                    # 计算 B-样条基函数与系数 c 的线性组合
                    out[ip, jp] = out[ip, jp] + c[interval + a - k, jp] * work[a]
# 定义函数 evaluate_all_bspl，用于计算在给定参数 xval 处的 B 样条函数值或导数值
def evaluate_all_bspl(const double[::1] t, int k, double xval, int m, int nu=0):
    """Evaluate the ``k+1`` B-splines which are non-zero on interval ``m``.

    Parameters
    ----------
    t : ndarray, shape (nt + k + 1,)
        sorted 1D array of knots，排序的结点数组
    k : int
        spline order，样条阶数
    xval: float
        argument at which to evaluate the B-splines，待求值的自变量
    m : int
        index of the left edge of the evaluation interval, ``t[m] <= x < t[m+1]``
        左边界索引，满足 t[m] <= x < t[m+1]
    nu : int, optional
        Evaluate derivatives order `nu`. Default is zero.，求导数的阶数，默认为零

    Returns
    -------
    ndarray, shape (k+1,)
        The values of B-splines :math:`[B_{m-k}(xval), ..., B_{m}(xval)]` if
        `nu` is zero, otherwise the derivatives of order `nu`.，
        B 样条基函数在 xval 处的值，如果 nu 为零，则返回基函数值，否则返回指定阶数的导数值。

    Examples
    --------

    A textbook use of this sort of routine is plotting the ``k+1`` polynomial
    pieces which make up a B-spline of order `k`.

    Consider a cubic spline

    >>> k = 3
    >>> t = [0., 1., 2., 3., 4.]   # internal knots，内结点
    >>> a, b = t[0], t[-1]    # base interval is [a, b)，基本区间是 [a, b)
    >>> t = np.array([a]*k + t + [b]*k)  # add boundary knots，添加边界结点

    >>> import matplotlib.pyplot as plt
    >>> xx = np.linspace(a, b, 100)
    >>> plt.plot(xx, BSpline.basis_element(t[k:-k])(xx),
    ...          lw=3, alpha=0.5, label='basis_element')

    Now we use slide an interval ``t[m]..t[m+1]`` along the base interval
    ``a..b`` and use `evaluate_all_bspl` to compute the restriction of
    the B-spline of interest to this interval:

    >>> for i in range(k+1):
    ...    x1, x2 = t[2*k - i], t[2*k - i + 1]
    ...    xx = np.linspace(x1 - 0.5, x2 + 0.5)
    ...    yy = [evaluate_all_bspl(t, k, x, 2*k - i)[i] for x in xx]
    ...    plt.plot(xx, yy, '--', label=str(i))
    ...
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.show()

    """
    # 使用 double 数组 bbb 存储 k+1 阶 B 样条的值
    bbb = np.empty(2*k+2, dtype=np.float64)
    # 定义 C 数组的视图 work，用于调用底层 C 函数 _deBoor_D
    cdef double[::1] work = bbb
    # 调用底层 C 函数 _deBoor_D 计算 B 样条值或导数
    _deBoor_D(&t[0], xval, k, m, nu, &work[0])
    # 返回前 k+1 个元素作为 B 样条函数值或导数值
    return bbb[:k+1]


# 定义函数 insert，用于在给定的结点集合 t 中插入一个结点 xval
def insert(double xval,
           const double[::1] t,
           const double_or_complex[:, ::1] c,
           int k,
           bint periodic=False
        ):
    """Insert a single knot at `xval`.
    """
    #
    # This is a port of the FORTRAN `insert` routine by P. Dierckx,
    # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/insert.f
    # which carries the following comment:
    #
    # subroutine insert inserts a new knot x into a spline function s(x)
    # of degree k and calculates the b-spline representation of s(x) with
    # respect to the new set of knots. in addition, if iopt.ne.0, s(x)
    # will be considered as a periodic spline with period per=t(n-k)-t(k+1)
    # satisfying the boundary constraints
    #      t(i+n-2*k-1) = t(i)+per  ,i=1,2,...,2*k+1
    #      c(i+n-2*k-1) = c(i)      ,i=1,2,...,k
    # in that case, the knots and b-spline coefficients returned will also
    # satisfy these boundary constraints, i.e.
    #      tt(i+nn-2*k-1) = tt(i)+per  ,i=1,2,...,2*k+1
    #      cc(i+nn-2*k-1) = cc(i)      ,i=1,2,...,k
    cdef:
        int interval, i  # 声明变量 interval 和 i，用于存储整数值

    interval = find_interval(t, k, xval, k, False)
    if interval < 0:
        # 如果插入点超出了范围，抛出异常
        raise ValueError(f"Cannot insert the knot at {xval}.")

    # 特殊情况处理：结点的重复度大于 k+1
    # 参考 https://github.com/scipy/scipy/commit/037204c3e91
    if t[interval] == t[interval + k + 1]:
        interval -= 1  # 调整 interval 的值以避免重复结点

    if periodic:
        if (interval + 1 <= 2*k) and (interval + 1 >= t.shape[0] - 2*k):
            # 对于周期样条曲线，必须至少有 k 个内部结点 t(j) 满足 t(k+1)<t(j)<=x
            # 或者至少有 k 个内部结点 t(j) 满足 x<=t(j)<t(n-k)
            raise ValueError("Not enough internal knots.")  # 抛出异常，内部结点不足

    # 构造新的结点向量 tt
    tt = np.r_[t[:interval+1], xval, t[interval+1:]]

    # 初始化系数矩阵 cc
    cc = np.zeros((c.shape[0]+1, c.shape[1]))

    # 复制已知的系数到新的 cc 中
    cc[interval+1:, ...] = c[interval:, ...]

    # 计算新插入结点处的系数
    for i in range(interval, interval-k, -1):
        fac = (xval - tt[i]) / (tt[i+k+1] - tt[i])
        cc[i, ...] = fac*c[i, ...] + (1. - fac)*c[i-1, ...]

    # 复制左边界的系数到新的 cc 中
    cc[:interval - k+1, ...] = c[:interval - k+1, ...]

    if periodic:
        # 周期样条曲线的边界条件处理
        n = tt.shape[0]
        nk = n - k - 1
        n2k = n - 2*k - 1
        T = tt[nk] - tt[k]   # 周期 T 的计算

        if interval >= nk - k:
            # 调整左边界结点和系数
            tt[:k] = tt[nk - k:nk] - T
            cc[:k, ...] = cc[n2k:n2k + k, ...]

        if interval <= 2*k-1:
            # 调整右边界结点和系数
            tt[n-k:] = tt[k+1:k+1+k] + T
            cc[n2k:n2k + k, ...] = cc[:k, ...]

    return tt, cc
@cython.wraparound(False)
@cython.boundscheck(False)
def _colloc(const double[::1] x, const double[::1] t, int k, double[::1, :] ab,
            int offset=0):
    """
    构建 B-spline 收集矩阵。

    收集矩阵定义为 :math:`B_{j,l} = B_l(x_j)`，
    因此第 ``j`` 行包含所有在 ``x_j`` 处非零的 B-spline。

    矩阵以 LAPACK 带状存储形式构建。
    对于一个 N×N 的矩阵 A，具有 ku 个上对角线和 kl 个下对角线，
    数组 Ab 的形状为 (2*kl + ku +1, N)，
    其中 Ab 的最后 kl+ku+1 行包含 A 的对角线，
    Ab 的前 kl 行不被引用。
    更多信息参见，例如 ``*gbsv`` 算法的文档。

    此例程不应直接调用，并且不进行错误检查。

    Parameters
    ----------
    x : ndarray, shape (n,)
        排序的 1D x 值数组
    t : ndarray, shape (nt + k + 1,)
        排序的 1D 节点数组
    k : int
        B-spline 阶数
    ab : ndarray, shape (2*kl + ku + 1, nt), F-order
        此参数被就地修改。
        返回时：被清零。
        返回时：带状存储中的 B-spline 收集矩阵，具有 ``ku`` 个上对角线和 ``kl`` 个下对角线。
        这里 ``kl = ku = k``。
    offset : int, optional
        跳过的行数
    """
    cdef int left, j, a, kl, ku, clmn
    cdef double xval

    kl = ku = k
    cdef double[::1] wrk = np.empty(2*k + 2, dtype=np.float64)

    # 收集矩阵
    with nogil:
        left = k
        for j in range(x.shape[0]):
            xval = x[j]
            # 寻找区间
            left = find_interval(t, k, xval, left, extrapolate=False)

            # 填充一行
            _deBoor_D(&t[0], xval, k, left, 0, &wrk[0])
            # 对于完整矩阵应为 ``A[j + offset, left-k:left+1] = bb``
            # 在带状存储中，需要扩展行
            for a in range(k+1):
                clmn = left - k + a
                ab[kl + ku + j + offset - clmn, clmn] = wrk[a]


@cython.wraparound(False)
@cython.boundscheck(False)
def _handle_lhs_derivatives(const double[::1]t, int k, double xval,
                            double[::1, :] ab,
                            int kl, int ku,
                            const cnp.npy_long[::1] deriv_ords,
                            int offset=0):
    """
    填写与 xval 处已知导数对应的收集矩阵条目。

    收集矩阵以 _colloc 准备的带状存储形式存在。
    不进行错误检查。

    Parameters
    ----------
    t : ndarray, shape (nt + k + 1,)
        节点
    k : integer
        B-spline 阶数
    xval : float
        要评估导数的值。
    ab : ndarray, shape(2*kl + ku + 1, nt), Fortran order
        B-spline 收集矩阵。
        此参数 *就地* 修改。
    """
    # kl : 整数
    # ab 的下部对角线的数量。
    kl : integer
    # ku : 整数
    # ab 的上部对角线的数量。
    ku : integer
    # deriv_ords : 1维 ndarray
    # 在 xval 处已知的导数阶数。
    deriv_ords : 1D ndarray
    # offset : 整数, 可选
    # 跳过矩阵 ab 的前 offset 行。

    """
    cdef:
        # left : 整数
        # 在 t 中找到的与 xval 最接近的区间的左端点。
        int left, nu, a, clmn, row
        # wrk : 1维 double 数组
        # 大小为 2*k+2，用于存储临时计算结果。
        double[::1] wrk = np.empty(2*k+2, dtype=np.float64)

    # derivatives @ xval
    with nogil:
        # left : 整数
        # 在 t 中找到的与 xval 最接近的区间的左端点。
        left = find_interval(t, k, xval, k, extrapolate=False)
        # 遍历每个导数阶数的行数。
        for row in range(deriv_ords.shape[0]):
            # nu : 整数
            # 当前行所代表的导数阶数。
            nu = deriv_ords[row]
            # 使用 C 函数 _deBoor_D 计算指定点 xval 处的导数值。
            _deBoor_D(&t[0], xval, k, left, nu, &wrk[0])
            # 如果 A 是一个完整的矩阵，那么可以简单地写成
            # ``A[row + offset, left-k:left+1] = bb``。
            # 遍历当前阶数下的每个系数。
            for a in range(k+1):
                # clmn : 整数
                # ab 矩阵中的列索引。
                clmn = left - k + a
                # 将 wrk 数组中的值写入到 ab 矩阵的指定位置。
                ab[kl + ku + offset + row - clmn, clmn] = wrk[a]
# 禁用 Cython 的数组访问越界检查
@cython.wraparound(False)
# 禁用 Cython 的边界检查
@cython.boundscheck(False)
# 定义一个 Cython 函数，用于构建 B-spline 最小二乘问题的正规方程
def _norm_eq_lsq(const double[::1] x,
                 const double[::1] t,
                 int k,
                 const double_or_complex[:, ::1] y,
                 const double[::1] w,
                 double[::1, :] ab,
                 double_or_complex[::1, :] rhs):
    """Construct the normal equations for the B-spline LSQ problem.

    The observation equations are ``A @ c = y``, and the normal equations are
    ``A.T @ A @ c = A.T @ y``. This routine fills in the rhs and lhs for the
    latter.

    The B-spline collocation matrix is defined as :math:`A_{j,l} = B_l(x_j)`,
    so that row ``j`` contains all the B-splines which are non-zero
    at ``x_j``.

    The normal eq matrix has at most `2k+1` bands and is constructed in the
    LAPACK symmetrix banded storage: ``A[i, j] == ab[i-j, j]`` with `i >= j`.
    See the doctsring for `scipy.linalg.cholesky_banded` for more info.

    This routine is not supposed to be called directly, and
    does no error checking.

    Parameters
    ----------
    x : ndarray, shape (n,)
        sorted 1D array of x values
    t : ndarray, shape (nt + k + 1,)
        sorted 1D array of knots
    k : int
        spline order
    y : ndarray, shape (n, s)
        a 2D array of y values. The second dimension contains all trailing
        dimensions of the original array of ordinates.
    w : ndarray, shape(n,)
        Weights.
    ab : ndarray, shape (k+1, n), in Fortran order.
        This parameter is modified in-place.
        On entry: should be zeroed out.
        On exit: LHS of the normal equations.
    rhs : ndarray, shape (n, s), in Fortran order.
        This parameter is modified in-place.
        On entry: should be zeroed out.
        On exit: RHS of the normal equations.

    """
    # 定义局部变量和数组，wrk 用于存储临时计算结果
    cdef:
        int j, r, s, row, clmn, left, ci
        double xval, wval
        double[::1] wrk = np.empty(2*k + 2, dtype=np.float64)

    # 使用 nogil 块以释放全局解释器锁（GIL），加速计算
    with nogil:
        # 初始化左边界为 k
        left = k
        # 遍历输入的 x 数组
        for j in range(x.shape[0]):
            # 获取当前 x 值和对应的权重值
            xval = x[j]
            wval = w[j] * w[j]
            # 查找 xval 所在的区间
            left = find_interval(t, k, xval, left, extrapolate=False)

            # 计算在 xval 处非零的 B-spline 值
            _deBoor_D(&t[0], xval, k, left, 0, &wrk[0])

            # 计算 A.T @ A 中的非零值，并以带状存储的形式保存在 ab 中
            for r in range(k+1):
                row = left - k + r
                for s in range(r+1):
                    clmn = left - k + s
                    ab[r-s, clmn] += wrk[r] * wrk[s] * wval

                # 计算 A.T @ y 中的非零值，并保存在 rhs 中
                for ci in range(rhs.shape[1]):
                    rhs[row, ci] = rhs[row, ci] + wrk[r] * y[j, ci] * wval
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
def _colloc_nd(const double[:, ::1] xvals, tuple t not None, const npy_int32[::1] k):
    """Construct the N-D tensor product collocation matrix as a CSR array.

    In the dense representation, each row of the collocation matrix corresponds
    to a data point and contains non-zero b-spline basis functions which are
    non-zero at this data point.

    Parameters
    ----------
    xvals : ndarray, shape(size, ndim)
        Data points. ``xvals[j, :]`` gives the ``j``-th data point as an
        ``ndim``-dimensional array.
    t : tuple of 1D arrays, length-ndim
        Tuple of knot vectors
    k : ndarray, shape (ndim,)
        Spline degrees

    Returns
    -------
    csr_data, csr_indices, csr_indptr
        The collocation matrix in the CSR array format.

    Notes
    -----
    This function constructs a collocation matrix where each row represents a data point,
    and the elements are the coefficients of the B-spline basis functions that are nonzero
    at that data point. The matrix is stored in Compressed Sparse Row (CSR) format.

    Cython Directives
    -----------------
    @cython.boundscheck(False):
        Disables bounds-checking on array accesses to improve performance.
    @cython.wraparound(False):
        Ensures that negative indices are not wrapped around, improving performance.
    @cython.nonecheck(False):
        Assumes that no Python objects are None, which can optimize performance by avoiding
        unnecessary null checks.

    """
    """
    Algorithm: given `xvals` and the tuple of knots `t`, we construct a tensor
    product spline, i.e. a linear combination of

       B(x1; i1, t1) * B(x2; i2, t2) * ... * B(xN; iN, tN)


    Here ``B(x; i, t)`` is the ``i``-th b-spline defined by the knot vector
    ``t`` evaluated at ``x``.

    Since ``B`` functions are localized, for each point `(x1, ..., xN)` we
    loop over the dimensions, and
    - find the location in the knot array, `t[i] <= x < t[i+1]`,
    - compute all non-zero `B` values
    - place these values into the relevant row

    In the dense representation, the collocation matrix would have had a row per
    data point, and each row has the values of the basis elements (i.e., tensor
    products of B-splines) evaluated at this data point. Since the matrix is very
    sparse (has size = len(x)**ndim, with only (k+1)**ndim non-zero elements per
    row), we construct it in the CSR format.
    """
    # Define variables using Cython's static typing
    cdef:
        npy_intp size = xvals.shape[0]  # Number of data points
        npy_intp ndim = xvals.shape[1]  # Number of dimensions (typically spatial dimensions)

        # 'intervals': indices for a point in xi into the knot arrays t
        npy_intp[::1] i = np.empty(ndim, dtype=np.intp)

        # container for non-zero b-splines at each point in xi
        double[:, ::1] b = np.empty((ndim, max(k) + 1), dtype=float)

        double xd               # d-th component of x
        const double[::1] td    # knots in the dimension d
        npy_intp kd             # d-th component of k

        npy_intp iflat    # index to loop over (k+1)**ndim non-zero terms
        npy_intp volume   # the number of non-zero terms
        npy_intp[:, ::1] _indices_k1d    # tabulated np.unravel_index

        # shifted indices into the data array
        npy_intp[::1] idx_c = np.ones(ndim, dtype=np.intp) * (-101)  # any sentinel would do, really
        npy_intp[::1] cstrides
        npy_intp idx_cflat

        npy_intp[::1] nu = np.zeros(ndim, dtype=np.intp)

        int out_of_bounds
        double factor
        double[::1] wrk = np.empty(2*max(k) + 2, dtype=float)

        # output
        double[::1] csr_data
        npy_int64[::1] csr_indices

        int j, d

    # the number of non-zero b-splines for each data point.
    k1_shape = tuple(kd + 1 for kd in k)
    volume = 1
    for d in range(ndim):
        volume *= k[d] + 1

    # Precompute the shape and strides of the coefficients array.
    # This would have been the NdBSpline coefficients; in the present context
    # this is a helper to compute the indices into the collocation matrix.
    c_shape = tuple(len(t[d]) - k1_shape[d] for d in range(ndim))

    # The computation is equivalent to
    # >>> x = np.empty(c_shape)
    # >>> cstrides = [s // 8 for s in x.strides]
    cs = c_shape[1:] + (1,)
    cstrides = np.cumprod(cs[::-1], dtype=np.intp)[::-1].copy()

    # tabulate flat indices for iterating over the (k+1)**ndim subarray of
    # non-zero b-spline elements
    indices = np.unravel_index(np.arange(volume), k1_shape)
    _indices_k1d = np.asarray(indices, dtype=np.intp).T.copy()
    # 将输入的索引转换为 NumPy 数组，并进行转置和复制操作，存储在 _indices_k1d 中

    # Allocate the collocation matrix in the CSR format.
    # If dense, this would have been
    # >>> matr = np.zeros((size, max_row_index), dtype=float)
    # 分配以 CSR 格式存储的配准矩阵空间。
    # 如果是密集矩阵，这将是
    # >>> matr = np.zeros((size, max_row_index), dtype=float)
    csr_indices = np.empty(shape=(size*volume,), dtype=np.int64)
    csr_data = np.empty(shape=(size*volume,), dtype=float)
    csr_indptr = np.arange(0, volume*size + 1, volume, dtype=np.int64)

    # ### Iterate over the data points ###
    # ### 迭代数据点 ###

    for j in range(size):
        xv = xvals[j, :]

        # For each point, iterate over the dimensions
        # 对于每个点，迭代各维度
        out_of_bounds = 0
        for d in range(ndim):
            td = t[d]
            xd = xv[d]
            kd = k[d]

            # get the location of x[d] in t[d]
            # 获取 x[d] 在 t[d] 中的位置
            i[d] = find_interval(td, kd, xd, kd, extrapolate=True)

            if i[d] < 0:
                out_of_bounds = 1
                break

            # compute non-zero b-splines at this value of xd in dimension d
            # 计算在维度 d 中，xd 处的非零 B 样条
            _deBoor_D(&td[0], xd, kd, i[d], nu[d], &wrk[0])
            b[d, :kd+1] = wrk[:kd+1]

        if out_of_bounds:
            raise ValueError(f"Out of bounds in {d = }, with {xv = }")

        # Iterate over the products of non-zero b-splines and place them
        # into the current row of the design matrix
        # 迭代非零 B 样条的乘积，并将它们放置在设计矩阵的当前行中
        for iflat in range(volume):
            # the line below is an unrolled version of
            # idx_b = np.unravel_index(iflat,  tuple(kd+1 for kd in k))
            # 下面的行是 np.unravel_index(iflat, tuple(kd+1 for kd in k)) 的展开版本
            idx_b = _indices_k1d[iflat, :]

            factor = 1.0
            idx_cflat = 0
            for d in range(ndim):
                factor *= b[d, idx_b[d]]
                idx_c[d] = idx_b[d] + i[d] - k[d]
                idx_cflat += idx_c[d] * cstrides[d]

            # The `idx_cflat` computation above is an unrolled version of
            # idx_cflat = np.ravel_multi_index(tuple(idx_c), c_shape)

            # Fill the row of the collocation matrix in the CSR format.
            # If it were dense, it would have been just
            # >>> matr[j, idx_cflat] = factor
            # 在 CSR 格式中填充配准矩阵的行。
            # 如果是密集矩阵，将会是
            # >>> matr[j, idx_cflat] = factor

            # Each row of the full matrix has `volume` non-zero elements.
            # Thus the CSR format `indptr` increases in steps of `volume`
            # 完整矩阵的每行有 `volume` 个非零元素。
            # 因此 CSR 格式的 `indptr` 每次递增 `volume`
            csr_indices[j*volume + iflat] = idx_cflat
            csr_data[j*volume + iflat] = factor

    return np.asarray(csr_data), np.asarray(csr_indices), csr_indptr
```