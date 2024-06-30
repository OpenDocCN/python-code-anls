# `D:\src\scipysrc\scipy\scipy\interpolate\_ppoly.pyx`

```
"""
Routines for evaluating and manipulating piecewise polynomials in
local power basis.

"""

# 导入 NumPy 库
import numpy as np

# 导入 Cython 模块
cimport cython

# 导入 libc.stdlib 和 libc.math 库
cimport libc.stdlib
cimport libc.math

# 从 scipy.linalg.cython_lapack 导入 dgeev 函数
from scipy.linalg.cython_lapack cimport dgeev

# 包含 "_poly_common.pxi" 文件
include "_poly_common.pxi"

# 定义常量 MAX_DIMS
DEF MAX_DIMS = 64

#------------------------------------------------------------------------------
# Piecewise power basis polynomials
#------------------------------------------------------------------------------

# 定义 evaluate 函数，用于评估分段多项式
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def evaluate(const double_or_complex[:,:,::1] c,
             const double[::1] x,
             const double[::1] xp,
             int dx,
             bint extrapolate,
             double_or_complex[:,::1] out):
    """
    Evaluate a piecewise polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
        There are `n` polynomials in each interval.
        Coefficient of highest order-term comes first.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials.
    xp : ndarray, shape (r,)
        Points to evaluate the piecewise polynomial at.
    dx : int
        Order of derivative to evaluate.  The derivative is evaluated
        piecewise and may have discontinuities.
    extrapolate : bint
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (r, n)
        Value of each polynomial at each of the input points.
        This argument is modified in-place.

    """
    cdef int ip, jp  # 声明整数变量 ip 和 jp
    cdef int interval  # 声明整数变量 interval
    cdef double xval  # 声明双精度浮点数变量 xval

    # 检查导数阶数是否为负数
    if dx < 0:
        raise ValueError("Order of derivative cannot be negative")

    # 检查形状是否匹配
    if out.shape[0] != xp.shape[0]:
        raise ValueError("out and xp have incompatible shapes")
    if out.shape[1] != c.shape[2]:
        raise ValueError("out and c have incompatible shapes")
    if c.shape[1] != x.shape[0] - 1:
        raise ValueError("x and c have incompatible shapes")

    interval = 0  # 初始化 interval 为 0
    cdef bint ascending = x[x.shape[0] - 1] >= x[0]  # 检查 x 是否升序排列

    # 开始评估
    for ip in range(len(xp)):
        xval = xp[ip]  # 获取当前评估点 xval

        # 寻找正确的区间
        if ascending:
            i = find_interval_ascending(&x[0], x.shape[0], xval, interval,
                                        extrapolate)
        else:
            i = find_interval_descending(&x[0], x.shape[0], xval, interval,
                                         extrapolate)
        if i < 0:
            # 如果未找到合适的区间，将结果设置为 NaN
            for jp in range(c.shape[2]):
                out[ip, jp] = libc.math.NAN
            continue
        else:
            interval = i  # 更新当前区间

        # 评估局部多项式
        for jp in range(c.shape[2]):
            out[ip, jp] = evaluate_poly1(xval - x[interval], c, interval,
                                         jp, dx)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
# 定义一个 Cython 函数，关闭数组索引的边界检查和负数索引的包装，启用 C 语言风格的除法
def evaluate_nd(const double_or_complex[:,:,::1] c,
                tuple xs,
                const int[:] ks,
                const double[:,:] xp,
                const int[:] dx,
                int extrapolate,
                double_or_complex[:,::1] out):
    """
    Evaluate a piecewise tensor-product polynomial.

    Parameters
    ----------
    c : ndarray, shape (k_1*...*k_d, m_1*...*m_d, n)
        Coefficients local polynomials of order `k-1` in
        `m_1`, ..., `m_d` intervals. There are `n` polynomials
        in each interval.
    ks : ndarray of int, shape (d,)
        Orders of polynomials in each dimension
    xs : d-tuple of ndarray of shape (m_d+1,) each
        Breakpoints of polynomials
    xp : ndarray, shape (r, d)
        Points to evaluate the piecewise polynomial at.
    dx : ndarray of int, shape (d,)
        Orders of derivative to evaluate.  The derivative is evaluated
        piecewise and may have discontinuities.
    extrapolate : int, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (r, n)
        Value of each polynomial at each of the input points.
        For points outside the span ``x[0] ... x[-1]``,
        ``nan`` is returned.
        This argument is modified in-place.

    """
    # 定义多个 Cython 变量和数组来存储计算过程中的中间值和索引
    cdef size_t ntot
    cdef ssize_t strides[MAX_DIMS]
    cdef ssize_t kstrides[MAX_DIMS]
    cdef double* xx[MAX_DIMS]
    cdef size_t nxx[MAX_DIMS]
    cdef double[::1] y
    cdef double_or_complex[:,:,::1] c2
    cdef int ip, jp, k, ndim
    cdef int interval[MAX_DIMS]
    cdef int pos, kpos, koutpos
    cdef int out_of_range
    cdef double xval

    # 获取维度数量
    ndim = len(xs)

    # 检查维度数量是否超过最大限制
    if ndim > MAX_DIMS:
        raise ValueError("Too many dimensions (maximum: %d)" % (MAX_DIMS,))

    # 检查各参数的形状是否符合预期
    if dx.shape[0] != ndim:
        raise ValueError("dx has incompatible shape")
    if xp.shape[1] != ndim:
        raise ValueError("xp has incompatible shape")
    if out.shape[0] != xp.shape[0]:
        raise ValueError("out and xp have incompatible shapes")
    if out.shape[1] != c.shape[2]:
        raise ValueError("out and c have incompatible shapes")

    # 计算每个维度的间隔步长
    ntot = 1
    for ip in range(ndim-1, -1, -1):
        # 检查导数阶数是否为负数
        if dx[ip] < 0:
            raise ValueError("Order of derivative cannot be negative")

        # 获取当前维度的断点数组和长度
        y = xs[ip]
        if y.shape[0] < 2:
            raise ValueError("each dimension must have >= 2 points")

        # 计算当前维度的步长和总数
        strides[ip] = ntot
        ntot *= y.shape[0] - 1

        # 获取断点数组的指针
        nxx[ip] = y.shape[0]
        xx[ip] = <double*>&y[0]

    # 检查系数数组的第二维度是否与总步长相符
    if c.shape[1] != ntot:
        raise ValueError("xs and c have incompatible shapes")

    # 计算多项式阶数的步长
    ntot = 1
    for ip in range(ndim):
        kstrides[ip] = ntot
        ntot *= ks[ip]
    # 检查数组 c 的第一个维度是否与 ntot 相等，如果不相等则抛出数值错误异常
    if c.shape[0] != ntot:
        raise ValueError("ks and c have incompatible shapes")

    # 临时存储空间初始化
    # 根据 double_or_complex 的类型选择合适的数据类型初始化 c2 数组
    if double_or_complex is double:
        c2 = np.zeros((c.shape[0], 1, 1), dtype=float)
    else:
        c2 = np.zeros((c.shape[0], 1, 1), dtype=complex)

    # 开始评估过程
    # 初始化 interval 数组
    for ip in range(ndim):
        interval[ip] = 0

    # 遍历 xp 数组的每一个元素
    for ip in range(xp.shape[0]):
        out_of_range = 0

        # 查找正确的区间
        for k in range(ndim):
            xval = xp[ip, k]

            # 使用 find_interval_ascending 函数查找 xx[k] 中 xval 对应的区间
            i = find_interval_ascending(xx[k],
                                        nxx[k],
                                        xval,
                                        interval[k],
                                        extrapolate)
            if i < 0:
                out_of_range = 1
                break
            else:
                interval[k] = i

        if out_of_range:
            # 如果超出范围，则将输出数组 out 中的对应元素设置为 NaN
            for jp in range(c.shape[2]):
                out[ip, jp] = libc.math.NAN
            continue

        # 计算 pos，用于确定在 c 中的起始位置
        pos = 0
        for k in range(ndim):
            pos += interval[k] * strides[k]

        # 通过嵌套的一维多项式求值评估局部多项式
        #
        # sum_{ijk} c[kx-i,ky-j,kz-k] x**i y**j z**k = sum_i a[i] x**i
        # a[i] = sum_j b[i,j] y**j
        # b[i,j] = sum_k c[kx-i,ky-j,kz-k] z**k
        #
        # 数组 c2 用于保存中间求和结果 a, b, ...
        for jp in range(c.shape[2]):
            # 将 c 中的数据复制到 c2 中的第一维
            c2[:,0,0] = c[:,pos,jp]

            # 倒序计算多项式
            for k in range(ndim-1, -1, -1):
                xval = xp[ip, k] - xx[k][interval[k]]
                kpos = 0
                for koutpos in range(kstrides[k]):
                    # 使用 evaluate_poly1 函数计算多项式的值并更新 c2 数组
                    c2[koutpos,0,0] = evaluate_poly1(xval, c2[kpos:kpos+ks[k],:,:], 0, 0, dx[k])
                    kpos += ks[k]

            # 将计算结果存储到输出数组 out 中的对应位置
            out[ip,jp] = c2[0,0,0]
# 设置 Cython 编译器选项：禁用数组包装边界检查，禁用索引包装边界检查，启用 C 语言风格的除法
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def fix_continuity(double_or_complex[:,:,::1] c,
                   const double[::1] x,
                   int order):
    """
    Make a piecewise polynomial continuously differentiable to given order.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
        There are `n` polynomials in each interval.
        Coefficient of highest order-term comes first.

        Coefficients c[-order-1:] are modified in-place.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials
    order : int
        Order up to which enforce piecewise differentiability.

    """

    cdef int ip, jp, kp, dx  # 声明 Cython 的整型变量
    cdef int interval  # 声明 Cython 的整型变量
    cdef double_or_complex res  # 声明 Cython 的双精度或复数类型变量
    cdef double xval  # 声明 Cython 的双精度类型变量

    # 检查导数阶数是否为负数
    if order < 0:
        raise ValueError("Order of derivative cannot be negative")

    # 检查形状是否匹配
    if c.shape[1] != x.shape[0] - 1:
        raise ValueError("x and c have incompatible shapes")
    # 检查导数阶数是否过大
    if order >= c.shape[0] - 1:
        raise ValueError("order too large")
    # 再次检查导数阶数是否为负数
    if order < 0:
        raise ValueError("order negative")

    # 开始计算
    for ip in range(1, len(x)-1):
        xval = x[ip]
        interval = ip - 1

        for jp in range(c.shape[2]):
            # 确保导数的连续性，从最高阶导数开始（低阶导数依赖于高阶导数，反之不然）
            for dx in range(order, -1, -1):
                # 计算前一个区间多项式的 dx 阶导数在 xval 处的值
                res = evaluate_poly1(xval - x[interval], c, interval, jp, dx)

                # 设置当前区间多项式的 dx 阶系数，以确保 dx 阶导数连续
                for kp in range(dx):
                    res /= kp + 1

                c[c.shape[0] - dx - 1, ip, jp] = res


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def integrate(const double_or_complex[:,:,::1] c,
              const double[::1] x,
              double a,
              double b,
              bint extrapolate,
              double_or_complex[::1] out):
    """
    Compute integral over a piecewise polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials
    a : double
        Start point of integration.
    b : double
        End point of integration.
    extrapolate : bint, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (n,)
        Integral of the piecewise polynomial, assuming the polynomial
        is zero outside the range (x[0], x[-1]).
        This argument is modified in-place.

    """

    cdef int jp  # 声明 Cython 的整型变量
    # 定义整数变量，用于存储起始和结束区间的索引以及当前区间的索引
    cdef int start_interval, end_interval, interval
    # 定义双精度或复数类型的变量，用于存储局部积分值和总积分值
    cdef double_or_complex va, vb, vtot

    # 形状检查
    if c.shape[1] != x.shape[0] - 1:
        # 如果 c 和 x 的列数不匹配，则抛出 ValueError 异常
        raise ValueError("x and c have incompatible shapes")
    if out.shape[0] != c.shape[2]:
        # 如果 out 的行数与 c 的第三维度不匹配，则抛出 ValueError 异常
        raise ValueError("x and c have incompatible shapes")

    # 确保积分上下限的顺序正确
    if not (b >= a):
        # 如果积分的上限小于下限，则抛出 ValueError 异常
        raise ValueError("Integral bounds not in order")

    # 确定 x 的升序或降序
    cdef bint ascending = x[x.shape[0] - 1] >= x[0]
    if ascending:
        # 如果是升序，则找到起始和结束区间的索引
        start_interval = find_interval_ascending(&x[0], x.shape[0], a, 0,
                                                 extrapolate)
        end_interval = find_interval_ascending(&x[0], x.shape[0], b, 0,
                                               extrapolate)
    else:
        # 如果是降序，则交换积分上下限，并找到起始和结束区间的索引
        a, b = b, a
        start_interval = find_interval_descending(&x[0], x.shape[0], a, 0,
                                                  extrapolate)
        end_interval = find_interval_descending(&x[0], x.shape[0], b, 0,
                                                extrapolate)

    # 如果找不到有效的区间索引，则将 out 数组置为 NaN，并返回
    if start_interval < 0 or end_interval < 0:
        out[:] = libc.math.NAN
        return

    # 计算积分值
    for jp in range(c.shape[2]):
        vtot = 0
        for interval in range(start_interval, end_interval+1):
            # 计算局部反导数，端点
            if interval == end_interval:
                vb = evaluate_poly1(b - x[interval], c, interval, jp, -1)
            else:
                vb = evaluate_poly1(x[interval+1] - x[interval], c, interval, jp, -1)

            # 计算局部反导数，起点
            if interval == start_interval:
                va = evaluate_poly1(a - x[interval], c, interval, jp, -1)
            else:
                va = evaluate_poly1(0, c, interval, jp, -1)

            # 计算积分值
            vtot += vb - va

        # 将计算得到的积分值存入 out 数组
        out[jp] = vtot

    # 如果是降序，则取负数
    if not ascending:
        for jp in range(c.shape[2]):
            out[jp] = -out[jp]
# 禁用 Cython 的边界检查
@cython.wraparound(False)
# 禁用 Cython 的索引包装
@cython.boundscheck(False)
# 启用 Cython 的 C 除法
@cython.cdivision(True)
# 定义函数 real_roots，计算实值分段多项式函数的实根
def real_roots(const double[:,:,::1] c, const double[::1] x, double y, bint report_discont,
               bint extrapolate):
    """
    Compute real roots of a real-valued piecewise polynomial function.

    If a section of the piecewise polynomial is identically zero, the
    values (x[begin], nan) are appended to the root list.

    If the piecewise polynomial is not continuous, and the sign
    changes across a breakpoint, the breakpoint is added to the root
    set if `report_discont` is True.

    Parameters
    ----------
    c, x
        Polynomial coefficients, as above
    y : float
        Find roots of ``pp(x) == y``.
    report_discont : bint, optional
        Whether to report discontinuities across zero at breakpoints
        as roots
    extrapolate : bint, optional
        Whether to consider roots obtained by extrapolating based
        on first and last intervals.

    """
    # 定义本地变量
    cdef list roots  # 存储所有找到的根
    cdef list cur_roots  # 存储当前计算的根
    cdef int interval, jp, k, i  # 定义整数变量

    cdef double *wr  # 双精度浮点数指针，用于实部
    cdef double *wi  # 双精度浮点数指针，用于虚部
    cdef double last_root, va, vb  # 上一个根，以及两个临时变量
    cdef double f, df, dx  # 函数值、导数值和步长
    cdef void *workspace  # 空指针，用于临时工作空间

    # 检查输入的多项式系数和断点是否兼容
    if c.shape[1] != x.shape[0] - 1:
        raise ValueError("x and c have incompatible shapes")

    # 若多项式系数为空，则返回一个空的 NumPy 数组
    if c.shape[0] == 0:
        return np.array([], dtype=float)

    # 分配内存给 wr 和 wi 指针数组
    wr = <double*>libc.stdlib.malloc(c.shape[0] * sizeof(double))
    wi = <double*>libc.stdlib.malloc(c.shape[0] * sizeof(double))
    if not wr or not wi:
        libc.stdlib.free(wr)
        libc.stdlib.free(wi)
        raise MemoryError("Failed to allocate memory in real_roots")

    # 初始化工作空间为 NULL
    workspace = NULL

    # 初始化上一个根为 NAN
    last_root = libc.math.NAN

    # 判断断点是否升序排列
    cdef bint ascending = x[x.shape[0] - 1] >= x[0]

    # 初始化根列表为空
    roots = []

    # 最终清理工作，释放内存
    finally:
        libc.stdlib.free(workspace)
        libc.stdlib.free(wr)
        libc.stdlib.free(wi)

    # 返回计算得到的根列表
    return roots


# 禁用 Cython 的边界检查
@cython.wraparound(False)
# 禁用 Cython 的索引包装
@cython.boundscheck(False)
# 启用 Cython 的 C 除法
@cython.cdivision(True)
# 定义函数 find_interval_descending，寻找降序排列的断点间隔
cdef int find_interval_descending(const double *x,
                                 size_t nx,
                                 double xval,
                                 int prev_interval=0,
                                 bint extrapolate=1) noexcept nogil:
    """
    Find an interval such that x[interval + 1] < xval <= x[interval], assuming
    that x are sorted in the descending order.
    If xval > x[0], then interval = 0, if xval < x[-1] then interval = n - 2.

    Parameters
    ----------
    x : array of double, shape (m,)
        Piecewise polynomial breakpoints sorted in descending order.
    xval : double
        Point to find.
    prev_interval : int, optional
        Interval where a previous point was found.
    extrapolate : bint, optional
        Whether to return the last of the first interval if the
        point is out-of-bounds.

    Returns
    -------
    interval : int
        Suitable interval or -1 if nan.

    """
    # 定义本地整数变量
    cdef int interval, high, low, mid
    # 定义本地双精度浮点数变量
    cdef double a, b
    # 将数组 x 的第一个元素赋值给 a
    a = x[0]
    # 将数组 x 的最后一个元素赋值给 b
    b = x[nx-1]

    # 将 prev_interval 的值赋给 interval
    interval = prev_interval
    # 如果 interval 小于 0 或者大于等于 nx，则将 interval 设为 0
    if interval < 0 or interval >= nx:
        interval = 0

    # 如果 xval 不在闭区间 [b, a] 内
    if not (b <= xval <= a):
        # 超出范围或者 xval 是 NaN
        if xval > a and extrapolate:
            # 在 a 的上方，如果允许外推，则将 interval 设为 0
            interval = 0
        elif xval < b and extrapolate:
            # 在 b 的下方，如果允许外推，则将 interval 设为 nx - 2
            interval = nx - 2
        else:
            # 不允许外推，将 interval 设为 -1
            interval = -1
    # 如果 xval 等于 b
    elif xval == b:
        # 使区间从左侧闭合，将 interval 设为 nx - 2
        interval = nx - 2
    else:
        # 在一般情况下应用二分查找。注意，low 和 high 表示区间号码，而不是横坐标。
        # 从 find_interval_ascending 转换过来时，仅仅是将比较操作中的 < 改为 >，>= 改为 <=。
        if xval <= x[interval]:
            # 如果 xval 小于等于 x[interval]，则将 low 设为 interval，high 设为 nx - 2
            low = interval
            high = nx - 2
        else:
            # 否则将 low 设为 0，high 设为 interval
            low = 0
            high = interval

        # 如果 xval 大于 x[low + 1]
        if xval > x[low + 1]:
            # 将 high 设为 low
            high = low

        # 二分查找
        while low < high:
            mid = (high + low) // 2
            if xval > x[mid]:
                # 如果 xval 大于 x[mid]，则将 high 设为 mid
                high = mid
            elif xval <= x[mid + 1]:
                # 如果 xval 小于等于 x[mid+1]，则将 low 设为 mid + 1
                low = mid + 1
            else:
                # x[mid] >= xval > x[mid+1]
                # 将 low 设为 mid
                low = mid
                break

        # 最终将 interval 设为 low
        interval = low

    # 返回最终确定的 interval
    return interval
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double_or_complex evaluate_poly1(double s, const double_or_complex[:,:,::1] c, int ci, int cj, int dx) noexcept nogil:
    """
    Evaluate polynomial, derivative, or antiderivative in a single interval.

    Antiderivatives are evaluated assuming zero integration constants.

    Parameters
    ----------
    s : double
        Polynomial x-value
    c : double[:,:,:]
        Polynomial coefficients. c[:,ci,cj] will be used
    ci, cj : int
        Which of the coefs to use
    dx : int
        Order of derivative (> 0) or antiderivative (< 0) to evaluate.

    """
    cdef int kp, k
    cdef double_or_complex res, z
    cdef double prefactor

    res = 0.0  # 初始化结果为0
    z = 1.0     # 初始化幂指数为1

    if dx < 0:
        for k in range(-dx):
            z *= s  # 计算 x 的负 dx 次幂

    for kp in range(c.shape[0]):
        # 计算微分后的项的系数
        if dx == 0:
            prefactor = 1.0
        elif dx > 0:
            # 导数
            if kp < dx:
                continue
            else:
                prefactor = 1.0
                for k in range(kp, kp - dx, -1):
                    prefactor *= k
        else:
            # 原函数
            prefactor = 1.0
            for k in range(kp, kp - dx):
                prefactor /= k + 1

        res = res + c[c.shape[0] - kp - 1, ci, cj] * z * prefactor  # 计算多项式的值

        # 计算 x 的 max(k-dx,0) 次幂
        if kp < c.shape[0] - 1 and kp >= dx:
            z *= s

    return res  # 返回计算结果


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int croots_poly1(const double[:,:,::1] c, double y, int ci, int cj,
                      double* wr, double* wi, void **workspace) except -10:
    """
    Find all complex roots of a local polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
         Coefficients of polynomials of order k
    y : float
        right-hand side of ``pp(x) == y``.
    ci, cj : int
         Index of the local polynomial whose coefficients c[:,ci,cj] to use
    wr, wi : double*
         Allocated double arrays of size `k`. The complex roots are stored
         here after call. The roots are sorted in increasing order according
         to the real part.
    workspace : double**
         Work space pointer. workspace[0] should be NULL on initial
         call.  Multiple subsequent calls with same `k` can share the
         same `workspace`.  If workspace[0] is non-NULL after the
         calls, it must be freed with libc.stdlib.free.

    Returns
    -------
    nroots : int
        How many roots found for the polynomial.
        If `-1`, the polynomial is identically zero.
        If `< -1`, an error occurred.

    Notes
    -----
    Uses LAPACK + the companion matrix method.

    """
    cdef double *a
    cdef double *work
    cdef double a0, a1, a2, d, br, bi, cc
    cdef int lwork, n, i, j, order
    cdef int nworkspace, info

    n = c.shape[0]  # 多项式的阶数
    # 检查实际的多项式阶数
    for j in range(n):
        # 如果系数矩阵中某个位置不为零，则确定多项式的阶数
        if c[j,ci,cj] != 0:
            order = n - 1 - j
            break
    else:
        # 如果所有系数均为零，则多项式阶数为 -1
        order = -1

    if order < 0:
        # 全部为零的多项式，除非右边的值正好等于系数，否则无解
        if y == 0:
            return -1
        else:
            return 0
    elif order == 0:
        # 非零常数多项式，没有实数根
        if c[n-1, ci, cj] == y:
            return -1
        else:
            return 0
    elif order == 1:
        # 低阶多项式：a0*x + a1
        a0 = c[n-1-order,ci,cj]
        a1 = c[n-1-order+1,ci,cj] - y
        wr[0] = -a1 / a0
        wi[0] = 0
        return 1
    elif order == 2:
        # 低阶多项式：a0*x**2 + a1*x + a2
        a0 = c[n-1-order,ci,cj]
        a1 = c[n-1-order+1,ci,cj]
        a2 = c[n-1-order+2,ci,cj] - y

        d = a1*a1 - 4*a0*a2
        if d < 0:
            # 无实数根
            d = libc.math.sqrt(-d)
            wr[0] = -a1/(2*a0)
            wi[0] = -d/(2*a0)
            wr[1] = -a1/(2*a0)
            wi[1] = d/(2*a0)
            return 2

        d = libc.math.sqrt(d)

        # 避免在减法中的取消问题
        if d == 0:
            wr[0] = -a1/(2*a0)
            wi[0] = 0
            wr[1] = -a1/(2*a0)
            wi[1] = 0
        elif a1 < 0:
            wr[0] = (2*a2) / (-a1 + d) # == (-a1 - d)/(2*a0)
            wi[0] = 0
            wr[1] = (-a1 + d) / (2*a0)
            wi[1] = 0
        else:
            wr[0] = (-a1 - d)/(2*a0)
            wi[0] = 0
            wr[1] = (2*a2) / (-a1 - d) # == (-a1 + d)/(2*a0)
            wi[1] = 0

        return 2

    # 计算所需的工作空间并分配它
    lwork = 1 + 8*n

    if workspace[0] == NULL:
        nworkspace = n*n + lwork
        workspace[0] = libc.stdlib.malloc(nworkspace * sizeof(double))
        if workspace[0] == NULL:
            raise MemoryError("Failed to allocate memory in croots_poly1")

    a = <double*>workspace[0]
    work = a + n*n

    # 初始化伴随矩阵，Fortran 风格顺序
    for j in range(order*order):
        a[j] = 0
    for j in range(order):
        cc = c[n-1-j,ci,cj]
        if j == 0:
            cc -= y
        a[j + (order-1)*order] = -cc / c[n-1-order,ci,cj]
        if j + 1 < order:
            a[j+1 + order*j] = 1

    # 计算伴随矩阵的特征值
    info = 0
    dgeev("N", "N", &order, a, &order, <double*>wr, <double*>wi,
          NULL, &order, NULL, &order, work, &lwork, &info)
    if info != 0:
        # 失败
        return -2

    # 对根进行排序（插入排序）
    for i in range(order):
        br = wr[i]
        bi = wi[i]
        for j in range(i - 1, -1, -1):
            if wr[j] > br:
                wr[j+1] = wr[j]
                wi[j+1] = wi[j]
            else:
                wr[j+1] = br
                wi[j+1] = bi
                break
        else:
            wr[0] = br
            wi[0] = bi
    # 返回变量 order，该变量包含经过排序后的根节点列表
    return order
# 定义函数 _croots_poly1，用于找到多项式的根
def _croots_poly1(const double[:,:,::1] c, double_complex[:,:,::1] w, double y=0):
    """
    Find roots of polynomials.

    This function is for testing croots_poly1

    Parameters
    ----------
    c : ndarray, (k, m, n)
        Coefficients of several order-k polynomials
    w : ndarray, (k, m, n)
        Output argument --- roots of the polynomials.

    """

    # 声明变量
    cdef double *wr
    cdef double *wi
    cdef void *workspace
    cdef int i, j, k, nroots

    # 检查输入的多维数组形状是否匹配
    if (c.shape[0] != w.shape[0] or c.shape[1] != w.shape[1]
            or c.shape[2] != w.shape[2]):
        raise ValueError("c and w have incompatible shapes")
    if c.shape[0] <= 0:
        return

    # 分配内存以存储实部和虚部的根
    wr = <double*>libc.stdlib.malloc(c.shape[0] * sizeof(double))
    wi = <double*>libc.stdlib.malloc(c.shape[0] * sizeof(double))
    if not wr or not wi:
        # 内存分配失败时释放已分配的内存并抛出异常
        libc.stdlib.free(wr)
        libc.stdlib.free(wi)
        raise MemoryError("Failed to allocate memory in _croots_poly1")

    workspace = NULL

    try:
        # 循环遍历多项式系数的每个元素
        for i in range(c.shape[1]):
            for j in range(c.shape[2]):
                for k in range(c.shape[0]):
                    w[k,i,j] = libc.math.NAN

                # 调用 C 扩展函数 croots_poly1 查找多项式的根
                nroots = croots_poly1(c, y, i, j, wr, wi, &workspace)

                # 处理根查找失败的情况
                if nroots == -1:
                    continue
                elif nroots < -1 or nroots >= c.shape[0]:
                    raise RuntimeError("root-finding failed")

                # 将找到的根赋值给输出数组 w
                for k in range(nroots):
                    w[k,i,j].real = wr[k]
                    w[k,i,j].imag = wi[k]
    finally:
        # 释放动态分配的内存
        libc.stdlib.free(workspace)
        libc.stdlib.free(wr)
        libc.stdlib.free(wi)


#------------------------------------------------------------------------------
# Piecewise Bernstein basis polynomials
#------------------------------------------------------------------------------

# 定义评估 Bernstein 基函数多项式的函数 evaluate_bpoly1
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double_or_complex evaluate_bpoly1(double_or_complex s,
                                       const double_or_complex[:,:,::1] c,
                                       int ci, int cj) noexcept nogil:
    """
    Evaluate polynomial in the Bernstein basis in a single interval.

    A Bernstein polynomial is defined as

        .. math:: b_{j, k} = comb(k, j) x^{j} (1-x)^{k-j}

    with ``0 <= x <= 1``.

    Parameters
    ----------
    s : double
        Polynomial x-value
    c : double[:,:,:]
        Polynomial coefficients. c[:,ci,cj] will be used
    ci, cj : int
        Which of the coefs to use

    """
    cdef int k, j
    cdef double_or_complex res, s1, comb

    k = c.shape[0] - 1  # 多项式的阶数
    s1 = 1. - s

    # 特殊处理最低阶次的情况
    if k == 0:
        res = c[0, ci, cj]
    elif k == 1:
        res = c[0, ci, cj] * s1 + c[1, ci, cj] * s
    elif k == 2:
        res = c[0, ci, cj] * s1*s1 + c[1, ci, cj] * 2.*s1*s + c[2, ci, cj] * s*s
    # 如果 k 等于 3，则使用三次贝塞尔曲线的求值公式计算结果
    elif k == 3:
        res = (c[0, ci, cj] * s1*s1*s1 + c[1, ci, cj] * 3.*s1*s1*s +
               c[2, ci, cj] * 3.*s1*s*s + c[3, ci, cj] * s*s*s)
    else:
        # 否则，使用 de Casteljau 算法计算贝塞尔曲线的值
        # 初始化结果 res 和组合数 comb
        res, comb = 0., 1.
        # 遍历从 0 到 k 的所有次数 j
        for j in range(k+1):
            # 计算当前阶数 j 的贝塞尔基函数值乘以权重系数，并累加到 res 中
            res += comb * s**j * s1**(k-j) * c[j, ci, cj]
            # 更新组合数 comb，用于计算下一个 j 的组合数
            comb *= 1. * (k-j) / (j+1.)

    # 返回计算得到的贝塞尔曲线值
    return res
# 设置 Cython 参数：禁用数组边界检查、循环包装和除法检查
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
# 定义函数 evaluate_bpoly1_deriv，用于计算 Bernstein 基础上的多项式导数
cdef double_or_complex evaluate_bpoly1_deriv(double_or_complex s,
                                             const double_or_complex[:,:,::1] c,
                                             int ci, int cj,
                                             int nu,
                                             double_or_complex[:,:,::1] wrk) noexcept nogil:
    """
    Evaluate the derivative of a polynomial in the Bernstein basis
    in a single interval.

    A Bernstein polynomial is defined as

        .. math:: b_{j, k} = comb(k, j) x^{j} (1-x)^{k-j}

    with ``0 <= x <= 1``.

    The algorithm is detailed in BPoly._construct_from_derivatives.

    Parameters
    ----------
    s : double
        Polynomial x-value
    c : double[:,:,:]
        Polynomial coefficients. c[:,ci,cj] will be used
    ci, cj : int
        Which of the coefs to use
    nu : int
        Order of the derivative to evaluate. Assumed strictly positive
        (no checks are made).
    wrk : double[:,:,::1]
        A work array, shape (c.shape[0]-nu, 1, 1).

    """
    cdef int k, j, a
    cdef double_or_complex res, term
    cdef double comb, poch

    # 确定多项式的阶数
    k = c.shape[0] - 1  # polynomial order

    # 如果求导数的阶数为0，则直接调用 evaluate_bpoly1 函数计算结果
    if nu == 0:
        res = evaluate_bpoly1(s, c, ci, cj)
    else:
        # 否则，根据 BPoly._construct_from_derivatives 中的算法计算导数
        poch = 1.
        for a in range(nu):
            poch *= k - a

        # 计算导数的每一项
        term = 0.
        for a in range(k - nu + 1):
            term, comb = 0., 1.
            for j in range(nu+1):
                # 计算组合数和每一项的贡献
                term += c[j+a, ci, cj] * (-1)**(j+nu) * comb
                comb *= 1. * (nu-j) / (j+1)
            wrk[a, 0, 0] = term * poch
        
        # 使用 wrk 数组计算 evaluate_bpoly1 函数的结果
        res = evaluate_bpoly1(s, wrk, 0, 0)

    # 返回计算结果
    return res

#
# Evaluation; only differs from _ppoly by evaluate_poly1 -> evaluate_bpoly1
#
# 设置 Cython 参数：禁用数组边界检查、循环包装和除法检查
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
# 定义函数 evaluate_bernstein，用于评估 Bernstein 基础上的分段多项式
def evaluate_bernstein(const double_or_complex[:,:,::1] c,
                       const double[::1] x,
                       const double[::1] xp,
                       int nu,
                       bint extrapolate,
                       double_or_complex[:,::1] out):
    """
    Evaluate a piecewise polynomial in the Bernstein basis.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
        There are `n` polynomials in each interval.
        Coefficient of highest order-term comes first.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials
    xp : ndarray, shape (r,)
        Points to evaluate the piecewise polynomial at.
    nu : int
        Order of derivative to evaluate.  The derivative is evaluated
        piecewise and may have discontinuities.
    extrapolate : bint, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (r, n)
        Value of each polynomial at each of the input points.
        This argument is modified in-place.

    """
    cdef int ip, jp
    cdef int interval
    cdef double xval
    cdef double_or_complex s, ds, ds_nu
    cdef double_or_complex[:,:,::1] wrk

    # 检查导数阶数是否合法
    if nu < 0:
        raise NotImplementedError("Cannot do antiderivatives in the B-basis yet.")

    # 检查形状是否匹配
    if out.shape[0] != xp.shape[0]:
        raise ValueError("out and xp have incompatible shapes")
    if out.shape[1] != c.shape[2]:
        raise ValueError("out and c have incompatible shapes")
    if c.shape[1] != x.shape[0] - 1:
        raise ValueError("x and c have incompatible shapes")

    # 如果导数阶数大于0，则初始化wrk数组
    if nu > 0:
        if double_or_complex is double_complex:
            wrk = np.empty((c.shape[0]-nu, 1, 1), dtype=np.complex128)
        else:
            wrk = np.empty((c.shape[0]-nu, 1, 1), dtype=np.float64)

    # 初始化interval为0，并根据x的最后一个元素是否大于等于第一个元素确定ascending的值
    interval = 0
    cdef bint ascending = x[x.shape[0] - 1] >= x[0]

    # 开始评估
    for ip in range(len(xp)):
        xval = xp[ip]

        # 确定正确的区间
        if ascending:
            i = find_interval_ascending(&x[0], x.shape[0], xval, interval,
                                        extrapolate)
        else:
            i = find_interval_descending(&x[0], x.shape[0], xval, interval,
                                         extrapolate)

        if i < 0:
            # 如果xval是nan等特殊值，则将out[ip, jp]设置为NaN
            for jp in range(c.shape[2]):
                out[ip, jp] = libc.math.NAN
            continue
        else:
            interval = i

        # 计算局部多项式的值
        ds = x[interval+1] - x[interval]
        ds_nu = ds**nu
        for jp in range(c.shape[2]):
            s = (xval - x[interval]) / ds
            if nu == 0:
                # 如果导数阶数为0，则调用evaluate_bpoly1计算多项式的值
                out[ip, jp] = evaluate_bpoly1(s, c, interval, jp)
            else:
                # 否则，调用evaluate_bpoly1_deriv计算多项式的导数值，并除以ds的nu次幂
                out[ip, jp] = evaluate_bpoly1_deriv(s, c, interval, jp,
                        nu, wrk) / ds_nu
```