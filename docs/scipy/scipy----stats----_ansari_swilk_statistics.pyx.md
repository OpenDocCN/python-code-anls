# `D:\src\scipysrc\scipy\scipy\stats\_ansari_swilk_statistics.pyx`

```
# 设置 Cython 编译器选项，禁用边界检查
# 设置 Cython 编译器选项，禁用初始化检查
# 设置 Cython 编译器选项，禁用循环边界检查
# 设置 Cython 编译器选项，启用 C 语言除法计算
# 设置 Cython 编译器选项，启用 C 语言幂运算

# 从 libc.math 中导入所需的数学函数
from libc.math cimport exp, sqrt, abs, log, acos
# 导入 NumPy 库并重命名为 np
import numpy as np
# 导入 cnp 模块，并允许导入 NumPy 数组接口
cimport numpy as cnp
cnp.import_array()

# 定义函数 gscale，用于计算 Ansari-Bradley W 统计量的空分布
def gscale(int test, int other):
    """
    Cython translation for the FORTRAN 77 code given in:

    Dinneen, L. C. and Blakesley, B. C., "Algorithm AS 93: A Generator for the
    Null Distribution of the Ansari-Bradley W Statistic", Applied Statistics,
    25(1), 1976, :doi:`10.2307/2346534`
    """
    # 定义局部整型变量并初始化
    cdef:
        int m = min(test, other)  # 取 test 和 other 中的较小值
        int n = max(test, other)  # 取 test 和 other 中的较大值
        int astart = ((test + 1) // 2) * (1 + (test // 2))  # 根据 test 计算 astart 的初始值
        int LL = (test * other) // 2 + 1  # 计算 LL 的初始值
        int ind  # 循环中使用的索引变量
        int n2b1  # 非零条目长度的变量
        int n2b2  # 非零条目长度的变量
        int loop_m = 3  # 循环 m 的次数
        int part_no  # 部分编号
        int ks  # ks 变量
        int len1 = 0, len2 = 0, len3 = 0  # 非零条目长度的变量

        bint symm = True if (m+n) % 2 == 0 else False  # 如果 m+n 是偶数则 symm 为 True，否则为 False
        bint odd = n % 2  # 如果 n 是奇数则 odd 为 True，否则为 False

        # 使用 NumPy 数组以支持大多数中缀运算符和切片机制
        # 然后在这些数组上定义内存视图，以避免在循环密集的辅助函数中的 NumPy 开销。
        # 未来可以通过更好的组织辅助函数来清理这些代码。
        # 在这里，我们仅提供了直接的 FORTRAN 翻译，以消除遗留代码。
        cnp.ndarray a1 = cnp.PyArray_ZEROS(1, [LL], cnp.NPY_FLOAT32, 0)  # 初始化长度为 LL 的浮点型数组 a1
        cnp.ndarray a2 = cnp.PyArray_ZEROS(1, [LL], cnp.NPY_FLOAT32, 0)  # 初始化长度为 LL 的浮点型数组 a2
        cnp.ndarray a3 = cnp.PyArray_ZEROS(1, [LL], cnp.NPY_FLOAT32, 0)  # 初始化长度为 LL 的浮点型数组 a3
        float[::1] a1v = a1  # a1 的内存视图
        float[::1] a2v = a2  # a2 的内存视图
        float[::1] a3v = a3  # a3 的内存视图

    # 如果 m 小于 0，直接返回
    if m < 0:
        return 0, np.array([], dtype=np.float32), 2

    # 对于较小的情况，特殊处理
    if m == 0:
        a1[0] = 1  # 设置 a1 的第一个元素为 1
        return astart, a1, 0  # 返回计算的 astart，a1 和 0

    # 当 m 等于 1 时的处理
    if m == 1:
        _start1(a1v, n)  # 调用 _start1 函数初始化 a1v
        # 如果不对称或 other 大于 test，则特殊设置
        if not (symm or (other > test)):
            a1v[0], a1v[LL - 1] = 1, 2  # 设置 a1v 的第一个和最后一个元素
        return astart, a1, 0  # 返回计算的 astart，a1 和 0

    # 当 m 等于 2 时的处理
    if m == 2:
        _start2(a1v, n)  # 调用 _start2 函数初始化 a1v
        # 如果不对称或 other 大于 test，则进行特殊设置
        if not (symm or (other > test)):
            for ind in range(LL//2):  # 循环 LL//2 次
                a1v[LL-1-ind], a1v[ind] = a1v[ind], a1v[LL-1-ind]  # 交换对称位置的元素
        return astart, a1, 0  # 返回计算的 astart，a1 和 0
    # 使用 nogil 上下文以避免全局解释器锁（GIL），提高并行性能
    with nogil:
        # 如果 odd 为真，进行奇数情况的初始化
        if odd:
            # 调用 _start1 和 _start2 函数初始化数组 a1v 和 a2v
            _start1(a1v, n)
            _start2(a2v, n-1)
            # 设置长度和其它变量，part_no 表示当前部分编号
            len1, len2, n2b1, n2b2 = 1 + (n // 2), n, 1, 2
            part_no = 0
        else:
            # 偶数情况的初始化，调用 _start2 函数初始化数组 a1v 和 a2v
            _start2(a1v, n)
            _start1(a2v, n-1)
            _start2(a3v, n-2)
            # 设置长度和其它变量，part_no 表示当前部分编号
            len1, len2, len3, n2b1, n2b2 = (n + 1), n / 2, (n - 1), 2, 1
            part_no = 1

        # 循环直到 loop_m 大于 m
        while loop_m <= m:
            if part_no == 0:
                # 在第一个部分执行的操作
                l1out = _frqadd(a1v, a2v, len2, n2b1)
                len1 += n
                len3 = _imply(a1v, l1out, len1, a3v, loop_m)
                n2b1 += 1
                loop_m += 1
                part_no = 1
            else:
                # 在第二个部分执行的操作
                l2out = _frqadd(a2v, a3v, len3, n2b2)
                len2 += n - 1
                _ = _imply(a2v, l2out, len2, a3v, loop_m)
                n2b2 += 1
                loop_m += 1
                part_no = 0

        # 如果不对称，将 a1v 的一部分加到另一部分
        if not symm:
            ks = (m + 3) / 2 - 1
            for ind in range(len2):
                a1v[ks+ind] += a2v[ind]

        # 如果 other 大于 test，反转数组 a1v 的一半元素
        if other > test:
            for ind in range(LL//2):
                a1v[LL-1-ind], a1v[ind] = a1v[ind], a1v[LL-1-ind]

    # 返回 astart, a1 和 0
    return astart, a1, 0
cdef inline void _start1(float[::1] a, int n) noexcept nogil:
    """
    Helper function for gscale function, see gscale docstring.
    """
    # 计算输出数组的长度
    cdef int lout = 1 + (n // 2)

    # 将数组的前 lout 个元素设置为 2
    a[:lout] = 2
    # 如果 n 是偶数，将数组的倒数第二个元素设置为 1
    if (n % 2) == 0:
        a[lout-1] = 1


cdef inline void _start2(float[::1] a, int n) noexcept nogil:
    """
    Helper function for gscale function, see gscale docstring.
    """
    # 初始化变量
    cdef:
        int odd = n % 2
        float A = 1.
        float B = 3.
        float C = 2. if odd else 0.
        int ndo = (n + 2 + odd) // 2 - odd

    # 填充前半部分数组
    for ind in range(ndo):
        a[ind] = A
        A += B
        B = 4 - B

    # 填充后半部分数组
    A, B = 1, 3
    for ind in range(n-odd, ndo-1, -1):
        a[ind] = A + C
        A += B
        B = 4 - B

    # 如果 n 是奇数，设置数组的倒数第二个元素为 2
    if odd == 1:
        a[(ndo*2)-1] = 2


cdef inline int _frqadd(float[::1] a, const float[::1] b, int lenb,
                        int offset) noexcept nogil:
    """
    Helper function for gscale function, see gscale docstring.
    """
    # 初始化常量和变量
    cdef:
        float two = 2
        int lout = lenb + offset
        int ind

    # 将数组 a 的 offset 开始的 lenb 个元素与数组 b 相乘后加到 a 上
    for ind in range(lenb):
        a[offset+ind] += two*b[ind]
    # 返回结果数组的长度
    return lout


cdef int _imply(float[::1] a, int curlen, int reslen, float[::1] b,
                int offset) noexcept nogil:
    """
    Helper function for gscale function, see gscale docstring.
    """
    # 初始化变量
    cdef:
        int i1
        int i2 = -offset
        int j2 = reslen-offset
        int j2min = (j2 + 1) // 2 - 1
        int nextlenb = j2
        int j1 = reslen-1
        float summ
        float diff

    # 减少 j2 的值
    j2 -= 1

    # 填充数组 a 和 b
    for i1 in range((reslen + 1) // 2):
        if i2 < 0:
            summ = a[i1]
        else:
            summ = a[i1] + b[i2]
            a[i1] = summ

        i2 += 1
        if j2 >= j2min:
            if j1 > curlen - 1:
                diff = summ
            else:
                diff = summ - a[j1]

            b[i1] = diff
            b[j2] = diff
            j2 -= 1

        a[j1] = summ
        j1 -= 1

    # 返回下一个 lenb 的长度
    return nextlenb


def swilk(const double[::1] x, double[::1] a, bint init=False, int n1=-1):
    """
    Calculates the Shapiro-Wilk W test and its significance level

    This is a double precision Cython translation (with modifications) of the
    FORTRAN 77 code given in:

    Royston P., "Remark AS R94: A Remark on Algorithm AS 181: The W-test for
    Normality", 1995, Applied Statistics, Vol. 44, :doi:`10.2307/2986146`

    IFAULT error code details from the R94 paper:
    - 0 for no fault
    - 1 if N, N1 < 3
    - 2 if N > 5000 (a non-fatal error)
    - 3 if N2 < N/2, so insufficient storage for A
    - 4 if N1 > N or (N1 < N and N < 20)
    - 5 if the proportion censored (N-N1)/N > 0.8
    - 6 if the data have zero range (if sorted on input)

    For SciPy, n1 is never used, set to a positive number to enable
    the functionality. Otherwise n1 = n is used.
    """
    pass  # 这个函数暂时没有实现内容，只是一个文档字符串的占位符
    cdef:
        int n = len(x)           # 计算列表 x 的长度，赋值给 n
        int n2 = len(a)          # 计算列表 a 的长度，赋值给 n2
        int ncens, nn2, ind1, i1, ind2  # 声明整型变量 ncens, nn2, ind1, i1, ind2
        bint upper = True        # 声明布尔变量 upper 并赋值为 True

        double[6] c1 = [0., 0.221157, -0.147981, -0.207119e1, 0.4434685e1,
                        -0.2706056e1]  # 定义长度为 6 的双精度浮点数数组 c1
        double[6] c2 = [0., 0.42981e-1, -0.293762, -0.1752461e1, 0.5682633e1,
                        -0.3582633e1]  # 定义长度为 6 的双精度浮点数数组 c2
        double[4] c3 = [0.5440, -0.39978, 0.25054e-1, -0.6714e-3]  # 定义长度为 4 的双精度浮点数数组 c3
        double[4] c4 = [0.13822e1, -0.77857, 0.62767e-1, -0.20322e-2]  # 定义长度为 4 的双精度浮点数数组 c4
        double[4] c5 = [-0.15861e1, -0.31082, -0.83751e-1, 0.38915e-2]  # 定义长度为 4 的双精度浮点数数组 c5
        double[3] c6 = [-0.4803, -0.82676e-1, 0.30302e-2]  # 定义长度为 3 的双精度浮点数数组 c6
        double[2] c7 = [0.164, 0.533]  # 定义长度为 2 的双精度浮点数数组 c7
        double[2] c8 = [0.1736, 0.315]  # 定义长度为 2 的双精度浮点数数组 c8
        double[2] c9 = [0.256, -0.635e-2]  # 定义长度为 2 的双精度浮点数数组 c9
        double[2] g = [-0.2273e1, 0.459]  # 定义长度为 2 的双精度浮点数数组 g
        double Z90 = 0.12816e1     # 定义双精度浮点数 Z90
        double Z95 = 0.16449e1     # 定义双精度浮点数 Z95
        double Z99 = 0.23263e1     # 定义双精度浮点数 Z99
        double ZM = 0.17509e1      # 定义双精度浮点数 ZM
        double ZSS = 0.56268       # 定义双精度浮点数 ZSS
        double BF1 = 0.8378        # 定义双精度浮点数 BF1
        double XX90 = 0.556        # 定义双精度浮点数 XX90
        double XX95 = 0.622        # 定义双精度浮点数 XX95
        double SQRTH = sqrt(2)/2   # 定义双精度浮点数 SQRTH，其值为 sqrt(2)/2
        double PI6 = 6/np.pi       # 定义双精度浮点数 PI6，其值为 6/π
        double SMALL=1e-19         # 定义双精度浮点数 SMALL，其值为 1e-19
        double w, pw, an, an25, summ2, ssumm2, rsn  # 声明双精度浮点数变量 w, pw, an, an25, summ2, ssumm2, rsn
        double A1, A2, fac, delta, w1, y, ld, bf, gamma, m, s  # 声明双精度浮点数变量 A1, A2, fac, delta, w1, y, ld, bf, gamma, m, s
        double RANGE, SA, SX, SSX, SSA, SAX, ASA, XSX, SSASSX, XX, XI  # 声明双精度浮点数变量 RANGE, SA, SX, SSX, SSA, SAX, ASA, XSX, SSASSX, XX, XI
        double Z90F, Z95F, Z99F, ZFM, ZSD, ZBAR  # 声明双精度浮点数变量 Z90F, Z95F, Z99F, ZFM, ZSD, ZBAR

    if n1 < 0:  # 如果 n1 小于 0
        n1 = n   # 将 n1 赋值为 n
    nn2 = n // 2  # 计算 n 的整数除以 2，结果赋值给 nn2
    if nn2 < n2:  # 如果 nn2 小于 n2
        return 1., 1., 3  # 返回三个值：1.0, 1.0, 3
    if n < 3:  # 如果 n 小于 3
        return 1., 1., 1  # 返回三个值：1.0, 1.0, 1
    w = 1.         # 将 w 初始化为 1.0
    pw = 1.        # 将 pw 初始化为 1.0
    an = n         # 将 an 初始化为 n

    if not init:   # 如果 init 不为真（即 init 为假）
        if n == 3:  # 如果 n 等于 3
            a[0] = SQRTH  # 将数组 a 的第一个元素赋值为 SQRTH 的值
        else:
            an25 = an + 0.25  # 计算 an + 0.25，结果赋值给 an25
            summ2 = 0.        # 初始化 summ2 为 0
            for ind1 in range(n2):  # 遍历范围在 n2 内的整数序列，ind1 依次取值
                temp = _ppnd((ind1+1-0.375)/an25)  # 计算 _ppnd 函数的返回值，传入的参数为 (ind1+1-0.375)/an25，结果赋值给 temp
                a[ind1] = temp  # 将 temp 的值赋给数组 a 的第 ind1 个元素
                summ2 += temp**2  # 将 temp 的平方加到 summ2 上

            summ2 *= 2.  # 将 summ2 乘以 2
            ssumm2 = sqrt(summ2)  # 计算 summ2 的平方根，结果赋值给 ssumm2
            rsn = 1 / sqrt(an)  # 计算 1/an 的平方根，结果赋值给 rsn
            A1 = _poly(c1, 6, rsn) - (a[0]/ssumm2)  # 计算 _poly 函数的返回值，传入的参数为 c1, 6, rsn，减去 a[0]/ssumm2 的值，结果赋给 A1
            if n > 5:  # 如果 n 大于 5
                i1 = 2  # 将 i1 赋值为 2
                A2 = -a[1]/ssumm2 + _poly(c2, 6, rsn)  # 计算 _poly 函数的返回值，传入的参数为 c2, 6, rsn，减去 -a[1]/ssumm2 的值，结果赋给 A2
                fac = sqrt((summ2 - (2 * a[0]**2) - 2 * a[1]**2) /
                           (1 - (2 * A1**2) - 2 *
    # 计算数据范围
    RANGE = x[n1-1] - x[0]
    # 如果数据范围小于预定义的阈值SMALL，则返回结果并指示异常代码6
    if RANGE < SMALL:
        return w, pw, 6

    # 计算归一化的第一个数据点的比例
    XX = x[0] / RANGE
    SX = XX
    SA = -a[0]
    ind2 = n - 2
    # 循环处理数据点
    for ind1 in range(1, n1):
        XI = x[ind1] / RANGE
        SX += XI
        # 计算SA，根据数据点位置和系数a的值来调整
        if ind1 != ind2:
            SA += (-1 if ind1 < ind2 else 1)*a[min(ind1, ind2)]
        XX = XI
        ind2 -= 1

    # 初始化ifault为0
    ifault = 0
    # 如果数据点数大于5000，则设置ifault为2
    if n > 5000:
        ifault = 2

    # 计算SA和SX的均值
    SA /= n1
    SX /= n1
    # 初始化SSA，SSX和SAX
    SSA, SSX, SAX = 0., 0., 0.
    ind2 = n - 1
    # 计算SSA，SSX和SAX
    for ind1 in range(n1):
        if ind1 != ind2:
            ASA = (-1 if ind1 < ind2 else 1)*a[min(ind1, ind2)] - SA
        else:
            ASA = -SA

        XSX = x[ind1]/RANGE - SX
        SSA += ASA * ASA
        SSX += XSX * XSX
        SAX += ASA * XSX
        ind2 -= 1

    # 计算SSA和SSX的平方根
    SSASSX = sqrt(SSA * SSX)
    # 计算w1
    w1 = (SSASSX - SAX) * (SSASSX + SAX)/(SSA * SSX)
    w = 1 - w1

    # 计算W的显著性水平（N=3时为精确值）
    if n == 3:
        # 当n=3时，计算pw的值，注意对小p值的处理
        if w < 0.75:
            return 0.75, 0., ifault
        else:
            pw = 1. - PI6 * acos(sqrt(w))
            return w, pw, ifault

    # 计算y和XX的自然对数
    y = log(w1)
    XX = log(an)
    # 根据数据点数选择不同的系数
    if n <= 11:
        gamma = _poly(g, 2, an)
        if y >= gamma:
            return w, SMALL, ifault
        y = -log(gamma - y)
        m = _poly(c3, 4, an)
        s = exp(_poly(c4, 4, an))
    else:
        m = _poly(c5, 4, XX)
        s = exp(_poly(c6, 3, XX))

    # 如果存在censoring数据点，则进行修正
    if ncens > 0:
        ld = -log(delta)
        bf = 1 + XX*BF1
        Z90F = Z90 + bf * _poly(c7, 2, XX90 ** XX)**ld
        Z95F = Z95 + bf * _poly(c8, 2, XX95 ** XX)**ld
        Z99F = Z99 + bf * _poly(c9, 2, XX)**ld
        ZFM = (Z90F + Z95F + Z99F)/3.
        ZSD = (Z90*(Z90F-ZFM)+Z95*(Z95F-ZFM)+Z99*(Z99F-ZFM))/ZSS
        ZBAR = ZFM - ZSD * ZM
        m += ZBAR * s
        s *= ZSD

    # 计算pw值，根据标准正态分布计算
    pw = _alnorm((y-m)/s, upper)

    return w, pw, ifault
cdef double _alnorm(double x, bint upper) noexcept nogil:
    """
    Helper function for swilk.

    Evaluates the tail area of the standardized normal curve from x to inf
    if upper is True or from -inf to x if upper is False

    Modification has been done to the Fortran version in November 2001 with the
    following note;

        MODIFY UTZERO.  ALTHOUGH NOT NECESSARY
        WHEN USING ALNORM FOR SIMPLY COMPUTING PERCENT POINTS,
        EXTENDING RANGE IS HELPFUL FOR USE WITH FUNCTIONS THAT
        USE ALNORM IN INTERMEDIATE COMPUTATIONS.

    The change is shown below as a commented utzero definition
    """
    cdef:
        double A1, A2, A3, A4, A5, A6, A7
        double B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12
        double ltone = 7.
        # double utzero = 18.66
        double utzero = 38.  # Constant defining upper limit for ALNORM computation
        double con = 1.28  # Constant for checking if z <= con
        double y, z, temp

    A1 = 0.398942280444
    A2 = 0.399903438504
    A3 = 5.75885480458
    A4 = 29.8213557808
    A5 = 2.62433121679
    A6 = 48.6959930692
    A7 = 5.92885724438
    B1 = 0.398942280385
    B2 = 3.8052e-8
    B3 = 1.00000615302
    B4 = 3.98064794e-4
    B5 = 1.98615381364
    B6 = 0.151679116635
    B7 = 5.29330324926
    B8 = 4.8385912808
    B9 = 15.1508972451
    B10 = 0.742380924027
    B11 = 30.789933034
    B12 = 3.99019417011
    z = x
    if not (z > 0):  # Check if z is not positive (handles NaNs)
        upper = False  # Set upper to False since z is negative
        z = -z  # Take the absolute value of z
    if not ((z <= ltone) or (upper and z <= utzero)):  # Check conditions for early return
        return 0. if upper else 1.  # Return 0. if upper is True, else return 1.
    y = 0.5 * z * z  # Compute y as half of z squared
    if z <= con:  # Check if z is less than or equal to con
        temp = 0.5 - z * (A1 - A2 * y / (y + A3 - A4 / (y + A5 + A6 / (y + A7))))  # Compute temp using constants A1-A7
    else:
        temp = B1 * exp(-y) / (z - B2 + B3 / (z + B4 + B5 / (z - B6 + B7 /
                             (z + B8 - B9 / (z + B10 + B11 / (z + B12))))))  # Compute temp using constants B1-B12

    return temp if upper else (1-temp)  # Return temp if upper is True, else return (1-temp)


cdef double _ppnd(double p) noexcept:
    """
    Helper function for swilk. Unused ifault return is
    commented out from return statements.
    """
    cdef:
        double A0, A1, A2, A3, B1, B2, B3, B4, C0, C1, C2, C3, D1, D2
        double q, r, temp
        double split = 0.42  # Constant split for determining calculation path

    # cdef int ifault = 0
    A0 = 2.50662823884
    A1 = -18.61500062529
    A2 = 41.39119773534
    A3 = -25.44106049637
    B1 = -8.47351093090
    B2 = 23.08336743743
    B3 = -21.06224101826
    B4 = 3.13082909833
    C0 = -2.78718931138
    C1 = -2.29796479134
    C2 = 4.85014127135
    C3 = 2.32121276858
    D1 = 3.54388924762
    D2 = 1.63706781897

    q = p - 0.5  # Compute q as p - 0.5
    if abs(q) <= split:  # Check if absolute value of q is less than or equal to split
        r = q * q  # Compute r as q squared
        temp = q * (((A3 * r + A2) * r + A1) * r + A0)  # Compute temp using constants A0-A3
        temp = temp / ((((B4 * r + B3) * r + B2) * r + B1) * r + 1.)  # Further compute temp
        return temp  #, 0

    r = p  # Set r to p
    if q > 0:  # Check if q is greater than 0
        r = 1 - p  # Set r to 1 - p
    if r > 0:  # Check if r is greater than 0
        r = sqrt(-log(r))  # Compute r as square root of -log(r)
    else:
        return 0.  #, 1

    temp = (((C3 * r + C2) * r + C1) * r + C0)  # Compute temp using constants C0-C3
    temp /= (D2 * r + D1) * r + 1.  # Further compute temp
    return (-temp if q < 0 else temp)  #, 0


cdef double _poly(const double[::1] c, int nord, double x) noexcept nogil:
    """
    Helper function for swilk that evaluates a polynomial at x.

    The polynomial is defined by its coefficients in c with length nord+1.
    """
    # 定义一个用于 swilk 函数的辅助函数，用于评估多项式。
    # 由于系数数组是以如下形式给出的：
    # [c0, cn, cn-1, ..., c2, c1]
    # 因此采用倒序循环。
    """
    Helper function for swilk function that evaluates polynomials.
    For some reason, the coefficients are given as

        [c0, cn, cn-1, ..., c2, c1]

    hence the backwards loop.
    """
    # 声明 C 语言风格的 double 类型变量 res 和 p
    cdef double res, p
    # 将第一个系数 c[0] 赋给 res
    res = c[0]
    # 如果 nord 等于 1，则直接返回 res
    if nord == 1:
        return res

    # 计算 p = x * c[nord-1]
    p = x * c[nord-1]
    # 如果 nord 等于 2，则返回 res + p
    if nord == 2:
        return res + p

    # 倒序遍历范围为 [nord-2, 0)，步长为 -1
    for ind in range(nord-2, 0, -1):
        # 更新 p = (p + c[ind]) * x
        p = (p + c[ind]) * x
    # 将计算结果加到 res 上
    res += p
    # 返回最终结果 res
    return res
```