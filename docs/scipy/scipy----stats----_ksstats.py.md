# `D:\src\scipysrc\scipy\scipy\stats\_ksstats.py`

```
# Compute the two-sided one-sample Kolmogorov-Smirnov Prob(Dn <= d) where:
#    D_n = sup_x{|F_n(x) - F(x)|},
#    F_n(x) is the empirical CDF for a sample of size n {x_i: i=1,...,n},
#    F(x) is the CDF of a probability distribution.
#
# Exact methods:
# Prob(D_n >= d) can be computed via a matrix algorithm of Durbin[1]
#   or a recursion algorithm due to Pomeranz[2].
# Marsaglia, Tsang & Wang[3] gave a computation-efficient way to perform
#   the Durbin algorithm.
#   D_n >= d <==>  D_n+ >= d or D_n- >= d (the one-sided K-S statistics), hence
#   Prob(D_n >= d) = 2*Prob(D_n+ >= d) - Prob(D_n+ >= d and D_n- >= d).
#   For d > 0.5, the latter intersection probability is 0.
#
# Approximate methods:
# For d close to 0.5, ignoring that intersection term may still give a
#   reasonable approximation.
# Li-Chien[4] and Korolyuk[5] gave an asymptotic formula extending
# Kolmogorov's initial asymptotic, suitable for large d. (See
#   scipy.special.kolmogorov for that asymptotic)
# Pelz-Good[6] used the functional equation for Jacobi theta functions to
#   transform the Li-Chien/Korolyuk formula produce a computational formula
#   suitable for small d.
#
# Simard and L'Ecuyer[7] provided an algorithm to decide when to use each of
#   the above approaches and it is that which is used here.
#
# Other approaches:
# Carvalho[8] optimizes Durbin's matrix algorithm for large values of d.
# Moscovich and Nadler[9] use FFTs to compute the convolutions.
#
# References:
# [1] Durbin J (1968).
#     "The Probability that the Sample Distribution Function Lies Between Two
#     Parallel Straight Lines."
#     Annals of Mathematical Statistics, 39, 398-411.
# [2] Pomeranz J (1974).
#     "Exact Cumulative Distribution of the Kolmogorov-Smirnov Statistic for
#     Small Samples (Algorithm 487)."
#     Communications of the ACM, 17(12), 703-704.
# [3] Marsaglia G, Tsang WW, Wang J (2003).
#     "Evaluating Kolmogorov's Distribution."
#     Journal of Statistical Software, 8(18), 1-4.
# [4] LI-CHIEN, C. (1956).
#     "On the exact distribution of the statistics of A. N. Kolmogorov and
#     their asymptotic expansion."
#     Acta Matematica Sinica, 6, 55-81.
# [5] KOROLYUK, V. S. (1960).
#     "Asymptotic analysis of the distribution of the maximum deviation in
#     the Bernoulli scheme."
#     Theor. Probability Appl., 4, 339-366.
# [6] Pelz W, Good IJ (1976).
#     "Approximating the Lower Tail-areas of the Kolmogorov-Smirnov One-sample
#     Statistic."
#     Journal of the Royal Statistical Society, Series B, 38(2), 152-156.
# [7] Simard, R., L'Ecuyer, P. (2011)
#       "Computing the Two-Sided Kolmogorov-Smirnov Distribution",
#       Journal of Statistical Software, Vol 39, 11, 1-18.
# [8] Carvalho, Luis (2015)
#     "An Improved Evaluation of Kolmogorov's Distribution"
#     Journal of Statistical Software, Code Snippets; Vol 65(3), 1-8.
# [9] Amit Moscovich, Boaz Nadler (2017)
#     "Fast calculation of boundary crossing probabilities for Poisson
# 导入所需的库
import numpy as np  # 导入NumPy库，用于数值计算
import scipy.special  # 导入SciPy库中的special模块，包含特殊函数
import scipy.special._ufuncs as scu  # 导入SciPy库中的_ufuncs模块，包含特殊函数的内部实现
from scipy._lib._finite_differences import _derivative  # 从SciPy库的_lib._finite_differences模块中导入_derivative函数

# 常量定义
_E128 = 128  # 定义整数常量_E128为128
_EP128 = np.ldexp(np.longdouble(1), _E128)  # 计算2的128次幂的长双精度数
_EM128 = np.ldexp(np.longdouble(1), -_E128)  # 计算2的-128次幂的长双精度数

_SQRT2PI = np.sqrt(2 * np.pi)  # 计算2π的平方根
_LOG_2PI = np.log(2 * np.pi)  # 计算2π的自然对数
_MIN_LOG = -708  # 定义最小的可接受对数值
_SQRT3 = np.sqrt(3)  # 计算3的平方根
_PI_SQUARED = np.pi ** 2  # 计算π的平方
_PI_FOUR = np.pi ** 4  # 计算π的四次方
_PI_SIX = np.pi ** 6  # 计算π的六次方

# Stirling系数列表
# [Lifted from _loggamma.pxd.] If B_m are the Bernoulli numbers,
# then Stirling coeffs are B_{2j}/(2j)/(2j-1) for j=8,...1.
_STIRLING_COEFFS = [-2.955065359477124183e-2, 6.4102564102564102564e-3,
                    -1.9175269175269175269e-3, 8.4175084175084175084e-4,
                    -5.952380952380952381e-4, 7.9365079365079365079e-4,
                    -2.7777777777777777778e-3, 8.3333333333333333333e-2]


def _log_nfactorial_div_n_pow_n(n):
    # 计算 n! / n**n 的对数值
    #    = (n-1)! / n**(n-1)
    # 使用Stirling近似公式，但将 n*log(n) 提前避免减法取消
    #    = log(n)/2 - n + log(sqrt(2pi)) + sum B_{2j}/(2j)/(2j-1)/n**(2j-1)
    rn = 1.0/n
    return np.log(n)/2 - n + _LOG_2PI/2 + rn * np.polyval(_STIRLING_COEFFS, rn/n)


def _clip_prob(p):
    """将概率裁剪到区间 0<=p<=1."""
    return np.clip(p, 0.0, 1.0)


def _select_and_clip_prob(cdfprob, sfprob, cdf=True):
    """选择CDF或SF，然后将其裁剪到区间 0<=p<=1."""
    p = np.where(cdf, cdfprob, sfprob)
    return _clip_prob(p)


def _kolmogn_DMTW(n, d, cdf=True):
    r"""使用MTW方法计算Kolmogorov分布的CDF: Pr(D_n <= d)。

    Durbin (1968); Marsaglia, Tsang, Wang (2003). [1], [3].
    """
    # 将 d = (k-h)/n，其中 k 是正整数且 0 <= h < 1
    # 生成大小为 m*m 的初始矩阵 H，其中 m=(2k-1)
    # 计算 (n!/n^n) * H^n 的第 k 行，并缩放中间结果。
    # 需要内存 O(m^2) 和计算时间 O(m^2 log(n))。
    # 适合小的 m。

    if d >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf)
    nd = n * d
    if nd <= 0.5:
        return _select_and_clip_prob(0.0, 1.0, cdf)
    k = int(np.ceil(nd))
    h = k - nd
    m = 2 * k - 1

    H = np.zeros([m, m])

    # 初始化：v 是 H 的第一列（也是 H 的最后一行）
    #  v[j] = (1-h^(j+1)/(j+1)!  (除了 v[-1])
    #  w[j] = 1/(j)!
    # q = H 的第 k 行（实际上是 i!/n^i*H^i）
    intm = np.arange(1, m + 1)
    v = 1.0 - h ** intm
    w = np.empty(m)
    fac = 1.0
    for j in intm:
        w[j - 1] = fac
        fac /= j  # 这可能会下溢。不过不是问题。
        v[j - 1] *= fac
    tt = max(2 * h - 1.0, 0)**m - 2*h**m
    v[-1] = (1.0 + tt) * fac

    for i in range(1, m):
        H[i - 1:, i] = w[:m - i + 1]
    H[:, 0] = v
    H[-1, :] = np.flip(v, axis=0)

    Hpwr = np.eye(np.shape(H)[0])  # 保存 H 的中间幂次
    nn = n
    expnt = 0  # Hpwr 的缩放
    Hexpnt = 0  # 初始化 H 的缩放指数为 0，用于记录 H 的缩放次数
    while nn > 0:
        # 如果 nn 是奇数
        if nn % 2:
            # 使用 H 矩阵的乘法更新 Hpwr 矩阵，并累加 Hexpnt 到 expnt 中
            Hpwr = np.matmul(Hpwr, H)
            expnt += Hexpnt
        # 计算 H 的平方
        H = np.matmul(H, H)
        # Hexpnt 增加两倍，用于下一次迭代
        Hexpnt *= 2
        # 根据需要进行缩放
        # 如果 H[k-1, k-1] 的绝对值大于 _EP128
        if np.abs(H[k - 1, k - 1]) > _EP128:
            # 将 H 缩小为 _EP128，同时更新 Hexpnt
            H /= _EP128
            Hexpnt += _E128
        # 将 nn 除以 2
        nn = nn // 2

    p = Hpwr[k - 1, k - 1]

    # 乘以 n!/n^n
    for i in range(1, n + 1):
        # 更新 p 的值为 i * p / n
        p = i * p / n
        # 如果 p 的绝对值小于 _EM128
        if np.abs(p) < _EM128:
            # 将 p 扩大为 _EP128，同时更新 expnt
            p *= _EP128
            expnt -= _E128

    # 还原缩放
    if expnt != 0:
        # 使用 expnt 还原 p 的值
        p = np.ldexp(p, expnt)

    # 调用 _select_and_clip_prob 函数，返回调用结果
    return _select_and_clip_prob(p, 1.0 - p, cdf)
# 计算第 i 行的区间端点
def _pomeranz_compute_j1j2(i, n, ll, ceilf, roundf):
    """Compute the endpoints of the interval for row i."""
    if i == 0:
        # 当 i 为 0 时，计算 j1 和 j2 的值
        j1, j2 = -ll - ceilf - 1, ll + ceilf - 1
    else:
        # 根据 Pomeranz 算法计算 j1 和 j2 的值
        # i + 1 = 2*ip1div2 + ip1mod2
        ip1div2, ip1mod2 = divmod(i + 1, 2)
        if ip1mod2 == 0:  # 如果 i 是奇数
            if ip1div2 == n + 1:
                j1, j2 = n - ll - ceilf - 1, n + ll + ceilf - 1
            else:
                j1, j2 = ip1div2 - 1 - ll - roundf - 1, ip1div2 + ll - 1 + ceilf - 1
        else:
            j1, j2 = ip1div2 - 1 - ll - 1, ip1div2 + ll + roundf - 1

    # 返回 j1 和 j2 的最大最小值
    return max(j1 + 2, 0), min(j2, n)


def _kolmogn_Pomeranz(n, x, cdf=True):
    r"""Computes Pr(D_n <= d) using the Pomeranz recursion algorithm.

    Pomeranz (1974) [2]
    """

    # V 是一个 n*(2n+2) 的矩阵。
    # 每一行是前一行与来自泊松分布的概率的卷积。
    # 所需的累积分布函数概率是 n! * V[n-1, 2n+1]（最后一行的最后一个条目）。
    # 在任何给定阶段只需要两行：V0 和 V1，交替使用。
    # 每行只有少量（连续的）条目可以是非零的。
    # j1 和 j2 跟踪每行中非零条目的起始和结束。
    # 根据需要缩放中间结果。
    t = n * x
    ll = int(np.floor(t))
    f = 1.0 * (t - ll)  # t 的小数部分
    g = min(f, 1.0 - f)
    ceilf = (1 if f > 0 else 0)
    roundf = (1 if f > 0.5 else 0)
    npwrs = 2 * (ll + 1)    # 卷积中可能需要的最大幂次数
    gpower = np.empty(npwrs)  # gpower = (g/n)^m/m!
    twogpower = np.empty(npwrs)  # twogpower = (2g/n)^m/m!
    onem2gpower = np.empty(npwrs)  # onem2gpower = ((1-2g)/n)^m/m!
    # gpower 等几乎是泊松分布概率，只是缺少归一化因子。

    gpower[0] = 1.0
    twogpower[0] = 1.0
    onem2gpower[0] = 1.0
    expnt = 0
    g_over_n, two_g_over_n, one_minus_two_g_over_n = g/n, 2*g/n, (1 - 2*g)/n
    for m in range(1, npwrs):
        gpower[m] = gpower[m - 1] * g_over_n / m
        twogpower[m] = twogpower[m - 1] * two_g_over_n / m
        onem2gpower[m] = onem2gpower[m - 1] * one_minus_two_g_over_n / m

    V0 = np.zeros([npwrs])
    V1 = np.zeros([npwrs])
    V1[0] = 1  # 第一行
    V0s, V1s = 0, 0  # 两行的起始索引

    # 计算第 0 行的 j1 和 j2
    j1, j2 = _pomeranz_compute_j1j2(0, n, ll, ceilf, roundf)
    for i in range(1, 2 * n + 2):
        # 保留上一次迭代的 j1, V1, V1s, V0s
        k1 = j1
        V0, V1 = V1, V0  # 交换 V0 和 V1 的引用
        V0s, V1s = V1s, V0s  # 交换 V0s 和 V1s 的引用
        V1.fill(0.0)  # 将 V1 数组填充为 0
        j1, j2 = _pomeranz_compute_j1j2(i, n, ll, ceilf, roundf)  # 计算 j1, j2
        if i == 1 or i == 2 * n + 1:
            pwrs = gpower
        else:
            pwrs = (twogpower if i % 2 else onem2gpower)
        ln2 = j2 - k1 + 1  # 计算 ln2
        if ln2 > 0:
            # 计算卷积，使用 V0 和 pwrs 的部分数据
            conv = np.convolve(V0[k1 - V0s:k1 - V0s + ln2], pwrs[:ln2])
            conv_start = j1 - k1  # conv 中用于开始的索引
            conv_len = j2 - j1 + 1  # conv 中要使用的条目数
            V1[:conv_len] = conv[conv_start:conv_start + conv_len]  # 将卷积结果放入 V1 中的对应部分
            # 缩放以避免下溢
            if 0 < np.max(V1) < _EM128:
                V1 *= _EP128
                expnt -= _E128
            V1s = V0s + j1 - k1  # 更新 V1s

    # 乘以 n!
    ans = V1[n - V1s]
    for m in range(1, n + 1):
        if np.abs(ans) > _EP128:
            ans *= _EM128
            expnt += _E128
        ans *= m

    # 恢复任何中间缩放
    if expnt != 0:
        ans = np.ldexp(ans, expnt)
    ans = _select_and_clip_prob(ans, 1.0 - ans, cdf)
    return ans
# 计算 Pelz-Good 方法下的 Kolmogorov-Smirnov 统计量的概率近似值
def _kolmogn_PelzGood(n, x, cdf=True):
    """Computes the Pelz-Good approximation to Prob(Dn <= x) with 0<=x<=1.

    Start with Li-Chien, Korolyuk approximation:
        Prob(Dn <= x) ~ K0(z) + K1(z)/sqrt(n) + K2(z)/n + K3(z)/n**1.5
    where z = x*sqrt(n).
    Transform each K_(z) using Jacobi theta functions into a form suitable
    for small z.
    Pelz-Good (1976). [6]
    """
    # 如果 x 小于等于 0.0，则返回概率为 0.0 或 1.0 的结果，根据 cdf 参数选择
    if x <= 0.0:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    # 如果 x 大于等于 1.0，则返回概率为 1.0 或 0.0 的结果，根据 cdf 参数选择
    if x >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf=cdf)

    # 计算 z = x * sqrt(n)
    z = np.sqrt(n) * x
    zsquared, zthree, zfour, zsix = z**2, z**3, z**4, z**6

    # 计算 qlog，并检查是否小于阈值 _MIN_LOG，如果是，则返回概率为 0.0 或 1.0 的结果，根据 cdf 参数选择
    qlog = -_PI_SQUARED / 8 / zsquared
    if qlog < _MIN_LOG:  # 当 z 大约为 0.041743441416853426 时
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)

    q = np.exp(qlog)

    # 计算 K1、K2 和 K3 的系数
    k1a = -zsquared
    k1b = _PI_SQUARED / 4

    k2a = 6 * zsix + 2 * zfour
    k2b = (2 * zfour - 5 * zsquared) * _PI_SQUARED / 4
    k2c = _PI_FOUR * (1 - 2 * zsquared) / 16

    k3d = _PI_SIX * (5 - 30 * zsquared) / 64
    k3c = _PI_FOUR * (-60 * zsquared + 212 * zfour) / 16
    k3b = _PI_SQUARED * (135 * zfour - 96 * zsix) / 4
    k3a = -30 * zsix - 90 * z**8

    K0to3 = np.zeros(4)
    # 使用 Horner 方案计算 sum c_i * q^(i^2)
    maxk = int(np.ceil(16 * z / np.pi))
    for k in range(maxk, 0, -1):
        m = 2 * k - 1
        msquared, mfour, msix = m**2, m**4, m**6
        qpower = np.power(q, 8 * k)
        coeffs = np.array([1.0,
                           k1a + k1b*msquared,
                           k2a + k2b*msquared + k2c*mfour,
                           k3a + k3b*msquared + k3c*mfour + k3d*msix])
        K0to3 *= qpower
        K0to3 += coeffs
    K0to3 *= q
    K0to3 *= _SQRT2PI
    # z**10 > 0 as z > 0.04
    K0to3 /= np.array([z, 6 * zfour, 72 * z**7, 6480 * z**10])

    # 计算第二个求和项中的额外部分
    q = np.exp(-_PI_SQUARED / 2 / zsquared)
    ks = np.arange(maxk, 0, -1)
    ksquared = ks ** 2
    sqrt3z = _SQRT3 * z
    kspi = np.pi * ks
    qpwers = q ** ksquared
    k2extra = np.sum(ksquared * qpwers)
    k2extra *= _PI_SQUARED * _SQRT2PI/(-36 * zthree)
    K0to3[2] += k2extra
    k3extra = np.sum((sqrt3z + kspi) * (sqrt3z - kspi) * ksquared * qpwers)
    k3extra *= _PI_SQUARED * _SQRT2PI/(216 * zsix)
    K0to3[3] += k3extra
    powers_of_n = np.power(n * 1.0, np.arange(len(K0to3)) / 2.0)
    K0to3 /= powers_of_n

    # 如果不是累积分布函数（cdf=False），则对 K0to3 进行反向处理
    if not cdf:
        K0to3 *= -1
        K0to3[0] += 1

    # 计算 K0to3 的总和
    Ksum = sum(K0to3)
    return Ksum


# 计算两侧 Kolmogorov-Smirnov 统计量的累积分布函数（或生存函数）
def _kolmogn(n, x, cdf=True):
    """Computes the CDF(or SF) for the two-sided Kolmogorov-Smirnov statistic.

    x must be of type float, n of type integer.

    Simard & L'Ecuyer (2011) [7].
    """
    if np.isnan(n):
        return n  # 如果 n 是 NaN，则返回原始的 NaN 类型
    if int(n) != n or n <= 0:
        return np.nan  # 如果 n 不是整数或者小于等于 0，则返回 NaN
    if x >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf=cdf)  # 如果 x 大于等于 1.0，则调用函数进行概率选择和裁剪
    if x <= 0.0:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)  # 如果 x 小于等于 0.0，则调用函数进行概率选择和裁剪
    t = n * x  # 计算 t = n * x
    if t <= 1.0:  # 如果 t 小于等于 1.0
        if t <= 0.5:  # 如果 t 小于等于 0.5
            return _select_and_clip_prob(0.0, 1.0, cdf=cdf)  # 调用函数进行概率选择和裁剪
        if n <= 140:  # 如果 n 小于等于 140
            prob = np.prod(np.arange(1, n+1) * (1.0/n) * (2*t - 1))  # 使用给定公式计算概率
        else:
            prob = np.exp(_log_nfactorial_div_n_pow_n(n) + n * np.log(2*t-1))  # 使用给定公式计算概率
        return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)  # 调用函数进行概率选择和裁剪
    if t >= n - 1:  # 如果 t 大于等于 n - 1
        prob = 2 * (1.0 - x)**n  # 使用给定公式计算概率
        return _select_and_clip_prob(1 - prob, prob, cdf=cdf)  # 调用函数进行概率选择和裁剪
    if x >= 0.5:  # 如果 x 大于等于 0.5
        prob = 2 * scipy.special.smirnov(n, x)  # 使用给定公式计算概率
        return _select_and_clip_prob(1.0 - prob, prob, cdf=cdf)  # 调用函数进行概率选择和裁剪

    nxsquared = t * x  # 计算 nxsquared = t * x
    if n <= 140:  # 如果 n 小于等于 140
        if nxsquared <= 0.754693:  # 如果 nxsquared 小于等于 0.754693
            prob = _kolmogn_DMTW(n, x, cdf=True)  # 调用函数计算概率
            return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)  # 调用函数进行概率选择和裁剪
        if nxsquared <= 4:  # 如果 nxsquared 小于等于 4
            prob = _kolmogn_Pomeranz(n, x, cdf=True)  # 调用函数计算概率
            return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)  # 调用函数进行概率选择和裁剪
        # 现在使用 Miller 近似的 2 * smirnov
        prob = 2 * scipy.special.smirnov(n, x)  # 使用给定公式计算概率
        return _select_and_clip_prob(1.0 - prob, prob, cdf=cdf)  # 调用函数进行概率选择和裁剪

    # 将 CDF 和 SF 分开，因为它们在 nxsquared 上有不同的截断。
    if not cdf:  # 如果不是累积分布函数（CDF）
        if nxsquared >= 370.0:  # 如果 nxsquared 大于等于 370.0
            return 0.0  # 返回 0.0
        if nxsquared >= 2.2:  # 如果 nxsquared 大于等于 2.2
            prob = 2 * scipy.special.smirnov(n, x)  # 使用给定公式计算概率
            return _clip_prob(prob)  # 调用函数裁剪概率
        # 继续计算 SF 作为 1.0-CDF
    if nxsquared >= 18.0:  # 如果 nxsquared 大于等于 18.0
        cdfprob = 1.0  # 累积分布函数的概率为 1.0
    elif n <= 100000 and n * x**1.5 <= 1.4:  # 如果 n 小于等于 100000 并且 n * x**1.5 小于等于 1.4
        cdfprob = _kolmogn_DMTW(n, x, cdf=True)  # 调用函数计算概率
    else:
        cdfprob = _kolmogn_PelzGood(n, x, cdf=True)  # 调用函数计算概率
    return _select_and_clip_prob(cdfprob, 1.0 - cdfprob, cdf=cdf)  # 调用函数进行概率选择和裁剪
# 计算两侧Kolmogorov-Smirnov统计量的概率密度函数（PDF）。

# 参数 n 必须是整数类型，x 必须是浮点类型。
def _kolmogn_p(n, x):
    if np.isnan(n):
        return n  # 返回与原来nan相同类型的值
    if int(n) != n or n <= 0:
        return np.nan  # 如果 n 不是正整数，返回nan
    if x >= 1.0 or x <= 0:
        return 0  # 如果 x 超出有效范围 [0, 1]，返回0
    t = n * x
    if t <= 1.0:
        # 对于 t <= 1.0 的情况，使用Ruben-Gambino公式计算PDF
        if t <= 0.5:
            return 0.0  # 如果 t <= 0.5，直接返回0.0
        if n <= 140:
            prd = np.prod(np.arange(1, n) * (1.0 / n) * (2 * t - 1))
        else:
            prd = np.exp(_log_nfactorial_div_n_pow_n(n) + (n-1) * np.log(2 * t - 1))
        return prd * 2 * n**2  # 计算并返回PDF的结果
    if t >= n - 1:
        # 对于 t >= n - 1 的情况，使用Ruben-Gambino公式计算PDF
        return 2 * (1.0 - x) ** (n-1) * n  # 计算并返回PDF的结果
    if x >= 0.5:
        return 2 * scipy.stats.ksone.pdf(x, n)  # 使用SciPy的ksone模块计算PDF

    # 对于其它情况，计算数值导数
    delta = x / 2.0**16  # 取一个小的增量
    delta = min(delta, x - 1.0/n)
    delta = min(delta, 0.5 - x)

    def _kk(_x):
        return kolmogn(n, _x)

    return _derivative(_kk, x, dx=delta, order=5)  # 返回导数的结果


# 计算kolmogn的P-PF/ISF。

# 参数 n 必须是整数类型，且 n >= 1；p 是累积分布函数（CDF），q 是生存函数（SF），满足 p + q = 1。
def _kolmogni(n, p, q):
    if np.isnan(n):
        return n  # 返回与原来nan相同类型的值
    if int(n) != n or n <= 0:
        return np.nan  # 如果 n 不是正整数，返回nan
    if p <= 0:
        return 1.0/n  # 如果 p <= 0，返回1/n
    if q <= 0:
        return 1.0  # 如果 q <= 0，返回1.0
    delta = np.exp((np.log(p) - scipy.special.loggamma(n+1))/n)  # 计算 delta
    if delta <= 1.0/n:
        return (delta + 1.0 / n) / 2  # 根据 delta 的值返回结果
    x = -np.expm1(np.log(q/2.0)/n)  # 计算 x
    if x >= 1 - 1.0/n:
        return x  # 如果 x 符合条件，直接返回 x
    x1 = scu._kolmogci(p)/np.sqrt(n)  # 计算 x1

    def _f(x):
        return _kolmogn(n, x) - p  # 定义函数 _f(x)

    return scipy.optimize.brentq(_f, 1.0/n, x1, xtol=1e-14)  # 使用 brentq 方法求解


# 计算两侧Kolmogorov-Smirnov分布的累积分布函数（CDF）或生存函数（SF）。

# 参数 n 可以是整数或者数组，x 是K-S统计量，取值范围在 [0, 1] 之间的浮点数，cdf 为 True 表示计算CDF，默认为 True。
def kolmogn(n, x, cdf=True):
    # 使用 numpy 的迭代器处理 n, x, cdf
    it = np.nditer([n, x, cdf, None],
                   op_dtypes=[None, np.float64, np.bool_, np.float64])
    for _n, _x, _cdf, z in it:
        # 遍历迭代器 `it` 返回的每个元组 (_n, _x, _cdf, z)
        
        if np.isnan(_n):
            # 如果 _n 是 NaN（Not a Number），则将 z 数组全部设置为 _n，然后继续下一轮循环
            z[...] = _n
            continue
        
        if int(_n) != _n:
            # 如果 _n 不是整数，则抛出 ValueError 异常，指明 _n 的值不是整数
            raise ValueError(f'n is not integral: {_n}')
        
        # 调用 _kolmogn 函数处理 _n、_x、_cdf，然后将返回的结果赋给 z 数组
        z[...] = _kolmogn(int(_n), _x, cdf=_cdf)
    
    # 将迭代器 `it` 的最后一个操作数作为结果赋给 result
    result = it.operands[-1]
    
    # 返回最终结果
    return result
# 计算两侧 Kolmogorov-Smirnov 分布的概率密度函数（PDF）

def kolmognp(n, x):
    """Computes the PDF for the two-sided Kolmogorov-Smirnov distribution.

    Parameters
    ----------
    n : integer, array_like
        样本数
    x : float, array_like
        K-S 统计量，取值范围为 0 到 1 的浮点数

    Returns
    -------
    pdf : ndarray
        指定位置的概率密度函数值

    返回值的形状由 numpy 广播 n 和 x 的结果决定。
    """
    it = np.nditer([n, x, None])
    for _n, _x, z in it:
        if np.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        z[...] = _kolmogn_p(int(_n), _x)
    result = it.operands[-1]
    return result


# 计算两侧 Kolmogorov-Smirnov 分布的百分点函数（PPF）或反向百分点函数（ISF）

def kolmogni(n, q, cdf=True):
    """Computes the PPF(or ISF) for the two-sided Kolmogorov-Smirnov distribution.

    Parameters
    ----------
    n : integer, array_like
        样本数
    q : float, array_like
        概率值，取值范围为 0 到 1 的浮点数
    cdf : bool, optional
        是否计算百分点函数（默认为 True），或反向百分点函数（ISF）

    Returns
    -------
    ppf : ndarray
        指定位置的百分点函数值（或反向百分点函数值，如果 cdf 为 False）

    返回值的形状由 numpy 广播 n 和 q 的结果决定。
    """
    it = np.nditer([n, q, cdf, None])
    for _n, _q, _cdf, z in it:
        if np.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        _pcdf, _psf = (_q, 1-_q) if _cdf else (1-_q, _q)
        z[...] = _kolmogni(int(_n), _pcdf, _psf)
    result = it.operands[-1]
    return result
```