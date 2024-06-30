# `D:\src\scipysrc\scipy\scipy\stats\_stats.pyx`

```
# 启用 Cython 的 cpow 编译器指令，用于支持复数的幂运算
# 导入必要的 Cython 类型和函数声明
from cpython cimport bool
from libc cimport math
from libc.math cimport NAN, INFINITY, M_PI as PI
cimport cython
cimport numpy as np
from numpy cimport ndarray, int64_t, float64_t, intp_t

# 导入警告模块和科学计算库
import warnings
import numpy as np
import scipy.stats, scipy.special
from scipy.linalg import solve_triangular
cimport scipy.special.cython_special as cs

# 初始化 NumPy 数组支持
np.import_array()

# 定义 von_mises_cdf_series 函数，计算 von Mises 分布的累积分布函数的级数部分
cdef double von_mises_cdf_series(double k, double x, unsigned int p) noexcept:
    # 计算正弦和余弦值
    cdef double s, c, sn, cn, R, V
    s = math.sin(x)
    c = math.cos(x)
    sn = math.sin(p * x)
    cn = math.cos(p * x)
    R = 0
    V = 0
    # 执行级数求和
    for n in range(p - 1, 0, -1):
        sn, cn = sn * c - cn * s, cn * c + sn * s
        R = k / (2 * n + k * R)
        V = R * (sn / n + V)

    # 使用 Cython 的 cdivision 来处理除法，返回 von Mises 分布的累积分布函数值
    with cython.cdivision(True):
        return 0.5 + x / (2 * PI) + V / PI


# 定义 von_mises_cdf_normalapprox 函数，使用正态分布近似计算 von Mises 分布的累积分布函数
cdef von_mises_cdf_normalapprox(k, x):
    # 预先计算常数 sqrt(2/pi)
    cdef double SQRT2_PI = 0.79788456080286535588  # sqrt(2/pi)

    # 计算参数 b 和 z
    b = SQRT2_PI / scipy.special.i0e(k)  # 检查 k 是否为负数
    z = b * np.sin(x / 2.)
    # 返回正态分布的累积分布函数值
    return scipy.stats.norm.cdf(z)


# 定义 von_mises_cdf 函数，计算 von Mises 分布的累积分布函数
@cython.boundscheck(False)
def von_mises_cdf(k_obj, x_obj):
    # 声明变量和初始化
    cdef double[:] temp, temp_xs, temp_ks
    cdef unsigned int i, p
    cdef double a1, a2, a3, a4, CK
    cdef np.ndarray k = np.asarray(k_obj)
    cdef np.ndarray x = np.asarray(x_obj)
    cdef bint zerodim = k.ndim == 0 and x.ndim == 0

    # 将 k 和 x 至少转换为一维数组
    k = np.atleast_1d(k)
    x = np.atleast_1d(x)
    # 对 x 进行周期性调整，使其落在 [0, 2*pi) 范围内
    ix = np.round(x / (2 * PI))
    x = x - ix * (2 * PI)

    # 初始化常数和系数
    CK = 50
    a1, a2, a3, a4 = 28., 0.5, 100., 5.

    # 广播数组以匹配大小
    bx, bk = np.broadcast_arrays(x, k)
    result = np.empty_like(bx, float)

    # 处理小 k 的情况，使用级数展开计算 von Mises 分布的累积分布函数
    c_small_k = bk < CK
    temp = result[c_small_k]
    temp_xs = bx[c_small_k].astype(float)
    temp_ks = bk[c_small_k].astype(float)
    for i in range(len(temp)):
        p = <int>(1 + a1 + a2 * temp_ks[i] - a3 / (temp_ks[i] + a4))
        temp[i] = von_mises_cdf_series(temp_ks[i], temp_xs[i], p)
        temp[i] = 0 if temp[i] < 0 else 1 if temp[i] > 1 else temp[i]
    result[c_small_k] = temp

    # 对于大 k 的情况，使用正态分布近似计算 von Mises 分布的累积分布函数
    result[~c_small_k] = von_mises_cdf_normalapprox(bk[~c_small_k], bx[~c_small_k])

    # 返回结果，根据输入是否为零维进行调整
    if not zerodim:
        return result + ix
    else:
        return (result + ix)[0]


# 定义 _kendall_dis 函数，计算 Kendall 距离的辅助函数
@cython.wraparound(False)
@cython.boundscheck(False)
def _kendall_dis(intp_t[:] x, intp_t[:] y):
    # 初始化变量和数组
    cdef:
        intp_t sup = 1 + np.max(y)
        # 使用 `>> 14` 改进 Fenwick 树的缓存性能（参见 gh-10108）
        intp_t[::1] arr = np.zeros(sup + ((sup - 1) >> 14), dtype=np.intp)
        intp_t i = 0, k = 0, size = x.size, idx
        int64_t dis = 0
    # 使用 nogil 上下文，表示以下代码将在没有全局解释器锁 (GIL) 的情况下执行
    with nogil:
        # 当 i 小于 size 时执行循环
        while i < size:
            # 在 k 小于 size 且 x[i] 等于 x[k] 的情况下执行内层循环
            while k < size and x[i] == x[k]:
                # dis 增加 i 的值
                dis += i
                # 将 y[k] 的值赋给 idx
                idx = y[k]
                # 当 idx 不等于 0 时执行循环
                while idx != 0:
                    # dis 减去 arr[idx + (idx >> 14)] 的值
                    dis -= arr[idx + (idx >> 14)]
                    # idx 更新为 idx 与 (idx - 1) 的按位与结果
                    idx = idx & (idx - 1)
    
                # k 值增加 1
                k += 1
    
            # 当 i 小于 k 时执行循环
            while i < k:
                # 将 y[i] 的值赋给 idx
                idx = y[i]
                # 当 idx 小于 sup 时执行内层循环
                while idx < sup:
                    # arr[idx + (idx >> 14)] 的值增加 1
                    arr[idx + (idx >> 14)] += 1
                    # idx 更新为 idx 加上 idx 的负数位运算结果
                    idx += idx & -idx
                # i 值增加 1
                i += 1
    
    # 返回 dis 的值作为函数的结果
    return dis
# 定义一个融合类型，包括 np.int32_t、np.int64_t、np.float32_t、np.float64_t，用于计算加权 tau
# 其他类型的数组将通过 _toint64() 转换为排名数组。
ctypedef fused ordered:
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t


# 原地反转排列的函数，根据 Boonstra 在 1965 年的文章实现
@cython.wraparound(False)
@cython.boundscheck(False)
cdef _invert_in_place(intp_t[:] perm):
    cdef intp_t n, i, j, k
    # 从后向前遍历排列
    for n in range(len(perm)-1, -1, -1):
        i = perm[n]
        if i < 0:
            perm[n] = -i - 1
        else:
            if i != n:
                k = n
                # 循环直到找到一个闭环
                while True:
                    j = perm[i]
                    perm[i] = -k - 1
                    if j == n:
                        perm[n] = i
                        break
                    k = i
                    i = j


@cython.wraparound(False)
@cython.boundscheck(False)
def _toint64(x):
    cdef intp_t i = 0, j = 0, l = len(x)
    # 使用快速排序对 x 进行排序，返回排列数组 perm
    cdef intp_t[::1] perm = np.argsort(x, kind='quicksort')
    # 创建一个 int64 类型的数组，用于存储排序后的排名结果
    cdef int64_t[::1] result = np.ndarray(l, dtype=np.int64)

    # 查找并处理 NaN 值，将其赋值为最小值
    for i in range(l - 1, -1, -1):
        if not np.isnan(x[perm[i]]):
            break
        result[perm[i]] = 0

    if i < l - 1:
        j = 1
        l = i + 1

    # 计算排名
    for i in range(l - 1):
        result[perm[i]] = j
        if x[perm[i]] != x[perm[i + 1]]:
            j += 1

    result[perm[l - 1]] = j
    return np.array(result, dtype=np.int64)


@cython.wraparound(False)
@cython.boundscheck(False)
def _weightedrankedtau(const ordered[:] x, const ordered[:] y, intp_t[:] rank, weigher, bool additive):
    # y_local 和 rank_local 是对 Cython bug 的解决方法，参见 gh-16718
    # 在 Cython 3.0 可用后，可以移除 y_local 和 rank_local，直接使用 y 和 rank
    cdef const ordered[:] y_local = y
    cdef intp_t i, first
    cdef float64_t t, u, v, w, s, sq
    cdef int64_t n = np.int64(len(x))
    cdef float64_t[::1] exchanges_weight = np.zeros(1, dtype=np.float64)
    # 初始按照 x 的值排序，并在值相同时按照 y 的值排序
    cdef intp_t[::1] perm = np.lexsort((y, x))
    cdef intp_t[::1] temp = np.empty(n, dtype=np.intp) # 支持结构

    if weigher is None:
        # 如果 weigher 为 None，则默认使用 lambda 函数计算权重
        weigher = lambda x: 1./(1 + x)

    if rank is None:
        # 如果排名数组为 None，则需要生成排名数组
        # 首先反转排列（以获取较高的排名在前），然后对其进行反转
        rank = np.empty(n, dtype=np.intp)
        rank[...] = perm[::-1]
        _invert_in_place(rank)

    cdef intp_t[:] rank_local = rank

    # 加权处理并计算 tau
    first = 0
    t = 0
    w = weigher(rank[perm[first]])
    s = w
    sq = w * w
    # 对排列 perm 中的元素进行循环处理，从第一个元素到第 n-1 个元素
    for i in range(1, n):
        # 检查当前处理的 perm[first] 和 perm[i] 在 x 和 y 上的值是否相同
        if x[perm[first]] != x[perm[i]] or y[perm[first]] != y[perm[i]]:
            # 根据参数 additive 决定更新 t 的方式
            t += s * (i - first - 1) if additive else (s * s - sq) / 2
            # 更新 first 的位置为当前处理的 i
            first = i
            # 重置 s 和 sq 为 0
            s = sq = 0
        
        # 计算权重 w，根据排名 rank[perm[i]] 使用 weigher 函数
        w = weigher(rank[perm[i]])
        # 更新 s 和 sq
        s += w
        sq += w * w

    # 处理最后一段相同 x 或 y 的数据段
    t += s * (n - first - 1) if additive else (s * s - sq) / 2

    # weigh ties in x
    # 初始化 first 和 u
    first = 0
    u = 0
    # 计算第一个元素的权重 w，并初始化 s 和 sq
    w = weigher(rank[perm[first]])
    s = w
    sq = w * w

    # 对排列 perm 中的元素进行循环处理，从第一个元素到第 n-1 个元素
    for i in range(1, n):
        # 检查当前处理的 perm[first] 和 perm[i] 在 x 上的值是否相同
        if x[perm[first]] != x[perm[i]]:
            # 根据参数 additive 决定更新 u 的方式
            u += s * (i - first - 1) if additive else (s * s - sq) / 2
            # 更新 first 的位置为当前处理的 i
            first = i
            # 重置 s 和 sq 为 0
        
        # 计算权重 w，根据排名 rank[perm[i]] 使用 weigher 函数
        w = weigher(rank[perm[i]])
        # 更新 s 和 sq
        s += w
        sq += w * w

    # 处理最后一段相同 x 的数据段
    u += s * (n - first - 1) if additive else (s * s - sq) / 2
    # 如果 first 仍为 0，说明所有元素在 x 上都相同，返回 np.nan
    if first == 0: # x is constant (all ties)
        return np.nan

    # this closure recursively sorts sections of perm[] by comparing
    # elements of y[perm[]] using temp[] as support

    # 定义内部函数 weigh，用于根据 y 的值对 perm 数组的部分进行递归排序
    def weigh(intp_t offset, intp_t length):
        cdef intp_t length0, length1, middle, i, j, k
        cdef float64_t weight, residual

        # 如果 length 为 1，直接返回 rank_local[perm[offset]] 的权重
        if length == 1:
            return weigher(rank_local[perm[offset]])
        
        # 分割数组为两部分
        length0 = length // 2
        length1 = length - length0
        middle = offset + length0
        # 递归调用 weigh 函数，处理左右两部分并计算权重
        residual = weigh(offset, length0)
        weight = weigh(middle, length1) + residual
        # 如果 perm[middle-1] 和 perm[middle] 对应的 y 值满足条件，直接返回权重
        if y_local[perm[middle - 1]] < y_local[perm[middle]]:
            return weight

        # 合并排序的过程
        i = j = k = 0
        while j < length0 and k < length1:
            if y_local[perm[offset + j]] <= y_local[perm[middle + k]]:
                temp[i] = perm[offset + j]
                residual -= weigher(rank_local[temp[i]])
                j += 1
            else:
                temp[i] = perm[middle + k]
                exchanges_weight[0] += weigher(rank_local[temp[i]]) * (
                    length0 - j) + residual if additive else weigher(
                    rank_local[temp[i]]) * residual
                k += 1
            i += 1
        
        perm[offset+i:offset+i+length0-j] = perm[offset+j:offset+length0]
        perm[offset:offset+i] = temp[0:i]
        return weight

    # weigh discordances
    # 调用 weigh 函数，处理 perm 数组中的 discordances
    weigh(0, n)

    # weigh ties in y
    # 初始化 first 和 v
    first = 0
    v = 0
    # 计算第一个元素的权重 w，并初始化 s 和 sq
    w = weigher(rank[perm[first]])
    s = w
    sq = w * w

    # 对排列 perm 中的元素进行循环处理，从第一个元素到第 n-1 个元素
    for i in range(1, n):
        # 检查当前处理的 perm[first] 和 perm[i] 在 y 上的值是否相同
        if y[perm[first]] != y[perm[i]]:
            # 根据参数 additive 决定更新 v 的方式
            v += s * (i - first - 1) if additive else (s * s - sq) / 2
            # 更新 first 的位置为当前处理的 i
            first = i
            # 重置 s 和 sq 为 0
        
        # 计算权重 w，根据排名 rank[perm[i]] 使用 weigher 函数
        w = weigher(rank[perm[i]])
        # 更新 s 和 sq
        s += w
        sq += w * w

    # 处理最后一段相同 y 的数据段
    v += s * (n - first - 1) if additive else (s * s - sq) / 2
    # 如果 first 仍为 0，说明所有元素在 y 上都相同，返回 np.nan
    if first == 0: # y is constant (all ties)
        return np.nan

    # weigh all pairs
    # 计算所有配对的权重
    s = sq = 0
    for i in range(n):
        w = weigher(rank[perm[i]])
        s += w
        sq += w * w

    # 计算总的权重 tot，根据参数 additive 决定方式
    tot = s * (n - 1) if additive else (s * s - sq) / 2
    # 计算 Kendall's tau 相关系数
    tau = ((tot - (v + u - t)) - 2. * exchanges_weight[0]
           ) / np.sqrt(tot - u) / np.sqrt(tot - v)
    # 将 tau 值限制在 [-1, 1] 的范围内
    return min(1., max(-1., tau))
# FROM MGCPY: https://github.com/neurodata/mgcpy
# Distance transforms used for MGC and Dcorr

# Columnwise ranking of data
@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个 Cython 函数来对二维数组 x 进行列方向的排名处理
cdef _dense_rank_data(ndarray x):
    # 使用 np.unique 函数获取 x 中唯一值并返回其逆序索引，表示排名
    _, v = np.unique(x, return_inverse=True)
    return v + 1


@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个 Cython 函数来对距离矩阵进行排名处理
def _rank_distance_matrix(distx):
    # 使用列表推导式在列方向上对 distx 中的每列应用 _dense_rank_data 函数
    # 并将结果水平堆叠成一个二维数组
    return np.hstack([_dense_rank_data(distx[:, i]).reshape(-1, 1) for i in range(distx.shape[1])])


@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个 Cython 函数来对距离矩阵进行居中处理
def _center_distance_matrix(distx, global_corr='mgc', is_ranked=True):
    cdef int n = distx.shape[0]  # 获取 distx 的行数
    cdef int m = distx.shape[1]  # 获取 distx 的列数
    cdef ndarray rank_distx = np.zeros(n * m)  # 创建一个大小为 n*m 的零数组

    if is_ranked:
        rank_distx = _rank_distance_matrix(distx)  # 如果 is_ranked 为 True，则进行排名处理

    if global_corr == "rank":
        distx = rank_distx.astype(np.float64, copy=False)  # 如果 global_corr 为 "rank"，将 distx 转换为 float64 类型

    # 计算 'mgc' 情况下的期望距离矩阵，即每列的均值乘以 n/(n-1)，然后在行方向上复制扩展
    cdef ndarray exp_distx = np.repeat(((distx.mean(axis=0) * n) / (n-1)), n).reshape(-1, n).T

    # 居中距离矩阵
    cdef ndarray cent_distx = distx - exp_distx

    if global_corr != "mantel" and global_corr != "biased":
        np.fill_diagonal(cent_distx, 0)  # 如果 global_corr 不是 "mantel" 或 "biased"，则将中心距离矩阵的对角线填充为 0

    return cent_distx, rank_distx  # 返回中心化后的距离矩阵和排名矩阵



# Centers each distance matrix and rank matrix
@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个 Cython 函数来对每个距离矩阵和排名矩阵进行转换处理
def _transform_distance_matrix(distx, disty, global_corr='mgc', is_ranked=True):
    if global_corr == "rank":
        is_ranked = True  # 如果 global_corr 为 "rank"，则设置 is_ranked 为 True

    cent_distx, rank_distx = _center_distance_matrix(distx, global_corr, is_ranked)  # 对 distx 进行中心化处理
    cent_disty, rank_disty = _center_distance_matrix(disty, global_corr, is_ranked)  # 对 disty 进行中心化处理

    transform_dist = {"cent_distx": cent_distx, "cent_disty": cent_disty,
                      "rank_distx": rank_distx, "rank_disty": rank_disty}

    return transform_dist  # 返回转换后的距离矩阵和排名矩阵的字典


# MGC specific functions
@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个 Cython 函数来计算期望协方差
cdef _expected_covar(const float64_t[:, :] distx, const float64_t[:, :] disty,
                     const int64_t[:, :] rank_distx, const int64_t[:, :] rank_disty,
                     float64_t[:, :] cov_xy, float64_t[:] expectx,
                     float64_t[:] expecty):
    # 对距离矩阵 distx 和 disty 中的每对元素进行遍历
    # 根据它们的排名，计算协方差矩阵和期望值
    cdef intp_t n = distx.shape[0]
    cdef float64_t a, b
    cdef intp_t i, j, k, l
    for i in range(n):
        for j in range(n):
            a = distx[i, j]
            b = disty[i, j]
            k = rank_distx[i, j]
            l = rank_disty[i, j]

            cov_xy[k, l] += a * b  # 更新协方差矩阵的对应位置
            expectx[k] += a  # 更新期望值 expectx
            expecty[l] += b  # 更新期望值 expecty

    return np.asarray(expectx), np.asarray(expecty)  # 返回更新后的 expectx 和 expecty


@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个 Cython 函数来生成协方差矩阵
cdef _covar_map(float64_t[:, :] cov_xy, intp_t nx, intp_t ny):
    # 返回给定尺寸的协方差矩阵 cov_xy
    cdef intp_t k, l
    # 遍历二维数组 cov_xy 中的元素，计算累加和
    for k in range(nx - 1):
        for l in range(ny - 1):
            # 对于每个元素 (k+1, l+1)，计算其累加和，根据左、上和左上的元素值
            cov_xy[k+1, l+1] += (cov_xy[k+1, l] + cov_xy[k, l+1] - cov_xy[k, l])
    
    # 将二维数组 cov_xy 转换为 NumPy 数组并返回
    return np.asarray(cov_xy)
# 设置 Cython 的优化指令，关闭数组边界检查和负索引处理，以提高性能
@cython.wraparound(False)
@cython.boundscheck(False)
def _local_covariance(distx, disty, rank_distx, rank_disty):
    # 将 float32 类型的 numpy 数组转换为 int64 类型，因为它将作为数组索引使用
    # 数组索引从 0 到 n-1
    rank_distx = np.asarray(rank_distx, np.int64) - 1
    rank_disty = np.asarray(rank_disty, np.int64) - 1

    # 获取数组的长度
    cdef intp_t n = distx.shape[0]
    # 计算排名的最大值并加一，作为数组的长度
    cdef intp_t nx = np.max(rank_distx) + 1
    cdef intp_t ny = np.max(rank_disty) + 1
    # 创建用于存储局部协方差的数组
    cdef ndarray cov_xy = np.zeros((nx, ny))
    # 创建用于存储数据 A 的期望值的数组
    cdef ndarray expectx = np.zeros(nx)
    # 创建用于存储数据 B 的期望值的数组
    cdef ndarray expecty = np.zeros(ny)

    # 计算期望协方差
    expectx, expecty = _expected_covar(distx, disty, rank_distx, rank_disty,
                                       cov_xy, expectx, expecty)

    # 对第一列进行累积和操作
    cov_xy[:, 0] = np.cumsum(cov_xy[:, 0])
    # 对 expectx 进行累积和操作
    expectx = np.cumsum(expectx)

    # 对第一行进行累积和操作
    cov_xy[0, :] = np.cumsum(cov_xy[0, :])
    # 对 expecty 进行累积和操作
    expecty = np.cumsum(expecty)

    # 将协方差数组映射为局部协方差
    cov_xy = _covar_map(cov_xy, nx, ny)
    # 对协方差进行居中处理
    cov_xy = cov_xy - ((expectx.reshape(-1, 1) @ expecty.reshape(-1, 1).T) / n**2)

    # 返回局部协方差数组
    return cov_xy


# 设置 Cython 的优化指令，关闭数组边界检查和负索引处理，以提高性能
@cython.wraparound(False)
@cython.boundscheck(False)
def _local_correlations(distx, disty, global_corr='mgc'):
    # 转换距离矩阵，根据全局相关性类型进行变换
    transformed = _transform_distance_matrix(distx, disty, global_corr)

    # 计算所有局部协方差
    cdef ndarray cov_mat = _local_covariance(
        transformed["cent_distx"],
        transformed["cent_disty"].T,
        transformed["rank_distx"],
        transformed["rank_disty"].T)

    # 计算数据 A 的局部方差
    cdef ndarray local_varx = _local_covariance(
        transformed["cent_distx"],
        transformed["cent_distx"].T,
        transformed["rank_distx"],
        transformed["rank_distx"].T)
    local_varx = local_varx.diagonal()

    # 计算数据 B 的局部方差
    cdef ndarray local_vary = _local_covariance(
        transformed["cent_disty"],
        transformed["cent_disty"].T,
        transformed["rank_disty"],
        transformed["rank_disty"].T)
    local_vary = local_vary.diagonal()

    # 对协方差进行归一化处理，得到局部相关性数组
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_mat = cov_mat / np.sqrt(local_varx.reshape(-1, 1) @ local_vary.reshape(-1, 1).T).real
        # 避免计算问题可能导致部分局部相关性略大于 1
        corr_mat[corr_mat > 1] = 1

    # 如果任何局部方差小于等于 0，则将相应的局部相关性设为 0
    corr_mat[local_varx <= 0, :] = 0
    corr_mat[:, local_vary <= 0] = 0

    # 返回局部相关性数组
    return corr_mat


# 使用 Cython 语法定义一个 C 函数，生成逆高斯分布的对数概率密度函数
cpdef double geninvgauss_logpdf(double x, double p, double b) noexcept nogil:
    return _geninvgauss_logpdf_kernel(x, p, b)


# 使用 Cython 语法定义一个 C 函数，实现逆高斯分布的对数概率密度函数的核心计算部分
cdef double _geninvgauss_logpdf_kernel(double x, double p, double b) noexcept nogil:
    cdef double z, c

    # 如果 x 小于等于 0，则返回负无穷大
    if x <= 0:
        return -INFINITY
    # 使用 cs 对象的 kve 方法计算给定参数 p 和 b 的值并赋给变量 z
    z = cs.kve(p, b)
    
    # 如果 z 的值是无穷大，则返回 NAN
    if math.isinf(z):
        return NAN
    
    # 计算变量 c 的值，其中包括数学常数 -log(2)、-log(z) 和 b 的负数
    c = -math.log(2) - math.log(z) + b
    
    # 返回根据给定公式计算得出的值，包括 c 的值、(p - 1) 乘以 x 的自然对数，以及 b 乘以 (x + 1/x) 的一半
    return c + (p - 1) * math.log(x) - b * (x + 1/x) / 2
cdef double _geninvgauss_pdf(double x, void *user_data) noexcept nogil:
    # 在 LowLevelCallable 中使用的函数，计算广义逆高斯分布的概率密度函数
    cdef double p, b

    if x <= 0:
        return 0.

    # 从用户数据中提取参数 p 和 b
    p = (<double *>user_data)[0]
    b = (<double *>user_data)[1]

    # 返回 x 的广义逆高斯分布的概率密度函数值
    return math.exp(_geninvgauss_logpdf_kernel(x, p, b))


cdef double _phi(double z) noexcept nogil:
    """计算正态分布的概率密度函数值。在 `studentized_range` 中使用"""
    cdef double inv_sqrt_2pi = 0.3989422804014327
    return inv_sqrt_2pi * math.exp(-0.5 * z * z)


cdef double _logphi(double z) noexcept nogil:
    """计算正态分布概率密度函数的对数值。在 `studentized_range` 中使用"""
    cdef double log_inv_sqrt_2pi = -0.9189385332046727
    return log_inv_sqrt_2pi - 0.5 * z * z


cdef double _Phi(double z) noexcept nogil:
    """计算正态分布的累积分布函数值。在 `studentized_range` 中使用"""
    # 使用自定义函数，因为在 32 位系统上使用 cs.ndtr 可能导致 q=0 时 PDF 错误
    # 使用硬编码的 1/sqrt(2) 常数，而不是 math 模块的常数，因为后者不是所有系统都可用
    cdef double inv_sqrt_2 = 0.7071067811865475
    return 0.5 * math.erfc(-z * inv_sqrt_2)


cpdef double _studentized_range_cdf_logconst(double k, double df) noexcept:
    """计算学生化范围分布累积分布函数中的对数常数项"""
    cdef double log_2 = 0.6931471805599453
    return (math.log(k) + (df / 2) * math.log(df)
            - (math.lgamma(df / 2) + (df / 2 - 1) * log_2))


cpdef double _studentized_range_pdf_logconst(double k, double df) noexcept:
    """计算学生化范围分布概率密度函数中的对数常数项"""
    cdef double log_2 = 0.6931471805599453
    return (math.log(k) + math.log(k - 1) + (df / 2) * math.log(df)
            - (math.lgamma(df / 2) + (df / 2 - 1) * log_2))


cdef double _studentized_range_cdf(int n, double[2] integration_var,
                                   void *user_data) noexcept nogil:
    # 计算 Batista 等人提出的方程 (3) 的积分项
    # 用于 LowLevelCallable
    q = (<double *> user_data)[0]
    k = (<double *> user_data)[1]
    df = (<double *> user_data)[2]
    log_cdf_const = (<double *> user_data)[3]

    s = integration_var[1]
    z = integration_var[0]

    # 适当的项在对数中计算，以避免溢出/下溢
    log_terms = (log_cdf_const
                 + (df - 1) * math.log(s)
                 - (df * s * s / 2)
                 + _logphi(z))

    # 将剩余的项在对数外乘以，因为可能为 0
    return math.exp(log_terms) * math.pow(_Phi(z + q * s) - _Phi(z), k - 1)


cdef double _studentized_range_cdf_asymptotic(double z, void *user_data) noexcept nogil:
    # 计算 Lund 等人在第 205 页提出的方程 (2) 的积分项
    # 用于 LowLevelCallable
    q = (<double *> user_data)[0]
    k = (<double *> user_data)[1]

    return k * _phi(z) * math.pow(_Phi(z + q) - _Phi(z), k - 1)
cdef double _studentized_range_pdf(int n, double[2] integration_var,
                                   void *user_data) noexcept nogil:
    # 计算 Batista 等人 [2] 中方程（4）的被积函数
    # 用于 LowLevelCallable
    q = (<double *> user_data)[0]  # 获取用户数据中的 q
    k = (<double *> user_data)[1]  # 获取用户数据中的 k
    df = (<double *> user_data)[2]  # 获取用户数据中的自由度 df
    log_pdf_const = (<double *> user_data)[3]  # 获取用户数据中的对数常数

    z = integration_var[0]  # 设置积分变量 z
    s = integration_var[1]  # 设置积分变量 s

    # 在对数中适当地计算术语，以避免下溢或上溢
    log_terms = (log_pdf_const
                 + df * math.log(s)
                 - df * s * s / 2
                 + _logphi(z)
                 + _logphi(s * q + z))

    # 将剩余的术语在对数之外相乘，因为它可能为 0
    return math.exp(log_terms) * math.pow(_Phi(s * q + z) - _Phi(z), k - 2)


cdef double _studentized_range_pdf_asymptotic(double z, void *user_data) noexcept nogil:
    # 计算 Lund 等人 [4] 中方程（2）的被积函数
    # 用于 LowLevelCallable
    q = (<double *> user_data)[0]  # 获取用户数据中的 q
    k = (<double *> user_data)[1]  # 获取用户数据中的 k

    return k * (k - 1) * _phi(z) * _phi(z + q) * math.pow(_Phi(z + q) - _Phi(z), k - 2)


cdef double _studentized_range_moment(int n, double[3] integration_var,
                                      void *user_data) noexcept nogil:
    # 用于 LowLevelCallable
    K = (<double *> user_data)[0]  # 要计算的第 K 阶矩
    k = (<double *> user_data)[1]  # 获取用户数据中的 k
    df = (<double *> user_data)[2]  # 获取用户数据中的自由度 df
    log_pdf_const = (<double *> user_data)[3]  # 获取用户数据中的对数常数

    # 将最外层积分变量提取出来作为传递给 PDF 的 q
    q = integration_var[2]

    cdef double pdf_data[4]  # 创建一个包含用户数据的数组
    pdf_data[0] = q
    pdf_data[1] = k
    pdf_data[2] = df
    pdf_data[3] = log_pdf_const

    return (math.pow(q, K) *
            _studentized_range_pdf(4, integration_var, pdf_data))


cpdef double genhyperbolic_pdf(double x, double p, double a, double b) noexcept nogil:
    # 计算广义双曲线分布的概率密度函数的值
    return math.exp(_genhyperbolic_logpdf_kernel(x, p, a, b))


cdef double _genhyperbolic_pdf(double x, void *user_data) noexcept nogil:
    # 计算广义双曲线分布的对数概率密度函数的被积函数
    # 用于 LowLevelCallable
    cdef double p, a, b

    p = (<double *>user_data)[0]  # 获取用户数据中的 p
    a = (<double *>user_data)[1]  # 获取用户数据中的 a
    b = (<double *>user_data)[2]  # 获取用户数据中的 b

    return math.exp(_genhyperbolic_logpdf_kernel(x, p, a, b))


cpdef double genhyperbolic_logpdf(
        double x, double p, double a, double b
        ) noexcept nogil:
    # 计算广义双曲线分布的对数概率密度函数的值
    return _genhyperbolic_logpdf_kernel(x, p, a, b)


# 对数概率密度函数总是负数，因此使用正值的异常值
cdef double _genhyperbolic_logpdf(double x, void *user_data) noexcept nogil:
    # 计算广义双曲线分布的对数概率密度函数的被积函数
    # 用于 LowLevelCallable
    cdef double p, a, b

    p = (<double *>user_data)[0]  # 获取用户数据中的 p
    a = (<double *>user_data)[1]  # 获取用户数据中的 a
    b = (<double *>user_data)[2]  # 获取用户数据中的 b

    return _genhyperbolic_logpdf_kernel(x, p, a, b)
cdef double _genhyperbolic_logpdf_kernel(
        double x, double p, double a, double b
        ) noexcept nogil:
    cdef double t1, t2, t3, t4, t5

    t1 = _log_norming_constant(p, a, b)  # 计算对数归一化常数
    t2 = math.sqrt(1.0 + x*x)  # 计算平方根
    t3 = (p - 0.5) * math.log(t2)  # 计算对数部分的指数函数
    t4 = math.log(cs.kve(p - 0.5, a * t2)) - a * t2  # 计算修改的贝塞尔函数的对数
    t5 = b * x  # 计算乘积

    return t1 + t3 + t4 + t5  # 返回结果


cdef double _log_norming_constant(double p, double a, double b) noexcept nogil:
    cdef double t1, t2, t3, t4, t5, t6

    t1 = (a + b)*(a - b)  # 计算差的平方
    t2 = p * 0.5 * math.log(t1)  # 计算对数部分的指数函数
    t3 = 0.5 * math.log(2 * PI)  # 计算对数部分的指数函数
    t4 = (p - 0.5) * math.log(a)  # 计算对数部分的指数函数
    t5 = math.sqrt(t1)  # 计算平方根
    t6 = math.log(cs.kve(p, t5)) - t5  # 计算修改的贝塞尔函数的对数

    return t2 - t3 - t4 - t6  # 返回结果


ctypedef fused real:
    float
    double
    long double


@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline int gaussian_kernel_estimate_inner(
    const real[:, :] points_,  const real[:, :] values_, const real[:, :] xi_,
    real[:, :] estimate, const real[:, :] cho_cov,
    int n, int m, int d, int p,
) noexcept nogil:
    cdef:
        int i, j, k
        real residual, arg, norm

    # Evaluate the normalisation
    norm = math.pow((2 * PI), (- d / 2.))  # 计算正则化常数
    for i in range(d):
        norm /= cho_cov[i, i]  # 根据 Cholesky 分解计算正则化常数的更新

    for i in range(n):
        for j in range(m):
            arg = 0
            for k in range(d):
                residual = (points_[i, k] - xi_[j, k])  # 计算残差
                arg += residual * residual  # 计算残差的平方和

            arg = math.exp(-arg / 2.) * norm  # 计算指数部分的指数函数
            for k in range(p):
                estimate[j, k] += values_[i, k] * arg  # 计算加权估计

    return 0


@cython.wraparound(False)
@cython.boundscheck(False)
def gaussian_kernel_estimate(points, values, xi, cho_cov, dtype,
                             real _=0):
    """
    Evaluate a multivariate Gaussian kernel estimate.

    Parameters
    ----------
    points : array_like with shape (n, d)
        Data points to estimate from in d dimensions.
    values : real[:, :] with shape (n, p)
        Multivariate values associated with the data points.
    xi : array_like with shape (m, d)
        Coordinates to evaluate the estimate at in d dimensions.
    cho_cov : array_like with shape (d, d)
        (Lower) Cholesky factor of the covariance.

    Returns
    -------
    estimate : double[:, :] with shape (m, p)
        Multivariate Gaussian kernel estimate evaluated at the input coordinates.
    """
    cdef:
        real[:, :] points_, xi_, values_, estimate, cho_cov_
        int n, d, m, p

    n = points.shape[0]  # 获取数据点的数量
    d = points.shape[1]  # 获取数据点的维度
    m = xi.shape[0]  # 获取估计点的数量
    p = values.shape[1]  # 获取值的维度

    if xi.shape[1] != d:
        raise ValueError("points and xi must have same trailing dim")  # 如果维度不匹配则抛出值错误异常
    if cho_cov.shape[0] != d or cho_cov.shape[1] != d:
        raise ValueError("Covariance matrix must match data dims")  # 如果协方差矩阵维度不匹配则抛出值错误异常

    # Rescale the data
    cho_cov_ = cho_cov.astype(dtype, copy=False)  # 将协方差矩阵转换为指定的数据类型
    # 使用 cho_cov 对 points.T 进行三角解法求解，并将结果转换为指定数据类型
    points_ = np.asarray(solve_triangular(cho_cov, points.T, lower=True).T,
                         dtype=dtype)
    # 使用 cho_cov 对 xi.T 进行三角解法求解，并将结果转换为指定数据类型
    xi_ = np.asarray(solve_triangular(cho_cov, xi.T, lower=True).T,
                     dtype=dtype)
    # 将 values 转换为指定数据类型，并确保原地修改
    values_ = values.astype(dtype, copy=False)

    # 创建结果数组并计算加权和
    estimate = np.zeros((m, p), dtype)

    # 使用 nogil 上下文调用 C 函数 gaussian_kernel_estimate_inner 进行计算
    with nogil:
        gaussian_kernel_estimate_inner(points_, values_, xi_,
                                       estimate, cho_cov_, n, m, d, p)

    # 返回结果数组的副本，确保返回的是指定数据类型的数组
    return np.asarray(estimate)
# 禁用 Cython 的边界检查和包装检查优化
@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个 C 语言级别的函数 logsumexp，计算两个实数的对数和指数运算的稳定版本
cdef real logsumexp(real a, real b):
    cdef:
        real c  # 本地实数变量 c
    # 计算 a 和 b 中较大的值
    c = max(a, b)
    # 返回稳定的对数求和结果
    return c + math.log(math.exp(a-c) + math.exp(b-c))


# 禁用 Cython 的边界检查和包装检查优化
@cython.wraparound(False)
@cython.boundscheck(False)
# 定义高斯核密度估计的对数版本函数
def gaussian_kernel_estimate_log(points, values, xi, cho_cov, dtype, real _=0):
    """
    def gaussian_kernel_estimate_log(points, real[:, :] values, xi, cho_cov)

    在提供的点集上评估估计的概率密度函数（pdf）的对数版本。

    Parameters
    ----------
    points : array_like with shape (n, d)
        要从中估计的数据点，具有 ``d`` 维度。
    values : real[:, :] with shape (n, p)
        与数据点相关联的多变量值。
    xi : array_like with shape (m, d)
        要在 ``d`` 维度上评估估计值的坐标。
    cho_cov : array_like with shape (d, d)
        协方差的（下三角）Cholesky 分解因子。

    Returns
    -------
    estimate : double[:, :] with shape (m, p)
        在输入坐标处评估的多变量高斯核密度估计的对数版本。
    """
    cdef:
        real[:, :] points_, xi_, values_, log_values_, estimate  # C 语言级别的局部变量声明
        int i, j, k  # 整数变量声明
        int n, d, m, p  # 整数变量声明
        real arg, residual, log_norm  # 实数变量声明

    n = points.shape[0]  # 数据点的数量
    d = points.shape[1]  # 数据点的维度
    m = xi.shape[0]  # 要评估的坐标的数量
    p = values.shape[1]  # 值的数量（多变量值的维度）

    if xi.shape[1] != d:  # 如果点集和要评估的坐标不匹配维度，则引发 ValueError
        raise ValueError("points and xi must have same trailing dim")
    if cho_cov.shape[0] != d or cho_cov.shape[1] != d:  # 如果协方差矩阵不匹配数据维度，则引发 ValueError
        raise ValueError("Covariance matrix must match data dims")

    # 重新缩放数据
    points_ = np.asarray(solve_triangular(cho_cov, points.T, lower=True).T,
                         dtype=dtype)
    xi_ = np.asarray(solve_triangular(cho_cov, xi.T, lower=True).T,
                     dtype=dtype)
    values_ = values.astype(dtype, copy=False)  # 将值转换为指定的数据类型

    log_values_ = np.empty((n, p), dtype)  # 创建一个空的数组用于存储对数值
    for i in range(n):
        for k in range(p):
            log_values_[i, k] = math.log(values_[i, k])  # 计算每个值的自然对数

    # 评估归一化常数
    log_norm = (- d / 2) * math.log(2 * PI)
    for i in range(d):
        log_norm -= math.log(cho_cov[i, i])

    # 创建结果数组并评估加权和
    estimate = np.full((m, p), fill_value=-np.inf, dtype=dtype)
    for i in range(n):
        for j in range(m):
            arg = 0
            for k in range(d):
                residual = (points_[i, k] - xi_[j, k])
                arg += residual * residual

            arg = -arg / 2 + log_norm
            for k in range(p):
                estimate[j, k] = logsumexp(estimate[j, k],
                                           arg + log_values_[i, k])  # 使用稳定的对数求和函数计算结果

    return np.asarray(estimate)  # 返回估计结果的数组表示
```