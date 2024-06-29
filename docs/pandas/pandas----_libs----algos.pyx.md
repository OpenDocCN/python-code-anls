# `D:\src\scipysrc\pandas\pandas\_libs\algos.pyx`

```
# 导入 Cython 模块声明
cimport cython
# 导入 Cython 中的 Py_ssize_t 类型声明
from cython cimport Py_ssize_t
# 导入 libc.math 中的数学函数声明
from libc.math cimport (
    fabs,    # 绝对值函数
    sqrt,    # 平方根函数
)
# 导入 libc.stdlib 中的内存分配函数声明
from libc.stdlib cimport (
    free,    # 释放内存函数
    malloc,  # 分配内存函数
)
# 导入 libc.string 中的内存操作函数声明
from libc.string cimport memmove  # 内存块移动函数

# 导入 NumPy 库
import numpy as np

# 使用 Cython 导入 NumPy C API
cimport numpy as cnp
# 从 NumPy C API 中导入数据类型和常量
from numpy cimport (
    NPY_FLOAT64,    # 64位浮点数类型常量
    NPY_INT8,       # 8位整数类型常量
    NPY_INT16,      # 16位整数类型常量
    NPY_INT32,      # 32位整数类型常量
    NPY_INT64,      # 64位整数类型常量
    NPY_OBJECT,     # 对象类型常量
    NPY_UINT64,     # 64位无符号整数类型常量
    float32_t,      # 32位浮点数类型
    float64_t,      # 64位浮点数类型
    int8_t,         # 8位整数类型
    int16_t,        # 16位整数类型
    int32_t,        # 32位整数类型
    int64_t,        # 64位整数类型
    intp_t,         # 平台相关的整数类型
    ndarray,        # NumPy 数组类型
    uint8_t,        # 8位无符号整数类型
    uint16_t,       # 16位无符号整数类型
    uint32_t,       # 32位无符号整数类型
    uint64_t,       # 64位无符号整数类型
)

# 导入 NumPy C API 的数组初始化函数
cnp.import_array()

# 导入 pandas 库中的实用工具函数
cimport pandas._libs.util as util
# 从 pandas 库的 dtypes 模块导入数据类型声明
from pandas._libs.dtypes cimport (
    numeric_object_t,   # 数值对象类型
    numeric_t,          # 数值类型
)
# 从 pandas 库的 khash 模块导入哈希表操作函数声明
from pandas._libs.khash cimport (
    kh_destroy_int64,   # 销毁 int64 哈希表
    kh_get_int64,       # 获取 int64 键的哈希表项
    kh_init_int64,      # 初始化 int64 哈希表
    kh_int64_t,         # int64 哈希表类型
    kh_put_int64,       # 插入 int64 键到哈希表
    kh_resize_int64,    # 调整 int64 哈希表大小
    khiter_t,           # 哈希表迭代器类型
)
# 从 pandas 库的 missing 模块导入缺失数据处理函数声明
from pandas._libs.missing cimport (
    checknull,      # 检查是否为缺失数据
    isnaobj,        # 检查是否为缺失对象
)
# 从 pandas 库的 util 模块导入获取 NAT 值的函数声明
from pandas._libs.util cimport get_nat

# 定义全局常量
cdef:
    float64_t FP_ERR = 1e-13      # 浮点数比较误差阈值
    float64_t NaN = <float64_t>np.nan  # NaN 值，使用 NumPy 定义
    int64_t NPY_NAT = get_nat()   # 获取 NAT（Not A Time）值

# 定义排序时的优先级策略字典
tiebreakers = {
    "average": TIEBREAK_AVERAGE,    # 平均值策略
    "min": TIEBREAK_MIN,            # 最小值策略
    "max": TIEBREAK_MAX,            # 最大值策略
    "first": TIEBREAK_FIRST,        # 首个值策略
    "dense": TIEBREAK_DENSE,        # 稠密值策略
}

# 定义函数，比较两个对象是否不同
cdef bint are_diff(object left, object right):
    try:
        return fabs(left - right) > FP_ERR   # 比较绝对值是否大于误差阈值
    except TypeError:
        return left != right   # 如果类型不兼容，直接比较值是否不同

# 定义类 Infinity，提供正无穷比较方法用于排序
class Infinity:
    """
    Provide a positive Infinity comparison method for ranking.
    """
    def __lt__(self, other):
        return False    # 总是返回 False

    def __le__(self, other):
        return isinstance(other, Infinity)   # 检查是否为 Infinity 类型

    def __eq__(self, other):
        return isinstance(other, Infinity)   # 检查是否为 Infinity 类型

    def __ne__(self, other):
        return not isinstance(other, Infinity)  # 检查是否不为 Infinity 类型

    def __gt__(self, other):
        return (not isinstance(other, Infinity) and
                not checknull(other))    # 检查是否不为 Infinity 并且不是缺失值

    def __ge__(self, other):
        return not checknull(other)   # 检查是否不是缺失值

# 定义类 NegInfinity，提供负无穷比较方法用于排序
class NegInfinity:
    """
    Provide a negative Infinity comparison method for ranking.
    """
    def __lt__(self, other):
        return  (not isinstance(other, NegInfinity) and
                 not checknull(other))   # 检查是否不是 NegInfinity 并且不是缺失值

    def __le__(self, other):
        return not checknull(other)   # 检查是否不是缺失值

    def __eq__(self, other):
        return isinstance(other, NegInfinity)   # 检查是否为 NegInfinity 类型

    def __ne__(self, other):
        return not isinstance(other, NegInfinity)   # 检查是否不为 NegInfinity 类型

    def __gt__(self, other):
        return False    # 总是返回 False

    def __ge__(self, other):
        return isinstance(other, NegInfinity)   # 检查是否为 NegInfinity 类型

# 定义函数，使用 Cython 的 wraparound 和 boundscheck 优化数组操作
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef ndarray[int64_t, ndim=1] unique_deltas(const int64_t[:] arr):
    """
    Efficiently find the unique first-differences of the given array.

    Parameters
    ----------
    arr : ndarray[int64_t]
        Input array of 64-bit integers.

    Returns
    -------
    ndarray[int64_t]
        An ordered array of unique first-differences as 64-bit integers.
    """
    # 声明变量 i 和 n，分别表示循环计数和数组 arr 的长度
    cdef:
        Py_ssize_t i, n = len(arr)
        # 声明变量 val，用于存储数组元素差值
        int64_t val
        # 声明变量 k，用于存储哈希表中的键
        khiter_t k
        # 声明指向哈希表的指针 table
        kh_int64_t *table
        # 初始化返回值 ret 为 0
        int ret = 0
        # 声明空列表 uniques，用于存储唯一的差值
        list uniques = []
        # 声明结果数组 result，数据类型为 int64_t 的一维 ndarray
        ndarray[int64_t, ndim=1] result
    
    # 初始化 int64 类型的哈希表 table
    table = kh_init_int64()
    # 调整哈希表 table 的大小为 10
    kh_resize_int64(table, 10)
    
    # 遍历数组 arr 中除最后一个元素外的所有元素
    for i in range(n - 1):
        # 计算相邻数组元素的差值
        val = arr[i + 1] - arr[i]
        # 在哈希表 table 中查找差值 val 对应的键 k
        k = kh_get_int64(table, val)
        # 如果 k 等于哈希表的桶数，说明 val 不在哈希表中
        if k == table.n_buckets:
            # 将差值 val 放入哈希表 table 中
            kh_put_int64(table, val, &ret)
            # 将差值 val 添加到 uniques 列表中
            uniques.append(val)
    
    # 销毁哈希表 table
    kh_destroy_int64(table)
    
    # 将 uniques 列表转换为 numpy 的 int64_t 数组，并按升序排序
    result = np.array(uniques, dtype=np.int64)
    result.sort()
    
    # 返回排序后的结果数组 result
    return result
# 使用 Cython 提供的 wraparound(False) 装饰器禁用数组的负索引环绕访问
# 使用 Cython 提供的 boundscheck(False) 装饰器禁用数组边界检查
@cython.wraparound(False)
@cython.boundscheck(False)
def is_lexsorted(list_of_arrays: list) -> bool:
    # 声明 C 语言扩展变量
    cdef:
        Py_ssize_t i  # Python ssize_t 类型变量 i
        Py_ssize_t n, nlevels  # Python ssize_t 类型变量 n 和 nlevels
        int64_t k, cur, pre  # C long long int 类型变量 k, cur, pre
        ndarray arr  # NumPy 数组对象 arr
        bint result = True  # Cython 布尔类型变量 result，默认为 True

    nlevels = len(list_of_arrays)  # 计算 list_of_arrays 中数组的数量
    n = len(list_of_arrays[0])  # 计算 list_of_arrays 中第一个数组的长度

    # 分配内存以保存 nlevels 个 int64_t* 类型的指针数组
    cdef int64_t **vecs = <int64_t**>malloc(nlevels * sizeof(int64_t*))
    if vecs is NULL:
        raise MemoryError()  # 内存分配失败时抛出 MemoryError 异常
    for i in range(nlevels):
        arr = list_of_arrays[i]
        assert arr.dtype.name == "int64"  # 断言 arr 的数据类型为 int64
        vecs[i] = <int64_t*>cnp.PyArray_DATA(arr)  # 将 arr 的数据指针转换为 int64_t*

    # 假设数组已经按字典顺序排序
    with nogil:
        for i in range(1, n):
            for k in range(nlevels):
                cur = vecs[k][i]  # 当前数组 k 中的第 i 个元素
                pre = vecs[k][i - 1]  # 当前数组 k 中的第 i-1 个元素
                if cur == pre:
                    continue  # 如果当前元素与前一个元素相同，则继续下一轮循环
                elif cur > pre:
                    break  # 如果当前元素大于前一个元素，则退出当前循环
                else:
                    result = False  # 如果当前元素小于前一个元素，则设置 result 为 False
                    break  # 退出当前循环
            if not result:
                break  # 如果 result 为 False，则退出外层循环

    free(vecs)  # 释放 vecs 占用的内存
    return result  # 返回排序结果的布尔值


# 使用 Cython 提供的 boundscheck(False) 装饰器禁用数组边界检查
# 使用 Cython 提供的 wraparound(False) 装饰器禁用数组的负索引环绕访问
def groupsort_indexer(const intp_t[:] index, Py_ssize_t ngroups):
    """
    计算一维索引器。

    索引器是传入索引的排序顺序，
    按组排序。

    参数
    ----------
    index: np.ndarray[np.intp]
        组 -> 位置的映射。
    ngroups: int64
        组的数量。

    返回
    -------
    ndarray[intp_t, ndim=1]
        索引器
    ndarray[intp_t, ndim=1]
        组计数

    注
    -----
    这是标签因子化过程的反向过程。
    """
    # 声明 C 语言扩展变量
    cdef:
        Py_ssize_t i, label, n  # Python ssize_t 类型变量 i, label, n
        intp_t[::1] indexer, where, counts  # NumPy intp_t 类型数组 indexer, where, counts

    counts = np.zeros(ngroups + 1, dtype=np.intp)  # 创建长度为 ngroups+1 的零数组 counts
    n = len(index)  # 计算 index 数组的长度
    indexer = np.zeros(n, dtype=np.intp)  # 创建长度为 n 的零数组 indexer
    where = np.zeros(ngroups + 1, dtype=np.intp)  # 创建长度为 ngroups+1 的零数组 where

    with nogil:
        # 计算每个组的大小，位置 0 为 NA
        for i in range(n):
            counts[index[i] + 1] += 1

        # 标记每个连续相同索引数据组的开始位置
        for i in range(1, ngroups + 1):
            where[i] = where[i - 1] + counts[i - 1]

        # 这是我们的索引器
        for i in range(n):
            label = index[i] + 1
            indexer[where[label]] = i
            where[label] += 1

    return indexer.base, counts.base


cdef Py_ssize_t swap(numeric_t *a, numeric_t *b) noexcept nogil:
    cdef:
        numeric_t t  # C 语言扩展变量 t

    # Cython 不允许指针解引用，因此使用数组语法
    t = a[0]
    a[0] = b[0]
    b[0] = t
    return 0


cdef numeric_t kth_smallest_c(numeric_t* arr,
                              Py_ssize_t k, Py_ssize_t n) noexcept nogil:
    """
    查找第 k 小的元素。附加参数 n 指定了在 arr 中考虑的最大元素数量，与 groupby.pyx 中的使用兼容。
    """
    # 声明 C 语言扩展变量
    cdef:
        Py_ssize_t i, j, left, m  # Python ssize_t 类型变量 i, j, left, m
        numeric_t x  # 数值类型变量 x

    left = 0
    m = n - 1
    # 当左边界 `left` 小于右边界 `m` 时，进行循环，确保对整个数组进行排序
    while left < m:
        # 选择数组中间的元素作为基准值 `x`
        x = arr[k]
        # 设定左指针 `i` 初始位置为 `left`
        i = left
        # 设定右指针 `j` 初始位置为 `m`

        # 在未完成排序之前持续进行循环
        while 1:
            # 从左向右找到第一个大于或等于基准值 `x` 的元素
            while arr[i] < x:
                i += 1
            # 从右向左找到第一个小于或等于基准值 `x` 的元素
            while x < arr[j]:
                j -= 1
            # 如果左指针 `i` 仍在右指针 `j` 左侧，则交换它们指向的元素
            if i <= j:
                swap(&arr[i], &arr[j])
                i += 1
                j -= 1

            # 如果左指针 `i` 超过了右指针 `j`，则退出内部循环
            if i > j:
                break

        # 如果右指针 `j` 指向的位置小于目标位置 `k`，则更新左边界 `left` 为 `i`
        if j < k:
            left = i
        # 如果左指针 `i` 指向的位置大于目标位置 `k`，则更新右边界 `m` 为 `j`
        if k < i:
            m = j
    
    # 返回数组中第 `k` 小的元素
    return arr[k]
# 用于优化性能的 Cython 声明，禁用数组越界检查
@cython.boundscheck(False)
@cython.wraparound(False)
def kth_smallest(numeric_t[::1] arr, Py_ssize_t k) -> numeric_t:
    """
    Compute the kth smallest value in arr. Note that the input
    array will be modified.

    Parameters
    ----------
    arr : numeric[::1]
        Array to compute the kth smallest value for, must be
        contiguous
    k : Py_ssize_t
        Index of the smallest value to find in the array

    Returns
    -------
    numeric
        The kth smallest value in arr
    """
    # 声明结果变量
    cdef:
        numeric_t result

    # 使用 nogil 上下文提高性能，调用 C 函数进行计算
    with nogil:
        result = kth_smallest_c(&arr[0], k, arr.shape[0])

    return result


# ----------------------------------------------------------------------
# Pairwise correlation/covariance


# 用于优化性能的 Cython 声明，禁用数组越界检查和负数索引包装
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def nancorr(const float64_t[:, :] mat, bint cov=False, minp=None):
    """
    Calculate pairwise correlation or covariance matrix with handling for NaN values.

    Parameters
    ----------
    mat : ndarray[float64_t, ndim=2]
        Input matrix for correlation/covariance calculation
    cov : bool, optional
        Flag indicating whether to compute covariance (True) or correlation (False)
    minp : int or None, optional
        Minimum number of observations required for valid correlation/covariance

    Returns
    -------
    ndarray
        Pairwise correlation or covariance matrix
    """
    # Cython 变量声明
    cdef:
        Py_ssize_t i, xi, yi, N, K  # Python 整数和循环变量
        int64_t minpv  # 最小观测数的整数值
        float64_t[:, ::1] result  # 结果矩阵，存储相关系数或协方差
        uint8_t[:, :] mask  # 表示有效值的掩码
        int64_t nobs = 0  # 观测数计数器
        float64_t vx, vy, dx, dy, meanx, meany, divisor, ssqdmx, ssqdmy, covxy  # 中间变量

    # 获取输入矩阵的形状
    N, K = (<object>mat).shape

    # 设置最小观测数的默认值
    if minp is None:
        minpv = 1
    else:
        minpv = <int64_t>minp

    # 初始化结果矩阵，和有效值掩码
    result = np.empty((K, K), dtype=np.float64)
    mask = np.isfinite(mat).view(np.uint8)

    # 使用 nogil 上下文进行计算，提高性能
    with nogil:
        # 双重循环计算每对变量之间的相关系数或协方差
        for xi in range(K):
            for yi in range(xi + 1):
                # 使用 Welford 方法计算方差和协方差
                nobs = ssqdmx = ssqdmy = covxy = meanx = meany = 0
                for i in range(N):
                    if mask[i, xi] and mask[i, yi]:
                        vx = mat[i, xi]
                        vy = mat[i, yi]
                        nobs += 1
                        dx = vx - meanx
                        dy = vy - meany
                        meanx += 1. / nobs * dx
                        meany += 1. / nobs * dy
                        ssqdmx += (vx - meanx) * dx
                        ssqdmy += (vy - meany) * dy
                        covxy += (vx - meanx) * dy

                # 根据观测数判断是否满足最小观测数要求，更新结果矩阵
                if nobs < minpv:
                    result[xi, yi] = result[yi, xi] = NaN
                else:
                    divisor = (nobs - 1.0) if cov else sqrt(ssqdmx * ssqdmy)

                    if divisor != 0:
                        result[xi, yi] = result[yi, xi] = covxy / divisor
                    else:
                        result[xi, yi] = result[yi, xi] = NaN

    return result.base

# ----------------------------------------------------------------------
# Pairwise Spearman correlation


# 用于优化性能的 Cython 声明，禁用数组越界检查和负数索引包装
@cython.boundscheck(False)
@cython.wraparound(False)
def nancorr_spearman(ndarray[float64_t, ndim=2] mat, Py_ssize_t minp=1) -> ndarray:
    """
    Calculate pairwise Spearman correlation matrix with handling for NaN values.

    Parameters
    ----------
    mat : ndarray[float64_t, ndim=2]
        Input matrix for correlation calculation
    minp : int, optional
        Minimum number of observations required for valid correlation

    Returns
    -------
    ndarray
        Pairwise Spearman correlation matrix
    """
    # 定义 C 语言扩展变量
    cdef:
        # 声明变量 i, xi, yi, N, K，这些都是 C 语言的类型和变量声明
        Py_ssize_t i, xi, yi, N, K
        # 声明结果矩阵和排名后的矩阵，都是二维数组
        ndarray[float64_t, ndim=2] result
        ndarray[float64_t, ndim=2] ranked_mat
        # 声明用于排名的一维数组 rankedx 和 rankedy
        float64_t[::1] rankedx, rankedy
        # 声明用于存储掩码后数据的一维数组 maskedx 和 maskedy
        float64_t[::1] maskedx, maskedy
        # 声明掩码数组 mask，用于标识数据的有效性
        ndarray[uint8_t, ndim=2] mask
        # 声明 nobs 变量并初始化为 0
        int64_t nobs = 0
        # 声明 no_nans 变量，用于表示是否存在 NaN 值的布尔变量
        bint no_nans
        # 声明一些统计计算时使用的浮点数变量
        float64_t vx, vy, sumx, sumxx, sumyy, mean, divisor

    # 获取输入矩阵 mat 的形状并赋值给 N 和 K
    N, K = (<object>mat).shape

    # 处理特殊情况，当样本数 N 小于指定的最小样本数 minp 时，结果矩阵 result 全部为 NaN
    if N < minp:
        # 创建一个 KxK 大小的结果矩阵，全部填充为 NaN
        result = np.full((K, K), np.nan, dtype=np.float64)
        return result

    # 初始化结果矩阵 result 为一个 KxK 的空矩阵，用于存储计算结果
    result = np.empty((K, K), dtype=np.float64)
    # 创建掩码 mask，标识矩阵 mat 中的有限值
    mask = np.isfinite(mat).view(np.uint8)
    # 检查是否所有的值都不是 NaN，即是否没有 NaN 值存在
    no_nans = mask.all()

    # 初始化排名后的矩阵 ranked_mat，形状为 N 行 K 列的空矩阵
    ranked_mat = np.empty((N, K), dtype=np.float64)

    # 注意：在循环中我们使用 nobs 作为索引访问 maskedx 和 maskedy，但在这里使用 N 是安全的，
    # 因为 N >= nobs，并且值是连续存储的
    # 初始化 maskedx 和 maskedy 为长度为 N 的空数组，用于存储掩码后的数据
    maskedx = np.empty(N, dtype=np.float64)
    maskedy = np.empty(N, dtype=np.float64)

    # 对每一列进行循环，将每列数据进行一维排名并存储到 ranked_mat 中
    for i in range(K):
        ranked_mat[:, i] = rank_1d(mat[:, i])
    # 使用 nogil 上下文以避免全局解释器锁(GIL)，允许并行执行代码段
    with nogil:
        # 遍历 K 次，xi 从 0 到 K-1
        for xi in range(K):
            # 内层循环，遍历从 0 到 xi 的所有值
            for yi in range(xi + 1):
                # 初始化 sumx、sumxx、sumyy 为 0
                sumx = sumxx = sumyy = 0

                # 如果数据中没有 NaN 或 Inf，可以加快处理速度，避免掩码检查和数组重新赋值
                if no_nans:
                    # 计算均值 mean，对应于 (N + 1) / 2
                    mean = (N + 1) / 2.

                    # 计算协方差的分子部分
                    for i in range(N):
                        vx = ranked_mat[i, xi] - mean
                        vy = ranked_mat[i, yi] - mean

                        sumx += vx * vy
                        sumxx += vx * vx
                        sumyy += vy * vy
                else:
                    nobs = 0
                    # 检查是否需要重新计算排名
                    all_ranks = True
                    for i in range(N):
                        all_ranks &= not (mask[i, xi] ^ mask[i, yi])
                        if mask[i, xi] and mask[i, yi]:
                            maskedx[nobs] = ranked_mat[i, xi]
                            maskedy[nobs] = ranked_mat[i, yi]
                            nobs += 1

                    # 如果有效观测值数小于最小观测数阈值 minp，则设定结果为 NaN 并继续下一次循环
                    if nobs < minp:
                        result[xi, yi] = result[yi, xi] = NaN
                        continue
                    else:
                        # 如果不是所有数据点都有相同的排名，则重新计算排名
                        if not all_ranks:
                            with gil:
                                # 需要将数组切片到 nobs 长度，因为 rank_1d 需要长度为 nobs 的数组
                                rankedx = rank_1d(np.asarray(maskedx)[:nobs])
                                rankedy = rank_1d(np.asarray(maskedy)[:nobs])
                            for i in range(nobs):
                                maskedx[i] = rankedx[i]
                                maskedy[i] = rankedy[i]

                        # 计算均值 mean，对应于 (nobs + 1) / 2
                        mean = (nobs + 1) / 2.

                        # 计算协方差的分子部分
                        for i in range(nobs):
                            vx = maskedx[i] - mean
                            vy = maskedy[i] - mean

                            sumx += vx * vy
                            sumxx += vx * vx
                            sumyy += vy * vy

                # 计算协方差的分母部分的平方根作为除数
                divisor = sqrt(sumxx * sumyy)

                # 如果除数不为零，则计算并存储协方差；否则，结果设为 NaN
                if divisor != 0:
                    result[xi, yi] = result[yi, xi] = sumx / divisor
                else:
                    result[xi, yi] = result[yi, xi] = NaN

    # 返回最终计算得到的结果矩阵
    return result
# ----------------------------------------------------------------------

def validate_limit(nobs: int | None, limit=None) -> int:
    """
    Check that the `limit` argument is a positive integer.

    Parameters
    ----------
    nobs : int
        Number of observations, used as default limit if `limit` is None.
    limit : object
        Optional limit value to check.

    Returns
    -------
    int
        The validated limit.

    Raises
    ------
    ValueError
        If `limit` is not an integer or if `limit` is less than 1.
    """
    if limit is None:
        lim = nobs  # Use nobs as the limit if limit is not provided
    else:
        if not util.is_integer_object(limit):  # Check if limit is an integer
            raise ValueError("Limit must be an integer")
        if limit < 1:  # Check if limit is positive
            raise ValueError("Limit must be greater than 0")
        lim = limit  # Use provided limit if valid

    return lim


# TODO: overlap with libgroupby.group_fillna_indexer?
@cython.boundscheck(False)
@cython.wraparound(False)
def get_fill_indexer(const uint8_t[:] mask, limit=None):
    """
    Find an indexer to use for ffill to `take` on the array being filled.

    Parameters
    ----------
    mask : memoryview of uint8_t
        Boolean mask indicating missing values.
    limit : object
        Optional limit on consecutive missing values to fill.

    Returns
    -------
    ndarray[intp_t, ndim=1]
        Indexer array for forward fill operation.
    """
    cdef:
        ndarray[intp_t, ndim=1] indexer
        Py_ssize_t i, N = len(mask), last_valid
        int lim

        # fill_count is the number of consecutive NAs we have seen.
        # If it exceeds the given limit, we stop padding.
        int fill_count = 0

    lim = validate_limit(N, limit)  # Validate and set the limit
    indexer = np.empty(N, dtype=np.intp)  # Initialize indexer array

    last_valid = -1  # Initialize last valid index to -1

    for i in range(N):
        if not mask[i]:  # If mask[i] is False (not missing value)
            indexer[i] = i  # Indexer points to itself
            last_valid = i  # Update last valid index
            fill_count = 0  # Reset fill_count
        else:
            if fill_count < lim:
                indexer[i] = last_valid  # Use last valid index
            else:
                indexer[i] = -1  # Beyond limit, mark as invalid
            fill_count += 1  # Increment fill_count for consecutive NAs

    return indexer


@cython.boundscheck(False)
@cython.wraparound(False)
def pad(
    const numeric_object_t[:] old,
    const numeric_object_t[:] new,
    limit=None
) -> ndarray:
    # -> ndarray[intp_t, ndim=1]
    cdef:
        Py_ssize_t i, j, nleft, nright
        ndarray[intp_t, ndim=1] indexer
        numeric_object_t cur, next_val
        int lim, fill_count = 0

    nleft = len(old)  # Length of old array
    nright = len(new)  # Length of new array
    indexer = np.empty(nright, dtype=np.intp)  # Initialize indexer array
    indexer[:] = -1  # Initialize indexer with -1 values

    lim = validate_limit(nright, limit)  # Validate and set the limit

    if nleft == 0 or nright == 0 or new[nright - 1] < old[0]:
        return indexer  # Return empty indexer if arrays are empty or not overlapping

    i = j = 0  # Initialize iterators i and j

    cur = old[0]  # Start with the first value of old array

    while j <= nright - 1 and new[j] < cur:
        j += 1  # Move j until new[j] >= cur

    while True:
        if j == nright:
            break  # Break if reached end of new array

        if i == nleft - 1:
            while j < nright:
                if new[j] == cur:
                    indexer[j] = i
                elif new[j] > cur and fill_count < lim:
                    indexer[j] = i
                    fill_count += 1
                j += 1
            break

        next_val = old[i + 1]  # Get the next value from old array

        while j < nright and cur <= new[j] < next_val:
            if new[j] == cur:
                indexer[j] = i
            elif fill_count < lim:
                indexer[j] = i
                fill_count += 1
            j += 1

        fill_count = 0  # Reset fill_count
        i += 1  # Move to the next value in old array
        cur = next_val  # Update current value

    return indexer


@cython.boundscheck(False)
# 使用 Cython 的 wraparound(False) 装饰器，禁止越界访问检查
@cython.wraparound(False)
# 原地填充函数，用于在给定条件下填充数据
def pad_inplace(numeric_object_t[:] values, uint8_t[:] mask, limit=None):
    cdef:
        Py_ssize_t i, N  # 声明循环变量 i 和长度 N
        numeric_object_t val  # 声明数值对象 val
        uint8_t prev_mask  # 声明前一个掩码值
        int lim, fill_count = 0  # 声明限制值 lim 和填充计数 fill_count

    N = len(values)  # 获取 values 的长度

    # GH#2778，处理特定问题编号为 2778 的情况
    if N == 0:
        return  # 如果 values 为空，则直接返回

    lim = validate_limit(N, limit)  # 调用 validate_limit 函数获取有效限制值 lim

    val = values[0]  # 初始化 val 为 values 的第一个元素
    prev_mask = mask[0]  # 初始化 prev_mask 为 mask 的第一个元素
    for i in range(N):  # 遍历 values 的每个元素
        if mask[i]:  # 如果当前位置的掩码为真
            if fill_count >= lim:  # 如果填充计数超过或等于限制值 lim
                continue  # 跳过本次循环，继续下一次循环
            fill_count += 1  # 填充计数加一
            values[i] = val  # 将当前位置的值设为 val
            mask[i] = prev_mask  # 将当前位置的掩码设为 prev_mask
        else:  # 如果当前位置的掩码为假
            fill_count = 0  # 重置填充计数为零
            val = values[i]  # 更新 val 为当前位置的值
            prev_mask = mask[i]  # 更新 prev_mask 为当前位置的掩码值


# 使用 Cython 的 boundscheck(False) 和 wraparound(False) 装饰器，禁止边界检查和越界访问检查
@cython.boundscheck(False)
@cython.wraparound(False)
# 二维原地填充函数，用于在给定条件下填充二维数据
def pad_2d_inplace(numeric_object_t[:, :] values, uint8_t[:, :] mask, limit=None):
    cdef:
        Py_ssize_t i, j, N, K  # 声明循环变量 i、j 和长度变量 N、K
        numeric_object_t val  # 声明数值对象 val
        int lim, fill_count = 0  # 声明限制值 lim 和填充计数 fill_count

    K, N = (<object>values).shape  # 获取 values 的行数 K 和列数 N

    # GH#2778，处理特定问题编号为 2778 的情况
    if N == 0:
        return  # 如果 values 为空，则直接返回

    lim = validate_limit(N, limit)  # 调用 validate_limit 函数获取有效限制值 lim

    for j in range(K):  # 遍历 values 的每一行
        fill_count = 0  # 初始化填充计数为零
        val = values[j, 0]  # 初始化 val 为当前行的第一个元素
        for i in range(N):  # 遍历当前行的每个元素
            if mask[j, i]:  # 如果当前位置的掩码为真
                if fill_count >= lim or i == 0:  # 如果填充计数超过或等于限制值 lim，或者是第一个元素
                    continue  # 跳过本次循环，继续下一次循环
                fill_count += 1  # 填充计数加一
                values[j, i] = val  # 将当前位置的值设为 val
                mask[j, i] = False  # 将当前位置的掩码设为假
            else:  # 如果当前位置的掩码为假
                fill_count = 0  # 重置填充计数为零
                val = values[j, i]  # 更新 val 为当前位置的值


# 使用 Cython 的 boundscheck(False) 和 wraparound(False) 装饰器，禁止边界检查和越界访问检查
@cython.boundscheck(False)
@cython.wraparound(False)
# 后向填充函数，生成填充向量的逻辑
def backfill(
    const numeric_object_t[:] old,  # 旧数据的常量数组
    const numeric_object_t[:] new,  # 新数据的常量数组
    limit=None  # 填充限制的可选参数
) -> ndarray:  # 返回类型声明为一维整数数组
    """
    Backfilling logic for generating fill vector

    Diagram of what's going on

    Old      New    Fill vector    Mask
            .        0               1
            .        0               1
            .        0               1
    A        A        0               1
            .        1               1
            .        1               1
            .        1               1
            .        1               1
            .        1               1
    B        B        1               1
            .        2               1
            .        2               1
            .        2               1
    C        C        2               1
            .                        0
            .                        0
    D
    """
    cdef:
        Py_ssize_t i, j, nleft, nright  # 声明循环变量 i、j 和左右数据长度变量 nleft、nright
        ndarray[intp_t, ndim=1] indexer  # 声明索引器 indexer
        numeric_object_t cur, prev  # 声明当前值 cur 和前一个值 prev
        int lim, fill_count = 0  # 声明限制值 lim 和填充计数 fill_count

    nleft = len(old)  # 获取旧数据的长度
    nright = len(new)  # 获取新数据的长度
    indexer = np.empty(nright, dtype=np.intp)  # 创建一个空的索引器，长度为新数据的长度，数据类型为整数指针
    indexer[:] = -1  # 将索引器所有位置初始化为 -1

    lim = validate_limit(nright, limit)  # 调用 validate_limit 函数获取有效限制值 lim

    # 如果旧数据或新数据为空，或者新数据的第一个值大于旧数据的最后一个值，则直接返回索引器
    if nleft == 0 or nright == 0 or new[0] > old[nleft - 1]:
        return indexer

    i = nleft - 1  # 初始化 i 为旧数据的最后一个索引
    j = nright - 1  # 初始化 j 为新数据的最后一个索引

    cur = old[nleft - 1]  # 初始化 cur 为旧数据的最后一个值

    while j >= 0 and new[j] > cur:  # 循环直到 j 小于零或新数据的值小于等于 cur
        j -= 1  # 更新 j 的值为 j - 1
    # 进入无限循环，直到条件被打破
    while True:
        # 如果索引 j 小于 0，退出循环
        if j < 0:
            break

        # 如果 i 等于 0，执行以下操作
        if i == 0:
            # 从 j 开始向前遍历 new 列表
            while j >= 0:
                # 如果 new[j] 等于 cur，将 indexer[j] 设置为 i
                if new[j] == cur:
                    indexer[j] = i
                # 如果 new[j] 小于 cur 并且 fill_count 小于 lim，则将 indexer[j] 设置为 i，并增加 fill_count
                elif new[j] < cur and fill_count < lim:
                    indexer[j] = i
                    fill_count += 1
                # 减少 j 的值
                j -= 1
            # 跳出当前循环
            break

        # 获取 old 列表中的前一个元素作为 prev
        prev = old[i - 1]

        # 当 j 大于等于 0 并且 prev 小于 new[j] 小于等于 cur 时，执行以下操作
        while j >= 0 and prev < new[j] <= cur:
            # 如果 new[j] 等于 cur，将 indexer[j] 设置为 i
            if new[j] == cur:
                indexer[j] = i
            # 如果 new[j] 小于 cur 并且 fill_count 小于 lim，则将 indexer[j] 设置为 i，并增加 fill_count
            elif new[j] < cur and fill_count < lim:
                indexer[j] = i
                fill_count += 1
            # 减少 j 的值
            j -= 1

        # 重置 fill_count 为 0
        fill_count = 0
        # 减少 i 的值
        i -= 1
        # 更新 cur 的值为 prev
        cur = prev

    # 返回 indexer 列表作为结果
    return indexer
# 向后填充给定的一维数值对象和掩码，直接修改原始数组
def backfill_inplace(numeric_object_t[:] values, uint8_t[:] mask, limit=None):
    pad_inplace(values[::-1], mask[::-1], limit=limit)


# 向后填充给定的二维数值对象和掩码，直接修改原始数组
def backfill_2d_inplace(numeric_object_t[:, :] values,
                        uint8_t[:, :] mask,
                        limit=None):
    pad_2d_inplace(values[:, ::-1], mask[:, ::-1], limit)


# 判断给定的一维数值对象是否单调，返回三个布尔值的元组
@cython.boundscheck(False)
@cython.wraparound(False)
def is_monotonic(const numeric_object_t[:] arr, bint timelike):
    """
    Returns
    -------
    tuple
        is_monotonic_inc : bool
        is_monotonic_dec : bool
        is_strict_monotonic : bool
    """
    cdef:
        Py_ssize_t i, n
        numeric_object_t prev, cur
        bint is_monotonic_inc = 1
        bint is_monotonic_dec = 1
        bint is_unique = 1
        bint is_strict_monotonic = 1

    n = len(arr)

    if n == 1:
        if arr[0] != arr[0] or (numeric_object_t is int64_t and timelike and
                                arr[0] == NPY_NAT):
            # 单个值为 NaN
            return False, False, True
        else:
            return True, True, True
    elif n < 2:
        return True, True, True

    if timelike and <int64_t>arr[0] == NPY_NAT:
        return False, False, False

    if numeric_object_t is not object:
        with nogil:
            prev = arr[0]
            for i in range(1, n):
                cur = arr[i]
                if timelike and <int64_t>cur == NPY_NAT:
                    is_monotonic_inc = 0
                    is_monotonic_dec = 0
                    break
                if cur < prev:
                    is_monotonic_inc = 0
                elif cur > prev:
                    is_monotonic_dec = 0
                elif cur == prev:
                    is_unique = 0
                else:
                    # cur 或 prev 是 NaN
                    is_monotonic_inc = 0
                    is_monotonic_dec = 0
                    break
                if not is_monotonic_inc and not is_monotonic_dec:
                    is_monotonic_inc = 0
                    is_monotonic_dec = 0
                    break
                prev = cur
    else:
        # 对象数据类型，与上面类似，但不能使用 `with nogil`
        prev = arr[0]
        for i in range(1, n):
            cur = arr[i]
            if timelike and <int64_t>cur == NPY_NAT:
                is_monotonic_inc = 0
                is_monotonic_dec = 0
                break
            if cur < prev:
                is_monotonic_inc = 0
            elif cur > prev:
                is_monotonic_dec = 0
            elif cur == prev:
                is_unique = 0
            else:
                # cur 或 prev 是 NaN
                is_monotonic_inc = 0
                is_monotonic_dec = 0
                break
            if not is_monotonic_inc and not is_monotonic_dec:
                is_monotonic_inc = 0
                is_monotonic_dec = 0
                break
            prev = cur
    # 检查是否严格单调，需满足唯一性且单调递增或单调递减
    is_strict_monotonic = is_unique and (is_monotonic_inc or is_monotonic_dec)
    # 返回三个布尔值：是否单调递增，是否单调递减，是否严格单调
    return is_monotonic_inc, is_monotonic_dec, is_strict_monotonic
# ----------------------------------------------------------------------
# rank_1d, rank_2d
# ----------------------------------------------------------------------

# 这个函数用于确定在排序时使用哪个数值来代表缺失值，取决于我们是否希望缺失值最终位于最高/最低。第二个参数未被使用，但是需要用于融合类型的特殊化。
cdef numeric_object_t get_rank_nan_fill_val(
    bint rank_nans_highest,
    numeric_object_t val,
    bint is_datetimelike=False,
):
    """
    Return the value we'll use to represent missing values when sorting depending
    on if we'd like missing values to end up at the top/bottom. (The second parameter
    is unused, but needed for fused type specialization)
    """
    if numeric_object_t is int64_t and is_datetimelike and not rank_nans_highest:
        return NPY_NAT + 1

    if rank_nans_highest:
        if numeric_object_t is object:
            return Infinity()
        elif numeric_object_t is int64_t:
            return util.INT64_MAX
        elif numeric_object_t is int32_t:
            return util.INT32_MAX
        elif numeric_object_t is int16_t:
            return util.INT16_MAX
        elif numeric_object_t is int8_t:
            return util.INT8_MAX
        elif numeric_object_t is uint64_t:
            return util.UINT64_MAX
        elif numeric_object_t is uint32_t:
            return util.UINT32_MAX
        elif numeric_object_t is uint16_t:
            return util.UINT16_MAX
        elif numeric_object_t is uint8_t:
            return util.UINT8_MAX
        else:
            return np.inf
    else:
        if numeric_object_t is object:
            return NegInfinity()
        elif numeric_object_t is int64_t:
            # Note(jbrockmendel) 2022-03-15 for reasons unknown, using util.INT64_MIN
            #  instead of NPY_NAT here causes build warnings and failure in
            #  test_cummax_i8_at_implementation_bound
            return NPY_NAT
        elif numeric_object_t is int32_t:
            return util.INT32_MIN
        elif numeric_object_t is int16_t:
            return util.INT16_MIN
        elif numeric_object_t is int8_t:
            return util.INT8_MIN
        elif numeric_object_t is uint64_t:
            return 0
        elif numeric_object_t is uint32_t:
            return 0
        elif numeric_object_t is uint16_t:
            return 0
        elif numeric_object_t is uint8_t:
            return 0
        else:
            return -np.inf


@cython.wraparound(False)
@cython.boundscheck(False)
def rank_1d(
    ndarray[numeric_object_t, ndim=1] values,
    const intp_t[:] labels=None,
    bint is_datetimelike=False,
    ties_method="average",
    bint ascending=True,
    bint pct=False,
    na_option="keep",
    const uint8_t[:] mask=None,
):
    """
    Fast NaN-friendly version of ``scipy.stats.rankdata``.

    Parameters
    ----------
    values : array of numeric_object_t values to be ranked
    labels : np.ndarray[np.intp] or None
        Array containing unique label for each group, with its ordering
        matching up to the corresponding record in `values`. If not called
        from a groupby operation, will be None.
    # is_datetimelike 表示 values 是否包含类似日期时间的条目，初始值为 False
    is_datetimelike : bool, default False
        True if `values` contains datetime-like entries.
    
    # ties_method 定义了处理并列情况的方式，默认为 'average'
    ties_method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        * average: 组内等级的平均值
        * min: 组内最低等级
        * max: 组内最高等级
        * first: 根据数组中的顺序分配等级
        * dense: 类似 'min'，但是等级在组之间总是递增的
    
    # ascending 定义了排序顺序，默认为 True 表示升序排列
    ascending : bool, default True
        False 表示从高（1）到低（N）的等级
    
    # na_option 定义了处理缺失值的方式，默认为 'keep'
    na_option : {'keep', 'top', 'bottom'}, default 'keep'
        * keep: 保留 NA 值在原位置
        * top: 如果是升序则最小等级
        * bottom: 如果是降序则最小等级
    
    # pct 表示是否计算每个组内数据的百分比等级，默认为 False
    pct : bool, default False
        Compute percentage rank of data within each group
    
    # mask 是一个可选的布尔型 NumPy 数组，用于指定需要视为 NA 的位置，例如分类数据
    mask : np.ndarray[bool], optional, default None
        Specify locations to be treated as NA, for e.g. Categorical.
    """
    # 定义 Cython 变量
    cdef:
        TiebreakEnumType tiebreak  # 定义处理并列情况的枚举类型变量
        Py_ssize_t N  # 定义变量 N，表示 values 的长度
        int64_t[::1] grp_sizes  # 定义整型数组，存储组的大小
        intp_t[:] lexsort_indexer  # 定义整型数组的切片，用于排序索引
        float64_t[::1] out  # 定义浮点数数组，存储输出结果
        ndarray[numeric_object_t, ndim=1] masked_vals  # 定义一维数值对象数组，存储被掩码的值
        numeric_object_t[:] masked_vals_memview  # 定义数值对象数组的内存视图，用于访问数据
        bint keep_na, nans_rank_highest, check_labels, check_mask  # 定义布尔型变量，用于条件检查
        numeric_object_t nan_fill_val  # 定义数值对象变量，用于填充缺失值

    # 根据 ties_method 获取对应的处理并列情况的方法
    tiebreak = tiebreakers[ties_method]
    if tiebreak == TIEBREAK_FIRST:
        # 如果 tiebreak 为 TIEBREAK_FIRST 且不是升序，则使用降序的首个元素作为 tiebreak 方法
        if not ascending:
            tiebreak = TIEBREAK_FIRST_DESCENDING

    # 根据 na_option 确定是否保留 NA 值
    keep_na = na_option == "keep"

    # 获取 values 的长度 N
    N = len(values)
    
    # 如果 labels 不为 None，则断言 labels 的长度与 N 相等
    if labels is not None:
        assert len(labels) == N
    
    # 创建一个长度为 N 的空数组 out，用于存储结果
    out = np.empty(N)
    
    # 创建一个长度为 N 的整型数组 grp_sizes，初始值为 1
    grp_sizes = np.ones(N, dtype=np.int64)

    # 如果不需要比较 labels，则可以在后续步骤中进行优化
    check_labels = labels is not None

    # 根据数值对象的类型和其他条件，确定是否需要检查 mask
    check_mask = (
        numeric_object_t is float32_t
        or numeric_object_t is float64_t
        or numeric_object_t is object
        or (numeric_object_t is int64_t and is_datetimelike)
    )
    check_mask = check_mask or mask is not None

    # 如果数值对象是 object 类型且 values 的 dtype 不是 np.object_，则将 values 转换为对象类型
    if numeric_object_t is object and values.dtype != np.object_:
        masked_vals = values.astype("O")
    else:
        masked_vals = values.copy()

    # 根据 mask 的情况处理不同数值对象类型的缺失值
    if mask is not None:
        pass
    elif numeric_object_t is object:
        mask = isnaobj(masked_vals)
    elif numeric_object_t is int64_t and is_datetimelike:
        mask = (masked_vals == NPY_NAT).astype(np.uint8)
    elif numeric_object_t is float64_t or numeric_object_t is float32_t:
        mask = np.isnan(masked_vals).astype(np.uint8)
    else:
        mask = np.zeros(shape=len(masked_vals), dtype=np.uint8)

    # 如果 `na_option == 'top'`，则无论是升序还是降序，我们都希望将最小的等级分配给 NaN
    # 因此，如果是升序，
    # 使用 np.lexsort 对给定的 order 列表进行排序，生成排序后的索引
    lexsort_indexer = np.lexsort(order).astype(np.intp, copy=False)

    # 如果排序顺序为降序，则将索引数组反转，以得到降序排列的效果
    if not ascending:
        lexsort_indexer = lexsort_indexer[::-1]

    # 使用 nogil 上下文进行并行计算，调用 rank_sorted_1d 函数进行排序
    with nogil:
        rank_sorted_1d(
            out,                    # 输出数组，用于存储排序结果
            grp_sizes,              # 分组大小数组，指示每个分组的元素数量
            lexsort_indexer,        # 使用的排序索引数组
            masked_vals_memview,    # 带有缺失值填充后的值数组的内存视图
            mask,                   # 布尔掩码数组，用于标记缺失值位置
            check_mask=check_mask,  # 是否检查掩码的标志
            N=N,                    # 数据集中的总元素数量
            tiebreak=tiebreak,      # 解决平级情况的方法
            keep_na=keep_na,        # 是否保留缺失值
            pct=pct,                # 百分位数
            labels=labels,          # 标签数组，用于分组
        )

    # 将输出数组转换为 np.ndarray 类型，并返回作为函数结果
    return np.asarray(out)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void rank_sorted_1d(
    float64_t[::1] out,
    int64_t[::1] grp_sizes,
    const intp_t[:] sort_indexer,
    const numeric_object_t[:] masked_vals,
    const uint8_t[:] mask,
    bint check_mask,
    Py_ssize_t N,
    TiebreakEnumType tiebreak=TIEBREAK_AVERAGE,
    bint keep_na=True,
    bint pct=False,
    # https://github.com/cython/cython/issues/1630, only trailing arguments can
    # currently be omitted for cdef functions, which is why we keep this at the end
    const intp_t[:] labels=None,
) noexcept nogil:
    """
    See rank_1d.__doc__. Handles only actual ranking, so sorting and masking should
    be handled in the caller. Note that `out` and `grp_sizes` are modified inplace.

    Parameters
    ----------
    out : float64_t[::1]
        Array to store computed ranks
    grp_sizes : int64_t[::1]
        Array to store group counts, only used if pct=True. Should only be None
        if labels is None.
    sort_indexer : intp_t[:]
        Array of indices which sorts masked_vals
    masked_vals : numeric_object_t[:]
        The values input to rank_1d, with missing values replaced by fill values
    mask : uint8_t[:]
        Array where entries are True if the value is missing, False otherwise.
    check_mask : bool
        If False, assumes the mask is all False to skip mask indexing
    N : Py_ssize_t
        The number of elements to rank. Note: it is not always true that
        N == len(out) or N == len(masked_vals) (see `nancorr_spearman` usage for why)
    tiebreak : TiebreakEnumType, default TIEBREAK_AVERAGE
        See rank_1d.__doc__ for the different modes
    keep_na : bool, default True
        Whether or not to keep nulls
    pct : bool, default False
        Compute percentage rank of data within each group
    labels : See rank_1d.__doc__, default None. None implies all labels are the same.
    """

    cdef:
        Py_ssize_t i, j, dups=0, sum_ranks=0,
        Py_ssize_t grp_start=0, grp_vals_seen=1, grp_na_count=0
        bint at_end, next_val_diff, group_changed, check_labels
        int64_t grp_size

    check_labels = labels is not None

    # Loop over the length of the value array
    # each incremental i value can be looked up in the lexsort_indexer
    # array that we sorted previously, which gives us the location of
    # that sorted value for retrieval back from the original
    # values / masked_vals arrays
    # TODO(cython3): de-duplicate once cython supports conditional nogil
    if pct:
        for i in range(N):
            if grp_sizes[i] != 0:
                out[i] = out[i] / grp_sizes[i]


def rank_2d(
    ndarray[numeric_object_t, ndim=2] in_arr,
    int axis=0,
    bint is_datetimelike=False,
    ties_method="average",
    bint ascending=True,
    na_option="keep",
    bint pct=False,
):
    """
    Fast NaN-friendly version of ``scipy.stats.rankdata``.
    """
    # 定义变量：k 表示列数，n 表示行数，col 表示当前列索引
    # out 是一个二维数组，存储结果，以列为主序（column-major），确保列在内存中是连续存储的
    # grp_sizes 是一个一维数组，存储每个分组的大小
    # values 是一个二维数组，存储输入数据的副本
    # masked_vals 是一个二维数组，与 values 相同维度，用于存储可能被遮蔽的数据值
    # sort_indexer 是一个二维数组，存储排序索引
    # mask 是一个二维数组，存储遮蔽数据的掩码
    # tiebreak 是一个枚举类型，用于指定如何处理平局
    # check_mask 是一个布尔值，指示是否需要检查掩码
    # keep_na 是一个布尔值，指示是否保留缺失值
    # nans_rank_highest 是一个布尔值，指示缺失值在排序中是否排在最高位
    # nan_fill_val 是一个数值，用于填充缺失值

    tiebreak = tiebreakers[ties_method]  # 根据 ties_method 选择对应的 tiebreak 枚举值

    if tiebreak == TIEBREAK_FIRST:
        if not ascending:
            tiebreak = TIEBREAK_FIRST_DESCENDING
    # 如果 tiebreak 是 TIEBREAK_FIRST 且不是升序排列，则使用 TIEBREAK_FIRST_DESCENDING

    keep_na = na_option == "keep"  # 设置 keep_na 为 True，如果 na_option 是 "keep"

    # 检查是否需要对 mask 进行检查，避免不必要的掩码检查
    check_mask = (
        numeric_object_t is float32_t
        or numeric_object_t is float64_t
        or numeric_object_t is object
        or (numeric_object_t is int64_t and is_datetimelike)
    )

    if axis == 1:
        values = np.asarray(in_arr).T.copy()  # 如果 axis 为 1，转置输入数组并复制
    else:
        values = np.asarray(in_arr).copy()  # 否则，直接复制输入数组

    if numeric_object_t is object:
        if values.dtype != np.object_:
            values = values.astype("O")  # 如果数据类型是对象，确保 values 的类型是 np.object_

    nans_rank_highest = ascending ^ (na_option == "top")
    # 根据 ascending 和 na_option 决定缺失值在排序中是排在最高位还是最低位

    if check_mask:
        # 如果需要检查 mask
        nan_fill_val = get_rank_nan_fill_val(nans_rank_highest, <numeric_object_t>0)
        # 获取用于填充缺失值的值

        if numeric_object_t is object:
            mask = isnaobj(values).view(np.uint8)  # 如果数据类型是对象，生成缺失值的掩码
        elif numeric_object_t is float64_t or numeric_object_t is float32_t:
            mask = np.isnan(values).view(np.uint8)  # 如果数据类型是浮点数，生成 NaN 的掩码
        else:
            # 否则，即 int64 和日期时间类型
            mask = (values == NPY_NAT).view(np.uint8)  # 生成缺失值的掩码
        np.putmask(values, mask, nan_fill_val)  # 使用 nan_fill_val 填充 values 中的缺失值
    else:
        mask = np.zeros_like(values, dtype=np.uint8)  # 否则，生成全零掩码数组

    if nans_rank_highest:
        order = (values, mask)  # 根据 nans_rank_highest 确定排序顺序
    else:
        order = (values, ~np.asarray(mask))  # 根据 nans_rank_highest 确定排序逆序

    n, k = (<object>values).shape  # 获取 values 的形状，n 为行数，k 为列数
    out = np.empty((n, k), dtype="f8", order="F")  # 创建输出数组 out，列主序，数据类型为 float64

    grp_sizes = np.ones(n, dtype=np.int64)  # 初始化分组大小数组为全一

    # 如果需要检查 mask，则使用 lexsort 进行排序，否则使用 argsort 进行排序
    if check_mask:
        sort_indexer = np.lexsort(order, axis=0).astype(np.intp, copy=False)
    else:
        kind = "stable" if ties_method == "first" else None
        sort_indexer = values.argsort(axis=0, kind=kind).astype(np.intp, copy=False)

    if not ascending:
        sort_indexer = sort_indexer[::-1, :]  # 如果不是升序排列，逆序排列索引

    masked_vals = values  # 将 values 赋值给 masked_vals

    # 使用并行计算进行排序
    with nogil:
        for col in range(k):
            rank_sorted_1d(
                out[:, col],  # 输出结果列
                grp_sizes,  # 分组大小数组
                sort_indexer[:, col],  # 当前列的排序索引
                masked_vals[:, col],  # 当前列的遮蔽值
                mask[:, col],  # 当前列的掩码
                check_mask=check_mask,  # 是否需要检查掩码
                N=n,  # 数据行数
                tiebreak=tiebreak,  # 平局处理方式
                keep_na=keep_na,  # 是否保留缺失值
                pct=pct,  # 百分比参数
            )

    if axis == 1:
        return np.asarray(out.T)  # 如果 axis 是 1，返回转置后的 out
    else:
        return np.asarray(out)  # 否则，返回 out
# 定义一个融合类型 diff_t，可以是 float64_t、float32_t、int8_t、int16_t、int32_t 或 int64_t
ctypedef fused diff_t:
    float64_t
    float32_t
    int8_t
    int16_t
    int32_t
    int64_t

# 定义一个融合类型 out_t，可以是 float32_t、float64_t 或 int64_t
ctypedef fused out_t:
    float32_t
    float64_t
    int64_t

# 使用 Cython 的装饰器设置 boundscheck 为 False，避免边界检查开销
# 使用 Cython 的装饰器设置 wraparound 为 False，避免数组越界检查开销
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义函数 diff_2d，接受以下参数：
# - arr: diff_t 类型的二维数组，const 修饰表示不可修改
# - out: out_t 类型的二维 NumPy 数组，存储计算结果
# - periods: 表示计算的周期数
# - axis: 指定计算的轴
# - datetimelike: 布尔值，指示是否处理日期时间类型数据，默认为 False
def diff_2d(
    const diff_t[:, :] arr,
    ndarray[out_t, ndim=2] out,
    Py_ssize_t periods,
    int axis,
    bint datetimelike=False,
):
    cdef:
        Py_ssize_t i, j, sx, sy, start, stop
        bint f_contig = arr.is_f_contig()  # 检查 arr 是否是按 Fortran 顺序存储的

    # 如果 out_t 是 float32_t，并且 diff_t 不是 float32_t、int8_t 或 int16_t，则抛出 NotImplementedError 异常
    if (out_t is float32_t
            and not (diff_t is float32_t or diff_t is int8_t or diff_t is int16_t)):
        raise NotImplementedError  # pragma: no cover
    # 如果 out_t 是 float64_t，并且 diff_t 是 float32_t、int8_t 或 int16_t，则抛出 NotImplementedError 异常
    elif (out_t is float64_t
          and (diff_t is float32_t or diff_t is int8_t or diff_t is int16_t)):
        raise NotImplementedError  # pragma: no cover
    # 如果 out_t 是 int64_t，并且 diff_t 不是 int64_t，则抛出 NotImplementedError 异常
    elif out_t is int64_t and diff_t is not int64_t:
        # 只有在 datetimelike 为 True 时才会有 int64_t 类型的 out_t
        raise NotImplementedError  # pragma: no cover

# 从模板生成的代码段，包含 algos_common_helper.pxi 中的内容
include "algos_common_helper.pxi"

# 从模板生成的代码段，包含 algos_take_helper.pxi 中的内容
include "algos_take_helper.pxi"
```