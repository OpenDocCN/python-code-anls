# `D:\src\scipysrc\scipy\scipy\stats\_sobol.pyx`

```
# cython: language_level=3  # 设置Cython的语言级别为Python 3
# cython: cdivision=True  # 启用Cython中的C风格整数除法

import importlib.resources  # 导入Python内置模块importlib.resources

cimport cython  # 导入Cython的cython模块
cimport numpy as cnp  # 导入Cython中的NumPy模块，并重命名为cnp

import numpy as np  # 导入Python中的NumPy模块，并重命名为np

cnp.import_array()  # 调用Cython中的NumPy的import_array函数，用于初始化数组接口

# Parameters are linked to the direction numbers list.
# See `_initialize_direction_numbers` for more details.
# Declared using DEF to be known at compilation time for ``poly`` et ``vinit``
DEF MAXDIM = 21201  # 定义最大维度为21201，用于多项式和vinit矩阵
DEF MAXDEG = 18  # 定义最大多项式次数为18

ctypedef fused uint_32_64:  # 定义一个联合类型，包括uint32_t和uint64_t
    cnp.uint32_t  # 对应Cython中的32位无符号整数
    cnp.uint64_t  # 对应Cython中的64位无符号整数

# Needed to be accessed with python
cdef extern from *:
    """
    int MAXDIM_DEFINE = 21201;
    int MAXDEG_DEFINE = 18;
    """
    int MAXDIM_DEFINE  # 声明从外部引入的最大维度常量
    int MAXDEG_DEFINE  # 声明从外部引入的最大多项式次数常量

_MAXDIM = MAXDIM_DEFINE  # 将外部常量MAXDIM_DEFINE赋值给本地变量_MAXDIM
_MAXDEG = MAXDEG_DEFINE  # 将外部常量MAXDEG_DEFINE赋值给本地变量_MAXDEG

_poly_dict = {}  # 初始化一个空字典，用于缓存poly矩阵
_vinit_dict = {}  # 初始化一个空字典，用于缓存vinit矩阵


def get_poly_vinit(kind, dtype):
    """Initialize and cache the direction numbers.

    Uses a dictionary to store the arrays. `kind` allows to select which
    dictionary to pull. The key of each dictionary corresponds to the `dtype`.
    If the key is not present in any of the dictionary, both dictionaries are
    initialized with `_initialize_direction_numbers`, for the given `dtype`.

    This is only used during the initialization step in `_initialize_v`.

    Parameters
    ----------
    kind : {'poly', 'vinit'}
        Select which dictionary to pull.
    dtype : {np.uint32, np.uint64}
        Which dtype to use.

    Returns
    -------
    poly_vinit : np.ndarray
        Either ``poly`` or ``vinit`` matrix.

    """
    if kind == 'poly':
        poly_vinit = _poly_dict.get(dtype)  # 从_poly_dict字典获取指定dtype的poly矩阵
    else:
        poly_vinit = _vinit_dict.get(dtype)  # 从_vinit_dict字典获取指定dtype的vinit矩阵

    if poly_vinit is None:
        _poly_dict[dtype] = np.empty((MAXDIM,), dtype=dtype)  # 初始化一个dtype类型的空poly矩阵
        _vinit_dict[dtype] = np.empty((MAXDIM, MAXDEG), dtype=dtype)  # 初始化一个dtype类型的空vinit矩阵

        _initialize_direction_numbers(_poly_dict[dtype], _vinit_dict[dtype], dtype)  # 调用函数初始化方向数

        if kind == 'poly':
            poly_vinit = _poly_dict.get(dtype)  # 再次尝试从_poly_dict获取poly矩阵
        else:
            poly_vinit = _vinit_dict.get(dtype)  # 再次尝试从_vinit_dict获取vinit矩阵

    return poly_vinit  # 返回获取到的poly_vinit矩阵


def _initialize_direction_numbers(poly, vinit, dtype):
    """Load direction numbers into two arrays.

    Parameters
    ----------
    poly, vinit : np.ndarray
        Direction numbers arrays to fill.
    dtype : {np.uint32, np.uint64}
        Which dtype to use.

    Notes
    -----
    Direction numbers obtained using the search criterion D(6)
    up to the dimension 21201. This is the recommended choice by the authors.

    Original data can be found at https://web.maths.unsw.edu.au/~fkuo/sobol/.
    For additional details on the quantities involved, see [1].

    [1] S. Joe and F. Y. Kuo. Remark on algorithm 659: Implementing sobol's
        quasirandom sequence generator. ACM Trans. Math. Softw., 29(1):49-57,
        Mar. 2003.

    The C-code generated from putting the numbers in as literals is obscenely
    large/inefficient. The data file was thus packaged and save as an .npz data

    """
    pass  # 此函数暂时为空，仅包含文档字符串和一个占位符pass
    """
    _curdir = importlib.resources.files("scipy.stats")
    _npzfile = _curdir.joinpath("_sobol_direction_numbers.npz")
    # 获取资源文件夹路径，此处假设资源文件位于scipy.stats中
    with importlib.resources.as_file(_npzfile) as f:
        # 打开资源文件_npzfile作为文件对象f
        dns = np.load(f)

    # 从加载的资源中提取多项式和初始向量
    dns_poly = dns["poly"].astype(dtype)
    dns_vinit = dns["vinit"].astype(dtype)

    # 将加载的多项式数据复制到poly数组中
    poly[...] = dns_poly
    # 将加载的初始向量数据复制到vinit数组中
    vinit[...] = dns_vinit
    ```
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int bit_length(uint_32_64 n) noexcept:
    cdef int bits = 0  # 初始化一个整数变量 bits 为 0，用于计算整数 n 的位数
    cdef uint_32_64 nloc = n  # 将输入参数 n 复制给 nloc，这样可以在不改变原始输入的情况下进行操作
    while nloc != 0:  # 循环直到 nloc 变为 0
        nloc >>= 1  # 将 nloc 右移一位
        bits += 1  # bits 加 1，统计右移的次数，即 n 的位数
    return bits  # 返回整数 n 的位数


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int low_0_bit(uint_32_64 x) noexcept nogil:
    """Get the position of the right-most 0 bit for an integer.

    Examples:
    >>> low_0_bit(0)
    1
    >>> low_0_bit(1)
    2
    >>> low_0_bit(2)
    1
    >>> low_0_bit(5)
    2
    >>> low_0_bit(7)
    4

    Parameters
    ----------
    x : int
        An integer.

    Returns
    -------
    position : int
        Position of the right-most 0 bit.

    """
    cdef int i = 0  # 初始化整数变量 i 为 0，用于记录右移的位数
    while x & (1 << i) != 0:  # 当 x 与 (1 左移 i 位) 的结果不为 0 时循环
        i += 1  # i 加 1，找到右起第一个为 0 的位的位置
    return i + 1  # 返回右起第一个为 0 的位的位置（从 1 开始计数）


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int ibits(uint_32_64 x, const int pos, const int length) noexcept nogil:
    """Extract a sequence of bits from the bit representation of an integer.

    Extract the sequence from position `pos` (inclusive) to ``pos + length``
    (not inclusive), leftwise.

    Examples:
    >>> ibits(1, 0, 1)
    1
    >>> ibits(1, 1, 1)
    0
    >>> ibits(2, 0, 1)
    0
    >>> ibits(2, 0, 2)
    2
    >>> ibits(25, 1, 5)
    12


    Parameters
    ----------
    x : int
        Integer to convert to bit representation.
    pos : int
        Starting position of sequence in bit representation of integer.
    length : int
        Length of sequence (number of bits).

    Returns
    -------
    ibits : int
        Integer value corresponding to bit sequence.

    """
    return (x >> pos) & ((1 << length) - 1)  # 返回从整数 x 的位表示中提取的从 pos 开始长度为 length 的位序列的整数值


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _initialize_v(
    uint_32_64[:, ::1] v, const int dim, const int bits
) noexcept:
    """Initialize matrix of size ``dim * bits`` with direction numbers."""
    cdef int d, i, j, k, m  # 声明整数变量 d, i, j, k, m
    cdef uint_32_64 p, newv, pow2  # 声明无符号整数变量 p, newv, pow2
    cdef uint_32_64[:] poly = get_poly_vinit(
        'poly',
        np.uint32 if uint_32_64 is cnp.uint32_t else np.uint64
    )  # 调用函数 get_poly_vinit 获取名为 'poly' 的多项式系数数组

    cdef uint_32_64[:, ::1] vinit = get_poly_vinit(
        'vinit',
        np.uint32 if uint_32_64 is cnp.uint32_t else np.uint64
    )  # 调用函数 get_poly_vinit 获取名为 'vinit' 的初始化矩阵

    if dim == 0:  # 如果 dim 为 0，直接返回，不做初始化操作
        return

    # first row of v is all 1s
    for i in range(bits):  # 循环遍历每列
        v[0, i] = 1  # 将第一行的每列都设为 1

    # Remaining rows of v (row 2 through dim, indexed by [1:dim])
    # 对于每一个维度 d，从 1 到 dim-1 进行循环
    for d in range(1, dim):
        # 从多项式数组 poly 中获取当前维度的多项式 p
        p = poly[d]
        # 计算多项式 p 的比特长度，并减去 1，得到 m
        m = bit_length(p) - 1

        # 将 vinit 中维度 d 的前 m 个元素复制到 v 的对应位置
        v[d, :m] = vinit[d, :m]

        # 根据 Bratley 和 Fox 的算法填充 v 的剩余元素，参考文献详见注释
        for j in range(m, bits):
            # 初始化 newv 为 v[d, j-m]
            newv = v[d, j - m]
            # 初始化 pow2 为 1
            pow2 = 1
            # 对于每个 k 在范围 m 内进行循环
            for k in range(m):
                # pow2 左移一位
                pow2 = pow2 << 1
                # 如果 p 的右移 m-1-k 位的第一个比特为 1
                if (p >> (m - 1 - k)) & 1:
                    # newv 异或 (pow2 乘以 v[d, j-k-1])
                    newv = newv ^ (pow2 * v[d, j - k - 1])
            # 设置 v[d, j] 为计算得到的 newv
            v[d, j] = newv

    # 将 v 的每一列乘以 2 的幂：
    # v * [2^(bits-1), 2^(bits-2),..., 2, 1]
    pow2 = 1
    # 对于每一个 d 在范围 bits 内进行循环
    for d in range(bits):
        # 对于每一个 i 在范围 dim 内进行循环
        for i in range(dim):
            # 将 v[i, bits-1-d] 乘以 pow2
            v[i, bits - 1 - d] *= pow2
        # pow2 左移一位
        pow2 = pow2 << 1
# 定义一个函数 `_draw`，作为对外接口，用于调用 Cython 实现的 `draw` 函数
def _draw(
    n,
    num_gen,
    const int dim,
    const cnp.float64_t scale,
    const uint_32_64[:, ::1] sv,
    uint_32_64[::1] quasi,
    cnp.float64_t[:, ::1] sample
):
    # 将输入参数类型转换为 Cython 可识别的类型并调用 `draw` 函数
    # 这里的 `cdef` 关键字用于定义 Cython 中的 C 变量
    cdef uint_32_64 n_ = n
    cdef uint_32_64 num_gen_ = num_gen
    draw(n_, num_gen_, dim, scale, sv, quasi, sample)


# 在 Cython 中声明函数 `draw`，实现了具体的随机数生成算法
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void draw(
    const uint_32_64 n,
    const uint_32_64 num_gen,
    const int dim,
    const cnp.float64_t scale,
    const uint_32_64[:, ::1] sv,
    uint_32_64[::1] quasi,
    cnp.float64_t[:, ::1] sample
) noexcept nogil:
    # 声明 C 语言中的变量
    cdef int j, l
    cdef uint_32_64 num_gen_loc = num_gen
    cdef uint_32_64 i, qtmp

    # 循环生成随机数样本
    for i in range(n):
        # 使用位操作找到 num_gen_loc 的最低位 0
        l = low_0_bit(num_gen_loc)
        for j in range(dim):
            # 执行混合和缩放操作，生成最终的样本数据
            qtmp = quasi[j] ^ sv[j, l - 1]
            quasi[j] = qtmp
            sample[i, j] = qtmp * scale
        num_gen_loc += 1


# 在 Cython 中声明函数 `_fast_forward`，用于快速前进状态变量 `quasi`
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _fast_forward(const uint_32_64 n,
                         const uint_32_64 num_gen,
                         const int dim,
                         const uint_32_64[:, ::1] sv,
                         uint_32_64[::1] quasi) noexcept nogil:
    # 声明 C 语言中的变量
    cdef int j, l
    cdef uint_32_64 num_gen_loc = num_gen
    cdef uint_32_64 i
    # 循环进行状态前进操作
    for i in range(n):
        # 使用位操作找到 num_gen_loc 的最低位 0
        l = low_0_bit(num_gen_loc)
        for j in range(dim):
            # 执行状态变量的前进操作
            quasi[j] = quasi[j] ^ sv[j, l - 1]
        num_gen_loc += 1


# 在 Cython 中声明函数 `cdot_pow2`，计算给定数组 `a` 的二进制表示对应的十进制值
@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint_32_64 cdot_pow2(const uint_32_64[::1] a) noexcept nogil:
    # 声明 C 语言中的变量
    cdef int i
    cdef int size = a.shape[0]
    cdef uint_32_64 z = 0
    cdef uint_32_64 pow2 = 1
    # 循环计算二进制值对应的十进制值
    for i in range(size):
        z += a[size - 1 - i] * pow2
        pow2 *= 2
    return z


# 在 Cython 中声明函数 `_cscramble`，实现使用线性矩阵乱序的数据混淆操作
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _cscramble(const int dim,
                      const int bits,
                      uint_32_64[:, :, ::1] ltm,
                      uint_32_64[:, ::1] sv) noexcept nogil:
    """Scrambling using (left) linear matrix scramble (LMS)."""
    # 声明 C 语言中的变量
    cdef int d, i, j, k, p
    cdef uint_32_64 l, lsmdp, t1, t2, vdj

    # 将线性矩阵的对角线元素设为 1
    for d in range(dim):
        for i in range(bits):
            ltm[d, i, i] = 1

    # 执行线性矩阵乱序算法
    for d in range(dim):
        for j in range(bits):
            vdj = sv[d, j]
            l = 1
            t2 = 0
            for p in range(bits - 1, -1, -1):
                lsmdp = cdot_pow2(ltm[d, p, :])
                t1 = 0
                for k in range(bits):
                    t1 += ibits(lsmdp, k, 1) * ibits(vdj, k, 1)
                t1 = t1 % 2
                t2 = t2 + t1 * l
                l = 2 * l
            sv[d, j] = t2


# 在 Cython 中声明函数 `_fill_p_cumulative`，用于计算累积概率分布
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _fill_p_cumulative(const cnp.float_t[::1] p,
                              cnp.float_t[::1] p_cumulative) noexcept nogil:
    # 声明 C 语言中的变量
    cdef int i
    # 定义变量 len_p，表示数组 p 的长度
    cdef int len_p = p.shape[0]
    # 定义变量 tot，初始化为 0，用于累计总和
    cdef float tot = 0
    # 定义变量 t，用于暂存累加值
    cdef float t
    # 遍历数组 p 的索引范围
    for i in range(len_p):
        # 计算累加值，并存储到变量 t 中
        t = tot + p[i]
        # 将累加值 t 存储到累计概率数组 p_cumulative 的对应位置 i
        p_cumulative[i] = t
        # 更新总累计值 tot 为 t，以备下一次迭代使用
        tot = t
# 使用 Cython 的装饰器设置边界检查和索引包装功能关闭
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个 CPython 可调用的函数 _categorize，接受三个参数：draws 数组，p_cumulative 数组和结果数组 result
cpdef void _categorize(const cnp.float_t[::1] draws,
                       const cnp.float_t[::1] p_cumulative,
                       cnp.intp_t[::1] result) noexcept nogil:
    # 声明变量 i 作为整数
    cdef int i
    # 获取 p_cumulative 数组的长度，赋值给 n_p
    cdef int n_p = p_cumulative.shape[0]
    # 循环遍历 draws 数组
    for i in range(draws.shape[0]):
        # 调用 _find_index 函数，查找 draws[i] 在 p_cumulative 中的索引，返回结果赋给 j
        j = _find_index(p_cumulative, n_p, draws[i])
        # 将结果数组 result 中索引为 j 的元素加一
        result[j] = result[j] + 1


# 使用 Cython 的装饰器设置边界检查和索引包装功能关闭
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个 CPython 可调用的函数 _find_index，接受 p_cumulative 数组、size 整数和 value 浮点数作为参数
cdef int _find_index(const cnp.float_t[::1] p_cumulative,
                     const int size,
                     const float value) noexcept nogil:
    # 声明变量 l 和 r 作为整数，分别初始化为 0 和 size - 1
    cdef int l = 0
    cdef int r = size - 1
    # 声明变量 m 作为整数
    cdef int m
    # 当 r 大于 l 时进行循环
    while r > l:
        # 计算中间值 m
        m = (l + r) // 2
        # 如果 value 大于 p_cumulative[m]，将 l 设置为 m + 1
        if value > p_cumulative[m]:
            l = m + 1
        # 否则将 r 设置为 m
        else:
            r = m
    # 返回 r，即查找到的索引值
    return r


# 定义一个 Python 函数 _test_find_index，用于测试 _find_index 函数
def _test_find_index(p_cumulative, size, value):
    # type: (np.ndarray, int, float) -> int
    """Wrapper for testing in python"""
    # 调用 _find_index 函数，返回其结果
    return _find_index(p_cumulative, size, value)
```