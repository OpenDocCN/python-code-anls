# `D:\src\scipysrc\scikit-learn\sklearn\utils\_random.pyx`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Random utility function
=======================
This module complements missing features of ``numpy.random``.

The module contains:
    * Several algorithms to sample integers without replacement.
    * Fast rand_r alternative based on xor shifts
"""

# 导入 NumPy 库并命名为 np
import numpy as np
# 导入本地模块 check_random_state
from . import check_random_state

# 导入 C 扩展中定义的 intp_t 类型
from ._typedefs cimport intp_t

# 定义默认种子为 1 的无符号 32 位整数
cdef uint32_t DEFAULT_SEED = 1

# 定义一个融合类型 fused 类型，用于兼容旧版和新版 NumPy 默认的整数类型
# 在 Windows 上，`long` 类型不总是与 `inp_t` 匹配
# 参见 `sample_without_replacement` Python 函数的注释获取更多细节
ctypedef fused default_int:
    intp_t
    long

# 定义 Python 函数 _sample_without_replacement_check_input，用于检查采样参数的一致性
cpdef _sample_without_replacement_check_input(default_int n_population,
                                              default_int n_samples):
    """ Check that input are consistent for sample_without_replacement"""
    # 如果 n_population 小于 0，则抛出 ValueError 异常
    if n_population < 0:
        raise ValueError('n_population should be greater than 0, got %s.'
                         % n_population)

    # 如果 n_samples 大于 n_population，则抛出 ValueError 异常
    if n_samples > n_population:
        raise ValueError('n_population should be greater or equal than '
                         'n_samples, got n_samples > n_population (%s > %s)'
                         % (n_samples, n_population))


# 定义 Python 函数 _sample_without_replacement_with_tracking_selection，用于无重复采样
cpdef _sample_without_replacement_with_tracking_selection(
        default_int n_population,
        default_int n_samples,
        random_state=None):
    r"""Sample integers without replacement.

    Select n_samples integers from the set [0, n_population) without
    replacement.

    Time complexity:
        - Worst-case: unbounded
        - Average-case:
            O(O(np.random.randint) * \sum_{i=1}^n_samples 1 /
                                              (1 - i / n_population)))
            <= O(O(np.random.randint) *
                   n_population * ln((n_population - 2)
                                     /(n_population - 1 - n_samples)))
            <= O(O(np.random.randint) *
                 n_population * 1 / (1 - n_samples / n_population))

    Space complexity of O(n_samples) in a python set.


    Parameters
    ----------
    n_population : int
        The size of the set to sample from.

    n_samples : int
        The number of integer to sample.

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    out : ndarray of shape (n_samples,)
        The sampled subsets of integer.
    """
    # 调用 _sample_without_replacement_check_input 函数，检查参数的合法性
    _sample_without_replacement_check_input(n_population, n_samples)

    # 定义 C 语言扩展类型的局部变量 i, j
    cdef default_int i
    cdef default_int j
    # 创建一个长度为 n_samples 的整数数组 out，用于存放采样结果
    cdef default_int[::1] out = np.empty((n_samples, ), dtype=int)

    # 使用 check_random_state 函数获取随机数生成器 rng
    rng = check_random_state(random_state)
    # 将rng的randint方法赋值给rng_randint变量，以便在后续代码中直接使用rng_randint代替rng.randint
    rng_randint = rng.randint

    # 创建一个空集合selected，用于存储选中的索引值
    # 以下代码受Python核心中random.sample启发而来
    cdef set selected = set()

    # 循环生成n_samples个随机样本
    for i in range(n_samples):
        # 从0到n_population-1中随机选择一个整数作为索引值j
        j = rng_randint(n_population)
        
        # 如果j已经在selected集合中存在，则重新生成j，直到j不在selected中
        while j in selected:
            j = rng_randint(n_population)
        
        # 将新生成的唯一索引j添加到selected集合中
        selected.add(j)
        
        # 将选中的索引j存储到输出数组out的第i个位置
        out[i] = j

    # 将输出数组out转换为NumPy数组，并返回
    return np.asarray(out)
# 定义一个 CPython 的扩展函数，用于无重复抽样（不放回抽样）的池化实现
def _sample_without_replacement_with_pool(default_int n_population,
                                          default_int n_samples,
                                          random_state=None):
    """Sample integers without replacement.

    Select n_samples integers from the set [0, n_population) without
    replacement.

    Time complexity: O(n_population + O(np.random.randint) * n_samples)
    时间复杂度：O(n_population + O(np.random.randint) * n_samples)

    Space complexity of O(n_population + n_samples).
    空间复杂度：O(n_population + n_samples)

    Parameters
    ----------
    n_population : int
        The size of the set to sample from. 抽样集合的大小。

    n_samples : int
        The number of integers to sample. 要抽样的整数数量。

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. 随机数种子或随机数生成器。

    Returns
    -------
    out : ndarray of shape (n_samples,)
        The sampled subsets of integers. 返回抽样的整数子集。
    """
    # 检查输入参数的有效性
    _sample_without_replacement_check_input(n_population, n_samples)

    cdef default_int i
    cdef default_int j
    cdef default_int[::1] out = np.empty((n_samples,), dtype=int)
    cdef default_int[::1] pool = np.empty((n_population,), dtype=int)

    # 使用给定的随机数生成器初始化
    rng = check_random_state(random_state)
    rng_randint = rng.randint

    # 初始化池
    for i in range(n_population):
        pool[i] = i

    # 下面的代码受 Python 核心中 random.sample 的启发
    for i in range(n_samples):
        # 从 [0, n_population - i) 区间随机选择一个整数
        j = rng_randint(n_population - i)
        out[i] = pool[j]
        # 将未选择的项移动到空缺位置
        pool[j] = pool[n_population - i - 1]

    return np.asarray(out)


# 定义一个 CPython 的扩展函数，用于无重复抽样（不放回抽样）的蓄水池抽样实现
cpdef _sample_without_replacement_with_reservoir_sampling(
    default_int n_population,
    default_int n_samples,
    random_state=None
):
    """Sample integers without replacement.

    Select n_samples integers from the set [0, n_population) without
    replacement.

    Time complexity of
        O((n_population - n_samples) * O(np.random.randint) + n_samples)
    时间复杂度：O((n_population - n_samples) * O(np.random.randint) + n_samples)

    Space complexity of O(n_samples)
    空间复杂度：O(n_samples)

    Parameters
    ----------
    n_population : int
        The size of the set to sample from. 抽样集合的大小。

    n_samples : int
        The number of integers to sample. 要抽样的整数数量。

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. 随机数种子或随机数生成器。

    Returns
    -------
    out : ndarray of shape (n_samples,)
        The sampled subsets of integers. The order of the items is not
        necessarily random. Use a random permutation of the array if the order
        of the items has to be randomized. 返回抽样的整数子集。如果需要随机排列项的顺序，
        可以使用数组的随机排列。
    """
    # 调用函数 _sample_without_replacement_check_input，验证输入参数 n_population 和 n_samples 是否合法
    _sample_without_replacement_check_input(n_population, n_samples)

    # 声明变量 i 和 j 为默认的整数类型，用于后续循环和索引操作
    cdef default_int i
    cdef default_int j

    # 创建一个一维的空整数数组 out，长度为 n_samples，用于存储采样结果
    cdef default_int[::1] out = np.empty((n_samples, ), dtype=int)

    # 使用给定的 random_state 检查并初始化随机数生成器 rng
    rng = check_random_state(random_state)
    # 从 rng 中获取随机整数生成函数 rng.randint
    rng_randint = rng.randint

    # 下面这段注释解释了这个 Cython 实现的基础来源和参考：
    # 这个 Cython 实现基于 Robert Kern 的实现：
    # http://mail.scipy.org/pipermail/numpy-discussion/2010-December/054289.html

    # 初始化 out 数组，将其填充为简单顺序：0, 1, 2, ..., n_samples-1
    for i in range(n_samples):
        out[i] = i

    # 对于从 n_samples 到 n_population-1 的范围内的每个 i 值进行采样
    for i from n_samples <= i < n_population:
        # 从 0 到 i (含) 中随机选择一个索引 j
        j = rng_randint(0, i + 1)
        # 如果随机选择的 j 在有效采样范围内（小于 n_samples），则用 i 替换 out[j]
        if j < n_samples:
            out[j] = i

    # 将 out 数组转换为 NumPy 数组并返回
    return np.asarray(out)
cdef _sample_without_replacement(default_int n_population,
                                 default_int n_samples,
                                 method="auto",
                                 random_state=None):
    """Sample integers without replacement.

    Private function for the implementation, see sample_without_replacement
    documentation for more details.
    """
    # 检查输入参数的有效性
    _sample_without_replacement_check_input(n_population, n_samples)

    # 定义可用的抽样方法列表
    all_methods = ("auto", "tracking_selection", "reservoir_sampling", "pool")

    # 计算抽样比例，避免除数为零的情况
    ratio = <double> n_samples / n_population if n_population != 0.0 else 1.0

    # 根据方法选择具体的抽样实现方式
    # 如果方法为 "auto" 并且比例在 0.01 到 0.99 之间，则使用排列方法
    if method == "auto" and ratio > 0.01 and ratio < 0.99:
        # 根据随机状态生成随机数生成器
        rng = check_random_state(random_state)
        # 返回随机排列的前 n_samples 个元素
        return rng.permutation(n_population)[:n_samples]

    # 如果方法为 "auto" 或者 "tracking_selection"
    if method == "auto" or method == "tracking_selection":
        # 如果比例小于 0.2，则使用基于跟踪选择的抽样方法
        if ratio < 0.2:
            return _sample_without_replacement_with_tracking_selection(
                n_population, n_samples, random_state)
        else:
            # 否则使用基于水库抽样的方法
            return _sample_without_replacement_with_reservoir_sampling(
                n_population, n_samples, random_state)

    # 如果方法为 "reservoir_sampling"
    elif method == "reservoir_sampling":
        # 使用基于水库抽样的方法
        return _sample_without_replacement_with_reservoir_sampling(
            n_population, n_samples, random_state)

    # 如果方法为 "pool"
    elif method == "pool":
        # 使用基于池的方法
        return _sample_without_replacement_with_pool(n_population, n_samples,
                                                     random_state)
    else:
        # 如果方法不在预期的方法列表中，则抛出 ValueError 异常
        raise ValueError('Expected a method name in %s, got %s. '
                         % (all_methods, method))


def sample_without_replacement(
        object n_population, object n_samples, method="auto", random_state=None):
    """Sample integers without replacement.

    Select n_samples integers from the set [0, n_population) without
    replacement.


    Parameters
    ----------
    n_population : int
        The size of the set to sample from.

    n_samples : int
        The number of integer to sample.

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    method : {"auto", "tracking_selection", "reservoir_sampling", "pool"}, \
            default='auto'
    # 方法选择参数，可选值包括 "auto", "tracking_selection", "reservoir_sampling", "pool"，默认为 'auto'
        If method == "auto", the ratio of n_samples / n_population is used
        to determine which algorithm to use:
        # 如果 method == "auto"，根据 n_samples / n_population 的比率确定使用哪种算法：
        If ratio is between 0 and 0.01, tracking selection is used.
        # 如果比率在 0 到 0.01 之间，则使用 tracking selection 算法。
        If ratio is between 0.01 and 0.99, numpy.random.permutation is used.
        # 如果比率在 0.01 到 0.99 之间，则使用 numpy.random.permutation 算法。
        If ratio is greater than 0.99, reservoir sampling is used.
        # 如果比率大于 0.99，则使用 reservoir sampling 算法。
        The order of the selected integers is undefined. If a random order is
        desired, the selected subset should be shuffled.
        # 所选整数的顺序是未定义的。如果需要随机顺序，应该对选定的子集进行洗牌。

        If method =="tracking_selection", a set based implementation is used
        which is suitable for `n_samples` <<< `n_population`.
        # 如果 method =="tracking_selection"，使用基于集合的实现，适用于 `n_samples` 远小于 `n_population`。

        If method == "reservoir_sampling", a reservoir sampling algorithm is
        used which is suitable for high memory constraint or when
        O(`n_samples`) ~ O(`n_population`).
        # 如果 method == "reservoir_sampling"，使用 reservoir sampling 算法，适用于高内存约束或当 O(`n_samples`) 与 O(`n_population`) 相当时。
        The order of the selected integers is undefined. If a random order is
        desired, the selected subset should be shuffled.
        # 所选整数的顺序是未定义的。如果需要随机顺序，应该对选定的子集进行洗牌。

        If method == "pool", a pool based algorithm is particularly fast, even
        faster than the tracking selection method. However, a vector containing
        the entire population has to be initialized.
        # 如果 method == "pool"，基于池的算法特别快，甚至比跟踪选择方法更快。但是，必须初始化一个包含整个总体的向量。
        If n_samples ~ n_population, the reservoir sampling method is faster.
        # 如果 n_samples 接近 n_population，reservoir sampling 方法更快速。

    Returns
    -------
    out : ndarray of shape (n_samples,)
    # 返回值：形状为 (n_samples,) 的 ndarray
        The sampled subsets of integer. The subset of selected integer might
        not be randomized, see the method argument.
        # 所选整数的子集。选定整数的子集可能不是随机化的，具体见 method 参数。

    Examples
    --------
    >>> from sklearn.utils.random import sample_without_replacement
    >>> sample_without_replacement(10, 5, random_state=42)
    array([8, 1, 5, 0, 7])
    """
    cdef:
        intp_t n_pop_intp, n_samples_intp
        long n_pop_long, n_samples_long
    # 使用 cdef 声明变量 n_pop_intp, n_samples_intp 为 intp_t 类型，n_pop_long, n_samples_long 为 long 类型

    # On most platforms `np.int_ is np.intp`.  However, before NumPy 2 the
    # default integer `np.int_` was a long which is 32bit on 64bit windows
    # while `intp` is 64bit on 64bit platforms and 32bit on 32bit ones.
    # 大多数平台上 `np.int_ is np.intp`。然而，在 NumPy 2 之前，默认整数 `np.int_` 是一个 long，在 64 位 Windows 上是 32 位，而在 64 位平台上是 64 位，在 32 位平台上是 32 位。

    if np.int_ is np.intp:
        # Branch always taken on NumPy >=2 (or when not on 64bit windows).
        # Cython has different rules for conversion of values to integers.
        # For NumPy <1.26.2 AND Cython 3, this first branch requires `int()`
        # called explicitly to allow e.g. floats.
        # 如果 np.int_ 等于 np.intp，则始终执行此分支（或者不在 64 位 Windows 上时）。Cython 对值转换为整数有不同的规则。对于 NumPy <1.26.2 和 Cython 3，这个分支需要显式调用 `int()` 来允许例如浮点数。

        n_pop_intp = int(n_population)
        n_samples_intp = int(n_samples)
        return _sample_without_replacement(
                n_pop_intp, n_samples_intp, method, random_state)
        # 调用 _sample_without_replacement 函数，传入 n_pop_intp, n_samples_intp, method, random_state 参数

    else:
        # Branch taken on 64bit windows with Numpy<2.0 where `long` is 32bit
        # 如果在 64 位 Windows 上且 NumPy <2.0，其中 `long` 是 32 位时执行此分支

        n_pop_long = n_population
        n_samples_long = n_samples
        return _sample_without_replacement(
                n_pop_long, n_samples_long, method, random_state)
        # 调用 _sample_without_replacement 函数，传入 n_pop_long, n_samples_long, method, random_state 参数
# 定义名为 _our_rand_r_py 的 Python 函数，用于测试 our_rand_r 函数
def _our_rand_r_py(seed):
    # 声明并初始化一个名为 my_seed 的 uint32_t 变量，其值为参数 seed
    cdef uint32_t my_seed = seed
    # 调用 C 函数 our_rand_r，传入 my_seed 的地址作为参数，并返回其结果
    return our_rand_r(&my_seed)
```