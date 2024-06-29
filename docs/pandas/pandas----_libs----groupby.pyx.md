# `D:\src\scipysrc\pandas\pandas\_libs\groupby.pyx`

```
# 使用 Cython 导入声明，允许在 Python 中使用 C 的数据类型和函数
cimport cython
from cython cimport (
    Py_ssize_t,  # 导入 C 中的 Py_ssize_t 类型，表示 Python 中的 ssize_t
    floating,    # 导入 C 中的 floating 类型，表示浮点数
)
from libc.math cimport (
    NAN,    # 导入 C 标准库中的 NAN 值
    sqrt,   # 导入 C 标准库中的平方根函数 sqrt
)
from libc.stdlib cimport (
    free,   # 导入 C 标准库中的 free 函数，用于释放内存
    malloc, # 导入 C 标准库中的 malloc 函数，用于动态分配内存
)

import numpy as np  # 导入 NumPy 库
cimport numpy as cnp  # 使用 Cython 导入 NumPy 库，以便在 Cython 中使用其功能
from numpy cimport (
    complex64_t,     # 导入 NumPy 中的复数类型 complex64_t
    complex128_t,    # 导入 NumPy 中的复数类型 complex128_t
    float32_t,       # 导入 NumPy 中的 32 位浮点数类型 float32_t
    float64_t,       # 导入 NumPy 中的 64 位浮点数类型 float64_t
    int8_t,          # 导入 NumPy 中的 8 位整数类型 int8_t
    int64_t,         # 导入 NumPy 中的 64 位整数类型 int64_t
    intp_t,          # 导入 NumPy 中的整数索引类型 intp_t
    ndarray,         # 导入 NumPy 中的 ndarray 类型
    uint8_t,         # 导入 NumPy 中的无符号 8 位整数类型 uint8_t
    uint64_t,        # 导入 NumPy 中的无符号 64 位整数类型 uint64_t
)

cnp.import_array()  # 调用 NumPy 的 import_array 函数，初始化 NumPy C API

from pandas._libs cimport util  # 使用 Cython 导入 pandas 库中的 util 模块
from pandas._libs.algos cimport (
    get_rank_nan_fill_val,   # 导入 pandas 库中的算法函数 get_rank_nan_fill_val
    kth_smallest_c,          # 导入 pandas 库中的算法函数 kth_smallest_c
)

from pandas._libs.algos import (
    groupsort_indexer,                # 导入 pandas 库中的排序索引函数 groupsort_indexer
    rank_1d,                         # 导入 pandas 库中的一维排名函数 rank_1d
    take_2d_axis1_bool_bool,         # 导入 pandas 库中的 take 函数，用于处理布尔型索引
    take_2d_axis1_float64_float64,   # 导入 pandas 库中的 take 函数，用于处理浮点型索引
)

from pandas._libs.dtypes cimport (
    numeric_object_t,   # 导入 pandas 库中的数据类型 numeric_object_t
    numeric_t,          # 导入 pandas 库中的数据类型 numeric_t
)
from pandas._libs.missing cimport checknull  # 使用 Cython 导入 pandas 库中的 missing 模块，引入 checknull 函数

# 定义一个 Cython 全局变量，存储 NumPy 中的 NAT 值
cdef int64_t NPY_NAT = util.get_nat()

# 定义一个 Cython 全局变量，存储 NaN 值的浮点数表示
cdef float64_t NaN = <float64_t>np.nan

# 定义一个枚举类型，表示线性插值的不同方式
cdef enum InterpolationEnumType:
    INTERPOLATION_LINEAR,   # 线性插值
    INTERPOLATION_LOWER,    # 较低插值
    INTERPOLATION_HIGHER,   # 较高插值
    INTERPOLATION_NEAREST,  # 最近邻插值
    INTERPOLATION_MIDPOINT   # 中点插值

# 定义一个内部函数，计算具有线性掩码的中位数
cdef float64_t median_linear_mask(float64_t* a, int n, uint8_t* mask) noexcept nogil:
    cdef:
        int i, j, na_count = 0
        float64_t* tmp
        float64_t result

    if n == 0:
        return NaN

    # 计算掩码中的缺失值数量
    for i in range(n):
        if mask[i]:
            na_count += 1

    # 如果存在缺失值
    if na_count:
        if na_count == n:
            return NaN

        # 分配内存以存储非缺失值
        tmp = <float64_t*>malloc((n - na_count) * sizeof(float64_t))
        if tmp is NULL:
            raise MemoryError()

        j = 0
        for i in range(n):
            if not mask[i]:
                tmp[j] = a[i]
                j += 1

        a = tmp
        n -= na_count

    # 计算线性插值后的中位数
    result = calc_median_linear(a, n)

    # 如果存在缺失值，释放临时数组
    if na_count:
        free(a)

    return result

# 定义一个内部函数，计算具有线性插值的中位数
cdef float64_t median_linear(
    float64_t* a,
    int n,
    bint is_datetimelike=False
) noexcept nogil:
    cdef:
        int i, j, na_count = 0
        float64_t* tmp
        float64_t result

    if n == 0:
        return NaN

    # 计算缺失值的数量
    if is_datetimelike:
        for i in range(n):
            if a[i] == NPY_NAT:
                na_count += 1
    else:
        for i in range(n):
            if a[i] != a[i]:
                na_count += 1

    # 如果存在缺失值
    if na_count:
        if na_count == n:
            return NaN

        # 分配内存以存储非缺失值
        tmp = <float64_t*>malloc((n - na_count) * sizeof(float64_t))
        if tmp is NULL:
            raise MemoryError()

        j = 0
        if is_datetimelike:
            for i in range(n):
                if a[i] != NPY_NAT:
                    tmp[j] = a[i]
                    j += 1
        else:
            for i in range(n):
                if a[i] == a[i]:
                    tmp[j] = a[i]
                    j += 1

        a = tmp
        n -= na_count

    # 计算线性插值后的中位数
    result = calc_median_linear(a, n)

    # 如果存在缺失值，释放临时数组
    if na_count:
        free(a)

    return result

# 定义一个内部函数，计算线性插值的中位数
cdef float64_t calc_median_linear(float64_t* a, int n) noexcept nogil:
    cdef:
        float64_t result

    # 计算中位数
    # 检查 n 是否为奇数，如果是奇数则执行以下代码块
    if n % 2:
        # 调用函数 kth_smallest_c 计算数组 a 中第 n//2 小的元素，并将结果赋给 result
        result = kth_smallest_c(a, n // 2, n)
    else:
        # 如果 n 是偶数，则执行以下代码块
        # 调用函数 kth_smallest_c 计算数组 a 中第 n//2 小的元素，并将结果保存在 result 中
        # 同时调用函数 kth_smallest_c 计算数组 a 中第 n//2 - 1 小的元素，
        # 将两者之和除以 2，并将结果赋给 result
        result = (kth_smallest_c(a, n // 2, n) +
                  kth_smallest_c(a, n // 2 - 1, n)) / 2

    # 返回计算得到的 result 结果
    return result
# 定义一个 C 语言风格的联合类型 `int64float_t`，可以是 int64_t、uint64_t、float32_t、float64_t 中的一种
ctypedef fused int64float_t:
    int64_t
    uint64_t
    float32_t
    float64_t

# 使用 Cython 的装饰器设置函数级优化，禁用边界检查和负数索引包装
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义函数 `group_median_float64`，用于计算浮点数数组的分组中位数
def group_median_float64(
    ndarray[float64_t, ndim=2] out,  # 输出的浮点数数组
    ndarray[int64_t] counts,  # 每个分组的元素计数数组
    ndarray[float64_t, ndim=2] values,  # 输入的浮点数值数组
    ndarray[intp_t] labels,  # 分组标签数组
    Py_ssize_t min_count=-1,  # 最小计数，仅在求和和乘积时使用，默认为 -1
    const uint8_t[:, :] mask=None,  # 可选的掩码数组
    uint8_t[:, ::1] result_mask=None,  # 可选的结果掩码数组
    bint is_datetimelike=False,  # 是否类似日期时间的数据
) -> None:  # 函数没有返回值

    """
    Only aggregates on axis=0
    """
    # 定义 C 语言风格的变量
    cdef:
        Py_ssize_t i, j, N, K, ngroups, size  # Python 整数变量和循环索引
        ndarray[intp_t] _counts  # 整数数组，存储分组后的计数
        ndarray[float64_t, ndim=2] data  # 双精度浮点数数组，存储中间数据
        ndarray[uint8_t, ndim=2] data_mask  # 无符号 8 位整数数组，存储掩码数据
        ndarray[intp_t] indexer  # 整数数组，存储排序后的索引
        float64_t* ptr  # 指向双精度浮点数的指针
        uint8_t* ptr_mask  # 指向无符号 8 位整数的指针
        float64_t result  # 存储中位数计算结果
        bint uses_mask = mask is not None  # 布尔值，指示是否使用掩码数组

    assert min_count == -1, "'min_count' only used in sum and prod"

    ngroups = len(counts)  # 获取分组数量
    N, K = (<object>values).shape  # 获取输入值数组的形状

    indexer, _counts = groupsort_indexer(labels, ngroups)  # 调用排序函数，获取索引和计数
    counts[:] = _counts[1:]  # 将计数值复制到 counts 数组中，排除第一个元素

    data = np.empty((K, N), dtype=np.float64)  # 创建 KxN 的空双精度浮点数数组
    ptr = <float64_t*>cnp.PyArray_DATA(data)  # 获取数据数组的指针

    take_2d_axis1_float64_float64(values.T, indexer, out=data)  # 调用函数填充数据数组

    if uses_mask:
        data_mask = np.empty((K, N), dtype=np.uint8)  # 创建 KxN 的空掩码数组
        ptr_mask = <uint8_t *>cnp.PyArray_DATA(data_mask)  # 获取掩码数组的指针

        take_2d_axis1_bool_bool(mask.T, indexer, out=data_mask, fill_value=1)  # 调用函数填充掩码数组

        with nogil:  # 进入无 GIL 状态的并行区域

            for i in range(K):
                # exclude NA group  # 排除 NA 组
                ptr += _counts[0]  # 移动数据指针到下一组数据起始位置
                ptr_mask += _counts[0]  # 移动掩码指针到下一组掩码起始位置

                for j in range(ngroups):  # 遍历所有分组
                    size = _counts[j + 1]  # 获取当前分组的大小
                    result = median_linear_mask(ptr, size, ptr_mask)  # 计算带有掩码的线性中位数
                    out[j, i] = result  # 将计算结果存储到输出数组中

                    if result != result:  # 检查是否为 NaN
                        result_mask[j, i] = 1  # 在结果掩码数组中标记为 1
                    ptr += size  # 移动数据指针到下一组数据起始位置
                    ptr_mask += size  # 移动掩码指针到下一组掩码起始位置

    else:
        with nogil:  # 进入无 GIL 状态的并行区域
            for i in range(K):
                # exclude NA group  # 排除 NA 组
                ptr += _counts[0]  # 移动数据指针到下一组数据起始位置

                for j in range(ngroups):  # 遍历所有分组
                    size = _counts[j + 1]  # 获取当前分组的大小
                    out[j, i] = median_linear(ptr, size, is_datetimelike)  # 计算线性中位数
                    ptr += size  # 移动数据指针到下一组数据起始位置


@cython.boundscheck(False)
@cython.wraparound(False)
# 定义函数 `group_cumprod`，计算值数组的累积乘积，按行分组
def group_cumprod(
    int64float_t[:, ::1] out,  # 输出的累积乘积数组
    ndarray[int64float_t, ndim=2] values,  # 输入的整数或浮点数值数组
    const intp_t[::1] labels,  # 分组标签数组
    int ngroups,  # 分组数量
    bint is_datetimelike,  # 是否类似日期时间的数据
    bint skipna=True,  # 是否跳过 NaN 值，默认为 True
    const uint8_t[:, :] mask=None,  # 可选的掩码数组
    uint8_t[:, ::1] result_mask=None,  # 可选的结果掩码数组
) -> None:  # 函数没有返回值

    """
    Cumulative product of columns of `values`, in row groups `labels`.
    """

    # 省略了对 min_count 的断言，因为不适用于此函数

    # 定义 C 语言风格的变量
    cdef:
        Py_ssize_t i, j, N, size  # Python 整数变量和循环索引
        ndarray[intp_t] _counts  # 整数数组，存储分组后的计数
        float64_t* ptr  # 指向双精度浮点数的指针
        uint8_t* ptr_mask  # 指向无符号 8 位整数的指针

    # 获取输入值数组的形状
    N, K = (<object>values).shape

    # 调用排序函数，获取索引和计数
    indexer, _counts = groupsort_indexer(labels, ngroups)

    # 创建 KxN 的空数组来存储输出
    out = np.empty((K, N), dtype=int64float_t)

    # 获取输出数组的指针
    ptr = <float64_t*>cnp.PyArray_DATA(out)

    # 进入无 GIL 状态的并行区域
    with nogil:

        for i in range(K):
            # exclude NA group  # 排除 NA 组
            ptr += _counts[0]  # 移动指针到下一组数据起始位置

            for j in range(ngroups):  # 遍历所有分组
                size = _counts[j + 1]  # 获取当前分组的大小
                out[j, i] = median_linear(ptr, size, is_datetimelike)  # 计算线性中位数
                ptr += size  #
    # 是否类似日期时间类型的标志，始终为假，因此`values`不会是日期时间类型。
    is_datetimelike : bool
        Always false, `values` is never datetime-like.
    # 是否跳过NaN值的标志，如果为真，则忽略`values`中的NaN值。
    skipna : bool
        If true, ignore nans in `values`.
    # 值的掩码，是一个np.ndarray数组，存储值的掩码信息。
    mask : np.ndarray[uint8], optional
        Mask of values
    # 结果数组的掩码，是一个np.ndarray数组，存储输出数组的掩码信息。
    result_mask : np.ndarray[int8], optional
        Mask of out array

    # 方法说明
    Notes
    -----
    This method modifies the `out` parameter, rather than returning an object.
    """
    # 使用Cython语法声明变量
    cdef:
        Py_ssize_t i, j, N, K
        int64float_t val, na_val
        int64float_t[:, ::1] accum
        intp_t lab
        uint8_t[:, ::1] accum_mask
        bint isna_entry, isna_prev = False
        bint uses_mask = mask is not None

    # 获取`values`的行数N和列数K
    N, K = (<object>values).shape
    # 初始化累积数组`accum`，全为1，与`values`相同的数据类型
    accum = np.ones((ngroups, K), dtype=(<object>values).dtype)
    # 获取NaN值，根据`is_datetimelike`来确定
    na_val = _get_na_val(<int64float_t>0, is_datetimelike)
    # 初始化累积掩码数组`accum_mask`，全为0，数据类型为uint8
    accum_mask = np.zeros((ngroups, K), dtype="uint8")

    # 使用GIL外部循环处理
    with nogil:
        # 外部循环遍历N次，N为`values`的行数
        for i in range(N):
            # 获取当前标签值`labels[i]`
            lab = labels[i]

            # 如果标签值小于0，跳过当前循环
            if lab < 0:
                continue
            # 内部循环遍历K次，K为`values`的列数
            for j in range(K):
                # 获取`values`中的值`values[i, j]`
                val = values[i, j]

                # 如果使用掩码，则获取当前位置的掩码值`mask[i, j]`
                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    # 否则调用函数判断是否为NaN值
                    isna_entry = _treat_as_na(val, False)

                # 如果当前值不是NaN
                if not isna_entry:
                    # 获取前一个累积掩码值`accum_mask[lab, j]`
                    isna_prev = accum_mask[lab, j]
                    # 如果前一个累积掩码值为真
                    if isna_prev:
                        # 将输出数组中的当前位置设为NaN值
                        out[i, j] = na_val
                        # 如果使用掩码，将结果数组掩码位置设为真
                        if uses_mask:
                            result_mask[i, j] = True

                    else:
                        # 否则累积乘积到累积数组中
                        accum[lab, j] *= val
                        # 将输出数组中的当前位置设为累积结果
                        out[i, j] = accum[lab, j]

                else:
                    # 如果当前值是NaN
                    if uses_mask:
                        # 如果使用掩码，将结果数组掩码位置设为真，输出数组当前位置设为0
                        result_mask[i, j] = True
                        out[i, j] = 0
                    else:
                        # 否则将输出数组当前位置设为NaN值
                        out[i, j] = na_val

                    # 如果不跳过NaN值，则将累积数组当前位置设为NaN值，累积掩码数组当前位置设为真
                    if not skipna:
                        accum[lab, j] = na_val
                        accum_mask[lab, j] = True
# 设置 Cython 的边界检查为 False，提高性能
@cython.boundscheck(False)
# 设置 Cython 的索引包装为 False，提高性能
@cython.wraparound(False)
def group_cumsum(
    # 输出累积和的二维数组，必须是 int64 或 float 类型
    int64float_t[:, ::1] out,
    # 需要计算累积和的值的二维数组
    ndarray[int64float_t, ndim=2] values,
    # 行分组的标签数组，必须是 intp 类型
    const intp_t[::1] labels,
    # 分组的数量，大于所有 labels 中的条目
    int ngroups,
    # 如果 values 包含类似日期时间的条目，则为 True
    bint is_datetimelike,
    # 如果为 True，在 values 中忽略 NaN 值
    bint skipna=True,
    # 可选的值的掩码数组，必须是 uint8 类型的二维数组
    const uint8_t[:, :] mask=None,
    # 可选的输出数组的掩码数组，必须是 int8 类型的二维数组
    uint8_t[:, ::1] result_mask=None,
) -> None:
    """
    对 `values` 中的列进行累积和，按 `labels` 中的行分组。

    Parameters
    ----------
    out : np.ndarray[ndim=2]
        用于存储累积和的数组。
    values : np.ndarray[ndim=2]
        需要进行累积和的值。
    labels : np.ndarray[np.intp]
        用于分组的标签。
    ngroups : int
        分组的数量，大于 `labels` 中的所有条目。
    is_datetimelike : bool
        如果 `values` 包含类似日期时间的条目，则为 True。
    skipna : bool, optional
        如果为 True，在 `values` 中忽略 NaN 值，默认为 True。
    mask : np.ndarray[uint8], optional
        值的掩码数组。
    result_mask : np.ndarray[int8], optional
        输出数组的掩码数组。

    Notes
    -----
    该方法修改 `out` 参数，而不是返回一个对象。
    """
    cdef:
        # 定义 C 的整型大小
        Py_ssize_t i, j, N, K
        # 定义累积值、临时值、掩码值、NA 值
        int64float_t val, y, t, na_val
        # 累积和数组、补偿数组、累积和的掩码数组
        int64float_t[:, ::1] accum, compensation
        uint8_t[:, ::1] accum_mask
        # 标签变量
        intp_t lab
        # 是否为 NaN 条目、上一个是否为 NaN 条目的标志
        bint isna_entry, isna_prev = False
        # 是否使用掩码的标志
        bint uses_mask = mask is not None

    # 获取 values 的形状信息
    N, K = (<object>values).shape

    # 如果使用掩码，则创建全零的累积和掩码数组
    if uses_mask:
        accum_mask = np.zeros((ngroups, K), dtype="uint8")

    # 创建全零的累积和数组和补偿数组
    accum = np.zeros((ngroups, K), dtype=np.asarray(values).dtype)
    compensation = np.zeros((ngroups, K), dtype=np.asarray(values).dtype)

    # 获取 NA 值，根据 is_datetimelike 的值
    na_val = _get_na_val(<int64float_t>0, is_datetimelike)
    # 使用 nogil 上下文以避免全局解释器锁(GIL)的影响，提高并行执行效率
    with nogil:
        # 遍历范围为 N 的索引
        for i in range(N):
            # 获取当前索引 i 处的标签值
            lab = labels[i]

            # 如果需要使用掩码并且标签值 lab 小于 0
            if uses_mask and lab < 0:
                # 设置 result_mask 的当前行全为 True
                result_mask[i, :] = True
                # 将 out 的当前行全部置为 0
                out[i, :] = 0
                # 继续下一个迭代
                continue
            # 如果标签值 lab 小于 0，则直接跳过当前迭代
            elif lab < 0:
                continue

            # 遍历范围为 K 的索引
            for j in range(K):
                # 获取 values 数组中 (i, j) 处的值
                val = values[i, j]

                # 如果需要使用掩码，则检查当前位置的掩码值
                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    # 否则，根据值和 is_datetimelike 判断是否视为空值
                    isna_entry = _treat_as_na(val, is_datetimelike)

                # 如果不跳过 NaN 值处理
                if not skipna:
                    # 如果需要使用掩码，则检查累积掩码中 (lab, j) 处的值
                    if uses_mask:
                        isna_prev = accum_mask[lab, j]
                    else:
                        # 否则，根据累积数组中 (lab, j) 处的值和 is_datetimelike 判断是否视为空值
                        isna_prev = _treat_as_na(accum[lab, j], is_datetimelike)

                    # 如果累积数组中先前的值为 NaN
                    if isna_prev:
                        # 如果使用掩码，则设置 result_mask 的当前位置为 True
                        result_mask[i, j] = True
                        # 设置 out 的当前位置为 0，保持确定性
                        out[i, j] = 0
                        # 继续下一个迭代
                        continue

                # 如果当前位置的值被视为空值
                if isna_entry:
                    # 如果使用掩码，则设置 result_mask 的当前位置为 True
                    result_mask[i, j] = True
                    # 设置 out 的当前位置为 0，保持确定性
                    out[i, j] = 0

                    # 如果不跳过 NaN 值处理
                    if not skipna:
                        # 如果使用掩码，则将 accum_mask 的当前位置设为 True
                        if uses_mask:
                            accum_mask[lab, j] = True
                        else:
                            # 否则，将 accum 的当前位置设为 NaN 值
                            accum[lab, j] = na_val

                else:
                    # 对于浮点数值，使用 Kahan 累加算法以减少浮点数误差
                    # （参考 https://en.wikipedia.org/wiki/Kahan_summation_algorithm）
                    if int64float_t == float32_t or int64float_t == float64_t:
                        y = val - compensation[lab, j]
                        t = accum[lab, j] + y
                        compensation[lab, j] = t - accum[lab, j] - y
                    else:
                        t = val + accum[lab, j]

                    # 更新累积数组的当前位置为 t
                    accum[lab, j] = t
                    # 更新 out 的当前位置为 t
                    out[i, j] = t
# 设置 Cython 参数：禁用数组边界检查
@cython.boundscheck(False)
# 设置 Cython 参数：禁用负数索引的检查
@cython.wraparound(False)
# 定义函数：将每个组的索引值进行偏移处理
def group_shift_indexer(
    int64_t[::1] out,                # 输出数组，存储结果索引
    const intp_t[::1] labels,        # 标签数组，每个元素对应一个组标识
    int ngroups,                     # 组的数量
    int periods,                     # 周期数
) -> None:
    cdef:
        Py_ssize_t N, i, ii, lab     # 定义 C 语言风格的变量声明
        int offset = 0, sign         # 偏移量和符号变量
        int64_t idxer, idxer_slot    # 索引器变量
        int64_t[::1] label_seen = np.zeros(ngroups, dtype=np.int64)  # 每个标签已见的计数数组
        int64_t[:, ::1] label_indexer  # 标签索引器数组

    N, = (<object>labels).shape  # 获取标签数组的长度 N

    if periods < 0:
        periods = -periods          # 如果周期为负数，取其相反数
        offset = N - 1              # 偏移量设为 N-1
        sign = -1                   # 符号为负数
    elif periods > 0:
        offset = 0                  # 否则偏移量为 0
        sign = 1                    # 符号为正数

    if periods == 0:
        with nogil:
            for i in range(N):
                out[i] = i         # 如果周期为 0，则直接将索引写入输出数组
    else:
        # 创建每个标签前期索引器的数组
        label_indexer = np.zeros((ngroups, periods), dtype=np.int64)
        with nogil:
            for i in range(N):
                # 如果向后移动，则使用反向迭代器
                ii = offset + sign * i
                lab = labels[ii]     # 获取当前标签值

                # 跳过空键
                if lab == -1:
                    out[ii] = -1    # 如果标签为 -1，则将输出索引设为 -1
                    continue

                label_seen[lab] += 1  # 增加该标签已见计数

                idxer_slot = label_seen[lab] % periods  # 计算索引槽位
                idxer = label_indexer[lab, idxer_slot]  # 获取索引器值

                if label_seen[lab] > periods:
                    out[ii] = idxer  # 如果已见计数大于周期数，则输出索引器值
                else:
                    out[ii] = -1     # 否则输出 -1

                label_indexer[lab, idxer_slot] = ii  # 更新标签索引器数组

@cython.wraparound(False)
@cython.boundscheck(False)
# 定义函数：确定组内值填充的索引
def group_fillna_indexer(
    Py_ssize_t[::1] out,            # 输出数组，存储填充的索引
    const intp_t[::1] labels,       # 标签数组，每个元素对应一个组标识
    const uint8_t[:] mask,          # 掩码数组，指示值是否为缺失值
    int64_t limit,                  # 填充限制的最大次数
    bint compute_ffill,             # 是否进行向前填充
    int ngroups,                    # 组的数量
) -> None:
    """
    索引如何在组内进行前向或后向填充值。

    参数
    ----------
    out : np.ndarray[np.intp]
        存储方法将写入其结果的值。
    labels : np.ndarray[np.intp]
        包含每个组的唯一标签的数组，其顺序与 `values` 中的对应记录匹配。
    mask : np.ndarray[np.uint8]
        指示值是否为 na 的数组。
    limit : int64_t
        在停止之前填充连续值的最大次数，或 -1 表示无限制。
    compute_ffill : bint
        是否计算前向填充或后向填充。
    ngroups : int
        大于 `labels` 的所有条目的组数。

    注意
    -----
    此方法修改 `out` 参数，而不是返回对象。
    """
    cdef:
        Py_ssize_t idx, N = len(out)  # 获取输出数组的长度 N
        intp_t label                  # 标签值
        intp_t[::1] last = -1 * np.ones(ngroups, dtype=np.intp)  # 每个组的上一个索引
        intp_t[::1] fill_count = np.zeros(ngroups, dtype=np.intp)  # 每个组的填充计数

    # 确保所有数组大小相同
    assert N == len(labels) == len(mask)
    # 使用 `nogil` 上下文，避免 GIL（全局解释器锁）限制
    with nogil:
        # 当不支持使用步长为正负号的 for 循环时，选择初始索引
        # 参考：https://github.com/cython/cython/issues/1106
        idx = 0 if compute_ffill else N-1
        # 遍历范围为 N 的循环
        for _ in range(N):
            # 获取当前索引位置的标签值
            label = labels[idx]
            # 如果标签值为 -1，表示 na-group 得到 na-values
            if label == -1:
                out[idx] = -1
            # 如果 mask 值为 1，表示当前位置缺失
            elif mask[idx] == 1:
                # 当填充次数已达限制时，停止填充
                if limit != -1 and fill_count[label] >= limit:
                    out[idx] = -1
                else:
                    out[idx] = last[label]  # 使用上一次的填充值填充当前位置
                    fill_count[label] += 1  # 填充计数加一
            else:
                fill_count[label] = 0  # 当位置不缺失时，重置填充计数
                last[label] = idx  # 更新 last 数组中的索引位置
                out[idx] = idx  # 当前位置的输出为其自身索引值

            # 根据 compute_ffill 决定是增加还是减少索引
            if compute_ffill:
                idx += 1
            else:
                idx -= 1
# 设置 Cython 的边界检查为 False，提升性能
@cython.boundscheck(False)
# 设置 Cython 的数组访问边界检查为 False，提升性能
@cython.wraparound(False)
# 定义函数 group_any_all，接受以下参数和返回类型
def group_any_all(
    # 输出结果的二维数组，数据类型为 int8
    int8_t[:, ::1] out,
    # 元素的真值数组，数据类型为 int8
    const int8_t[:, :] values,
    # 每个分组的唯一标签数组，数据类型为 intp
    const intp_t[::1] labels,
    # 表示值是否为 NA 的掩码数组，数据类型为 uint8
    const uint8_t[:, :] mask,
    # 值测试类型，可选值为 'any' 或 'all'，数据类型为 str
    str val_test,
    # 是否跳过 NA 值的标志，数据类型为 bool
    bint skipna,
    # 结果掩码数组，数据类型为 uint8 的二维数组，可选参数
    uint8_t[:, ::1] result_mask,
) -> None:
    """
    Aggregated boolean values to show truthfulness of group elements. If the
    input is a nullable type (result_mask is not None), the result will be computed
    using Kleene logic.

    Parameters
    ----------
    out : np.ndarray[np.int8]
        Values into which this method will write its results.
    labels : np.ndarray[np.intp]
        Array containing unique label for each group, with its
        ordering matching up to the corresponding record in `values`
    values : np.ndarray[np.int8]
        Containing the truth value of each element.
    mask : np.ndarray[np.uint8]
        Indicating whether a value is na or not.
    val_test : {'any', 'all'}
        String object dictating whether to use any or all truth testing
    skipna : bool
        Flag to ignore nan values during truth testing
    result_mask : ndarray[bool, ndim=2], optional
        If not None, these specify locations in the output that are NA.
        Modified in-place.

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object.
    The returned values will either be 0, 1 (False or True, respectively), or
    -1 to signify a masked position in the case of a nullable input.
    """
    # 定义循环变量和常量，N 为 labels 的长度，K 为 out 的第二维度长度
    cdef:
        Py_ssize_t i, j, N = len(labels), K = out.shape[1]
        intp_t lab
        int8_t flag_val, val
        bint uses_mask = result_mask is not None

    # 根据 val_test 的取值初始化 flag_val
    if val_test == "all":
        # 如果 val_test 为 'all'，则 flag_val 初始化为 0
        # 因为 Python 中空可迭代对象的 'all' 值为 True，故全部设为 1，遇到 False 设置为 0
        flag_val = 0
    elif val_test == "any":
        # 如果 val_test 为 'any'，则 flag_val 初始化为 1
        # 因为 Python 中空可迭代对象的 'any' 值为 False，故全部设为 0，遇到 True 设置为 1
        flag_val = 1
    else:
        # 如果 val_test 不是 'any' 或 'all'，则抛出 ValueError 异常
        raise ValueError("'val_test' must be either 'any' or 'all'!")

    # 将 out 数组全部初始化为 1 或 0，取决于 flag_val 的值
    out[:] = 1 - flag_val
    # 使用 `nogil` 上下文以释放全局解锁，提高性能和并发性
    with nogil:
        # 遍历标签列表，处理每个标签的操作
        for i in range(N):
            # 获取当前标签值
            lab = labels[i]
            # 若标签值为负数，则跳过当前循环
            if lab < 0:
                continue

            # 遍历每个类别，处理每个类别下的操作
            for j in range(K):
                # 如果 `skipna` 为真且当前位置被标记为跳过，则继续下一个循环
                if skipna and mask[i, j]:
                    continue

                # 如果使用了掩码并且当前位置被掩码标记，则执行以下操作
                if uses_mask and mask[i, j]:
                    # 如果 `out[lab, j]` 不等于 `flag_val`，表示当前位置的结果尚未确定
                    # 根据克里因逻辑将结果位置标记为掩码
                    if out[lab, j] != flag_val:
                        result_mask[lab, j] = 1
                    continue

                # 获取当前值
                val = values[i, j]

                # 如果当前值等于 `flag_val`，表示结果已经确定
                if val == flag_val:
                    # 将结果设置为 `flag_val`
                    out[lab, j] = flag_val
                    # 如果使用了掩码，则将结果掩码位置标记为非掩码
                    if uses_mask:
                        result_mask[lab, j] = 0
# ----------------------------------------------------------------------
# group_sum, group_prod, group_var, group_mean, group_ohlc
# ----------------------------------------------------------------------

# 定义一个融合类型 mean_t，可以是 float64_t、float32_t、complex64_t、complex128_t 中的一种
ctypedef fused mean_t:
    float64_t
    float32_t
    complex64_t
    complex128_t

# 定义一个融合类型 sum_t，可以是 mean_t、int64_t、uint64_t、object 中的一种
ctypedef fused sum_t:
    mean_t
    int64_t
    uint64_t
    object

# 使用 Cython 的装饰器指定不进行负索引和边界检查
@cython.wraparound(False)
@cython.boundscheck(False)
def group_sum(
    # out 是一个二维数组，其中元素类型为 sum_t，每行对应一个聚合结果
    sum_t[:, ::1] out,
    # counts 是一个一维整数数组，记录每个聚合组中的数据点数
    int64_t[::1] counts,
    # values 是一个二维数组，包含待聚合的数据
    ndarray[sum_t, ndim=2] values,
    # labels 是一个一维整数数组，标识每个数据点所属的聚合组
    const intp_t[::1] labels,
    # mask 是一个二维布尔数组，指示哪些数据点参与聚合计算
    const uint8_t[:, :] mask,
    # result_mask 是一个可选的二维布尔数组，标识聚合结果是否有效
    uint8_t[:, ::1] result_mask=None,
    # min_count 是一个整数，指定参与聚合计算的最小数据点数
    Py_ssize_t min_count=0,
    # is_datetimelike 是一个布尔值，指示数据是否类似日期时间
    bint is_datetimelike=False,
) -> None:
    """
    Only aggregates on axis=0 using Kahan summation
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        sum_t val, t, y
        sum_t[:, ::1] sumx, compensation
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)
        bint uses_mask = mask is not None
        bint isna_entry

    # 如果 values 和 labels 的长度不一致，抛出 ValueError 异常
    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    # 初始化 nobs 数组，与 out 形状相同，用于记录每个聚合组的数据点数
    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    # 初始化 sumx 数组，与 out 形状相同，用于保存聚合结果
    sumx = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)
    # 初始化 compensation 数组，与 out 形状相同，用于 Kahan 累加算法的补偿项
    compensation = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)

    # 获取 values 的行数 N 和列数 K
    N, K = (<object>values).shape
    # 使用 nogil 关键字来声明一个无 GIL（全局解释器锁）的上下文
    with nogil(sum_t is not object):
        # 循环遍历标签数组的每个元素
        for i in range(N):
            # 获取当前索引处的标签
            lab = labels[i]
            # 如果标签小于 0，则跳过当前循环
            if lab < 0:
                continue

            # 增加对应标签的计数器
            counts[lab] += 1

            # 循环遍历值数组的每个元素
            for j in range(K):
                # 获取当前索引处的值
                val = values[i, j]

                # 根据 uses_mask 决定是否使用掩码
                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    # 判断当前值是否应被视为缺失值
                    isna_entry = _treat_as_na(val, is_datetimelike)

                # 如果不是缺失值
                if not isna_entry:
                    # 增加对应标签和列索引的观测值计数
                    nobs[lab, j] += 1

                    # 如果 sum_t 是对象类型
                    if sum_t is object:
                        # 注意：此处不像非对象类型那样使用“补偿”
                        # 如果当前观测计数为 1，则直接将当前值作为 t
                        if nobs[lab, j] == 1:
                            t = val
                        else:
                            # 否则将当前 sumx 值与当前值相加作为 t
                            t = sumx[lab, j] + val
                        sumx[lab, j] = t

                    else:
                        # 如果 sum_t 不是对象类型
                        y = val - compensation[lab, j]
                        t = sumx[lab, j] + y
                        # 更新补偿值
                        compensation[lab, j] = t - sumx[lab, j] - y
                        # 如果补偿值为 NaN，则设置为 0，以避免结果为 NaN
                        if compensation[lab, j] != compensation[lab, j]:
                            compensation[lab, j] = 0
                        sumx[lab, j] = t

    # 检查并调整结果数组中小于最小计数的值
    _check_below_mincount(
        out, uses_mask, result_mask, ncounts, K, nobs, min_count, sumx
    )
@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个函数group_prod，接受多个参数并且没有返回值
def group_prod(
    # 定义一个输出参数out，类型为int64float_t的二维数组，不进行包装
    int64float_t[:, ::1] out,
    # 定义一个计数数组counts，类型为int64_t的一维数组
    int64_t[::1] counts,
    # 定义一个数值数组values，类型为ndarray[int64float_t]的二维数组
    ndarray[int64float_t, ndim=2] values,
    # 定义一个标签数组labels，类型为const intp_t的一维数组
    const intp_t[::1] labels,
    # 定义一个掩码数组mask，类型为const uint8_t的二维数组
    const uint8_t[:, ::1] mask,
    # 定义一个结果掩码数组result_mask，默认值为None
    uint8_t[:, ::1] result_mask=None,
    # 定义一个最小计数min_count，默认值为0
    Py_ssize_t min_count=0,
) -> None:
    """
    Only aggregates on axis=0
    """
    # 定义多个Cython变量
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        int64float_t val
        int64float_t[:, ::1] prodx
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)
        bint isna_entry, uses_mask = mask is not None

    # 如果values和labels的长度不同，抛出值错误异常
    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    # 初始化nobs为全零的int64_t类型数组
    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    # 初始化prodx为全一的int64float_t类型数组
    prodx = np.ones((<object>out).shape, dtype=(<object>out).base.dtype)

    # 获取values的形状(N, K)
    N, K = (<object>values).shape

    # 使用nogil语句块进行无GIL的并行操作
    with nogil:
        # 循环遍历N次
        for i in range(N):
            # 获取当前标签lab
            lab = labels[i]
            # 如果lab小于0，跳过当前循环
            if lab < 0:
                continue

            # 增加对应标签lab的计数
            counts[lab] += 1
            # 循环遍历K次
            for j in range(K):
                # 获取values数组中的值val
                val = values[i, j]

                # 如果使用掩码，则获取当前位置的掩码值
                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    # 否则调用_treat_as_na函数判断是否为NA值
                    isna_entry = _treat_as_na(val, False)

                # 如果不是NA值
                if not isna_entry:
                    # 增加对应标签lab和列j的观测数目计数
                    nobs[lab, j] += 1
                    # 计算对应标签lab和列j的乘积
                    prodx[lab, j] *= val

    # 调用_check_below_mincount函数，检查并处理最小计数以下的情况
    _check_below_mincount(
        out, uses_mask, result_mask, ncounts, K, nobs, min_count, prodx
    )


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
# 定义一个函数group_var，接受多个参数并且没有返回值
def group_var(
    # 定义一个输出参数out，类型为floating的二维数组，不进行包装
    floating[:, ::1] out,
    # 定义一个计数数组counts，类型为int64_t的一维数组
    int64_t[::1] counts,
    # 定义一个数值数组values，类型为ndarray[floating]的二维数组
    ndarray[floating, ndim=2] values,
    # 定义一个标签数组labels，类型为const intp_t的一维数组
    const intp_t[::1] labels,
    # 定义一个最小计数min_count，默认值为-1
    Py_ssize_t min_count=-1,
    # 定义一个自由度修正项ddof，默认值为1
    int64_t ddof=1,
    # 定义一个掩码数组mask，默认值为None
    const uint8_t[:, ::1] mask=None,
    # 定义一个结果掩码数组result_mask，默认值为None
    uint8_t[:, ::1] result_mask=None,
    # 定义一个布尔型变量is_datetimelike，默认值为False
    bint is_datetimelike=False,
    # 定义一个字符串变量name，默认值为"var"
    str name="var",
) -> None:
    # 定义多个Cython变量
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        floating val, ct, oldmean
        floating[:, ::1] mean
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)
        bint isna_entry, uses_mask = mask is not None
        bint is_std = name == "std"
        bint is_sem = name == "sem"

    # 如果min_count不等于-1，断言失败并输出相关信息
    assert min_count == -1, "'min_count' only used in sum and prod"

    # 如果values和labels的长度不同，抛出值错误异常
    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    # 初始化nobs为全零的int64_t类型数组
    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    # 初始化mean为全零的floating类型数组
    mean = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)

    # 获取values的形状(N, K)
    N, K = (<object>values).shape

    # 将输出数组out的所有元素设置为0.0
    out[:, :] = 0.0
    # 使用 nogil 上下文，可能表示这段代码需要在没有全局解释器锁的情况下执行（针对并行处理优化）
    with nogil:
        # 遍历标签数组中的每一个索引 i
        for i in range(N):
            # 获取当前索引处的标签值
            lab = labels[i]
            # 如果标签值小于 0，则跳过当前循环，继续下一个
            if lab < 0:
                continue

            # 增加对应标签 lab 的计数值
            counts[lab] += 1

            # 遍历每个标签 lab 下的 K 个值
            for j in range(K):
                # 获取 values 数组中 (i, j) 处的值
                val = values[i, j]

                # 如果使用掩码，则检查当前位置是否为缺失值
                if uses_mask:
                    isna_entry = mask[i, j]
                # 如果是日期时间型数据，处理缺失值判断
                elif is_datetimelike:
                    # 对于 group_var，不能简单地使用 _treat_as_na，
                    # 因为日期时间类型会转换为 float64 而不是 int64。
                    isna_entry = val == NPY_NAT
                else:
                    # 使用 _treat_as_na 函数判断当前值是否为缺失值
                    isna_entry = _treat_as_na(val, is_datetimelike)

                # 如果不是缺失值
                if not isna_entry:
                    # 增加对应标签 lab 和列 j 的观测计数
                    nobs[lab, j] += 1
                    # 获取旧的均值
                    oldmean = mean[lab, j]
                    # 更新均值和离差平方和
                    mean[lab, j] += (val - oldmean) / nobs[lab, j]
                    out[lab, j] += (val - mean[lab, j]) * (val - oldmean)

        # 遍历每个计数值数组的行索引 i
        for i in range(ncounts):
            # 遍历每个计数值数组的列索引 j
            for j in range(K):
                # 获取 nobs 数组中的观测数
                ct = nobs[i, j]
                # 如果观测数小于等于自由度调整因子 ddof
                if ct <= ddof:
                    # 如果使用掩码，则将结果掩码数组中对应位置置为 True
                    if uses_mask:
                        result_mask[i, j] = True
                    # 否则将 out 数组中对应位置置为 NaN
                    else:
                        out[i, j] = NAN
                else:
                    # 如果计算标准差
                    if is_std:
                        # 计算标准差并赋值给 out 数组
                        out[i, j] = sqrt(out[i, j] / (ct - ddof))
                    # 如果计算标准误差
                    elif is_sem:
                        # 计算标准误差并赋值给 out 数组
                        out[i, j] = sqrt(out[i, j] / (ct - ddof) / ct)
                    else:
                        # 否则，执行方差的计算
                        out[i, j] /= (ct - ddof)
@cython.wraparound(False)
@cython.boundscheck(False)
# 设置 Cython 选项，禁用边界检查和负数索引包装
@cython.cdivision(True)
@cython.cpow
# 启用 Cython 的 C 除法和幂运算优化
def group_skew(
    float64_t[:, ::1] out,  # 输出数组，用于存储计算结果
    int64_t[::1] counts,  # 记录每个组中的观测数
    ndarray[float64_t, ndim=2] values,  # 输入数据的二维数组
    const intp_t[::1] labels,  # 每个数据点对应的分组标签
    const uint8_t[:, ::1] mask=None,  # 可选的遮罩数组
    uint8_t[:, ::1] result_mask=None,  # 可选的结果遮罩数组
    bint skipna=True,  # 是否跳过 NA 值
) -> None:
    cdef:
        Py_ssize_t i, j, N, K, lab, ngroups = len(counts)
        int64_t[:, ::1] nobs  # 记录每个组中每个特征的观测数
        Py_ssize_t len_values = len(values), len_labels = len(labels)
        bint isna_entry, uses_mask = mask is not None  # 是否使用遮罩数组的标志
        float64_t[:, ::1] M1, M2, M3  # 记录组内每个特征的一、二、三阶矩
        float64_t delta, delta_n, term1, val  # 用于计算偏差、增量、项和当前值
        int64_t n1, n  # 用于计算观测数和计数

    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    # 初始化 nobs 数组，用于记录每个组中每个特征的观测数

    # 初始化 M1, M2, M3 数组，分别用于记录一、二、三阶矩
    M1 = np.zeros((<object>out).shape, dtype=np.float64)
    M2 = np.zeros((<object>out).shape, dtype=np.float64)
    M3 = np.zeros((<object>out).shape, dtype=np.float64)

    N, K = (<object>values).shape
    # 获取输入数据的行数 N 和列数 K

    out[:, :] = 0.0
    # 初始化输出数组为 0

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue
            # 如果标签 lab 小于 0，则跳过当前循环

            counts[lab] += 1
            # 对应组的观测数加一

            for j in range(K):
                val = values[i, j]
                # 获取当前值

                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    isna_entry = _treat_as_na(val, False)
                # 根据是否使用遮罩数组，确定当前值是否为 NA

                if not isna_entry:
                    # 如果当前值不是 NA

                    # 根据 RunningStats::Push 算法更新一、二、三阶矩
                    n1 = nobs[lab, j]
                    n = n1 + 1

                    nobs[lab, j] = n
                    # 更新当前组当前特征的观测数

                    delta = val - M1[lab, j]
                    delta_n = delta / n
                    term1 = delta * delta_n * n1

                    M1[lab, j] += delta_n
                    M3[lab, j] += term1 * delta_n * (n - 2) - 3 * delta_n * M2[lab, j]
                    M2[lab, j] += term1
                    # 更新一、二、三阶矩

                elif not skipna:
                    M1[lab, j] = NaN
                    M2[lab, j] = NaN
                    M3[lab, j] = NaN
                    # 如果跳过 NA 值被禁用，则设置对应矩为 NaN

        for i in range(ngroups):
            for j in range(K):
                ct = <float64_t>nobs[i, j]
                if ct < 3:
                    if result_mask is not None:
                        result_mask[i, j] = 1
                    out[i, j] = NaN
                    # 如果观测数小于 3，则输出为 NaN
                elif M2[i, j] == 0:
                    out[i, j] = 0
                    # 如果 M2 等于 0，则输出为 0
                else:
                    out[i, j] = (
                        (ct * (ct - 1) ** 0.5 / (ct - 2))
                        * (M3[i, j] / M2[i, j] ** 1.5)
                    )
                    # 否则，根据公式计算偏斜度
    const uint8_t[:, ::1] mask=None,
    # 定义一个名为 mask 的常量，类型为 uint8_t 的二维数组，维度顺序为 ::1，初始值为 None

    uint8_t[:, ::1] result_mask=None,
    # 定义一个名为 result_mask 的变量，类型为 uint8_t 的二维数组，维度顺序为 ::1，初始值为 None
    """
    Compute the mean per label given a label assignment for each value.
    NaN values are ignored.

    Parameters
    ----------
    out : np.ndarray[floating]
        Values into which this method will write its results.
    counts : np.ndarray[int64]
        A zeroed array of the same shape as labels,
        populated by group sizes during algorithm.
    values : np.ndarray[floating]
        2-d array of the values to find the mean of.
    labels : np.ndarray[np.intp]
        Array containing unique label for each group, with its
        ordering matching up to the corresponding record in `values`.
    min_count : Py_ssize_t
        Only used in sum and prod. Always -1.
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    mask : ndarray[bool, ndim=2], optional
        Mask of the input values.
    result_mask : ndarray[bool, ndim=2], optional
        Mask of the out array

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object.
    `counts` is modified to hold group sizes
    """

    # Declare variables and types using Cython's cdef
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        mean_t val, count, y, t, nan_val
        mean_t[:, ::1] sumx, compensation
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)
        bint isna_entry, uses_mask = mask is not None

    # Check if min_count is as expected
    assert min_count == -1, "'min_count' only used in sum and prod"

    # Validate lengths of values and labels arrays
    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    # Initialize arrays for computations, optimized for performance
    # Equivalent to np.zeros_like(out) but faster in Cython
    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    sumx = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)
    compensation = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)

    # Determine NaN value based on input data characteristics
    N, K = (<object>values).shape
    if uses_mask:
        nan_val = 0
    elif is_datetimelike:
        nan_val = NPY_NAT
    else:
        nan_val = NAN
    # 使用 nogil 上下文，表示这段代码需要在没有全局解释器锁 (GIL) 的情况下执行，通常用于提高多线程并行代码的性能
    with nogil:
        # 遍历范围为 N 的循环，处理每个索引 i
        for i in range(N):
            # 获取标签数组中索引为 i 的标签值
            lab = labels[i]
            # 如果标签值小于 0，则跳过当前循环
            if lab < 0:
                continue

            # 对应标签值的计数器加一
            counts[lab] += 1

            # 遍历范围为 K 的循环，处理索引为 i 的每个值
            for j in range(K):
                # 获取 values 数组中索引为 (i, j) 的值
                val = values[i, j]

                # 如果使用了掩码 (mask)
                if uses_mask:
                    # 获取掩码数组中索引为 (i, j) 的值
                    isna_entry = mask[i, j]
                elif is_datetimelike:
                    # 处理日期时间型数据的情况下，判断是否为 NA 值
                    # 对于 group_mean，不能简单地使用 _treat_as_na，因为
                    # 日期时间型数据会转换为 float64 而不是 int64
                    isna_entry = val == NPY_NAT
                else:
                    # 判断是否为 NA 值，根据数据类型和 is_datetimelike 参数调用 _treat_as_na 函数
                    isna_entry = _treat_as_na(val, is_datetimelike)

                # 如果不是 NA 值
                if not isna_entry:
                    # 对应标签和索引的观测值计数加一
                    nobs[lab, j] += 1

                    # 计算修正后的值 y，更新 sumx[lab, j] 的值
                    y = val - compensation[lab, j]
                    t = sumx[lab, j] + y
                    compensation[lab, j] = t - sumx[lab, j] - y

                    # 如果修正值为 NaN，则设置为 0，解决 NaN 问题
                    if compensation[lab, j] != compensation[lab, j]:
                        # GH#50367
                        # 如果 val 是正负无穷，修正值为 NaN，结果可能变为 NaN 而不是正负无穷
                        # 不能使用 util.is_nan，因为没有 GIL
                        compensation[lab, j] = 0.

                    # 更新 sumx[lab, j] 的值为 t
                    sumx[lab, j] = t

        # 遍历范围为 ncounts 的循环，处理每个索引 i
        for i in range(ncounts):
            # 遍历范围为 K 的循环，处理索引为 i 的每个值
            for j in range(K):
                # 获取 nobs 数组中索引为 (i, j) 的计数值
                count = nobs[i, j]

                # 如果计数值为 0
                if nobs[i, j] == 0:
                    # 如果使用了掩码，则将 result_mask[i, j] 设置为 True
                    if uses_mask:
                        result_mask[i, j] = True
                    else:
                        # 否则，将 out[i, j] 设置为 NaN 值
                        out[i, j] = nan_val

                else:
                    # 计算平均值，并将结果存储到 out[i, j] 中
                    out[i, j] = sumx[i, j] / count
@cython.wraparound(False)
@cython.boundscheck(False)
# 定义函数 `group_ohlc`，用于按照指定标签进行 OHLC 聚合操作
def group_ohlc(
    int64float_t[:, ::1] out,  # 输出数组，存储聚合结果
    int64_t[::1] counts,  # 记录每个标签出现次数的数组
    ndarray[int64float_t, ndim=2] values,  # 输入的值数组，二维数组
    const intp_t[::1] labels,  # 标签数组，用于分组
    Py_ssize_t min_count=-1,  # 最小计数，默认为 -1，只在 sum 和 prod 中使用
    const uint8_t[:, ::1] mask=None,  # 可选的掩码数组
    uint8_t[:, ::1] result_mask=None,  # 可选的结果掩码数组
) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, N, K, lab  # 定义循环变量和标签变量
        int64float_t val  # 定义值变量
        uint8_t[::1] first_element_set  # 标记每个标签是否已经设置过初始值
        bint isna_entry, uses_mask = mask is not None  # 定义是否为 NA 值的标记和是否使用掩码的标记

    assert min_count == -1, "'min_count' only used in sum and prod"

    if len(labels) == 0:
        return  # 如果标签数组为空，直接返回

    N, K = (<object>values).shape  # 获取输入值数组的形状信息

    if out.shape[1] != 4:
        raise ValueError("Output array must have 4 columns")  # 输出数组必须有 4 列，否则引发值错误异常

    if K > 1:
        raise NotImplementedError("Argument 'values' must have only one dimension")  # 如果输入值数组有多于一维，引发未实现错误异常

    if int64float_t is float32_t or int64float_t is float64_t:
        out[:] = NAN  # 如果数据类型是浮点数，用 NaN 填充输出数组
    else:
        out[:] = 0  # 否则用 0 填充输出数组

    first_element_set = np.zeros((<object>counts).shape, dtype=np.uint8)  # 初始化一个用于标记是否已设置初始值的数组

    if uses_mask:
        result_mask[:] = True  # 如果使用掩码，初始化结果掩码数组为 True

    with nogil:
        for i in range(N):
            lab = labels[i]  # 获取当前循环的标签值
            if lab == -1:
                continue  # 如果标签值为 -1，跳过当前循环

            counts[lab] += 1  # 对应标签的计数加一
            val = values[i, 0]  # 获取输入值数组中当前行第一列的值

            if uses_mask:
                isna_entry = mask[i, 0]  # 获取掩码数组中当前行第一列的值
            else:
                isna_entry = _treat_as_na(val, False)  # 判断当前值是否为 NA 值

            if isna_entry:
                continue  # 如果是 NA 值，跳过当前循环

            if not first_element_set[lab]:
                # 如果当前标签的第一个元素未设置初始值，则将其设置为当前值
                out[lab, 0] = out[lab, 1] = out[lab, 2] = out[lab, 3] = val
                first_element_set[lab] = True  # 标记该标签已设置初始值
                if uses_mask:
                    result_mask[lab] = False  # 若使用掩码，将结果掩码数组中对应位置设为 False
            else:
                # 如果当前标签已设置初始值，则更新最高价、最低价和收盘价
                out[lab, 1] = max(out[lab, 1], val)
                out[lab, 2] = min(out[lab, 2], val)
                out[lab, 3] = val


@cython.boundscheck(False)
@cython.wraparound(False)
# 定义函数 `group_quantile`，用于按照指定标签计算分位数
def group_quantile(
    float64_t[:, ::1] out,  # 输出数组，存储计算后的分位数
    ndarray[numeric_t, ndim=1] values,  # 输入的值数组
    const intp_t[::1] labels,  # 标签数组，用于分组
    const uint8_t[:] mask,  # 掩码数组，标记 NA 值
    const float64_t[:] qs,  # 分位数数组，指定要计算的分位数
    const int64_t[::1] starts,  # 每个组开始的位置
    const int64_t[::1] ends,  # 每个组结束的位置
    str interpolation,  # 插值方法，如线性插值、最近值等
    uint8_t[:, ::1] result_mask,  # 结果掩码数组，标记计算结果是否有效
    bint is_datetimelike,  # 标记值是否为日期时间类型
) -> None:
    """
    Calculate the quantile per group.

    Parameters
    ----------
    out : np.ndarray[np.float64, ndim=2]
        Array of aggregated values that will be written to.
    values : np.ndarray
        Array containing the values to apply the function against.
    labels : ndarray[np.intp]
        Array containing the unique group labels.
    qs : ndarray[float64_t]
        The quantile values to search for.
    starts : ndarray[int64]
        Positions at which each group begins.
    ends : ndarray[int64]
        Positions at which each group ends.
    interpolation : {'linear', 'lower', 'highest', 'nearest', 'midpoint'}
        Method to use when interpolating between data points to compute quantile.
    result_mask : ndarray[bool, ndim=2] or None
        Optional mask array indicating invalid results.
    """
    # 定义变量 `is_datetimelike`，表示 int64 值是否代表 datetime64 类型的值
    is_datetimelike : bool
        Whether int64 values represent datetime64-like values.

    # 函数说明部分，此函数修改提供的 `out` 参数而非显式返回值
    Notes
    -----
    Rather than explicitly returning a value, this function modifies the
    provided `out` parameter.
    """
    # 定义 Cython 变量
    cdef:
        # 定义多个 Py_ssize_t 类型的变量，并初始化其值
        Py_ssize_t i, N=len(labels), ngroups, non_na_sz, k, nqs
        Py_ssize_t idx=0
        Py_ssize_t grp_size
        # 定义枚举类型变量 InterpolationEnumType 和 float64_t 类型变量
        InterpolationEnumType interp
        float64_t q_val, q_idx, frac, val, next_val
        # 定义 bint 类型变量 uses_result_mask，并根据 result_mask 是否为 None 进行初始化
        bint uses_result_mask = result_mask is not None
        Py_ssize_t start, end
        # 定义 ndarray 类型变量 grp 和 intp_t[::1] 类型变量 sort_indexer
        ndarray[numeric_t] grp
        intp_t[::1] sort_indexer
        # 定义 const uint8_t[:] 类型变量 sub_mask

    # 断言语句，确保 values 的行数等于 N
    assert values.shape[0] == N
    # 断言语句，确保 starts 不为 None
    assert starts is not None
    # 断言语句，确保 ends 不为 None，且 starts 和 ends 的长度相等
    assert ends is not None
    assert len(starts) == len(ends)

    # 检查 qs 中是否有不在 [0, 1] 范围内的值，若有则引发 ValueError 异常
    if any(not (0 <= q <= 1) for q in qs):
        wrong = [x for x in qs if not (0 <= x <= 1)][0]
        raise ValueError(
            f"Each 'q' must be between 0 and 1. Got '{wrong}' instead"
        )

    # 定义字典 inter_methods，用于存储不同插值方法的字符串与对应的枚举值映射关系
    inter_methods = {
        "linear": INTERPOLATION_LINEAR,
        "lower": INTERPOLATION_LOWER,
        "higher": INTERPOLATION_HIGHER,
        "nearest": INTERPOLATION_NEAREST,
        "midpoint": INTERPOLATION_MIDPOINT,
    }
    # 将 interpolation 对应的插值方法字符串映射为枚举值赋给 interp
    interp = inter_methods[interpolation]

    # 获取 qs 的长度赋给 nqs
    nqs = len(qs)
    # 获取 out 的长度赋给 ngroups
    ngroups = len(out)

    # TODO: get cnp.PyArray_ArgSort to work with nogil so we can restore the rest
    #  of this function as being `with nogil:`
    # （待完成）使 cnp.PyArray_ArgSort 在 nogil 模式下正常工作，以便可以将该函数的其余部分恢复为 `with nogil:` 模式
    # 遍历每个分组
    for i in range(ngroups):
        # 获取当前分组的起始和结束索引
        start = starts[i]
        end = ends[i]

        # 提取当前分组的数值
        grp = values[start:end]

        # 计算当前分组中非缺失值的数量
        sub_mask = mask[start:end]
        grp_size = sub_mask.size
        non_na_sz = 0
        for k in range(grp_size):
            if sub_mask[k] == 0:
                non_na_sz += 1

        # 根据数据类型是否为日期时间类型，选择合适的排序方式
        if is_datetimelike:
            # 当数据为日期时间类型时，需要将 NaT 放在排序的末尾
            sort_indexer = cnp.PyArray_ArgSort(grp.view("M8[ns]"), 0, cnp.NPY_QUICKSORT)
        else:
            sort_indexer = cnp.PyArray_ArgSort(grp, 0, cnp.NPY_QUICKSORT)

        # 如果当前分组没有非缺失值，根据插值策略填充结果
        if non_na_sz == 0:
            for k in range(nqs):
                if uses_result_mask:
                    result_mask[i, k] = 1
                else:
                    out[i, k] = NaN
        else:
            # 当存在非缺失值时，计算请求的分位数值
            for k in range(nqs):
                q_val = qs[k]

                # 计算请求分位数对应的索引
                # 将浮点数转换为整数，意图是截断结果
                idx = <int64_t>(q_val * <float64_t>(non_na_sz - 1))

                # 获取分位数对应的值
                val = grp[sort_indexer[idx]]

                # 如果请求的分位数正好对应一个索引，直接输出该索引的值；否则插值计算
                q_idx = q_val * (non_na_sz - 1)
                frac = q_idx % 1

                if frac == 0.0 or interp == INTERPOLATION_LOWER:
                    out[i, k] = val
                else:
                    next_val = grp[sort_indexer[idx + 1]]
                    if interp == INTERPOLATION_LINEAR:
                        out[i, k] = val + (next_val - val) * frac
                    elif interp == INTERPOLATION_HIGHER:
                        out[i, k] = next_val
                    elif interp == INTERPOLATION_MIDPOINT:
                        out[i, k] = (val + next_val) / 2.0
                    elif interp == INTERPOLATION_NEAREST:
                        if frac > .5 or (frac == .5 and idx % 2 == 1):
                            # 如果分位数处于两个索引的中间，则取偶数索引的值，与 np.quantile 类似
                            out[i, k] = next_val
                        else:
                            out[i, k] = val
# ----------------------------------------------------------------------
# group_nth, group_last, group_rank
# ----------------------------------------------------------------------

# 定义一个融合类型，包括numeric_object_t、complex64_t和complex128_t
ctypedef fused numeric_object_complex_t:
    numeric_object_t
    complex64_t
    complex128_t

# 定义一个Cython函数，用于判断给定值是否为NA（不可用值）
cdef bint _treat_as_na(numeric_object_complex_t val,
                       bint is_datetimelike) noexcept nogil:
    # 如果val的类型是object，使用GIL检查是否为null
    if numeric_object_complex_t is object:
        with gil:
            return checknull(val)
    
    # 如果val的类型是int64_t，且is_datetimelike为真，则返回是否为NPY_NAT的比较结果
    elif numeric_object_complex_t is int64_t:
        return is_datetimelike and val == NPY_NAT
    # 如果val的类型是float32_t、float64_t、complex64_t或complex128_t，则返回是否为NaN的比较结果
    elif (
        numeric_object_complex_t is float32_t
        or numeric_object_complex_t is float64_t
        or numeric_object_complex_t is complex64_t
        or numeric_object_complex_t is complex128_t
    ):
        return val != val
    else:
        # 非日期时间型整数，返回False
        return False

# 定义一个Cython函数，用于获取numeric_object_t类型的最小或最大值
cdef numeric_object_t _get_min_or_max(
    numeric_object_t val,
    bint compute_max,
    bint is_datetimelike,
):
    """
    找到numeric_object_t类型的最小值或最大值；'val'是一个占位符，使numeric_object_t成为一个参数。
    """
    return get_rank_nan_fill_val(
        not compute_max,
        val=val,
        is_datetimelike=is_datetimelike,
    )

# 定义一个Cython函数，用于获取numeric_t类型的NA值
cdef numeric_t _get_na_val(numeric_t val, bint is_datetimelike):
    cdef:
        numeric_t na_val
    
    # 根据numeric_t的类型和is_datetimelike的值，确定返回的NA值
    if numeric_t == float32_t or numeric_t == float64_t:
        na_val = NaN
    elif numeric_t is int64_t and is_datetimelike:
        na_val = NPY_NAT
    else:
        # 用于掩码情况下的默认值
        na_val = 0
    return na_val

# 定义一个融合类型，包括numeric_object_t、complex64_t和complex128_t
ctypedef fused mincount_t:
    numeric_object_t
    complex64_t
    complex128_t

# 使用Cython的装饰器定义一个内联函数，检查每个组的观测数量是否低于min_count，
# 如果是，则将该组的结果设置为适当的NA-like值。
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _check_below_mincount(
    mincount_t[:, ::1] out,
    bint uses_mask,
    uint8_t[:, ::1] result_mask,
    Py_ssize_t ncounts,
    Py_ssize_t K,
    const int64_t[:, ::1] nobs,
    int64_t min_count,
    mincount_t[:, ::1] resx,
) noexcept:
    """
    检查每个组的观测数量是否低于min_count，
    如果是，则将该组的结果设置为适当的NA-like值。
    """
    cdef:
        Py_ssize_t i, j
    # 使用 `nogil` 上下文管理器，确保在 Cython 中取消全局解释器锁以提高性能
    with nogil(mincount_t is not object):
        # 循环遍历每个观测的计数
        for i in range(ncounts):
            # 遍历每个聚类中的元素
            for j in range(K):
                # 检查当前观测在聚类 j 中的计数是否大于等于最小计数阈值
                if nobs[i, j] >= min_count:
                    # 若满足条件，则将 resx 中的值赋给 out[i, j]
                    out[i, j] = resx[i, j]
                else:
                    # 若不满足最小计数阈值条件，则根据数据类型和配置进行处理
                    # 如果是整数类型且非日期时间类型，并且不使用掩码，
                    # 则意味着 counts[i] < min_count，在 WrappedCythonOp._call_cython_op
                    # 的结尾将会被转换为 float64 并掩码处理。因此可以安全地
                    # 在 out[i, j] 中设置一个占位值。
                    if uses_mask:
                        # 如果使用掩码，则将 result_mask[i, j] 设为 True
                        result_mask[i, j] = True
                        # 将 out[i, j] 设置为 0，以确保结果的确定性，因为它是用 np.empty 初始化的。
                        # 这也确保我们在适当时可以向下转换 out。
                        out[i, j] = 0
                    elif (
                        mincount_t is float32_t
                        or mincount_t is float64_t
                        or mincount_t is complex64_t
                        or mincount_t is complex128_t
                    ):
                        # 根据 mincount_t 的类型设置 out[i, j] 为 NAN
                        out[i, j] = NAN
                    elif mincount_t is int64_t:
                        # 在非日期时间类型的情况下，作为占位符设置为 NPY_NAT
                        out[i, j] = NPY_NAT
                    elif mincount_t is object:
                        # 如果 mincount_t 是对象类型，则设置 out[i, j] 为 None
                        out[i, j] = None
                    else:
                        # 其他情况下，设置 out[i, j] 为 0，作为占位符，参见上文说明
                        out[i, j] = 0
@cython.wraparound(False)
@cython.boundscheck(False)
# 定义函数 group_last，用于在 axis=0 上进行聚合操作
def group_last(
    numeric_object_t[:, ::1] out,            # 输出数组，类型为 numeric_object_t 的二维数组
    int64_t[::1] counts,                     # 计数数组，长度为 N 的一维数组
    const numeric_object_t[:, :] values,     # 值数组，类型为 numeric_object_t 的二维常量数组
    const intp_t[::1] labels,                # 标签数组，长度为 N 的一维常量数组
    const uint8_t[:, :] mask,                # 掩码数组，类型为 uint8_t 的二维常量数组
    uint8_t[:, ::1] result_mask=None,        # 结果掩码数组，可选参数，默认为 None
    Py_ssize_t min_count=-1,                 # 最小计数阈值，可选参数，默认为 -1
    bint is_datetimelike=False,              # 是否是日期时间类型，布尔型参数，默认为 False
    bint skipna=True,                        # 是否跳过 NA 值，布尔型参数，默认为 True
) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)  # 定义变量 i, j, N, K, lab, ncounts，以及对 counts 的长度赋值
        numeric_object_t val                    # 定义变量 val，类型为 numeric_object_t
        numeric_object_t[:, ::1] resx           # 定义结果数组 resx，类型为 numeric_object_t 的二维数组
        int64_t[:, ::1] nobs                    # 定义观测计数数组 nobs，类型为 int64_t 的二维数组
        bint uses_mask = mask is not None       # 是否使用掩码的布尔型变量，如果 mask 不为 None 则为 True
        bint isna_entry                         # 是否是 NA 值的布尔型变量

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")  # 如果 values 和 labels 的长度不相等则引发断言错误

    min_count = max(min_count, 1)  # 确保 min_count 至少为 1
    nobs = np.zeros((<object>out).shape, dtype=np.int64)  # 初始化 nobs 数组为全零数组，形状与 out 相同，数据类型为 int64
    if numeric_object_t is object:
        resx = np.empty((<object>out).shape, dtype=object)  # 如果 numeric_object_t 是 object 类型，初始化 resx 为空数组，形状与 out 相同，数据类型为 object
    else:
        resx = np.empty_like(out)  # 否则，初始化 resx 为与 out 相同形状的空数组，数据类型与 out 相同

    N, K = (<object>values).shape  # 获取 values 的形状，分别赋值给 N 和 K

    with nogil(numeric_object_t is not object):
        # 使用 nogil 块，允许没有全局解释器锁的情况下执行，要求 numeric_object_t 不是 object 类型
        for i in range(N):  # 迭代 values 的行数 N
            lab = labels[i]  # 获取当前行对应的标签值
            if lab < 0:
                continue  # 如果标签值小于 0，则跳过当前循环

            counts[lab] += 1  # 对应标签的计数加一
            for j in range(K):  # 迭代 values 的列数 K
                val = values[i, j]  # 获取当前位置的值

                if skipna:
                    if uses_mask:
                        isna_entry = mask[i, j]  # 如果使用掩码，则获取掩码值
                    else:
                        isna_entry = _treat_as_na(val, is_datetimelike)  # 否则，调用 _treat_as_na 函数判断是否为 NA 值
                    if isna_entry:
                        continue  # 如果是 NA 值，则跳过当前循环

                nobs[lab, j] += 1  # 对应标签和列的观测计数加一
                resx[lab, j] = val  # 将当前值赋给 resx 对应位置

                if uses_mask and not skipna:
                    result_mask[lab, j] = mask[i, j]  # 如果使用掩码且不跳过 NA 值，则将掩码值赋给 result_mask

    _check_below_mincount(
        out, uses_mask, result_mask, ncounts, K, nobs, min_count, resx
    )  # 调用 _check_below_mincount 函数，检查是否满足最小计数要求


@cython.wraparound(False)
@cython.boundscheck(False)
# 定义函数 group_nth，用于在 axis=0 上进行聚合操作
def group_nth(
    numeric_object_t[:, ::1] out,            # 输出数组，类型为 numeric_object_t 的二维数组
    int64_t[::1] counts,                     # 计数数组，长度为 N 的一维数组
    const numeric_object_t[:, :] values,     # 值数组，类型为 numeric_object_t 的二维常量数组
    const intp_t[::1] labels,                # 标签数组，长度为 N 的一维常量数组
    const uint8_t[:, :] mask,                # 掩码数组，类型为 uint8_t 的二维常量数组
    uint8_t[:, ::1] result_mask=None,        # 结果掩码数组，可选参数，默认为 None
    int64_t min_count=-1,                    # 最小计数阈值，可选参数，默认为 -1
    int64_t rank=1,                          # 排名值，可选参数，默认为 1
    bint is_datetimelike=False,              # 是否是日期时间类型，布尔型参数，默认为 False
    bint skipna=True,                        # 是否跳过 NA 值，布尔型参数，默认为 True
) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)  # 定义变量 i, j, N, K, lab, ncounts，以及对 counts 的长度赋值
        numeric_object_t val                    # 定义变量 val，类型为 numeric_object_t
        numeric_object_t[:, ::1] resx           # 定义结果数组 resx，类型为 numeric_object_t 的二维数组
        int64_t[:, ::1] nobs                    # 定义观测计数数组 nobs，类型为 int64_t 的二维数组
        bint uses_mask = mask is not None       # 是否使用掩码的布尔型变量，如果 mask 不为 None 则为 True
        bint isna_entry                         # 是否是 NA 值的布尔型变量

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")  # 如果 values 和 labels 的长度不相等则引发断言错误

    min_count = max(min_count, 1)  # 确保 min_count 至少为 1
    nobs = np.zeros((<object>out).shape, dtype=np.int64)  # 初始化 nobs 数组为全零数组，形状与 out 相同，数据类型为 int64
    if numeric_object_t is object:
        resx = np.empty((<object>out).shape, dtype=object)  # 如果 numeric_object_t 是 object 类型，初始化 resx 为空数组，形状与 out 相同，数据类型为 object
    else:
        resx = np.empty_like(out)  # 否则，初始化 resx 为与 out 相同形状的空数组，数据类型与 out 相同

    N, K = (<object>values).shape  # 获取 values 的形状，分别赋值给 N 和 K
    # 使用 nogil 上下文管理器，此处可能是 Cython 中的特定语法，用于提高代码执行效率或者处理特定类型的对象
    with nogil(numeric_object_t is not object):
        # 遍历范围为 N 的索引，执行以下操作
        for i in range(N):
            # 获取标签数组中第 i 个元素的值
            lab = labels[i]
            # 如果标签值小于 0，则跳过当前循环继续下一次迭代
            if lab < 0:
                continue
    
            # 增加 counts 数组中 lab 索引位置的计数值
            counts[lab] += 1
            # 遍历范围为 K 的索引，执行以下操作
            for j in range(K):
                # 获取 values 数组中 (i, j) 位置的值
                val = values[i, j]
    
                # 如果 skipna 参数为 True，则执行以下条件判断
                if skipna:
                    # 如果 uses_mask 为 True，则获取 mask 数组中 (i, j) 位置的值
                    if uses_mask:
                        isna_entry = mask[i, j]
                    else:
                        # 否则，调用 _treat_as_na 函数判断 val 是否为缺失值
                        isna_entry = _treat_as_na(val, is_datetimelike)
                    # 如果 isna_entry 为 True，则跳过当前循环继续下一次迭代
                    if isna_entry:
                        continue
    
                # 增加 nobs 数组中 (lab, j) 索引位置的计数值
                nobs[lab, j] += 1
                # 如果 nobs 数组中 (lab, j) 索引位置的计数值等于 rank
                if nobs[lab, j] == rank:
                    # 设置 resx 数组中 (lab, j) 索引位置的值为 val
                    resx[lab, j] = val
                    # 如果 uses_mask 为 True 并且 skipna 为 False，则设置 result_mask 数组中 (lab, j) 索引位置的值为 mask[i, j]
                    if uses_mask and not skipna:
                        result_mask[lab, j] = mask[i, j]
    
    # 调用 _check_below_mincount 函数，对输出结果 out 进行检查并可能修改，传递的参数包括 uses_mask, result_mask, ncounts, K, nobs, min_count, resx
    _check_below_mincount(
        out, uses_mask, result_mask, ncounts, K, nobs, min_count, resx
    )
# 设置 Cython 的边界检查为 False
@cython.boundscheck(False)
# 设置 Cython 的数组访问越界检查为 False
@cython.wraparound(False)
# 定义函数 group_rank，计算每个组内值的排名
def group_rank(
    # 输出结果的二维数组，存储排名结果
    float64_t[:, ::1] out,
    # 待排名的二维数值数组
    ndarray[numeric_object_t, ndim=2] values,
    # 每个值对应的组标签数组，用于标识数据属于哪个组
    const intp_t[::1] labels,
    # 组的数量，此参数在本函数中未使用，用于匹配其他 groupby 函数的签名
    int ngroups,
    # 布尔值，指示 values 是否包含类似日期时间的条目
    bint is_datetimelike,
    # 定义如何处理并列值的方法，如 'average', 'min', 'max' 等
    str ties_method="average",
    # 布尔值，指示排名是升序还是降序排列
    bint ascending=True,
    # 布尔值，指示是否计算数据在组内的百分比排名
    bint pct=False,
    # 定义处理缺失值的方法，如 'keep', 'top', 'bottom'
    str na_option="keep",
    # 可选的屏蔽掩码数组，用于标识不参与排名的值
    const uint8_t[:, :] mask=None,
) -> None:
    """
    Provides the rank of values within each group.

    Parameters
    ----------
    out : np.ndarray[np.float64, ndim=2]
        Values to which this method will write its results.
    values : np.ndarray of numeric_object_t values to be ranked
    labels : np.ndarray[np.intp]
        Array containing unique label for each group, with its ordering
        matching up to the corresponding record in `values`
    ngroups : int
        This parameter is not used, is needed to match signatures of other
        groupby functions.
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    ties_method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        * average: average rank of group
        * min: lowest rank in group
        * max: highest rank in group
        * first: ranks assigned in order they appear in the array
        * dense: like 'min', but rank always increases by 1 between groups
    ascending : bool, default True
        False for ranks by high (1) to low (N)
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
    pct : bool, default False
        Compute percentage rank of data within each group
    na_option : {'keep', 'top', 'bottom'}, default 'keep'
        * keep: leave NA values where they are
        * top: smallest rank if ascending
        * bottom: smallest rank if descending
    mask : np.ndarray[bool] or None, default None

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object
    """
    cdef:
        # 声明循环中使用的变量
        Py_ssize_t i, k, N
        # 用于存储单个维度排名结果的数组
        ndarray[float64_t, ndim=1] result
        # 子屏蔽掩码数组，用于存储循环中的特定列的掩码值

    # 获取 values 的列数 N
    N = values.shape[1]

    # 对每一列进行排名计算
    for k in range(N):
        # 如果掩码为空，则子掩码为 None，否则为指定列的掩码值
        if mask is None:
            sub_mask = None
        else:
            sub_mask = mask[:, k]

        # 调用 rank_1d 函数计算单个维度的排名结果
        result = rank_1d(
            values=values[:, k],
            labels=labels,
            is_datetimelike=is_datetimelike,
            ties_method=ties_method,
            ascending=ascending,
            pct=pct,
            na_option=na_option,
            mask=sub_mask,
        )
        # 将计算得到的排名结果写入输出数组 out 的对应位置
        for i in range(len(result)):
            if labels[i] >= 0:
                out[i, k] = result[i]


# ----------------------------------------------------------------------
# group_min, group_max
# ----------------------------------------------------------------------

# 设置 Cython 的数组访问越界检查为 False
@cython.wraparound(False)
# 设置 Cython 的边界检查为 False
@cython.boundscheck(False)
# 定义 Cython 函数 group_min_max，用于计算每个组内的最小值和最大值
cdef group_min_max(
    # 输出结果的二维数组，存储每个组的最小和最大值
    numeric_t[:, ::1] out,
    # 计数数组，用于存储每个组的计数
    int64_t[::1] counts,
    # 待计算的二维数值数组
    ndarray[numeric_t, ndim=2] values,
    # 每个值对应的组标签数组，用于标识数据属于哪个组
    const intp_t[::1] labels,
    # 最小计数值，默认为 -1，表示无最小计数限制
    Py_ssize_t min_count=-1,
    # 布尔值，指示 values 是否包含类似日期时间的条目
    bint is_datetimelike=False,
    # 声明一个布尔型参数 compute_max，默认为 True
    bint compute_max=True,
    # 声明一个二维数组参数 mask，类型为 uint8_t，可以是任意方向的数组
    const uint8_t[:, ::1] mask=None,
    # 声明一个二维数组参数 result_mask，类型为 uint8_t，可以是任意方向的数组，默认为 None
    uint8_t[:, ::1] result_mask=None,
):
    """
    Compute minimum/maximum of columns of `values`, in row groups `labels`.

    Parameters
    ----------
    out : np.ndarray[numeric_t, ndim=2]
        Array to store result in.
    counts : np.ndarray[int64]
        Input as a zeroed array, populated by group sizes during algorithm
    values : array
        Values to find column-wise min/max of.
    labels : np.ndarray[np.intp]
        Labels to group by.
    min_count : Py_ssize_t, default -1
        The minimum number of non-NA group elements, NA result if threshold
        is not met
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    compute_max : bint, default True
        True to compute group-wise max, False to compute min
    mask : ndarray[bool, ndim=2], optional
        If not None, indices represent missing values,
        otherwise the mask will not be used
    result_mask : ndarray[bool, ndim=2], optional
        If not None, these specify locations in the output that are NA.
        Modified in-place.

    Notes
    -----
    This method modifies the `out` parameter, rather than returning an object.
    `counts` is modified to hold group sizes
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ngroups = len(counts)
        numeric_t val
        numeric_t[:, ::1] group_min_or_max
        int64_t[:, ::1] nobs
        bint uses_mask = mask is not None
        bint isna_entry

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    # 确保最小计数不小于1
    min_count = max(min_count, 1)
    # 初始化观测值计数数组
    nobs = np.zeros((<object>out).shape, dtype=np.int64)

    # 初始化分组最小/最大值数组
    group_min_or_max = np.empty_like(out)
    group_min_or_max[:] = _get_min_or_max(<numeric_t>0, compute_max, is_datetimelike)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            # 增加当前组的计数
            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    isna_entry = _treat_as_na(val, is_datetimelike)

                # 如果不是缺失值，增加观测值计数，并更新分组的最小/最大值
                if not isna_entry:
                    nobs[lab, j] += 1
                    if compute_max:
                        if val > group_min_or_max[lab, j]:
                            group_min_or_max[lab, j] = val
                    else:
                        if val < group_min_or_max[lab, j]:
                            group_min_or_max[lab, j] = val

    # 检查是否低于最小计数阈值，可能将结果标记为缺失值
    _check_below_mincount(
        out, uses_mask, result_mask, ngroups, K, nobs, min_count, group_min_or_max
    )


@cython.wraparound(False)
@cython.boundscheck(False)
def group_idxmin_idxmax(
    intp_t[:, ::1] out,
    int64_t[::1] counts,
    ndarray[numeric_object_t, ndim=2] values,
    const intp_t[::1] labels,
    Py_ssize_t min_count=-1,
    bint is_datetimelike=False,
    const uint8_t[:, ::1] mask=None,
    str name="idxmin",
    bint skipna=True,
    # 声明一个参数 result_mask，类型为 uint8_t 的二维数组（即 uint8_t[:, ::1]）
    uint8_t[:, ::1] result_mask=None,
    """
    Compute index of minimum/maximum of columns of `values`, in row groups `labels`.

    This function only computes the row number where the minimum/maximum occurs, we'll
    take the corresponding index value after this function.

    Parameters
    ----------
    out : np.ndarray[intp, ndim=2]
        Array to store result in.
    counts : np.ndarray[int64]
        Input as a zeroed array, populated by group sizes during algorithm
    values : np.ndarray[numeric_object_t, ndim=2]
        Values to find column-wise min/max of.
    labels : np.ndarray[np.intp]
        Labels to group by.
    min_count : Py_ssize_t, default -1
        The minimum number of non-NA group elements, NA result if threshold
        is not met.
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    name : {"idxmin", "idxmax"}, default "idxmin"
        Whether to compute idxmin or idxmax.
    mask : ndarray[bool, ndim=2], optional
        If not None, indices represent missing values,
        otherwise the mask will not be used
    skipna : bool, default True
        Flag to ignore nan values during truth testing
    result_mask : ndarray[bool, ndim=2], optional
        If not None, these specify locations in the output that are NA.
        Modified in-place.

    Notes
    -----
    This method modifies the `out` parameter, rather than returning an object.
    `counts` is modified to hold group sizes
    """
    # 声明变量和类型
    cdef:
        Py_ssize_t i, j, N, K, lab  # 声明整数变量
        numeric_object_t val  # 声明数值对象类型的变量
        numeric_object_t[:, ::1] group_min_or_max  # 二维数组，用于存储最小或最大值的索引
        uint8_t[:, ::1] seen  # 二维数组，用于标记是否已处理过
        bint uses_mask = mask is not None  # 布尔变量，指示是否使用了掩码
        bint isna_entry  # 布尔变量，指示当前条目是否为NA
        bint compute_max = name == "idxmax"  # 布尔变量，指示是否计算最大值

    # 断言检查参数 name 的合法性
    assert name == "idxmin" or name == "idxmax"

    # 检查 values 和 labels 的长度是否相同
    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    # 获取 values 的行数 N 和列数 K
    N, K = (<object>values).shape

    # 根据 numeric_object_t 类型创建适当类型的数组
    if numeric_object_t is object:
        group_min_or_max = np.empty((<object>out).shape, dtype=object)
        seen = np.zeros((<object>out).shape, dtype=np.uint8)
    else:
        group_min_or_max = np.empty_like(out, dtype=values.dtype)
        seen = np.zeros_like(out, dtype=np.uint8)

    # 使用 transform 时，确保 out 数组有有效值来处理未观察到的类别
    out[:] = 0
    # 使用 nogil 上下文管理器，针对 numeric_object_t 不是 object 的情况
    with nogil(numeric_object_t is not object):
        # 循环遍历范围为 N 的索引 i
        for i in range(N):
            # 获取标签 labels[i]
            lab = labels[i]
            # 如果 lab 小于 0，则跳过当前循环
            if lab < 0:
                continue

            # 遍历范围为 K 的索引 j
            for j in range(K):
                # 如果不跳过 NA 并且 out[lab, j] 等于 -1，则继续下一轮循环
                if not skipna and out[lab, j] == -1:
                    # 一旦遇到 NA，就无法回头
                    continue

                # 获取 values[i, j] 的值
                val = values[i, j]

                # 如果使用掩码，则检查 mask[i, j] 是否为 NA
                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    # 否则调用 _treat_as_na 函数判断是否为 NA，根据 is_datetimelike 决定
                    isna_entry = _treat_as_na(val, is_datetimelike)

                # 如果是 NA 条目
                if isna_entry:
                    # 如果不跳过 NA 或者 (不跳过 NA 且 (lab, j) 位置未见过)
                    if not skipna or not seen[lab, j]:
                        # 将 out[lab, j] 置为 -1
                        out[lab, j] = -1
                else:
                    # 如果 (lab, j) 位置未见过
                    if not seen[lab, j]:
                        # 将 (lab, j) 位置标记为已见过
                        seen[lab, j] = True
                        # 将 group_min_or_max[lab, j] 设置为当前值 val
                        group_min_or_max[lab, j] = val
                        # 将 out[lab, j] 设置为 i
                        out[lab, j] = i
                    # 否则，如果计算最大值
                    elif compute_max:
                        # 如果当前值 val 大于 group_min_or_max[lab, j] 的值
                        if val > group_min_or_max[lab, j]:
                            # 更新 group_min_or_max[lab, j] 为当前值 val
                            group_min_or_max[lab, j] = val
                            # 更新 out[lab, j] 为 i
                            out[lab, j] = i
                    # 否则，即计算最小值
                    else:
                        # 如果当前值 val 小于 group_min_or_max[lab, j] 的值
                        if val < group_min_or_max[lab, j]:
                            # 更新 group_min_or_max[lab, j] 为当前值 val
                            group_min_or_max[lab, j] = val
                            # 更新 out[lab, j] 为 i
                            out[lab, j] = i
# 设置 Cython 函数装饰器，禁用数组访问的边界检查
@cython.wraparound(False)
# 设置 Cython 函数装饰器，禁用负数索引的边界检查
@cython.boundscheck(False)
def group_max(
    # 输出结果的二维数组，每行存储每个组的最大值
    numeric_t[:, ::1] out,
    # 存储每个组中非缺失值的计数数组
    int64_t[::1] counts,
    # 存储需要计算最大值的数据的二维数组
    ndarray[numeric_t, ndim=2] values,
    # 表示每行数据所属的组的整数标签数组
    const intp_t[::1] labels,
    # 最小非缺失值数量，默认为-1表示没有限制
    Py_ssize_t min_count=-1,
    # 是否处理类似于日期时间的数据类型
    bint is_datetimelike=False,
    # 可选的缺失值掩码数组
    const uint8_t[:, ::1] mask=None,
    # 可选的结果缺失值掩码数组，用于标记输出中的缺失值位置
    uint8_t[:, ::1] result_mask=None,
) -> None:
    """See group_min_max.__doc__"""
    # 调用 group_min_max 函数计算每个组的最大值
    group_min_max(
        out,
        counts,
        values,
        labels,
        min_count=min_count,
        is_datetimelike=is_datetimelike,
        compute_max=True,
        mask=mask,
        result_mask=result_mask,
    )


# 设置 Cython 函数装饰器，禁用数组访问的边界检查
@cython.wraparound(False)
# 设置 Cython 函数装饰器，禁用负数索引的边界检查
@cython.boundscheck(False)
def group_min(
    # 输出结果的二维数组，每行存储每个组的最小值
    numeric_t[:, ::1] out,
    # 存储每个组中非缺失值的计数数组
    int64_t[::1] counts,
    # 存储需要计算最小值的数据的二维数组
    ndarray[numeric_t, ndim=2] values,
    # 表示每行数据所属的组的整数标签数组
    const intp_t[::1] labels,
    # 最小非缺失值数量，默认为-1表示没有限制
    Py_ssize_t min_count=-1,
    # 是否处理类似于日期时间的数据类型
    bint is_datetimelike=False,
    # 可选的缺失值掩码数组
    const uint8_t[:, ::1] mask=None,
    # 可选的结果缺失值掩码数组，用于标记输出中的缺失值位置
    uint8_t[:, ::1] result_mask=None,
) -> None:
    """See group_min_max.__doc__"""
    # 调用 group_min_max 函数计算每个组的最小值
    group_min_max(
        out,
        counts,
        values,
        labels,
        min_count=min_count,
        is_datetimelike=is_datetimelike,
        compute_max=False,
        mask=mask,
        result_mask=result_mask,
    )


# 设置 Cython 函数装饰器，禁用负数索引的边界检查
@cython.boundscheck(False)
# 设置 Cython 函数装饰器，禁用数组访问的边界检查
@cython.wraparound(False)
cdef group_cummin_max(
    # 输出结果的二维数组，每行存储每个组的累计最小或最大值
    numeric_t[:, ::1] out,
    # 存储需要计算累计最小或最大值的数据的二维数组
    ndarray[numeric_t, ndim=2] values,
    # 缺失值掩码数组，标识数据中的缺失值位置
    const uint8_t[:, ::1] mask,
    # 结果缺失值掩码数组，用于标记输出中的缺失值位置
    uint8_t[:, ::1] result_mask,
    # 表示每行数据所属的组的整数标签数组
    const intp_t[::1] labels,
    # 组的数量，大于标签数组中的所有值
    int ngroups,
    # 是否处理类似于日期时间的数据类型
    bint is_datetimelike,
    # 是否跳过缺失值进行计算
    bint skipna,
    # 是否计算累计最大值，否则计算累计最小值
    bint compute_max,
):
    """
    计算 `values` 数据的列在 `labels` 中的行组的累计最小/最大值。

    Parameters
    ----------
    out : np.ndarray[numeric_t, ndim=2]
        用于存储累计最小/最大值的数组。
    values : np.ndarray[numeric_t, ndim=2]
        需要计算累计最小/最大值的数据。
    mask : np.ndarray[bool] or None
        如果不为 None，则索引表示缺失值，否则不使用掩码。
    result_mask : ndarray[bool, ndim=2], optional
        如果不为 None，则指定输出中的 NA 位置。原地修改。
    labels : np.ndarray[np.intp]
        分组标签。
    ngroups : int
        组数，大于 `labels` 中的所有条目。
    is_datetimelike : bool
        如果 `values` 包含类似日期时间的条目，则为 True。
    skipna : bool
        如果为 True，则忽略 `values` 中的 NaN。
    compute_max : bool
        如果应计算累计最大值，则为 True；否则应计算累计最小值。

    Notes
    -----
    此方法修改 `out` 参数，而不是返回对象。
    """
    cdef:
        numeric_t[:, ::1] accum  # 累加值数组
        Py_ssize_t i, j, N, K  # 循环索引变量
        numeric_t val, mval, na_val  # 当前值、最小/最大值、NA 值
        uint8_t[:, ::1] seen_na  # 观察到的 NA 值掩码数组
        intp_t lab  # 当前标签值
        bint na_possible  # 是否可能有 NA 值
        bint uses_mask = mask is not None  # 是否使用掩码
        bint isna_entry  # 当前条目是否为 NA 值

    # 创建一个与 `values` 形状相同的空数组，用于累计最小/最大值的计算
    accum = np.empty((ngroups, (<object>values).shape[1]), dtype=values.dtype)
    # 将累积数组初始化为指定类型的最小值或最大值
    accum[:] = _get_min_or_max(<numeric_t>0, compute_max, is_datetimelike)

    # 获取表示缺失值的数值，根据数据类型和是否为日期时间类型确定
    na_val = _get_na_val(<numeric_t>0, is_datetimelike)

    # 如果使用掩码，标记可能存在缺失值
    if uses_mask:
        na_possible = True
        # 仅为了避免未初始化警告，na_val 将不会被使用
        na_val = 0
    elif numeric_t is float64_t or numeric_t is float32_t:
        na_possible = True
    elif is_datetimelike:
        na_possible = True
    else:
        # 仅为了避免未初始化警告，na_possible 将不会为 False
        na_possible = False

    # 如果可能存在缺失值，初始化一个用于跟踪是否见过缺失值的数组
    if na_possible:
        seen_na = np.zeros((<object>accum).shape, dtype=np.uint8)

    # 获取数据集的行数 N 和列数 K
    N, K = (<object>values).shape

    # 使用 nogil 块来提高性能，遍历数据集
    with nogil:
        for i in range(N):
            lab = labels[i]
            # 如果标签为负数，跳过当前循环
            if lab < 0:
                continue
            for j in range(K):

                # 如果不跳过缺失值并且该位置可能有缺失值且已经见过缺失值
                if not skipna and na_possible and seen_na[lab, j]:
                    if uses_mask:
                        # 如果使用掩码，将结果掩码置为 1
                        result_mask[i, j] = 1
                        # 将输出值置为 0，确保结果可确定并可以适当地降级
                        out[i, j] = 0
                    else:
                        # 将输出值置为缺失值 na_val
                        out[i, j] = na_val
                else:
                    val = values[i, j]

                    # 如果使用掩码，判断当前条目是否为缺失值
                    if uses_mask:
                        isna_entry = mask[i, j]
                    else:
                        # 否则根据数值及是否为日期时间类型判断当前条目是否为缺失值
                        isna_entry = _treat_as_na(val, is_datetimelike)

                    # 如果当前条目不是缺失值
                    if not isna_entry:
                        mval = accum[lab, j]
                        # 如果需要计算最大值，更新累积数组中的最大值
                        if compute_max:
                            if val > mval:
                                accum[lab, j] = mval = val
                        else:
                            # 如果需要计算最小值，更新累积数组中的最小值
                            if val < mval:
                                accum[lab, j] = mval = val
                        # 将输出值设置为累积数组中的值
                        out[i, j] = mval
                    else:
                        # 如果当前条目是缺失值，标记为已见过缺失值，并将输出值设置为当前值
                        seen_na[lab, j] = 1
                        out[i, j] = val
# 应用 Cython 的优化指令，关闭数组边界检查和负数索引包装
@cython.boundscheck(False)
@cython.wraparound(False)
def group_cummin(
    numeric_t[:, ::1] out,  # 输出的二维数组，用于存储每组的累计最小值
    ndarray[numeric_t, ndim=2] values,  # 输入的二维数值数组
    const intp_t[::1] labels,  # 整型一维数组，表示每个元素的组标签
    int ngroups,  # 组的数量
    bint is_datetimelike,  # 布尔值，指示数值是否类似日期时间
    const uint8_t[:, ::1] mask=None,  # 可选的二维字节流掩码数组
    uint8_t[:, ::1] result_mask=None,  # 可选的二维字节流结果掩码数组
    bint skipna=True,  # 是否跳过 NaN 值
) -> None:
    """See group_cummin_max.__doc__"""
    # 调用 group_cummin_max 函数计算每组的累计最小值，详见其文档
    group_cummin_max(
        out=out,
        values=values,
        mask=mask,
        result_mask=result_mask,
        labels=labels,
        ngroups=ngroups,
        is_datetimelike=is_datetimelike,
        skipna=skipna,
        compute_max=False,  # 指示计算累计最小值
    )


# 应用 Cython 的优化指令，关闭数组边界检查和负数索引包装
@cython.boundscheck(False)
@cython.wraparound(False)
def group_cummax(
    numeric_t[:, ::1] out,  # 输出的二维数组，用于存储每组的累计最大值
    ndarray[numeric_t, ndim=2] values,  # 输入的二维数值数组
    const intp_t[::1] labels,  # 整型一维数组，表示每个元素的组标签
    int ngroups,  # 组的数量
    bint is_datetimelike,  # 布尔值，指示数值是否类似日期时间
    const uint8_t[:, ::1] mask=None,  # 可选的二维字节流掩码数组
    uint8_t[:, ::1] result_mask=None,  # 可选的二维字节流结果掩码数组
    bint skipna=True,  # 是否跳过 NaN 值
) -> None:
    """See group_cummin_max.__doc__"""
    # 调用 group_cummin_max 函数计算每组的累计最大值，详见其文档
    group_cummin_max(
        out=out,
        values=values,
        mask=mask,
        result_mask=result_mask,
        labels=labels,
        ngroups=ngroups,
        is_datetimelike=is_datetimelike,
        skipna=skipna,
        compute_max=True,  # 指示计算累计最大值
    )
```