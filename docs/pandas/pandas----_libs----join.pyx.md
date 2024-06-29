# `D:\src\scipysrc\pandas\pandas\_libs\join.pyx`

```
# 导入 Cython 模块
cimport cython
# 导入 Cython 所需的 Py_ssize_t 类型
from cython cimport Py_ssize_t
# 导入 NumPy 库
import numpy as np

# 导入 Cython 中的 NumPy 接口
cimport numpy as cnp
# 从 NumPy 中导入特定的类型
from numpy cimport (
    int64_t,
    intp_t,
    ndarray,
)

# 调用 NumPy 的 import_array() 函数
cnp.import_array()

# 从 pandas 库中导入 groupsort_indexer 函数
from pandas._libs.algos import groupsort_indexer

# 从 pandas 库中导入数值相关的数据类型
from pandas._libs.dtypes cimport (
    numeric_object_t,
    numeric_t,
)

# 定义一个内连接函数，设置不进行负索引包装和越界检查
@cython.wraparound(False)
@cython.boundscheck(False)
def inner_join(const intp_t[:] left, const intp_t[:] right,
               Py_ssize_t max_groups, bint sort=True):
    # 定义 Cython 本地变量
    cdef:
        Py_ssize_t i, j, k, count = 0
        intp_t[::1] left_sorter, right_sorter
        intp_t[::1] left_count, right_count
        intp_t[::1] left_indexer, right_indexer
        intp_t lc, rc
        Py_ssize_t left_pos = 0, right_pos = 0, position = 0
        Py_ssize_t offset

    # 调用 groupsort_indexer 函数，获取左侧和右侧数组的排序器和计数器
    left_sorter, left_count = groupsort_indexer(left, max_groups)
    right_sorter, right_count = groupsort_indexer(right, max_groups)

    # 使用 nogil 块，进行无全局解锁的并行处理
    with nogil:
        # 第一遍扫描，确定结果集的大小，不使用 NA 组
        for i in range(1, max_groups + 1):
            lc = left_count[i]
            rc = right_count[i]

            if rc > 0 and lc > 0:
                count += lc * rc

    # 创建用于存储结果索引的 NumPy 数组
    left_indexer = np.empty(count, dtype=np.intp)
    right_indexer = np.empty(count, dtype=np.intp)

    # 再次使用 nogil 块，进行无全局解锁的并行处理
    with nogil:
        # 排除 NA 组
        left_pos = left_count[0]
        right_pos = right_count[0]
        for i in range(1, max_groups + 1):
            lc = left_count[i]
            rc = right_count[i]

            if rc > 0 and lc > 0:
                for j in range(lc):
                    offset = position + j * rc
                    for k in range(rc):
                        left_indexer[offset + k] = left_pos + j
                        right_indexer[offset + k] = right_pos + k
                position += lc * rc
            left_pos += lc
            right_pos += rc

        # 使用 _get_result_indexer 函数填充左右索引器的结果
        _get_result_indexer(left_sorter, left_indexer)
        _get_result_indexer(right_sorter, right_indexer)

    # 如果不需要排序，则按原始顺序返回结果
    if not sort:
        if len(left) == len(left_indexer):
            # 左侧没有多个匹配项
            rev = np.empty(len(left), dtype=np.intp)
            rev.put(np.asarray(left_sorter), np.arange(len(left)))
        else:
            # 使用 groupsort_indexer 函数获取索引
            rev, _ = groupsort_indexer(left_indexer, len(left))

        return np.asarray(left_indexer).take(rev), np.asarray(right_indexer).take(rev)
    else:
        # 如果需要排序，则直接返回结果
        return np.asarray(left_indexer), np.asarray(right_indexer)


# 定义左外连接函数，设置不进行负索引包装和越界检查
@cython.wraparound(False)
@cython.boundscheck(False)
def left_outer_join(const intp_t[:] left, const intp_t[:] right,
                    Py_ssize_t max_groups, bint sort=True):
    cdef:
        # 定义多个 C 语言级别的变量
        Py_ssize_t i, j, k, count = 0
        # 定义一个 ndarray 类型的变量 rev，用于存储索引
        ndarray[intp_t] rev
        # 定义两个 intp_t 类型的一维数组 left_count 和 right_count，存储分组计数
        intp_t[::1] left_count, right_count
        # 定义两个 intp_t 类型的一维数组 left_sorter 和 right_sorter，存储排序后的索引
        intp_t[::1] left_sorter, right_sorter
        # 定义两个 intp_t 类型的一维数组 left_indexer 和 right_indexer，存储结果索引
        intp_t[::1] left_indexer, right_indexer
        # 定义 lc 和 rc 两个变量，存储左右数组中各组的计数
        intp_t lc, rc
        # 定义三个 Py_ssize_t 类型的变量，分别记录左右数组的位置和结果索引的位置
        Py_ssize_t left_pos = 0, right_pos = 0, position = 0
        # 定义一个 Py_ssize_t 类型的变量 offset，用于偏移计算
        Py_ssize_t offset

    # 调用 groupsort_indexer 函数，获取左数组的排序索引和计数
    left_sorter, left_count = groupsort_indexer(left, max_groups)
    # 调用 groupsort_indexer 函数，获取右数组的排序索引和计数
    right_sorter, right_count = groupsort_indexer(right, max_groups)

    with nogil:
        # 第一次遍历，确定结果集的大小，不使用 NA 组
        for i in range(1, max_groups + 1):
            # 获取左数组和右数组当前组的计数
            lc = left_count[i]
            rc = right_count[i]

            # 计算结果集的大小
            if rc > 0:
                count += lc * rc
            else:
                count += lc

    # 创建两个空的 intp_t 类型的一维数组 left_indexer 和 right_indexer，用于存储结果索引
    left_indexer = np.empty(count, dtype=np.intp)
    right_indexer = np.empty(count, dtype=np.intp)

    with nogil:
        # 排除 NA 组
        left_pos = left_count[0]
        right_pos = right_count[0]
        for i in range(1, max_groups + 1):
            # 获取左数组和右数组当前组的计数
            lc = left_count[i]
            rc = right_count[i]

            if rc == 0:
                # 处理右数组当前组为空的情况
                for j in range(lc):
                    left_indexer[position + j] = left_pos + j
                    right_indexer[position + j] = -1
                position += lc
            else:
                # 处理右数组当前组非空的情况
                for j in range(lc):
                    offset = position + j * rc
                    for k in range(rc):
                        left_indexer[offset + k] = left_pos + j
                        right_indexer[offset + k] = right_pos + k
                position += lc * rc
            left_pos += lc
            right_pos += rc

        # 将结果存储在 left_indexer 和 right_indexer 中
        _get_result_indexer(left_sorter, left_indexer)
        _get_result_indexer(right_sorter, right_indexer)

    if not sort:  # 如果不需要排序，恢复到原始顺序
        if len(left) == len(left_indexer):
            # 左数组中没有任何行有多个匹配项
            # 这是一个快捷方式，避免再次调用 groupsort_indexer
            rev = np.empty(len(left), dtype=np.intp)
            rev.put(np.asarray(left_sorter), np.arange(len(left)))
        else:
            # 获取 left_indexer 的排序索引
            rev, _ = groupsort_indexer(left_indexer, len(left))

        # 返回按照 rev 排序后的 left_indexer 和 right_indexer
        return np.asarray(left_indexer).take(rev), np.asarray(right_indexer).take(rev)
    else:
        # 如果需要排序，直接返回 left_indexer 和 right_indexer
        return np.asarray(left_indexer), np.asarray(right_indexer)
@cython.wraparound(False)
@cython.boundscheck(False)
def full_outer_join(const intp_t[:] left, const intp_t[:] right,
                    Py_ssize_t max_groups):
    cdef:
        Py_ssize_t i, j, k, count = 0
        intp_t[::1] left_sorter, right_sorter
        intp_t[::1] left_count, right_count
        intp_t[::1] left_indexer, right_indexer
        intp_t lc, rc
        intp_t left_pos = 0, right_pos = 0
        Py_ssize_t offset, position = 0

    # 调用groupsort_indexer函数对left和right数组进行分组排序，并获取排序后的索引和每个组的计数
    left_sorter, left_count = groupsort_indexer(left, max_groups)
    right_sorter, right_count = groupsort_indexer(right, max_groups)

    with nogil:
        # 第一遍扫描，确定结果集的大小，不包括NA组
        for i in range(1, max_groups + 1):
            lc = left_count[i]  # 获取左侧第i组的计数
            rc = right_count[i]  # 获取右侧第i组的计数

            if rc > 0 and lc > 0:
                count += lc * rc  # 如果左右组都不为空，结果集大小加上左右组的笛卡尔积大小
            else:
                count += lc + rc  # 否则加上左组和右组的元素总数（其中一个为0时）

    # 创建结果索引数组
    left_indexer = np.empty(count, dtype=np.intp)
    right_indexer = np.empty(count, dtype=np.intp)

    with nogil:
        # 不包括NA组
        left_pos = left_count[0]  # 获取左侧NA组的位置
        right_pos = right_count[0]  # 获取右侧NA组的位置
        for i in range(1, max_groups + 1):
            lc = left_count[i]  # 获取左侧第i组的计数
            rc = right_count[i]  # 获取右侧第i组的计数

            if rc == 0:
                # 如果右侧第i组为空，将左侧第i组的索引填入left_indexer，右侧索引设为-1
                for j in range(lc):
                    left_indexer[position + j] = left_pos + j
                    right_indexer[position + j] = -1
                position += lc
            elif lc == 0:
                # 如果左侧第i组为空，将右侧第i组的索引填入right_indexer，左侧索引设为-1
                for j in range(rc):
                    left_indexer[position + j] = -1
                    right_indexer[position + j] = right_pos + j
                position += rc
            else:
                # 否则，对于左侧第i组的每个元素，填充左右组的笛卡尔积索引到left_indexer和right_indexer
                for j in range(lc):
                    offset = position + j * rc
                    for k in range(rc):
                        left_indexer[offset + k] = left_pos + j
                        right_indexer[offset + k] = right_pos + k
                position += lc * rc
            left_pos += lc  # 更新左侧位置
            right_pos += rc  # 更新右侧位置

        # 使用_get_result_indexer函数，根据排序后的索引获取最终的结果索引
        _get_result_indexer(left_sorter, left_indexer)
        _get_result_indexer(right_sorter, right_indexer)

    # 返回结果索引数组
    return np.asarray(left_indexer), np.asarray(right_indexer)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _get_result_indexer(intp_t[::1] sorter, intp_t[::1] indexer) noexcept nogil:
    """NOTE: overwrites indexer with the result to avoid allocating another array"""
    cdef:
        Py_ssize_t i, n, idx

    if len(sorter) > 0:
        # Cython版本的 `res = algos.take_nd(sorter, indexer, fill_value=-1)`
        n = indexer.shape[0]
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                indexer[i] = -1
            else:
                indexer[i] = sorter[idx]
    else:
        # 长度为0时的情况，直接将索引数组全部置为-1
        indexer[:] = -1
# 定义一个函数，接受一个名为 `indexer` 的常量整型数组，并返回一个 NumPy 整型数组
def ffill_indexer(const intp_t[:] indexer) -> np.ndarray:
    # 定义变量 i 和 n，分别为索引器的长度和迭代索引
    cdef:
        Py_ssize_t i, n = len(indexer)
        # 定义数组 result，存储整型数据
        ndarray[intp_t] result
        # 定义变量 val 和 last_obs，分别存储当前值和上一个观察到的值
        intp_t val, last_obs

    # 创建一个空的 NumPy 数组 result，用于存储结果
    result = np.empty(n, dtype=np.intp)
    # 初始化 last_obs 为 -1
    last_obs = -1

    # 遍历索引器中的每个元素
    for i in range(n):
        # 获取当前索引处的值
        val = indexer[i]
        # 如果当前值为 -1，则将结果数组中当前位置的值设为上一个观察到的值
        if val == -1:
            result[i] = last_obs
        else:
            # 否则，将结果数组中当前位置的值设为当前值，并更新 last_obs
            result[i] = val
            last_obs = val

    # 返回填充后的结果数组
    return result


# ----------------------------------------------------------------------
# left_join_indexer, inner_join_indexer, outer_join_indexer
# ----------------------------------------------------------------------

# Joins on ordered, unique indices

# right might contain non-unique values

# 使用 Cython 的 wraparound(False) 和 boundscheck(False) 装饰器来优化性能
@cython.wraparound(False)
@cython.boundscheck(False)
def left_join_indexer_unique(
    ndarray[numeric_object_t] left,
    ndarray[numeric_object_t] right
):
    """
    Both left and right are strictly monotonic increasing.
    """
    # 定义变量 i, j, nleft, nright，分别为左右数组的迭代器和长度
    cdef:
        Py_ssize_t i, j, nleft, nright
        # 定义整型数组 indexer，用于存储索引结果
        ndarray[intp_t] indexer
        # 定义变量 rval，存储右数组中的当前值
        numeric_object_t rval

    # 初始化迭代器 i 和 j 为 0，获取左右数组的长度
    i = 0
    j = 0
    nleft = len(left)
    nright = len(right)

    # 创建一个空的 NumPy 整型数组 indexer，用于存储左数组的索引结果
    indexer = np.empty(nleft, dtype=np.intp)
    # 进入循环，开始左右数组的索引匹配
    while True:
        # 如果 i 已经达到左数组的末尾，则跳出循环
        if i == nleft:
            break

        # 如果 j 已经达到右数组的末尾，则将当前左数组位置的索引设为 -1，并继续下一个位置
        if j == nright:
            indexer[i] = -1
            i += 1
            continue

        # 获取右数组中当前位置的值 rval
        rval = right[j]

        # 在左数组中找到第一个与 rval 相等的值的位置
        while i < nleft - 1 and left[i] == rval:
            indexer[i] = j
            i += 1

        # 如果当前左数组位置的值与 rval 相等，则将其索引设为 j，并继续向后找到所有相等的位置
        if left[i] == rval:
            indexer[i] = j
            i += 1
            while i < nleft - 1 and left[i] == rval:
                indexer[i] = j
                i += 1
            j += 1
        # 如果当前左数组位置的值大于 rval，则将其索引设为 -1，并增加 j
        elif left[i] > rval:
            indexer[i] = -1
            j += 1
        # 否则，将其索引设为 -1，并增加 i
        else:
            indexer[i] = -1
            i += 1

    # 返回左数组的索引结果
    return indexer


# 使用 Cython 的 wraparound(False) 和 boundscheck(False) 装饰器来优化性能
@cython.wraparound(False)
@cython.boundscheck(False)
def left_join_indexer(ndarray[numeric_object_t] left, ndarray[numeric_object_t] right):
    """
    Two-pass algorithm for monotonic indexes. Handles many-to-one merges.

    Both left and right are monotonic increasing, but at least one of them
    is non-unique (if both were unique we'd use left_join_indexer_unique).
    """
    # 定义变量 i, j, nright, nleft, count，分别为迭代器和数组长度，以及输出计数
    cdef:
        Py_ssize_t i, j, nright, nleft, count
        # 定义变量 lval, rval，存储左右数组中的当前值
        numeric_object_t lval, rval
        # 定义整型数组 lindexer, rindexer，存储左右数组的索引结果
        ndarray[intp_t] lindexer, rindexer
        # 定义数值对象数组 result，用于存储结果

    # 获取左右数组的长度
    nleft = len(left)
    nright = len(right)

    # 第一遍循环用于计算输出索引器的大小 'count'
    i = 0
    j = 0
    count = 0
    # 如果左侧数组还有剩余元素
    if nleft > 0:
        # 循环直到处理完左侧数组的所有元素
        while i < nleft:
            # 如果右侧数组已经处理完
            if j == nright:
                # 计算剩余未匹配的左侧元素个数，并添加到计数器中
                count += nleft - i
                break

            # 获取当前左右两侧数组的元素值
            lval = left[i]
            rval = right[j]

            # 如果左右两侧元素值相等
            if lval == rval:
                # 执行相同的代码块，适用于 left_join_indexer、inner_join_indexer、outer_join_indexer
                count += 1
                # 检查左侧是否还有相同值的元素
                if i < nleft - 1:
                    # 如果右侧还有未处理的元素且下一个元素与当前右侧元素相等
                    if j < nright - 1 and right[j + 1] == rval:
                        j += 1
                    else:
                        i += 1
                        # 如果下一个左侧元素不等于当前右侧元素，则右侧索引向前移动
                        if left[i] != rval:
                            j += 1
                elif j < nright - 1:
                    j += 1
                    # 如果左侧元素不等于下一个右侧元素，则左侧索引向前移动
                    if lval != right[j]:
                        i += 1
                else:
                    # 已经处理到两个数组的末尾
                    break
            elif lval < rval:
                # 左侧元素不在右侧数组中，添加到计数器并移动左侧索引
                count += 1
                i += 1
            else:
                # 右侧元素不在左侧数组中，右侧索引向前移动
                j += 1

    # 确定结果大小后再次处理

    # 创建左侧索引器、右侧索引器和结果数组
    lindexer = np.empty(count, dtype=np.intp)
    rindexer = np.empty(count, dtype=np.intp)
    result = np.empty(count, dtype=left.dtype)

    # 重置索引和计数器
    i = 0
    j = 0
    count = 0
    if nleft > 0:
        # 再次循环处理左侧数组的元素
        while i < nleft:
            # 如果右侧数组已经处理完
            if j == nright:
                # 将剩余未匹配的左侧元素添加到索引器和结果数组中
                while i < nleft:
                    lindexer[count] = i
                    rindexer[count] = -1
                    result[count] = left[i]
                    i += 1
                    count += 1
                break

            # 获取当前左右两侧数组的元素值
            lval = left[i]
            rval = right[j]

            # 如果左右两侧元素值相等
            if lval == rval:
                # 将匹配的元素索引添加到索引器中，将元素值添加到结果数组中
                lindexer[count] = i
                rindexer[count] = j
                result[count] = lval
                count += 1
                # 检查左侧是否还有相同值的元素
                if i < nleft - 1:
                    # 如果右侧还有未处理的元素且下一个元素与当前右侧元素相等
                    if j < nright - 1 and right[j + 1] == rval:
                        j += 1
                    else:
                        i += 1
                        # 如果下一个左侧元素不等于当前右侧元素，则右侧索引向前移动
                        if left[i] != rval:
                            j += 1
                elif j < nright - 1:
                    j += 1
                    # 如果左侧元素不等于下一个右侧元素，则左侧索引向前移动
                    if lval != right[j]:
                        i += 1
                else:
                    # 已经处理到两个数组的末尾
                    break
            elif lval < rval:
                # 左侧元素不在右侧数组中，添加到索引器并移动左侧索引
                lindexer[count] = i
                rindexer[count] = -1
                result[count] = lval
                count += 1
                i += 1
            else:
                # 右侧元素不在左侧数组中，右侧索引向前移动
                j += 1

    # 返回结果数组、左侧索引器和右侧索引器
    return result, lindexer, rindexer
@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个函数，执行内部连接操作并返回结果索引
def inner_join_indexer(ndarray[numeric_object_t] left, ndarray[numeric_object_t] right):
    """
    Two-pass algorithm for monotonic indexes. Handles many-to-one merges.

    Both left and right are monotonic increasing but not necessarily unique.
    """
    # 声明变量
    cdef:
        Py_ssize_t i, j, nright, nleft, count
        numeric_object_t lval, rval
        ndarray[intp_t] lindexer, rindexer
        ndarray[numeric_object_t] result

    # 获取左右数组的长度
    nleft = len(left)
    nright = len(right)

    # 第一次遍历用于确定输出索引数组的大小 'count'
    i = 0
    j = 0
    count = 0
    if nleft > 0 and nright > 0:
        while True:
            if i == nleft:
                break
            if j == nright:
                break

            # 获取当前位置的值
            lval = left[i]
            rval = right[j]
            # 比较左右值，如果相等则加入结果索引中
            if lval == rval:
                count += 1
                # 判断是否还有相同值，调整索引以保证唯一性
                if i < nleft - 1:
                    if j < nright - 1 and right[j + 1] == rval:
                        j += 1
                    else:
                        i += 1
                        if left[i] != rval:
                            j += 1
                elif j < nright - 1:
                    j += 1
                    if lval != right[j]:
                        i += 1
                else:
                    # 已经遍历到末尾
                    break
            elif lval < rval:
                # 左值不在右边，丢弃，继续下一个左值
                i += 1
            else:
                # 右值不在左边，丢弃，继续下一个右值
                j += 1

    # 根据确定的 'count' 创建索引数组
    lindexer = np.empty(count, dtype=np.intp)
    rindexer = np.empty(count, dtype=np.intp)
    result = np.empty(count, dtype=left.dtype)

    # 第二次遍历以填充索引数组和结果数组
    i = 0
    j = 0
    count = 0
    if nleft > 0 and nright > 0:
        while True:
            if i == nleft:
                break
            if j == nright:
                break

            lval = left[i]
            rval = right[j]
            if lval == rval:
                # 将匹配的索引加入到对应位置
                lindexer[count] = i
                rindexer[count] = j
                result[count] = lval
                count += 1
                if i < nleft - 1:
                    if j < nright - 1 and right[j + 1] == rval:
                        j += 1
                    else:
                        i += 1
                        if left[i] != rval:
                            j += 1
                elif j < nright - 1:
                    j += 1
                    if lval != right[j]:
                        i += 1
                else:
                    # 已经遍历到末尾
                    break
            elif lval < rval:
                # 左值不在右边，丢弃，继续下一个左值
                i += 1
            else:
                # 右值不在左边，丢弃，继续下一个右值
                j += 1

    # 返回结果数组和两个索引数组
    return result, lindexer, rindexer
# 设置 Cython 的 wraparound 为 False，禁用边界检查
# 设置 Cython 的 boundscheck 为 False，禁用越界检查
@cython.wraparound(False)
@cython.boundscheck(False)
def outer_join_indexer(ndarray[numeric_object_t] left, ndarray[numeric_object_t] right):
    """
    Both left and right are monotonic increasing but not necessarily unique.
    左右两个数组都是单调递增的，但不一定是唯一的。
    """
    cdef:
        Py_ssize_t i, j, nright, nleft, count  # 定义 Cython 类型的变量
        numeric_object_t lval, rval  # 定义数值对象的变量
        ndarray[intp_t] lindexer, rindexer  # 定义整数指针数组类型的变量
        ndarray[numeric_object_t] result  # 定义数值对象数组类型的变量

    nleft = len(left)  # 获取左数组的长度
    nright = len(right)  # 获取右数组的长度

    # 第一遍循环是为了找到输出索引数组的大小 'count'
    # count 将等于左数组的长度加上右数组中不在左数组中的元素个数（包括重复的）
    i = 0
    j = 0
    count = 0
    if nleft == 0:
        count = nright
    elif nright == 0:
        count = nleft
    else:
        while True:
            if i == nleft:
                count += nright - j
                break
            if j == nright:
                count += nleft - i
                break

            lval = left[i]
            rval = right[j]
            if lval == rval:
                count += 1
                if i < nleft - 1:
                    if j < nright - 1 and right[j + 1] == rval:
                        j += 1
                    else:
                        i += 1
                        if left[i] != rval:
                            j += 1
                elif j < nright - 1:
                    j += 1
                    if lval != right[j]:
                        i += 1
                else:
                    # end of the road
                    break
            elif lval < rval:
                count += 1
                i += 1
            else:
                count += 1
                j += 1

    lindexer = np.empty(count, dtype=np.intp)  # 创建一个大小为 count 的空整数指针数组
    rindexer = np.empty(count, dtype=np.intp)  # 创建一个大小为 count 的空整数指针数组
    result = np.empty(count, dtype=left.dtype)  # 创建一个大小为 count 的空数组，数据类型与左数组相同

    # 再次进行循环，但是填充索引器和结果数组

    i = 0
    j = 0
    count = 0
    if nleft == 0:
        for j in range(nright):
            lindexer[j] = -1
            rindexer[j] = j
            result[j] = right[j]
    elif nright == 0:
        for i in range(nleft):
            lindexer[i] = i
            rindexer[i] = -1
            result[i] = left[i]
    # 如果条件不满足，执行以下代码块
    else:
        # 无限循环，直到条件改变
        while True:
            # 如果左侧索引与剩余计数相等
            if i == nleft:
                # 在右侧索引未遍历完时执行以下代码块
                while j < nright:
                    # 将左侧索引标记为未匹配
                    lindexer[count] = -1
                    # 记录右侧索引
                    rindexer[count] = j
                    # 将右侧对应元素存入结果
                    result[count] = right[j]
                    # 增加计数器
                    count += 1
                    # 移动右侧索引
                    j += 1
                # 退出循环
                break
            # 如果右侧索引与剩余计数相等
            if j == nright:
                # 在左侧索引未遍历完时执行以下代码块
                while i < nleft:
                    # 记录左侧索引
                    lindexer[count] = i
                    # 将右侧索引标记为未匹配
                    rindexer[count] = -1
                    # 将左侧对应元素存入结果
                    result[count] = left[i]
                    # 增加计数器
                    count += 1
                    # 移动左侧索引
                    i += 1
                # 退出循环
                break

            # 获取当前左右侧索引对应的值
            lval = left[i]
            rval = right[j]

            # 如果左右侧值相等
            if lval == rval:
                # 记录左右侧索引
                lindexer[count] = i
                rindexer[count] = j
                # 将左侧值存入结果
                result[count] = lval
                # 增加计数器
                count += 1
                # 如果左侧索引不是最后一个且右侧下一个值与当前值相等
                if i < nleft - 1:
                    if j < nright - 1 and right[j + 1] == rval:
                        j += 1
                    else:
                        i += 1
                        if left[i] != rval:
                            j += 1
                # 如果右侧索引不是最后一个
                elif j < nright - 1:
                    j += 1
                    if lval != right[j]:
                        i += 1
                else:
                    # 结束循环
                    break
            # 如果左侧值小于右侧值
            elif lval < rval:
                # 记录左侧索引
                lindexer[count] = i
                # 将右侧索引标记为未匹配
                rindexer[count] = -1
                # 将左侧值存入结果
                result[count] = lval
                # 增加计数器
                count += 1
                # 移动左侧索引
                i += 1
            else:
                # 记录左侧索引标记为未匹配
                lindexer[count] = -1
                # 记录右侧索引
                rindexer[count] = j
                # 将右侧值存入结果
                result[count] = rval
                # 增加计数器
                count += 1
                # 移动右侧索引
                j += 1

    # 返回三个结果数组
    return result, lindexer, rindexer
# ----------------------------------------------------------------------
# asof_join_by
# ----------------------------------------------------------------------

# 导入必要的库
from pandas._libs.hashtable cimport Int64HashTable

# 定义函数，通过右侧值进行向后的近似连接
def asof_join_backward_on_X_by_Y(ndarray[numeric_t] left_values,
                                 ndarray[numeric_t] right_values,
                                 const int64_t[:] left_by_values,
                                 const int64_t[:] right_by_values,
                                 bint allow_exact_matches=True,
                                 tolerance=None,
                                 bint use_hashtable=True):

    # 声明变量
    cdef:
        Py_ssize_t left_pos, right_pos, left_size, right_size, found_right_pos
        ndarray[intp_t] left_indexer, right_indexer
        bint has_tolerance = False
        numeric_t tolerance_ = 0
        numeric_t diff = 0
        Int64HashTable hash_table

    # 如果使用容差，设置容差相关的对象
    if tolerance is not None:
        has_tolerance = True
        tolerance_ = tolerance

    # 获取左侧和右侧数组的长度
    left_size = len(left_values)
    right_size = len(right_values)

    # 初始化左侧和右侧的索引数组
    left_indexer = np.empty(left_size, dtype=np.intp)
    right_indexer = np.empty(left_size, dtype=np.intp)

    # 如果使用哈希表，则创建一个 Int64HashTable
    if use_hashtable:
        hash_table = Int64HashTable(right_size)

    # 初始化右侧位置为0，并遍历左侧数组
    right_pos = 0
    for left_pos in range(left_size):
        # 如果右侧位置为负数，则重新设置为0
        if right_pos < 0:
            right_pos = 0

        # 查找右侧最后一个小于或等于左侧值的位置
        if allow_exact_matches:
            while (right_pos < right_size and
                   right_values[right_pos] <= left_values[left_pos]):
                if use_hashtable:
                    hash_table.set_item(right_by_values[right_pos], right_pos)
                right_pos += 1
        else:
            while (right_pos < right_size and
                   right_values[right_pos] < left_values[left_pos]):
                if use_hashtable:
                    hash_table.set_item(right_by_values[right_pos], right_pos)
                right_pos += 1
        right_pos -= 1

        # 将找到的右侧位置保存到索引数组中
        if use_hashtable:
            by_value = left_by_values[left_pos]
            found_right_pos = (hash_table.get_item(by_value)
                               if by_value in hash_table else -1)
        else:
            found_right_pos = right_pos

        left_indexer[left_pos] = left_pos
        right_indexer[left_pos] = found_right_pos

        # 如果需要，验证容差是否满足
        if has_tolerance and found_right_pos != -1:
            diff = left_values[left_pos] - right_values[found_right_pos]
            if diff > tolerance_:
                right_indexer[left_pos] = -1

    # 返回左侧和右侧的索引数组
    return left_indexer, right_indexer
# 定义一个函数，执行基于某个数值数组的条件前向连接操作
def asof_join_forward_on_X_by_Y(ndarray[numeric_t] left_values,
                                ndarray[numeric_t] right_values,
                                const int64_t[:] left_by_values,
                                const int64_t[:] right_by_values,
                                bint allow_exact_matches=1,
                                tolerance=None,
                                bint use_hashtable=True):

    cdef:
        # 定义循环中使用的变量
        Py_ssize_t left_pos, right_pos, left_size, right_size, found_right_pos
        # 用于保存左右索引的数组
        ndarray[intp_t] left_indexer, right_indexer
        # 是否存在容差值和容差的具体数值
        bint has_tolerance = False
        numeric_t tolerance_ = 0
        # 差值计算时使用的变量
        numeric_t diff = 0
        # 整数哈希表对象，用于加速查找
        Int64HashTable hash_table

    # 如果指定了容差值，则设置相关变量
    if tolerance is not None:
        has_tolerance = True
        tolerance_ = tolerance

    # 计算左右数组的长度
    left_size = len(left_values)
    right_size = len(right_values)

    # 初始化左右索引数组，大小为左数组长度
    left_indexer = np.empty(left_size, dtype=np.intp)
    right_indexer = np.empty(left_size, dtype=np.intp)

    # 如果允许使用哈希表，则初始化哈希表对象
    if use_hashtable:
        hash_table = Int64HashTable(right_size)

    # 初始化右侧位置为右数组的最后一个索引
    right_pos = right_size - 1
    # 从左数组的最后一个元素向前遍历
    for left_pos in range(left_size - 1, -1, -1):
        # 如果右侧位置超出了数组范围，则重新设置为最后一个索引
        if right_pos == right_size:
            right_pos = right_size - 1

        # 找到右侧第一个数值大于等于左侧当前位置数值的位置
        if allow_exact_matches:
            while (right_pos >= 0 and
                   right_values[right_pos] >= left_values[left_pos]):
                # 如果使用哈希表，将右侧值与索引存入哈希表
                if use_hashtable:
                    hash_table.set_item(right_by_values[right_pos], right_pos)
                right_pos -= 1
        else:
            while (right_pos >= 0 and
                   right_values[right_pos] > left_values[left_pos]):
                if use_hashtable:
                    hash_table.set_item(right_by_values[right_pos], right_pos)
                right_pos -= 1
        right_pos += 1

        # 获取左侧值对应的右侧索引位置
        if use_hashtable:
            by_value = left_by_values[left_pos]
            found_right_pos = (hash_table.get_item(by_value)
                               if by_value in hash_table else -1)
        else:
            found_right_pos = (right_pos
                               if right_pos != right_size else -1)

        # 将左右索引保存到对应的数组中
        left_indexer[left_pos] = left_pos
        right_indexer[left_pos] = found_right_pos

        # 如果存在容差要求，并且找到了右侧索引位置，则验证容差是否满足
        if has_tolerance and found_right_pos != -1:
            diff = right_values[found_right_pos] - left_values[left_pos]
            if diff > tolerance_:
                right_indexer[left_pos] = -1

    # 返回左右索引数组
    return left_indexer, right_indexer
# 定义一个函数，执行基于 X 和 Y 的最近邻近连接操作
def asof_join_nearest_on_X_by_Y(ndarray[numeric_t] left_values,
                                ndarray[numeric_t] right_values,
                                const int64_t[:] left_by_values,
                                const int64_t[:] right_by_values,
                                bint allow_exact_matches=True,
                                tolerance=None,
                                bint use_hashtable=True):

    # 声明变量用于存储结果索引
    cdef:
        # 用于存储后向搜索结果的数组索引
        ndarray[intp_t] bli, bri
        # 用于存储前向搜索结果的数组索引
        ndarray[intp_t] fli, fri

        # 用于存储左侧数据集索引的数组
        ndarray[intp_t] left_indexer
        # 用于存储右侧数据集索引的数组
        ndarray[intp_t] right_indexer

        # 左侧数据集的大小
        Py_ssize_t left_size
        # 循环变量
        Py_ssize_t i

        # 左侧和右侧时间戳之间的差异，用于比较
        numeric_t bdiff, fdiff

    # 执行向后搜索操作，并获取结果索引
    bli, bri = asof_join_backward_on_X_by_Y(
        left_values,
        right_values,
        left_by_values,
        right_by_values,
        allow_exact_matches,
        tolerance,
        use_hashtable
    )

    # 执行向前搜索操作，并获取结果索引
    fli, fri = asof_join_forward_on_X_by_Y(
        left_values,
        right_values,
        left_by_values,
        right_by_values,
        allow_exact_matches,
        tolerance,
        use_hashtable
    )

    # 初始化左侧数据集的索引数组
    left_size = len(left_values)
    left_indexer = np.empty(left_size, dtype=np.intp)
    # 初始化右侧数据集的索引数组
    right_indexer = np.empty(left_size, dtype=np.intp)

    # 遍历结果索引数组
    for i in range(len(bri)):
        # 选择右侧时间戳差异较小的索引作为最终索引
        if bri[i] != -1 and fri[i] != -1:
            bdiff = left_values[bli[i]] - right_values[bri[i]]
            fdiff = right_values[fri[i]] - left_values[fli[i]]
            right_indexer[i] = bri[i] if bdiff <= fdiff else fri[i]
        else:
            # 如果没有有效索引，则直接使用存在的索引
            right_indexer[i] = bri[i] if bri[i] != -1 else fri[i]
        left_indexer[i] = bli[i]

    # 返回左侧和右侧数据集的最终索引结果
    return left_indexer, right_indexer
```