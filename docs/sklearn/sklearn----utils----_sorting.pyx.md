# `D:\src\scipysrc\scikit-learn\sklearn\utils\_sorting.pyx`

```
# 导入 Cython 中的 floating 类型
from cython cimport floating

# 定义一个内联函数 dual_swap，用于交换 darr 和 iarr 数组中索引 a 和 b 处的值
cdef inline void dual_swap(
    floating* darr,   # 浮点数数组 darr
    intp_t *iarr,     # 整数指针数组 iarr
    intp_t a,         # 要交换的索引 a
    intp_t b,         # 要交换的索引 b
) noexcept nogil:
    """Swap the values at index a and b of both darr and iarr"""
    cdef floating dtmp = darr[a]   # 临时存储 darr[a] 的值
    darr[a] = darr[b]              # 将 darr[b] 的值赋给 darr[a]
    darr[b] = dtmp                 # 将临时存储的值赋给 darr[b]

    cdef intp_t itmp = iarr[a]     # 临时存储 iarr[a] 的值
    iarr[a] = iarr[b]              # 将 iarr[b] 的值赋给 iarr[a]
    iarr[b] = itmp                 # 将临时存储的值赋给 iarr[b]


# 定义一个 C 函数 simultaneous_sort，用于同时对 values 和 indices 数组进行递归快速排序
cdef int simultaneous_sort(
    floating* values,    # 浮点数数组 values
    intp_t* indices,     # 整数指针数组 indices
    intp_t size,         # 数组的大小
) noexcept nogil:
    """
    Perform a recursive quicksort on the values array as to sort them ascendingly.
    This simultaneously performs the swaps on both the values and the indices arrays.

    The numpy equivalent is:

        def simultaneous_sort(dist, idx):
             i = np.argsort(dist)
             return dist[i], idx[i]

    Notes
    -----
    Arrays are manipulated via a pointer to there first element and their size
    as to ease the processing of dynamically allocated buffers.
    """
    # TODO: In order to support discrete distance metrics, we need to have a
    # simultaneous sort which breaks ties on indices when distances are identical.
    # The best might be using a std::stable_sort and a Comparator which might need
    # an Array of Structures (AoS) instead of the Structure of Arrays (SoA)
    # currently used.
    
    cdef:
        intp_t pivot_idx, i, store_idx
        floating pivot_val

    # 对于小数组的情况，进行高效处理
    if size <= 1:
        pass
    elif size == 2:
        if values[0] > values[1]:
            dual_swap(values, indices, 0, 1)
    elif size == 3:
        if values[0] > values[1]:
            dual_swap(values, indices, 0, 1)
        if values[1] > values[2]:
            dual_swap(values, indices, 1, 2)
            if values[0] > values[1]:
                dual_swap(values, indices, 0, 1)
    else:
        # 根据中值法确定枢轴值。
        # 将三个值中最小的移到数组的开头，
        # 中间值（枢轴值）移到末尾，最大值移到枢轴索引处。
        pivot_idx = size // 2
        if values[0] > values[size - 1]:
            dual_swap(values, indices, 0, size - 1)
        if values[size - 1] > values[pivot_idx]:
            dual_swap(values, indices, size - 1, pivot_idx)
            if values[0] > values[size - 1]:
                dual_swap(values, indices, 0, size - 1)
        pivot_val = values[size - 1]

        # 围绕枢轴对索引进行分区。在此操作结束后，
        # pivot_idx 将包含枢轴值，左侧所有元素将小于枢轴值，
        # 右侧所有元素将大于枢轴值。
        store_idx = 0
        for i in range(size - 1):
            if values[i] < pivot_val:
                dual_swap(values, indices, i, store_idx)
                store_idx += 1
        dual_swap(values, indices, store_idx, size - 1)
        pivot_idx = store_idx

        # 递归地对枢轴两侧进行排序
        if pivot_idx > 1:
            simultaneous_sort(values, indices, pivot_idx)
        if pivot_idx + 2 < size:
            simultaneous_sort(values + pivot_idx + 1,
                              indices + pivot_idx + 1,
                              size - pivot_idx - 1)
    return 0
```