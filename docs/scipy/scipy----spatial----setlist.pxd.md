# `D:\src\scipysrc\scipy\scipy\spatial\setlist.pxd`

```
# -*- cython -*-  # 指明这是一个 Cython 文件
"""
List of sets of integers, low-level C implementation

Works similarly as

    setlist = [set() for j in range(n)]

but with integer values.

"""

cimport libc.stdlib  # 导入 Cython 的 C 标准库接口
cimport numpy as np  # 导入 Cython 的 NumPy 接口
import numpy as np  # 导入 NumPy 库

cdef struct setlist_t:  # 定义 C 结构体 setlist_t
    size_t n  # 集合数目
    size_t *sizes  # 每个集合的当前大小
    size_t *alloc_sizes  # 每个集合的分配大小
    int **sets  # 指向每个集合的指针数组

cdef inline int init(setlist_t *setlist, size_t n, size_t size_guess) except -1:
    """
    Initialise a list of `n` sets with a given guessed size
    初始化包含 `n` 个集合的列表，每个集合预先分配给定大小的空间
    """
    cdef int j  # 循环变量

    setlist.n = n  # 设置集合数量

    setlist.sets = <int**>libc.stdlib.malloc(sizeof(int*) * n)  # 分配集合指针数组内存
    if setlist.sets == NULL:  # 内存分配失败处理
        raise MemoryError("Failed to allocate memory in setlist.init()")

    setlist.sizes = <size_t*>libc.stdlib.malloc(sizeof(size_t) * n)  # 分配集合大小数组内存
    if setlist.sizes == NULL:  # 内存分配失败处理
        libc.stdlib.free(setlist.sets)
        raise MemoryError("Failed to allocate memory in setlist.init()")

    setlist.alloc_sizes = <size_t*>libc.stdlib.malloc(sizeof(size_t) * n)  # 分配集合分配大小数组内存
    if setlist.alloc_sizes == NULL:  # 内存分配失败处理
        libc.stdlib.free(setlist.sets)
        libc.stdlib.free(setlist.sizes)
        raise MemoryError("Failed to allocate memory in setlist.init()")

    for j in range(n):
        setlist.sizes[j] = 0  # 初始化每个集合的当前大小为 0
        setlist.alloc_sizes[j] = size_guess  # 设置每个集合的初始分配大小
        setlist.sets[j] = <int*>libc.stdlib.malloc(sizeof(int) * size_guess)  # 分配每个集合的初始空间
        if setlist.sets[j] == NULL:  # 内存分配失败处理
            for i in range(j):
                libc.stdlib.free(setlist.sets[i])
            libc.stdlib.free(setlist.sets)
            libc.stdlib.free(setlist.sizes)
            libc.stdlib.free(setlist.alloc_sizes)
            raise MemoryError("Failed to allocate memory in setlist.init()")

    return 0  # 初始化成功返回 0

cdef inline void free(setlist_t *setlist) noexcept:
    """
    Free the set list
    释放集合列表的内存
    """

    cdef int j  # 循环变量
    for j in range(setlist.n):
        libc.stdlib.free(setlist.sets[j])  # 释放每个集合的内存
    libc.stdlib.free(setlist.sets)  # 释放集合指针数组的内存
    libc.stdlib.free(setlist.sizes)  # 释放集合大小数组的内存
    libc.stdlib.free(setlist.alloc_sizes)  # 释放集合分配大小数组的内存
    setlist.sets = NULL  # 设置集合指针为空
    setlist.sizes = NULL  # 设置集合大小为空
    setlist.alloc_sizes = NULL  # 设置集合分配大小为空
    setlist.n = 0  # 设置集合数量为 0

cdef inline int add(setlist_t *setlist, int n, int value) noexcept nogil:
    """
    Add a value to set `n`
    向集合 `n` 中添加一个值
    """

    cdef size_t i, sz  # 循环变量和集合大小
    cdef int *p  # 指向集合的指针

    if n < 0 or n >= setlist.n:
        return 1  # 如果集合索引无效，返回 1

    for i in range(setlist.sizes[n]):
        if setlist.sets[n][i] == value:
            return 0  # 如果值已经存在于集合中，返回 0

    if setlist.sizes[n] >= setlist.alloc_sizes[n]:
        sz = 2*setlist.alloc_sizes[n] + 1  # 扩展集合空间的新大小
        p = <int*>libc.stdlib.realloc(<void*>setlist.sets[n], sz * sizeof(int))  # 重新分配集合的内存
        if p == NULL:
            return -1  # 内存重新分配失败，返回 -1
        setlist.sets[n] = p  # 更新集合的指针
        setlist.alloc_sizes[n] = sz  # 更新集合的分配大小

    setlist.sets[n][setlist.sizes[n]] = value  # 将值添加到集合中
    setlist.sizes[n] += 1  # 更新集合大小

    return 0  # 添加成功返回 0

cdef inline object tocsr(setlist_t *setlist):
    """
    Convert list of sets to CSR format

    Integers for set `i` reside in data[indptr[i]:indptr[i+1]]

    Returns
    -------
    indptr
        CSR indptr
    data
        CSR data

    """
    # 声明变量 i, j, pos，它们都是 size_t 类型
    cdef size_t i, j, pos
    # 声明变量 total_size，初始化为 0
    cdef size_t total_size
    
    # 声明数组变量 indptr 和 data，它们的类型分别为一维 np.npy_int 数组
    indptr, data
    # 初始化 total_size 为 0
    total_size = 0
    
    # 计算所有集合中元素的总数
    for j in range(setlist.n):
        total_size += setlist.sizes[j]

    # 创建一个长度为 setlist.n+1 的空数组 indptr，元素类型为 np.intc
    indptr = np.empty((setlist.n+1,), dtype=np.intc)
    # 创建一个长度为 total_size 的空数组 data，元素类型为 np.intc
    data = np.empty((total_size,), dtype=np.intc)

    # 初始化 pos 为 0
    pos = 0
    # 遍历每个集合
    for i in range(setlist.n):
        # 设置 indptr[i] 为当前位置 pos
        indptr[i] = pos
        # 遍历集合 i 中的元素
        for j in range(setlist.sizes[i]):
            # 将集合 i 中的元素赋值给 data 中的对应位置 pos
            data[pos] = setlist.sets[i][j]
            # 更新 pos
            pos += 1
    # 设置 indptr[setlist.n] 为最终位置 pos
    indptr[setlist.n] = pos

    # 返回计算得到的 indptr 和 data 数组
    return indptr, data
```