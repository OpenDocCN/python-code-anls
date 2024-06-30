# `D:\src\scipysrc\scipy\scipy\stats\_rcont\rcont.pyx`

```
cimport numpy as np
import numpy as np

np.import_array()

from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
from libc.stdint cimport uint32_t, uint64_t, int64_t

# 定义整型数组类型
ctypedef int64_t tab_t

# 引入外部头文件中的结构体定义和函数声明
cdef extern from "./_rcont.h":
    ctypedef struct bitgen_t:
        void *state
        uint64_t (*next_uint64)(void *st) nogil
        uint32_t (*next_uint32)(void *st) nogil
        double (*next_double)(void *st) nogil
        uint64_t (*next_raw)(void *st) nogil

    void rcont1_init(tab_t*, int, const tab_t*)
    void rcont1(tab_t*, int, const tab_t*, int, const tab_t*,
                tab_t, tab_t*, bitgen_t*)
    void rcont2(tab_t*, int, const tab_t*, int, const tab_t*,
                tab_t, bitgen_t*)

# 根据给定的随机状态对象获取随机数生成器结构体指针
cdef bitgen_t* get_bitgen(random_state):
    if isinstance(random_state, np.random.RandomState):
        bg = random_state._bit_generator
    elif isinstance(random_state, np.random.Generator):
        bg = random_state.bit_generator
    else:
        raise ValueError('random_state is not RandomState or Generator')
    capsule = bg.capsule

    cdef:
        const char *capsule_name = "BitGenerator"

    # 检查 PyCapsule 是否有效
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("invalid pointer to anon_func_state")

    # 获取 PyCapsule 指向的结构体指针
    return <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)


# 从给定的行、列和总数生成随机变量，使用 rcont1 算法
def rvs_rcont1(const tab_t[::1] row, const tab_t[::1] col, tab_t ntot,
               int size, random_state):

    cdef:
        bitgen_t *rstate = get_bitgen(random_state)
        int nr = row.shape[0]  # 获取行数
        int nc = col.shape[0]  # 获取列数

    # 初始化结果数组，全零
    cdef np.ndarray[tab_t, ndim=3, mode="c"] result = np.zeros(
        (size, nr, nc), dtype=np.int64
    )

    # 创建临时工作数组，指定大小和数据类型
    cdef np.ndarray[tab_t, ndim=1, mode="c"] work = np.empty(
        ntot, dtype=np.int64
    )

    # 如果行数、列数或总数为零，直接返回全零的结果数组
    if nc == 0 or nr == 0 or ntot == 0:
        return result

    # 初始化 rcont1 算法所需的工作数组
    rcont1_init(&work[0], nc, &col[0])

    # 循环生成随机变量，使用 rcont1 算法填充结果数组
    for i in range(size):
        rcont1(&result[i, 0, 0], nr, &row[0], nc, &col[0], ntot,
               &work[0], rstate)

    # 返回生成的随机变量结果数组
    return result


# 从给定的行、列和总数生成随机变量，使用 rcont2 算法
def rvs_rcont2(const tab_t[::1] row, const tab_t[::1] col, tab_t ntot,
               int size, random_state):
    cdef:
        bitgen_t *rstate = get_bitgen(random_state)
        int nr = row.shape[0]  # 获取行数
        int nc = col.shape[0]  # 获取列数

    # 初始化结果数组，全零
    cdef np.ndarray[tab_t, ndim=3, mode="c"] result = np.zeros(
        (size, nr, nc), dtype=np.int64
    )

    # 如果行数、列数或总数为零，直接返回全零的结果数组
    if nc == 0 or nr == 0 or ntot == 0:
        return result

    # 循环生成随机变量，使用 rcont2 算法填充结果数组
    for i in range(size):
        rcont2(&result[i, 0, 0], nr, &row[0], nc, &col[0], ntot,
               rstate)

    # 返回生成的随机变量结果数组
    return result
```