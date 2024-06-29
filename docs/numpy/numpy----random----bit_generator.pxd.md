# `D:\src\scipysrc\numpy\numpy\random\bit_generator.pxd`

```py
# 导入必要的Cython模块和类型定义
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

# 从numpy/random/bitgen.h中导入bitgen结构体的定义
cdef extern from "numpy/random/bitgen.h":
    struct bitgen:
        void *state
        uint64_t (*next_uint64)(void *st) nogil
        uint32_t (*next_uint32)(void *st) nogil
        double (*next_double)(void *st) nogil
        uint64_t (*next_raw)(void *st) nogil

    ctypedef bitgen bitgen_t

# 定义一个Cython类BitGenerator，实现了随机数生成器的接口
cdef class BitGenerator():
    cdef readonly object _seed_seq  # 私有成员变量，用于种子序列
    cdef readonly object lock  # 锁对象，用于多线程同步
    cdef bitgen_t _bitgen  # bitgen结构体对象，实际的随机数生成器
    cdef readonly object _ctypes  # ctypes对象，用于C语言类型的转换
    cdef readonly object _cffi  # cffi对象，用于C语言函数接口
    cdef readonly object capsule  # Python中的胶囊对象

# 定义一个Cython类SeedSequence，用于生成种子序列
cdef class SeedSequence():
    cdef readonly object entropy  # 私有成员变量，熵源对象
    cdef readonly tuple spawn_key  # 只读元组，用于生成子序列的键
    cdef readonly Py_ssize_t pool_size  # 只读整数，池大小
    cdef readonly object pool  # 只读对象，池对象
    cdef readonly uint32_t n_children_spawned  # 只读无符号32位整数，生成的子序列数

    # 用于混合熵和熵数组的方法定义
    cdef mix_entropy(self, np.ndarray[np.npy_uint32, ndim=1] mixer,
                     np.ndarray[np.npy_uint32, ndim=1] entropy_array)
    # 获取组装的熵的方法定义
    cdef get_assembled_entropy(self)

# 定义一个简单的Cython类SeedlessSequence，表示没有种子的序列
cdef class SeedlessSequence():
    pass
```