# `.\numpy\numpy\random\__init__.pxd`

```py
# 导入必要的 Cython 模块和类型定义
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

# 从 numpy/random/bitgen.h 头文件中导入 bitgen 结构体定义
cdef extern from "numpy/random/bitgen.h":
    struct bitgen:
        void *state
        uint64_t (*next_uint64)(void *st) nogil
        uint32_t (*next_uint32)(void *st) nogil
        double (*next_double)(void *st) nogil
        uint64_t (*next_raw)(void *st) nogil

    # 定义 bitgen_t 类型作为 bitgen 结构体的别名
    ctypedef bitgen bitgen_t

# 从 numpy.random.bit_generator 模块中导入 BitGenerator 和 SeedSequence 类
from numpy.random.bit_generator cimport BitGenerator, SeedSequence
```