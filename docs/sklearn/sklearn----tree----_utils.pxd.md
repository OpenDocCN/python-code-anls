# `D:\src\scipysrc\scikit-learn\sklearn\tree\_utils.pxd`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _utils.pyx for details.

# 导入必要的 C 库模块
cimport numpy as cnp
from ._tree cimport Node  # 导入 Node 类型定义
from ..neighbors._quad_tree cimport Cell  # 导入 Cell 类型定义
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint32_t  # 导入各种类型定义

cdef enum:
    # 我们替换 rand_r 函数的最大值设定（见底部说明）。
    # 我们不使用 RAND_MAX，因为它在不同平台上不同，并且在 Windows/MSVC 上特别小。
    # 这对应于 32 位有符号整数的最大可表示值（即 2^31 - 1）。
    RAND_R_MAX = 2147483647


# safe_realloc(&p, n) 重新调整 p 的分配大小为 n * sizeof(*p) 字节，或者抛出 MemoryError 异常。
# 它不会调用 free，因为这是 __dealloc__ 的工作。
#   cdef float32_t *p = NULL
#   safe_realloc(&p, n)
# 相当于带有错误检查的 malloc(n * sizeof(*p))。
ctypedef fused realloc_ptr:
    # 根据需要添加指针类型。
    (float32_t*)
    (intp_t*)
    (unsigned char*)
    (WeightedPQueueRecord*)  # WeightedPQueueRecord 类型的指针
    (float64_t*)
    (float64_t**)
    (Node*)  # Node 类型的指针
    (Cell*)  # Cell 类型的指针
    (Node**)  # Node 类型的指针的指针

# 安全的重新分配函数声明
cdef int safe_realloc(realloc_ptr* p, size_t nelems) except -1 nogil


# 将 sizet_ptr 转换为 ndarray 的 C 函数声明
cdef cnp.ndarray sizet_ptr_to_ndarray(intp_t* data, intp_t size)


# 生成指定范围内随机整数的函数声明
cdef intp_t rand_int(intp_t low, intp_t high,
                     uint32_t* random_state) noexcept nogil


# 生成指定范围内均匀分布的随机数的函数声明
cdef float64_t rand_uniform(float64_t low, float64_t high,
                            uint32_t* random_state) noexcept nogil


# 对数函数的声明
cdef float64_t log(float64_t x) noexcept nogil

# =============================================================================
# WeightedPQueue 数据结构
# =============================================================================

# WeightedPQueue 中存储的记录
cdef struct WeightedPQueueRecord:
    float64_t data
    float64_t weight

# WeightedPQueue 类的声明
cdef class WeightedPQueue:
    cdef intp_t capacity
    cdef intp_t array_ptr
    cdef WeightedPQueueRecord* array_

    # 是否为空的方法声明
    cdef bint is_empty(self) noexcept nogil

    # 重置方法声明
    cdef int reset(self) except -1 nogil

    # 返回大小的方法声明
    cdef intp_t size(self) noexcept nogil

    # 压入数据的方法声明
    cdef int push(self, float64_t data, float64_t weight) except -1 nogil

    # 移除数据的方法声明
    cdef int remove(self, float64_t data, float64_t weight) noexcept nogil

    # 弹出数据的方法声明
    cdef int pop(self, float64_t* data, float64_t* weight) noexcept nogil

    # 查看顶部数据的方法声明
    cdef int peek(self, float64_t* data, float64_t* weight) noexcept nogil

    # 根据索引获取权重的方法声明
    cdef float64_t get_weight_from_index(self, intp_t index) noexcept nogil

    # 根据索引获取值的方法声明
    cdef float64_t get_value_from_index(self, intp_t index) noexcept nogil


# =============================================================================
# WeightedMedianCalculator 数据结构
# =============================================================================

# WeightedMedianCalculator 类的声明
cdef class WeightedMedianCalculator:
    cdef intp_t initial_capacity
    cdef WeightedPQueue samples
    cdef float64_t total_weight
    cdef intp_t k
    cdef float64_t sum_w_0_k  # 表示 sum(weights[0:k]) = w[0] + w[1] + ... + w[k-1]
    # 返回队列中元素的个数，使用Cython语法声明一个返回整数类型的函数，无异常抛出，无需全局解释锁
    cdef intp_t size(self) noexcept nogil
    
    # 将指定的数据和权重推入队列中，使用Cython语法声明一个返回整数类型的函数，如果失败返回-1，无异常抛出，无需全局解释锁
    cdef int push(self, float64_t data, float64_t weight) except -1 nogil
    
    # 重置队列，使用Cython语法声明一个返回整数类型的函数，如果失败返回-1，无异常抛出，无需全局解释锁
    cdef int reset(self) except -1 nogil
    
    # 在推入数据后更新中位数计算所需的参数，使用Cython语法声明一个返回整数类型的函数，无异常抛出，无需全局解释锁
    cdef int update_median_parameters_post_push(
        self, float64_t data, float64_t weight,
        float64_t original_median) noexcept nogil
    
    # 从队列中移除指定的数据和权重，使用Cython语法声明一个返回整数类型的函数，无异常抛出，无需全局解释锁
    cdef int remove(self, float64_t data, float64_t weight) noexcept nogil
    
    # 从队列中弹出数据和权重，使用Cython语法声明一个返回整数类型的函数，无异常抛出，无需全局解释锁
    cdef int pop(self, float64_t* data, float64_t* weight) noexcept nogil
    
    # 在移除数据后更新中位数计算所需的参数，使用Cython语法声明一个返回整数类型的函数，无异常抛出，无需全局解释锁
    cdef int update_median_parameters_post_remove(
        self, float64_t data, float64_t weight,
        float64_t original_median) noexcept nogil
    
    # 获取队列的中位数，使用Cython语法声明一个返回双精度浮点数类型的函数，无异常抛出，无需全局解释锁
    cdef float64_t get_median(self) noexcept nogil
```