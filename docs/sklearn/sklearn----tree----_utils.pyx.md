# `D:\src\scipysrc\scikit-learn\sklearn\tree\_utils.pyx`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入 C 标准库中的部分函数和变量
from libc.stdlib cimport free
from libc.stdlib cimport realloc
from libc.math cimport log as ln
from libc.math cimport isnan

# 导入 NumPy 库并使用 C 接口
import numpy as np
cimport numpy as cnp
cnp.import_array()

# 导入本地的随机数生成函数
from ..utils._random cimport our_rand_r

# =============================================================================
# 辅助函数
# =============================================================================

# 定义一个安全的重新分配内存的函数
cdef int safe_realloc(realloc_ptr* p, size_t nelems) except -1 nogil:
    # 计算所需的内存字节数
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    # 检查是否有溢出
    if nbytes / sizeof(p[0][0]) != nelems:
        # 内存分配失败，抛出内存错误
        raise MemoryError(f"could not allocate ({nelems} * {sizeof(p[0][0])}) bytes")

    # 使用 realloc 分配内存
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        # 内存分配失败，抛出内存错误
        raise MemoryError(f"could not allocate {nbytes} bytes")

    # 更新指针指向新分配的内存区域
    p[0] = tmp
    return 0


# 用于测试的辅助函数，尝试分配一个绝对会溢出的内存量
def _realloc_test():
    cdef intp_t* p = NULL
    safe_realloc(&p, <size_t>(-1) / 2)
    if p != NULL:
        free(p)
        assert False


# 将一个 intp_t 类型的指针转换为 NumPy 一维数组
cdef inline cnp.ndarray sizet_ptr_to_ndarray(intp_t* data, intp_t size):
    """Return copied data as 1D numpy array of intp's."""
    # 创建数组的形状
    cdef cnp.npy_intp shape[1]
    shape[0] = <cnp.npy_intp> size
    # 使用给定的数据和形状创建 NumPy 数组，并进行复制
    return cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_INTP, data).copy()


# 生成一个在指定范围内的随机整数
cdef inline intp_t rand_int(intp_t low, intp_t high,
                            uint32_t* random_state) noexcept nogil:
    """Generate a random integer in [low; end)."""
    return low + our_rand_r(random_state) % (high - low)


# 生成一个在指定范围内的随机浮点数
cdef inline float64_t rand_uniform(float64_t low, float64_t high,
                                   uint32_t* random_state) noexcept nogil:
    """Generate a random float64_t in [low; high)."""
    # 使用线性变换将随机数映射到指定范围
    return ((high - low) * <float64_t> our_rand_r(random_state) /
            <float64_t> RAND_R_MAX) + low


# 重定义 log 函数，将其作为对数函数基底为 2 的版本实现
cdef inline float64_t log(float64_t x) noexcept nogil:
    return ln(x) / ln(2.0)

# =============================================================================
# WeightedPQueue 数据结构
# =============================================================================

# 定义一个加权优先队列类
cdef class WeightedPQueue:
    """A priority queue class, always sorted in increasing order.

    Attributes
    ----------
    capacity : intp_t
        The capacity of the priority queue.

    array_ptr : intp_t
        The water mark of the priority queue; the priority queue grows from
        left to right in the array ``array_``. ``array_ptr`` is always
        less than ``capacity``.
    """
    array_ : WeightedPQueueRecord*
        The array of priority queue records. The minimum element is on the
        left at index 0, and the maximum element is on the right at index
        ``array_ptr-1``.
    """
    # 声明一个指向WeightedPQueueRecord类型的指针array_，
    # 用于存储优先队列记录的数组。最小元素位于索引0处，
    # 最大元素位于索引array_ptr-1处。

    def __cinit__(self, intp_t capacity):
        self.capacity = capacity
        self.array_ptr = 0
        # 使用safe_realloc函数为array_分配内存，大小为capacity
        safe_realloc(&self.array_, capacity)

    def __dealloc__(self):
        # 释放array_指向的内存空间
        free(self.array_)

    cdef int reset(self) except -1 nogil:
        """Reset the WeightedPQueue to its state at construction

        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.array_ptr = 0
        # 重新调整array_的内存空间大小为capacity，
        # 如果内存分配失败则抛出MemoryError并返回-1
        safe_realloc(&self.array_, self.capacity)
        return 0

    cdef bint is_empty(self) noexcept nogil:
        # 检查队列是否为空，如果array_ptr小于等于0则为空
        return self.array_ptr <= 0

    cdef intp_t size(self) noexcept nogil:
        # 返回当前队列中元素的个数，即array_ptr的值
        return self.array_ptr

    cdef int push(self, float64_t data, float64_t weight) except -1 nogil:
        """Push record on the array.

        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef intp_t array_ptr = self.array_ptr
        cdef WeightedPQueueRecord* array = NULL
        cdef intp_t i

        # 如果当前数组元素个数已经超过容量，则扩展容量为原来的两倍
        if array_ptr >= self.capacity:
            self.capacity *= 2
            # 使用safe_realloc函数重新分配array_的内存空间
            safe_realloc(&self.array_, self.capacity)

        # 将数据data和权重weight作为新记录插入数组末尾
        array = self.array_
        array[array_ptr].data = data
        array[array_ptr].weight = weight

        # 将插入的元素依次向上冒泡，直到数组按升序排列
        i = array_ptr
        while(i != 0 and array[i].data < array[i-1].data):
            array[i], array[i-1] = array[i-1], array[i]
            i -= 1

        # 增加数组元素计数器
        self.array_ptr = array_ptr + 1
        return 0

    cdef int remove(self, float64_t data, float64_t weight) noexcept nogil:
        """Remove a specific value/weight record from the array.
        Returns 0 if successful, -1 if record not found."""
        cdef intp_t array_ptr = self.array_ptr
        cdef WeightedPQueueRecord* array = self.array_
        cdef intp_t idx_to_remove = -1
        cdef intp_t i

        if array_ptr <= 0:
            return -1

        # 查找要移除的元素
        for i in range(array_ptr):
            if array[i].data == data and array[i].weight == weight:
                idx_to_remove = i
                break

        if idx_to_remove == -1:
            return -1

        # 将被移除元素后面的元素向左移动一个位置
        for i in range(idx_to_remove, array_ptr-1):
            array[i] = array[i+1]

        # 减少数组元素计数器
        self.array_ptr = array_ptr - 1
        return 0
    # 定义一个函数 pop，从优先级队列中移除顶部（最小值）的元素。
    # 如果成功移除返回 0，如果队列为空返回 -1。
    cdef int pop(self, float64_t* data, float64_t* weight) noexcept nogil:
        """Remove the top (minimum) element from array.
        Returns 0 if successful, -1 if nothing to remove."""
        # 获取队列当前元素个数的指针
        cdef intp_t array_ptr = self.array_ptr
        # 获取队列数组的指针
        cdef WeightedPQueueRecord* array = self.array_
        # 定义循环变量 i

        # 如果队列为空，则返回 -1
        if array_ptr <= 0:
            return -1

        # 将队列中第一个元素的数据和权重赋值给传入的 data 和 weight 数组
        data[0] = array[0].data
        weight[0] = array[0].weight

        # 将被移除的元素后面的元素依次向左移动一个位置
        for i in range(0, array_ptr-1):
            array[i] = array[i+1]

        # 更新队列元素个数
        self.array_ptr = array_ptr - 1
        return 0

    # 定义一个函数 peek，将队列顶部元素的数据和权重写入指针中。
    # 如果成功写入返回 0，如果队列为空返回 -1。
    cdef int peek(self, float64_t* data, float64_t* weight) noexcept nogil:
        """Write the top element from array to a pointer.
        Returns 0 if successful, -1 if nothing to write."""
        # 获取队列数组的指针
        cdef WeightedPQueueRecord* array = self.array_
        # 如果队列为空，则返回 -1
        if self.array_ptr <= 0:
            return -1
        # 将队列中第一个元素的数据和权重赋值给传入的 data 和 weight 数组
        data[0] = array[0].data
        weight[0] = array[0].weight
        return 0

    # 定义一个函数 get_weight_from_index，根据索引获取指定位置的元素的权重。
    # 索引范围为 [0, self.current_capacity]。
    cdef float64_t get_weight_from_index(self, intp_t index) noexcept nogil:
        """Given an index between [0,self.current_capacity], access
        the appropriate heap and return the requested weight"""
        # 获取队列数组的指针
        cdef WeightedPQueueRecord* array = self.array_

        # 返回指定索引位置元素的权重
        return array[index].weight

    # 定义一个函数 get_value_from_index，根据索引获取指定位置的元素的数据值。
    # 索引范围为 [0, self.current_capacity]。
    cdef float64_t get_value_from_index(self, intp_t index) noexcept nogil:
        """Given an index between [0,self.current_capacity], access
        the appropriate heap and return the requested value"""
        # 获取队列数组的指针
        cdef WeightedPQueueRecord* array = self.array_

        # 返回指定索引位置元素的数据值
        return array[index].data
# =============================================================================
# WeightedMedianCalculator data structure
# =============================================================================

cdef class WeightedMedianCalculator:
    """A class to handle calculation of the weighted median from streams of
    data. To do so, it maintains a parameter ``k`` such that the sum of the
    weights in the range [0,k) is greater than or equal to half of the total
    weight. By minimizing the value of ``k`` that fulfills this constraint,
    calculating the median is done by either taking the value of the sample
    at index ``k-1`` of ``samples`` (samples[k-1].data) or the average of
    the samples at index ``k-1`` and ``k`` of ``samples``
    ((samples[k-1] + samples[k]) / 2).

    Attributes
    ----------
    initial_capacity : intp_t
        The initial capacity of the WeightedMedianCalculator.

    samples : WeightedPQueue
        Holds the samples (consisting of values and their weights) used in the
        weighted median calculation.

    total_weight : float64_t
        The sum of the weights of items in ``samples``. Represents the total
        weight of all samples used in the median calculation.

    k : intp_t
        Index used to calculate the median.

    sum_w_0_k : float64_t
        The sum of the weights from samples[0:k]. Used in the weighted
        median calculation; minimizing the value of ``k`` such that
        ``sum_w_0_k`` >= ``total_weight / 2`` provides a mechanism for
        calculating the median in constant time.

    """

    def __cinit__(self, intp_t initial_capacity):
        # 初始化方法，设置初始容量和各个属性的初始值
        self.initial_capacity = initial_capacity
        self.samples = WeightedPQueue(initial_capacity)  # 使用初始容量创建带权重优先队列
        self.total_weight = 0  # 初始化总权重为0
        self.k = 0  # 初始化中位数计算所需的索引k为0
        self.sum_w_0_k = 0  # 初始化sum_w_0_k为0，用于中位数计算

    cdef intp_t size(self) noexcept nogil:
        """Return the number of samples in the
        WeightedMedianCalculator"""
        # 返回当前样本数量
        return self.samples.size()

    cdef int reset(self) except -1 nogil:
        """Reset the WeightedMedianCalculator to its state at construction

        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # 重置WeightedMedianCalculator对象到构造状态
        # 调用WeightedPQueue的reset方法清空队列
        self.samples.reset()
        self.total_weight = 0  # 重置总权重为0
        self.k = 0  # 重置中位数计算所需的索引k为0
        self.sum_w_0_k = 0  # 重置sum_w_0_k为0
        return 0
    # 定义一个用于向 WeightedMedianCalculator 中推送值和其权重的函数
    cdef int push(self, float64_t data, float64_t weight) except -1 nogil:
        """Push a value and its associated weight to the WeightedMedianCalculator

        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef int return_value  # 声明返回值变量
        cdef float64_t original_median = 0.0  # 声明原始中位数变量并初始化为0.0

        if self.size() != 0:
            original_median = self.get_median()  # 获取当前的中位数

        # 调用 samples.push 方法，将数据和权重推送到样本堆中
        # 使用 except -1 表示在分配内存失败时返回 -1
        return_value = self.samples.push(data, weight)

        # 在推送后更新中位数计算所需的参数
        self.update_median_parameters_post_push(data, weight, original_median)

        return return_value  # 返回推送操作的结果值

    # 更新推送操作后的中位数参数
    cdef int update_median_parameters_post_push(
            self, float64_t data, float64_t weight,
            float64_t original_median) noexcept nogil:
        """Update the parameters used in the median calculation,
        namely `k` and `sum_w_0_k` after an insertion"""

        # 特殊情况：插入第一个元素
        if self.size() == 1:
            self.k = 1  # 设置 k 为 1
            self.total_weight = weight  # 设置总权重为当前数据的权重
            self.sum_w_0_k = self.total_weight  # 设置初始累积权重为当前数据的权重
            return 0

        # 更新总权重
        self.total_weight += weight

        if data < original_median:
            # 如果插入的数据小于原始中位数
            self.k += 1  # 增加 k 值
            self.sum_w_0_k += weight  # 更新 sum_w_0_k 加上插入数据的权重

            # 调整 k 的最小值，使得 sum(W[0:k]) >= total_weight / 2
            # k 的最小值为 1
            while (self.k > 1 and ((self.sum_w_0_k -
                                   self.samples.get_weight_from_index(self.k-1))
                                  >= self.total_weight / 2.0)):
                self.k -= 1
                self.sum_w_0_k -= self.samples.get_weight_from_index(self.k)

            return 0

        if data >= original_median:
            # 如果插入的数据大于等于原始中位数
            # 调整 k 的最小值，使得 sum(W[0:k]) >= total_weight / 2
            while (self.k < self.samples.size() and
                   (self.sum_w_0_k < self.total_weight / 2.0)):
                self.k += 1
                self.sum_w_0_k += self.samples.get_weight_from_index(self.k-1)

            return 0

    # 从 MedianHeap 中移除一个值，不再参与中位数的计算
    cdef int remove(self, float64_t data, float64_t weight) noexcept nogil:
        """Remove a value from the MedianHeap, removing it
        from consideration in the median calculation
        """
        cdef int return_value  # 声明返回值变量
        cdef float64_t original_median = 0.0  # 声明原始中位数变量并初始化为0.0

        if self.size() != 0:
            original_median = self.get_median()  # 获取当前的中位数

        # 调用 samples.remove 方法，从样本堆中移除数据和权重
        return_value = self.samples.remove(data, weight)

        # 在移除数据后更新中位数计算所需的参数
        self.update_median_parameters_post_remove(data, weight, original_median)

        return return_value  # 返回移除操作的结果值
    # 定义一个 Cython 的 cdef 函数，用于从 MedianHeap 中弹出一个值
    # data 和 weight 是传入的 float64_t 类型指针参数，用于返回弹出的数据和权重
    # noexcept 和 nogil 表示此函数是无异常、无全局解锁的优化声明
    cdef int pop(self, float64_t* data, float64_t* weight) noexcept nogil:
        """Pop a value from the MedianHeap, starting from the
        left and moving to the right.
        """
        # 定义返回值
        cdef int return_value
        # 原始中位数初始化为 0.0
        cdef float64_t original_median = 0.0

        # 如果 MedianHeap 非空，获取当前的中位数
        if self.size() != 0:
            original_median = self.get_median()

        # 如果 samples 中没有元素，则无法弹出，返回 -1
        # 没有元素可弹出
        if self.samples.size() == 0:
            return -1

        # 调用 samples 对象的 pop 方法弹出一个元素，并获取返回值
        return_value = self.samples.pop(data, weight)
        # 调用 update_median_parameters_post_remove 方法更新删除元素后的中位数参数
        self.update_median_parameters_post_remove(data[0],
                                                  weight[0],
                                                  original_median)
        return return_value

    # 更新删除元素后的中位数计算参数
    # data: 删除的数据值
    # weight: 删除的数据权重
    # original_median: 原始的中位数值
    cdef int update_median_parameters_post_remove(
            self, float64_t data, float64_t weight,
            float64_t original_median) noexcept nogil:
        """Update the parameters used in the median calculation,
        namely `k` and `sum_w_0_k` after a removal"""
        # 如果 samples 中没有元素，则重置所有参数并返回 0
        if self.samples.size() == 0:
            self.k = 0
            self.total_weight = 0
            self.sum_w_0_k = 0
            return 0

        # 如果 samples 中只有一个元素，则设置 k=1，并更新 total_weight 和 sum_w_0_k
        if self.samples.size() == 1:
            self.k = 1
            self.total_weight -= weight
            self.sum_w_0_k = self.total_weight
            return 0

        # 减去删除元素的权重
        self.total_weight -= weight

        # 如果删除的数据小于原始中位数
        if data < original_median:
            # 删除的数据在中位数以下，所以减少 k
            self.k -= 1
            # 更新 sum_w_0_k，减去删除的权重
            self.sum_w_0_k -= weight

            # 调整 k，使得 sum(W[0:k]) >= total_weight / 2
            # 通过增加 k 并相应更新 sum_w_0_k，直到满足条件为止
            while(self.k < self.samples.size() and
                  (self.sum_w_0_k < self.total_weight / 2.0)):
                self.k += 1
                self.sum_w_0_k += self.samples.get_weight_from_index(self.k-1)
            return 0

        # 如果删除的数据大于等于原始中位数
        if data >= original_median:
            # 删除的数据在中位数以上
            # 调整 k，使得 sum(W[0:k]) >= total_weight / 2
            while(self.k > 1 and ((self.sum_w_0_k -
                                   self.samples.get_weight_from_index(self.k-1))
                                  >= self.total_weight / 2.0)):
                self.k -= 1
                self.sum_w_0_k -= self.samples.get_weight_from_index(self.k)
            return 0
    # 定义一个 C++ 方法，计算中位数并写入指针，考虑样本权重，不抛出异常，无需全局解锁
    cdef float64_t get_median(self) noexcept nogil:
        """Write the median to a pointer, taking into account
        sample weights."""
        # 如果前半部分样本权重等于总权重的一半，返回分割中位数
        if self.sum_w_0_k == (self.total_weight / 2.0):
            # split median
            return (self.samples.get_value_from_index(self.k) +
                    self.samples.get_value_from_index(self.k-1)) / 2.0
        # 如果前半部分样本权重大于总权重的一半，返回整体中位数
        if self.sum_w_0_k > (self.total_weight / 2.0):
            # whole median
            return self.samples.get_value_from_index(self.k-1)
# 定义一个函数，用于检查给定二维数组 X 每列是否包含 NaN 值，返回结果数组
def _any_isnan_axis0(const float32_t[:, :] X):
    """Same as np.any(np.isnan(X), axis=0)"""
    # 声明变量 i, j 作为循环索引
    cdef:
        intp_t i, j
        # 获取数组 X 的行数和列数
        intp_t n_samples = X.shape[0]
        intp_t n_features = X.shape[1]
        # 创建一个布尔类型的数组 isnan_out，用于存储每列是否有 NaN 的信息
        unsigned char[::1] isnan_out = np.zeros(X.shape[1], dtype=np.bool_)

    # 使用 nogil 上下文进行并行化处理
    with nogil:
        # 遍历 X 的每一行
        for i in range(n_samples):
            # 遍历 X 的每一列
            for j in range(n_features):
                # 如果当前列 j 已经检测到 NaN，则跳过
                if isnan_out[j]:
                    continue
                # 检查 X[i, j] 是否为 NaN
                if isnan(X[i, j]):
                    # 如果是 NaN，则将 isnan_out[j] 置为 True，并终止当前列的检查
                    isnan_out[j] = True
                    break
    # 将 isnan_out 转换为 NumPy 数组并返回
    return np.asarray(isnan_out)
```