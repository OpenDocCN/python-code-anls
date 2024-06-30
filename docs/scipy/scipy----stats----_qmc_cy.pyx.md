# `D:\src\scipysrc\scipy\scipy\stats\_qmc_cy.pyx`

```
# cython: language_level=3
# 设定 Cython 的语言级别为 Python 3
# cython: boundscheck=False
# 禁用 Cython 边界检查优化
# cython: wraparound=False
# 禁用 Cython 索引超界检查优化
# cython: cdivision=True
# 启用 Cython 的 C 语言风格的除法优化
# cython: cpow=True
# 启用 Cython 的 C 语言风格的乘方优化

import numpy as np
# 导入 NumPy 库
cimport numpy as np
# 以 C 的方式导入 NumPy 库，以提高性能
from libc.math cimport fabs, sqrt, pow
# 从 C 标准库中导入 fabs、sqrt、pow 函数，用于数学计算

np.import_array()
# 调用 NumPy 的 import_array 函数，初始化 NumPy 数组接口

cdef extern from "<thread>" namespace "std" nogil:
    # 从 C++ 的 <thread> 头文件中导入 thread 类，使用 nogil 表示不需要 GIL
    cdef cppclass thread:
        thread()
        # thread 类的默认构造函数声明
        void thread[A, B, C, D, E, F, G](A, B, C, D, E, F, G)
        # thread 类的模板化构造函数声明，用于创建线程
        void join()
        # thread 类的 join 方法声明，用于等待线程执行完毕

cdef extern from "<mutex>" namespace "std" nogil:
    # 从 C++ 的 <mutex> 头文件中导入 mutex 类，使用 nogil 表示不需要 GIL
    cdef cppclass mutex:
        void lock()
        # mutex 类的 lock 方法声明，用于锁定互斥量
        void unlock()
        # mutex 类的 unlock 方法声明，用于解锁互斥量

cdef extern from "<functional>" namespace "std" nogil:
    # 从 C++ 的 <functional> 头文件中导入 reference_wrapper 类，使用 nogil 表示不需要 GIL
    cdef cppclass reference_wrapper[T]:
        pass
    cdef reference_wrapper[T] ref[T](T&)
    # reference_wrapper 类的 ref 函数模板声明，用于引用包装对象

from libcpp.vector cimport vector
# 从 libcpp.vector 中导入 vector 类，用于定义 C++ 标准库的 vector 容器

cdef mutex threaded_sum_mutex
# 定义一个名为 threaded_sum_mutex 的互斥量

def _cy_wrapper_centered_discrepancy(const double[:, ::1] sample, bint iterative,
                                     workers):
    # 定义 Cython 包装函数 _cy_wrapper_centered_discrepancy
    # 接受一个双精度浮点型二维数组 sample、一个布尔型 iterative 和一个整数 workers 作为参数
    return centered_discrepancy(sample, iterative, workers)
    # 调用 C 函数 centered_discrepancy，并返回其结果

def _cy_wrapper_wrap_around_discrepancy(const double[:, ::1] sample,
                                        bint iterative, workers):
    # 定义 Cython 包装函数 _cy_wrapper_wrap_around_discrepancy
    # 接受一个双精度浮点型二维数组 sample、一个布尔型 iterative 和一个整数 workers 作为参数
    return wrap_around_discrepancy(sample, iterative, workers)
    # 调用 C 函数 wrap_around_discrepancy，并返回其结果

def _cy_wrapper_mixture_discrepancy(const double[:, ::1] sample,
                                    bint iterative, workers):
    # 定义 Cython 包装函数 _cy_wrapper_mixture_discrepancy
    # 接受一个双精度浮点型二维数组 sample、一个布尔型 iterative 和一个整数 workers 作为参数
    return mixture_discrepancy(sample, iterative, workers)
    # 调用 C 函数 mixture_discrepancy，并返回其结果

def _cy_wrapper_l2_star_discrepancy(const double[:, ::1] sample,
                                    bint iterative, workers):
    # 定义 Cython 包装函数 _cy_wrapper_l2_star_discrepancy
    # 接受一个双精度浮点型二维数组 sample、一个布尔型 iterative 和一个整数 workers 作为参数
    return l2_star_discrepancy(sample, iterative, workers)
    # 调用 C 函数 l2_star_discrepancy，并返回其结果

cdef double centered_discrepancy(const double[:, ::1] sample_view,
                                 bint iterative, unsigned int workers) noexcept nogil:
    # 定义 C 函数 centered_discrepancy
    # 接受一个双精度浮点型二维数组 sample_view、一个布尔型 iterative 和一个无符号整数 workers 作为参数
    cdef:
        Py_ssize_t n = sample_view.shape[0]
        # 定义变量 n，表示 sample_view 的行数
        Py_ssize_t d = sample_view.shape[1]
        # 定义变量 d，表示 sample_view 的列数
        Py_ssize_t i = 0, j = 0
        # 定义变量 i 和 j，用于循环计数
        double prod, disc1 = 0
        # 定义双精度浮点型变量 prod 和 disc1，初始化为 0

    for i in range(n):
        # 循环遍历 n 次，i 从 0 到 n-1
        prod = 1
        # 初始化 prod 为 1
        for j in range(d):
            # 循环遍历 d 次，j 从 0 到 d-1
            prod *= (
                1 + 0.5 * fabs(sample_view[i, j] - 0.5) - 0.5
                * fabs(sample_view[i, j] - 0.5) ** 2
            )
            # 计算 prod 的乘积
        disc1 += prod
        # 累加 prod 到 disc1

    cdef double disc2 = threaded_loops(centered_discrepancy_loop, sample_view,
                                       workers)
    # 定义双精度浮点型变量 disc2，调用 threaded_loops 函数，并传递 centered_discrepancy_loop 和 workers 参数

    if iterative:
        n += 1
        # 如果 iterative 为真，则增加 n 的值

    return ((13.0 / 12.0) ** d - 2.0 / n * disc1
            + 1.0 / (n ** 2) * disc2)
    # 返回计算结果

cdef double centered_discrepancy_loop(const double[:, ::1] sample_view,
                                      Py_ssize_t istart, Py_ssize_t istop) noexcept nogil:
    # 定义 C 函数 centered_discrepancy_loop
    # 接受一个双精度浮点型二维数组 sample_view，以及两个 Py_ssize_t 类型的参数 istart 和 istop
    cdef:
        Py_ssize_t i, j, k
        # 定义变量 i、j 和 k，用于循环计数
        double prod, disc2 = 0
        # 定义双精度浮点型变量 prod 和 disc2，初始化为 0
    # 循环遍历 istart 到 istop 范围内的索引 i
    for i in range(istart, istop):
        # 对 sample_view 的每一行进行遍历，使用 j 作为索引
        for j in range(sample_view.shape[0]):
            # 初始化乘积变量 prod
            prod = 1
            # 对 sample_view 的每一列进行遍历，使用 k 作为索引
            for k in range(sample_view.shape[1]):
                # 计算当前位置元素与 0.5 的绝对差值的一半，以及两个位置元素之间的绝对差值的一半，并计算乘积
                prod *= (
                    1 + 0.5 * fabs(sample_view[i, k] - 0.5)
                    + 0.5 * fabs(sample_view[j, k] - 0.5)
                    - 0.5 * fabs(sample_view[i, k] - sample_view[j, k])
                )
            # 将计算得到的 prod 累加到 disc2 中
            disc2 += prod
    
    # 返回计算得到的 disc2 结果
    return disc2
# 计算环绕差异度量（wrap-around discrepancy）的函数，返回一个双精度浮点数
cdef double wrap_around_discrepancy(const double[:, ::1] sample_view,
                                    bint iterative, unsigned int workers) noexcept nogil:
    # 获取样本视图的行数和列数
    cdef:
        Py_ssize_t n = sample_view.shape[0]
        Py_ssize_t d = sample_view.shape[1]
        double disc

    # 调用多线程循环函数来计算环绕差异
    disc = threaded_loops(wrap_around_loop, sample_view,
                          workers)

    # 如果启用迭代，则增加样本数 n
    if iterative:
        n += 1

    # 返回环绕差异度量的计算结果
    return - (4.0 / 3.0) ** d + 1.0 / (n ** 2) * disc


# 环绕差异度量的内部循环函数，返回一个双精度浮点数
cdef double wrap_around_loop(const double[:, ::1] sample_view,
                             Py_ssize_t istart, Py_ssize_t istop) noexcept nogil:

    # 定义循环中用到的变量
    cdef:
        Py_ssize_t i, j, k
        double prod, disc = 0

    # 循环计算环绕差异
    for i in range(istart, istop):
        for j in range(sample_view.shape[0]):
            prod = 1
            for k in range(sample_view.shape[1]):
                # 计算差值并更新乘积
                x_kikj = fabs(sample_view[i, k] - sample_view[j, k])
                prod *= 3.0 / 2.0 - x_kikj + x_kikj ** 2
            # 更新环绕差异度量
            disc += prod

    # 返回环绕差异度量的累积结果
    return disc


# 混合差异度量的函数，返回一个双精度浮点数
cdef double mixture_discrepancy(const double[:, ::1] sample_view,
                                bint iterative, unsigned int workers) noexcept nogil:
    # 获取样本视图的行数和列数，以及其他初始化变量
    cdef:
        Py_ssize_t n = sample_view.shape[0]
        Py_ssize_t d = sample_view.shape[1]
        Py_ssize_t i = 0, j = 0
        double prod = 1, disc = 0, disc1 = 0

    # 计算混合差异度量的第一个部分
    for i in range(n):
        for j in range(d):
            prod *= (
                5.0 / 3.0 - 0.25 * fabs(sample_view[i, j] - 0.5)
                - 0.25 * fabs(sample_view[i, j] - 0.5) ** 2
            )
        # 累加第一个部分的乘积
        disc1 += prod
        prod = 1

    # 调用多线程循环函数来计算混合差异度量的第二部分
    cdef double disc2 = threaded_loops(mixture_loop, sample_view, workers)

    # 如果启用迭代，则增加样本数 n
    if iterative:
        n += 1

    # 计算混合差异度量的最终结果
    disc = (19.0 / 12.0) ** d
    disc1 = 2.0 / n * disc1
    disc2 = 1.0 / (n ** 2) * disc2

    # 返回混合差异度量的计算结果
    return disc - disc1 + disc2


# 混合差异度量的内部循环函数，返回一个双精度浮点数
cdef double mixture_loop(const double[:, ::1] sample_view, Py_ssize_t istart,
                         Py_ssize_t istop) noexcept nogil:

    # 定义循环中用到的变量
    cdef:
        Py_ssize_t i, j, k
        double prod, disc2 = 0

    # 循环计算混合差异度量
    for i in range(istart, istop):
        for j in range(sample_view.shape[0]):
            prod = 1
            for k in range(sample_view.shape[1]):
                # 计算差值并更新乘积
                prod *= (15.0 / 8.0
                         - 0.25 * fabs(sample_view[i, k] - 0.5)
                         - 0.25 * fabs(sample_view[j, k] - 0.5)
                         - 3.0 / 4.0 * fabs(sample_view[i, k]
                                            - sample_view[j, k])
                         + 0.5
                         * fabs(sample_view[i, k] - sample_view[j, k]) ** 2)
            # 更新混合差异度量
            disc2 += prod

    # 返回混合差异度量的累积结果
    return disc2


# L2* 差异度量的函数，返回一个双精度浮点数
cdef double l2_star_discrepancy(const double[:, ::1] sample_view,
                                bint iterative, unsigned int workers) noexcept nogil:
    # 获取样本视图的行数和列数，以及初始化一些变量
    cdef:
        Py_ssize_t n = sample_view.shape[0]
        Py_ssize_t d = sample_view.shape[1]
        Py_ssize_t i = 0, j = 0
        double prod = 1, disc1 = 0
    # 循环计算在给定的 n 值下的 disc1
    for i in range(n):
        # 对于每个 i，计算在给定的 d 值下的 prod，即累乘 (1 - sample_view[i, j]^2) 的结果
        for j in range(d):
            prod *= 1 - sample_view[i, j] ** 2

        # 将当前的 prod 加到 disc1 中
        disc1 += prod
        # 重置 prod 为 1，为下一次循环准备
        prod = 1

    # 调用 C 函数 threaded_loops 计算 disc2，传入参数为 l2_star_loop、sample_view 和 workers
    cdef double disc2 = threaded_loops(l2_star_loop, sample_view, workers)

    # 如果 iterative 为真，增加 n 的值
    if iterative:
        n += 1

    # 计算 1/n 的值，并将其转换为 double 类型，存储在 one_div_n 中
    cdef double one_div_n = <double> 1 / n

    # 返回最终的结果，计算 sqrt(pow(3, -d) - one_div_n * pow(2, 1 - d) * disc1 + 1 / pow(n, 2) * disc2)
    return sqrt(
        pow(3, -d) - one_div_n * pow(2, 1 - d) * disc1 + 1 / pow(n, 2) * disc2
    )
cdef double l2_star_loop(const double[:, ::1] sample_view, Py_ssize_t istart,
                         Py_ssize_t istop) noexcept nogil:
    # 定义一个 Cython 的函数 l2_star_loop，计算特定范围内的 double 类型结果，不涉及 Python 全局解释器锁（GIL）的操作

    cdef:
        Py_ssize_t i, j, k
        double prod = 1, disc2 = 0, tmp_sum = 0
        # 定义循环中用到的变量：i, j, k，以及计算过程中的累积乘积 prod，累积和 disc2，临时和 tmp_sum

    for i in range(istart, istop):
        # 外层循环，遍历 istart 到 istop 范围内的索引 i
        for j in range(sample_view.shape[0]):
            # 内层循环，遍历 sample_view 的第一维度的索引 j
            prod = 1
            # 初始化 prod 为 1
            for k in range(sample_view.shape[1]):
                # 内部最深层循环，遍历 sample_view 的第二维度的索引 k
                prod *= (
                    1 - max(sample_view[i, k], sample_view[j, k])
                )
                # 计算乘积，使用 max 函数获取 sample_view[i, k] 和 sample_view[j, k] 中的最大值
            tmp_sum += prod
            # 将当前乘积累加到 tmp_sum 中

        disc2 += tmp_sum
        # 将 tmp_sum 的值累加到 disc2 中
        tmp_sum = 0
        # 清空 tmp_sum，为下一次循环做准备

    return disc2
    # 返回计算结果 disc2


def _cy_wrapper_update_discrepancy(const double[::1] x_new_view,
                                   const double[:, ::1] sample_view,
                                   double initial_disc):
    # 定义一个 Python 包装器函数 _cy_wrapper_update_discrepancy，用于调用 C 函数 c_update_discrepancy

    return c_update_discrepancy(x_new_view, sample_view, initial_disc)
    # 调用底层的 C 函数 c_update_discrepancy，并返回其计算结果


cdef double c_update_discrepancy(const double[::1] x_new_view,
                                 const double[:, ::1] sample_view,
                                 double initial_disc) noexcept:
    # 定义一个 Cython 的函数 c_update_discrepancy，计算并返回一个 double 类型的值，不涉及 Python GIL

    cdef:
        Py_ssize_t n = sample_view.shape[0] + 1
        # 计算 sample_view 的第一维度大小加一的值，存储在 n 中
        Py_ssize_t d = sample_view.shape[1]
        # 计算 sample_view 的第二维度大小，存储在 d 中
        Py_ssize_t i = 0, j = 0
        # 定义循环中使用的索引变量 i 和 j
        double prod = 1
        # 初始化一个 double 类型变量 prod 为 1
        double  disc1 = 0, disc2 = 0, disc3 = 0
        # 初始化三个 double 类型变量 disc1, disc2, disc3 为 0
        double[::1] abs_ = np.empty(d, dtype=np.float64)
        # 使用 NumPy 创建一个长度为 d 的 double 类型数组 abs_

    # derivation from P.T. Roy (@tupui)
    # 从 P.T. Roy (@tupui) 推导得来的算法
    for i in range(d):
        # 循环遍历第二维度 d 次
        abs_[i] = fabs(x_new_view[i] - 0.5)
        # 计算 x_new_view[i] 和 0.5 之间的绝对值，存储在 abs_[i] 中
        prod *= (
            1 + 0.5 * abs_[i]
            - 0.5 * pow(abs_[i], 2)
        )
        # 计算乘积，使用 pow 函数计算 abs_[i] 的平方

    disc1 = (- 2 / <double> n) * prod
    # 计算 disc1，使用 C 语法的类型转换

    prod = 1
    # 重置 prod 为 1
    for i in range(n - 1):
        # 循环遍历 n-1 次
        for j in range(d):
            # 内层循环，遍历 d 次
            prod *= (
                1 + 0.5 * abs_[j]
                + 0.5 * fabs(sample_view[i, j] - 0.5)
                - 0.5 * fabs(x_new_view[j] - sample_view[i, j])
            )
            # 计算乘积，涉及 abs_, sample_view, x_new_view

        disc2 += prod
        # 将当前乘积累加到 disc2 中
        prod = 1
        # 重置 prod 为 1

    disc2 *= 2 / pow(n, 2)
    # 计算 disc2，使用 pow 函数计算 n 的平方

    for i in range(d):
        # 循环遍历第二维度 d 次
        prod *= 1 + abs_[i]
        # 计算乘积，涉及 abs_

    disc3 = 1 / pow(n, 2) * prod
    # 计算 disc3，使用 pow 函数计算 n 的平方

    return initial_disc + disc1 + disc2 + disc3
    # 返回计算结果，涉及 initial_disc, disc1, disc2, disc3


ctypedef double (*func_type)(const double[:, ::1], Py_ssize_t,
                             Py_ssize_t) noexcept nogil
# 定义一个函数指针类型 func_type，接受三个参数，返回 double 类型的值，不涉及 Python GIL


cdef double threaded_loops(func_type loop_func,
                           const double[:, ::1] sample_view,
                           unsigned int workers) noexcept nogil:
    # 定义一个 Cython 函数 threaded_loops，使用多线程计算结果，不涉及 Python GIL

    cdef:
        Py_ssize_t n = sample_view.shape[0]
        # 计算 sample_view 的第一维度大小，存储在 n 中
        double disc2 = 0
        # 初始化一个 double 类型变量 disc2 为 0

    if workers <= 1:
        # 如果 workers 小于等于 1
        return loop_func(sample_view, 0, n)
        # 调用 loop_func 函数，计算并返回结果

    cdef:
        vector[thread] threads
        # 定义一个 vector 类型的线程数组 threads
        unsigned int tid
        # 定义一个 unsigned int 类型的线程索引变量 tid
        Py_ssize_t istart, istop
        # 定义循环的起始和终止索引变量 istart 和 istop

    for tid in range(workers):
        # 外层循环，遍历 workers 次
        istart = <Py_ssize_t> (n / workers * tid)
        # 计算 istart 的值
        istop = <Py_ssize_t> (
            n / workers * (tid + 1)) if tid < workers - 1 else n
        # 计算 istop 的值
        threads.push_back(
            thread(one_thread_loop, loop_func, ref(disc2),
                   sample_view, istart, istop, None)
        )
        # 将新线程对象添加到 threads 中，调用 one_thread_loop 函数

    for tid in range(workers):
        # 内层循环，遍历 workers 次
        threads[tid].join()
        # 等待线程 tid 的完成

    return disc2
    # 返回计算结果 disc2
# 定义 Cython 函数，执行单线程循环计算，将结果累加到 disc 中
cdef void one_thread_loop(func_type loop_func,
                          double& disc,
                          const double[:, ::1] sample_view,
                          Py_ssize_t istart,
                          Py_ssize_t istop,
                          _) noexcept nogil:
    # 调用 loop_func 计算 sample_view 在 istart 到 istop 范围内的结果
    cdef double tmp = loop_func(sample_view, istart, istop)

    # 加锁，将 tmp 累加到 disc 中（Cython 问题 #1863 的临时解决方案）
    threaded_sum_mutex.lock()
    (&disc)[0] += tmp
    # 解锁
    threaded_sum_mutex.unlock()


# 定义 Cython 函数，执行 Van der Corput 序列生成（非线程化版本）
def _cy_van_der_corput(Py_ssize_t n,
                       long base,
                       long start_index,
                       unsigned int workers):
    # 初始化长度为 n 的双精度浮点数序列
    sequence = np.zeros(n, dtype=np.double)

    cdef:
        # 使用 Cython 声明数组视图和线程向量
        double[::1] sequence_view = sequence
        vector[thread] threads
        unsigned int tid
        Py_ssize_t istart, istop

    # 如果 worker 数量小于等于 1，则单线程执行
    if workers <= 1:
        _cy_van_der_corput_threaded_loop(0, n, base, start_index,
                                         sequence_view, None)
        return sequence

    # 使用多线程生成 Van der Corput 序列
    for tid in range(workers):
        # 计算每个线程的起始和结束索引
        istart = <Py_ssize_t> (n / workers * tid)
        istop = <Py_ssize_t> (n / workers * (tid + 1)) if tid < workers - 1 else n
        # 启动线程，每个线程执行 _cy_van_der_corput_threaded_loop 函数
        threads.push_back(
            thread(_cy_van_der_corput_threaded_loop, istart, istop, base,
                   start_index, sequence_view, None)
        )

    # 等待所有线程执行完成
    for tid in range(workers):
        threads[tid].join()

    # 返回生成的 Van der Corput 序列
    return sequence


# 定义 Cython 函数，执行 Van der Corput 序列生成（线程化版本）
cdef _cy_van_der_corput_threaded_loop(Py_ssize_t istart,
                                      Py_ssize_t istop,
                                      long base,
                                      long start_index,
                                      double[::1] sequence_view,
                                      _):
    cdef:
        long quotient, remainder
        Py_ssize_t i
        double b2r

    # 遍历指定范围的索引，生成 Van der Corput 序列
    for i in range(istart, istop):
        quotient = start_index + i
        b2r = 1.0 / base
        while quotient > 0:
            remainder = quotient % base
            sequence_view[i] += remainder * b2r
            b2r /= base
            quotient //= base


# 定义 Cython 函数，执行混淆的 Van der Corput 序列生成
def _cy_van_der_corput_scrambled(Py_ssize_t n,
                                 long base,
                                 long start_index,
                                 const np.int64_t[:,::1] permutations,
                                 unsigned int workers):
    # 初始化长度为 n 的双精度浮点数序列
    sequence = np.zeros(n)

    cdef:
        # 使用 Cython 声明数组视图和线程向量
        double[::1] sequence_view = sequence
        vector[thread] threads
        unsigned int tid
        Py_ssize_t istart, istop

    # 如果 worker 数量小于等于 1，则单线程执行
    if workers <= 1:
        _cy_van_der_corput_scrambled_loop(0, n, base, start_index,
                                          permutations, sequence_view)
        return sequence
    # 遍历每个线程的标识符，范围是从 0 到 workers-1
    for tid in range(workers):
        # 计算当前线程开始处理的索引位置
        istart = <Py_ssize_t> (n / workers * tid)
        # 计算当前线程结束处理的索引位置
        istop = <Py_ssize_t> (
                n / workers * (tid + 1)) if tid < workers - 1 else n
        # 将一个新线程对象加入到线程列表中，用于执行指定的函数和参数
        threads.push_back(
            thread(_cy_van_der_corput_scrambled_loop, istart, istop, base,
                   start_index, permutations, sequence_view)
        )

    # 等待每个线程执行完毕
    for tid in range(workers):
        threads[tid].join()

    # 返回生成的序列
    return sequence
# 定义一个 Cython（Cython 是用于编写 C 扩展的 Python 风格语法的编译器）函数，实现 Van der Corput 序列的混淆生成
cdef _cy_van_der_corput_scrambled_loop(Py_ssize_t istart,
                                       Py_ssize_t istop,
                                       long base,
                                       long start_index,
                                       const np.int64_t[:,::1] permutations,
                                       double[::1] sequence_view):
    # 定义循环中的变量
    cdef:
        long i, j, quotient, remainder  # 定义整型变量和余数
        double b2r  # 定义双精度浮点数变量

    # 外层循环，遍历指定范围的索引
    for i in range(istart, istop):
        quotient = start_index + i  # 计算当前迭代的商
        b2r = 1.0 / base  # 计算当前迭代的基数的倒数
        # 内层循环，遍历置换数组的行数
        for j in range(permutations.shape[0]):
            remainder = quotient % base  # 计算当前迭代的余数
            remainder = permutations[j, remainder]  # 使用置换数组进行置换
            sequence_view[i] += remainder * b2r  # 更新序列视图中的值
            b2r /= base  # 更新基数的倒数
            quotient //= base  # 更新商
```