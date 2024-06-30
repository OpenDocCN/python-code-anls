# `D:\src\scipysrc\scipy\scipy\signal\_sosfilt.pyx`

```
# 导入 NumPy 库，并使其在 Cython 中可用
cimport numpy as np
# 导入 Cython 功能
cimport cython

# 调用 NumPy 的 import_array() 函数，使 NumPy 数组支持在 Cython 中使用
np.import_array()

# 定义浮点类型的数据类型集合，包括单精度、双精度及其复数形式
ctypedef fused DTYPE_floating_t:
    float
    float complex
    double
    double complex
    long double
    long double complex

# 定义数据类型集合，包括浮点类型和对象类型
ctypedef fused DTYPE_t:
    DTYPE_floating_t
    object

# Cython 3.0 发布后可以使用 nogil 的简化语法，此处暂时需要分别使用 nogil 和 gil 两种情况的循环复制
# 实现 _sosfilt_float 函数，对 sos、x 和 zi 进行滤波操作，无异常抛出，使用 nogil 提升性能
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _sosfilt_float(DTYPE_floating_t [:, ::1] sos,
                         DTYPE_floating_t [:, ::1] x,
                         DTYPE_floating_t [:, :, ::1] zi) noexcept nogil:
    # 在函数中直接修改 x 和 zi 的值
    cdef Py_ssize_t n_signals = x.shape[0]  # 信号数目
    cdef Py_ssize_t n_samples = x.shape[1]  # 每个信号的样本数
    cdef Py_ssize_t n_sections = sos.shape[0]  # SOS 滤波器节段数目
    cdef Py_ssize_t i, n, s
    cdef DTYPE_floating_t x_new, x_cur
    cdef DTYPE_floating_t[:, ::1] zi_slice
    cdef DTYPE_floating_t const_1 = 1.0

    # 使用内存视图来减少数组查找次数，提升效率
    for i in xrange(n_signals):  # 遍历每个信号
        zi_slice = zi[i, :, :]  # 获取当前信号的 zi 数据
        for n in xrange(n_samples):  # 遍历每个样本
            x_cur = const_1 * x[i, n]  # 确保 x_cur 是一个副本

            for s in xrange(n_sections):  # 遍历每个 SOS 滤波器节段
                x_new = sos[s, 0] * x_cur + zi_slice[s, 0]
                zi_slice[s, 0] = (sos[s, 1] * x_cur - sos[s, 4] * x_new
                                  + zi_slice[s, 1])
                zi_slice[s, 1] = sos[s, 2] * x_cur - sos[s, 5] * x_new
                x_cur = x_new

            x[i, n] = x_cur  # 更新原始输入数组的值

# 实现 _sosfilt_object 函数，对 sos、x 和 zi 进行对象类型的滤波操作，无异常抛出
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _sosfilt_object(object [:, ::1] sos,
                    object [:, ::1] x,
                    object [:, :, ::1] zi):
    # 在函数中直接修改 x 和 zi 的值
    cdef Py_ssize_t n_signals = x.shape[0]  # 信号数目
    cdef Py_ssize_t n_samples = x.shape[1]  # 每个信号的样本数
    cdef Py_ssize_t n_sections = sos.shape[0]  # SOS 滤波器节段数目
    cdef Py_ssize_t i, n, s
    cdef object x_n

    for i in xrange(n_signals):  # 遍历每个信号
        for n in xrange(n_samples):  # 遍历每个样本
            for s in xrange(n_sections):  # 遍历每个 SOS 滤波器节段
                x_n = x[i, n]  # 创建临时副本
                # 使用直接 II 转置结构：
                x[i, n] = sos[s, 0] * x_n + zi[i, s, 0]
                zi[i, s, 0] = (sos[s, 1] * x_n - sos[s, 4] * x[i, n] +
                               zi[i, s, 1])
                zi[i, s, 1] = (sos[s, 2] * x_n - sos[s, 5] * x[i, n])

# 实现 _sosfilt 函数，根据输入的数据类型调用相应的函数处理 sos、x 和 zi
def _sosfilt(DTYPE_t [:, ::1] sos,
             DTYPE_t [:, ::1] x,
             DTYPE_t [:, :, ::1] zi):
    if DTYPE_t is object:  # 如果数据类型是对象类型
        _sosfilt_object(sos, x, zi)  # 调用对象类型的滤波函数处理
    else:  # 否则使用非对象类型的快速滤波函数，并在 nogil 环境中执行以提高性能
        with nogil:
            _sosfilt_float(sos, x, zi)
```