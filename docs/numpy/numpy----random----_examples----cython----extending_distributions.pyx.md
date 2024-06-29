# `D:\src\scipysrc\numpy\numpy\random\_examples\cython\extending_distributions.pyx`

```py
#!/usr/bin/env python3
#cython: language_level=3
"""
This file shows how the to use a BitGenerator to create a distribution.
"""
import numpy as np                     # 导入 NumPy 库
cimport numpy as np                    # 从 Cython 中导入 NumPy 库
cimport cython                         # 导入 Cython 声明
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
                                        # 从 CPython 中导入相关函数
from libc.stdint cimport uint16_t, uint64_t
                                        # 从 C 标准库中导入整数类型
from numpy.random cimport bitgen_t      # 从 NumPy 中导入 bitgen_t 类型
from numpy.random import PCG64          # 从 NumPy 中导入 PCG64 类型
from numpy.random.c_distributions cimport (
      random_standard_uniform_fill, random_standard_uniform_fill_f)
                                        # 从 NumPy 中导入随机分布相关函数

@cython.boundscheck(False)
@cython.wraparound(False)
def uniforms(Py_ssize_t n):
    """
    Create an array of `n` uniformly distributed doubles.
    A 'real' distribution would want to process the values into
    some non-uniform distribution
    """
    cdef Py_ssize_t i                   # 声明一个 C 语言风格的 ssize_t 类型变量 i
    cdef bitgen_t *rng                  # 声明一个指向 bitgen_t 结构体的指针 rng
    cdef const char *capsule_name = "BitGenerator"
                                        # 声明一个指向字符串常量的指针 capsule_name
    cdef double[::1] random_values      # 声明一个连续的内存视图，存储双精度浮点数

    x = PCG64()                         # 创建一个 PCG64 类型的随机数生成器对象 x
    capsule = x.capsule                 # 获取生成器对象的胶囊对象
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
                                        # 检查胶囊对象是否有效，否则抛出异常
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
                                        # 将胶囊对象的指针转换为 bitgen_t 指针
    random_values = np.empty(n, dtype='float64')
                                        # 创建一个空的 NumPy 数组，存储双精度浮点数
    with x.lock, nogil:                 # 进入无 GIL 的并行区域
        for i in range(n):              # 循环 n 次
            # Call the function
            random_values[i] = rng.next_double(rng.state)
                                        # 调用 rng 的 next_double 方法生成随机数
    randoms = np.asarray(random_values) # 将 random_values 转换为 NumPy 数组

    return randoms                      # 返回生成的随机数数组

# cython example 2
@cython.boundscheck(False)
@cython.wraparound(False)
def uint10_uniforms(Py_ssize_t n):
    """Uniform 10 bit integers stored as 16-bit unsigned integers"""
    cdef Py_ssize_t i                   # 声明一个 C 语言风格的 ssize_t 类型变量 i
    cdef bitgen_t *rng                  # 声明一个指向 bitgen_t 结构体的指针 rng
    cdef const char *capsule_name = "BitGenerator"
                                        # 声明一个指向字符串常量的指针 capsule_name
    cdef uint16_t[::1] random_values    # 声明一个连续的内存视图，存储 16 位无符号整数
    cdef int bits_remaining             # 声明一个整数 bits_remaining
    cdef int width = 10                 # 声明一个整数 width 并赋值为 10
    cdef uint64_t buff, mask = 0x3FF    # 声明两个 64 位无符号整数并赋初值

    x = PCG64()                         # 创建一个 PCG64 类型的随机数生成器对象 x
    capsule = x.capsule                 # 获取生成器对象的胶囊对象
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
                                        # 检查胶囊对象是否有效，否则抛出异常
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
                                        # 将胶囊对象的指针转换为 bitgen_t 指针
    random_values = np.empty(n, dtype='uint16')
                                        # 创建一个空的 NumPy 数组，存储 16 位无符号整数
    # Best practice is to release GIL and acquire the lock
    bits_remaining = 0                  # 初始化 bits_remaining 为 0
    with x.lock, nogil:                 # 进入无 GIL 的并行区域
        for i in range(n):              # 循环 n 次
            if bits_remaining < width:  # 如果剩余位数小于 width
                buff = rng.next_uint64(rng.state)
                                        # 调用 rng 的 next_uint64 方法生成 64 位随机数
            random_values[i] = buff & mask
                                        # 将 buff 与 mask 进行按位与操作赋值给 random_values[i]
            buff >>= width              # 右移 buff width 位

    randoms = np.asarray(random_values)
                                        # 将 random_values 转换为 NumPy 数组
    return randoms                      # 返回生成的随机数数组

# cython example 3
def uniforms_ex(bit_generator, Py_ssize_t n, dtype=np.float64):
    """
    Create an array of `n` uniformly distributed doubles via a "fill" function.

    A 'real' distribution would want to process the values into
    some non-uniform distribution

    Parameters
    ----------
    bit_generator: BitGenerator instance
    n: int
        Output vector length
    dtype: {str, dtype}, optional
        Desired dtype, either 'd' (or 'float64') or 'f' (or 'float32'). The
        default dtype value is 'd'
    """
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef np.ndarray randoms

    # 获取保存 BitGenerator 的 PyCapsule 对象
    capsule = bit_generator.capsule
    # 可选：验证该 PyCapsule 对象是否指向一个 BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # 将 PyCapsule 指针转换为 bitgen_t 类型的指针
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    # 根据指定的 dtype 创建一个空的 numpy 数组
    _dtype = np.dtype(dtype)
    randoms = np.empty(n, dtype=_dtype)
    
    # 根据 dtype 类型进行不同的随机数生成操作
    if _dtype == np.float32:
        # 使用互斥锁保护的情况下，调用 C 函数生成 float32 类型的随机数
        with bit_generator.lock:
            random_standard_uniform_fill_f(rng, n, <float*>np.PyArray_DATA(randoms))
    elif _dtype == np.float64:
        # 使用互斥锁保护的情况下，调用 C 函数生成 float64 类型的随机数
        with bit_generator.lock:
            random_standard_uniform_fill(rng, n, <double*>np.PyArray_DATA(randoms))
    else:
        # 若不支持的 dtype 类型，则抛出错误
        raise TypeError('Unsupported dtype %r for random' % _dtype)
    
    # 返回生成的随机数数组
    return randoms
```