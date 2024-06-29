# `D:\src\scipysrc\numpy\numpy\random\_examples\cython\extending.pyx`

```
# 使用 Python3 解释器执行该脚本
#!/usr/bin/env python3

# 设定 Cython 语言级别为 3
#cython: language_level=3

# 从 libc 库中导入 uint32_t 类型
from libc.stdint cimport uint32_t

# 从 cpython.pycapsule 库中导入 PyCapsule_IsValid 和 PyCapsule_GetPointer 函数
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer

# 导入 NumPy 库，并声明导入的类型和函数
import numpy as np
cimport numpy as np
cimport cython

# 从 numpy.random 库中导入 bitgen_t 类型
from numpy.random cimport bitgen_t

# 导入 PCG64 类型
from numpy.random import PCG64

# 初始化 NumPy 数组支持
np.import_array()

# 使用 Cython 的装饰器声明边界检查关闭和数组溢出检查关闭的函数
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义 uniform_mean 函数，接受一个 Py_ssize_t 类型的参数 n
def uniform_mean(Py_ssize_t n):
    # 声明变量 i 和 rng
    cdef Py_ssize_t i
    cdef bitgen_t *rng

    # 声明并初始化字符串常量 capsule_name
    cdef const char *capsule_name = "BitGenerator"

    # 声明并初始化 double 类型的一维数组 random_values
    cdef double[::1] random_values

    # 声明 NumPy 的 ndarray 对象 randoms
    cdef np.ndarray randoms

    # 创建 PCG64 对象 x
    x = PCG64()

    # 获取 PCG64 对象的胶囊（capsule）
    capsule = x.capsule

    # 检查胶囊的有效性
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")

    # 获取胶囊中的指针，转换为 bitgen_t 指针类型赋值给 rng
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    # 初始化 random_values 数组为长度 n 的空数组
    random_values = np.empty(n)

    # 使用 with 语句获取锁并释放 GIL，以便多线程安全生成随机数
    with x.lock, nogil:
        # 循环生成 n 个随机数，并存储到 random_values 数组中
        for i in range(n):
            random_values[i] = rng.next_double(rng.state)

    # 将 random_values 转换为 NumPy ndarray 对象
    randoms = np.asarray(random_values)

    # 返回随机数的平均值
    return randoms.mean()


# 声明 cdef 函数 bounded_uint，接受 lb, ub, rng 三个参数，并声明为 nogil 函数
cdef uint32_t bounded_uint(uint32_t lb, uint32_t ub, bitgen_t *rng) nogil:
    # 声明变量 mask, delta, val，并初始化 mask 为 ub - lb
    cdef uint32_t mask, delta, val
    mask = delta = ub - lb

    # 计算 mask 的位掩码
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16

    # 生成一个在指定范围内的随机数并返回
    val = rng.next_uint32(rng.state) & mask
    while val > delta:
        val = rng.next_uint32(rng.state) & mask

    return lb + val


# 使用 Cython 的装饰器声明边界检查关闭和数组溢出检查关闭的函数
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义 bounded_uints 函数，接受 lb, ub 和 Py_ssize_t 类型的参数 n
def bounded_uints(uint32_t lb, uint32_t ub, Py_ssize_t n):
    # 声明变量 i 和 rng
    cdef Py_ssize_t i
    cdef bitgen_t *rng

    # 声明并初始化 uint32_t 类型的一维数组 out
    cdef uint32_t[::1] out

    # 声明并初始化字符串常量 capsule_name
    cdef const char *capsule_name = "BitGenerator"

    # 创建 PCG64 对象 x
    x = PCG64()

    # 初始化长度为 n 的 uint32_t 类型的空数组 out
    out = np.empty(n, dtype=np.uint32)

    # 获取 PCG64 对象的胶囊（capsule）
    capsule = x.capsule

    # 检查胶囊的有效性
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")

    # 获取胶囊中的指针，转换为 bitgen_t 指针类型赋值给 rng
    rng = <bitgen_t *>PyCapsule_GetPointer(capsule, capsule_name)

    # 使用 with 语句获取锁并释放 GIL，以便多线程安全生成随机数
    with x.lock, nogil:
        # 循环调用 bounded_uint 函数生成 n 个位于 lb 和 ub 之间的随机数，并存储到 out 数组中
        for i in range(n):
            out[i] = bounded_uint(lb, ub, rng)

    # 将 out 数组转换为 NumPy ndarray 对象并返回
    return np.asarray(out)
```