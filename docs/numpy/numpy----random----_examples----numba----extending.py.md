# `.\numpy\numpy\random\_examples\numba\extending.py`

```
# 导入必要的库：numpy 和 numba
import numpy as np
import numba as nb

# 从 numpy.random 中导入 PCG64 随机数生成器
from numpy.random import PCG64
# 从 timeit 中导入 timeit 函数
from timeit import timeit

# 创建 PCG64 随机数生成器的实例
bit_gen = PCG64()
# 获取 PCG64 随机数生成器的下一个双精度随机数方法
next_d = bit_gen.cffi.next_double
# 获取 PCG64 随机数生成器的状态地址方法
state_addr = bit_gen.cffi.state_address

# 定义生成正态分布随机数的函数，使用 Numba 进行加速编译
def normals(n, state):
    out = np.empty(n)
    for i in range((n + 1) // 2):
        # 生成两个均匀分布随机数，并转换为标准正态分布随机数
        x1 = 2.0 * next_d(state) - 1.0
        x2 = 2.0 * next_d(state) - 1.0
        r2 = x1 * x1 + x2 * x2
        # 如果随机数不在单位圆内则重新生成
        while r2 >= 1.0 or r2 == 0.0:
            x1 = 2.0 * next_d(state) - 1.0
            x2 = 2.0 * next_d(state) - 1.0
            r2 = x1 * x1 + x2 * x2
        f = np.sqrt(-2.0 * np.log(r2) / r2)
        out[2 * i] = f * x1
        if 2 * i + 1 < n:
            out[2 * i + 1] = f * x2
    return out

# 使用 Numba 加速编译 normals 函数
normalsj = nb.jit(normals, nopython=True)

# 设定生成随机数的个数
n = 10000

# 定义使用 Numba 加速编译后的生成正态分布随机数的函数
def numbacall():
    return normalsj(n, state_addr)

# 创建 NumPy 的 PCG64 随机数生成器实例
rg = np.random.Generator(PCG64())

# 定义使用 NumPy 生成正态分布随机数的函数
def numpycall():
    return rg.normal(size=n)

# 检查两种生成随机数的函数输出的形状是否一致
r1 = numbacall()
r2 = numpycall()
assert r1.shape == (n,)
assert r1.shape == r2.shape

# 测试 Numba 加速编译后的生成正态分布随机数函数的性能
t1 = timeit(numbacall, number=1000)
print(f'{t1:.2f} secs for {n} PCG64 (Numba/PCG64) gaussian randoms')

# 测试 NumPy 生成正态分布随机数函数的性能
t2 = timeit(numpycall, number=1000)
print(f'{t2:.2f} secs for {n} PCG64 (NumPy/PCG64) gaussian randoms')

# 示例 2

# 获取 PCG64 随机数生成器的下一个 32 位无符号整数随机数方法
next_u32 = bit_gen.ctypes.next_uint32
# 获取 PCG64 随机数生成器的状态
ctypes_state = bit_gen.ctypes.state

# 使用 Numba 加速编译定义的函数，生成在给定范围内的随机数
@nb.jit(nopython=True)
def bounded_uint(lb, ub, state):
    # 计算用于掩码的值
    mask = delta = ub - lb
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16

    # 生成随机数，并确保在指定范围内
    val = next_u32(state) & mask
    while val > delta:
        val = next_u32(state) & mask

    return lb + val

# 打印在指定范围内生成的随机数
print(bounded_uint(323, 2394691, ctypes_state.value))

# 使用 Numba 加速编译定义的函数，生成多个在给定范围内的随机数
@nb.jit(nopython=True)
def bounded_uints(lb, ub, n, state):
    out = np.empty(n, dtype=np.uint32)
    for i in range(n):
        out[i] = bounded_uint(lb, ub, state)

# 生成一定数量的在指定范围内的随机数
bounded_uints(323, 2394691, 10000000, ctypes_state.value)
```