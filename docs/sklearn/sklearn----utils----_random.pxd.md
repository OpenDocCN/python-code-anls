# `D:\src\scipysrc\scikit-learn\sklearn\utils\_random.pxd`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的类型定义
from ._typedefs cimport uint32_t

# 默认种子值为1
cdef inline uint32_t DEFAULT_SEED = 1

# 定义枚举常量
cdef enum:
    # 我们自己实现的 rand_r 替代方案的最大值
    # 我们不使用 RAND_MAX，因为它在不同平台上有所不同，
    # 特别是在 Windows/MSVC 上非常小。
    # 这对应于32位有符号整数的最大可表示值（即2^31 - 1）。
    RAND_R_MAX = 2147483647


# 使用32位 XorShift 生成器实现的 rand_r 替代方案
# 参考文献：http://www.jstatsoft.org/v08/i14/paper
cdef inline uint32_t our_rand_r(uint32_t* seed) nogil:
    """使用 np.uint32 种子生成伪随机数 np.uint32"""
    # 如果种子为0，则设置为默认种子值
    if (seed[0] == 0):
        seed[0] = DEFAULT_SEED

    seed[0] ^= <uint32_t>(seed[0] << 13)
    seed[0] ^= <uint32_t>(seed[0] >> 17)
    seed[0] ^= <uint32_t>(seed[0] << 5)

    # 使用取模确保不返回超出有符号32位整数最大值的值（即2^31 - 1）
    # 注意需要使用括号以避免溢出：这里 RAND_R_MAX 在加1前被转换为 uint32_t 类型。
    return seed[0] % ((<uint32_t>RAND_R_MAX) + 1)
```