# `D:\src\scipysrc\scikit-learn\sklearn\utils\murmurhash.pxd`

```
# 导入所需的 C 语言头文件和数据类型定义
from ..utils._typedefs cimport int32_t, uint32_t

# C API 目前被禁用，因为即使不使用这些函数，也需要 -I 标志来确保编译正常。
# cdef extern from "MurmurHash3.h":
#     void MurmurHash3_x86_32(void* key, int len, unsigned int seed,
#                             void* out)
#
#     void MurmurHash3_x86_128(void* key, int len, unsigned int seed,
#                              void* out)
#
#     void MurmurHash3_x64_128(void* key, int len, unsigned int seed,
#                              void* out)

# 定义使用 Cython 包装的 MurmurHash3 散列算法函数，提供不同输入类型的版本

cpdef uint32_t murmurhash3_int_u32(int key, unsigned int seed)
# 使用 MurmurHash3 算法对输入整数 key 进行无符号 32 位整数哈希，使用 seed 作为种子

cpdef int32_t murmurhash3_int_s32(int key, unsigned int seed)
# 使用 MurmurHash3 算法对输入整数 key 进行有符号 32 位整数哈希，使用 seed 作为种子

cpdef uint32_t murmurhash3_bytes_u32(bytes key, unsigned int seed)
# 使用 MurmurHash3 算法对输入字节串 key 进行无符号 32 位整数哈希，使用 seed 作为种子

cpdef int32_t murmurhash3_bytes_s32(bytes key, unsigned int seed)
# 使用 MurmurHash3 算法对输入字节串 key 进行有符号 32 位整数哈希，使用 seed 作为种子
```