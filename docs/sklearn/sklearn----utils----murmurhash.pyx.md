# `D:\src\scipysrc\scikit-learn\sklearn\utils\murmurhash.pyx`

```
"""Cython wrapper for MurmurHash3 non-cryptographic hash function.

MurmurHash is an extensively tested and very fast hash function that has
good distribution properties suitable for machine learning use cases
such as feature hashing and random projections.

The original C++ code by Austin Appleby is released the public domain
and can be found here:

  https://code.google.com/p/smhasher/

"""
# 上述为模块的简介和背景信息

# 导入必要的类型定义
from ..utils._typedefs cimport int32_t, uint32_t

# 导入 NumPy 库
import numpy as np

# 从外部头文件导入 MurmurHash3 算法的函数签名
cdef extern from "src/MurmurHash3.h":
    void MurmurHash3_x86_32(void *key, int len, uint32_t seed, void *out)
    void MurmurHash3_x86_128(void *key, int len, uint32_t seed, void *out)
    void MurmurHash3_x64_128 (void *key, int len, uint32_t seed, void *out)

# 定义一个 Cython 可调用的函数，计算 32 位整数键的 MurmurHash3 值
cpdef uint32_t murmurhash3_int_u32(int key, unsigned int seed):
    """Compute the 32bit murmurhash3 of a int key at seed."""
    cdef uint32_t out
    MurmurHash3_x86_32(&key, sizeof(int), seed, &out)
    return out

# 定义一个 Cython 可调用的函数，计算 32 位带符号整数键的 MurmurHash3 值
cpdef int32_t murmurhash3_int_s32(int key, unsigned int seed):
    """Compute the 32bit murmurhash3 of a int key at seed."""
    cdef int32_t out
    MurmurHash3_x86_32(&key, sizeof(int), seed, &out)
    return out

# 定义一个 Cython 可调用的函数，计算字节串键的 MurmurHash3 值（32 位无符号整数）
cpdef uint32_t murmurhash3_bytes_u32(bytes key, unsigned int seed):
    """Compute the 32bit murmurhash3 of a bytes key at seed."""
    cdef uint32_t out
    MurmurHash3_x86_32(<char*> key, len(key), seed, &out)
    return out

# 定义一个 Cython 可调用的函数，计算字节串键的 MurmurHash3 值（32 位带符号整数）
cpdef int32_t murmurhash3_bytes_s32(bytes key, unsigned int seed):
    """Compute the 32bit murmurhash3 of a bytes key at seed."""
    cdef int32_t out
    MurmurHash3_x86_32(<char*> key, len(key), seed, &out)
    return out

# 定义一个函数，计算整数数组键的 MurmurHash3 值（32 位无符号整数）
def _murmurhash3_bytes_array_u32(
    const int32_t[:] key,
    unsigned int seed,
):
    """Compute 32bit murmurhash3 hashes of a key int array at seed."""
    # TODO make it possible to pass preallocated output array
    cdef:
        uint32_t[:] out = np.zeros(key.size, np.uint32)
        Py_ssize_t i
    for i in range(key.shape[0]):
        out[i] = murmurhash3_int_u32(key[i], seed)
    return np.asarray(out)

# 定义一个函数，计算整数数组键的 MurmurHash3 值（32 位带符号整数）
def _murmurhash3_bytes_array_s32(
    const int32_t[:] key,
    unsigned int seed,
):
    """Compute 32bit murmurhash3 hashes of a key int array at seed."""
    # TODO make it possible to pass preallocated output array
    cdef:
        int32_t[:] out = np.zeros(key.size, np.int32)
        Py_ssize_t i
    for i in range(key.shape[0]):
        out[i] = murmurhash3_int_s32(key[i], seed)
    return np.asarray(out)

# 定义一个函数，计算任意键的 MurmurHash3 值（32 位整数）
def murmurhash3_32(key, seed=0, positive=False):
    """Compute the 32bit murmurhash3 of key at seed.

    The underlying implementation is MurmurHash3_x86_32 generating low
    latency 32bits hash suitable for implementing lookup tables, Bloom
    filters, count min sketch or feature hashing.

    Parameters
    ----------
    key : np.int32, bytes, unicode or ndarray of dtype=np.int32
        The physical object to hash.
    # 如果 key 是 bytes 类型
    if isinstance(key, bytes):
        # 如果 positive 为 True，返回无符号整数的 murmurhash3 值
        if positive:
            return murmurhash3_bytes_u32(key, seed)
        # 如果 positive 为 False，返回有符号整数的 murmurhash3 值
        else:
            return murmurhash3_bytes_s32(key, seed)
    
    # 如果 key 是 unicode 类型（Python 2.x 中的 str 类型）
    elif isinstance(key, unicode):
        # 如果 positive 为 True，返回无符号整数的 murmurhash3 值（使用 UTF-8 编码转换为 bytes）
        if positive:
            return murmurhash3_bytes_u32(key.encode('utf-8'), seed)
        # 如果 positive 为 False，返回有符号整数的 murmurhash3 值（使用 UTF-8 编码转换为 bytes）
        else:
            return murmurhash3_bytes_s32(key.encode('utf-8'), seed)
    
    # 如果 key 是 int 或 np.int32 类型
    elif isinstance(key, int) or isinstance(key, np.int32):
        # 如果 positive 为 True，返回无符号整数的 murmurhash3 值
        if positive:
            return murmurhash3_int_u32(<int32_t>key, seed)  # 使用 key 的整数值进行哈希
        # 如果 positive 为 False，返回有符号整数的 murmurhash3 值
        else:
            return murmurhash3_int_s32(<int32_t>key, seed)  # 使用 key 的整数值进行哈希
    
    # 如果 key 是 np.ndarray 类型
    elif isinstance(key, np.ndarray):
        # 检查 key 的 dtype 是否为 np.int32
        if key.dtype != np.int32:
            raise TypeError(
                "key.dtype should be int32, got %s" % key.dtype)
        # 如果 positive 为 True，返回无符号整数的 murmurhash3 值（将数组展平后进行哈希）
        if positive:
            return _murmurhash3_bytes_array_u32(key.ravel(), seed).reshape(key.shape)
        # 如果 positive 为 False，返回有符号整数的 murmurhash3 值（将数组展平后进行哈希）
        else:
            return _murmurhash3_bytes_array_s32(key.ravel(), seed).reshape(key.shape)
    
    # 如果 key 的类型不支持上述任何一种
    else:
        raise TypeError(
            "key %r with type %s is not supported. "
            "Explicit conversion to bytes is required" % (key, type(key)))
```