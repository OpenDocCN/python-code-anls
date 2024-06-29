# `D:\src\scipysrc\pandas\pandas\_libs\hashing.pyx`

```
# Translated from the reference implementation
# at https://github.com/veorq/SipHash

# 导入Cython的声明
cimport cython
# 从C标准库中导入所需函数
from libc.stdlib cimport (
    free,       # 释放内存的函数
    malloc,     # 分配内存的函数
)

import numpy as np   # 导入NumPy库

from numpy cimport (   # 导入NumPy的C语言级接口
    import_array,     # 初始化NumPy C API
    ndarray,          # NumPy数组类型
    uint8_t,          # 8位无符号整数类型
    uint64_t,         # 64位无符号整数类型
)

import_array()   # 初始化NumPy数组接口

from pandas._libs.util cimport is_nan   # 导入Pandas的内部工具函数

# 声明Cython函数，禁用边界检查以提高性能
@cython.boundscheck(False)
def hash_object_array(
    ndarray[object, ndim=1] arr, str key, str encoding="utf8"
) -> np.ndarray[np.uint64]:
    """
    Parameters
    ----------
    arr : 1-d object ndarray of objects
        待哈希的对象数组
    key : hash key, must be 16 byte len encoded
        哈希密钥，必须为16字节长的编码
    encoding : encoding for key & arr, default to 'utf8'
        编码格式，默认为'utf8'

    Returns
    -------
    1-d uint64 ndarray of hashes.
    返回一个一维的64位无符号整数数组，存储哈希值。

    Raises
    ------
    TypeError
        If the array contains mixed types.
        如果数组包含混合类型，则抛出TypeError异常。

    Notes
    -----
    Allowed values must be strings, or nulls
    mixed array types will raise TypeError.
    允许的值必须为字符串或null，
    混合类型数组将引发TypeError异常。
    """
    cdef:
        Py_ssize_t i, n    # 循环计数器和数组长度
        uint64_t[::1] result   # 存储哈希结果的数组
        bytes data, k     # 数据字节和密钥字节
        uint8_t *kb       # 指向密钥字节的指针
        uint64_t *lens    # 数据长度数组
        char **vecs       # 字符串指针数组
        char *cdata       # 字符串数据指针
        object val        # 数组元素
        list data_list = []   # 存储数据字节的列表

    k = <bytes>key.encode(encoding)   # 将密钥编码为字节流
    kb = <uint8_t *>k     # 获取密钥字节的指针
    if len(k) != 16:      # 如果密钥长度不为16字节，抛出值错误异常
        raise ValueError(
            f"key should be a 16-byte string encoded, got {k} (len {len(k)})"
        )

    n = len(arr)   # 获取数组长度

    # 创建一个字节指针数组
    vecs = <char **>malloc(n * sizeof(char *))
    if vecs is NULL:   # 检查分配内存是否成功
        raise MemoryError()
    lens = <uint64_t*>malloc(n * sizeof(uint64_t))
    if lens is NULL:   # 检查分配内存是否成功
        raise MemoryError()

    for i in range(n):   # 遍历数组
        val = arr[i]     # 获取数组元素
        if isinstance(val, bytes):   # 如果是字节类型，直接赋值给data
            data = <bytes>val
        elif isinstance(val, str):   # 如果是字符串类型，编码为字节流
            data = <bytes>val.encode(encoding)
        elif val is None or is_nan(val):   # 如果是空值或NaN，转换为字符串并编码为字节流
            data = <bytes>str(val).encode(encoding)

        elif isinstance(val, tuple):   # 如果是元组类型，尝试哈希元组内部的元素
            hash(val)
            data = <bytes>str(val).encode(encoding)
        else:   # 其他类型抛出类型错误异常
            raise TypeError(
                f"{val} of type {type(val)} is not a valid type for hashing, "
                "must be string or null"
            )

        lens[i] = len(data)   # 记录数据的长度
        cdata = data   # 获取数据的指针

        # 通过将数据添加到列表中保持引用有效
        data_list.append(data)
        vecs[i] = cdata   # 将数据指针存入指针数组中

    result = np.empty(n, dtype=np.uint64)   # 创建存储结果的NumPy数组
    with nogil:   # 禁用GIL以提高性能
        for i in range(n):
            result[i] = low_level_siphash(<uint8_t *>vecs[i], lens[i], kb)   # 调用低级SipHash函数计算哈希值

    free(vecs)   # 释放分配的内存
    free(lens)   # 释放分配的内存
    return result.base  # 返回底层的np.ndarray，以便正确地返回哈希数组
    # 返回一个64位整数，将一个包含8个字节的数组p的元素按位组合成一个整数
    return (<uint64_t>p[0] |
            <uint64_t>p[1] << 8 |
            <uint64_t>p[2] << 16 |
            <uint64_t>p[3] << 24 |
            <uint64_t>p[4] << 32 |
            <uint64_t>p[5] << 40 |
            <uint64_t>p[6] << 48 |
            <uint64_t>p[7] << 56)
# 定义一个名为 _sipround 的内联函数，接受四个指向 uint64_t 类型变量的指针参数，无异常抛出且不使用全局解锁机制
cdef void _sipround(uint64_t* v0, uint64_t* v1,
                    uint64_t* v2, uint64_t* v3) noexcept nogil:
    # 第一轮 SIP 轮函数运算
    v0[0] += v1[0]
    v1[0] = _rotl(v1[0], 13)
    v1[0] ^= v0[0]
    v0[0] = _rotl(v0[0], 32)
    # 第二轮 SIP 轮函数运算
    v2[0] += v3[0]
    v3[0] = _rotl(v3[0], 16)
    v3[0] ^= v2[0]
    # 第三轮 SIP 轮函数运算
    v0[0] += v3[0]
    v3[0] = _rotl(v3[0], 21)
    v3[0] ^= v0[0]
    # 第四轮 SIP 轮函数运算
    v2[0] += v1[0]
    v1[0] = _rotl(v1[0], 17)
    v1[0] ^= v2[0]
    v2[0] = _rotl(v2[0], 32)


# 使用 Cython 声明，启用除法检查，定义一个名为 low_level_siphash 的内联函数，接受三个 uint8_t* 类型的参数并且不抛出异常，不使用全局解锁机制
cdef uint64_t low_level_siphash(uint8_t* data, size_t datalen,
                                uint8_t* key) noexcept nogil:
    # 初始化 SipHash 算法所需的四个 64 位变量
    cdef uint64_t v0 = 0x736f6d6570736575ULL
    cdef uint64_t v1 = 0x646f72616e646f6dULL
    cdef uint64_t v2 = 0x6c7967656e657261ULL
    cdef uint64_t v3 = 0x7465646279746573ULL
    cdef uint64_t b
    # 使用给定的 key 生成两个 64 位的密钥
    cdef uint64_t k0 = u8to64_le(key)
    cdef uint64_t k1 = u8to64_le(key + 8)
    cdef uint64_t m
    cdef int i
    # 指向数据结尾的指针，数据长度按 uint64_t 对齐
    cdef uint8_t* end = data + datalen - (datalen % sizeof(uint64_t))
    cdef int left = datalen & 7  # 数据最后的不完整字节数
    cdef int cROUNDS = 2  # 压缩轮数
    cdef int dROUNDS = 4  # 拓展轮数

    # 计算并应用初始值
    b = (<uint64_t>datalen) << 56
    v3 ^= k1
    v2 ^= k0
    v1 ^= k1
    v0 ^= k0

    # 遍历数据块，每个块为 64 位，应用 SIP 轮函数
    while (data != end):
        m = u8to64_le(data)
        v3 ^= m
        for i in range(cROUNDS):
            _sipround(&v0, &v1, &v2, &v3)
        v0 ^= m
        data += sizeof(uint64_t)

    # 处理剩余的字节数据
    for i in range(left-1, -1, -1):
        b |= (<uint64_t>data[i]) << (i * 8)

    v3 ^= b

    # 应用额外的 SIP 轮函数
    for i in range(cROUNDS):
        _sipround(&v0, &v1, &v2, &v3)

    v0 ^= b
    v2 ^= 0xff

    # 再次应用 SIP 轮函数
    for i in range(dROUNDS):
        _sipround(&v0, &v1, &v2, &v3)

    # 计算最终的哈希值并返回
    b = v0 ^ v1 ^ v2 ^ v3

    return b
```