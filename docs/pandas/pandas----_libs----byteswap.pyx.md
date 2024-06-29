# `D:\src\scipysrc\pandas\pandas\_libs\byteswap.pyx`

```
"""
以下是避免 Python 函数调用开销的 struct.unpack 更快的版本。

在 SAS7BDAT 解析器中，它们可能会被调用多达 (n_rows * n_cols) 次。
"""
# 导入必要的 Cython 类型和函数
from cython cimport Py_ssize_t
from libc.stdint cimport (
    uint16_t,
    uint32_t,
    uint64_t,
)
from libc.string cimport memcpy


# 读取单精度浮点数并进行字节交换
def read_float_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    cdef uint32_t value
    # 确保偏移量加上 value 的大小不超过数据长度
    assert offset + sizeof(value) < len(data)
    # 定义指针，指向数据起始位置加上偏移量处
    cdef const void *ptr = <unsigned char*>(data) + offset
    # 从指定位置复制数据到 value 中
    memcpy(&value, ptr, sizeof(value))
    # 如果需要字节交换，则调用 _byteswap4 函数
    if byteswap:
        value = _byteswap4(value)

    cdef float res
    # 将 value 的内容复制到 res 中作为返回结果
    memcpy(&res, &value, sizeof(res))
    return res


# 读取双精度浮点数并进行字节交换
def read_double_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    cdef uint64_t value
    assert offset + sizeof(value) < len(data)
    cdef const void *ptr = <unsigned char*>(data) + offset
    memcpy(&value, ptr, sizeof(value))
    if byteswap:
        value = _byteswap8(value)

    cdef double res
    memcpy(&res, &value, sizeof(res))
    return res


# 读取 uint16_t 类型整数并进行字节交换
def read_uint16_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    cdef uint16_t res
    assert offset + sizeof(res) < len(data)
    memcpy(&res, <const unsigned char*>(data) + offset, sizeof(res))
    if byteswap:
        res = _byteswap2(res)
    return res


# 读取 uint32_t 类型整数并进行字节交换
def read_uint32_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    cdef uint32_t res
    assert offset + sizeof(res) < len(data)
    memcpy(&res, <const unsigned char*>(data) + offset, sizeof(res))
    if byteswap:
        res = _byteswap4(res)
    return res


# 读取 uint64_t 类型整数并进行字节交换
def read_uint64_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    cdef uint64_t res
    assert offset + sizeof(res) < len(data)
    memcpy(&res, <const unsigned char*>(data) + offset, sizeof(res))
    if byteswap:
        res = _byteswap8(res)
    return res


# 字节交换功能的定义

cdef extern from *:
    """
    #ifdef _MSC_VER
        #define _byteswap2 _byteswap_ushort
        #define _byteswap4 _byteswap_ulong
        #define _byteswap8 _byteswap_uint64
    #else
        #define _byteswap2 __builtin_bswap16
        #define _byteswap4 __builtin_bswap32
        #define _byteswap8 __builtin_bswap64
    #endif
    """
    uint16_t _byteswap2(uint16_t)    # 字节交换函数，用于 uint16_t
    uint32_t _byteswap4(uint32_t)    # 字节交换函数，用于 uint32_t
    uint64_t _byteswap8(uint64_t)    # 字节交换函数，用于 uint64_t
```