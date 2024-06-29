# `D:\src\scipysrc\pandas\pandas\_libs\byteswap.pyi`

```
# 从给定的字节数组中读取一个浮点数，根据需要进行字节顺序的交换
def read_float_with_byteswap(data: bytes, offset: int, byteswap: bool) -> float:
    ...

# 从给定的字节数组中读取一个双精度浮点数，根据需要进行字节顺序的交换
def read_double_with_byteswap(data: bytes, offset: int, byteswap: bool) -> float:
    ...

# 从给定的字节数组中读取一个无符号 16 位整数，根据需要进行字节顺序的交换
def read_uint16_with_byteswap(data: bytes, offset: int, byteswap: bool) -> int:
    ...

# 从给定的字节数组中读取一个无符号 32 位整数，根据需要进行字节顺序的交换
def read_uint32_with_byteswap(data: bytes, offset: int, byteswap: bool) -> int:
    ...

# 从给定的字节数组中读取一个无符号 64 位整数，根据需要进行字节顺序的交换
def read_uint64_with_byteswap(data: bytes, offset: int, byteswap: bool) -> int:
    ...


这些函数声明了一系列从给定的字节数组中读取特定类型数据的函数，并可以选择是否进行字节顺序的交换。
```