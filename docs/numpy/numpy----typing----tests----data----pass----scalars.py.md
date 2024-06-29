# `.\numpy\numpy\typing\tests\data\pass\scalars.py`

```py
import sys  # 导入系统模块sys，用于访问系统相关功能
import datetime as dt  # 导入datetime模块并命名为dt，用于处理日期和时间

import pytest  # 导入pytest模块，用于编写和运行测试用例
import numpy as np  # 导入numpy库并命名为np，用于数值计算

b = np.bool()  # 创建一个布尔型对象b
b_ = np.bool_()  # 创建一个布尔型对象b_
u8 = np.uint64()  # 创建一个64位无符号整数对象u8
i8 = np.int64()  # 创建一个64位有符号整数对象i8
f8 = np.float64()  # 创建一个64位浮点数对象f8
c16 = np.complex128()  # 创建一个128位复数对象c16
U = np.str_()  # 创建一个空字符串对象U
S = np.bytes_()  # 创建一个空字节串对象S

# Construction

class D:
    def __index__(self) -> int:
        return 0

class C:
    def __complex__(self) -> complex:
        return 3j

class B:
    def __int__(self) -> int:
        return 4

class A:
    def __float__(self) -> float:
        return 4.0

np.complex64(3j)  # 创建一个32位复数对象，使用Python复数值3j
np.complex64(A())  # 创建一个32位复数对象，使用A类返回的浮点数值
np.complex64(C())  # 创建一个32位复数对象，使用C类返回的复数值
np.complex128(3j)  # 创建一个64位复数对象，使用Python复数值3j
np.complex128(C())  # 创建一个64位复数对象，使用C类返回的复数值
np.complex128(None)  # 创建一个64位复数对象，使用Python None

np.complex64("1.2")  # 尝试使用字符串创建32位复数对象，会抛出错误
np.complex128(b"2j")  # 尝试使用字节串创建64位复数对象，会抛出错误

np.int8(4)  # 创建一个8位有符号整数对象，使用整数4
np.int16(3.4)  # 创建一个16位有符号整数对象，尝试使用浮点数3.4，将会截断为整数3
np.int32(4)  # 创建一个32位有符号整数对象，使用整数4
np.int64(-1)  # 创建一个64位有符号整数对象，使用整数-1
np.uint8(B())  # 创建一个8位无符号整数对象，使用B类返回的整数值
np.uint32()  # 创建一个32位无符号整数对象，未提供值，默认为0
np.int32("1")  # 尝试使用字符串创建32位有符号整数对象，会抛出错误
np.int64(b"2")  # 尝试使用字节串创建64位有符号整数对象，会抛出错误

np.float16(A())  # 创建一个16位浮点数对象，使用A类返回的浮点数值
np.float32(16)  # 创建一个32位浮点数对象，使用整数16
np.float64(3.0)  # 创建一个64位浮点数对象，使用浮点数3.0
np.float64(None)  # 创建一个64位浮点数对象，使用Python None
np.float32("1")  # 尝试使用字符串创建32位浮点数对象，会抛出错误
np.float16(b"2.5")  # 尝试使用字节串创建16位浮点数对象，会抛出错误

np.uint64(D())  # 创建一个64位无符号整数对象，使用D类返回的整数值
np.float32(D())  # 创建一个32位浮点数对象，使用D类返回的浮点数值
np.complex64(D())  # 创建一个32位复数对象，使用D类返回的复数值

np.bytes_(b"hello")  # 创建一个字节串对象，使用字节串b"hello"
np.bytes_("hello", 'utf-8')  # 创建一个字节串对象，使用字符串"hello"，指定编码为utf-8
np.bytes_("hello", encoding='utf-8')  # 创建一个字节串对象，使用字符串"hello"，指定编码为utf-8
np.str_("hello")  # 创建一个字符串对象，使用字符串"hello"
np.str_(b"hello", 'utf-8')  # 创建一个字符串对象，使用字节串b"hello"，指定编码为utf-8
np.str_(b"hello", encoding='utf-8')  # 创建一个字符串对象，使用字节串b"hello"，指定编码为utf-8

# Array-ish semantics

np.int8().real  # 访问一个8位整数对象的实部属性
np.int16().imag  # 访问一个16位整数对象的虚部属性
np.int32().data  # 访问一个32位整数对象的数据属性
np.int64().flags  # 访问一个64位整数对象的标志属性

np.uint8().itemsize * 2  # 计算一个8位无符号整数对象的字节大小乘以2
np.uint16().ndim + 1  # 访问一个16位无符号整数对象的维度属性，并加1
np.uint32().strides  # 访问一个32位无符号整数对象的步幅属性
np.uint64().shape  # 访问一个64位无符号整数对象的形状属性

# Time structures

np.datetime64()  # 创建一个datetime64对象，未提供参数，默认为'1970-01-01'
np.datetime64(0, "D")  # 创建一个datetime64对象，使用0和时间单位"D"表示天
np.datetime64(0, b"D")  # 创建一个datetime64对象，使用0和时间单位b"D"表示天
np.datetime64(0, ('ms', 3))  # 创建一个datetime64对象，使用0和时间单位('ms', 3)表示毫秒
np.datetime64("2019")  # 创建一个datetime64对象，使用字符串"2019"
np.datetime64(b"2019")  # 创建一个datetime64对象，使用字节串b"2019"
np.datetime64("2019", "D")  # 创建一个datetime64对象，使用字符串"2019"和时间单位"D"表示天
np.datetime64(np.datetime64())  # 创建一个datetime64对象，使用当前的datetime64对象
np.datetime64(dt.datetime(2000, 5, 3))  # 创建一个datetime64对象，使用datetime.datetime对象表示的日期时间
np.datetime64(dt.date(2000, 5, 3))  # 创建一个datetime64对象，使用datetime.date对象表示的日期
np.datetime64(None)  # 创建一个datetime64对象，使用Python None
np.datetime64(None, "D")  # 创建一个datetime64对象，使用Python None和时间单位"D"表示天

np.timedelta64()  # 创建一个timedelta64对象，未提供参数，默认为0
np.timedelta64(0)  # 创建一个timedelta64对象，使用整数0
np.timedelta64(0, "D")  # 创建一个timedelta64对象，使用0和时间单位"D"表示天
np.timedelta64(0, ('ms', 3))  # 创建一个timedelta64对象，使用0和时间单位('ms', 3)表示毫秒
np.timedelta64(0, b"D")  # 创建一个timedelta64对象，使用0和时间单位b"D"表示天
np.timedelta64("3")  # 创建一个timedelta64对象，使用字符串"3"
np.timedelta64(b"5")  # 创建一个timedelta64对象，使用字节串b"5"
np.timedelta64(np.timedelta64(2))  # 创建一个timedelta64对象，使用当前的timedelta64对象
np.timedelta64(dt.timedelta(2))  # 创建一个timedelta64对象，使用datetime.timedelta对象表示的时间差
np.timedelta64(None)  # 创建一个timedelta64对象，使用Python None
np.timedelta64(None, "D")  # 创建一个timedelta64对象，使用Python None和时间单位"D"表示天

np.void(1)  # 创建一个void对象，使用整数1
np.void(np.int64(1))  # 创建一个void对象，使用np.int64对象表示的整数1
np.
# 获取数组的单个元素
u8.item()
f8.item()
c16.item()
U.item()
S.item()

# 将数组转换为列表
b.tolist()
i8.tolist()
u8.tolist()
f8.tolist()
c16.tolist()
U.tolist()
S.tolist()

# 将数组展开为一维数组
b.ravel()
i8.ravel()
u8.ravel()
f8.ravel()
c16.ravel()
U.ravel()
S.ravel()

# 将数组展开为一维数组
b.flatten()
i8.flatten()
u8.flatten()
f8.flatten()
c16.flatten()
U.flatten()
S.flatten()

# 将数组重新整形为指定形状
b.reshape(1)
i8.reshape(1)
u8.reshape(1)
f8.reshape(1)
c16.reshape(1)
U.reshape(1)
S.reshape(1)
```