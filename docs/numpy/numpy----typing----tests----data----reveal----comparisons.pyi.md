# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\comparisons.pyi`

```py
import sys  # 导入sys模块，用于访问与Python解释器相关的系统功能
import fractions  # 导入fractions模块，支持有理数的运算
import decimal  # 导入decimal模块，支持高精度的十进制浮点数运算
from typing import Any  # 从typing模块导入Any类型，表示任意类型的对象

import numpy as np  # 导入NumPy库并用np别名表示
import numpy.typing as npt  # 导入NumPy的类型标注模块

if sys.version_info >= (3, 11):
    from typing import assert_type  # 如果Python版本大于等于3.11，则从typing模块导入assert_type函数
else:
    from typing_extensions import assert_type  # 否则从typing_extensions模块导入assert_type函数

c16 = np.complex128()  # 创建一个复数类型为complex128的NumPy对象并赋值给c16
f8 = np.float64()  # 创建一个浮点数类型为float64的NumPy对象并赋值给f8
i8 = np.int64()  # 创建一个整数类型为int64的NumPy对象并赋值给i8
u8 = np.uint64()  # 创建一个无符号整数类型为uint64的NumPy对象并赋值给u8

c8 = np.complex64()  # 创建一个复数类型为complex64的NumPy对象并赋值给c8
f4 = np.float32()  # 创建一个浮点数类型为float32的NumPy对象并赋值给f4
i4 = np.int32()  # 创建一个整数类型为int32的NumPy对象并赋值给i4
u4 = np.uint32()  # 创建一个无符号整数类型为uint32的NumPy对象并赋值给u4

dt = np.datetime64(0, "D")  # 创建一个日期时间类型为datetime64的NumPy对象并赋值给dt
td = np.timedelta64(0, "D")  # 创建一个时间间隔类型为timedelta64的NumPy对象并赋值给td

b_ = np.bool()  # 创建一个布尔类型为bool的NumPy对象并赋值给b_

b = bool()  # 创建一个Python内置的布尔对象并赋值给b
c = complex()  # 创建一个Python内置的复数对象并赋值给c
f = float()  # 创建一个Python内置的浮点数对象并赋值给f
i = int()  # 创建一个Python内置的整数对象并赋值给i

AR = np.array([0], dtype=np.int64)  # 创建一个NumPy数组，包含一个元素0，数据类型为int64，并赋值给AR
AR.setflags(write=False)  # 设置AR数组为不可写

SEQ = (0, 1, 2, 3, 4)  # 创建一个元组SEQ，包含5个整数元素

# object-like comparisons

assert_type(i8 > fractions.Fraction(1, 5), Any)  # 断言i8与1/5的比较结果的类型为Any
assert_type(i8 > [fractions.Fraction(1, 5)], Any)  # 断言i8与包含1/5的列表的比较结果的类型为Any
assert_type(i8 > decimal.Decimal("1.5"), Any)  # 断言i8与Decimal类型的1.5的比较结果的类型为Any
assert_type(i8 > [decimal.Decimal("1.5")], Any)  # 断言i8与包含Decimal类型1.5的列表的比较结果的类型为Any

# Time structures

assert_type(dt > dt, np.bool)  # 断言日期时间对象dt与自身的比较结果的类型为NumPy布尔类型
assert_type(td > td, np.bool)  # 断言时间间隔对象td与自身的比较结果的类型为NumPy布尔类型
assert_type(td > i, np.bool)  # 断言时间间隔对象td与整数对象i的比较结果的类型为NumPy布尔类型
assert_type(td > i4, np.bool)  # 断言时间间隔对象td与整数对象i4的比较结果的类型为NumPy布尔类型
assert_type(td > i8, np.bool)  # 断言时间间隔对象td与整数对象i8的比较结果的类型为NumPy布尔类型

assert_type(td > AR, npt.NDArray[np.bool])  # 断言时间间隔对象td与NumPy数组AR的比较结果的类型为NumPy布尔数组
assert_type(td > SEQ, npt.NDArray[np.bool])  # 断言时间间隔对象td与元组SEQ的比较结果的类型为NumPy布尔数组
assert_type(AR > SEQ, npt.NDArray[np.bool])  # 断言NumPy数组AR与元组SEQ的比较结果的类型为NumPy布尔数组
assert_type(AR > td, npt.NDArray[np.bool])  # 断言NumPy数组AR与时间间隔对象td的比较结果的类型为NumPy布尔数组
assert_type(SEQ > td, npt.NDArray[np.bool])  # 断言元组SEQ与时间间隔对象td的比较结果的类型为NumPy布尔数组
assert_type(SEQ > AR, npt.NDArray[np.bool])  # 断言元组SEQ与NumPy数组AR的比较结果的类型为NumPy布尔数组

# boolean

assert_type(b_ > b, np.bool)  # 断言NumPy布尔对象b_与Python内置布尔对象b的比较结果的类型为NumPy布尔类型
assert_type(b_ > b_, np.bool)  # 断言NumPy布尔对象b_与自身的比较结果的类型为NumPy布尔类型
assert_type(b_ > i, np.bool)  # 断言NumPy布尔对象b_与整数对象i的比较结果的类型为NumPy布尔类型
assert_type(b_ > i8, np.bool)  # 断言NumPy布尔对象b_与整数对象i8的比较结果的类型为NumPy布尔类型
assert_type(b_ > i4, np.bool)  # 断言NumPy布尔对象b_与整数对象i4的比较结果的类型为NumPy布尔类型
assert_type(b_ > u8, np.bool)  # 断言NumPy布尔对象b_与无符号整数对象u8的比较结果的类型为NumPy布尔类型
assert_type(b_ > u4, np.bool)  # 断言NumPy布尔对象b_与无符号整数对象u4的比较结果的类型为NumPy布尔类型
assert_type(b_ > f, np.bool)  # 断言NumPy布尔对象b_与浮点数对象f的比较结果的类型为NumPy布尔类型
assert_type(b_ > f8, np.bool)  # 断言NumPy布尔对象b_与浮点数对象f8的比较结果的类型为NumPy布尔类型
assert_type(b_ > f4, np.bool)  # 断言NumPy布尔对象b_与浮点数对象f4的比较结果的类型为NumPy布尔类型
assert_type(b_ > c, np.bool)  # 断言NumPy布尔对象b_与复数对象c的比较结果的类型为NumPy布尔类型
assert_type(b_ > c16, np.bool)  # 断言NumPy布尔对象b_与复数对象c16的比较结果的类型为NumPy布尔类型
assert_type(b_ > c8, np.bool)  # 断言NumPy布尔对象b_与复数对象c8的比较结果的类型为NumPy布尔类型
assert_type(b_ > AR, npt.NDArray[np.bool])  # 断言NumPy布尔对象b_与NumPy数组AR的比较结果的类型为NumPy布尔数组
assert_type(b_ > SEQ, npt.NDArray[np.bool])  # 断言NumPy布尔对象b_与元组SEQ的比较结果的类型为NumPy布尔数组

# Complex

assert_type(c16 > c16, np.bool)  # 断言复数对象c16与自身的比较结果的类型为NumPy布尔类型
# 确保 c8 大于 SEQ，返回结果应为布尔值的 NumPy 数组
assert_type(c8 > SEQ, npt.NDArray[np.bool])

# 检查 c16 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(c16 > c8, np.bool)

# 检查 f8 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > c8, np.bool)

# 检查 i8 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(i8 > c8, np.bool)

# 检查 c8 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(c8 > c8, np.bool)

# 检查 f4 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > c8, np.bool)

# 检查 i4 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(i4 > c8, np.bool)

# 检查 b_ 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(b_ > c8, np.bool)

# 检查 b 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(b > c8, np.bool)

# 检查 c 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(c > c8, np.bool)

# 检查 f 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(f > c8, np.bool)

# 检查 i 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(i > c8, np.bool)

# 检查 AR 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(AR > c8, npt.NDArray[np.bool])

# 检查 SEQ 是否大于 c8，返回结果应为布尔值的 NumPy 数组
assert_type(SEQ > c8, npt.NDArray[np.bool])

# Float

# 检查 f8 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > f8, np.bool)

# 检查 f8 是否大于 i8，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > i8, np.bool)

# 检查 f8 是否大于 f4，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > f4, np.bool)

# 检查 f8 是否大于 i4，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > i4, np.bool)

# 检查 f8 是否大于 b_，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > b_, np.bool)

# 检查 f8 是否大于 b，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > b, np.bool)

# 检查 f8 是否大于 c，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > c, np.bool)

# 检查 f8 是否大于 f，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > f, np.bool)

# 检查 f8 是否大于 i，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > i, np.bool)

# 检查 f8 是否大于 AR，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > AR, npt.NDArray[np.bool])

# 检查 f8 是否大于 SEQ，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > SEQ, npt.NDArray[np.bool])

# 检查 f8 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > f8, np.bool)

# 检查 i8 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(i8 > f8, np.bool)

# 检查 f4 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > f8, np.bool)

# 检查 i4 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(i4 > f8, np.bool)

# 检查 b_ 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(b_ > f8, np.bool)

# 检查 b 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(b > f8, np.bool)

# 检查 c 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(c > f8, np.bool)

# 检查 f 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(f > f8, np.bool)

# 检查 i 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(i > f8, np.bool)

# 检查 AR 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(AR > f8, npt.NDArray[np.bool])

# 检查 SEQ 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(SEQ > f8, npt.NDArray[np.bool])

# 检查 f4 是否大于 f8，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > f8, np.bool)

# 检查 f4 是否大于 i8，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > i8, np.bool)

# 检查 f4 是否大于 f4，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > f4, np.bool)

# 检查 f4 是否大于 i4，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > i4, np.bool)

# 检查 f4 是否大于 b_，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > b_, np.bool)

# 检查 f4 是否大于 b，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > b, np.bool)

# 检查 f4 是否大于 c，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > c, np.bool)

# 检查 f4 是否大于 f，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > f, np.bool)

# 检查 f4 是否大于 i，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > i, np.bool)

# 检查 f4 是否大于 AR，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > AR, npt.NDArray[np.bool])

# 检查 f4 是否大于 SEQ，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > SEQ, npt.NDArray[np.bool])

# 检查 f8 是否大于 f4，返回结果应为布尔值的 NumPy 数组
assert_type(f8 > f4, np.bool)

# 检查 i8 是否大于 f4，返回结果应为布尔值的 NumPy 数组
assert_type(i8 > f4, np.bool)

# 检查 f4 是否大于 f4，返回结果应为布尔值的 NumPy 数组
assert_type(f4 > f4, np.bool)

# 检查 i4 是否大于 f4，返回结果应为布尔值的 NumPy 数组
assert_type(i4 > f4, np.bool)

# 检查 b_ 是否大于 f4，返回结果应为布尔值的 NumPy 数组
assert_type(b_ > f4, np.bool)

# 检查 b 是否大于 f4，返回结果应为布尔值的 NumPy 数组
assert_type(b > f4, np.bool)

# 检查 c 是否大于 f4，返回结果应为布尔值的 NumPy 数组
assert_type(c > f4, np.bool)

# 检查 f 是否大于 f4，返回结果应为布尔值的 NumPy 数组
assert_type(f > f4, np.bool)

# 检查 i 是否大于 f4，返回结果应为
# 确保变量 u4 大于变量 u8，返回布尔值
assert_type(u4 > u8, np.bool)
# 确保变量 b_ 大于变量 u8，返回布尔值
assert_type(b_ > u8, np.bool)
# 确保变量 b 大于变量 u8，返回布尔值
assert_type(b > u8, np.bool)
# 确保变量 c 大于变量 u8，返回布尔值
assert_type(c > u8, np.bool)
# 确保变量 f 大于变量 u8，返回布尔值
assert_type(f > u8, np.bool)
# 确保变量 i 大于变量 u8，返回布尔值
assert_type(i > u8, np.bool)
# 确保变量 AR 大于变量 u8，返回类型为 npt.NDArray[np.bool] 的数组
assert_type(AR > u8, npt.NDArray[np.bool])
# 确保变量 SEQ 大于变量 u8，返回类型为 npt.NDArray[np.bool] 的数组
assert_type(SEQ > u8, npt.NDArray[np.bool])

# 确保变量 i4 大于变量 i8，返回布尔值
assert_type(i4 > i8, np.bool)
# 确保变量 i4 大于变量 i4，返回布尔值
assert_type(i4 > i4, np.bool)
# 确保变量 i4 大于变量 i，返回布尔值
assert_type(i4 > i, np.bool)
# 确保变量 i4 大于变量 b_，返回布尔值
assert_type(i4 > b_, np.bool)
# 确保变量 i4 大于变量 b，返回布尔值
assert_type(i4 > b, np.bool)
# 确保变量 i4 大于变量 AR，返回类型为 npt.NDArray[np.bool] 的数组
assert_type(i4 > AR, npt.NDArray[np.bool])
# 确保变量 i4 大于变量 SEQ，返回类型为 npt.NDArray[np.bool] 的数组
assert_type(i4 > SEQ, npt.NDArray[np.bool])

# 确保变量 u4 大于变量 i8，返回布尔值
assert_type(u4 > i8, np.bool)
# 确保变量 u4 大于变量 i4，返回布尔值
assert_type(u4 > i4, np.bool)
# 确保变量 u4 大于变量 u8，返回布尔值
assert_type(u4 > u8, np.bool)
# 确保变量 u4 大于变量 u4，返回布尔值
assert_type(u4 > u4, np.bool)
# 确保变量 u4 大于变量 i，返回布尔值
assert_type(u4 > i, np.bool)
# 确保变量 u4 大于变量 b_，返回布尔值
assert_type(u4 > b_, np.bool)
# 确保变量 u4 大于变量 b，返回布尔值
assert_type(u4 > b, np.bool)
# 确保变量 u4 大于变量 AR，返回类型为 npt.NDArray[np.bool] 的数组
assert_type(u4 > AR, npt.NDArray[np.bool])
# 确保变量 u4 大于变量 SEQ，返回类型为 npt.NDArray[np.bool] 的数组
assert_type(u4 > SEQ, npt.NDArray[np.bool])

# 确保变量 i8 大于变量 i4，返回布尔值
assert_type(i8 > i4, np.bool)
# 确保变量 i4 大于变量 i4，返回布尔值
assert_type(i4 > i4, np.bool)
# 确保变量 i 大于变量 i4，返回布尔值
assert_type(i > i4, np.bool)
# 确保变量 b_ 大于变量 i4，返回布尔值
assert_type(b_ > i4, np.bool)
# 确保变量 b 大于变量 i4，返回布尔值
assert_type(b > i4, np.bool)
# 确保变量 AR 大于变量 i4，返回类型为 npt.NDArray[np.bool] 的数组
assert_type(AR > i4, npt.NDArray[np.bool])
# 确保变量 SEQ 大于变量 i4，返回类型为 npt.NDArray[np.bool] 的数组
assert_type(SEQ > i4, npt.NDArray[np.bool])

# 确保变量 i8 大于变量 u4，返回布尔值
assert_type(i8 > u4, np.bool)
# 确保变量 i4 大于变量 u4，返回布尔值
assert_type(i4 > u4, np.bool)
# 确保变量 u8 大于变量 u4，返回布尔值
assert_type(u8 > u4, np.bool)
# 确保变量 u4 大于变量 u4，返回布尔值
assert_type(u4 > u4, np.bool)
# 确保变量 b_ 大于变量 u4，返回布尔值
assert_type(b_ > u4, np.bool)
# 确保变量 b 大于变量 u4，返回布尔值
assert_type(b > u4, np.bool)
# 确保变量 i 大于变量 u4，返回布尔值
assert_type(i > u4, np.bool)
# 确保变量 AR 大于变量 u4，返回类型为 npt.NDArray[np.bool] 的数组
assert_type(AR > u4, npt.NDArray[np.bool])
# 确保变量 SEQ 大于变量 u4，返回类型为 npt.NDArray[np.bool] 的数组
assert_type(SEQ > u4, npt.NDArray[np.bool])
```