# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\multiarray.pyi`

```py
import numpy as np  # 导入 NumPy 库

i8: np.int64  # 定义别名 i8 为 np.int64

AR_b: npt.NDArray[np.bool]  # 定义 AR_b 为布尔类型的 NumPy 数组
AR_u1: npt.NDArray[np.uint8]  # 定义 AR_u1 为无符号 8 位整数类型的 NumPy 数组
AR_i8: npt.NDArray[np.int64]  # 定义 AR_i8 为有符号 64 位整数类型的 NumPy 数组
AR_f8: npt.NDArray[np.float64]  # 定义 AR_f8 为双精度浮点数类型的 NumPy 数组
AR_M: npt.NDArray[np.datetime64]  # 定义 AR_M 为日期时间类型的 NumPy 数组

M: np.datetime64  # 定义 M 为日期时间类型的 NumPy 对象

AR_LIKE_f: list[float]  # 定义 AR_LIKE_f 为浮点数列表

def func(a: int) -> None: ...  # 定义一个函数 func，接收一个整数参数 a，无返回值

np.where(AR_b, 1)  # 使用 np.where 函数，根据条件 AR_b 返回 1 的索引位置

np.can_cast(AR_f8, 1)  # 检查是否可以将 AR_f8 类型转换为整数 1

np.vdot(AR_M, AR_M)  # 计算 AR_M 和 AR_M 的向量点积

np.copyto(AR_LIKE_f, AR_f8)  # 将 AR_f8 的值复制到 AR_LIKE_f，要求类型兼容

np.putmask(AR_LIKE_f, [True, True, False], 1.5)  # 根据掩码修改 AR_LIKE_f 中的值

np.packbits(AR_f8)  # 将 AR_f8 中的数据打包为位字段
np.packbits(AR_u1, bitorder=">")  # 将 AR_u1 中的数据按大端顺序打包为位字段

np.unpackbits(AR_i8)  # 解包 AR_i8 中的位字段为数组
np.unpackbits(AR_u1, bitorder=">")  # 按大端顺序解包 AR_u1 中的位字段为数组

np.shares_memory(1, 1, max_work=i8)  # 检查两个对象是否共享内存，设置最大工作大小为 i8

np.may_share_memory(1, 1, max_work=i8)  # 检查两个对象是否可能共享内存，设置最大工作大小为 i8

np.arange(M)  # 创建从 0 开始到 M-1 的整数数组
np.arange(stop=10)  # 创建一个从 0 到 9 的整数数组

np.datetime_data(int)  # 获取整数类型的日期时间数据

np.busday_offset("2012", 10)  # 计算从 "2012" 开始的第 10 个工作日的日期

np.datetime_as_string("2012")  # 将日期时间 "2012" 转换为字符串表示形式

np.char.compare_chararrays("a", b"a", "==", False)  # 比较两个字符数组是否相等，不考虑大小写

np.nested_iters([AR_i8, AR_i8])  # 创建嵌套迭代器，传入数组列表，但缺少位置参数
np.nested_iters([AR_i8, AR_i8], 0)  # 创建嵌套迭代器，但类型不兼容
np.nested_iters([AR_i8, AR_i8], [0])  # 创建嵌套迭代器，但类型不兼容
np.nested_iters([AR_i8, AR_i8], [[0], [1]], flags=["test"])  # 创建嵌套迭代器，但类型不兼容
np.nested_iters([AR_i8, AR_i8], [[0], [1]], op_flags=[["test"]])  # 创建嵌套迭代器，但类型不兼容
np.nested_iters([AR_i8, AR_i8], [[0], [1]], buffersize=1.0)  # 创建嵌套迭代器，但类型不兼容
```