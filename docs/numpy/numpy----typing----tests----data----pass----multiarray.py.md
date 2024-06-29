# `.\numpy\numpy\typing\tests\data\pass\multiarray.py`

```
# 导入 numpy 库并为变量指定类型
import numpy as np
import numpy.typing as npt

# 声明一个包含单个浮点数的一维数组 AR_f8
AR_f8: npt.NDArray[np.float64] = np.array([1.0])
# 声明一个包含单个整数的一维数组 AR_i4，并指定数据类型为 int32
AR_i4 = np.array([1], dtype=np.int32)
# 声明一个包含单个无符号整数的一维数组 AR_u1，并指定数据类型为 uint8
AR_u1 = np.array([1], dtype=np.uint8)

# 使用 AR_f8 创建一个广播对象 b_f8
b_f8 = np.broadcast(AR_f8)
# 调用广播对象 b_f8 的方法 next()，获取下一个元素
next(b_f8)
# 重置广播对象 b_f8 的迭代状态
b_f8.reset()
# 返回广播对象 b_f8 的当前索引
b_f8.index
# 返回广播对象 b_f8 的迭代器元组
b_f8.iters
# 返回广播对象 b_f8 的维度
b_f8.nd
# 返回广播对象 b_f8 的维度数
b_f8.ndim
# 返回广播对象 b_f8 的迭代器数量
b_f8.numiter
# 返回广播对象 b_f8 的形状
b_f8.shape
# 返回广播对象 b_f8 的大小
b_f8.size

# 使用 AR_i4, AR_f8, AR_f8 创建一个多元广播对象 b_i4_f8_f8
next(b_i4_f8_f8)
# 重置多元广播对象 b_i4_f8_f8 的迭代状态
b_i4_f8_f8.reset()
# 返回多元广播对象 b_i4_f8_f8 的维度
b_i4_f8_f8.ndim
# 返回多元广播对象 b_i4_f8_f8 的当前索引
b_i4_f8_f8.index
# 返回多元广播对象 b_i4_f8_f8 的迭代器元组
b_i4_f8_f8.iters
# 返回多元广播对象 b_i4_f8_f8 的维度
b_i4_f8_f8.nd
# 返回多元广播对象 b_i4_f8_f8 的迭代器数量
b_i4_f8_f8.numiter
# 返回多元广播对象 b_i4_f8_f8 的形状
b_i4_f8_f8.shape
# 返回多元广播对象 b_i4_f8_f8 的大小
b_i4_f8_f8.size

# 计算 AR_f8 和 AR_i4 的内积
np.inner(AR_f8, AR_i4)

# 返回数组中满足条件的索引
np.where([True, True, False])
# 根据条件返回数组中的值
np.where([True, True, False], 1, 0)

# 使用给定的键排序多维数组
np.lexsort([0, 1, 2])

# 检查是否可以将指定类型转换为另一类型
np.can_cast(np.dtype("i8"), int)
# 检查是否可以将 AR_f8 转换为指定类型 "f8"
np.can_cast(AR_f8, "f8")
# 检查是否可以将 AR_f8 转换为复数类型 np.complex128，使用不安全转换
np.can_cast(AR_f8, np.complex128, casting="unsafe")

# 返回数组中能够精确表示所有值的最小标量类型
np.min_scalar_type([1])
# 返回 AR_f8 中能够精确表示所有值的最小标量类型
np.min_scalar_type(AR_f8)

# 返回数组的结果类型
np.result_type(int, AR_i4)
# 返回 AR_f8 和 AR_u1 的结果类型
np.result_type(AR_f8, AR_u1)
# 返回 AR_f8 和 np.complex128 的结果类型
np.result_type(AR_f8, np.complex128)

# 计算向量的点积
np.dot(AR_LIKE_f, AR_i4)
# 计算标量与数组的乘积
np.dot(AR_u1, 1)
# 计算复数与标量的乘积
np.dot(1.5j, 1)
# 计算标量与数组的乘积，并将结果存储到 AR_f8
np.dot(AR_u1, 1, out=AR_f8)

# 计算向量的内积
np.vdot(AR_LIKE_f, AR_i4)
# 计算标量与数组的内积
np.vdot(AR_u1, 1)
# 计算复数与标量的内积
np.vdot(1.5j, 1)

# 计算数组中每个元素的出现次数
np.bincount(AR_i4)

# 将源数组的内容复制到目标数组
np.copyto(AR_f8, [1.6])

# 根据条件将值放入目标数组
np.putmask(AR_f8, [True], 1.5)

# 将整数数组打包为二进制位数组
np.packbits(AR_i4)
# 将无符号整数数组打包为二进制位数组
np.packbits(AR_u1)

# 将二进制位数组解包为整数数组
np.unpackbits(AR_u1)

# 检查两个对象是否共享内存
np.shares_memory(1, 2)
# 检查多个对象是否共享内存，设置最大工作数为 1
np.shares_memory(AR_f8, AR_f8, max_work=1)

# 检查两个对象是否可能共享内存
np.may_share_memory(1, 2)
# 检查多个对象是否可能共享内存，设置最大工作数为 1
np.may_share_memory(AR_f8, AR_f8, max_work=1)
```