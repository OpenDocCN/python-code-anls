# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\fromnumeric.pyi`

```
"""Tests for :mod:`numpy._core.fromnumeric`."""

# 导入 numpy 库，并使用类型定义模块 numpy.typing
import numpy as np
import numpy.typing as npt

# 创建一个布尔类型的 2 维数组 A
A = np.array(True, ndmin=2, dtype=bool)
# 设置数组 A 为不可写
A.setflags(write=False)
# 定义一个类型为 np.str_ 的 numpy 数组 AR_U
AR_U: npt.NDArray[np.str_]

# 创建一个布尔类型的标量数组 a
a = np.bool(True)

# 调用 np.take 函数，尝试取出标量数组 a 的元素，但未指定索引位置，导致类型错误
np.take(a, None)  # E: No overload variant
# 调用 np.take 函数，尝试在轴上使用浮点数索引，导致类型错误
np.take(a, axis=1.0)  # E: No overload variant
# 调用 np.take 函数，尝试将结果输出到整数 1，导致类型错误
np.take(A, out=1)  # E: No overload variant
# 调用 np.take 函数，尝试在模式中使用字符串 "bob"，导致类型错误
np.take(A, mode="bob")  # E: No overload variant

# 调用 np.reshape 函数，尝试使用 None 作为新形状，导致类型错误
np.reshape(a, None)  # E: No overload variant
# 调用 np.reshape 函数，尝试指定新形状为 1，并使用字符串 "bob" 作为排序顺序，导致类型错误
np.reshape(A, 1, order="bob")  # E: No overload variant

# 调用 np.choose 函数，尝试使用 None 作为索引数组，导致类型错误
np.choose(a, None)  # E: No overload variant
# 调用 np.choose 函数，尝试将结果输出到浮点数 1.0，导致类型错误
np.choose(a, out=1.0)  # E: No overload variant
# 调用 np.choose 函数，尝试在模式中使用字符串 "bob"，导致类型错误
np.choose(A, mode="bob")  # E: No overload variant

# 调用 np.repeat 函数，尝试使用 None 作为重复次数，导致类型错误
np.repeat(a, None)  # E: No overload variant
# 调用 np.repeat 函数，尝试在轴上使用浮点数 1.0 作为重复次数，导致类型错误
np.repeat(A, 1, axis=1.0)  # E: No overload variant

# 调用 np.swapaxes 函数，尝试使用 None 作为一个轴，导致类型错误
np.swapaxes(A, None, 1)  # E: No overload variant
# 调用 np.swapaxes 函数，尝试在轴上使用整数 1 和列表 [0]，导致类型错误
np.swapaxes(A, 1, [0])  # E: No overload variant

# 调用 np.transpose 函数，尝试在轴参数中使用浮点数 1.0，导致类型错误
np.transpose(A, axes=1.0)  # E: No overload variant

# 调用 np.partition 函数，尝试使用 None 作为分区索引，导致类型错误
np.partition(a, None)  # E: No overload variant
# 调用 np.partition 函数，尝试在轴参数中使用字符串 "bob"，导致类型错误
np.partition(
    a, 0, axis="bob"
)
# 调用 np.partition 函数，尝试在 kind 参数中使用字符串 "bob"，导致类型错误
np.partition(
    A, 0, kind="bob"
)
# 调用 np.partition 函数，尝试在 order 参数中使用 range(5)，导致类型错误
np.partition(
    A, 0, order=range(5)  # E: Argument "order" to "partition" has incompatible type
)

# 调用 np.argpartition 函数，尝试使用 None 作为分区索引，导致类型错误
np.argpartition(
    a, None  # E: incompatible type
)
# 调用 np.argpartition 函数，尝试在轴参数中使用字符串 "bob"，导致类型错误
np.argpartition(
    a, 0, axis="bob"  # E: incompatible type
)
# 调用 np.argpartition 函数，尝试在 kind 参数中使用字符串 "bob"，导致类型错误
np.argpartition(
    A, 0, kind="bob"  # E: incompatible type
)
# 调用 np.argpartition 函数，尝试在 order 参数中使用 range(5)，导致类型错误
np.argpartition(
    A, 0, order=range(5)  # E: Argument "order" to "argpartition" has incompatible type
)

# 调用 np.sort 函数，尝试在轴参数中使用字符串 "bob"，导致类型错误
np.sort(A, axis="bob")  # E: No overload variant
# 调用 np.sort 函数，尝试在 kind 参数中使用字符串 "bob"，导致类型错误
np.sort(A, kind="bob")  # E: No overload variant
# 调用 np.sort 函数，尝试在 order 参数中使用 range(5)，导致类型错误
np.sort(A, order=range(5))  # E: Argument "order" to "sort" has incompatible type

# 调用 np.argsort 函数，尝试在轴参数中使用字符串 "bob"，导致类型错误
np.argsort(A, axis="bob")  # E: Argument "axis" to "argsort" has incompatible type
# 调用 np.argsort 函数，尝试在 kind 参数中使用字符串 "bob"，导致类型错误
np.argsort(A, kind="bob")  # E: Argument "kind" to "argsort" has incompatible type
# 调用 np.argsort 函数，尝试在 order 参数中使用 range(5)，导致类型错误
np.argsort(A, order=range(5))  # E: Argument "order" to "argsort" has incompatible type

# 调用 np.argmax 函数，尝试在轴参数中使用字符串 "bob"，导致类型错误
np.argmax(A, axis="bob")  # E: No overload variant of "argmax" matches argument type
# 调用 np.argmax 函数，尝试在 kind 参数中使用字符串 "bob"，导致类型错误
np.argmax(A, kind="bob")  # E: No overload variant of "argmax" matches argument type

# 调用 np.argmin 函数，尝试在轴参数中使用字符串 "bob"，导致类型错误
np.argmin(A, axis="bob")  # E: No overload variant of "argmin" matches argument type
# 调用 np.argmin 函数，尝试在 kind 参数中使用字符串 "bob"，导致类型错误
np.argmin(A, kind="bob")  # E: No overload variant of "argmin" matches argument type

# 调用 np.searchsorted 函数，尝试在 side 参数中使用字符串 "bob"，导致类型错误
np.searchsorted(
    A[0], 0, side="bob"
)
# 调用 np.searchsorted 函数，尝试在 sorter 参数中使用浮点数 1.0，导致类型错误
np.searchsorted(
    A[0], 0, sorter=1.0
)

# 调用 np.resize 函数，尝试在新形状参数中使用浮点数 1.0，导致类型错误
np.resize(A, 1.0)  # E: No overload variant

# 调用 np.squeeze 函数，尝试在轴参数中使用浮点数 1.0，导致类型错误
np.squeeze(A, 1.0)  # E: No overload variant of "squeeze" matches argument type

# 调用 np.diagonal 函数，尝试在 offset 参数中使用 None，导致类型错误
np.diagonal(A, offset=None)  # E: No overload variant
# 调用 np.diagonal 函数，尝试在 axis1 参数中使用字符串 "bob"，导致类型错误
np.diagonal(A, axis
    # 创建一个包含单个元素True的列表
    [True],
    # A 是一个变量或对象，用于某种计算或操作
    A,
    # axis 参数设置为 1.0，可能指定某些操作在特定轴上进行
    axis=1.0
np.clip(a, 1, 2, out=1)  # E: No overload variant of "clip" matches argument type
# 将数组 a 中的元素限制在 [1, 2] 的范围内，并将结果写入数组 out 中，但指定的输出参数类型不匹配

np.sum(a, axis=1.0)  # E: No overload variant
# 计算数组 a 沿指定轴的元素和，但指定的轴参数类型不匹配

np.sum(a, keepdims=1.0)  # E: No overload variant
# 计算数组 a 的元素和，并保持输入数组的维度，但指定的 keepdims 参数类型不匹配

np.sum(a, initial=[1])  # E: No overload variant
# 计算数组 a 的元素和，但指定的 initial 参数类型不匹配

np.all(a, axis=1.0)  # E: No overload variant
# 检查数组 a 沿指定轴的所有元素是否都为 True，但指定的轴参数类型不匹配

np.all(a, keepdims=1.0)  # E: No overload variant
# 检查数组 a 的所有元素是否都为 True，并保持输入数组的维度，但指定的 keepdims 参数类型不匹配

np.all(a, out=1.0)  # E: No overload variant
# 检查数组 a 的所有元素是否都为 True，但指定的输出参数类型不匹配

np.any(a, axis=1.0)  # E: No overload variant
# 检查数组 a 沿指定轴的任一元素是否为 True，但指定的轴参数类型不匹配

np.any(a, keepdims=1.0)  # E: No overload variant
# 检查数组 a 的任一元素是否为 True，并保持输入数组的维度，但指定的 keepdims 参数类型不匹配

np.any(a, out=1.0)  # E: No overload variant
# 检查数组 a 的任一元素是否为 True，但指定的输出参数类型不匹配

np.cumsum(a, axis=1.0)  # E: No overload variant
# 计算数组 a 沿指定轴的累积和，但指定的轴参数类型不匹配

np.cumsum(a, dtype=1.0)  # E: No overload variant
# 计算数组 a 的累积和，但指定的 dtype 参数类型不匹配

np.cumsum(a, out=1.0)  # E: No overload variant
# 计算数组 a 的累积和，但指定的输出参数类型不匹配

np.ptp(a, axis=1.0)  # E: No overload variant
# 计算数组 a 沿指定轴的最大值和最小值之差，但指定的轴参数类型不匹配

np.ptp(a, keepdims=1.0)  # E: No overload variant
# 计算数组 a 的最大值和最小值之差，并保持输入数组的维度，但指定的 keepdims 参数类型不匹配

np.ptp(a, out=1.0)  # E: No overload variant
# 计算数组 a 的最大值和最小值之差，但指定的输出参数类型不匹配

np.amax(a, axis=1.0)  # E: No overload variant
# 返回数组 a 沿指定轴的最大值，但指定的轴参数类型不匹配

np.amax(a, keepdims=1.0)  # E: No overload variant
# 返回数组 a 的最大值，并保持输入数组的维度，但指定的 keepdims 参数类型不匹配

np.amax(a, out=1.0)  # E: No overload variant
# 返回数组 a 的最大值，但指定的输出参数类型不匹配

np.amax(a, initial=[1.0])  # E: No overload variant
# 返回数组 a 的最大值，但指定的 initial 参数类型不匹配

np.amax(a, where=[1.0])  # E: incompatible type
# 返回数组 a 的最大值，但指定的 where 参数类型不匹配

np.amin(a, axis=1.0)  # E: No overload variant
# 返回数组 a 沿指定轴的最小值，但指定的轴参数类型不匹配

np.amin(a, keepdims=1.0)  # E: No overload variant
# 返回数组 a 的最小值，并保持输入数组的维度，但指定的 keepdims 参数类型不匹配

np.amin(a, out=1.0)  # E: No overload variant
# 返回数组 a 的最小值，但指定的输出参数类型不匹配

np.amin(a, initial=[1.0])  # E: No overload variant
# 返回数组 a 的最小值，但指定的 initial 参数类型不匹配

np.amin(a, where=[1.0])  # E: incompatible type
# 返回数组 a 的最小值，但指定的 where 参数类型不匹配

np.prod(a, axis=1.0)  # E: No overload variant
# 计算数组 a 沿指定轴的元素乘积，但指定的轴参数类型不匹配

np.prod(a, out=False)  # E: No overload variant
# 计算数组 a 的元素乘积，但指定的输出参数类型不匹配

np.prod(a, keepdims=1.0)  # E: No overload variant
# 计算数组 a 的元素乘积，并保持输入数组的维度，但指定的 keepdims 参数类型不匹配

np.prod(a, initial=int)  # E: No overload variant
# 计算数组 a 的元素乘积，但指定的 initial 参数类型不匹配

np.prod(a, where=1.0)  # E: No overload variant
# 计算数组 a 的元素乘积，但指定的 where 参数类型不匹配

np.prod(AR_U)  # E: incompatible type
# 计算数组 AR_U 的元素乘积，但输入的数组类型不匹配

np.cumprod(a, axis=1.0)  # E: No overload variant
# 计算数组 a 沿指定轴的累积乘积，但指定的轴参数类型不匹配

np.cumprod(a, out=False)  # E: No overload variant
# 计算数组 a 的累积乘积，但指定的输出参数类型不匹配

np.cumprod(AR_U)  # E: incompatible type
# 计算数组 AR_U 的累积乘积，但输入的数组类型不匹配

np.size(a, axis=1.0)  # E: Argument "axis" to "size" has incompatible type
# 返回数组 a 沿指定轴的元素个数，但指定的轴参数类型不匹配

np.around(a, decimals=1.0)  # E: No overload variant
# 将数组 a 中的元素四舍五入到指定的小数位数，但指定的 decimals 参数类型不匹配

np.around(a, out=type)  # E: No overload variant
# 将数组 a 中的元素四舍五入，但指定的输出参数类型不匹配

np.around(AR_U)  # E: incompatible type
# 将数组 AR_U 中的元素四舍五入，但输入的数组类型不匹配

np.mean(a, axis=1.0)  # E: No overload variant
# 计算数组 a 沿指定轴的均值，但指定的轴参数类型不匹配

np.mean(a, out=False)  # E: No overload variant
# 计算数组 a 的均值，但指定的输出参数类型不匹配

np.mean(a, keepdims=1.0)  # E: No overload variant
# 计算数组 a 的均值，并保持输入数组的维度，但指定的 keepdims 参数类型不匹配

np.mean(AR_U)  # E: incompatible type
# 计算数组 AR_U 的均值，但输入的数组类型
```