# `.\numpy\numpy\typing\tests\data\pass\simple.py`

```py
"""Simple expression that should pass with mypy."""
import operator  # 导入 operator 模块

import numpy as np  # 导入 numpy 库
import numpy.typing as npt  # 导入 numpy.typing 模块，用于类型注解
from collections.abc import Iterable  # 从 collections.abc 模块导入 Iterable 类型

# Basic checks
array = np.array([1, 2])  # 创建一个 NumPy 数组 `array`，包含元素 [1, 2]

def ndarray_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return x  # 函数接收一个 np.float64 类型的 NumPy 数组，并返回相同类型的数组

ndarray_func(np.array([1, 2], dtype=np.float64))  # 调用函数 `ndarray_func`，传入一个 np.float64 类型的数组
array == 1  # 检查数组 `array` 中的每个元素是否等于 1，返回布尔值数组
array.dtype == float  # 检查数组 `array` 的数据类型是否为 float，返回布尔值

# Dtype construction
np.dtype(float)  # 创建一个描述 float 类型的 NumPy dtype 对象
np.dtype(np.float64)  # 创建一个描述 np.float64 类型的 NumPy dtype 对象
np.dtype(None)  # 创建一个未指定类型的 NumPy dtype 对象
np.dtype("float64")  # 创建一个描述 float64 类型的 NumPy dtype 对象
np.dtype(np.dtype(float))  # 使用已有的 float 类型的 dtype 创建一个新的 NumPy dtype 对象
np.dtype(("U", 10))  # 创建一个长度为 10 的 Unicode 字符串数组的 dtype 对象
np.dtype((np.int32, (2, 2)))  # 创建一个 2x2 的 int32 类型的结构化 dtype 对象

# Define the arguments on the previous line to prevent bidirectional
# type inference in mypy from broadening the types.
two_tuples_dtype = [("R", "u1"), ("G", "u1"), ("B", "u1")]
np.dtype(two_tuples_dtype)  # 创建一个包含三个字段的结构化 dtype 对象

three_tuples_dtype = [("R", "u1", 2)]
np.dtype(three_tuples_dtype)  # 创建一个包含一个字段和其形状的结构化 dtype 对象

mixed_tuples_dtype = [("R", "u1"), ("G", np.str_, 1)]
np.dtype(mixed_tuples_dtype)  # 创建一个包含一个字符字段和一个字符串字段的结构化 dtype 对象

shape_tuple_dtype = [("R", "u1", (2, 2))]
np.dtype(shape_tuple_dtype)  # 创建一个包含一个形状为 (2, 2) 的字符字段的结构化 dtype 对象

shape_like_dtype = [("R", "u1", (2, 2)), ("G", np.str_, 1)]
np.dtype(shape_like_dtype)  # 创建一个包含两个字段的结构化 dtype 对象，其中一个是形状为 (2, 2) 的字符字段

object_dtype = [("field1", object)]
np.dtype(object_dtype)  # 创建一个包含一个 object 类型字段的结构化 dtype 对象

np.dtype((np.int32, (np.int8, 4)))  # 创建一个嵌套的 dtype 对象，包含 int32 和 int8 组成的元组

# Dtype comparison
np.dtype(float) == float  # 检查 np.dtype(float) 是否等于 float，返回布尔值
np.dtype(float) != np.float64  # 检查 np.dtype(float) 是否不等于 np.float64，返回布尔值
np.dtype(float) < None  # 检查 np.dtype(float) 是否小于 None，返回布尔值
np.dtype(float) <= "float64"  # 检查 np.dtype(float) 是否小于等于 "float64"，返回布尔值
np.dtype(float) > np.dtype(float)  # 检查 np.dtype(float) 是否大于 np.dtype(float)，返回布尔值
np.dtype(float) >= np.dtype(("U", 10))  # 检查 np.dtype(float) 是否大于等于包含长度为 10 的 Unicode 字符串数组的 dtype 对象

# Iteration and indexing
def iterable_func(x: Iterable[object]) -> Iterable[object]:
    return x  # 函数接收一个可迭代对象，返回相同的可迭代对象

iterable_func(array)  # 调用函数 `iterable_func`，传入数组 `array`
[element for element in array]  # 列表推导式，生成数组 `array` 的所有元素组成的列表
iter(array)  # 返回一个迭代器，迭代数组 `array` 中的元素
zip(array, array)  # 创建一个迭代器，同时迭代两个数组 `array` 中的元素
array[1]  # 获取数组 `array` 中索引为 1 的元素
array[:]  # 获取数组 `array` 的所有元素（切片）
array[...]  # 获取数组 `array` 的所有元素（完整切片）
array[:] = 0  # 将数组 `array` 的所有元素设置为 0

array_2d = np.ones((3, 3))  # 创建一个全为 1 的 3x3 的二维数组 `array_2d`
array_2d[:2, :2]  # 获取二维数组 `array_2d` 的左上角 2x2 子数组
array_2d[..., 0]  # 获取二维数组 `array_2d` 的第一列元素
array_2d[:2, :2] = 0  # 将二维数组 `array_2d` 的左上角 2x2 子数组的元素设置为 0

# Other special methods
len(array)  # 返回数组 `array` 的长度
str(array)  # 返回数组 `array` 的字符串表示
array_scalar = np.array(1)  # 创建一个标量值为 1 的数组 `array_scalar`
int(array_scalar)  # 将数组 `array_scalar` 转换为整数
float(array_scalar)  # 将数组 `array_scalar` 转换为浮点数
bytes(array_scalar)  # 返回数组 `array_scalar` 的字节表示
operator.index(array_scalar)  # 返回数组 `array_scalar` 的索引

# comparisons
array < 1  # 比较数组 `array` 中的每个元素是否小于 1，返回布尔值数组
array <= 1  # 比较数组 `array` 中的每个元素是否小于等于 1，返回布尔值数组
array == 1  # 比较数组 `array` 中的每个元素是否等于 1，返回布尔值数组
array != 1  # 比较数组 `array` 中的每个元素是否不等于 1，返回布尔值数组
array > 1  # 比较数组 `array` 中的每个元素是否大于 1，返回布尔值数组
array >= 1  # 比较数组 `array` 中的每个元素是否大于等于 1，返回布尔值数组
1 < array  # 比较数组 `array` 中的每个元素是否大于 1，返回布尔值数组
1 <= array  # 比较数组 `array` 中的每个元素是否大于等于 1，返回布尔值数组
1 == array  # 比较数组 `array` 中的每个元素是否等于 1，返回布尔值数组
1 != array  # 比较数组 `array` 中的每个元素是否不等于 1，返回布尔值数组
1 > array  # 比较数组 `array` 中的每个元素是否小于 1，返回布尔值数组
1 >= array  # 比较数组 `array` 中的每个元素是否小于等于 1，返回布尔值数组

# binary arithmetic
array + 1  # 将数组 `array` 中的每个元素加上 1
1 + array  # 将数组 `array` 中的每个元素加上 1
array += 1  # 将数组 `array` 中的每个元素加上 1

array - 1  # 将数组 `array` 中的每个元素减去 1
1 - array  # 将 1 减去数组 `array` 中的每个元素
array -= 1  # 将
```