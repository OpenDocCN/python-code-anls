# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\twodim_base.pyi`

```py
# 导入必要的类型和函数
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt


# 定义一个接受任意类型数组和整数的函数，返回一个字符串类型数组
def func1(ar: npt.NDArray[Any], a: int) -> npt.NDArray[np.str_]:
    pass


# 定义一个接受任意类型数组和浮点数的函数，返回一个浮点数
def func2(ar: npt.NDArray[Any], a: float) -> float:
    pass


# 定义一个布尔类型的NumPy数组
AR_b: npt.NDArray[np.bool]
# 定义一个时间间隔类型的NumPy数组
AR_m: npt.NDArray[np.timedelta64]

# 定义一个布尔类型的列表
AR_LIKE_b: list[bool]

# 创建一个10x20的单位矩阵，返回错误：没有匹配的函数重载
np.eye(10, M=20.0)  # E: No overload variant
# 创建一个10x10的单位矩阵，偏移为2.5，数据类型为整数，返回错误：没有匹配的函数重载
np.eye(10, k=2.5, dtype=int)  # E: No overload variant

# 从AR_b创建对角矩阵，偏移为0.5，返回错误：没有匹配的函数重载
np.diag(AR_b, k=0.5)  # E: No overload variant
# 从AR_b创建扁平对角矩阵，偏移为0.5，返回错误：没有匹配的函数重载
np.diagflat(AR_b, k=0.5)  # E: No overload variant

# 创建一个10x20的下三角矩阵，返回错误：没有匹配的函数重载
np.tri(10, M=20.0)  # E: No overload variant
# 创建一个10x10的下三角矩阵，偏移为2.5，数据类型为整数，返回错误：没有匹配的函数重载
np.tri(10, k=2.5, dtype=int)  # E: No overload variant

# 从AR_b创建一个下三角矩阵，偏移为0.5，返回错误：没有匹配的函数重载
np.tril(AR_b, k=0.5)  # E: No overload variant
# 从AR_b创建一个上三角矩阵，偏移为0.5，返回错误：没有匹配的函数重载
np.triu(AR_b, k=0.5)  # E: No overload variant

# 从AR_m创建一个Vandermonde矩阵，返回错误：不兼容的类型
np.vander(AR_m)  # E: incompatible type

# 计算AR_m的二维直方图，返回错误：没有匹配的函数重载
np.histogram2d(AR_m)  # E: No overload variant

# 使用函数func1计算长度为10的遮罩索引数组，返回错误：不兼容的类型
np.mask_indices(10, func1)  # E: incompatible type
# 使用函数func2和参数10.5计算长度为10的遮罩索引数组，返回错误：不兼容的类型
np.mask_indices(10, func2, 10.5)  # E: incompatible type
```