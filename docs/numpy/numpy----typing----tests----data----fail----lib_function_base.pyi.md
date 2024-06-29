# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\lib_function_base.pyi`

```py
from typing import Any

import numpy as np
import numpy.typing as npt

# 声明不同数据类型的数组变量
AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]
AR_m: npt.NDArray[np.timedelta64]
AR_M: npt.NDArray[np.datetime64]
AR_O: npt.NDArray[np.object_]

# 定义一个空函数 func，参数为整数，无返回值
def func(a: int) -> None: ...

# 计算 AR_m 数组的平均值，但存在类型不兼容的错误
np.average(AR_m)  # E: incompatible type

# 在 AR_f8 数组中选择元素，但存在类型不兼容的错误
np.select(1, [AR_f8])  # E: incompatible type

# 计算 AR_m 数组的角度，但存在类型不兼容的错误
np.angle(AR_m)  # E: incompatible type

# 对 AR_m 数组进行解包，但存在类型不兼容的错误
np.unwrap(AR_m)  # E: incompatible type

# 对 AR_c16 数组进行解包，但存在类型不兼容的错误
np.unwrap(AR_c16)  # E: incompatible type

# 对整数值进行去零处理，但存在类型不兼容的错误
np.trim_zeros(1)  # E: incompatible type

# 在整数值中应用条件替换，但存在类型不兼容的错误
np.place(1, [True], 1.5)  # E: incompatible type

# 创建一个对整数进行向量化的函数，但存在类型不兼容的错误
np.vectorize(1)  # E: incompatible type

# 在 AR_f8 数组的切片位置插入值 5，但存在类型不兼容的错误
np.place(AR_f8, slice(None), 5)  # E: incompatible type

# 使用插值计算 AR_f8 和 AR_c16 数组之间的插值，但存在类型不兼容的错误
np.interp(AR_f8, AR_c16, AR_f8)  # E: incompatible type

# 使用插值计算 AR_c16 和 AR_f8 数组之间的插值，但存在类型不兼容的错误
np.interp(AR_c16, AR_f8, AR_f8)  # E: incompatible type

# 使用插值计算 AR_f8 和 AR_O 数组之间的插值，但存在类型不兼容的错误
np.interp(AR_f8, AR_f8, AR_O)  # E: incompatible type

# 计算 AR_m 数组的协方差，但存在类型不兼容的错误
np.cov(AR_m)  # E: incompatible type

# 计算 AR_O 数组的协方差，但存在类型不兼容的错误
np.cov(AR_O)  # E: incompatible type

# 计算 AR_m 数组的相关系数，但存在类型不兼容的错误
np.corrcoef(AR_m)  # E: incompatible type

# 计算 AR_O 数组的相关系数，但存在类型不兼容的错误
np.corrcoef(AR_O)  # E: incompatible type

# 计算 AR_f8 数组相关系数时，指定了不支持的参数，不存在匹配的重载变体
np.corrcoef(AR_f8, bias=True)  # E: No overload variant

# 计算 AR_f8 数组相关系数时，指定了不支持的参数，不存在匹配的重载变体
np.corrcoef(AR_f8, ddof=2)  # E: No overload variant

# 创建复数数列时使用了不兼容的参数，存在类型不兼容的错误
np.blackman(1j)  # E: incompatible type

# 创建复数数列时使用了不兼容的参数，存在类型不兼容的错误
np.bartlett(1j)  # E: incompatible type

# 创建复数数列时使用了不兼容的参数，存在类型不兼容的错误
np.hanning(1j)  # E: incompatible type

# 创建复数数列时使用了不兼容的参数，存在类型不兼容的错误
np.hamming(1j)  # E: incompatible type

# 创建复数数列时使用了不兼容的参数，存在类型不兼容的错误
np.hamming(AR_c16)  # E: incompatible type

# 创建复数数列时使用了不兼容的参数，存在类型不兼容的错误
np.kaiser(1j, 1)  # E: incompatible type

# 在 AR_O 数组上应用 sinc 函数，但存在类型不兼容的错误
np.sinc(AR_O)  # E: incompatible type

# 计算 AR_M 数组的中位数，但存在类型不兼容的错误
np.median(AR_M)  # E: incompatible type

# 计算 AR_f8 数组百分位数时，使用了不兼容的参数，不存在匹配的重载变体
np.percentile(AR_f8, 50j)  # E: No overload variant

# 计算 AR_f8 数组百分位数时，使用了不兼容的参数，不存在匹配的重载变体
np.percentile(AR_f8, 50, interpolation="bob")  # E: No overload variant

# 计算 AR_f8 数组分位数时，使用了不兼容的参数，不存在匹配的重载变体
np.quantile(AR_f8, 0.5j)  # E: No overload variant

# 计算 AR_f8 数组分位数时，使用了不兼容的参数，不存在匹配的重载变体
np.quantile(AR_f8, 0.5, interpolation="bob")  # E: No overload variant

# 创建网格时使用了不兼容的参数，存在类型不兼容的错误
np.meshgrid(AR_f8, AR_f8, indexing="bob")  # E: incompatible type

# 在 AR_f8 数组中删除 AR_f8 数组的元素，但存在类型不兼容的错误
np.delete(AR_f8, AR_f8)  # E: incompatible type

# 在 AR_f8 数组中插入元素 AR_f8，但存在类型不兼容的错误
np.insert(AR_f8, AR_f8, 1.5)  # E: incompatible type

# 在 AR_f8 数组中执行数位化操作时使用了不兼容的参数，不存在匹配的重载变体
np.digitize(AR_f8, 1j)  # E: No overload variant
```