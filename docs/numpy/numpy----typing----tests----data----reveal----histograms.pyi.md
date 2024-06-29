# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\histograms.pyi`

```py
import sys  # 导入sys模块，用于系统相关操作
from typing import Any  # 从typing模块导入Any类型，用于类型提示

import numpy as np  # 导入NumPy库，并用np作为别名
import numpy.typing as npt  # 导入NumPy的类型提示模块

if sys.version_info >= (3, 11):
    from typing import assert_type  # 如果Python版本大于等于3.11，则从typing模块直接导入assert_type函数
else:
    from typing_extensions import assert_type  # 否则从typing_extensions模块导入assert_type函数

AR_i8: npt.NDArray[np.int64]  # 定义AR_i8变量，类型为np.int64的NumPy数组
AR_f8: npt.NDArray[np.float64]  # 定义AR_f8变量，类型为np.float64的NumPy数组

assert_type(np.histogram_bin_edges(AR_i8, bins="auto"), npt.NDArray[Any])  # 断言np.histogram_bin_edges返回值为npt.NDArray[Any]
assert_type(np.histogram_bin_edges(AR_i8, bins="rice", range=(0, 3)), npt.NDArray[Any])  # 断言np.histogram_bin_edges返回值为npt.NDArray[Any]
assert_type(np.histogram_bin_edges(AR_i8, bins="scott", weights=AR_f8), npt.NDArray[Any])  # 断言np.histogram_bin_edges返回值为npt.NDArray[Any]

assert_type(np.histogram(AR_i8, bins="auto"), tuple[npt.NDArray[Any], npt.NDArray[Any]])  # 断言np.histogram返回值为元组，包含两个npt.NDArray[Any]类型
assert_type(np.histogram(AR_i8, bins="rice", range=(0, 3)), tuple[npt.NDArray[Any], npt.NDArray[Any]])  # 断言np.histogram返回值为元组，包含两个npt.NDArray[Any]类型
assert_type(np.histogram(AR_i8, bins="scott", weights=AR_f8), tuple[npt.NDArray[Any], npt.NDArray[Any]])  # 断言np.histogram返回值为元组，包含两个npt.NDArray[Any]类型
assert_type(np.histogram(AR_f8, bins=1, density=True), tuple[npt.NDArray[Any], npt.NDArray[Any]])  # 断言np.histogram返回值为元组，包含两个npt.NDArray[Any]类型

assert_type(np.histogramdd(AR_i8, bins=[1]), tuple[npt.NDArray[Any], tuple[npt.NDArray[Any], ...]])  # 断言np.histogramdd返回值为元组，第一个元素为npt.NDArray[Any]，第二个元素为元组类型
assert_type(np.histogramdd(AR_i8, range=[(0, 3)]), tuple[npt.NDArray[Any], tuple[npt.NDArray[Any], ...]])  # 断言np.histogramdd返回值为元组，第一个元素为npt.NDArray[Any]，第二个元素为元组类型
assert_type(np.histogramdd(AR_i8, weights=AR_f8), tuple[npt.NDArray[Any], tuple[npt.NDArray[Any], ...]])  # 断言np.histogramdd返回值为元组，第一个元素为npt.NDArray[Any]，第二个元素为元组类型
assert_type(np.histogramdd(AR_f8, density=True), tuple[npt.NDArray[Any], tuple[npt.NDArray[Any], ...]])  # 断言np.histogramdd返回值为元组，第一个元素为npt.NDArray[Any]，第二个元素为元组类型
```