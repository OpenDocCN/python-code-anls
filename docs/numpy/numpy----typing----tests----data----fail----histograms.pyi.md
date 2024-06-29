# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\histograms.pyi`

```py
import numpy as np  # 导入NumPy库，通常用np作为别名
import numpy.typing as npt  # 导入NumPy类型定义模块，用于类型提示

AR_i8: npt.NDArray[np.int64]  # 声明AR_i8为一个NumPy数组，其中元素类型为np.int64
AR_f8: npt.NDArray[np.float64]  # 声明AR_f8为一个NumPy数组，其中元素类型为np.float64

np.histogram_bin_edges(AR_i8, range=(0, 1, 2))  # 计算AR_i8的直方图边缘值，但类型不兼容

np.histogram(AR_i8, range=(0, 1, 2))  # 计算AR_i8的直方图，但类型不兼容

np.histogramdd(AR_i8, range=(0, 1))  # 计算AR_i8的多维直方图，但类型不兼容
np.histogramdd(AR_i8, range=[(0, 1, 2)])  # 计算AR_i8的多维直方图，但类型不兼容
```