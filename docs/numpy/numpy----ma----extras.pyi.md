# `D:\src\scipysrc\numpy\numpy\ma\extras.pyi`

```py
from typing import Any

from numpy.lib._index_tricks_impl import AxisConcatenator

from numpy.ma.core import (
    dot as dot,
    mask_rowcols as mask_rowcols,
)

__all__: list[str]

# 定义函数 count_masked，用于计算数组中的掩码元素数量
def count_masked(arr, axis=...): ...

# 定义函数 masked_all，创建指定形状和数据类型的全掩码数组
def masked_all(shape, dtype = ...): ...

# 定义函数 masked_all_like，创建与给定数组 arr 相同形状和数据类型的全掩码数组
def masked_all_like(arr): ...

# 定义类 _fromnxfunction，用于包装来自 numpy 函数的名称和文档字符串
class _fromnxfunction:
    __name__: Any
    __doc__: Any
    # 初始化方法接受函数名称 funcname 作为参数
    def __init__(self, funcname): ...
    # 返回函数的文档字符串
    def getdoc(self): ...
    # 调用方法，接受任意参数并传递给包装的函数
    def __call__(self, *args, **params): ...

# 定义类 _fromnxfunction_single，继承自 _fromnxfunction，用于包装接受单个参数的函数
class _fromnxfunction_single(_fromnxfunction):
    def __call__(self, x, *args, **params): ...

# 定义类 _fromnxfunction_seq，继承自 _fromnxfunction，用于包装接受序列参数的函数
class _fromnxfunction_seq(_fromnxfunction):
    def __call__(self, x, *args, **params): ...

# 定义类 _fromnxfunction_allargs，继承自 _fromnxfunction，用于包装接受任意参数的函数
class _fromnxfunction_allargs(_fromnxfunction):
    def __call__(self, *args, **params): ...

# 定义函数 apply_along_axis，沿指定轴应用函数 func1d 到数组 arr
def apply_along_axis(func1d, axis, arr, *args, **kwargs): ...

# 定义函数 apply_over_axes，对数组 a 应用函数 func，沿指定轴 axes
def apply_over_axes(func, a, axes): ...

# 定义函数 average，计算数组 a 的加权平均值
def average(a, axis=..., weights=..., returned=..., keepdims=...): ...

# 定义函数 median，计算数组 a 沿指定轴的中位数
def median(a, axis=..., out=..., overwrite_input=..., keepdims=...): ...

# 定义函数 compress_nd，对数组 x 沿指定轴 axis 压缩
def compress_nd(x, axis=...): ...

# 定义函数 compress_rowcols，对数组 x 压缩掩码行和列
def compress_rowcols(x, axis=...): ...

# 定义函数 compress_rows，对数组 a 压缩行
def compress_rows(a): ...

# 定义函数 compress_cols，对数组 a 压缩列
def compress_cols(a): ...

# 定义函数 mask_rows，对数组 a 掩码行
def mask_rows(a, axis = ...): ...

# 定义函数 mask_cols，对数组 a 掩码列
def mask_cols(a, axis = ...): ...

# 定义函数 ediff1d，计算数组 arr 的一阶差分
def ediff1d(arr, to_end=..., to_begin=...): ...

# 定义函数 unique，返回数组 ar1 的唯一元素
def unique(ar1, return_index=..., return_inverse=...): ...

# 定义函数 intersect1d，返回两个数组 ar1 和 ar2 的交集
def intersect1d(ar1, ar2, assume_unique=...): ...

# 定义函数 setxor1d，返回两个数组 ar1 和 ar2 的对称差集
def setxor1d(ar1, ar2, assume_unique=...): ...

# 定义函数 in1d，检查数组 ar1 的元素是否包含在 ar2 中
def in1d(ar1, ar2, assume_unique=..., invert=...): ...

# 定义函数 isin，检查元素 element 是否包含在 test_elements 中
def isin(element, test_elements, assume_unique=..., invert=...): ...

# 定义函数 union1d，返回两个数组 ar1 和 ar2 的并集
def union1d(ar1, ar2): ...

# 定义函数 setdiff1d，返回两个数组 ar1 和 ar2 的差集
def setdiff1d(ar1, ar2, assume_unique=...): ...

# 定义函数 cov，计算两个数组 x 和 y 的协方差
def cov(x, y=..., rowvar=..., bias=..., allow_masked=..., ddof=...): ...

# 定义函数 corrcoef，计算两个数组 x 和 y 的相关系数
def corrcoef(x, y=..., rowvar=..., bias = ..., allow_masked=..., ddof = ...): ...

# 定义类 MAxisConcatenator，继承自 AxisConcatenator，用于数组轴的拼接
class MAxisConcatenator(AxisConcatenator):
    concatenate: Any
    # 类方法 makemat，将数组 arr 转换为矩阵
    @classmethod
    def makemat(cls, arr): ...
    # 获取数组的切片，key 为切片参数
    def __getitem__(self, key): ...

# 定义类 mr_class，继承自 MAxisConcatenator，用于数组操作
class mr_class(MAxisConcatenator):
    def __init__(self): ...

# 定义全局变量 mr_，为 mr_class 的实例
mr_: mr_class

# 定义函数 ndenumerate，枚举数组 a 中的非掩码元素和索引
def ndenumerate(a, compressed=...): ...

# 定义函数 flatnotmasked_edges，查找数组 a 中的非掩码边缘元素
def flatnotmasked_edges(a): ...

# 定义函数 notmasked_edges，查找数组 a 沿指定轴的非掩码边缘元素
def notmasked_edges(a, axis=...): ...

# 定义函数 flatnotmasked_contiguous，查找数组 a 中的非掩码连续元素
def flatnotmasked_contiguous(a): ...

# 定义函数 notmasked_contiguous，查找数组 a 沿指定轴的非掩码连续元素
def notmasked_contiguous(a, axis=...): ...

# 定义函数 clump_unmasked，查找数组 a 中非掩码数据块
def clump_unmasked(a): ...

# 定义函数 clump_masked，查找数组 a 中的掩码数据块
def clump_masked(a): ...

# 定义函数 vander，生成 Vandermonde 矩阵
def vander(x, n=...): ...

# 定义函数 polyfit，多项式拟合函数
def polyfit(x, y, deg, rcond=..., full=..., w=..., cov=...): ...
```