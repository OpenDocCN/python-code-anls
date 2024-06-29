# `D:\src\scipysrc\pandas\pandas\_libs\arrays.pyi`

```
# 从 typing 模块导入必要的类型声明
from typing import Sequence

# 导入 numpy 库并重命名为 np
import numpy as np

# 从 pandas._typing 模块导入特定的类型声明
from pandas._typing import (
    AxisInt,
    DtypeObj,
    Self,
    Shape,
)

# 定义 NDArrayBacked 类，用于封装 numpy 数组
class NDArrayBacked:
    # 类变量 _dtype，表示封装的 numpy 数组的数据类型
    _dtype: DtypeObj
    # 类变量 _ndarray，表示封装的 numpy 数组对象
    _ndarray: np.ndarray
    
    # 初始化方法，接受一个 numpy 数组和数据类型作为参数
    def __init__(self, values: np.ndarray, dtype: DtypeObj) -> None: ...

    # 类方法 _simple_new，用于创建 NDArrayBacked 类的新实例
    @classmethod
    def _simple_new(cls, values: np.ndarray, dtype: DtypeObj) -> Self: ...

    # 实例方法 _from_backing_data，从给定的数据创建一个新的实例
    def _from_backing_data(self, values: np.ndarray) -> Self: ...

    # 特殊方法 __setstate__，用于反序列化对象状态
    def __setstate__(self, state) -> None: ...

    # 实例方法 __len__，返回封装的 numpy 数组的长度
    def __len__(self) -> int: ...

    # 属性方法 shape，返回封装的 numpy 数组的形状
    @property
    def shape(self) -> Shape: ...

    # 属性方法 ndim，返回封装的 numpy 数组的维度
    @property
    def ndim(self) -> int: ...

    # 属性方法 size，返回封装的 numpy 数组的元素个数
    @property
    def size(self) -> int: ...

    # 属性方法 nbytes，返回封装的 numpy 数组占用的字节数
    @property
    def nbytes(self) -> int: ...

    # 实例方法 copy，复制当前实例
    def copy(self, order=...) -> Self: ...

    # 实例方法 delete，删除指定轴上的元素
    def delete(self, loc, axis=...) -> Self: ...

    # 实例方法 swapaxes，交换指定的两个轴
    def swapaxes(self, axis1, axis2) -> Self: ...

    # 实例方法 repeat，沿指定轴重复元素
    def repeat(self, repeats: int | Sequence[int], axis: int | None = ...) -> Self: ...

    # 实例方法 reshape，重塑数组的形状
    def reshape(self, *args, **kwargs) -> Self: ...

    # 实例方法 ravel，展平数组
    def ravel(self, order=...) -> Self: ...

    # 属性方法 T，返回数组的转置
    @property
    def T(self) -> Self: ...

    # 类方法 _concat_same_type，用于连接相同类型的实例数组
    @classmethod
    def _concat_same_type(
        cls, to_concat: Sequence[Self], axis: AxisInt = ...
    ) -> Self: ...
```