# `D:\src\scipysrc\numpy\numpy\ma\mrecords.pyi`

```py
# 导入类型提示的必要模块
from typing import Any, TypeVar

# 导入 numpy 库中的 dtype 和 MaskedArray 类
from numpy import dtype
from numpy.ma import MaskedArray

# 定义 __all__ 列表，用于指定模块中公开的符号
__all__: list[str]

# 设置 _ShapeType 和 _DType_co 作为类型变量，并指定边界和协变性
_ShapeType = TypeVar("_ShapeType", bound=Any)
_DType_co = TypeVar("_DType_co", bound=dtype[Any], covariant=True)

# 定义 MaskedRecords 类，继承自 MaskedArray 类
class MaskedRecords(MaskedArray[_ShapeType, _DType_co]):
    # 定义 __new__ 方法，用于创建新的 MaskedRecords 实例
    def __new__(
        cls,
        shape,
        dtype=...,
        buf=...,
        offset=...,
        strides=...,
        formats=...,
        names=...,
        titles=...,
        byteorder=...,
        aligned=...,
        mask=...,
        hard_mask=...,
        fill_value=...,
        keep_mask=...,
        copy=...,
        **options,
    ): ...

    # 定义 _mask 属性
    _mask: Any
    # 定义 _fill_value 属性
    _fill_value: Any

    # 定义 _data 属性的属性方法
    @property
    def _data(self): ...

    # 定义 _fieldmask 属性的属性方法
    @property
    def _fieldmask(self): ...

    # 定义 __array_finalize__ 方法，用于继承属性
    def __array_finalize__(self, obj): ...

    # 定义 __len__ 方法，返回 MaskedRecords 的长度
    def __len__(self): ...

    # 定义 __getattribute__ 方法，用于获取属性
    def __getattribute__(self, attr): ...

    # 定义 __setattr__ 方法，用于设置属性
    def __setattr__(self, attr, val): ...

    # 定义 __getitem__ 方法，用于获取元素
    def __getitem__(self, indx): ...

    # 定义 __setitem__ 方法，用于设置元素
    def __setitem__(self, indx, value): ...

    # 定义 view 方法，用于返回类型视图
    def view(self, dtype=..., type=...): ...

    # 定义 harden_mask 方法，用于强化掩码
    def harden_mask(self): ...

    # 定义 soften_mask 方法，用于软化掩码
    def soften_mask(self): ...

    # 定义 copy 方法，用于复制 MaskedRecords 对象
    def copy(self): ...

    # 定义 tolist 方法，将 MaskedRecords 转换为列表
    def tolist(self, fill_value=...): ...

    # 定义 __reduce__ 方法，用于序列化 MaskedRecords 对象
    def __reduce__(self): ...

# 为便捷起见，创建别名 mrecarray 指向 MaskedRecords 类
mrecarray = MaskedRecords

# 定义 fromarrays 函数，从数组列表创建 MaskedRecords 对象
def fromarrays(
    arraylist,
    dtype=...,
    shape=...,
    formats=...,
    names=...,
    titles=...,
    aligned=...,
    byteorder=...,
    fill_value=...,
): ...

# 定义 fromrecords 函数，从记录列表创建 MaskedRecords 对象
def fromrecords(
    reclist,
    dtype=...,
    shape=...,
    formats=...,
    names=...,
    titles=...,
    aligned=...,
    byteorder=...,
    fill_value=...,
    mask=...,
): ...

# 定义 fromtextfile 函数，从文本文件创建 MaskedRecords 对象
def fromtextfile(
    fname,
    delimiter=...,
    commentchar=...,
    missingchar=...,
    varnames=...,
    vartypes=...,
    # 注意: 该参数已在 NumPy 1.22.0 中废弃于 2021-09-23
    # delimitor=...,
): ...

# 定义 addfield 函数，用于向 mrecord 添加新字段
def addfield(mrecord, newfield, newfieldname=...): ...
```