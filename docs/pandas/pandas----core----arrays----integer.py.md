# `D:\src\scipysrc\pandas\pandas\core\arrays\integer.py`

```
from __future__ import annotations
# 导入未来支持的类型注释功能，用于声明类型注释的语法

from typing import ClassVar
# 导入ClassVar用于声明类变量的类型

import numpy as np
# 导入NumPy库，用于处理数值数据

from pandas.core.dtypes.base import register_extension_dtype
# 从pandas核心数据类型中导入注册扩展数据类型的函数

from pandas.core.dtypes.common import is_integer_dtype
# 从pandas核心数据类型中导入判断是否为整数数据类型的函数

from pandas.core.arrays.numeric import (
    NumericArray,
    NumericDtype,
)
# 从pandas核心数值数组中导入NumericArray和NumericDtype类

class IntegerDtype(NumericDtype):
    """
    An ExtensionDtype to hold a single size & kind of integer dtype.

    These specific implementations are subclasses of the non-public
    IntegerDtype. For example, we have Int8Dtype to represent signed int 8s.

    The attributes name & type are set when these subclasses are created.
    """

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value = 1
    # 用于填充'_data'以避免向上转型的值

    _default_np_dtype = np.dtype(np.int64)
    # 默认的NumPy数据类型为64位有符号整数

    _checker = is_integer_dtype
    # 检查函数，用于检查是否为整数数据类型的函数引用

    @classmethod
    def construct_array_type(cls) -> type[IntegerArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return IntegerArray
        # 返回与此数据类型关联的数组类型IntegerArray的类对象

    @classmethod
    def _get_dtype_mapping(cls) -> dict[np.dtype, IntegerDtype]:
        # 获取数据类型映射的类方法，返回一个NumPy数据类型到IntegerDtype的字典映射
        return NUMPY_INT_TO_DTYPE  # NUMPY_INT_TO_DTYPE应为全局变量或类的属性

    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        """
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless. e.g. if 'values'
        has a floating dtype, each value must be an integer.
        """
        try:
            return values.astype(dtype, casting="safe", copy=copy)
            # 尝试以"safe"模式将值安全地转换为给定的数据类型dtype
        except TypeError as err:
            casted = values.astype(dtype, copy=copy)
            # 如果转换失败，以普通模式进行转换
            if (casted == values).all():
                return casted
            # 检查是否所有值都相等，若相等则返回转换后的数组，否则抛出错误
            raise TypeError(
                f"cannot safely cast non-equivalent {values.dtype} to {np.dtype(dtype)}"
            ) from err
            # 抛出类型错误，指示无法安全地将不等效的dtype转换为目标dtype


class IntegerArray(NumericArray):
    """
    Array of integer (optional missing) values.

    Uses :attr:`pandas.NA` as the missing value.

    .. warning::

       IntegerArray is currently experimental, and its API or internal
       implementation may change without warning.

    We represent an IntegerArray with 2 numpy arrays:

    - data: contains a numpy integer array of the appropriate dtype
    - mask: a boolean array holding a mask on the data, True is missing

    To construct an IntegerArray from generic array-like input, use
    :func:`pandas.array` with one of the integer dtypes (see examples).

    See :ref:`integer_na` for more.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d integer-dtype array.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values.
    copy : bool, default False
        Whether to copy the `values` and `mask`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    IntegerArray

    Examples
    --------
    Create an IntegerArray with :func:`pandas.array`.

    """

# 这部分代码定义了一个扩展整数数据类型IntegerDtype和整数数组IntegerArray，用于处理整数值及其缺失值的表示和操作。
    # 创建一个包含整数和缺失值的 Pandas 整数数组，使用 Int32 数据类型
    >>> int_array = pd.array([1, None, 3], dtype=pd.Int32Dtype())
    
    # 打印整数数组，展示其中的值及其类型信息
    >>> int_array
    <IntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: Int32
    
    # 还可以使用字符串别名来指定数据类型，这些别名是大写的
    >>> pd.array([1, None, 3], dtype="Int32")
    <IntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: Int32
    
    # 可以使用不同的字符串别名来指定其他整数数据类型，例如 UInt16
    >>> pd.array([1, None, 3], dtype="UInt16")
    <IntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: UInt16
    
    
    
    # 将 IntegerDtype 类赋值给 _dtype_cls 变量
    _dtype_cls = IntegerDtype
# 定义一个多行字符串，包含关于整数数据类型的文档字符串模板，用于各个数据类型的扩展类
_dtype_docstring = """
An ExtensionDtype for {dtype} integer data.

Uses :attr:`pandas.NA` as its missing value, rather than :attr:`numpy.nan`.

Attributes
----------
None

Methods
-------
None

See Also
--------
Int8Dtype : 8-bit nullable integer type.
Int16Dtype : 16-bit nullable integer type.
Int32Dtype : 32-bit nullable integer type.
Int64Dtype : 64-bit nullable integer type.

Examples
--------
For Int8Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.Int8Dtype())
>>> ser.dtype
Int8Dtype()

For Int16Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.Int16Dtype())
>>> ser.dtype
Int16Dtype()

For Int32Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.Int32Dtype())
>>> ser.dtype
Int32Dtype()

For Int64Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.Int64Dtype())
>>> ser.dtype
Int64Dtype()

For UInt8Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.UInt8Dtype())
>>> ser.dtype
UInt8Dtype()

For UInt16Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.UInt16Dtype())
>>> ser.dtype
UInt16Dtype()

For UInt32Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.UInt32Dtype())
>>> ser.dtype
UInt32Dtype()

For UInt64Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.UInt64Dtype())
>>> ser.dtype
UInt64Dtype()
"""

# 创建一个注册为扩展数据类型的类，并指定相应的 numpy 类型和文档字符串
@register_extension_dtype
class Int8Dtype(IntegerDtype):
    type = np.int8
    name: ClassVar[str] = "Int8"
    __doc__ = _dtype_docstring.format(dtype="int8")


@register_extension_dtype
class Int16Dtype(IntegerDtype):
    type = np.int16
    name: ClassVar[str] = "Int16"
    __doc__ = _dtype_docstring.format(dtype="int16")


@register_extension_dtype
class Int32Dtype(IntegerDtype):
    type = np.int32
    name: ClassVar[str] = "Int32"
    __doc__ = _dtype_docstring.format(dtype="int32")


@register_extension_dtype
class Int64Dtype(IntegerDtype):
    type = np.int64
    name: ClassVar[str] = "Int64"
    __doc__ = _dtype_docstring.format(dtype="int64")


@register_extension_dtype
class UInt8Dtype(IntegerDtype):
    type = np.uint8
    name: ClassVar[str] = "UInt8"
    __doc__ = _dtype_docstring.format(dtype="uint8")


@register_extension_dtype
class UInt16Dtype(IntegerDtype):
    type = np.uint16
    name: ClassVar[str] = "UInt16"
    __doc__ = _dtype_docstring.format(dtype="uint16")


@register_extension_dtype
class UInt32Dtype(IntegerDtype):
    type = np.uint32
    name: ClassVar[str] = "UInt32"
    __doc__ = _dtype_docstring.format(dtype="uint32")


@register_extension_dtype
class UInt64Dtype(IntegerDtype):
    type = np.uint64
    name: ClassVar[str] = "UInt64"
    __doc__ = _dtype_docstring.format(dtype="uint64")


# 创建一个映射字典，将 numpy 的整数类型映射到对应的扩展数据类型对象
NUMPY_INT_TO_DTYPE: dict[np.dtype, IntegerDtype] = {
    np.dtype(np.int8): Int8Dtype(),
    np.dtype(np.int16): Int16Dtype(),
    np.dtype(np.int32): Int32Dtype(),
    np.dtype(np.int64): Int64Dtype(),
    np.dtype(np.uint8): UInt8Dtype(),
    np.dtype(np.uint16): UInt16Dtype(),
    np.dtype(np.uint32): UInt32Dtype(),
    np.dtype(np.uint64): UInt64Dtype(),
}
```