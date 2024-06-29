# `D:\src\scipysrc\pandas\pandas\core\arrays\floating.py`

```
from __future__ import annotations
# 引入将来的注解支持，使得在类型提示中可以引用定义的类自身

from typing import ClassVar
# 引入ClassVar用于声明类变量

import numpy as np
# 引入numpy库并使用np作为别名

from pandas.core.dtypes.base import register_extension_dtype
# 从pandas核心数据类型基础模块中引入register_extension_dtype函数

from pandas.core.dtypes.common import is_float_dtype
# 从pandas核心数据类型通用模块中引入is_float_dtype函数

from pandas.core.arrays.numeric import (
    NumericArray,
    NumericDtype,
)
# 从pandas核心数字数组模块中引入NumericArray和NumericDtype类

class FloatingDtype(NumericDtype):
    """
    An ExtensionDtype to hold a single size of floating dtype.

    These specific implementations are subclasses of the non-public
    FloatingDtype. For example we have Float32Dtype to represent float32.

    The attributes name & type are set when these subclasses are created.
    """

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value = np.nan
    # 内部用于填充'_data'以避免向上转型的值，设为np.nan

    _default_np_dtype = np.dtype(np.float64)
    # 默认的numpy数据类型设为np.float64

    _checker = is_float_dtype
    # 设定检查浮点数据类型的函数为is_float_dtype

    @classmethod
    def construct_array_type(cls) -> type[FloatingArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return FloatingArray
        # 返回与此数据类型相关联的数组类型FloatingArray的类对象

    @classmethod
    def _get_dtype_mapping(cls) -> dict[np.dtype, FloatingDtype]:
        return NUMPY_FLOAT_TO_DTYPE
        # 返回一个映射，将numpy数据类型映射到FloatingDtype类的字典

    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        """
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        """
        # This is really only here for compatibility with IntegerDtype
        # Here for compat with IntegerDtype
        return values.astype(dtype, copy=copy)
        # 将数值安全地转换为给定的数据类型dtype，以numpy数组的形式返回

class FloatingArray(NumericArray):
    """
    Array of floating (optional missing) values.

    .. warning::

       FloatingArray is currently experimental, and its API or internal
       implementation may change without warning. Especially the behaviour
       regarding NaN (distinct from NA missing values) is subject to change.

    We represent a FloatingArray with 2 numpy arrays:

    - data: contains a numpy float array of the appropriate dtype
    - mask: a boolean array holding a mask on the data, True is missing

    To construct an FloatingArray from generic array-like input, use
    :func:`pandas.array` with one of the float dtypes (see examples).

    See :ref:`integer_na` for more.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d float-dtype array.
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
    FloatingArray

    Examples
    --------
    Create an FloatingArray with :func:`pandas.array`:

    >>> pd.array([0.1, None, 0.3], dtype=pd.Float32Dtype())
    <FloatingArray>
    [0.1, <NA>, 0.3]
    Length: 3, dtype: Float32

    String aliases for the dtypes are also available. They are capitalized.

    >>> pd.array([0.1, None, 0.3], dtype="Float32")
    <FloatingArray>
    """
    # 表示浮点（可选缺失）值的数组

    def __init__(self, values: np.ndarray, mask: np.ndarray, copy: bool = False):
        super().__init__(values, mask, copy=copy)
        # 调用父类NumericArray的构造函数，初始化FloatingArray对象

# 代码块结束
    # 创建一个包含浮点数和缺失值的 Pandas Series 对象
    [0.1, <NA>, 0.3]
    # Series 包含 3 个元素，数据类型为 Float32
    Length: 3, dtype: Float32
    """
    这是一个多行注释的示例，注释了上面 Series 对象的定义和信息。
    
    # 将 FloatingDtype 类型赋值给 _dtype_cls 变量
    _dtype_cls = FloatingDtype
# 定义一个文档字符串，描述了一个用于 {dtype} 数据的 ExtensionDtype。
# 该 dtype 使用 pd.NA 作为缺失值指示符。

_dtype_docstring = """
An ExtensionDtype for {dtype} data.

This dtype uses ``pd.NA`` as missing value indicator.

Attributes
----------
None

Methods
-------
None

See Also
--------
CategoricalDtype : Type for categorical data with the categories and orderedness.
IntegerDtype : An ExtensionDtype to hold a single size & kind of integer dtype.
StringDtype : An ExtensionDtype for string data.

Examples
--------
For Float32Dtype:

>>> ser = pd.Series([2.25, pd.NA], dtype=pd.Float32Dtype())
>>> ser.dtype
Float32Dtype()

For Float64Dtype:

>>> ser = pd.Series([2.25, pd.NA], dtype=pd.Float64Dtype())
>>> ser.dtype
Float64Dtype()
"""

# 注册 Float32Dtype 类为扩展的浮点型 dtype
@register_extension_dtype
class Float32Dtype(FloatingDtype):
    # 指定 dtype 的类型为 np.float32
    type = np.float32
    # 类名字符串为 "Float32"
    name: ClassVar[str] = "Float32"
    # 设置类的文档字符串为预定义的 _dtype_docstring，填充 dtype 为 "float32"
    __doc__ = _dtype_docstring.format(dtype="float32")


# 注册 Float64Dtype 类为扩展的浮点型 dtype
@register_extension_dtype
class Float64Dtype(FloatingDtype):
    # 指定 dtype 的类型为 np.float64
    type = np.float64
    # 类名字符串为 "Float64"
    name: ClassVar[str] = "Float64"
    # 设置类的文档字符串为预定义的 _dtype_docstring，填充 dtype 为 "float64"
    __doc__ = _dtype_docstring.format(dtype="float64")


# 创建一个映射，将 numpy 的浮点型 dtype 映射到对应的 FloatingDtype 类实例
NUMPY_FLOAT_TO_DTYPE: dict[np.dtype, FloatingDtype] = {
    np.dtype(np.float32): Float32Dtype(),
    np.dtype(np.float64): Float64Dtype(),
}
```