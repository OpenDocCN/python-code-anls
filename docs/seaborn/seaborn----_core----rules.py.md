# `D:\src\scipysrc\seaborn\seaborn\_core\rules.py`

```
from __future__ import annotations
# 引入 future 模块，以支持类型注释的新语法，即使在旧版本的 Python 中也可以使用

import warnings
# 引入 warnings 模块，用于处理警告信息

from collections import UserString
# 从 collections 模块中引入 UserString 类，用于自定义字符串对象

from numbers import Number
# 从 numbers 模块中引入 Number 类型，用于数字类型的处理

from datetime import datetime
# 从 datetime 模块中引入 datetime 类型，用于日期时间的处理

import numpy as np
# 引入 NumPy 库，并将其命名为 np ，用于数值计算

import pandas as pd
# 引入 Pandas 库，并将其命名为 pd ，用于数据分析和处理

from typing import TYPE_CHECKING
# 从 typing 模块中引入 TYPE_CHECKING ，用于类型检查

if TYPE_CHECKING:
    from typing import Literal
    from pandas import Series
    # 如果在类型检查模式下，则进一步引入 Literal 和 Series 类型定义

class VarType(UserString):
    """
    Prevent comparisons elsewhere in the library from using the wrong name.

    Errors are simple assertions because users should not be able to trigger
    them. If that changes, they should be more verbose.
    """
    # 自定义类 VarType ，继承自 UserString 类

    # 预定义允许的变量类型
    allowed = "numeric", "datetime", "categorical", "boolean", "unknown"

    def __init__(self, data):
        # 构造函数，确保输入数据在允许的类型列表中
        assert data in self.allowed, data
        super().__init__(data)

    def __eq__(self, other):
        # 定义等于操作符，确保比较的对象在允许的类型列表中
        assert other in self.allowed, other
        return self.data == other

def variable_type(
    vector: Series,
    boolean_type: Literal["numeric", "categorical", "boolean"] = "numeric",
    strict_boolean: bool = False,
) -> VarType:
    """
    Determine whether a vector contains numeric, categorical, or datetime data.

    This function differs from the pandas typing API in a few ways:

    - Python sequences or object-typed PyData objects are considered numeric if
      all of their entries are numeric.
    - String or mixed-type data are considered categorical even if not
      explicitly represented as a :class:`pandas.api.types.CategoricalDtype`.
    - There is some flexibility about how to treat binary / boolean data.

    Parameters
    ----------
    vector : :func:`pandas.Series`, :func:`numpy.ndarray`, or Python sequence
        Input data to test.
    boolean_type : 'numeric', 'categorical', or 'boolean'
        Type to use for vectors containing only 0s and 1s (and NAs).
    strict_boolean : bool
        If True, only consider data to be boolean when the dtype is bool or Boolean.

    Returns
    -------
    var_type : 'numeric', 'categorical', or 'datetime'
        Name identifying the type of data in the vector.
    """

    # 如果 vector 的 dtype 是 pd.CategoricalDtype 类型，则判断为 'categorical'
    if isinstance(getattr(vector, 'dtype', None), pd.CategoricalDtype):
        return VarType("categorical")

    # 特殊情况处理全为 NA 的数据，始终判断为 'numeric'
    if pd.isna(vector).all():
        return VarType("numeric")

    # 删除 NA 值，简化后续类型推断过程
    vector = vector.dropna()

    # 特殊情况处理二元/布尔数据，允许调用者确定处理方式
    # 当 vector 包含字符串/对象时，触发 NumPy 警告
    # 当 vector 包含日期时间数据时，触发 DeprecationWarning
    # 据 numpy 的问题追踪网址，此处存在一个与未来警告和废弃警告相关的问题
    # numpy 认为这是一个 bug，并可能会消失。
    with warnings.catch_warnings():
        # 设置警告过滤器，忽略特定类别的警告
        warnings.simplefilter(
            action='ignore',
            category=(FutureWarning, DeprecationWarning)  # type: ignore  # mypy bug?
        )
        
        # 如果 strict_boolean 为真
        if strict_boolean:
            # 如果向量的数据类型是 pandas 的扩展数据类型
            if isinstance(vector.dtype, pd.core.dtypes.base.ExtensionDtype):
                boolean_dtypes = ["bool", "boolean"]
            else:
                boolean_dtypes = ["bool"]
            # 检查向量的数据类型是否在布尔值类型列表中
            boolean_vector = vector.dtype in boolean_dtypes
        else:
            try:
                # 尝试判断向量是否完全是由 0 和 1 组成的布尔向量
                boolean_vector = bool(np.isin(vector, [0, 1]).all())
            except TypeError:
                # 如果在 NumPy 的类型转换规则下无法比较 .isin，通常是由于 'vector' 的数据类型（未知）的原因
                boolean_vector = False
        
        # 如果向量是布尔向量，则返回布尔类型
        if boolean_vector:
            return VarType(boolean_type)

    # 如果向量是数值类型，使用 pandas API 进行检查
    if pd.api.types.is_numeric_dtype(vector):
        return VarType("numeric")

    # 如果向量是日期时间类型，使用 pandas API 进行检查
    if pd.api.types.is_datetime64_dtype(vector):
        return VarType("datetime")

    # --- 如果程序运行到这里，需要检查向量中的条目

    # 检查集合中的所有条目是否都是数字
    def all_numeric(x):
        for x_i in x:
            if not isinstance(x_i, Number):
                return False
        return True

    # 如果所有条目都是数字，则返回数值类型
    if all_numeric(vector):
        return VarType("numeric")

    # 检查集合中的所有条目是否都是日期时间类型
    def all_datetime(x):
        for x_i in x:
            if not isinstance(x_i, (datetime, np.datetime64)):
                return False
        return True

    # 如果所有条目都是日期时间类型，则返回日期时间类型
    if all_datetime(vector):
        return VarType("datetime")

    # 否则，返回分类类型
    return VarType("categorical")
def categorical_order(vector: Series, order: list | None = None) -> list:
    """
    Return a list of unique data values using seaborn's ordering rules.

    Parameters
    ----------
    vector : Series
        Vector of "categorical" values
    order : list
        Desired order of category levels to override the order determined
        from the `data` object.

    Returns
    -------
    order : list
        Ordered list of category levels not including null values.

    """
    # 如果指定了order参数，则直接返回该参数值
    if order is not None:
        return order

    # 如果vector的数据类型为category
    if vector.dtype.name == "category":
        # 使用vector的分类值作为order
        order = list(vector.cat.categories)
    else:
        # 如果vector不是category类型，则筛选出非空值的唯一值作为order
        order = list(filter(pd.notnull, vector.unique()))
        # 检查order的类型，如果是数值类型，则进行排序
        if variable_type(pd.Series(order)) == "numeric":
            order.sort()

    # 返回最终确定的order列表
    return order
```