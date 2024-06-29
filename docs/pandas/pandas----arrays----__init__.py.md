# `D:\src\scipysrc\pandas\pandas\arrays\__init__.py`

```
"""
All of pandas' ExtensionArrays.

See :ref:`extending.extension-types` for more.
"""

# 从 pandas 核心模块导入各种扩展数组类型
from pandas.core.arrays import (
    ArrowExtensionArray,    # 导入 Arrow 扩展数组类型
    ArrowStringArray,       # 导入 Arrow 字符串数组类型
    BooleanArray,           # 导入布尔数组类型
    Categorical,            # 导入分类数据类型
    DatetimeArray,          # 导入日期时间数组类型
    FloatingArray,          # 导入浮点数数组类型
    IntegerArray,           # 导入整数数组类型
    IntervalArray,          # 导入区间数组类型
    NumpyExtensionArray,    # 导入 NumPy 扩展数组类型
    PeriodArray,            # 导入周期（时间段）数组类型
    SparseArray,            # 导入稀疏数组类型
    StringArray,            # 导入字符串数组类型
    TimedeltaArray,         # 导入时间差数组类型
)

# __all__ 列表定义了在模块导入时暴露给外部的符号（变量和类）
__all__ = [
    "ArrowExtensionArray",    # Arrow 扩展数组类型
    "ArrowStringArray",       # Arrow 字符串数组类型
    "BooleanArray",           # 布尔数组类型
    "Categorical",            # 分类数据类型
    "DatetimeArray",          # 日期时间数组类型
    "FloatingArray",          # 浮点数数组类型
    "IntegerArray",           # 整数数组类型
    "IntervalArray",          # 区间数组类型
    "NumpyExtensionArray",    # NumPy 扩展数组类型
    "PeriodArray",            # 周期（时间段）数组类型
    "SparseArray",            # 稀疏数组类型
    "StringArray",            # 字符串数组类型
    "TimedeltaArray",         # 时间差数组类型
]
```