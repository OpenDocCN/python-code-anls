# `D:\src\scipysrc\seaborn\seaborn\_core\typing.py`

```
# 从 __future__ 导入 annotations 模块，允许在类型注解中使用字符串形式的类型
from __future__ import annotations

# 导入 collections.abc 模块中的 Iterable 和 Mapping 类，用于类型注解
from collections.abc import Iterable, Mapping

# 导入 datetime 模块中的 date, datetime, timedelta 类
from datetime import date, datetime, timedelta

# 导入 typing 模块中的各种类型注解
from typing import Any, Optional, Union, Tuple, List, Dict

# 导入 numpy 模块中的 ndarray 类型
from numpy import ndarray  # TODO use ArrayLike?

# 导入 pandas 模块中的 Series, Index, Timestamp, Timedelta 类型
from pandas import Series, Index, Timestamp, Timedelta

# 导入 matplotlib.colors 模块中的 Colormap, Normalize 类型
from matplotlib.colors import Colormap, Normalize

# 定义 ColumnName 类型，可以是 str, bytes, date, datetime, timedelta, bool, complex,
# Timestamp, Timedelta 中的一种
ColumnName = Union[
    str, bytes, date, datetime, timedelta, bool, complex, Timestamp, Timedelta
]

# 定义 Vector 类型，可以是 Series, Index, ndarray 中的一种
Vector = Union[Series, Index, ndarray]

# 定义 VariableSpec 类型，可以是 ColumnName, Vector, None 中的一种
VariableSpec = Union[ColumnName, Vector, None]

# 定义 VariableSpecList 类型，可以是 List[VariableSpec], Index, None 中的一种
VariableSpecList = Union[List[VariableSpec], Index, None]

# 定义 DataSource 类型，可以是 object, Mapping, None 中的一种
# DataSource 表示数据源，可能是实现了 __dataframe__ 方法的对象，或者是 Mapping，或者为 None
DataSource = Union[object, Mapping, None]

# 定义 OrderSpec 类型，可以是 Iterable, None 中的一种
# OrderSpec 表示排序规格，可能是可迭代对象或者为 None
OrderSpec = Union[Iterable, None]  # TODO technically str is iterable

# 定义 NormSpec 类型，可以是 Tuple[Optional[float], Optional[float]], Normalize, None 中的一种
# NormSpec 表示规范化规格，可能是包含两个可选浮点数的元组，Normalize 对象，或者为 None
NormSpec = Union[Tuple[Optional[float], Optional[float]], Normalize, None]

# 定义 PaletteSpec 类型，可以是 str, list, dict, Colormap, None 中的一种
# PaletteSpec 表示调色板规格，可能是字符串、列表、字典、Colormap 对象，或者为 None
PaletteSpec = Union[str, list, dict, Colormap, None]

# 定义 DiscreteValueSpec 类型，可以是 dict, list, None 中的一种
# DiscreteValueSpec 表示离散值规格，可能是字典、列表，或者为 None
DiscreteValueSpec = Union[dict, list, None]

# 定义 ContinuousValueSpec 类型，可以是 Tuple[float, float], List[float], Dict[Any, float], None 中的一种
# ContinuousValueSpec 表示连续值规格，可能是包含两个浮点数的元组、浮点数列表、任意键的字典值为浮点数的字典，或者为 None
ContinuousValueSpec = Union[
    Tuple[float, float], List[float], Dict[Any, float], None,
]

# 定义 Default 类，实现 __repr__ 方法返回字符串 "<default>"
class Default:
    def __repr__(self):
        return "<default>"

# 定义 Deprecated 类，实现 __repr__ 方法返回字符串 "<deprecated>"
class Deprecated:
    def __repr__(self):
        return "<deprecated>"

# 创建 Default 类的实例 default，用于表示默认值
default = Default()

# 创建 Deprecated 类的实例 deprecated，用于表示已废弃的值
deprecated = Deprecated()
```