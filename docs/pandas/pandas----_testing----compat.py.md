# `D:\src\scipysrc\pandas\pandas\_testing\compat.py`

```
"""
Helpers for sharing tests between DataFrame/Series
"""

# 从 __future__ 模块导入 annotations 特性，用于支持注解类型提示
from __future__ import annotations

# 导入类型检查模块，用于在运行时检查类型
from typing import TYPE_CHECKING

# 从 pandas 库中导入 DataFrame 类型
from pandas import DataFrame

# 如果是类型检查模式，导入 DtypeObj 类型
if TYPE_CHECKING:
    from pandas._typing import DtypeObj


# 定义函数 get_dtype，用于获取对象的数据类型
def get_dtype(obj) -> DtypeObj:
    # 如果 obj 是 DataFrame 类型
    if isinstance(obj, DataFrame):
        # 注意：这里假设 DataFrame 只有一列数据
        # 返回 DataFrame 的第一列数据类型
        return obj.dtypes.iat[0]
    else:
        # 返回 obj 的数据类型
        return obj.dtype


# 定义函数 get_obj，用于获取 DataFrame 或其第一列作为 Series
def get_obj(df: DataFrame, klass):
    """
    For sharing tests using frame_or_series, either return the DataFrame
    unchanged or return it's first column as a Series.
    """
    # 如果 klass 是 DataFrame 类型
    if klass is DataFrame:
        # 直接返回原始的 DataFrame
        return df
    # 返回 DataFrame 的第一列数据作为 Series
    return df._ixs(0, axis=1)
```