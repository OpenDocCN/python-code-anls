# `D:\src\scipysrc\pandas\pandas\api\types\__init__.py`

```
"""
Public toolkit API.
"""

# 从 pandas._libs.lib 中导入 infer_dtype 函数
from pandas._libs.lib import infer_dtype

# 从 pandas.core.dtypes.api 中导入所有内容（除了 F403 错误）
from pandas.core.dtypes.api import *  # noqa: F403

# 从 pandas.core.dtypes.concat 中导入 union_categoricals 函数
from pandas.core.dtypes.concat import union_categoricals

# 从 pandas.core.dtypes.dtypes 中导入以下数据类型类别
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,   # 分类数据类型
    DatetimeTZDtype,    # 带时区的日期时间数据类型
    IntervalDtype,      # 区间数据类型
    PeriodDtype,        # 时期数据类型
)

# 将以下标识符添加到 __all__ 列表中，以便导入时可见的标识符
__all__ = [
    "infer_dtype",       # 推断数据类型的函数
    "union_categoricals",# 合并分类数据的函数
    "CategoricalDtype",  # 分类数据类型类
    "DatetimeTZDtype",   # 带时区的日期时间数据类型类
    "IntervalDtype",     # 区间数据类型类
    "PeriodDtype",       # 时期数据类型类
]
```