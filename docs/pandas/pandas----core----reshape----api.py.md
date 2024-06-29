# `D:\src\scipysrc\pandas\pandas\core\reshape\api.py`

```
# 从 pandas 库的 reshape 模块中导入 concat 函数
from pandas.core.reshape.concat import concat
# 从 pandas 库的 reshape 模块中导入 from_dummies 和 get_dummies 函数
from pandas.core.reshape.encoding import (
    from_dummies,
    get_dummies,
)
# 从 pandas 库的 reshape 模块中导入 lreshape、melt 和 wide_to_long 函数
from pandas.core.reshape.melt import (
    lreshape,
    melt,
    wide_to_long,
)
# 从 pandas 库的 reshape 模块中导入 merge、merge_asof 和 merge_ordered 函数
from pandas.core.reshape.merge import (
    merge,
    merge_asof,
    merge_ordered,
)
# 从 pandas 库的 reshape 模块中导入 crosstab、pivot 和 pivot_table 函数
from pandas.core.reshape.pivot import (
    crosstab,
    pivot,
    pivot_table,
)
# 从 pandas 库的 reshape 模块中导入 cut 和 qcut 函数
from pandas.core.reshape.tile import (
    cut,
    qcut,
)

# __all__ 列表，列出了该模块中公开的所有函数名
__all__ = [
    "concat",
    "crosstab",
    "cut",
    "from_dummies",
    "get_dummies",
    "lreshape",
    "melt",
    "merge",
    "merge_asof",
    "merge_ordered",
    "pivot",
    "pivot_table",
    "qcut",
    "wide_to_long",
]
```