# `D:\src\scipysrc\pandas\pandas\api\indexers\__init__.py`

```
"""
Public API for Rolling Window Indexers.
"""

# 导入 pandas 库中的相关模块和函数
from pandas.core.indexers import check_array_indexer
from pandas.core.indexers.objects import (
    BaseIndexer,
    FixedForwardWindowIndexer,
    VariableOffsetWindowIndexer,
)

# 将以下符号导出到模块的公共接口中，供外部使用
__all__ = [
    "check_array_indexer",                 # 函数：检查数组索引器
    "BaseIndexer",                         # 类：基础索引器
    "FixedForwardWindowIndexer",           # 类：固定前向滚动窗口索引器
    "VariableOffsetWindowIndexer",         # 类：可变偏移滚动窗口索引器
]
```