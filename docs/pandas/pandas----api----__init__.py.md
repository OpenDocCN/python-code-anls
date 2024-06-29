# `D:\src\scipysrc\pandas\pandas\api\__init__.py`

```
"""public toolkit API"""

# 导入 pandas 的 API 扩展模块
from pandas.api import (
    extensions,    # 导入扩展模块
    indexers,      # 导入索引器模块
    interchange,   # 导入交换模块
    types,         # 导入类型模块
    typing,        # 导入类型提示模块
)

# 定义公开的 API 列表，包含可以被导出的模块名称
__all__ = [
    "interchange",  # 可导出的交换模块
    "extensions",   # 可导出的扩展模块
    "indexers",     # 可导出的索引器模块
    "types",        # 可导出的类型模块
    "typing",       # 可导出的类型提示模块
]
```