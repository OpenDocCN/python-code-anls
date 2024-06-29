# `D:\src\scipysrc\pandas\pandas\tseries\__init__.py`

```
# 忽略类型检查警告 TCH004，该注释通常在类型检查工具时使用
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 导入具有公共类/函数的模块：
    from pandas.tseries import (
        frequencies,  # 导入频率相关模块
        offsets,      # 导入偏移量相关模块
    )

    # 仅将这些模块标记为公共模块
    __all__ = ["frequencies", "offsets"]
```