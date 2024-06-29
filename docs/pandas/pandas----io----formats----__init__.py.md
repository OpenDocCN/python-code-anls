# `D:\src\scipysrc\pandas\pandas\io\formats\__init__.py`

```
# 禁止类型检查时忽略 TCH004 错误
# 根据 TYPE_CHECKING 的值进行条件判断，用于在类型检查期间引入类型注解
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 仅当在类型检查时，从 pandas.io.formats 模块导入 style 类型
    from pandas.io.formats import style

    # 将 style 标记为公共导出的模块
    __all__ = ["style"]
```