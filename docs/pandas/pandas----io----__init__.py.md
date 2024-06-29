# `D:\src\scipysrc\pandas\pandas\io\__init__.py`

```
# ruff: noqa: TCH004
# 引入类型检查模块的标志，表示该文件不需要进行类型检查 TCH004

# 从 typing 模块导入 TYPE_CHECKING 类型
from typing import TYPE_CHECKING

# 如果 TYPE_CHECKING 为真，则执行以下代码块
if TYPE_CHECKING:
    # 从 pandas.io 模块导入以下公共类和函数模块
    from pandas.io import (
        formats,  # 导入 formats 模块
        json,     # 导入 json 模块
        stata,    # 导入 stata 模块
    )

    # 将导入的模块列入公共接口 __all__
    __all__ = ["formats", "json", "stata"]
```