# `.\numpy\numpy\_core\getlimits.pyi`

```py
# 从 numpy 模块中导入特定名称的函数和对象，并使用 as 关键字进行重命名
from numpy import (
    finfo as finfo,  # 导入 finfo 函数并重命名为 finfo
    iinfo as iinfo,  # 导入 iinfo 函数并重命名为 iinfo
)

# 定义一个列表类型的全局变量 __all__，用于指定在使用 from ... import * 时导入的符号（symbols）
__all__: list[str]
```