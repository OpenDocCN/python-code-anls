# `D:\src\scipysrc\numpy\numpy\rec\__init__.pyi`

```py
# 导入 numpy._core.records 模块中的特定组件
from numpy._core.records import (
    record as record,              # 导入 record 别名为 record
    recarray as recarray,          # 导入 recarray 别名为 recarray
    format_parser as format_parser,# 导入 format_parser 别名为 format_parser
    fromarrays as fromarrays,      # 导入 fromarrays 别名为 fromarrays
    fromrecords as fromrecords,    # 导入 fromrecords 别名为 fromrecords
    fromstring as fromstring,      # 导入 fromstring 别名为 fromstring
    fromfile as fromfile,          # 导入 fromfile 别名为 fromfile
    array as array                 # 导入 array 别名为 array
)

# 定义 __all__ 变量，指定在使用 from ... import * 时导入的符号列表
__all__: list[str]

# 定义 __path__ 变量，指定模块的搜索路径列表
__path__: list[str]
```