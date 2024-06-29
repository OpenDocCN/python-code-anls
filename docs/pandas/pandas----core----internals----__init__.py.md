# `D:\src\scipysrc\pandas\pandas\core\internals\__init__.py`

```
# 导入模块中的函数 make_block，该函数在 pyarrow 的某个版本中被使用（2023 年 9 月 18 日的注释）
from pandas.core.internals.api import make_block  # 2023-09-18 pyarrow uses this

# 导入模块中的函数 concatenate_managers，用于连接管理器
from pandas.core.internals.concat import concatenate_managers

# 导入模块中的类 BlockManager 和 SingleBlockManager，这些类用于 pandas 内部数据块管理
from pandas.core.internals.managers import (
    BlockManager,
    SingleBlockManager,
)

# 定义 __all__ 列表，包含了在此模块中希望公开的函数和类的名称
__all__ = [
    "make_block",            # 将 make_block 函数添加到公开接口中
    "BlockManager",          # 将 BlockManager 类添加到公开接口中
    "SingleBlockManager",    # 将 SingleBlockManager 类添加到公开接口中
    "concatenate_managers",  # 将 concatenate_managers 函数添加到公开接口中
]
```