# `.\pytorch\torch\testing\_internal\opinfo\definitions\__init__.py`

```py
# 忽略类型检查错误，这是为了防止类型检查器(mypy)报告不必要的错误
mypy: ignore-errors

# 从typing模块导入List类型
from typing import List

# 从torch.testing._internal.opinfo.core模块中导入OpInfo类
from torch.testing._internal.opinfo.core import OpInfo
# 从torch.testing._internal.opinfo.definitions模块中导入需要的子模块
from torch.testing._internal.opinfo.definitions import (
    _masked,
    fft,
    linalg,
    signal,
    special,
)

# 操作符数据库，初始化为一个OpInfo对象的列表，包含多个子模块的op_db
op_db: List[OpInfo] = [
    *fft.op_db,        # 添加fft模块的操作符信息
    *linalg.op_db,     # 添加linalg模块的操作符信息
    *signal.op_db,     # 添加signal模块的操作符信息
    *special.op_db,    # 添加special模块的操作符信息
    *_masked.op_db,    # 添加_masked模块的操作符信息
]

# Python参考数据库，初始化为一个OpInfo对象的列表，包含部分子模块的python_ref_db
python_ref_db: List[OpInfo] = [
    *fft.python_ref_db,        # 添加fft模块的Python参考信息
    *linalg.python_ref_db,     # 添加linalg模块的Python参考信息
    *special.python_ref_db,    # 添加special模块的Python参考信息
]
```