# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\__init__.py`

```py
# 从当前包中导入 process 模块中的指定成员
from .process import (
    uvr5_names,       # 导入 uvr5_names 变量/函数
    uvr,              # 导入 uvr 变量/函数
    uvr_prediction    # 导入 uvr_prediction 变量/函数
)

# 指定当前模块中可以通过 `from package import *` 导入的成员列表
__all__ = [
    "uvr5_names",      # 将 uvr5_names 添加到 `__all__` 列表中
    "uvr",             # 将 uvr 添加到 `__all__` 列表中
    "uvr_prediction"   # 将 uvr_prediction 添加到 `__all__` 列表中
]
```