# `.\pytorch\torch\_subclasses\__init__.py`

```
# 导入 torch 模块
import torch

# 从 torch._subclasses.fake_tensor 模块导入指定的异常和类
from torch._subclasses.fake_tensor import (
    DynamicOutputShapeException,  # 导入 DynamicOutputShapeException 异常类
    FakeTensor,                   # 导入 FakeTensor 类
    FakeTensorMode,               # 导入 FakeTensorMode 枚举类
    UnsupportedFakeTensorException,  # 导入 UnsupportedFakeTensorException 异常类
)

# 从 torch._subclasses.fake_utils 模块导入 CrossRefFakeMode 类
from torch._subclasses.fake_utils import CrossRefFakeMode

# __all__ 列表定义了该模块导出的公共接口，方便使用者知道可以导入哪些对象
__all__ = [
    "FakeTensor",                      # 将 FakeTensor 类添加到 __all__ 列表中
    "FakeTensorMode",                  # 将 FakeTensorMode 枚举类添加到 __all__ 列表中
    "UnsupportedFakeTensorException",  # 将 UnsupportedFakeTensorException 异常类添加到 __all__ 列表中
    "DynamicOutputShapeException",     # 将 DynamicOutputShapeException 异常类添加到 __all__ 列表中
    "CrossRefFakeMode",                # 将 CrossRefFakeMode 类添加到 __all__ 列表中
]
```