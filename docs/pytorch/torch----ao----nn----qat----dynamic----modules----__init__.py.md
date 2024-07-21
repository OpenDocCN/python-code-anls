# `.\pytorch\torch\ao\nn\qat\dynamic\modules\__init__.py`

```py
# 从当前包中导入 linear 模块中的 Linear 类
from .linear import Linear

# 定义一个列表 __all__，包含字符串 "Linear"，表明在使用 from 包名 import * 导入时只导入 Linear 这个符号
__all__ = ["Linear"]
```