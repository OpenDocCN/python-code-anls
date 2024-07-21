# `.\pytorch\torch\autograd\_functions\__init__.py`

```
# 从当前包的 tensor 模块中导入所有内容，忽略 F403 错误（即 F403 表示禁止导入某些内容的警告）
from .tensor import *  # noqa: F403
```