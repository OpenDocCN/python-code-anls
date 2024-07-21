# `.\pytorch\torch\nn\qat\dynamic\__init__.py`

```
# 禁止 flake8 检查引入的模块中未使用的变量或导入
# 用于包装整段文本，说明这是关于 QAT 动态模块的文档字符串
r"""QAT Dynamic Modules.

This package is in the process of being deprecated.
Please, use `torch.ao.nn.qat.dynamic` instead.
"""
# 从当前包中导入所有模块（除了 F403 指定的例外）
from .modules import *  # noqa: F403
```