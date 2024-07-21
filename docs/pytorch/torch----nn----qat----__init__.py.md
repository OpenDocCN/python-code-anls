# `.\pytorch\torch\nn\qat\__init__.py`

```py
# flake8: noqa: F401
r"""QAT Dynamic Modules.

This package is in the process of being deprecated.
Please, use `torch.ao.nn.qat.dynamic` instead.
"""
# 从当前包导入dynamic模块，禁止Flake8对未使用的导入进行警告
from . import dynamic  # noqa: F403
# 从当前包导入modules模块，禁止Flake8对未使用的导入进行警告
from . import modules  # noqa: F403
# 从modules模块中导入所有内容，禁止Flake8对未使用的导入进行警告
from .modules import *  # noqa: F403

# 定义__all__列表，包含所有公开的模块名称
__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "EmbeddingBag",
]
```