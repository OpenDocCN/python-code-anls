# `.\pytorch\torch\nn\qat\modules\__init__.py`

```py
# flake8: noqa: F401
r"""QAT Modules.

This package is in the process of being deprecated.
Please, use `torch.ao.nn.qat.modules` instead.
"""

# 导入 QAT 模块中的线性层类
from torch.ao.nn.qat.modules.linear import Linear
# 导入 QAT 模块中的一维卷积类
from torch.ao.nn.qat.modules.conv import Conv1d
# 导入 QAT 模块中的二维卷积类
from torch.ao.nn.qat.modules.conv import Conv2d
# 导入 QAT 模块中的三维卷积类
from torch.ao.nn.qat.modules.conv import Conv3d
# 导入 QAT 模块中的嵌入袋和嵌入类
from torch.ao.nn.qat.modules.embedding_ops import EmbeddingBag, Embedding

# 导入当前目录下的 conv 模块
from . import conv
# 导入当前目录下的 embedding_ops 模块
from . import embedding_ops
# 导入当前目录下的 linear 模块
from . import linear

# 声明当前模块中公开的符号列表
__all__ = [
    "Linear",       # 线性层类
    "Conv1d",       # 一维卷积类
    "Conv2d",       # 二维卷积类
    "Conv3d",       # 三维卷积类
    "Embedding",    # 嵌入类
    "EmbeddingBag", # 嵌入袋类
]
```