# `.\pytorch\torch\ao\nn\qat\modules\__init__.py`

```py
# 导入模块中的特定类和函数：从linear模块导入Linear类，从conv模块分别导入Conv1d、Conv2d和Conv3d类，
# 从embedding_ops模块导入EmbeddingBag和Embedding类。
from .linear import Linear
from .conv import Conv1d
from .conv import Conv2d
from .conv import Conv3d
from .embedding_ops import EmbeddingBag, Embedding

# 定义一个列表，包含了该模块中希望对外公开的类和函数的名称。
__all__ = [
    "Linear",         # 将Linear类添加到__all__列表中，使其在模块外可用
    "Conv1d",         # 将Conv1d类添加到__all__列表中，使其在模块外可用
    "Conv2d",         # 将Conv2d类添加到__all__列表中，使其在模块外可用
    "Conv3d",         # 将Conv3d类添加到__all__列表中，使其在模块外可用
    "Embedding",      # 将Embedding类添加到__all__列表中，使其在模块外可用
    "EmbeddingBag",   # 将EmbeddingBag类添加到__all__列表中，使其在模块外可用
]
```