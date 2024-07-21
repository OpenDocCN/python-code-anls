# `.\pytorch\torch\nn\quantized\modules\embedding_ops.py`

```
# flake8: noqa: F401
# 在导入模块时忽略 flake8 的 F401 错误，即未使用的导入警告

r"""Quantized Modules.

This file is in the process of migration to `torch/ao/nn/quantized`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/modules`,
while adding an import statement here.
"""

# 定义导出的模块列表，仅包含三个类名
__all__ = ['EmbeddingPackedParams', 'Embedding', 'EmbeddingBag']

# 从 embedding_ops 模块中导入三个类
from torch.ao.nn.quantized.modules.embedding_ops import Embedding
from torch.ao.nn.quantized.modules.embedding_ops import EmbeddingBag
from torch.ao.nn.quantized.modules.embedding_ops import EmbeddingPackedParams
```