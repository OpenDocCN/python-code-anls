# `.\pytorch\torch\nn\qat\modules\embedding_ops.py`

```py
# flake8: noqa: F401
r"""QAT Modules.

This file is in the process of migration to `torch/ao/nn/qat`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/qat/modules`,
while adding an import statement here.
"""

# 导出模块列表，这些模块在当前文件中可以被外部导入使用
__all__ = ['Embedding', 'EmbeddingBag']

# 导入 Embedding 和 EmbeddingBag 模块，以便在当前文件中使用
from torch.ao.nn.qat.modules.embedding_ops import Embedding
from torch.ao.nn.qat.modules.embedding_ops import EmbeddingBag
```