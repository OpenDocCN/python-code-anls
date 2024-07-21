# `.\pytorch\torch\nn\quantized\_reference\modules\sparse.py`

```
# flake8: noqa: F401
r"""Quantized Reference Modules.

This module is in the process of migration to
`torch/ao/nn/quantized/reference`, and is kept here for
compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/reference`,
while adding an import statement here.
"""

# 从 torch.ao.nn.quantized.reference.modules.sparse 中导入 Embedding 类
from torch.ao.nn.quantized.reference.modules.sparse import Embedding
# 从 torch.ao.nn.quantized.reference.modules.sparse 中导入 EmbeddingBag 类
from torch.ao.nn.quantized.reference.modules.sparse import EmbeddingBag
```