# `.\pytorch\torch\nn\quantized\_reference\modules\__init__.py`

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

# 导入线性模块
from torch.ao.nn.quantized.reference.modules.linear import Linear
# 导入一维卷积模块及其转置模块
from torch.ao.nn.quantized.reference.modules.conv import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
# 导入循环神经网络模块及其单元模块
from torch.ao.nn.quantized.reference.modules.rnn import RNNCell, LSTMCell, GRUCell, LSTM
# 导入稀疏表示模块
from torch.ao.nn.quantized.reference.modules.sparse import Embedding, EmbeddingBag

# 导出的模块列表
__all__ = [
    'Linear',
    'Conv1d',
    'Conv2d',
    'Conv3d',
    'ConvTranspose1d',
    'ConvTranspose2d',
    'ConvTranspose3d',
    'RNNCell',
    'LSTMCell',
    'GRUCell',
    'LSTM',
    'Embedding',
    'EmbeddingBag',
]
```