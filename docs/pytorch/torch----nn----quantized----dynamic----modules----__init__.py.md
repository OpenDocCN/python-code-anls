# `.\pytorch\torch\nn\quantized\dynamic\modules\__init__.py`

```py
# flake8: noqa: F401
r"""Quantized Dynamic Modules.

This file is in the process of migration to `torch/ao/nn/quantized/dynamic`,
and is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/dynamic`,
while adding an import statement here.
"""

# 从torch/ao/nn/quantized/dynamic/modules导入conv模块
from torch.ao.nn.quantized.dynamic.modules import conv
# 从torch/ao/nn/quantized/dynamic/modules导入linear模块
from torch.ao.nn.quantized.dynamic.modules import linear
# 从torch/ao/nn/quantized/dynamic/modules导入rnn模块
from torch.ao.nn.quantized.dynamic.modules import rnn

# 从torch/ao/nn/quantized/dynamic/modules/conv导入不同类型的卷积层类
from torch.ao.nn.quantized.dynamic.modules.conv import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
# 从torch/ao/nn/quantized/dynamic/modules/linear导入线性层类
from torch.ao.nn.quantized.dynamic.modules.linear import Linear
# 从torch/ao/nn/quantized/dynamic/modules/rnn导入循环神经网络类和单元类
from torch.ao.nn.quantized.dynamic.modules.rnn import LSTM, GRU, LSTMCell, RNNCell, GRUCell

# 导出给外部使用的类列表
__all__ = [
    'Linear',
    'LSTM',
    'GRU',
    'LSTMCell',
    'RNNCell',
    'GRUCell',
    'Conv1d',
    'Conv2d',
    'Conv3d',
    'ConvTranspose1d',
    'ConvTranspose2d',
    'ConvTranspose3d',
]
```