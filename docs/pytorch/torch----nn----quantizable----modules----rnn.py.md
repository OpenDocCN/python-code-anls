# `.\pytorch\torch\nn\quantizable\modules\rnn.py`

```py
# flake8: noqa: F401
r"""Quantizable Modules.

This file is in the process of migration to `torch/ao/nn/quantizable`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantizable/modules`,
while adding an import statement here.
"""

# 导入需要的模块以支持量化操作的神经网络层
from torch.ao.nn.quantizable.modules.rnn import LSTM
from torch.ao.nn.quantizable.modules.rnn import LSTMCell
```