# `.\pytorch\torch\ao\nn\quantized\dynamic\modules\__init__.py`

```py
# 导入linear模块中的Linear类
from .linear import Linear
# 导入rnn模块中的LSTM, GRU, LSTMCell, RNNCell, GRUCell类
from .rnn import LSTM, GRU, LSTMCell, RNNCell, GRUCell
# 导入conv模块中的Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d类
from .conv import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d

# 定义一个列表，包含了该模块中要导出的所有类名
__all__ = [
    'Linear',              # 线性层类
    'LSTM',                # 长短期记忆网络类
    'GRU',                 # 门控循环单元类
    'LSTMCell',            # 单个长短期记忆单元类
    'RNNCell',             # 单个循环神经网络单元类
    'GRUCell',             # 单个门控循环单元类
    'Conv1d',              # 1维卷积类
    'Conv2d',              # 2维卷积类
    'Conv3d',              # 3维卷积类
    'ConvTranspose1d',     # 1维转置卷积类
    'ConvTranspose2d',     # 2维转置卷积类
    'ConvTranspose3d',     # 3维转置卷积类
]
```