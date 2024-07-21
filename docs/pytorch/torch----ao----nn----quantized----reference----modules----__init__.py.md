# `.\pytorch\torch\ao\nn\quantized\reference\modules\__init__.py`

```py
# 导入模块中的具体类或函数：从linear模块中导入Linear类
from .linear import Linear
# 导入模块中的具体类或函数：从conv模块中导入Conv1d、Conv2d、Conv3d、ConvTranspose1d、ConvTranspose2d、ConvTranspose3d类
from .conv import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
# 导入模块中的具体类或函数：从rnn模块中导入RNNCell、LSTMCell、GRUCell、LSTM、GRU类或函数
from .rnn import RNNCell, LSTMCell, GRUCell, LSTM, GRU
# 导入模块中的具体类或函数：从sparse模块中导入Embedding、EmbeddingBag类
from .sparse import Embedding, EmbeddingBag

# 将下列类和函数添加到当前模块的公共接口列表中，供外部使用
__all__ = [
    'Linear',  # 线性层类
    'Conv1d',  # 一维卷积类
    'Conv2d',  # 二维卷积类
    'Conv3d',  # 三维卷积类
    'ConvTranspose1d',  # 一维转置卷积类
    'ConvTranspose2d',  # 二维转置卷积类
    'ConvTranspose3d',  # 三维转置卷积类
    'RNNCell',  # 循环神经网络单元类
    'LSTMCell',  # LSTM单元类
    'GRUCell',  # GRU单元类
    'LSTM',  # 长短时记忆网络类
    'GRU',  # 门控循环单元类
    'Embedding',  # 嵌入层类
    'EmbeddingBag',  # 嵌入包层类
]
```