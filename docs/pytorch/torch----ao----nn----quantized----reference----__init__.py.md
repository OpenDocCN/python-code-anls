# `.\pytorch\torch\ao\nn\quantized\reference\__init__.py`

```
# 导入当前包内所有模块，忽略 F403 错误（模块不是通过正规的导入语句导入的情况）
from .modules import *  # noqa: F403

# 指定该模块（当前脚本）中可以导出的符号（变量、函数等）
__all__ = [
    'Linear',               # 线性层（全连接层）
    'Conv1d',               # 一维卷积层
    'Conv2d',               # 二维卷积层
    'Conv3d',               # 三维卷积层
    'ConvTranspose1d',      # 一维转置卷积层
    'ConvTranspose2d',      # 二维转置卷积层
    'ConvTranspose3d',      # 三维转置卷积层
    'RNNCell',              # 循环神经网络单元
    'LSTMCell',             # 长短时记忆网络单元
    'GRUCell',              # 门控循环单元
    'LSTM',                 # 长短时记忆网络
    'GRU',                  # 门控循环网络
    'Embedding',            # 嵌入层
    'EmbeddingBag',         # 带填充的嵌入层
]
```