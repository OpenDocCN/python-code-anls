# `.\pytorch\torch\ao\nn\quantizable\modules\__init__.py`

```py
# 导入本地的 activation 模块中的 MultiheadAttention 类
from .activation import MultiheadAttention
# 导入本地的 rnn 模块中的 LSTM 类
from .rnn import LSTM
# 导入本地的 rnn 模块中的 LSTMCell 类
from .rnn import LSTMCell

# 定义一个公开接口 __all__，包含需要导出的模块和类名列表
__all__ = [
    'LSTM',  # 导出 LSTM 类
    'LSTMCell',  # 导出 LSTMCell 类
    'MultiheadAttention',  # 导出 MultiheadAttention 类
]
```