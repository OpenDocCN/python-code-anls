# `.\pytorch\torch\nn\quantizable\modules\__init__.py`

```py
# 导入多头注意力机制模块（Quantized版本）从torch.ao.nn.quantizable.modules.activation中
from torch.ao.nn.quantizable.modules.activation import MultiheadAttention
# 导入长短时记忆网络模块（Quantized版本）从torch.ao.nn.quantizable.modules.rnn中
from torch.ao.nn.quantizable.modules.rnn import LSTM
# 导入LSTM单元模块（Quantized版本）从torch.ao.nn.quantizable.modules.rnn中

# 定义一个列表，包含了模块的公开接口名称，这些模块用于量化计算
__all__ = [
    'LSTM',  # LSTM模块
    'LSTMCell',  # LSTM单元模块
    'MultiheadAttention',  # 多头注意力机制模块
]
```