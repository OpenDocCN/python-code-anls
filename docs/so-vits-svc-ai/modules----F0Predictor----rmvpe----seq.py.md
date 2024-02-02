# `so-vits-svc\modules\F0Predictor\rmvpe\seq.py`

```py
# 导入 PyTorch 中的神经网络模块
import torch.nn as nn

# 定义双向 GRU 模型
class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        # 调用父类的初始化方法
        super(BiGRU, self).__init__()
        # 创建一个双向 GRU 层
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    # 定义前向传播方法
    def forward(self, x):
        # 对输入数据进行双向 GRU 运算，并返回结果
        return self.gru(x)[0]

# 定义双向 LSTM 模型
class BiLSTM(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        # 调用父类的初始化方法
        super(BiLSTM, self).__init__()
        # 创建一个双向 LSTM 层
        self.lstm = nn.LSTM(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    # 定义前向传播方法
    def forward(self, x):
        # 对输入数据进行双向 LSTM 运算，并返回结果
        return self.lstm(x)[0]
```