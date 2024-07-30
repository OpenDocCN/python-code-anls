# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\modules\sequence_modeling.py`

```py
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    """
    双向 LSTM 模型类定义，继承自 nn.Module
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        初始化方法，定义模型结构和参数
        
        Args:
        - input_size: 输入特征的维度
        - hidden_size: LSTM 隐藏状态的维度
        - output_size: 输出特征的维度
        """
        super(BidirectionalLSTM, self).__init__()
        # 定义双向 LSTM 层，输入维度为 input_size，隐藏状态维度为 hidden_size
        # batch_first=True 表示输入数据的第一个维度是 batch_size
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           bidirectional=True,
                           batch_first=True)
        # 线性层，输入维度为 2*hidden_size，输出维度为 output_size
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        前向传播方法
        
        Args:
        - x: 输入张量，维度为 [batch_size x T x input_size]
        
        Returns:
        - output: 输出张量，维度为 [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        # LSTM 前向计算，输入 x 的维度是 [batch_size x T x input_size]
        # 返回 recurrent 是 LSTM 输出的张量，维度为 [batch_size x T x (2*hidden_size)]
        recurrent, _ = self.rnn(x)
        # 使用线性层处理 LSTM 输出，将维度从 [batch_size x T x (2*hidden_size)] 转换为 [batch_size x T x output_size]
        output = self.linear(recurrent)
        return output
```