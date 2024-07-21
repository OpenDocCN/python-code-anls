# `.\pytorch\test\onnx\model_defs\lstm_flattening_result.py`

```py
# 导入 PyTorch 的 nn 模块中的神经网络定义
from torch import nn
# 导入 nn 模块中用于处理压缩序列的工具类 PackedSequence
from torch.nn.utils.rnn import PackedSequence

# 定义一个自定义的 LSTM 模型，继承自 nn.LSTM 类
class LstmFlatteningResult(nn.LSTM):
    # 重写 forward 方法，接受输入 input 和任意的额外参数 fargs 和 fkwargs
    def forward(self, input, *fargs, **fkwargs):
        # 调用 nn.LSTM 的 forward 方法进行前向传播
        output, (hidden, cell) = nn.LSTM.forward(self, input, *fargs, **fkwargs)
        # 返回 LSTM 输出 output，以及隐藏状态 hidden 和细胞状态 cell
        return output, hidden, cell

# 定义一个带有序列长度信息的 LSTM 模型类，继承自 nn.Module
class LstmFlatteningResultWithSeqLength(nn.Module):
    # 初始化方法，定义输入大小 input_size、隐藏层大小 hidden_size、层数 layers、是否双向 bidirect、dropout 等参数
    def __init__(self, input_size, hidden_size, layers, bidirect, dropout, batch_first):
        super().__init__()
        # 设置是否按照 batch_first 维度进行批处理
        self.batch_first = batch_first
        # 创建一个 nn.LSTM 内部模型对象
        self.inner_model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=batch_first,
        )

    # 重写 forward 方法，接受压缩的序列输入 input 和可选的初始隐藏状态 hx
    def forward(self, input: PackedSequence, hx=None):
        # 调用内部 LSTM 模型的 forward 方法进行前向传播
        output, (hidden, cell) = self.inner_model.forward(input, hx)
        # 返回 LSTM 输出 output，隐藏状态 hidden 和细胞状态 cell
        return output, hidden, cell

# 定义一个不带序列长度信息的 LSTM 模型类，继承自 nn.Module
class LstmFlatteningResultWithoutSeqLength(nn.Module):
    # 初始化方法，定义输入大小 input_size、隐藏层大小 hidden_size、层数 layers、是否双向 bidirect、dropout 等参数
    def __init__(self, input_size, hidden_size, layers, bidirect, dropout, batch_first):
        super().__init__()
        # 设置是否按照 batch_first 维度进行批处理
        self.batch_first = batch_first
        # 创建一个 nn.LSTM 内部模型对象
        self.inner_model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=batch_first,
        )

    # 重写 forward 方法，接受输入 input 和可选的初始隐藏状态 hx
    def forward(self, input, hx=None):
        # 调用内部 LSTM 模型的 forward 方法进行前向传播
        output, (hidden, cell) = self.inner_model.forward(input, hx)
        # 返回 LSTM 输出 output，隐藏状态 hidden 和细胞状态 cell
        return output, hidden, cell
```