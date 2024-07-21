# `.\pytorch\test\onnx\model_defs\rnn_model_with_packed_sequence.py`

```
from torch import nn
from torch.nn.utils import rnn as rnn_utils

# 使用了 PyTorch 中的 nn.Module 来定义一个支持 packed sequence 的 RNN 模型
class RnnModelWithPackedSequence(nn.Module):
    def __init__(self, model, batch_first):
        super().__init__()
        self.model = model  # 初始化模型
        self.batch_first = batch_first  # 指示输入是否为 batch_first 形式

    def forward(self, input, *args):
        args, seq_lengths = args[:-1], args[-1]  # 从参数中分离出额外的参数和序列长度
        input = rnn_utils.pack_padded_sequence(input, seq_lengths, self.batch_first)
        rets = self.model(input, *args)  # 执行模型的前向传播
        ret, rets = rets[0], rets[1:]  # 分离出第一个返回值和其余返回值
        ret, _ = rnn_utils.pad_packed_sequence(ret, self.batch_first)  # 对 packed sequence 进行解包和填充
        return tuple([ret] + list(rets))  # 返回结果的元组形式


class RnnModelWithPackedSequenceWithoutState(nn.Module):
    def __init__(self, model, batch_first):
        super().__init__()
        self.model = model
        self.batch_first = batch_first

    def forward(self, input, seq_lengths):
        input = rnn_utils.pack_padded_sequence(input, seq_lengths, self.batch_first)
        rets = self.model(input)  # 执行模型的前向传播
        ret, rets = rets[0], rets[1:]  # 分离出第一个返回值和其余返回值
        ret, _ = rnn_utils.pad_packed_sequence(ret, self.batch_first)  # 对 packed sequence 进行解包和填充
        return list([ret] + list(rets))  # 返回结果的列表形式


class RnnModelWithPackedSequenceWithState(nn.Module):
    def __init__(self, model, batch_first):
        super().__init__()
        self.model = model
        self.batch_first = batch_first

    def forward(self, input, hx, seq_lengths):
        input = rnn_utils.pack_padded_sequence(input, seq_lengths, self.batch_first)
        rets = self.model(input, hx)  # 执行模型的前向传播，带有状态 hx
        ret, rets = rets[0], rets[1:]  # 分离出第一个返回值和其余返回值
        ret, _ = rnn_utils.pad_packed_sequence(ret, self.batch_first)  # 对 packed sequence 进行解包和填充
        return list([ret] + list(rets))  # 返回结果的列表形式
```