# `.\pytorch\test\onnx\model_defs\word_language_model.py`

```
# 导入必要的模块和类
from typing import Optional, Tuple  # 引入类型提示

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
from torch import Tensor  # 导入张量类型


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        rnn_type,
        ntoken,
        ninp,
        nhid,
        nlayers,
        dropout=0.5,
        tie_weights=False,
        batchsize=2,
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # 定义 dropout 层
        self.encoder = nn.Embedding(ntoken, ninp)  # 定义嵌入层，将输入标记映射为嵌入向量
        # 根据指定的循环神经网络类型创建 RNN 层
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                ) from None
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
            )
        self.decoder = nn.Linear(nhid, ntoken)  # 定义线性层，用于输出层的预测

        # 如果设置了 tie_weights 标志，则将解码器的权重与编码器的权重绑定
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self.init_weights()  # 初始化权重
        self.rnn_type = rnn_type  # 保存循环神经网络的类型
        self.nhid = nhid  # 保存隐藏层单元数
        self.nlayers = nlayers  # 保存循环层的层数
        self.hidden = self.init_hidden(batchsize)  # 初始化隐藏状态

    @staticmethod
    def repackage_hidden(h):
        """Detach hidden states from their history."""
        # 分离隐藏状态，使其不再依赖历史记录
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple([RNNModel.repackage_hidden(v) for v in h])

    def init_weights(self):
        # 初始化权重
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        # 定义前向传播过程
        emb = self.drop(self.encoder(input))  # 嵌入层 + dropout
        output, hidden = self.rnn(emb, hidden)  # RNN 层的前向计算
        output = self.drop(output)  # dropout 输出
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )  # 解码器的前向计算
        self.hidden = RNNModel.repackage_hidden(hidden)  # 分离隐藏状态，以防止梯度传播
        return decoded.view(output.size(0), output.size(1), decoded.size(1))  # 返回解码后的输出
    # 初始化隐藏状态
    def init_hidden(self, bsz):
        # 获取模型参数的第一个参数，获取其数据
        weight = next(self.parameters()).data
        # 如果循环神经网络类型为 LSTM
        if self.rnn_type == "LSTM":
            # 返回一个包含全零张量的元组，形状为 (层数, batch size, 隐藏单元数)
            return (
                weight.new(self.nlayers, bsz, self.nhid).zero_(),
                weight.new(self.nlayers, bsz, self.nhid).zero_(),
            )
        else:
            # 返回一个全零张量，形状为 (层数, batch size, 隐藏单元数)
            return weight.new(self.nlayers, bsz, self.nhid).zero_()
class RNNModelWithTensorHidden(RNNModel):
    """Supports GRU scripting."""

    @staticmethod
    def repackage_hidden(h):
        """Detach hidden states from their history."""
        # 将隐藏状态从其历史中分离出来并返回
        return h.detach()

    def forward(self, input: Tensor, hidden: Tensor):
        # 对输入进行编码并进行 dropout 处理
        emb = self.drop(self.encoder(input))
        # 使用 RNN 进行前向传播，得到输出和更新后的隐藏状态
        output, hidden = self.rnn(emb, hidden)
        # 对输出进行 dropout 处理
        output = self.drop(output)
        # 将输出重塑成适合解码器的形状
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        # 将隐藏状态分离并更新为新的分离状态
        self.hidden = RNNModelWithTensorHidden.repackage_hidden(hidden)
        # 返回解码后的输出
        return decoded.view(output.size(0), output.size(1), decoded.size(1))


class RNNModelWithTupleHidden(RNNModel):
    """Supports LSTM scripting."""

    @staticmethod
    def repackage_hidden(h: Tuple[Tensor, Tensor]):
        """Detach hidden states from their history."""
        # 将元组中的隐藏状态从其历史中分离出来并返回新的元组
        return (h[0].detach(), h[1].detach())

    def forward(self, input: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None):
        # 对输入进行编码并进行 dropout 处理
        emb = self.drop(self.encoder(input))
        # 使用 LSTM 进行前向传播，得到输出和更新后的隐藏状态
        output, hidden = self.rnn(emb, hidden)
        # 对输出进行 dropout 处理
        output = self.drop(output)
        # 将输出重塑成适合解码器的形状
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        # 将隐藏状态分离并更新为新的分离状态
        self.hidden = self.repackage_hidden(tuple(hidden))
        # 返回解码后的输出
        return decoded.view(output.size(0), output.size(1), decoded.size(1))
```