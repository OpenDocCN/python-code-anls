# `.\pytorch\benchmarks\functional_autograd_benchmark\torchaudio_models.py`

```py
# 导入所需的库和模块
import math  # 导入数学库
from collections import OrderedDict  # 导入有序字典模块
from typing import Optional, Tuple  # 导入类型提示相关的模块

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数库
from torch import nn, Tensor  # 从PyTorch中导入神经网络模块和张量模块

__all__ = ["Wav2Letter"]  # 定义该模块中公开的接口

class Wav2Letter(nn.Module):
    r"""Wav2Letter model architecture from the `"Wav2Letter: an End-to-End ConvNet-based Speech Recognition System"
     <https://arxiv.org/abs/1609.03193>`_ paper.
     :math:`\text{padding} = \frac{\text{ceil}(\text{kernel} - \text{stride})}{2}`
    Args:
        num_classes (int, optional): Number of classes to be classified. (Default: ``40``)
        input_type (str, optional): Wav2Letter can use as input: ``waveform``, ``power_spectrum``
         or ``mfcc`` (Default: ``waveform``).
        num_features (int, optional): Number of input features that the network will receive (Default: ``1``).
    """

    def __init__(
        self, num_classes: int = 40, input_type: str = "waveform", num_features: int = 1
    ):
        # 调用父类构造函数初始化模块
        super(Wav2Letter, self).__init__()
        # 初始化模型中的类别数
        self.num_classes = num_classes
        # 初始化输入类型（波形、功率谱或MFCC）
        self.input_type = input_type
        # 初始化输入特征数量
        self.num_features = num_features
    ) -> None:
        # 调用父类的构造函数初始化对象
        super().__init__()

        # 根据输入类型选择声学特征的数量
        acoustic_num_features = 250 if input_type == "waveform" else num_features
        
        # 定义声学模型的结构
        acoustic_model = nn.Sequential(
            nn.Conv1d(
                in_channels=acoustic_num_features,
                out_channels=250,
                kernel_size=48,
                stride=2,
                padding=23,
            ),
            nn.ReLU(inplace=True),  # 使用ReLU作为激活函数
            nn.Conv1d(
                in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=250, out_channels=2000, kernel_size=32, stride=1, padding=16
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=2000, out_channels=2000, kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=2000,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),  # 最后一层的激活函数，保留非负输出

        )

        # 如果输入类型为波形信号，则添加波形模型的层，并将声学模型作为最终模型
        if input_type == "waveform":
            waveform_model = nn.Sequential(
                nn.Conv1d(
                    in_channels=num_features,
                    out_channels=250,
                    kernel_size=250,
                    stride=160,
                    padding=45,
                ),
                nn.ReLU(inplace=True),
            )
            self.acoustic_model = nn.Sequential(waveform_model, acoustic_model)

        # 如果输入类型为功率谱或MFCC，则直接将声学模型作为最终模型
        if input_type in ["power_spectrum", "mfcc"]:
            self.acoustic_model = acoustic_model
    # 定义前向传播函数，接受一个三维张量作为输入，并返回一个经过预测处理后的张量
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): 维度为 (batch_size, num_features, input_length) 的张量，表示输入数据。
        Returns:
            Tensor: 维度为 (batch_size, number_of_classes, input_length) 的张量，表示预测结果。
        """

        # 使用声学模型处理输入张量 x，得到处理后的张量
        x = self.acoustic_model(x)
        # 对处理后的张量进行对数softmax操作，以获取预测概率
        x = nn.functional.log_softmax(x, dim=1)
        # 返回预测结果张量
        return x
# 从 https://github.com/SeanNaren/deepspeech.pytorch 中获取，并进行修改的代码
class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        将输入的维度从 T*N*H 收缩到 (T*N)*H，并应用于一个模块。
        允许处理可变长度的序列和小批量大小。
        :param module: 要应用输入的模块。
        """
        super().__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        # 改变输入张量的形状以便应用于模块
        x = x.view(t * n, -1)
        x = self.module(x)
        # 恢复原来的形状
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + " (\n"
        tmpstr += self.module.__repr__()
        tmpstr += ")"
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        根据给定的长度向模块的输出添加填充。这是为了确保在推断期间批处理大小变化时，模型的结果不会改变。
        输入需要是形状为 (BxCxDxT) 的张量。
        :param seq_module: 包含卷积堆栈的顺序模块。
        """
        super().__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: 大小为 BxCxDxT 的输入张量
        :param lengths: 批处理中每个序列的实际长度
        :return: 模块的掩码输出
        """
        for module in self.seq_module:
            x = module(x)
            # 创建与 x 相同形状的布尔掩码张量，填充为 0
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                # 将长度之后的部分掩码填充为 1
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            # 使用掩码将 x 中相应位置的值置为 0
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            # 如果不是训练模式，则对输入张量应用 softmax 操作
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        rnn_type=nn.LSTM,
        bidirectional=False,
        batch_norm=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        # 如果启用批归一化，则对输入应用 SequenceWise 包装的批归一化
        self.batch_norm = (
            SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        )
        # 创建 RNN 层，根据参数设置双向性和偏置
        self.rnn = rnn_type(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            bias=True,
        )
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        # 将 RNN 层的参数展开
        self.rnn.flatten_parameters()
    # 定义前向传播函数，接受输入 x 和输出长度 output_lengths
    def forward(self, x, output_lengths):
        # 如果存在批量归一化层，则对输入 x 进行归一化处理
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        # 将输入 x 和输出长度 output_lengths 打包成填充序列
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, enforce_sorted=False)
        
        # 将打包的序列 x 输入到循环神经网络（RNN）中，返回输出 x 和最终隐藏状态 h
        x, h = self.rnn(x)
        
        # 对打包后的序列 x 进行解包成填充序列
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        
        # 如果是双向循环神经网络，则进行下列操作：
        if self.bidirectional:
            # 将 x 的维度调整为 (TxNx2xH)，然后在第二个维度上求和，得到 (TxNxH)
            x = (
                x.view(x.size(0), x.size(1), 2, -1)
                .sum(2)
                .view(x.size(0), x.size(1), -1)
            )  # (TxNxH*2) -> (TxNxH) by sum
        
        # 返回处理后的输出 x
        return x
class Lookahead(nn.Module):
    # Wang et al., 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input

    def __init__(self, n_features, context):
        super().__init__()
        assert context > 0
        self.context = context  # 设置上下文大小，用于Lookahead层
        self.n_features = n_features  # 特征数目
        self.pad = (0, self.context - 1)  # 填充元组，左边0个元素，右边context-1个元素
        self.conv = nn.Conv1d(
            self.n_features,
            self.n_features,
            kernel_size=self.context,  # 卷积核大小为context
            stride=1,
            groups=self.n_features,  # 分组数目为特征数目
            padding=0,
            bias=None,
        )  # 一维卷积层的初始化

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)  # 调整输入张量的维度顺序
        x = F.pad(x, pad=self.pad, value=0)  # 在张量的两端进行零填充，以适应卷积操作
        x = self.conv(x)  # 应用卷积操作
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # 调整张量的维度顺序，使其与输入相同
        return x  # 返回处理后的张量作为输出

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "n_features="
            + str(self.n_features)
            + ", context="
            + str(self.context)
            + ")"
        )  # 返回描述该类实例的字符串表示形式


class DeepSpeech(nn.Module):
    def __init__(
        self,
        rnn_type,
        labels,
        rnn_hidden_size,
        nb_layers,
        audio_conf,
        bidirectional,
        context=20,
    ):
        super().__init__()
        # 初始化DeepSpeech模型的参数
        ):
            # 调用父类的构造函数
            super().__init__()

            # 初始化隐藏层大小
            self.hidden_size = rnn_hidden_size
            # 初始化隐藏层数量
            self.hidden_layers = nb_layers
            # 初始化 RNN 类型
            self.rnn_type = rnn_type
            # 初始化音频配置
            self.audio_conf = audio_conf
            # 初始化标签
            self.labels = labels
            # 是否双向 RNN
            self.bidirectional = bidirectional

            # 提取音频配置中的采样率和窗口大小
            sample_rate = self.audio_conf["sample_rate"]
            window_size = self.audio_conf["window_size"]
            # 计算标签数量
            num_classes = len(self.labels)

            # 定义卷积层
            self.conv = MaskConv(
                nn.Sequential(
                    # 第一层卷积
                    nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                    nn.BatchNorm2d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                    # 第二层卷积
                    nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                    nn.BatchNorm2d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                )
            )

            # 根据卷积层和频谱大小计算 RNN 的输入大小
            rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
            rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
            rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
            rnn_input_size *= 32

            # 初始化 RNN 列表
            rnns = []
            # 第一个 RNN 层
            rnn = BatchRNN(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                batch_norm=False,
            )
            rnns.append(("0", rnn))
            # 其余 RNN 层
            for x in range(nb_layers - 1):
                rnn = BatchRNN(
                    input_size=rnn_hidden_size,
                    hidden_size=rnn_hidden_size,
                    rnn_type=rnn_type,
                    bidirectional=bidirectional,
                )
                rnns.append(("%d" % (x + 1), rnn))
            self.rnns = nn.Sequential(OrderedDict(rnns))

            # 初始化 lookahead 层，如果是单向 RNN
            self.lookahead = (
                nn.Sequential(
                    Lookahead(rnn_hidden_size, context=context),
                    nn.Hardtanh(0, 20, inplace=True),
                )
                if not bidirectional
                else None
            )

            # 全连接层
            fully_connected = nn.Sequential(
                nn.BatchNorm1d(rnn_hidden_size),
                nn.Linear(rnn_hidden_size, num_classes, bias=False),
            )
            self.fc = nn.Sequential(
                SequenceWise(fully_connected),
            )
            # 推断时的 softmax 层
            self.inference_softmax = InferenceBatchSoftmax()
    def forward(self, x, lengths):
        # 将长度转移到 CPU 上，并转换为整型
        lengths = lengths.cpu().int()
        # 根据输入长度获取输出长度
        output_lengths = self.get_seq_lens(lengths)
        # 使用卷积层处理输入 x，并返回处理后的结果和输出长度
        x, _ = self.conv(x, output_lengths)

        # 获取处理后的 x 的尺寸信息
        sizes = x.size()
        # 将 x 的特征维度展平
        x = x.view(
            sizes[0], sizes[1] * sizes[2], sizes[3]
        )  # Collapse feature dimension
        # 调整 x 的维度顺序为 TxNxH
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        # 遍历 RNN 层并处理 x
        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        # 如果不是双向的，则进行前瞻层处理
        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        # 使用全连接层处理最终的 x
        x = self.fc(x)
        # 再次转置 x 的维度顺序
        x = x.transpose(0, 1)
        # 在训练模式下是恒等映射，在评估模式下是 softmax
        x = self.inference_softmax(x)
        return x, output_lengths

    def get_seq_lens(self, input_length):
        """
        给定包含整数序列长度的 1D 张量或变量，返回网络将输出的序列大小的 1D 张量或变量。
        :param input_length: 1D 张量
        :return: 通过模型缩放后的 1D 张量
        """
        # 将输入长度命名为 seq_len
        seq_len = input_length
        # 遍历卷积模块，根据不同的卷积层计算输出的序列长度
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = (
                    seq_len
                    + 2 * m.padding[1]
                    - m.dilation[1] * (m.kernel_size[1] - 1)
                    - 1
                )
                # 计算输出长度并进行浮点数除法
                seq_len = seq_len.true_divide(m.stride[1]) + 1
        # 返回计算得到的输出序列长度，转换为整型
        return seq_len.int()
# 从PyTorch示例代码中导入需要的模块和类
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        # 初始化父类(nn.Module)中的特性
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个全零的张量作为位置编码(pe)，形状为(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成一个位置索引张量，形状为(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算除数项，用于位置编码中的正弦和余弦函数
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # 计算位置编码中的正弦和余弦函数值，填充到位置编码张量(pe)中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 调整位置编码张量(pe)的形状并进行转置，使其符合模型输入要求
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 将位置编码张量(pe)注册为模型的缓冲区，以便在模型保存和加载时保持不变
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        # 将输入序列(x)与位置编码张量(pe)相加，以便将位置信息加入到输入中
        x = x + self.pe[: x.size(0), :]
        # 对输出进行dropout操作，以减少过拟合风险
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        # 尝试导入所需的Transformer相关模块，如果失败则抛出ImportError
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except Exception as e:
            raise ImportError(
                "TransformerEncoder module does not exist in PyTorch 1.1 or lower."
            ) from e
        # 设定模型类型为Transformer
        self.model_type = "Transformer"
        self.src_mask = None
        # 初始化位置编码器(positional encoder)，并传入必要的参数
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        # 初始化Transformer编码器，包含多个TransformerEncoderLayer
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # 初始化词嵌入层(embedding layer)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        # 初始化解码器线性层，将输出维度映射回词汇表大小
        self.decoder = nn.Linear(ninp, ntoken)

        # 初始化模型权重
        self.init_weights()
    # 初始化模型权重
    def init_weights(self):
        # 初始权重范围
        initrange = 0.1
        # 对编码器权重进行均匀分布初始化
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        # 对解码器权重进行均匀分布初始化
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    # 前向传播函数
    def forward(self, src, has_mask=True):
        # 如果有掩码需要处理
        if has_mask:
            device = src.device
            # 如果掩码不存在或者掩码大小不匹配输入数据长度，则创建掩码
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                # 生成一个方形的逐步生成掩码，并将其移到指定设备上
                mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(
                    device
                )
                self.src_mask = mask
        else:
            # 如果没有掩码需求，则置空掩码
            self.src_mask = None

        # 对输入进行编码并乘以缩放因子
        src = self.encoder(src) * math.sqrt(self.ninp)
        # 对位置编码后的输入进行处理
        src = self.pos_encoder(src)
        # 经过变换器编码器处理输入和掩码后的输出
        output = self.transformer_encoder(src, self.src_mask)
        # 对输出进行解码
        output = self.decoder(output)
        # 对输出进行 log_softmax 处理并返回
        return F.log_softmax(output, dim=-1)
# 定义一个多头注意力容器类，用于实现多头注意力机制
class MultiheadAttentionContainer(torch.nn.Module):
    def __init__(self, nhead, in_proj_container, attention_layer, out_proj):
        r"""A multi-head attention container
        Args:
            nhead: the number of heads in the multiheadattention model
            in_proj_container: A container of multi-head in-projection linear layers (a.k.a nn.Linear).
            attention_layer: The attention layer.
            out_proj: The multi-head out-projection layer (a.k.a nn.Linear).
        Examples::
            >>> import torch
            >>> embed_dim, num_heads, bsz = 10, 5, 64
            >>> in_proj_container = InProjContainer(torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim))
            >>> MHA = MultiheadAttentionContainer(num_heads,
                                                  in_proj_container,
                                                  ScaledDotProduct(),
                                                  torch.nn.Linear(embed_dim, embed_dim))
            >>> query = torch.rand((21, bsz, embed_dim))
            >>> key = value = torch.rand((16, bsz, embed_dim))
            >>> attn_output, attn_weights = MHA(query, key, value)
            >>> print(attn_output.shape)
            >>> torch.Size([21, 64, 10])
        """
        super().__init__()
        self.nhead = nhead
        self.in_proj_container = in_proj_container
        self.attention_layer = attention_layer
        self.out_proj = out_proj

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        bias_k: Optional[torch.Tensor] = None,
        bias_v: Optional[torch.Tensor] = None,
    ):
        # 实现多头注意力机制的前向传播
        # 其中 query, key, value 分别为查询、键和值的张量表示
        # attn_mask, bias_k, bias_v 为可选的注意力掩码和偏置张量
        pass  # 在实际使用中，这里会填入具体的多头注意力机制的计算逻辑
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            query, key, value (Tensor): map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            attn_mask, bias_k and bias_v (Tensor, optional): keyword arguments passed to the attention layer.
                See the definitions in the attention.
        Shape:
            - Inputs:
                - query: :math:`(L, N, E)`
                - key: :math:`(S, N, E)`
                - value: :math:`(S, N, E)`
                - attn_mask, bias_k and bias_v: same with the shape of the corresponding args in attention layer.
            - Outputs:
                - attn_output: :math:`(L, N, E)`
                - attn_output_weights: :math:`(N * H, L, S)`
            where where L is the target length, S is the sequence length, H is the number of attention heads,
                N is the batch size, and E is the embedding dimension.
        """
        # Extract dimensions from query tensor
        tgt_len, src_len, bsz, embed_dim = (
            query.size(-3),
            key.size(-3),
            query.size(-2),
            query.size(-1),
        )
        # Project query, key, and value using the in_proj_container method
        q, k, v = self.in_proj_container(query, key, value)
        # Check if query's embedding dimension is divisible by the number of heads
        assert (
            q.size(-1) % self.nhead == 0
        ), "query's embed_dim must be divisible by the number of heads"
        # Calculate dimension of each attention head
        head_dim = q.size(-1) // self.nhead
        # Reshape q tensor for multi-head attention
        q = q.reshape(tgt_len, bsz * self.nhead, head_dim)

        # Check if key's embedding dimension is divisible by the number of heads
        assert (
            k.size(-1) % self.nhead == 0
        ), "key's embed_dim must be divisible by the number of heads"
        # Calculate dimension of each attention head for key tensor
        head_dim = k.size(-1) // self.nhead
        # Reshape k tensor for multi-head attention
        k = k.reshape(src_len, bsz * self.nhead, head_dim)

        # Check if value's embedding dimension is divisible by the number of heads
        assert (
            v.size(-1) % self.nhead == 0
        ), "value's embed_dim must be divisible by the number of heads"
        # Calculate dimension of each attention head for value tensor
        head_dim = v.size(-1) // self.nhead
        # Reshape v tensor for multi-head attention
        v = v.reshape(src_len, bsz * self.nhead, head_dim)

        # Apply the attention layer to q, k, v tensors
        attn_output, attn_output_weights = self.attention_layer(
            q, k, v, attn_mask=attn_mask, bias_k=bias_k, bias_v=bias_v
        )
        # Reshape attention output tensor to original shape
        attn_output = attn_output.reshape(tgt_len, bsz, embed_dim)
        # Project attn_output using the out_proj method
        attn_output = self.out_proj(attn_output)
        # Return the processed attention output and weights
        return attn_output, attn_output_weights
class ScaledDotProduct(torch.nn.Module):
    def __init__(self, dropout=0.0):
        r"""Processes a projected query and key-value pair to apply
        scaled dot product attention.
        Args:
            dropout (float): probability of dropping an attention weight.
        Examples::
            >>> SDP = torchtext.models.ScaledDotProduct(0.1)
            >>> q = torch.randn(256, 21, 3)
            >>> k = v = torch.randn(256, 21, 3)
            >>> attn_output, attn_weights = SDP(q, k, v)
            >>> print(attn_output.shape, attn_weights.shape)
            torch.Size([256, 21, 3]) torch.Size([256, 21, 21])
        """
        super().__init__()
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        bias_k: Optional[torch.Tensor] = None,
        bias_v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute scaled dot-product attention given query, key, and value tensors.
        Args:
            query, key, value (torch.Tensor): Tensors containing sequences of queries, keys, and values.
            attn_mask (torch.Tensor, optional): Mask tensor to avoid attention on certain positions.
            bias_k, bias_v (torch.Tensor, optional): Bias tensors for key and value projections.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing attention output and attention weights.
        """
        # Implementation of scaled dot product attention
        # Step 1: Compute scaled scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        
        # Step 2: Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Step 3: Apply softmax to obtain attention weights
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        # Step 4: Apply dropout
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Step 5: Compute attention output
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights


class InProjContainer(torch.nn.Module):
    def __init__(self, query_proj, key_proj, value_proj):
        r"""A container to process inputs with projection layers.
        Args:
            query_proj: projection layer for query.
            key_proj: projection layer for key.
            value_proj: projection layer for value.
        """
        super().__init__()
        self.query_proj = query_proj
        self.key_proj = key_proj
        self.value_proj = value_proj

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Projects the input sequences using projection layers.
        Args:
            query, key, value (torch.Tensor): Tensors containing sequences of queries, keys, and values.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing projected queries, keys, and values.
        """
        return self.query_proj(query), self.key_proj(key), self.value_proj(value)
```