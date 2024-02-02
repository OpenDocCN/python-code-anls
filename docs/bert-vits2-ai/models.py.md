# `Bert-VITS2\models.py`

```py
# 导入数学库
import math
# 导入 PyTorch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn 模块中导入 functional 模块并重命名为 F
from torch.nn import functional as F

# 导入自定义的 commons 模块
import commons
# 导入自定义的 modules 模块
import modules
# 导入自定义的 attentions 模块
import attentions
# 导入自定义的 monotonic_align 模块
import monotonic_align

# 从 torch.nn 模块中导入 Conv1d、ConvTranspose1d、Conv2d 类
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
# 从 torch.nn.utils 模块中导入 weight_norm、remove_weight_norm、spectral_norm 函数
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

# 从 commons 模块中导入 init_weights、get_padding 函数
from commons import init_weights, get_padding
# 从 text 模块中导入 symbols、num_tones、num_languages 变量
from text import symbols, num_tones, num_languages

# 定义 DurationDiscriminator 类，继承自 nn.Module 类
class DurationDiscriminator(nn.Module):  # vits2
    # 初始化方法
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化类的属性
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # 创建一个丢弃层，用于随机丢弃输入张量中的部分元素
        self.drop = nn.Dropout(p_dropout)
        # 创建一个一维卷积层
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建一个 LayerNorm 层
        self.norm_1 = modules.LayerNorm(filter_channels)
        # 创建一个一维卷积层
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建一个 LayerNorm 层
        self.norm_2 = modules.LayerNorm(filter_channels)
        # 创建一个一维卷积层，用于持续时间的投影
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        # 创建一个双向 LSTM 层
        self.LSTM = nn.LSTM(
            2 * filter_channels, filter_channels, batch_first=True, bidirectional=True
        )

        # 如果 gin_channels 不为 0，则创建一个一维卷积层
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        # 创建一个包含线性层和 Sigmoid 激活函数的序列模块
        self.output_layer = nn.Sequential(
            nn.Linear(2 * filter_channels, 1), nn.Sigmoid()
        )

    # 前向传播方法，计算输出概率
    def forward_probability(self, x, dur):
        # 对持续时间进行投影
        dur = self.dur_proj(dur)
        # 将输入张量和持续时间张量拼接在一起
        x = torch.cat([x, dur], dim=1)
        # 调换张量的维度
        x = x.transpose(1, 2)
        # 将张量输入到 LSTM 层中
        x, _ = self.LSTM(x)
        # 将 LSTM 层的输出输入到输出层中，计算输出概率
        output_prob = self.output_layer(x)
        # 返回输出概率
        return output_prob
    # 定义一个前向传播函数，接受输入 x、输入掩码 x_mask、实际持续时间 dur_r、预测持续时间 dur_hat 和条件 g（可选）
    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        # 将输入 x 转换为不需要梯度的张量
        x = torch.detach(x)
        # 如果条件 g 存在
        if g is not None:
            # 将条件 g 转换为不需要梯度的张量
            g = torch.detach(g)
            # 将输入 x 加上条件 g 的条件信息
            x = x + self.cond(g)
        # 使用第一个卷积层对输入 x 进行卷积操作
        x = self.conv_1(x * x_mask)
        # 对卷积结果进行 ReLU 激活函数处理
        x = torch.relu(x)
        # 对结果进行归一化处理
        x = self.norm_1(x)
        # 对结果进行随机失活处理
        x = self.drop(x)
        # 使用第二个卷积层对输入 x 进行卷积操作
        x = self.conv_2(x * x_mask)
        # 对卷积结果进行 ReLU 激活函数处理
        x = torch.relu(x)
        # 对结果进行归一化处理
        x = self.norm_2(x)
        # 对结果进行随机失活处理

        # 初始化一个空列表用于存储输出概率
        output_probs = []
        # 遍历实际持续时间和预测持续时间
        for dur in [dur_r, dur_hat]:
            # 调用 forward_probability 函数计算输出概率
            output_prob = self.forward_probability(x, dur)
            # 将计算得到的输出概率添加到列表中
            output_probs.append(output_prob)

        # 返回输出概率列表
        return output_probs
# 定义一个 TransformerCouplingBlock 类，继承自 nn.Module
class TransformerCouplingBlock(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
        share_parameter=False,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化类的属性
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # 创建一个空的 nn.ModuleList 用于存储流的模块
        self.flows = nn.ModuleList()

        # 如果 share_parameter 为 True，则创建一个 FFT 对象并赋值给 self.wn，否则 self.wn 为 None
        self.wn = (
            attentions.FFT(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=self.gin_channels,
            )
            if share_parameter
            else None
        )

        # 循环 n_flows 次，每次创建一个 TransformerCouplingLayer 对象和一个 Flip 对象，并添加到 self.flows 中
        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                    gin_channels=self.gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    # 前向传播函数，接受输入 x, x_mask, g 和 reverse 参数
    def forward(self, x, x_mask, g=None, reverse=False):
        # 如果 reverse 为 False，则对 self.flows 中的每个流进行前向传播
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        # 如果 reverse 为 True，则对 self.flows 中的每个流进行反向传播
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        # 返回结果 x
        return x


# 定义一个 StochasticDurationPredictor 类，继承自 nn.Module
class StochasticDurationPredictor(nn.Module):
    # 初始化函数，设置模型的参数
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 将 filter_channels 设置为 in_channels，这行代码需要在将来的版本中移除
        filter_channels = in_channels  # it needs to be removed from future version.
        # 设置模型的各个参数
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # 初始化 Log 模块
        self.log_flow = modules.Log()
        # 初始化流模块列表
        self.flows = nn.ModuleList()
        # 添加 ElementwiseAffine 模块到流模块列表
        self.flows.append(modules.ElementwiseAffine(2))
        # 循环创建 n_flows 个 ConvFlow 模块和 Flip 模块，添加到流模块列表
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        # 初始化 post_pre 模块
        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        # 初始化 post_proj 模块
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        # 初始化 post_convs 模块
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        # 初始化 post_flows 模块列表
        self.post_flows = nn.ModuleList()
        # 添加 ElementwiseAffine 模块到 post_flows 模块列表
        self.post_flows.append(modules.ElementwiseAffine(2))
        # 循环创建 4 个 ConvFlow 模块和 Flip 模块，添加到 post_flows 模块列表
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        # 初始化 pre 模块
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        # 初始化 proj 模块
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        # 初始化 convs 模块
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        # 如果 gin_channels 不为 0，则初始化 cond 模块
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)
# 定义一个名为 DurationPredictor 的类，继承自 nn.Module
class DurationPredictor(nn.Module):
    # 初始化函数，接受输入通道数、滤波器通道数、卷积核大小、丢失率和 gin 通道数作为参数
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 初始化类的属性
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # 初始化丢失层
        self.drop = nn.Dropout(p_dropout)
        # 初始化第一个卷积层
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 初始化第一个归一化层
        self.norm_1 = modules.LayerNorm(filter_channels)
        # 初始化第二个卷积层
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 初始化第二个归一化层
        self.norm_2 = modules.LayerNorm(filter_channels)
        # 初始化投影层
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        # 如果 gin_channels 不为 0，则初始化条件卷积层
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    # 前向传播函数，接受输入 x、x_mask 和 g（可选）作为参数
    def forward(self, x, x_mask, g=None):
        # 对输入 x 进行去除梯度操作
        x = torch.detach(x)
        # 如果 g 不为 None，则对 g 进行去除梯度操作，并将 x 加上条件卷积层对 g 的处理结果
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        # 对 x 进行第一个卷积操作，并乘以 x_mask
        x = self.conv_1(x * x_mask)
        # 对 x 进行 ReLU 激活函数操作
        x = torch.relu(x)
        # 对 x 进行第一个归一化操作
        x = self.norm_1(x)
        # 对 x 进行丢失操作
        x = self.drop(x)
        # 对 x 进行第二个卷积操作，并乘以 x_mask
        x = self.conv_2(x * x_mask)
        # 对 x 进行 ReLU 激活函数操作
        x = torch.relu(x)
        # 对 x 进行第二个归一化操作
        x = self.norm_2(x)
        # 对 x 进行丢失操作
        x = self.drop(x)
        # 对 x 进行投影操作，并乘以 x_mask
        x = self.proj(x * x_mask)
        # 返回 x 乘以 x_mask 的结果
        return x * x_mask


# 定义一个名为 Bottleneck 的类，继承自 nn.Sequential
class Bottleneck(nn.Sequential):
    # 初始化函数，接受输入维度和隐藏维度作为参数
    def __init__(self, in_dim, hidden_dim):
        # 初始化第一个全连接层
        c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        # 初始化第二个全连接层
        c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        # 调用父类的初始化函数
        super().__init__(* [c_fc1, c_fc2])


# 定义一个名为 Block 的类，继承自 nn.Module
class Block(nn.Module):
    # 初始化函数，接受输入维度和隐藏维度作为参数
    def __init__(self, in_dim, hidden_dim) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 初始化归一化层
        self.norm = nn.LayerNorm(in_dim)
        # 初始化 MLP 模块
        self.mlp = MLP(in_dim, hidden_dim)

    # 前向传播函数，接受输入 x，并返回处理后的结果
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入 x 加上 MLP 模块对归一化层处理后的结果
        x = x + self.mlp(self.norm(x))
        # 返回处理后的结果
        return x


# 定义一个名为 MLP 的类，继承自 nn.Module
class MLP(nn.Module):
    # 初始化神经网络模型的参数
    def __init__(self, in_dim, hidden_dim):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度为in_dim，输出维度为hidden_dim，不使用偏置
        self.c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        # 创建一个线性层，输入维度为in_dim，输出维度为hidden_dim，不使用偏置
        self.c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        # 创建一个线性层，输入维度为hidden_dim，输出维度为in_dim，不使用偏置
        self.c_proj = nn.Linear(hidden_dim, in_dim, bias=False)
    
    # 前向传播函数
    def forward(self, x: torch.Tensor):
        # 使用SILU激活函数对c_fc1的输出进行激活，然后与c_fc2的输出相乘
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        # 将上一步的结果输入到c_proj中
        x = self.c_proj(x)
        # 返回输出结果
        return x
class TextEncoder(nn.Module):
    # 定义一个名为TextEncoder的类，继承自nn.Module
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
    ):
        # 初始化函数，接受多个参数，包括词汇量、输出通道数、隐藏通道数、过滤器通道数、注意力头数、层数、卷积核大小、丢弃概率等
        super().__init__()
        # 调用父类的初始化函数
        self.n_vocab = n_vocab
        # 初始化词汇量
        self.out_channels = out_channels
        # 初始化输出通道数
        self.hidden_channels = hidden_channels
        # 初始化隐藏通道数
        self.filter_channels = filter_channels
        # 初始化过滤器通道数
        self.n_heads = n_heads
        # 初始化注意力头数
        self.n_layers = n_layers
        # 初始化层数
        self.kernel_size = kernel_size
        # 初始化卷积核大小
        self.p_dropout = p_dropout
        # 初始化丢弃概率
        self.gin_channels = gin_channels
        # 初始化GIN通道数，默认为0
        self.emb = nn.Embedding(len(symbols), hidden_channels)
        # 创建一个嵌入层，用于将符号映射到隐藏通道数的向量
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        # 对嵌入层的权重进行正态分布初始化
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        # 创建一个嵌入层，用于将音调映射到隐藏通道数的向量
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        # 对嵌入层的权重进行正态分布初始化
        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        # 创建一个嵌入层，用于将语言映射到隐藏通道数的向量
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)
        # 对嵌入层的权重进行正态分布初始化
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)
        # 创建一个一维卷积层，用于将BERT的输出映射到隐藏通道数的向量
        self.ja_bert_proj = nn.Conv1d(1024, hidden_channels, 1)
        # 创建一个一维卷积层，用于将日语BERT的输出映射到隐藏通道数的向量
        self.en_bert_proj = nn.Conv1d(1024, hidden_channels, 1)
        # 创建一个一维卷积层，用于将英语BERT的输出映射到隐藏通道数的向量

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        # 创建一个名为encoder的注意力模型，接受隐藏通道数、过滤器通道数、注意力头数、层数、卷积核大小、丢弃概率等参数
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        # 创建一个一维卷积层，用于将隐藏通道数映射到输出通道数乘以2的向量
    # 定义一个前向传播函数，接受输入 x、x_lengths、tone、language、bert、ja_bert、en_bert 和可选参数 g
    def forward(self, x, x_lengths, tone, language, bert, ja_bert, en_bert, g=None):
        # 对输入的 bert 数据进行投影，并进行转置
        bert_emb = self.bert_proj(bert).transpose(1, 2)
        # 对输入的 ja_bert 数据进行投影，并进行转置
        ja_bert_emb = self.ja_bert_proj(ja_bert).transpose(1, 2)
        # 对输入的 en_bert 数据进行投影，并进行转置
        en_bert_emb = self.en_bert_proj(en_bert).transpose(1, 2)
        # 将输入 x 进行嵌入操作，并加上 tone、language、bert、ja_bert、en_bert 的嵌入结果，再乘以 sqrt(hidden_channels)
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + bert_emb
            + ja_bert_emb
            + en_bert_emb
        ) * math.sqrt(
            self.hidden_channels
        )  # [b, t, h]
        # 对 x 进行维度转置，将第二维和最后一维交换
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        # 生成 x 的 mask，保留有效长度的部分
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        # 将 x 乘以 mask，然后传入编码器进行处理
        x = self.encoder(x * x_mask, x_mask, g=g)
        # 对处理后的结果进行投影
        stats = self.proj(x) * x_mask
        # 将投影后的结果按照通道数进行分割，得到均值和标准差
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 返回处理后的结果 x、均值 m、标准差 logs 和输入的 mask
        return x, m, logs, x_mask
# 定义残差耦合块的神经网络模块
class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        # 初始化残差耦合块的参数
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # 创建一个模块列表用于存储流的模块
        self.flows = nn.ModuleList()
        # 循环创建指定数量的残差耦合层和翻转层，并添加到模块列表中
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    # 前向传播函数
    def forward(self, x, x_mask, g=None, reverse=False):
        # 如果不是反向传播
        if not reverse:
            # 对每个流中的模块进行前向传播
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            # 如果是反向传播，则对流中的模块进行反向传播
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        # 返回结果
        return x


# 定义后验编码器的神经网络模块
class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    # 初始化函数，设置模型的输入通道数、输出通道数、隐藏通道数、卷积核大小、膨胀率、层数和输入通道数
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型的输入通道数
        self.in_channels = in_channels
        # 设置模型的输出通道数
        self.out_channels = out_channels
        # 设置模型的隐藏通道数
        self.hidden_channels = hidden_channels
        # 设置模型的卷积核大小
        self.kernel_size = kernel_size
        # 设置模型的膨胀率
        self.dilation_rate = dilation_rate
        # 设置模型的层数
        self.n_layers = n_layers
        # 设置模型的输入通道数
        self.gin_channels = gin_channels

        # 创建一个 1 维卷积层，输入通道数为 in_channels，输出通道数为 hidden_channels，卷积核大小为 1
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        # 创建一个 WaveNet 模块，输入通道数为 hidden_channels，卷积核大小为 kernel_size，膨胀率为 dilation_rate，层数为 n_layers，输入通道数为 gin_channels
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        # 创建一个 1 维卷积层，输入通道数为 hidden_channels，输出通道数为 out_channels * 2，卷积核大小为 1
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    # 前向传播函数，接收输入 x、输入长度 x_lengths、全局特征 g（可选）
    def forward(self, x, x_lengths, g=None):
        # 生成输入 x 的掩码，保留有效长度部分
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        # 对输入 x 进行预处理，并乘以掩码
        x = self.pre(x) * x_mask
        # 对输入 x 进行编码，传入掩码和全局特征 g
        x = self.enc(x, x_mask, g=g)
        # 对编码后的结果进行投影，并乘以掩码
        stats = self.proj(x) * x_mask
        # 将投影后的结果按照输出通道数的一半进行分割，得到均值 m 和标准差 logs
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 生成服从正态分布的随机数，乘以标准差 logs，加上均值 m，再乘以掩码，得到最终输出 z
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        # 返回最终输出 z、均值 m、标准差 logs 和掩码
        return z, m, logs, x_mask
class Generator(torch.nn.Module):
    # 定义生成器类，继承自 torch.nn.Module
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        # 初始化函数，接受生成器的各种参数
        super(Generator, self).__init__()
        # 初始化父类的构造函数
        self.num_kernels = len(resblock_kernel_sizes)
        # 记录残差块的核大小数量
        self.num_upsamples = len(upsample_rates)
        # 记录上采样率的数量
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        # 创建一个一维卷积层，用于预处理输入数据

        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2
        # 根据 resblock 参数选择不同的残差块类型

        self.ups = nn.ModuleList()
        # 创建一个空的模块列表，用于存储上采样层
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # 遍历上采样率和核大小
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            # 添加带权重归一化的一维转置卷积层到上采样列表中

        self.resblocks = nn.ModuleList()
        # 创建一个空的模块列表，用于存储残差块
        for i in range(len(self.ups)):
            # 遍历上采样层
            ch = upsample_initial_channel // (2 ** (i + 1))
            # 计算当前通道数
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                # 遍历残差块的核大小和膨胀大小
                self.resblocks.append(resblock(ch, k, d))
                # 添加残差块到残差块列表中

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # 创建一个一维卷积层，用于后处理生成的数据，不使用偏置
        self.ups.apply(init_weights)
        # 对上采样层应用初始化权重的函数

        if gin_channels != 0:
            # 如果条件通道数不为0
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
            # 创建一个一维卷积层，用于条件输入
    # 定义前向传播函数，接受输入 x 和条件 g（可选）
    def forward(self, x, g=None):
        # 使用预处理卷积层处理输入 x
        x = self.conv_pre(x)
        # 如果条件 g 不为空，则将条件 g 作用于输入 x
        if g is not None:
            x = x + self.cond(g)

        # 循环执行上采样操作
        for i in range(self.num_upsamples):
            # 对输入 x 执行 LeakyReLU 激活函数
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 对输入 x 执行上采样操作
            x = self.ups[i](x)
            xs = None
            # 循环执行残差块操作
            for j in range(self.num_kernels):
                # 如果 xs 为空，则将当前残差块的处理结果赋值给 xs
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                # 如果 xs 不为空，则将当前残差块的处理结果加到 xs 上
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # 对 xs 求平均值
            x = xs / self.num_kernels
        # 对最终输出 x 执行 LeakyReLU 激活函数
        x = F.leaky_relu(x)
        # 使用后处理卷积层处理最终输出 x
        x = self.conv_post(x)
        # 对输出 x 执行 tanh 激活函数
        x = torch.tanh(x)

        # 返回处理后的输出 x
        return x

    # 定义移除权重归一化的函数
    def remove_weight_norm(self):
        # 打印提示信息
        print("Removing weight norm...")
        # 遍历上采样层，移除权重归一化
        for layer in self.ups:
            remove_weight_norm(layer)
        # 遍历残差块，移除权重归一化
        for layer in self.resblocks:
            layer.remove_weight_norm()
class DiscriminatorP(torch.nn.Module):
    # 定义一个名为 DiscriminatorP 的类，继承自 torch.nn.Module
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        # 初始化函数，接受 period、kernel_size、stride 和 use_spectral_norm 四个参数
        super(DiscriminatorP, self).__init__()
        # 调用父类的初始化函数
        self.period = period
        # 设置 period 属性为传入的 period 参数
        self.use_spectral_norm = use_spectral_norm
        # 设置 use_spectral_norm 属性为传入的 use_spectral_norm 参数
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 根据 use_spectral_norm 参数选择使用 weight_norm 还是 spectral_norm，并赋值给 norm_f
        self.convs = nn.ModuleList(
            # 创建一个 nn.ModuleList 对象，包含多个卷积层
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 使用 norm_f 包装一个卷积层，并添加到 ModuleList 中
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 使用 norm_f 包装一个卷积层，并添加到 ModuleList 中
                # ...（以下类似，省略）
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        # 使用 norm_f 包装一个卷积层，并赋值给 conv_post 属性
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 初始化特征图列表
        fmap = []

        # 将输入从1维转换为2维
        b, c, t = x.shape
        # 如果时间步 t 不能被周期 period 整除，进行填充
        if t % self.period != 0:  # pad first
            # 计算需要填充的数量
            n_pad = self.period - (t % self.period)
            # 对输入进行反射填充
            x = F.pad(x, (0, n_pad), "reflect")
            # 更新时间步数
            t = t + n_pad
        # 将输入重塑为指定形状
        x = x.view(b, c, t // self.period, self.period)

        # 遍历卷积层列表
        for layer in self.convs:
            # 对输入进行卷积操作
            x = layer(x)
            # 对卷积结果进行 LeakyReLU 激活
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 将激活后的结果添加到特征图列表中
            fmap.append(x)
        # 对卷积后的结果进行后续卷积操作
        x = self.conv_post(x)
        # 将后续卷积结果添加到特征图列表中
        fmap.append(x)
        # 对结果进行展平操作
        x = torch.flatten(x, 1, -1)

        # 返回展平后的结果和特征图列表
        return x, fmap
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        # 初始化 DiscriminatorS 类
        super(DiscriminatorS, self).__init__()
        # 根据 use_spectral_norm 参数选择使用 weight_norm 或 spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 创建卷积层列表
        self.convs = nn.ModuleList(
            [
                # 使用 norm_f 对 Conv1d 进行归一化处理
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        # 创建后续的卷积层
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        # 保存每一层的特征图
        fmap = []

        # 遍历卷积层列表
        for layer in self.convs:
            # 对输入进行卷积操作
            x = layer(x)
            # 使用 LeakyReLU 激活函数
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 将特征图添加到列表中
            fmap.append(x)
        # 对最后一层卷积层进行操作
        x = self.conv_post(x)
        # 将特征图添加到列表中
        fmap.append(x)
        # 对特征图进行展平操作
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        # 初始化 MultiPeriodDiscriminator 类
        super(MultiPeriodDiscriminator, self).__init__()
        # 定义多个周期
        periods = [2, 3, 5, 7, 11]

        # 创建包含不同周期的 DiscriminatorS 实例列表
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        # 将实例列表转换为 ModuleList
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        # 存储真实音频和生成音频的判别结果
        y_d_rs = []
        y_d_gs = []
        # 存储真实音频和生成音频的特征图
        fmap_rs = []
        fmap_gs = []
        # 遍历不同周期的 Discriminator 实例
        for i, d in enumerate(self.discriminators):
            # 获取真实音频的判别结果和特征图
            y_d_r, fmap_r = d(y)
            # 获取生成音频的判别结果和特征图
            y_d_g, fmap_g = d(y_hat)
            # 将判别结果和特征图添加到对应的列表中
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class WavLMDiscriminator(nn.Module):
    """docstring for Discriminator."""
    # 初始化函数，设置初始参数
    def __init__(
        self, slm_hidden=768, slm_layers=13, initial_channel=64, use_spectral_norm=False
    ):
        # 调用父类的初始化函数
        super(WavLMDiscriminator, self).__init__()
        # 根据是否使用谱归一化选择不同的归一化函数
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # 创建预处理层，使用1x1卷积将输入通道数转换为初始通道数
        self.pre = norm_f(
            Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0)
        )

        # 创建卷积层列表
        self.convs = nn.ModuleList(
            [
                # 使用不同的卷积核大小和通道数创建卷积层
                norm_f(
                    nn.Conv1d(
                        initial_channel, initial_channel * 2, kernel_size=5, padding=2
                    )
                ),
                norm_f(
                    nn.Conv1d(
                        initial_channel * 2,
                        initial_channel * 4,
                        kernel_size=5,
                        padding=2,
                    )
                ),
                norm_f(
                    nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)
                ),
            ]
        )

        # 创建后处理层，将通道数转换为1
        self.conv_post = norm_f(Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    # 前向传播函数
    def forward(self, x):
        # 经过预处理层
        x = self.pre(x)

        # 保存每一层的特征图
        fmap = []
        # 遍历卷积层列表
        for l in self.convs:
            # 经过卷积层
            x = l(x)
            # 经过LeakyReLU激活函数
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 将特征图添加到列表中
            fmap.append(x)
        # 经过后处理层
        x = self.conv_post(x)
        # 将多维张量展平为一维张量
        x = torch.flatten(x, 1, -1)

        # 返回结果张量
        return x
class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        # 创建卷积层列表
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)]) # noqa: E501

        # 计算输出通道数
        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        # 创建 GRU 层
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        # 创建线性层
        self.proj = nn.Linear(128, gin_channels)

    def forward(self, inputs, mask=None):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        # 通过卷积层和激活函数处理输入
        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L
class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=256,
        gin_channels=256,
        use_sdp=True,
        n_flow_layer=4,
        n_layers_trans_flow=4,
        flow_share_parameter=False,
        use_transformer_flow=True,
        **kwargs
    ):
    # 初始化合成器模型，设置各种参数和超参数

    def forward(
        self,
        x,
        x_lengths,
        y,
        y_lengths,
        sid,
        tone,
        language,
        bert,
        ja_bert,
        en_bert,
    ):
    # 合成器模型的前向传播，接收输入和标签等参数

    def infer(
        self,
        x,
        x_lengths,
        sid,
        tone,
        language,
        bert,
        ja_bert,
        en_bert,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        max_len=None,
        sdp_ratio=0,
        y=None,
    ):
    # 推断模式下的合成器模型，接收输入和一些超参数
        # 如果说话者数量大于0，则使用说话者嵌入作为全局特征
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            # 否则，使用参考编码器对音频进行编码，并添加维度
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        # 使用编码器对音频进行编码，得到声码器的输出、声码器的log标准差、输入的mask、以及注意力权重
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, en_bert, g=g
        )
        # 使用SDP和DP模块计算log权重
        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
            sdp_ratio
        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)
        # 计算权重
        w = torch.exp(logw) * x_mask * length_scale
        # 对权重向上取整
        w_ceil = torch.ceil(w)
        # 计算输出长度
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        # 生成输出mask
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        # 生成注意力mask
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        # 生成注意力权重
        attn = commons.generate_path(w_ceil, attn_mask)

        # 使用注意力权重对声码器的输出和log标准差进行加权求和
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        # 生成服从正态分布的噪声并与声码器的输出和log标准差相乘
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        # 使用逆变换器对z_p进行逆变换，得到输出
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        # 使用解码器对z进行解码，得到最终输出
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        # 返回最终输出、注意力权重、输出mask、以及(z, z_p, m_p, logs_p)元组
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
```