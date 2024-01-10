# `Bert-VITS2\oldVersion\V101\models.py`

```
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
# 从当前目录下的 text 模块中导入 symbols、num_tones、num_languages 变量
from .text import symbols, num_tones, num_languages

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
        # 创建一个层归一化层
        self.norm_1 = modules.LayerNorm(filter_channels)
        # 创建一个一维卷积层
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建一个层归一化层
        self.norm_2 = modules.LayerNorm(filter_channels)
        # 创建一个一维卷积层，用于将持续时间投影到指定维度
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        # 创建一个预输出一维卷积层
        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建一个层归一化层
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        # 创建一个预输出一维卷积层
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建一个层归一化层
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        # 如果 gin_channels 不为 0，则创建一个条件一维卷积层
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        # 创建一个包含线性层和 Sigmoid 激活函数的序列模块
        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())
    # 计算前向传播的概率，给定输入 x、输入掩码 x_mask、持续时间 dur 和条件 g（可选）
    def forward_probability(self, x, x_mask, dur, g=None):
        # 使用持续时间的投影进行处理
        dur = self.dur_proj(dur)
        # 将输入 x 和持续时间 dur 连接起来
        x = torch.cat([x, dur], dim=1)
        # 使用预处理卷积层 1 进行处理
        x = self.pre_out_conv_1(x * x_mask)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 使用预处理规范化层 1 进行处理
        x = self.pre_out_norm_1(x)
        # 使用 dropout 进行处理
        x = self.drop(x)
        # 使用预处理卷积层 2 进行处理
        x = self.pre_out_conv_2(x * x_mask)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 使用预处理规范化层 2 进行处理
        x = self.pre_out_norm_2(x)
        # 使用 dropout 进行处理
        x = self.drop(x)
        # 将输入掩码应用到 x 上
        x = x * x_mask
        # 转置 x 的维度
        x = x.transpose(1, 2)
        # 使用输出层进行处理，得到输出概率
        output_prob = self.output_layer(x)
        # 返回输出概率
        return output_prob
    
    # 前向传播函数，给定输入 x、输入掩码 x_mask、真实持续时间 dur_r、预测持续时间 dur_hat 和条件 g（可选）
    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        # 将输入 x 转换为不需要梯度的张量
        x = torch.detach(x)
        # 如果条件 g 存在
        if g is not None:
            # 将条件 g 转换为不需要梯度的张量
            g = torch.detach(g)
            # 将条件 g 作为输入 x 的条件信息
            x = x + self.cond(g)
        # 使用卷积层 1 进行处理
        x = self.conv_1(x * x_mask)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 使用规范化层 1 进行处理
        x = self.norm_1(x)
        # 使用 dropout 进行处理
        x = self.drop(x)
        # 使用卷积层 2 进行处理
        x = self.conv_2(x * x_mask)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 使用规范化层 2 进行处理
        x = self.norm_2(x)
        # 使用 dropout 进行处理
        x = self.drop(x)
    
        # 存储输出概率的列表
        output_probs = []
        # 对于真实持续时间 dur_r 和预测持续时间 dur_hat
        for dur in [dur_r, dur_hat]:
            # 计算前向传播的概率，并添加到输出概率列表中
            output_prob = self.forward_probability(x, x_mask, dur, g)
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

        # 循环 n_flows 次，创建 TransformerCouplingLayer 和 Flip 模块，并添加到 self.flows 中
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
        # 如果 reverse 为 False，则对 self.flows 中的每个模块进行前向传播
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        # 如果 reverse 为 True，则对 self.flows 中的每个模块进行反向传播
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
        # 将 filter_channels 设置为 in_channels，这行代码在将来的版本中需要移除
        filter_channels = in_channels  # it needs to be removed from future version.
        # 设置模型的各个参数
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # 初始化模型的流程
        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        # 循环创建 n_flows 个 ConvFlow 模块和 Flip 模块
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        # 初始化后处理的模块
        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        # 创建后处理的 ConvFlow 模块和 Flip 模块
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        # 初始化预处理的模块
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        # 创建预处理的 ConvFlow 模块和 Flip 模块
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        # 如果有条件输入，创建条件模块
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)
# 定义一个名为 DurationPredictor 的神经网络模型类
class DurationPredictor(nn.Module):
    # 初始化函数，定义模型的参数
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        # 初始化模型的各个参数
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # 初始化模型的各个层
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        # 如果有条件输入通道，则初始化条件输入层
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    # 前向传播函数，定义模型的前向计算过程
    def forward(self, x, x_mask, g=None):
        # 对输入数据进行去梯度操作
        x = torch.detach(x)
        # 如果有条件输入，则对条件输入进行去梯度操作，并将其加到输入数据上
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        # 第一卷积层
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        # 第二卷积层
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        # 投影层
        x = self.proj(x * x_mask)
        # 返回结果
        return x * x_mask


# 定义一个名为 TextEncoder 的神经网络模型类
class TextEncoder(nn.Module):
    # 初始化函数，定义模型的参数
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
    # 初始化函数，设置模型的参数
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
        gin_channels,
    ):
        super().__init__()
        # 设置模型的词汇量大小
        self.n_vocab = n_vocab
        # 设置输出通道数
        self.out_channels = out_channels
        # 设置隐藏层通道数
        self.hidden_channels = hidden_channels
        # 设置滤波器通道数
        self.filter_channels = filter_channels
        # 设置注意力头数
        self.n_heads = n_heads
        # 设置层数
        self.n_layers = n_layers
        # 设置卷积核大小
        self.kernel_size = kernel_size
        # 设置丢弃率
        self.p_dropout = p_dropout
        # 设置GIN通道数
        self.gin_channels = gin_channels
        # 创建词嵌入层，将词汇表中的符号映射为隐藏层通道数维度的向量
        self.emb = nn.Embedding(len(symbols), hidden_channels)
        # 初始化词嵌入层的权重
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        # 创建音调嵌入层，将音调映射为隐藏层通道数维度的向量
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        # 初始化音调嵌入层的权重
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        # 创建语言嵌入层，将语言映射为隐藏层通道数维度的向量
        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        # 初始化语言嵌入层的权重
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)
        # 创建BERT投影层，将BERT的输出进行卷积投影到隐藏层通道数维度
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)

        # 创建编码器层
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        # 创建卷积投影层，将编码器的输出进行卷积投影到输出通道数*2维度
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    # 前向传播函数
    def forward(self, x, x_lengths, tone, language, bert, g=None):
        # 对输入进行词嵌入、音调嵌入、语言嵌入、BERT投影，并进行加权缩放
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + self.bert_proj(bert).transpose(1, 2)
        ) * math.sqrt(
            self.hidden_channels
        )  # [b, t, h]
        # 调换张量维度
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        # 生成掩码
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        # 对输入进行编码
        x = self.encoder(x * x_mask, x_mask, g=g)
        # 对编码后的结果进行卷积投影
        stats = self.proj(x) * x_mask

        # 将结果分割为均值和标准差
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 返回编码结果、均值、标准差、掩码
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

        # 创建流模块的列表
        self.flows = nn.ModuleList()
        # 根据 n_flows 参数循环创建残差耦合层和翻转层
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
            # 对每个流模块进行前向传播
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            # 如果是反向传播，对流模块进行反向传播
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
    # 初始化函数，继承父类的初始化方法
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
        super().__init__()
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置输出通道数
        self.out_channels = out_channels
        # 设置隐藏层通道数
        self.hidden_channels = hidden_channels
        # 设置卷积核大小
        self.kernel_size = kernel_size
        # 设置膨胀率
        self.dilation_rate = dilation_rate
        # 设置卷积层数
        self.n_layers = n_layers
        # 设置GIN通道数
        self.gin_channels = gin_channels

        # 创建一个 1x1 的卷积层，输入通道数为 in_channels，输出通道数为 hidden_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        # 创建一个 WaveNet 模块
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        # 创建一个 1x1 的卷积层，输入通道数为 hidden_channels，输出通道数为 out_channels * 2
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    # 前向传播函数
    def forward(self, x, x_lengths, g=None):
        # 生成输入数据的掩码
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        # 对输入数据进行预处理，并乘以掩码
        x = self.pre(x) * x_mask
        # 对输入数据进行编码
        x = self.enc(x, x_mask, g=g)
        # 对编码后的数据进行投影
        stats = self.proj(x) * x_mask
        # 将投影后的数据分割成均值和标准差
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 生成服从正态分布的随机数，乘以标准差并加上均值，再乘以掩码
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        # 返回生成的数据 z，均值 m，标准差 logs，以及输入数据的掩码
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
        # 创建一个一维卷积层，用于预处理

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
            # 计算通道数
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                # 遍历残差块的核大小和膨胀大小
                self.resblocks.append(resblock(ch, k, d))
                # 添加残差块到残差块列表中

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # 创建一个一维卷积层，用于后处理
        self.ups.apply(init_weights)
        # 对上采样层应用初始化权重的函数

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
            # 如果有条件通道数，则创建一个一维卷积层
    # 定义一个前向传播函数，接受输入 x 和条件 g（可选）
    def forward(self, x, g=None):
        # 对输入 x 进行预处理卷积操作
        x = self.conv_pre(x)
        # 如果条件 g 不为空，则将条件 g 作用于输入 x
        if g is not None:
            x = x + self.cond(g)

        # 循环进行上采样操作
        for i in range(self.num_upsamples):
            # 对输入 x 进行 LeakyReLU 激活函数操作
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 对输入 x 进行上采样操作
            x = self.ups[i](x)
            xs = None
            # 循环进行残差块操作
            for j in range(self.num_kernels):
                # 如果 xs 为空，则将当前残差块的操作结果赋值给 xs
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                # 如果 xs 不为空，则将当前残差块的操作结果加到 xs 上
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # 对 xs 进行均值操作
            x = xs / self.num_kernels
        # 对输入 x 进行 LeakyReLU 激活函数操作
        x = F.leaky_relu(x)
        # 对输入 x 进行后处理卷积操作
        x = self.conv_post(x)
        # 对输入 x 进行 tanh 激活函数操作
        x = torch.tanh(x)

        # 返回处理后的结果 x
        return x

    # 定义一个移除权重归一化的函数
    def remove_weight_norm(self):
        # 打印移除权重归一化的提示信息
        print("Removing weight norm...")
        # 遍历所有上采样层，移除权重归一化
        for l in self.ups:
            remove_weight_norm(l)
        # 遍历所有残差块，移除权重归一化
        for l in self.resblocks:
            l.remove_weight_norm()
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
        # 将输入重新调整形状
        x = x.view(b, c, t // self.period, self.period)

        # 遍历卷积层列表
        for l in self.convs:
            # 对输入进行卷积操作
            x = l(x)
            # 对卷积结果进行 LeakyReLU 激活
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 将激活后的结果添加到特征图列表中
            fmap.append(x)
        # 对最终卷积结果进行后续卷积操作
        x = self.conv_post(x)
        # 将后续卷积结果添加到特征图列表中
        fmap.append(x)
        # 对最终结果进行展平操作
        x = torch.flatten(x, 1, -1)

        # 返回展平后的结果和特征图列表
        return x, fmap
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        # 定义权重归一化函数，如果使用谱归一化则使用谱归一化函数
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 定义卷积层列表
        self.convs = nn.ModuleList(
            [
                # 使用权重归一化函数对卷积层进行归一化
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        # 定义后续的卷积层并使用权重归一化函数进行归一化
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        # 遍历卷积层列表
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        # 创建包含不同周期的DiscriminatorS对象列表
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        # 遍历不同周期的DiscriminatorS对象列表
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    """
    outputs --- [N, ref_enc_gru_size]
    """

    # 初始化函数，设置输入通道数和条件信息通道数
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
        # 计算输出通道数
        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        # 创建 GRU 层
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        # 创建线性投影层
        self.proj = nn.Linear(128, gin_channels)

    # 前向传播函数
    def forward(self, inputs, mask=None):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        # 通过卷积层和激活函数处理输入
        for conv in self.convs:
            out = conv(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    # 计算输出通道数的函数
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
        n_vocab,  # 词汇表大小
        spec_channels,  # 频谱通道数
        segment_size,  # 分段大小
        inter_channels,  # 中间通道数
        hidden_channels,  # 隐藏通道数
        filter_channels,  # 过滤器通道数
        n_heads,  # 多头注意力机制头数
        n_layers,  # 层数
        kernel_size,  # 卷积核大小
        p_dropout,  # 丢弃概率
        resblock,  # 是否使用残差块
        resblock_kernel_sizes,  # 残差块卷积核大小
        resblock_dilation_sizes,  # 残差块膨胀大小
        upsample_rates,  # 上采样率
        upsample_initial_channel,  # 上采样初始通道数
        upsample_kernel_sizes,  # 上采样卷积核大小
        n_speakers=256,  # 说话人数，默认为256
        gin_channels=256,  # GIN通道数，默认为256
        use_sdp=True,  # 是否使用SDP，默认为True
        n_flow_layer=4,  # 流层数，默认为4
        n_layers_trans_flow=3,  # 转换流层数，默认为3
        flow_share_parameter=False,  # 流共享参数，默认为False
        use_transformer_flow=True,  # 是否使用Transformer流，默认为True
        **kwargs  # 其他参数
    def infer(
        self,
        x,  # 输入数据
        x_lengths,  # 输入数据长度
        sid,  # 说话人ID
        tone,  # 音调
        language,  # 语言
        bert,  # BERT
        noise_scale=0.667,  # 噪声比例，默认为0.667
        length_scale=1,  # 长度比例，默认为1
        noise_scale_w=0.8,  # 噪声比例w，默认为0.8
        max_len=None,  # 最大长度，默认为None
        sdp_ratio=0,  # SDP比例，默认为0
        y=None,  # 输出数据，默认为None
        # 如果说话者数量大于0，则使用说话者嵌入作为全局特征
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            # 否则，使用参考编码器对音频进行编码，并添加维度
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        # 使用编码器对音频进行编码，得到音频编码、均值、标准差、掩码
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert, g=g)
        # 使用自注意力机制和动态规划计算对齐路径的对数权重
        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
            sdp_ratio
        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)
        # 计算权重并应用长度缩放和掩码
        w = torch.exp(logw) * x_mask * length_scale
        # 对权重向上取整
        w_ceil = torch.ceil(w)
        # 计算输出长度并生成输出掩码
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        # 生成注意力掩码
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        # 生成对齐路径
        attn = commons.generate_path(w_ceil, attn_mask)

        # 根据对齐路径对均值和标准差进行加权求和
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        # 生成服从正态分布的随机噪声并应用标准差和噪声缩放
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        # 使用逆变换网络生成音频特征
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        # 使用解码器生成最终输出
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        # 返回输出、注意力权重、输出掩码和中间变量
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
```