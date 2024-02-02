# `Bert-VITS2\oldVersion\V220\models.py`

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
# 从 vector_quantize_pytorch 模块中导入 VectorQuantize 类
from vector_quantize_pytorch import VectorQuantize

# 定义 DurationDiscriminator 类，继承自 nn.Module 类
class DurationDiscriminator(nn.Module):  # vits2
    # 初始化函数，接受输入通道数、滤波器通道数、卷积核大小、丢弃概率、全局输入通道数作为参数
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 设置类的属性
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # 创建丢弃层，设置丢弃概率
        self.drop = nn.Dropout(p_dropout)
        # 创建第一个卷积层，设置输入通道数、滤波器通道数、卷积核大小和填充方式
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建第一个归一化层
        self.norm_1 = modules.LayerNorm(filter_channels)
        # 创建第二个卷积层，设置输入通道数、滤波器通道数、卷积核大小和填充方式
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建第二个归一化层
        self.norm_2 = modules.LayerNorm(filter_channels)
        # 创建持续时间投影层，设置输入通道数、滤波器通道数和卷积核大小
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        # 创建预输出的第一个卷积层，设置输入通道数、滤波器通道数、卷积核大小和填充方式
        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建预输出的第一个归一化层
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        # 创建预输出的第二个卷积层，设置输入通道数、滤波器通道数、卷积核大小和填充方式
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建预输出的第二个归一化层
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        # 如果全局输入通道数不为 0
        if gin_channels != 0:
            # 创建条件卷积层，设置输入通道数、输出通道数和卷积核大小
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        # 创建输出层，包含线性层和 Sigmoid 激活函数
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
# 定义一个名为TextEncoder的神经网络模块
class TextEncoder(nn.Module):
    # 初始化函数，定义模型的各个参数
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
        n_speakers,
        gin_channels=0,
    ):
    # 前向传播函数，接收输入x，x_lengths，tone，language，bert，ja_bert，en_bert，emo，sid，g等参数
    def forward(
        self, x, x_lengths, tone, language, bert, ja_bert, en_bert, emo, sid, g=None
    ):
        # 将sid转移到CPU上
        sid = sid.cpu()
        # 对bert进行投影并转置
        bert_emb = self.bert_proj(bert).transpose(1, 2)
        ja_bert_emb = self.ja_bert_proj(ja_bert).transpose(1, 2)
        en_bert_emb = self.en_bert_proj(en_bert).transpose(1, 2)
        # 对emo进行特征提取
        emo_emb = self.in_feature_net(emo)
        # 对emo_emb进行量化和解码
        emo_emb, _, loss_commit = self.emo_vq(emo_emb.unsqueeze(1))
        loss_commit = loss_commit.mean()
        emo_emb = self.out_feature_net(emo_emb)
        # 计算x的表示，包括各种嵌入和特征
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + bert_emb
            + ja_bert_emb
            + en_bert_emb
            + emo_emb
        ) * math.sqrt(
            self.hidden_channels
        )  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        # 生成x的mask
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        # 对x进行编码
        x = self.encoder(x * x_mask, x_mask, g=g)
        # 对编码结果进行投影
        stats = self.proj(x) * x_mask
        # 将stats分割成均值和标准差
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 返回编码结果x，均值m，标准差logs，x的mask，以及loss_commit
        return x, m, logs, x_mask, loss_commit


# 定义一个名为ResidualCouplingBlock的神经网络模块
class ResidualCouplingBlock(nn.Module):
    # 初始化函数，定义模型的各个参数
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    # 初始化函数，继承父类的初始化方法
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows,
        gin_channels
    ):
        super().__init__()
        # 初始化各种参数
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # 创建流模块列表
        self.flows = nn.ModuleList()
        # 循环创建 n_flows 个 ResidualCouplingLayer 和 Flip 模块
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
        # 如果是反向传播
        else:
            # 对每个流模块进行反向传播
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        # 返回结果
        return x
# 定义后验编码器类，继承自 nn.Module
class PosteriorEncoder(nn.Module):
    # 初始化方法，接受输入通道数、输出通道数、隐藏通道数、卷积核大小、扩张率、层数和 gin 通道数
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化各个参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        # 创建 1 维卷积层，用于预处理输入数据
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        # 创建 WN 模块，用于编码输入数据
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        # 创建 1 维卷积层，用于将编码后的数据映射到输出通道数的两倍
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    # 前向传播方法，接受输入数据 x、输入数据长度 x_lengths 和条件信息 g
    def forward(self, x, x_lengths, g=None):
        # 根据输入数据长度创建掩码
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        # 对输入数据进行预处理并应用掩码
        x = self.pre(x) * x_mask
        # 对预处理后的数据进行编码
        x = self.enc(x, x_mask, g=g)
        # 将编码后的数据映射到输出通道数的两倍，并应用掩码
        stats = self.proj(x) * x_mask
        # 将映射后的数据分割为均值 m 和标准差 logs
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 生成服从正态分布的随机数，乘以标准差并加上均值，再应用掩码
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        # 返回生成的数据 z、均值 m、标准差 logs 和掩码 x_mask
        return z, m, logs, x_mask


# 定义生成器类，继承自 torch.nn.Module
class Generator(torch.nn.Module):
    # 初始化方法，接受初始通道数、残差块、残差块卷积核大小、残差块扩张率、上采样率、上采样初始通道数、上采样卷积核大小和 gin 通道数
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
        # 调用父类的构造函数
        super(Generator, self).__init__()
        # 计算残差块的数量
        self.num_kernels = len(resblock_kernel_sizes)
        # 计算上采样率的数量
        self.num_upsamples = len(upsample_rates)
        # 创建一个一维卷积层，用于预处理
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        # 根据不同的残差块类型选择不同的模块
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        # 创建上采样模块列表
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # 添加上采样模块
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

        # 创建残差块模块列表
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                # 添加残差块模块
                self.resblocks.append(resblock(ch, k, d))

        # 创建后处理的一维卷积层
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # 初始化上采样模块的权重
        self.ups.apply(init_weights)

        # 如果有条件输入通道，则创建一个一维卷积层
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
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
        super(DiscriminatorS, self).__init__()
        # 根据是否使用谱归一化选择不同的归一化函数
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 定义卷积层列表
        self.convs = nn.ModuleList(
            [
                # 使用选择的归一化函数创建卷积层
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        # 使用选择的归一化函数创建后续卷积层
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        # 遍历卷积层列表
        for layer in self.convs:
            x = layer(x)
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
        # 遍历不同周期的Discriminator对象列表
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

    def __init__(self, spec_channels, gin_channels=0):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            # 使用卷积层对输入进行特征提取
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
        # 创建 GRU 模型
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
        n_vocab,  # 词汇表大小
        spec_channels,  # 频谱通道数
        segment_size,  # 分段大小
        inter_channels,  # 中间层通道数
        hidden_channels,  # 隐藏层通道数
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
        n_speakers=256,  # 说话人数量，默认为256
        gin_channels=256,  # GIN通道数，默认为256
        use_sdp=True,  # 是否使用SDP，默认为True
        n_flow_layer=4,  # 流层数量，默认为4
        n_layers_trans_flow=4,  # 转换流层数量，默认为4
        flow_share_parameter=False,  # 流共享参数，默认为False
        use_transformer_flow=True,  # 是否使用Transformer流，默认为True
        **kwargs  # 其他参数
    def forward(
        self,
        x,  # 输入x
        x_lengths,  # 输入x的长度
        y,  # 输入y
        y_lengths,  # 输入y的长度
        sid,  # 说话人ID
        tone,  # 音调
        language,  # 语言
        bert,  # BERT
        ja_bert,  # 日语BERT
        en_bert,  # 英语BERT
        emo=None,  # 情感，可选
    def infer(
        self,
        x,  # 输入x
        x_lengths,  # 输入x的长度
        sid,  # 说话人ID
        tone,  # 音调
        language,  # 语言
        bert,  # BERT
        ja_bert,  # 日语BERT
        en_bert,  # 英语BERT
        emo=None,  # 情感，可选
        noise_scale=0.667,  # 噪声比例，默认为0.667
        length_scale=1,  # 长度比例，默认为1
        noise_scale_w=0.8,  # 噪声比例w，默认为0.8
        max_len=None,  # 最大长度，可选
        sdp_ratio=0,  # SDP比例，默认为0
        y=None,  # 输入y，可选
        # 如果说话者数量大于0
        if self.n_speakers > 0:
            # 通过说话者 ID 获取对应的嵌入向量，并在最后添加一个维度
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            # 通过参考音频进行编码，并在最后添加一个维度
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        # 使用编码器对输入进行编码，获取编码后的结果、概率、对数概率、输入的掩码以及额外信息
        x, m_p, logs_p, x_mask, _ = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, en_bert, emo, sid, g=g
        )
        # 使用自适应路径注意力机制进行计算，得到对数权重
        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
            sdp_ratio
        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)
        # 将对数权重转换为权重，并乘以输入的掩码和长度缩放因子
        w = torch.exp(logw) * x_mask * length_scale
        # 对权重向上取整
        w_ceil = torch.ceil(w)
        # 计算输出的长度
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        # 生成输出的掩码
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        # 生成注意力掩码
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        # 生成注意力权重
        attn = commons.generate_path(w_ceil, attn_mask)

        # 使用注意力权重对编码后的结果进行加权求和
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        # 使用注意力权重对对数概率进行加权求和
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        # 对加权后的结果添加高斯噪声
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        # 使用流模型进行解码，得到输出
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        # 返回输出、注意力权重、输出的掩码以及额外信息
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
```