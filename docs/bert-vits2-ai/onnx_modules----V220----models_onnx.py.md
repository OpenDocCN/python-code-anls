# `Bert-VITS2\onnx_modules\V220\models_onnx.py`

```
# 导入数学库
import math
# 导入 PyTorch 库
import torch
# 从 PyTorch 库中导入 nn 模块
from torch import nn
# 从 PyTorch 的 nn 模块中导入 functional 模块并重命名为 F
from torch.nn import functional as F

# 导入自定义的 commons 模块
import commons
# 导入自定义的 modules 模块
import modules
# 从当前目录下的 attentions_onnx 模块中导入所有内容
from . import attentions_onnx
# 从 vector_quantize_pytorch 模块中导入 VectorQuantize 类
from vector_quantize_pytorch import VectorQuantize

# 从 PyTorch 的 nn 模块中导入 Conv1d、ConvTranspose1d、Conv2d 类
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
# 从 PyTorch 的 nn.utils 模块中导入 weight_norm、remove_weight_norm、spectral_norm 函数
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

        # 创建一个丢弃层，用于随机丢弃输入张量的部分元素
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
        # 创建一个一维卷积层
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        # 创建一个一维卷积层
        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建一个 LayerNorm 层
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        # 创建一个一维卷积层
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建一个 LayerNorm 层
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        # 如果 gin_channels 不为 0，则创建一个一维卷积层
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

        # 如果 share_parameter 为 True，则创建一个 FFT 模块，否则为 None
        self.wn = (
            attentions_onnx.FFT(
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

        # 循环创建 n_flows 个 TransformerCouplingLayer 和 Flip 模块，并添加到 flows 中
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
    def forward(self, x, x_mask, g=None, reverse=True):
        # 如果 reverse 为 False，则按顺序对 flows 中的模块进行前向传播
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        # 如果 reverse 为 True，则按相反顺序对 flows 中的模块进行前向传播
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        # 返回前向传播的结果 x
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

        # 初始化模型的一些组件
        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        # 循环创建 n_flows 个 ConvFlow 模块和 Flip 模块
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        # 初始化后处理的一些组件
        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        # 循环创建 4 个 ConvFlow 模块和 Flip 模块
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        # 初始化前处理的一些组件
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        # 如果 gin_channels 不为 0，则初始化条件模块
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)
    # 定义一个前向传播函数，接受输入 x, x_mask, z, g（可选）
    def forward(self, x, x_mask, z, g=None):
        # 将输入 x 转换为不需要梯度的张量
        x = torch.detach(x)
        # 使用预处理层对输入 x 进行处理
        x = self.pre(x)
        # 如果 g 不为空
        if g is not None:
            # 将 g 转换为不需要梯度的张量
            g = torch.detach(g)
            # 将条件 g 作用于输入 x
            x = x + self.cond(g)
        # 使用卷积层对输入 x 进行处理
        x = self.convs(x, x_mask)
        # 使用投影层对输入 x 进行处理，并乘以输入 x 的掩码
        x = self.proj(x) * x_mask

        # 将流列表反转，并去除最后一个元素（一个无用的 vflow）
        flows = list(reversed(self.flows))
        flows = flows[:-2] + [flows[-1]]
        # 遍历流列表中的每个流
        for flow in flows:
            # 对 z 进行反向传播
            z = flow(z, x_mask, g=x, reverse=True)
        # 将 z 拆分成 z0 和 z1
        z0, z1 = torch.split(z, [1, 1], 1)
        # 将 z0 作为 logw 返回
        logw = z0
        return logw
# 定义一个名为 DurationPredictor 的神经网络模型类
class DurationPredictor(nn.Module):
    # 初始化函数，定义模型的参数和层
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        # 初始化模型的参数
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # 初始化模型的层
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

        # 如果有条件输入通道，则添加条件输入层
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    # 前向传播函数，定义模型的前向计算过程
    def forward(self, x, x_mask, g=None):
        # 对输入进行去除梯度操作
        x = torch.detach(x)
        # 如果有条件输入，则对条件输入进行去除梯度操作，并将其加到输入上
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        # 第一层卷积操作
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        # 第二层卷积操作
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        # 投影层操作
        x = self.proj(x * x_mask)
        return x * x_mask


# 定义一个名为 Bottleneck 的神经网络模型类，继承自 nn.Sequential
class Bottleneck(nn.Sequential):
    # 初始化函数，定义模型的线性层
    def __init__(self, in_dim, hidden_dim):
        c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        super().__init__(* [c_fc1, c_fc2])


# 定义一个名为 Block 的神经网络模型类，继承自 nn.Module
class Block(nn.Module):
    # 初始化函数，定义模型的层
    def __init__(self, in_dim, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, hidden_dim)

    # 前向传播函数，定义模型的前向计算过程
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(self.norm(x))
        return x


# 定义一个名为 MLP 的神经网络模型类，继承自 nn.Module
class MLP(nn.Module):
    # 这里需要继续添加 MLP 类的初始化函数和前向传播函数
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
        # 初始化文本编码器模型
        def forward(
            self, x, x_lengths, tone, language, bert, ja_bert, en_bert, emo, g=None
        ):
            # 创建一个与输入张量 x 相同形状的全为1的掩码张量
            x_mask = torch.ones_like(x).unsqueeze(0)
            # 对 BERT 输出进行投影，并转置维度
            bert_emb = self.bert_proj(bert.transpose(0, 1).unsqueeze(0)).transpose(1, 2)
            ja_bert_emb = self.ja_bert_proj(ja_bert.transpose(0, 1).unsqueeze(0)).transpose(
                1, 2
            )
            en_bert_emb = self.en_bert_proj(en_bert.transpose(0, 1).unsqueeze(0)).transpose(
                1, 2
            )
            # 对情感向量进行特征提取和量化
            emo_emb = self.in_feature_net(emo.transpose(0, 1))
            emo_emb, _, _ = self.emo_vq(emo_emb.unsqueeze(1))
            # 对情感向量进行特征提取
            emo_emb = self.out_feature_net(emo_emb)
            # 将各种嵌入向量相加，并乘以一个缩放因子
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
            # 转置张量的特定维度
            x = torch.transpose(x, 1, -1)  # [b, h, t]
            x_mask = x_mask.to(x.dtype)
            # 使用编码器对输入进行编码
            x = self.encoder(x * x_mask, x_mask, g=g)
            # 对编码结果进行投影
            stats = self.proj(x) * x_mask
            # 将结果张量按通道数分割成两部分
            m, logs = torch.split(stats, self.out_channels, dim=1)
            return x, m, logs, x_mask


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
        # 初始化残差耦合块模型
        # 初始化函数，继承父类的初始化方法
        super().__init__()
        # 设置神经网络的通道数
        self.channels = channels
        # 设置隐藏层的通道数
        self.hidden_channels = hidden_channels
        # 设置卷积核大小
        self.kernel_size = kernel_size
        # 设置膨胀率
        self.dilation_rate = dilation_rate
        # 设置流的层数
        self.n_layers = n_layers
        # 设置流的数量
        self.n_flows = n_flows
        # 设置GIN通道数
        self.gin_channels = gin_channels

        # 创建流的模块列表
        self.flows = nn.ModuleList()
        # 循环创建流的模块
        for i in range(n_flows):
            # 添加残差耦合层模块到流的模块列表
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
            # 添加翻转模块到流的模块列表
            self.flows.append(modules.Flip())

    # 前向传播函数
    def forward(self, x, x_mask, g=None, reverse=True):
        # 如果是反向传播
        if not reverse:
            # 对每个流的模块进行前向传播
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        # 如果是正向传播
        else:
            # 对每个流的模块进行反向传播
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        # 返回传播结果
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

    # 前向传播方法，接受输入数据 x、数据长度 x_lengths 和条件输入 g
    def forward(self, x, x_lengths, g=None):
        # 生成输入数据的掩码，保留有效数据部分
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        # 对输入数据进行预处理，并应用掩码
        x = self.pre(x) * x_mask
        # 对预处理后的数据进行编码，并应用掩码
        x = self.enc(x, x_mask, g=g)
        # 对编码后的数据进行投影，并应用掩码
        stats = self.proj(x) * x_mask
        # 将投影后的数据分割成均值 m 和标准差 logs
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 生成服从正态分布的随机数，并根据均值和标准差进行采样，再应用掩码
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        # 返回采样结果 z、均值 m、标准差 logs 和掩码 x_mask
        return z, m, logs, x_mask


# 定义生成器类，继承自 torch.nn.Module
class Generator(torch.nn.Module):
    # 初始化方法，接受初始通道数、残差块、残差块的卷积核大小、扩张率、上采样率、上采样初始通道数、上采样卷积核大小和 gin 通道数
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

        # 创建一个模块列表，用于存储上采样层
        self.ups = nn.ModuleList()
        # 遍历上采样率和上采样核大小，创建上采样层并添加到模块列表中
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
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

        # 创建一个模块列表，用于存储残差块
        self.resblocks = nn.ModuleList()
        # 遍历上采样层，根据残差块核大小和膨胀大小创建残差块并添加到模块列表中
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        # 创建一个一维卷积层，用于后处理
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # 初始化上采样层的权重
        self.ups.apply(init_weights)

        # 如果 gin_channels 不为 0，则创建一个一维卷积层作为条件输入
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
    # 定义前向传播函数，接受输入 x 和条件 g（可选）
    def forward(self, x, g=None):
        # 使用预处理卷积层处理输入 x
        x = self.conv_pre(x)
        # 如果条件 g 存在，则将条件 g 作用于 x
        if g is not None:
            x = x + self.cond(g)

        # 循环执行上采样操作
        for i in range(self.num_upsamples):
            # 对 x 执行 LeakyReLU 激活函数
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 使用上采样层对 x 进行上采样操作
            x = self.ups[i](x)
            xs = None
            # 循环执行残差块操作
            for j in range(self.num_kernels):
                # 如果 xs 为空，则将当前残差块的处理结果赋值给 xs
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                # 否则，将当前残差块的处理结果加到 xs 上
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # 对 xs 求平均值
            x = xs / self.num_kernels
        # 对 x 执行 LeakyReLU 激活函数
        x = F.leaky_relu(x)
        # 使用后处理卷积层处理 x
        x = self.conv_post(x)
        # 对 x 执行 tanh 激活函数
        x = torch.tanh(x)

        # 返回处理后的 x
        return x

    # 移除权重归一化
    def remove_weight_norm(self):
        # 打印信息
        print("Removing weight norm...")
        # 遍历上采样层，移除权重归一化
        for layer in self.ups:
            remove_weight_norm(layer)
        # 遍历残差块，移除权重归一化
        for layer in self.resblocks:
            layer.remove_weight_norm()
class DiscriminatorP(torch.nn.Module):
    # 定义一个名为DiscriminatorP的类，继承自torch.nn.Module
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        # 初始化函数，接受period、kernel_size、stride和use_spectral_norm等参数
        super(DiscriminatorP, self).__init__()
        # 调用父类的初始化函数
        self.period = period
        # 设置类的period属性为传入的period参数
        self.use_spectral_norm = use_spectral_norm
        # 设置类的use_spectral_norm属性为传入的use_spectral_norm参数
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 根据use_spectral_norm参数选择使用weight_norm或spectral_norm，并赋值给norm_f变量
        self.convs = nn.ModuleList(
            # 创建一个包含多个卷积层的ModuleList
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
                # 使用norm_f对Conv2d进行归一化，并设置卷积层的参数
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 使用norm_f对Conv2d进行归一化，并设置卷积层的参数
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 使用norm_f对Conv2d进行归一化，并设置卷积层的参数
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 使用norm_f对Conv2d进行归一化，并设置卷积层的参数
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 使用norm_f对Conv2d进行归一化，并设置卷积层的参数
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        # 使用norm_f对Conv2d进行归一化，并设置卷积层的参数
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 初始化特征图列表
        fmap = []

        # 将输入从1维转换为2维
        b, c, t = x.shape
        # 如果时间步 t 不能被 self.period 整除，则进行填充
        if t % self.period != 0:  # pad first
            # 计算需要填充的数量
            n_pad = self.period - (t % self.period)
            # 对输入进行反射填充
            x = F.pad(x, (0, n_pad), "reflect")
            # 更新时间步 t
            t = t + n_pad
        # 将输入重新调整形状为 (b, c, t // self.period, self.period)
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
        # 定义权重归一化函数，如果使用谱归一化则使用谱归一化函数
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 定义卷积层列表
        self.convs = nn.ModuleList(
            [
                # 使用权重归一化函数对输入进行卷积操作
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        # 定义后续卷积层
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        # 遍历卷积层列表，对输入进行卷积操作并使用LeakyReLU激活函数
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        # 对最后一个卷积层进行卷积操作
        x = self.conv_post(x)
        fmap.append(x)
        # 对输出进行展平操作
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        # 定义包含不同周期的判别器列表
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
        # 遍历不同周期的判别器，对真实数据和生成数据进行判别
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
        ref_enc_filters = [32, 32, 64, 64, 128, 128]  # 定义卷积层的滤波器数量
        K = len(ref_enc_filters)  # 获取滤波器数量
        filters = [1] + ref_enc_filters  # 构建滤波器列表
        convs = [  # 创建卷积层列表
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
        self.convs = nn.ModuleList(convs)  # 将卷积层列表转换为模块列表
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)]) # noqa: E501

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)  # 计算输出通道数量
        self.gru = nn.GRU(  # 创建GRU层
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)  # 创建线性层

    def forward(self, inputs, mask=None):
        N = inputs.size(0)  # 获取输入的批量大小
        out = inputs.view(N, 1, -1, self.spec_channels)  # 将输入重塑为指定形状
        for conv in self.convs:  # 遍历卷积层列表
            out = conv(out)  # 应用卷积操作
            # out = wn(out)
            out = F.relu(out)  # 应用ReLU激活函数

        out = out.transpose(1, 2)  # 转置张量维度
        T = out.size(1)  # 获取转置后的张量维度
        N = out.size(0)  # 获取转置后的张量维度
        out = out.contiguous().view(N, T, -1)  # 将张量重塑为指定形状

        self.gru.flatten_parameters()  # 展平GRU层的参数
        memory, out = self.gru(out)  # 应用GRU层

        return self.proj(out.squeeze(0))  # 返回线性层的输出

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):  # 遍历卷积层数量
            L = (L - kernel_size + 2 * pad) // stride + 1  # 计算输出通道数量
        return L  # 返回输出通道数量
class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,  # 词汇量大小
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
        n_speakers=256,  # 说话人数量，默认为256
        gin_channels=256,  # GIN通道数，默认为256
        use_sdp=True,  # 是否使用SDP，默认为True
        n_flow_layer=4,  # 流层数量，默认为4
        n_layers_trans_flow=4,  # 转换流层数量，默认为4
        flow_share_parameter=False,  # 流共享参数，默认为False
        use_transformer_flow=True,  # 是否使用Transformer流，默认为True
        **kwargs,  # 其他参数
    def export_onnx(
        self,
        path,  # 导出路径
        max_len=None,  # 最大长度，默认为None
        sdp_ratio=0,  # SDP比率，默认为0
        y=None,  # 目标值，默认为None
```