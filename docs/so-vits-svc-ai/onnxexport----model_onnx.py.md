# `so-vits-svc\onnxexport\model_onnx.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn 模块中导入 Conv1d, Conv2d 模块
from torch.nn import Conv1d, Conv2d
# 从 torch.nn 模块中导入 functional 模块并重命名为 F
from torch.nn import functional as F
# 从 torch.nn.utils 模块中导入 spectral_norm, weight_norm 模块
from torch.nn.utils import spectral_norm, weight_norm
# 导入 attentions, commons, modules 模块
import modules.attentions as attentions
import modules.commons as commons
import modules.modules as modules
# 导入 utils 模块
import utils
# 从 modules.commons 模块中导入 get_padding 模块
from modules.commons import get_padding
# 从 utils 模块中导入 f0_to_coarse 模块
from utils import f0_to_coarse
# 从 vdecoder.hifigan.models 模块中导入 Generator 模块
from vdecoder.hifigan.models import Generator

# 定义 ResidualCouplingBlock 类，继承自 nn.Module 类
class ResidualCouplingBlock(nn.Module):
    # 初始化函数，接收 channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows, gin_channels 参数
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0):
        # 调用父类的初始化函数
        super().__init__()
        # 将参数赋值给对应的属性
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # 创建 nn.ModuleList 对象 flows
        self.flows = nn.ModuleList()
        # 循环 n_flows 次
        for i in range(n_flows):
            # 向 flows 中添加 ResidualCouplingLayer 和 Flip 模块
            self.flows.append(
                modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                              gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    # 前向传播函数，接收 x, x_mask, g=None, reverse=False 参数
    def forward(self, x, x_mask, g=None, reverse=False):
        # 如果不是反向传播
        if not reverse:
            # 遍历 flows 中的模块，对 x 进行处理
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        # 如果是反向传播
        else:
            # 反向遍历 flows 中的模块，对 x 进行处理
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        # 返回处理后的 x
        return x


# 定义 Encoder 类，继承自 nn.Module 类
class Encoder(nn.Module):
    # 初始化函数，设置模型的输入通道数、输出通道数、隐藏通道数、卷积核大小、扩张率、层数和输入图像的全局信息通道数
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
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
        # 设置模型的扩张率
        self.dilation_rate = dilation_rate
        # 设置模型的层数
        self.n_layers = n_layers
        # 设置模型的输入图像的全局信息通道数
        self.gin_channels = gin_channels

        # 创建一个 1x1 的卷积层，将输入通道数转换为隐藏通道数
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        # 创建一个 WaveNet 模块
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        # 创建一个 1x1 的卷积层，将隐藏通道数转换为输出通道数的两倍
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    # 前向传播函数，接收输入张量 x、输入长度 x_lengths 和全局信息张量 g（可选）
    def forward(self, x, x_lengths, g=None):
        # 生成输入张量的掩码，用于处理变长序列
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        # 对输入张量进行 1x1 卷积，并乘以掩码
        x = self.pre(x) * x_mask
        # 将输入张量传入 WaveNet 模块进行处理
        x = self.enc(x, x_mask, g=g)
        # 将处理后的张量传入 1x1 卷积层进行处理
        stats = self.proj(x) * x_mask
        # 将处理后的张量按输出通道数的一半进行分割，得到均值和标准差
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 对均值加上服从标准正态分布的随机噪声，并乘以掩码
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        # 返回处理后的张量 z、均值 m、标准差 logs 和掩码 x_mask
        return z, m, logs, x_mask
# 定义一个名为TextEncoder的类，继承自nn.Module
class TextEncoder(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 n_layers,
                 gin_channels=0,
                 filter_channels=None,
                 n_heads=None,
                 p_dropout=None):
        super().__init__()  # 调用父类的初始化函数
        # 初始化各个参数
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        # 创建一个卷积层，将隐藏通道数转换为输出通道数的两倍
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        # 创建一个大小为256的嵌入层，将256维的输入转换为隐藏通道数
        self.f0_emb = nn.Embedding(256, hidden_channels)

        # 创建一个名为enc_的Encoder对象
        self.enc_ = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

    # 前向传播函数，接受多个参数
    def forward(self, x, x_mask, f0=None, z=None):
        # 将输入x与f0的嵌入结果相加，并进行转置
        x = x + self.f0_emb(f0).transpose(1, 2)
        # 对输入x进行编码
        x = self.enc_(x * x_mask, x_mask)
        # 对编码结果进行投影
        stats = self.proj(x) * x_mask
        # 将投影结果分割为均值m和标准差logs
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 计算最终输出z，并考虑输入z和标准差logs
        z = (m + z * torch.exp(logs)) * x_mask
        # 返回最终输出z、均值m、标准差logs和输入掩码x_mask
        return z, m, logs, x_mask


# 定义一个名为DiscriminatorP的类，继承自torch.nn.Module
class DiscriminatorP(torch.nn.Module):
    # 初始化函数，设置周期、卷积核大小、步长、是否使用谱归一化
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        # 调用父类的初始化函数
        super(DiscriminatorP, self).__init__()
        # 设置周期
        self.period = period
        # 设置是否使用谱归一化
        self.use_spectral_norm = use_spectral_norm
        # 根据是否使用谱归一化选择合适的归一化函数
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 创建卷积层列表
        self.convs = nn.ModuleList([
            # 第一层卷积
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            # 第二层卷积
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            # 第三层卷积
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            # 第四层卷积
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            # 第五层卷积
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        # 创建后续卷积层
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    # 前向传播函数
    def forward(self, x):
        # 特征图列表
        fmap = []

        # 1维转2维
        b, c, t = x.shape
        # 如果时间步不是周期的整数倍，进行填充
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        # 遍历卷积层列表
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        # 后续卷积层
        x = self.conv_post(x)
        fmap.append(x)
        # 展平
        x = torch.flatten(x, 1, -1)

        return x, fmap
# 定义鉴别器模型类
class DiscriminatorS(torch.nn.Module):
    # 初始化函数
    def __init__(self, use_spectral_norm=False):
        # 调用父类的初始化函数
        super(DiscriminatorS, self).__init__()
        # 根据是否使用谱归一化选择不同的归一化函数
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 定义卷积层列表
        self.convs = nn.ModuleList([
            # 使用归一化函数对卷积层进行归一化
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        # 定义后续的卷积层
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    # 前向传播函数
    def forward(self, x):
        # 特征图列表
        fmap = []

        # 遍历卷积层列表
        for l in self.convs:
            # 对输入进行卷积操作
            x = l(x)
            # 使用 LeakyReLU 激活函数
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 将结果添加到特征图列表中
            fmap.append(x)
        # 对最后一个卷积层进行操作
        x = self.conv_post(x)
        # 将结果添加到特征图列表中
        fmap.append(x)
        # 对结果进行展平操作
        x = torch.flatten(x, 1, -1)

        # 返回结果和特征图列表
        return x, fmap


class F0Decoder(nn.Module):
    # 初始化函数，设置模型的参数
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 spk_channels=0):
        super().__init__()
        # 设置模型的输出通道数
        self.out_channels = out_channels
        # 设置模型的隐藏通道数
        self.hidden_channels = hidden_channels
        # 设置模型的滤波器通道数
        self.filter_channels = filter_channels
        # 设置模型的注意力头数
        self.n_heads = n_heads
        # 设置模型的层数
        self.n_layers = n_layers
        # 设置模型的卷积核大小
        self.kernel_size = kernel_size
        # 设置模型的丢弃率
        self.p_dropout = p_dropout
        # 设置说话人通道数，默认为0
        self.spk_channels = spk_channels

        # 创建预网络层，使用隐藏通道数作为输入和输出通道数，卷积核大小为3，填充为1
        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        # 创建注意力层，使用隐藏通道数、滤波器通道数、注意力头数、层数、卷积核大小和丢弃率作为参数
        self.decoder = attentions.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        # 创建投影层，使用隐藏通道数作为输入通道数，输出通道数为模型的输出通道数，卷积核大小为1
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        # 创建F0预网络层，使用输入通道数为1，输出通道数为隐藏通道数，卷积核大小为3，填充为1
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        # 创建条件层，使用说话人通道数作为输入通道数，隐藏通道数作为输出通道数，卷积核大小为1
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    # 前向传播函数，接收输入x、归一化的F0、输入的掩码x_mask、说话人嵌入spk_emb
    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        # 将输入x转换为不需要梯度的张量
        x = torch.detach(x)
        # 如果存在说话人嵌入spk_emb，则将输入x与条件层对说话人嵌入的处理结果相加
        if spk_emb is not None:
            x = x + self.cond(spk_emb)
        # 将输入x与归一化的F0经过F0预网络层处理后的结果相加
        x += self.f0_prenet(norm_f0)
        # 将输入x经过预网络层处理，并乘以输入的掩码x_mask
        x = self.prenet(x) * x_mask
        # 将处理后的输入x和输入的掩码x_mask传入注意力层进行处理
        x = self.decoder(x * x_mask, x_mask)
        # 将处理后的结果经过投影层处理，并乘以输入的掩码x_mask
        x = self.proj(x) * x_mask
        # 返回处理后的结果
        return x
class SynthesizerTrn(nn.Module):
    """
  Synthesizer for Training
  """

    def forward(self, c, f0, mel2ph, uv, noise=None, g=None):

        # 对输入的c进行填充，[0, 0, 1, 0]表示填充左右0个单位，上1个单位，下0个单位
        decoder_inp = F.pad(c, [0, 0, 1, 0])
        # 将mel2ph扩展一个维度并重复c.shape[-1]次，然后与decoder_inp进行按索引取值，再转置
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, c.shape[-1]])
        c = torch.gather(decoder_inp, 1, mel2ph_).transpose(1, 2)  # [B, T, H]

        # 创建一个与c相同大小的张量，值为c.size(-1)，并转移到c所在的设备
        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        # 在g的第0维度上增加一个维度
        g = g.unsqueeze(0)
        # 使用emb_g对g进行嵌入，并转置
        g = self.emb_g(g).transpose(1, 2)
        # 创建一个与c相同大小的掩码张量，值为1，数据类型与c相同
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        # 使用pre对c进行处理，再乘以x_mask，再加上emb_uv对uv进行嵌入并转置
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2)

        # 如果需要预测f0
        if self.predict_f0:
            # 计算lf0，然后对lf0进行归一化
            lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv, random_scale=False)
            # 使用f0_decoder对x、norm_lf0、x_mask进行处理，得到预测的lf0
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            # 根据预测的lf0计算f0
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)

        # 使用enc_p对x、x_mask、f0_to_coarse(f0)、noise进行编码
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), z=noise)
        # 使用flow对z_p、c_mask、g进行处理，得到z
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        # 使用dec对z乘以c_mask、g、f0进行处理，得到输出o
        o = self.dec(z * c_mask, g=g, f0=f0)
        # 返回输出o
        return o
```