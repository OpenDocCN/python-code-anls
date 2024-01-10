# `so-vits-svc\onnxexport\model_onnx_speaker_mix.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn 模块中导入 functional 模块并重命名为 F
from torch.nn import functional as F

# 导入自定义模块 attentions、commons、modules 和 utils
import modules.attentions as attentions
import modules.commons as commons
import modules.modules as modules
import utils
# 从 utils 模块中导入 f0_to_coarse 函数
from utils import f0_to_coarse

# 定义 ResidualCouplingBlock 类，继承自 nn.Module 类
class ResidualCouplingBlock(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0,
                 share_parameter=False
                 ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化类的属性
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # 创建 nn.ModuleList 类型的对象 flows
        self.flows = nn.ModuleList()

        # 如果 share_parameter 为 False，则创建 WN 对象并赋值给 self.wn，否则 self.wn 为 None
        self.wn = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=gin_channels) if share_parameter else None

        # 循环 n_flows 次，向 flows 中添加 ResidualCouplingLayer 和 Flip 对象
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                              gin_channels=gin_channels, mean_only=True, wn_sharing_parameter=self.wn))
            self.flows.append(modules.Flip())

    # 前向传播函数，接受多个参数
    def forward(self, x, x_mask, g=None, reverse=False):
        # 如果 reverse 为 False，则对 flows 中的每个 flow 调用前向传播函数
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        # 如果 reverse 为 True，则对 flows 中的每个 flow 调用反向传播函数
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        # 返回 x
        return x

# 定义 TransformerCouplingBlock 类，继承自 nn.Module 类
class TransformerCouplingBlock(nn.Module):
    # 初始化函数，设置模型的各项参数
    def __init__(self,
                 channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 n_flows=4,
                 gin_channels=0,
                 share_parameter=False
                 ):
            
        super().__init__()
        # 设置模型的通道数
        self.channels = channels
        # 设置模型的隐藏层通道数
        self.hidden_channels = hidden_channels
        # 设置模型的卷积核大小
        self.kernel_size = kernel_size
        # 设置模型的层数
        self.n_layers = n_layers
        # 设置模型的流数
        self.n_flows = n_flows
        # 设置模型的GIN通道数
        self.gin_channels = gin_channels

        # 初始化模型的流列表
        self.flows = nn.ModuleList()

        # 如果共享参数，则初始化FFT注意力模块
        self.wn = attentions.FFT(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, isflow = True, gin_channels = self.gin_channels) if share_parameter else None

        # 根据流数循环添加TransformerCouplingLayer和Flip模块到流列表中
        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(channels, hidden_channels, kernel_size, n_layers, n_heads, p_dropout, filter_channels, mean_only=True, wn_sharing_parameter=self.wn, gin_channels = self.gin_channels))
            self.flows.append(modules.Flip())

    # 前向传播函数，根据reverse参数决定正向传播还是反向传播
    def forward(self, x, x_mask, g=None, reverse=False):
        # 如果是正向传播
        if not reverse:
            # 对每个流模块进行正向传播
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        # 如果是反向传播
        else:
            # 对每个流模块进行反向传播
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        # 返回传播结果
        return x
# 定义一个名为 Encoder 的类，继承自 nn.Module
class Encoder(nn.Module):
    # 初始化函数，接受输入通道数、输出通道数、隐藏通道数、卷积核大小、扩张率、层数和 gin 通道数
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        # 初始化各个参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        # 创建一个 1 维卷积层，输入通道数为 in_channels，输出通道数为 hidden_channels，卷积核大小为 1
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        # 创建一个 WN 模块，参数为 hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        # 创建一个 1 维卷积层，输入通道数为 hidden_channels，输出通道数为 out_channels * 2，卷积核大小为 1
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    # 前向传播函数，接受输入 x, x_lengths, g（可选）
    def forward(self, x, x_lengths, g=None):
        # 生成 x 的掩码，保留有效长度部分
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        # 对输入 x 进行预处理，并乘以掩码
        x = self.pre(x) * x_mask
        # 对处理后的 x 进行编码
        x = self.enc(x, x_mask, g=g)
        # 对编码后的结果进行投影
        stats = self.proj(x) * x_mask
        # 将结果 stats 按照输出通道数的一半进行分割，得到均值 m 和标准差 logs
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 生成服从正态分布的随机数，乘以标准差 logs，加上均值 m，再乘以掩码
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        # 返回 z, m, logs, x_mask
        return z, m, logs, x_mask


class TextEncoder(nn.Module):
    # 初始化函数，定义了模型的参数和层
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 n_layers,
                 gin_channels=0,
                 filter_channels=None,
                 n_heads=None,
                 p_dropout=None):
        # 调用父类的初始化函数
        super().__init__()
        # 设置输出通道数
        self.out_channels = out_channels
        # 设置隐藏通道数
        self.hidden_channels = hidden_channels
        # 设置卷积核大小
        self.kernel_size = kernel_size
        # 设置层数
        self.n_layers = n_layers
        # 设置输入图像的通道数
        self.gin_channels = gin_channels
        # 创建一个卷积层，将隐藏通道数映射到输出通道数的两倍
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        # 创建一个大小为256的嵌入层，将256维的输入映射到隐藏通道数
        self.f0_emb = nn.Embedding(256, hidden_channels)

        # 创建一个注意力机制的编码器
        self.enc_ = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

    # 前向传播函数，定义了模型的前向计算过程
    def forward(self, x, x_mask, f0=None, z=None):
        # 将输入x与f0的嵌入结果相加，并进行转置
        x = x + self.f0_emb(f0).transpose(1, 2)
        # 将加了嵌入的输入x和掩码x_mask传入编码器进行计算
        x = self.enc_(x * x_mask, x_mask)
        # 使用卷积层对编码器的输出进行统计
        stats = self.proj(x) * x_mask
        # 将统计结果按照输出通道数进行分割，得到均值m和标准差logs
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 对隐变量z进行重参数化，并与掩码相乘
        z = (m + z * torch.exp(logs)) * x_mask

        # 返回重参数化后的z，均值m，标准差logs和掩码x_mask
        return z, m, logs, x_mask
class F0Decoder(nn.Module):
    # 定义 F0Decoder 类，继承自 nn.Module
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 spk_channels=0):
        # 初始化函数，接受输出通道数、隐藏通道数、滤波通道数、注意力头数、层数、卷积核大小、dropout 概率和说话人通道数等参数
        super().__init__()
        # 调用父类的初始化函数
        self.out_channels = out_channels
        # 设置输出通道数
        self.hidden_channels = hidden_channels
        # 设置隐藏通道数
        self.filter_channels = filter_channels
        # 设置滤波通道数
        self.n_heads = n_heads
        # 设置注意力头数
        self.n_layers = n_layers
        # 设置层数
        self.kernel_size = kernel_size
        # 设置卷积核大小
        self.p_dropout = p_dropout
        # 设置dropout 概率
        self.spk_channels = spk_channels
        # 设置说话人通道数

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        # 定义预网络，使用 1 维卷积层
        self.decoder = attentions.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        # 定义解码器，使用 FFT 注意力机制
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        # 定义投影层，使用 1 维卷积层
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        # 定义 F0 预网络，使用 1 维卷积层
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)
        # 定义条件卷积层，使用 1 维卷积层

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        # 前向传播函数，接受输入 x、归一化 F0、掩码 x_mask 和说话人嵌入 spk_emb
        x = torch.detach(x)
        # 将输入 x 转换为不需要梯度的张量
        if (spk_emb is not None):
            # 如果存在说话人嵌入
            x = x + self.cond(spk_emb)
            # 输入 x 加上说话人嵌入的条件
        x += self.f0_prenet(norm_f0)
        # 输入 x 加上 F0 预网络的处理结果
        x = self.prenet(x) * x_mask
        # 输入 x 经过预网络处理并乘以掩码
        x = self.decoder(x * x_mask, x_mask)
        # 输入 x 经过解码器处理并乘以掩码
        x = self.proj(x) * x_mask
        # 输入 x 经过投影层处理并乘以掩码
        return x
        # 返回处理结果


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def export_chara_mix(self, speakers_mix):
        # 导出角色混合
        self.speaker_map = torch.zeros((len(speakers_mix), 1, 1, self.gin_channels))
        # 初始化说话人映射
        i = 0
        # 初始化计数器
        for key in speakers_mix.keys():
            # 遍历说话人混合的键
            spkidx = speakers_mix[key]
            # 获取说话人索引
            self.speaker_map[i] = self.emb_g(torch.LongTensor([[spkidx]]))
            # 使用 emb_g 获取说话人映射
            i = i + 1
            # 计数器加一
        self.speaker_map = self.speaker_map.unsqueeze(0)
        # 对说话人映射进行维度扩展
        self.export_mix = True
        # 设置导出混合标志为 True
    # 定义一个方法，用于模型的前向传播
    def forward(self, c, f0, mel2ph, uv, noise=None, g=None, vol = None):
        # 对输入的c进行填充，填充上方1行，下方0行
        decoder_inp = F.pad(c, [0, 0, 1, 0])
        # 将mel2ph在第二个维度上进行复制，使其与c的维度相同
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, c.shape[-1]])
        # 从decoder_inp中根据mel2ph_的索引取值，并进行转置
        c = torch.gather(decoder_inp, 1, mel2ph_).transpose(1, 2)  # [B, T, H]

        # 如果使用导出混合
        if self.export_mix:   # [N, S]  *  [S, B, 1, H]
            # 将g的形状进行调整
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1)  # [N, S, B, 1, 1]
            # 将g与speaker_map相乘
            g = g * self.speaker_map  # [N, S, B, 1, H]
            # 对g进行求和
            g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
            # 对g进行转置和压缩维度
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
        else:
            # 如果g的维度为1，则在第0维度上进行扩展
            if g.dim() == 1:
                g = g.unsqueeze(0)
            # 对g进行嵌入操作，并进行转置
            g = self.emb_g(g).transpose(1, 2)
        
        # 对f0进行扩展，使其与c的维度相同
        x_mask = torch.unsqueeze(torch.ones_like(f0), 1).to(c.dtype)
        
        # 如果vol不为空且开启了vol_embedding，则对vol进行嵌入操作并进行转置，否则为0
        vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol is not None and self.vol_embedding else 0

        # 对c进行预处理，并与uv进行嵌入操作，再加上vol
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        # 如果使用自动f0预测并且开启了预测f0
        if self.use_automatic_f0_prediction and self.predict_f0:
            # 对f0进行转换
            lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
            # 对lf0进行归一化
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv, random_scale=False)
            # 对归一化后的lf0进行预测
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            # 对预测的lf0进行转换
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        
        # 对z_p, m_p, logs_p, c_mask进行编码
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), z=noise)
        # 对z进行流动操作
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        # 对z进行解码操作
        o = self.dec(z * c_mask, g=g, f0=f0)
        # 返回o
        return o
```