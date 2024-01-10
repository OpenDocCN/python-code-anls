# `so-vits-svc\vdecoder\hifiganwithsnake\models.py`

```
# 导入所需的库
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from vdecoder.hifiganwithsnake.alias.act import SnakeAlias  # 导入自定义模块

from .env import AttrDict  # 从当前目录下的 env 模块中导入 AttrDict 类
from .utils import get_padding, init_weights  # 从当前目录下的 utils 模块中导入 get_padding 和 init_weights 函数

LRELU_SLOPE = 0.1  # 定义全局变量 LRELU_SLOPE，赋值为 0.1

# 加载模型函数，接受模型路径和设备类型作为参数
def load_model(model_path, device='cuda'):
    # 从模型路径中获取配置文件路径
    config_file = os.path.join(os.path.split(model_path)[0], 'config.json')
    # 打开配置文件，读取其中的内容
    with open(config_file) as f:
        data = f.read()

    # 定义全局变量 h，将配置文件内容解析为 JSON 格式
    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)  # 使用解析后的 JSON 数据创建 AttrDict 对象

    generator = Generator(h).to(device)  # 创建生成器对象，并将其移动到指定设备上

    cp_dict = torch.load(model_path)  # 加载模型参数
    generator.load_state_dict(cp_dict['generator'])  # 加载生成器的状态字典
    generator.eval()  # 设置生成器为评估模式
    generator.remove_weight_norm()  # 移除生成器的权重归一化
    del cp_dict  # 删除模型参数字典
    return generator, h  # 返回生成器对象和配置信息


class ResBlock1(torch.nn.Module):
    # 这里是 ResBlock1 类的定义，后续的代码需要继续注释
    # 初始化 ResBlock1 类，设置参数 h, channels, kernel_size, dilation 和 C
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), C=None):
        # 调用父类的初始化方法
        super(ResBlock1, self).__init__()
        # 设置属性 h
        self.h = h
        # 初始化卷积层列表 convs1，包含三个卷积层，每个卷积层都使用权重归一化
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))
        ])
        # 对 convs1 中的每个卷积层应用初始化权重的函数
        self.convs1.apply(init_weights)

        # 初始化卷积层列表 convs2，包含三个卷积层，每个卷积层都使用权重归一化
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        # 对 convs2 中的每个卷积层应用初始化权重的函数
        self.convs2.apply(init_weights)

        # 计算总层数
        self.num_layers = len(self.convs1) + len(self.convs2)
        # 初始化激活函数列表，每个激活函数都使用 SnakeAlias，共有 self.num_layers 个
        self.activations = nn.ModuleList([
            SnakeAlias(channels, C=C) for _ in range(self.num_layers)
        ])

    # 前向传播函数，接收输入 x 和维度 DIM
    def forward(self, x, DIM=None):
        # 将激活函数列表分为两部分，分别赋值给 acts1 和 acts2
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        # 遍历 convs1, convs2, acts1, acts2 中的每个元素，分别赋值给 c1, c2, a1, a2
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            # 使用 a1 对输入 x 进行处理
            xt = a1(x, DIM)
            # 使用 c1 对处理后的数据 xt 进行卷积操作
            xt = c1(xt)
            # 使用 a2 对卷积后的数据 xt 进行处理
            xt = a2(xt, DIM)
            # 使用 c2 对处理后的数据 xt 进行卷积操作
            xt = c2(xt)
            # 将处理后的数据 xt 与输入 x 相加，得到输出 x
            x = xt + x
        # 返回输出 x
        return x

    # 移除权重归一化
    def remove_weight_norm(self):
        # 遍历 convs1 中的每个卷积层，移除权重归一化
        for l in self.convs1:
            remove_weight_norm(l)
        # 遍历 convs2 中的每个卷积层，移除权重归一化
        for l in self.convs2:
            remove_weight_norm(l)
class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), C=None):
        super(ResBlock2, self).__init__()
        self.h = h  # 初始化ResBlock2对象的h属性
        self.convs = nn.ModuleList([  # 初始化ResBlock2对象的convs属性为包含两个卷积层的ModuleList
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],  # 使用weight_norm函数创建卷积层
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],  # 使用weight_norm函数创建卷积层
                               padding=get_padding(kernel_size, dilation[1]))
        ])
        self.convs.apply(init_weights)  # 对convs属性中的所有卷积层应用init_weights函数进行初始化
        
        self.num_layers = len(self.convs)  # 初始化ResBlock2对象的num_layers属性为convs属性的长度
        self.activations = nn.ModuleList([  # 初始化ResBlock2对象的activations属性为包含SnakeAlias对象的ModuleList
            SnakeAlias(channels, C=C) for _ in range(self.num_layers)  # 使用SnakeAlias类创建SnakeAlias对象
        ])

    def forward(self, x, DIM=None):
        for c,a in zip(self.convs, self.activations):  # 遍历convs和activations属性中的卷积层和SnakeAlias对象
            xt = a(x, DIM)  # 使用SnakeAlias对象对输入x进行处理
            xt = c(xt)  # 使用卷积层对处理后的数据进行卷积操作
            x = xt + x  # 将卷积操作的结果与输入x相加
        return x  # 返回处理后的数据

    def remove_weight_norm(self):
        for l in self.convs:  # 遍历convs属性中的卷积层
            remove_weight_norm(l)  # 移除卷积层的权重归一化


def padDiff(x):
    return F.pad(F.pad(x, (0,0,-1,1), 'constant', 0) - x, (0,0,0,-1), 'constant', 0)  # 对输入x进行填充和差分操作

class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """
    # 初始化函数，设置默认参数，并初始化对象属性
    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        # 调用父类的初始化函数
        super(SineGen, self).__init__()
        # 设置正弦振幅
        self.sine_amp = sine_amp
        # 设置噪声标准差
        self.noise_std = noise_std
        # 设置谐波数
        self.harmonic_num = harmonic_num
        # 设置维度
        self.dim = self.harmonic_num + 1
        # 设置采样率
        self.sampling_rate = samp_rate
        # 设置有声门阈值
        self.voiced_threshold = voiced_threshold
        # 设置脉冲标志
        self.flag_for_pulse = flag_for_pulse
        # 初始化 ONNX 标志
        self.onnx = False
    
    # 生成 UV 信号的私有方法
    def _f02uv(self, f0):
        # 生成 UV 信号，如果基频大于有声门阈值则为 1，否则为 0
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv
class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp=None):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs.to(self.l_linear.weight.dtype)))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class Generator(torch.nn.Module):
    # 导出 ONNX 模型的方法
    def OnnxExport(self):
        # 设置 ONNX 标志为 True
        self.onnx = True
        # 设置源数据的 ONNX 标志为 True
        self.m_source.l_sin_gen.onnx = True
        
    # 前向传播方法
    def forward(self, x, f0, g=None):
        # 如果不是 ONNX 模式
        if not self.onnx:
            # 对 f0 进行上采样并转置
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        # 获取声音源、噪声源和声门特征
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        # 转置声音源
        har_source = har_source.transpose(1, 2)
        # 对输入进行预处理卷积
        x = self.conv_pre(x)
        # 将输入与条件信息相加
        x = x + self.cond(g)
        # 循环进行上采样
        for i in range(self.num_upsamples):
            # 对输入进行变形
            x = self.snakes[i](x)
            # 上采样
            x = self.ups[i](x)
            # 对声音源进行噪声卷积
            x_source = self.noise_convs[i](har_source)
            # 将输入与声音源相加
            x = x + x_source
            xs = None
            # 循环进行残差块处理
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # 对残差块处理结果进行平均
            x = xs / self.num_kernels
        # 对输出进行后处理
        x = self.snake_post(x)
        x = self.conv_post(x)
        # 对输出进行 tanh 激活
        x = torch.tanh(x)

        return x

    # 移除权重归一化的方法
    def remove_weight_norm(self):
        # 打印移除权重归一化的提示信息
        print('Removing weight norm...')
        # 对上采样层移除权重归一化
        for l in self.ups:
            remove_weight_norm(l)
        # 对残差块移除权重归一化
        for l in self.resblocks:
            l.remove_weight_norm()
        # 移除输入预处理卷积的权重归一化
        remove_weight_norm(self.conv_pre)
        # 移除输出后处理卷积的权重归一化
        remove_weight_norm(self.conv_post)
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 创建卷积层列表，每个卷积层使用权重归一化或谱归一化
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        # 创建最后的卷积层，使用权重归一化或谱归一化
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # 如果时间步不是周期的整数倍，进行填充
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)  # 进行卷积操作
            x = F.leaky_relu(x, LRELU_SLOPE)  # 使用 Leaky ReLU 激活函数
            fmap.append(x)  # 将中间特征图添加到列表中
        x = self.conv_post(x)  # 进行最后一层卷积操作
        fmap.append(x)  # 将最终特征图添加到列表中
        x = torch.flatten(x, 1, -1)  # 将特征图展平为一维向量

        return x, fmap  # 返回最终结果和中间特征图列表


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=None):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(DiscriminatorP(period))  # 创建多个周期判别器
    # 定义一个前向传播函数，接受真实数据 y 和生成数据 y_hat 作为输入
    def forward(self, y, y_hat):
        # 初始化真实数据在每个鉴别器上的判别结果列表和特征图列表
        y_d_rs = []
        fmap_rs = []
        # 初始化生成数据在每个鉴别器上的判别结果列表和特征图列表
        y_d_gs = []
        fmap_gs = []
        # 遍历每个鉴别器
        for i, d in enumerate(self.discriminators):
            # 对真实数据进行前向传播，获取判别结果和特征图
            y_d_r, fmap_r = d(y)
            # 对生成数据进行前向传播，获取判别结果和特征图
            y_d_g, fmap_g = d(y_hat)
            # 将真实数据的判别结果和特征图添加到对应的列表中
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            # 将生成数据的判别结果和特征图添加到对应的列表中
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        # 返回真实数据和生成数据在每个鉴别器上的判别结果列表和特征图列表
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        # 初始化判别器模型
        super(DiscriminatorS, self).__init__()
        # 根据是否使用谱归一化选择不同的归一化函数
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 定义卷积层列表
        self.convs = nn.ModuleList([
            # 使用归一化函数对卷积层进行归一化
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        # 定义后续的卷积层
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        # 存储每一层卷积的特征图
        fmap = []
        # 遍历每一层卷积
        for l in self.convs:
            # 对输入进行卷积操作
            x = l(x)
            # 对卷积结果进行 LeakyReLU 激活
            x = F.leaky_relu(x, LRELU_SLOPE)
            # 将卷积结果存储到特征图列表中
            fmap.append(x)
        # 对最后一层卷积进行后续卷积操作
        x = self.conv_post(x)
        # 将最后一层卷积结果存储到特征图列表中
        fmap.append(x)
        # 对最后一层卷积结果进行展平操作
        x = torch.flatten(x, 1, -1)
        # 返回展平后的结果和特征图列表
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        # 初始化多尺度判别器模型
        super(MultiScaleDiscriminator, self).__init__()
        # 定义多个判别器模型
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        # 定义多个平均池化层
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])
    # 定义一个前向传播函数，接受真实数据 y 和生成数据 y_hat 作为输入
    def forward(self, y, y_hat):
        # 初始化真实数据的判别结果列表和特征图列表
        y_d_rs = []
        fmap_rs = []
        # 初始化生成数据的判别结果列表和特征图列表
        y_d_gs = []
        fmap_gs = []
        # 遍历所有判别器
        for i, d in enumerate(self.discriminators):
            # 如果不是第一个判别器，对真实数据和生成数据进行平均池化
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            # 对真实数据进行判别，并获取判别结果和特征图
            y_d_r, fmap_r = d(y)
            # 对生成数据进行判别，并获取判别结果和特征图
            y_d_g, fmap_g = d(y_hat)
            # 将真实数据的判别结果和特征图添加到对应的列表中
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            # 将生成数据的判别结果和特征图添加到对应的列表中
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        # 返回真实数据的判别结果列表、生成数据的判别结果列表、真实数据的特征图列表和生成数据的特征图列表
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
# 计算特征图之间的损失
def feature_loss(fmap_r, fmap_g):
    # 初始化损失值
    loss = 0
    # 遍历红色和绿色特征图，计算它们之间的绝对值平均损失，并累加到总损失中
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    # 返回总损失的两倍
    return loss * 2


# 计算鉴别器的损失
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    # 初始化损失值
    loss = 0
    # 初始化真实样本和生成样本的损失列表
    r_losses = []
    g_losses = []
    # 遍历真实样本和生成样本的输出，计算鉴别器损失，并累加到总损失中
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        # 将每个样本的损失值添加到对应的列表中
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    # 返回总损失和真实样本、生成样本的损失列表
    return loss, r_losses, g_losses


# 计算生成器的损失
def generator_loss(disc_outputs):
    # 初始化损失值
    loss = 0
    # 初始化生成器损失列表
    gen_losses = []
    # 遍历生成器输出，计算生成器损失，并累加到总损失中
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    # 返回总损失和生成器损失列表
    return loss, gen_losses
```