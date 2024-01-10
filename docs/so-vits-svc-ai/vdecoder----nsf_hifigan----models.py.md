# `so-vits-svc\vdecoder\nsf_hifigan\models.py`

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

# 从自定义的 env 模块中导入 AttrDict 类
from .env import AttrDict
# 从自定义的 utils 模块中导入 get_padding 和 init_weights 函数
from .utils import get_padding, init_weights

# 定义 Leaky ReLU 的斜率
LRELU_SLOPE = 0.1

# 加载模型的函数，返回生成器和配置信息
def load_model(model_path, device='cuda'):
    # 加载模型的配置信息
    h = load_config(model_path)
    # 创建生成器并将其移动到指定设备上
    generator = Generator(h).to(device)
    # 加载模型参数
    cp_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(cp_dict['generator'])
    # 设置生成器为评估模式
    generator.eval()
    # 移除参数中的权重归一化
    generator.remove_weight_norm()
    # 释放内存
    del cp_dict
    # 返回生成器和配置信息
    return generator, h

# 加载模型的配置信息
def load_config(model_path):
    # 从模型路径中获取配置文件路径
    config_file = os.path.join(os.path.split(model_path)[0], 'config.json')
    # 读取配置文件内容
    with open(config_file) as f:
        data = f.read()
    # 将配置文件内容解析为 JSON 格式
    json_config = json.loads(data)
    # 使用 AttrDict 封装配置信息
    h = AttrDict(json_config)
    # 返回封装后的配置信息
    return h

# 定义 ResBlock1 类，继承自 torch.nn.Module
class ResBlock1(torch.nn.Module):
    # 初始化函数，定义了一个残差块模型
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        # 调用父类的初始化函数
        super(ResBlock1, self).__init__()
        # 保存参数 h
        self.h = h
        # 定义第一组卷积层，使用 nn.ModuleList 包装多个卷积层
        self.convs1 = nn.ModuleList([
            # 使用权重归一化的一维卷积层
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))
        ])
        # 对第一组卷积层应用初始化权重函数
        self.convs1.apply(init_weights)

        # 定义第二组卷积层，使用 nn.ModuleList 包装多个卷积层
        self.convs2 = nn.ModuleList([
            # 使用权重归一化的一维卷积层
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        # 对第二组卷积层应用初始化权重函数
        self.convs2.apply(init_weights)

    # 前向传播函数
    def forward(self, x):
        # 遍历第一组和第二组卷积层
        for c1, c2 in zip(self.convs1, self.convs2):
            # 使用 LeakyReLU 激活函数
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 第一组卷积操作
            xt = c1(xt)
            # 使用 LeakyReLU 激活函数
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            # 第二组卷积操作
            xt = c2(xt)
            # 残差连接
            x = xt + x
        # 返回结果
        return x

    # 移除权重归一化
    def remove_weight_norm(self):
        # 遍历第一组卷积层，移除权重归一化
        for l in self.convs1:
            remove_weight_norm(l)
        # 遍历第二组卷积层，移除权重归一化
        for l in self.convs2:
            remove_weight_norm(l)
class ResBlock2(torch.nn.Module):
    # 定义一个名为ResBlock2的torch模块
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        # 初始化函数，接受参数h, channels, kernel_size, dilation
        super(ResBlock2, self).__init__()
        # 调用父类的初始化函数
        self.h = h
        # 设置实例变量h为传入的参数h
        self.convs = nn.ModuleList([
            # 创建一个包含两个卷积层的模块列表
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            # 第一个卷积层，使用weight_norm进行权重归一化
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
            # 第二个卷积层，使用weight_norm进行权重归一化
        ])
        self.convs.apply(init_weights)
        # 对模块列表中的所有模块应用初始化权重的函数

    def forward(self, x):
        # 前向传播函数，接受输入x
        for c in self.convs:
            # 遍历模块列表中的所有模块
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 使用LeakyReLU激活函数处理输入x
            xt = c(xt)
            # 将处理后的输入传入当前遍历的卷积层
            x = xt + x
            # 将卷积层的输出与输入相加
        return x
        # 返回最终的输出x

    def remove_weight_norm(self):
        # 定义一个移除权重归一化的函数
        for l in self.convs:
            # 遍历模块列表中的所有模块
            remove_weight_norm(l)
            # 移除当前遍历模块的权重归一化


class SineGen(torch.nn.Module):
    # 定义一个名为SineGen的torch模块，继承自torch.nn.Module
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
    # SineGen类的说明文档，包括参数说明和注意事项

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0):
        # 初始化函数，接受参数samp_rate, harmonic_num, sine_amp, noise_std, voiced_threshold
        super(SineGen, self).__init__()
        # 调用父类的初始化函数
        self.sine_amp = sine_amp
        # 设置实例变量sine_amp为传入的参数sine_amp
        self.noise_std = noise_std
        # 设置实例变量noise_std为传入的参数noise_std
        self.harmonic_num = harmonic_num
        # 设置实例变量harmonic_num为传入的参数harmonic_num
        self.dim = self.harmonic_num + 1
        # 设置实例变量dim为harmonic_num加1
        self.sampling_rate = samp_rate
        # 设置实例变量sampling_rate为传入的参数samp_rate
        self.voiced_threshold = voiced_threshold
        # 设置实例变量voiced_threshold为传入的参数voiced_threshold
    # 定义一个私有方法，用于生成 UV 信号
    def _f02uv(self, f0):
        # 生成一个与 f0 相同形状的全为1的张量作为初始的 UV 信号
        uv = torch.ones_like(f0)
        # 根据 voiced_threshold 将 UV 信号中对应 voiced 部分的值设为1，非 voiced 部分的值保持为0
        uv = uv * (f0 > self.voiced_threshold)
        # 返回生成的 UV 信号
        return uv

    # 禁止在该方法中进行梯度计算
    @torch.no_grad()
# 定义名为 SourceModuleHnNSF 的类，用于 hn-nsf 模型的源模块
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

    # 初始化方法，接受采样率、谐波数、正弦信号振幅、添加噪声标准差和有声门限作为参数
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        # 调用父类的初始化方法
        super(SourceModuleHnNSF, self).__init__()

        # 设置正弦信号振幅和噪声标准差
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # 创建 SineGen 实例，用于生成正弦波形
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # 创建 Linear 层，用于将源谐波合并为单一激励
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    # 前向传播方法，接受输入 x 和 upp
    def forward(self, x, upp):
        # 使用 SineGen 生成正弦波形、声音类型和无用类型
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        # 将正弦波形输入到 Linear 层和 Tanh 激活函数中，得到合并后的正弦波形
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        # 返回合并后的正弦波形
        return sine_merge


class Generator(torch.nn.Module):
    # 初始化生成器对象
    def __init__(self, h):
        # 调用父类的初始化方法
        super(Generator, self).__init__()
        # 保存参数
        self.h = h
        # 计算残差块的数量
        self.num_kernels = len(h.resblock_kernel_sizes)
        # 计算上采样率的数量
        self.num_upsamples = len(h.upsample_rates)
        # 创建音频源模块对象
        self.m_source = SourceModuleHnNSF(
            sampling_rate=h.sampling_rate,
            harmonic_num=8
        )
        # 初始化噪声卷积层列表
        self.noise_convs = nn.ModuleList()
        # 创建预处理卷积层
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        # 根据参数选择不同的残差块类型
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        # 初始化上采样层列表
        self.ups = nn.ModuleList()
        # 遍历上采样率和卷积核大小
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            # 计算当前通道数
            c_cur = h.upsample_initial_channel // (2 ** (i + 1))
            # 添加上采样卷积层
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))
            # 如果不是最后一层上采样
            if i + 1 < len(h.upsample_rates):  #
                # 计算步长
                stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                # 添加噪声卷积层
                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
            else:
                # 添加噪声卷积层
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        # 初始化残差块列表
        self.resblocks = nn.ModuleList()
        # 初始化通道数
        ch = h.upsample_initial_channel
        # 遍历上采样层
        for i in range(len(self.ups)):
            # 通道数减半
            ch //= 2
            # 遍历残差块的卷积核大小和扩张率
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                # 添加残差块
                self.resblocks.append(resblock(h, ch, k, d))

        # 创建后处理卷积层
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        # 初始化上采样层和后处理卷积层的权重
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        # 计算上采样率的乘积
        self.upp = int(np.prod(h.upsample_rates))
    # 定义前向传播函数，接受输入 x 和条件 f0
    def forward(self, x, f0):
        # 从条件 f0 和上采样参数 upp 中获取哈尔小波源
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        # 对输入 x 进行预处理卷积
        x = self.conv_pre(x)
        # 循环进行上采样
        for i in range(self.num_upsamples):
            # 对 x 进行 LeakyReLU 激活函数处理
            x = F.leaky_relu(x, LRELU_SLOPE)
            # 使用上采样层对 x 进行上采样
            x = self.ups[i](x)
            # 使用哈尔小波源对 x 进行噪声卷积
            x_source = self.noise_convs[i](har_source)
            # 将上采样结果和哈尔小波源卷积结果相加
            x = x + x_source
            # 初始化 xs 为 None
            xs = None
            # 循环进行残差块处理
            for j in range(self.num_kernels):
                # 如果 xs 为 None，则将残差块处理结果赋值给 xs
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                # 否则，将残差块处理结果累加到 xs 上
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # 对 xs 进行均值处理
            x = xs / self.num_kernels
        # 对 x 进行 LeakyReLU 激活函数处理
        x = F.leaky_relu(x)
        # 对 x 进行后处理卷积
        x = self.conv_post(x)
        # 对 x 进行 tanh 激活函数处理
        x = torch.tanh(x)

        # 返回处理后的 x
        return x

    # 移除权重归一化
    def remove_weight_norm(self):
        # 打印移除权重归一化的提示信息
        print('Removing weight norm...')
        # 遍历上采样层，移除权重归一化
        for l in self.ups:
            remove_weight_norm(l)
        # 遍历残差块，移除权重归一化
        for l in self.resblocks:
            l.remove_weight_norm()
        # 移除前处理卷积的权重归一化
        remove_weight_norm(self.conv_pre)
        # 移除后处理卷积的权重归一化
        remove_weight_norm(self.conv_post)
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 创建卷积层列表，每个卷积层都经过权重归一化处理
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        # 创建最后的卷积层，经过权重归一化处理
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

        for l in self.convs:  # 遍历卷积层列表
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)  # 使用 Leaky ReLU 激活函数
            fmap.append(x)  # 将中间特征图添加到列表中
        x = self.conv_post(x)  # 经过最后的卷积层
        fmap.append(x)  # 将最终特征图添加到列表中
        x = torch.flatten(x, 1, -1)  # 将特征图展平为一维向量

        return x, fmap  # 返回一维向量和中间特征图列表


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=None):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:  # 遍历周期列表
            self.discriminators.append(DiscriminatorP(period))  # 创建对应周期的判别器实例并添加到列表中
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
            # 对真实数据进行鉴别，得到判别结果和特征图
            y_d_r, fmap_r = d(y)
            # 对生成数据进行鉴别，得到判别结果和特征图
            y_d_g, fmap_g = d(y_hat)
            # 将真实数据的判别结果和特征图添加到对应的列表中
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            # 将生成数据的判别结果和特征图添加到对应的列表中
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        # 返回真实数据在每个鉴别器上的判别结果列表、生成数据在每个鉴别器上的判别结果列表、真实数据在每个鉴别器上的特征图列表、生成数据在每个鉴别器上的特征图列表
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        # 初始化函数，定义判别器模型
        super(DiscriminatorS, self).__init__()
        # 根据是否使用谱归一化选择相应的归一化函数
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
        # 对最后一层卷积进行处理
        x = self.conv_post(x)
        # 将最后一层卷积结果存储到特征图列表中
        fmap.append(x)
        # 对最后一层卷积结果进行展平操作
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        # 初始化函数，定义多尺度判别器模型
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
    # 定义一个前向传播函数，接受真实值 y 和预测值 y_hat 作为输入
    def forward(self, y, y_hat):
        # 初始化真实值的判别器输出列表和特征图列表
        y_d_rs = []
        fmap_rs = []
        # 初始化预测值的判别器输出列表和特征图列表
        y_d_gs = []
        fmap_gs = []
        # 遍历所有的判别器
        for i, d in enumerate(self.discriminators):
            # 如果不是第一个判别器，对真实值和预测值进行平均池化
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            # 对真实值输入判别器，获取判别器输出和特征图
            y_d_r, fmap_r = d(y)
            # 对预测值输入判别器，获取判别器输出和特征图
            y_d_g, fmap_g = d(y_hat)
            # 将真实值的判别器输出和特征图添加到对应的列表中
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            # 将预测值的判别器输出和特征图添加到对应的列表中
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        # 返回真实值判别器输出列表、预测值判别器输出列表、真实值特征图列表、预测值特征图列表
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
# 计算特征图之间的损失
def feature_loss(fmap_r, fmap_g):
    # 初始化损失值
    loss = 0
    # 遍历红色和绿色特征图，计算它们之间的损失
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            # 计算每个像素点的绝对差值，并求平均
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


# 计算鉴别器的损失
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    # 初始化损失值
    loss = 0
    r_losses = []
    g_losses = []
    # 遍历真实输出和生成输出，计算它们的损失
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        # 计算真实输出的损失
        r_loss = torch.mean((1 - dr) ** 2)
        # 计算生成输出的损失
        g_loss = torch.mean(dg ** 2)
        # 总损失为真实输出损失和生成输出损失之和
        loss += (r_loss + g_loss)
        # 将真实输出和生成输出的损失值添加到列表中
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


# 计算生成器的损失
def generator_loss(disc_outputs):
    # 初始化损失值
    loss = 0
    gen_losses = []
    # 遍历生成器输出，计算其损失
    for dg in disc_outputs:
        # 计算生成器输出的损失
        l = torch.mean((1 - dg) ** 2)
        # 将生成器输出的损失值添加到列表中
        gen_losses.append(l)
        # 总损失为所有生成器输出的损失之和
        loss += l

    return loss, gen_losses
```