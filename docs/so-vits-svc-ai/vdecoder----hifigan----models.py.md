# `so-vits-svc\vdecoder\hifigan\models.py`

```py
# 导入所需的库
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

# 从自定义的环境模块中导入属性字典
from .env import AttrDict
# 从自定义的工具模块中导入函数
from .utils import get_padding, init_weights

# 定义 Leaky ReLU 的斜率
LRELU_SLOPE = 0.1

# 加载模型的函数
def load_model(model_path, device='cuda'):
    # 从模型路径中获取配置文件路径
    config_file = os.path.join(os.path.split(model_path)[0], 'config.json')
    # 打开配置文件并读取内容
    with open(config_file) as f:
        data = f.read()

    # 全局变量 h，用于存储配置信息
    global h
    # 将配置文件内容解析为 JSON 格式
    json_config = json.loads(data)
    # 将 JSON 格式的配置信息转换为属性字典
    h = AttrDict(json_config)

    # 创建生成器对象，并将其移动到指定设备上
    generator = Generator(h).to(device)

    # 加载模型参数
    cp_dict = torch.load(model_path)
    # 加载生成器的状态字典
    generator.load_state_dict(cp_dict['generator'])
    # 将生成器设置为评估模式
    generator.eval()
    # 移除生成器中的权重归一化
    generator.remove_weight_norm()
    # 释放模型参数的内存
    del cp_dict
    # 返回生成器对象和配置信息
    return generator, h

# 定义一个 ResNet 模块
class ResBlock1(torch.nn.Module):
    # 初始化函数，定义了ResBlock1类的构造方法
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        # 调用父类的初始化方法
        super(ResBlock1, self).__init__()
        # 保存参数h
        self.h = h
        # 定义一组卷积层，使用nn.ModuleList包装
        self.convs1 = nn.ModuleList([
            # 使用weight_norm对卷积层进行权重归一化
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))
        ])
        # 对convs1中的所有卷积层应用初始化权重的函数
        self.convs1.apply(init_weights)

        # 定义另一组卷积层，使用nn.ModuleList包装
        self.convs2 = nn.ModuleList([
            # 使用weight_norm对卷积层进行权重归一化
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        # 对convs2中的所有卷积层应用初始化权重的函数
        self.convs2.apply(init_weights)

    # 前向传播函数
    def forward(self, x):
        # 遍历convs1和convs2中的卷积层
        for c1, c2 in zip(self.convs1, self.convs2):
            # 使用LeakyReLU激活函数对输入进行处理
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 将输入通过第一个卷积层
            xt = c1(xt)
            # 使用LeakyReLU激活函数对输出进行处理
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            # 将输出通过第二个卷积层
            xt = c2(xt)
            # 将输入和输出相加，得到残差连接的结果
            x = xt + x
        # 返回最终的输出
        return x

    # 移除权重归一化
    def remove_weight_norm(self):
        # 遍历convs1中的所有卷积层，移除权重归一化
        for l in self.convs1:
            remove_weight_norm(l)
        # 遍历convs2中的所有卷积层，移除权重归一化
        for l in self.convs2:
            remove_weight_norm(l)
class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h  # 初始化ResBlock2对象的h属性
        self.convs = nn.ModuleList([  # 创建包含两个卷积层的ModuleList
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],  # 第一个卷积层
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],  # 第二个卷积层
                               padding=get_padding(kernel_size, dilation[1]))
        ])
        self.convs.apply(init_weights)  # 对所有卷积层应用初始化权重的函数

    def forward(self, x):
        for c in self.convs:  # 遍历所有卷积层
            xt = F.leaky_relu(x, LRELU_SLOPE)  # 使用LeakyReLU激活函数
            xt = c(xt)  # 对输入数据进行卷积操作
            x = xt + x  # 将卷积结果与输入数据相加
        return x  # 返回最终结果

    def remove_weight_norm(self):
        for l in self.convs:  # 遍历所有卷积层
            remove_weight_norm(l)  # 移除权重归一化

def padDiff(x):
    return F.pad(F.pad(x, (0,0,-1,1), 'constant', 0) - x, (0,0,0,-1), 'constant', 0)  # 对输入数据进行填充和差分操作

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
        # 生成 UV 信号，如果 f0 大于有声门阈值则为 1，否则为 0
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
        # 设置源声音生成器的 ONNX 标志为 True
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
        # 对输入进行预处理
        x = self.conv_pre(x)
        # 将输入与条件信息相加
        x = x + self.cond(g)
        # 进行多次上采样和残差连接
        for i in range(self.num_upsamples):
            # 对输入进行 LeakyReLU 激活
            x = F.leaky_relu(x, LRELU_SLOPE)
            # 进行上采样
            x = self.ups[i](x)
            # 对声音源进行卷积
            x_source = self.noise_convs[i](har_source)
            # 将上采样结果与声音源相加
            x = x + x_source
            xs = None
            # 对残差块进行处理
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # 对结果进行平均处理
            x = xs / self.num_kernels
        # 对结果进行 LeakyReLU 激活
        x = F.leaky_relu(x)
        # 对结果进行后处理
        x = self.conv_post(x)
        # 对结果进行 tanh 处理
        x = torch.tanh(x)

        return x

    # 移除权重归一化的方法
    def remove_weight_norm(self):
        # 打印移除权重归一化的提示信息
        print('Removing weight norm...')
        # 对上采样层进行权重归一化移除
        for l in self.ups:
            remove_weight_norm(l)
        # 对残差块进行权重归一化移除
        for l in self.resblocks:
            l.remove_weight_norm()
        # 对预处理卷积层进行权重归一化移除
        remove_weight_norm(self.conv_pre)
        # 对后处理卷积层进行权重归一化移除
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
            self.discriminators.append(DiscriminatorP(period))  # 创建对应周期的 DiscriminatorP 实例
    # 定义一个前向传播函数，接受真实数据 y 和生成数据 y_hat 作为输入
    def forward(self, y, y_hat):
        # 初始化真实数据在每个鉴别器的判别结果列表和特征图列表
        y_d_rs = []
        fmap_rs = []
        # 初始化生成数据在每个鉴别器的判别结果列表和特征图列表
        y_d_gs = []
        fmap_gs = []
        # 遍历每个鉴别器
        for i, d in enumerate(self.discriminators):
            # 对真实数据进行鉴别器判别，得到判别结果和特征图
            y_d_r, fmap_r = d(y)
            # 对生成数据进行鉴别器判别，得到判别结果和特征图
            y_d_g, fmap_g = d(y_hat)
            # 将真实数据的判别结果和特征图添加到对应的列表中
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            # 将生成数据的判别结果和特征图添加到对应的列表中
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        # 返回真实数据在每个鉴别器的判别结果列表、生成数据在每个鉴别器的判别结果列表、真实数据在每个鉴别器的特征图列表、生成数据在每个鉴别器的特征图列表
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        # 初始化判别器模型
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
    # 定义一个前向传播函数，接受真实值 y 和预测值 y_hat 作为输入
    def forward(self, y, y_hat):
        # 初始化用于存储每个判别器输出的真实值得分和特征图的列表
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        # 遍历每个判别器
        for i, d in enumerate(self.discriminators):
            # 如果不是第一个判别器，对 y 和 y_hat 进行平均池化
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            # 分别将 y 和 y_hat 输入判别器，获取真实值得分和特征图
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            # 将真实值得分和特征图添加到对应的列表中
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        # 返回所有判别器输出的真实值得分和特征图
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
    # 遍历真实输出和生成输出，计算它们之间的损失
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
        # 总损失为所有生成器输出损失之和
        loss += l

    return loss, gen_losses
```