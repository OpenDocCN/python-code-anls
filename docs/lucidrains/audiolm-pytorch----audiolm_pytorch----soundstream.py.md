# `.\lucidrains\audiolm-pytorch\audiolm_pytorch\soundstream.py`

```py
# 导入必要的库
import functools
from pathlib import Path
from functools import partial, wraps
from itertools import cycle, zip_longest
from typing import Optional, List

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from torch.linalg import vector_norm

import torchaudio.transforms as T
from torchaudio.functional import resample

from einops import rearrange, reduce, pack, unpack

# 导入自定义模块
from vector_quantize_pytorch import (
    GroupedResidualVQ,
    GroupedResidualLFQ,
    GroupedResidualFSQ
)

from local_attention import LocalMHA
from local_attention.transformer import FeedForward, DynamicPositionBias

from gateloop_transformer import SimpleGateLoopLayer as GateLoop

from audiolm_pytorch.utils import curtail_to_multiple

from audiolm_pytorch.version import __version__
from packaging import version
parsed_version = version.parse(__version__)

import pickle

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将元组转换为指定长度的元组
def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

# 根据键过滤字典
def filter_by_keys(fn, d):
    return {k: v for k, v in d.items() if fn(k)}

# 映射字典键
def map_keys(fn, d):
    return {fn(k): v for k, v in d.items()}

# GAN 损失函数

# 对数函数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# Hinge 判别器损失
def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

# Hinge 生成器损失
def hinge_gen_loss(fake):
    return -fake.mean()

# Leaky ReLU 激活函数
def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

# 梯度惩罚
def gradient_penalty(wave, output, weight = 10):
    batch_size, device = wave.shape[0], wave.device

    gradients = torch_grad(
        outputs = output,
        inputs = wave,
        grad_outputs = torch.ones_like(output),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((vector_norm(gradients, dim = 1) - 1) ** 2).mean()

# 更好的序列化函数

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# 判别器

class MultiScaleDiscriminator(Module):
    def __init__(
        self,
        channels = 16,
        layers = 4,
        groups = (4, 16, 64, 256),
        chan_max = 1024,
        input_channels = 1
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(input_channels, channels, 15, padding = 7)
        self.conv_layers = ModuleList([])

        curr_channels = channels

        for _, group in zip(range(layers), groups):
            chan_out = min(curr_channels * 4, chan_max)

            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(curr_channels, chan_out, 41, stride = 4, padding = 20, groups = group),
                leaky_relu()
            ))

            curr_channels = chan_out

        self.final_conv = nn.Sequential(
            nn.Conv1d(curr_channels, curr_channels, 5, padding = 2),
            leaky_relu(),
            nn.Conv1d(curr_channels, 1, 3, padding = 1),
        )

    def forward(
        self,
        x,
        return_intermediates = False
    ):
        x = self.init_conv(x)
        intermediates = []

        for layer in self.conv_layers:
            x = layer(x)
            intermediates.append(x)

        out = self.final_conv(x)

        if not return_intermediates:
            return out

        return out, intermediates

# 自回归挤压激励
# https://arxiv.org/abs/1709.01507

class SqueezeExcite(Module):
    def __init__(self, dim, reduction_factor = 4, dim_minimum = 8):
        super().__init__()
        dim_inner = max(dim_minimum, dim // reduction_factor)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, 1),
            nn.SiLU(),
            nn.Conv1d(dim_inner, dim, 1),
            nn.Sigmoid()
        )
    # 定义前向传播函数，输入参数 x
    def forward(self, x):
        # 获取输入 x 的序列长度和设备信息
        seq, device = x.shape[-2], x.device

        # 计算累积均值 - 因为是自回归的

        # 沿着倒数第二个维度对 x 进行累积求和
        cum_sum = x.cumsum(dim = -2)
        # 创建一个序列长度范围的张量，转换为浮点数类型，并移动到指定设备
        denom = torch.arange(1, seq + 1, device = device).float()
        # 计算累积均值，即累积和除以对应的序号
        cum_mean = cum_sum / rearrange(denom, 'n -> n 1')

        # glu 门

        # 通过神经网络计算门控值
        gate = self.net(cum_mean)

        # 返回输入 x 与门控值的乘积
        return x * gate
# 定义一个复杂的短时傅里叶变换鉴别器

class ModReLU(Module):
    """
    https://arxiv.org/abs/1705.09792
    https://github.com/pytorch/pytorch/issues/47052#issuecomment-718948801
    """
    # 定义一个自定义的激活函数模块，参考论文和GitHub链接
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        # 返回修正的ReLU激活函数应用于输入 x 的结果
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.j * torch.angle(x))

class ComplexConv2d(Module):
    # 定义一个复杂卷积层模块
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size,
        stride = 1,
        padding = 0
    ):
        super().__init__()
        # 创建一个普通的卷积层对象
        conv = nn.Conv2d(dim, dim_out, kernel_size, dtype = torch.complex64)
        # 将卷积层的权重和偏置参数转换为复数类型
        self.weight = nn.Parameter(torch.view_as_real(conv.weight))
        self.bias = nn.Parameter(torch.view_as_real(conv.bias))

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # 将权重和偏置参数转换为复数类型
        weight, bias = map(torch.view_as_complex, (self.weight, self.bias))

        x = x.to(weight.dtype)
        # 返回卷积操作的结果
        return F.conv2d(x, weight, bias, stride = self.stride, padding = self.padding)

def ComplexSTFTResidualUnit(chan_in, chan_out, strides):
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))

    return nn.Sequential(
        # 定义一个复杂短时傅里叶变换残差单元
        Residual(Sequential(
            ComplexConv2d(chan_in, chan_in, 3, padding = 1),
            ModReLU(),
            ComplexConv2d(chan_in, chan_in, 3, padding = 1)
        )),
        ComplexConv2d(chan_in, chan_out, kernel_sizes, stride = strides, padding = paddings)
    )

class ComplexSTFTDiscriminator(Module):
    # 定义一个复杂短时傅里叶变换鉴别器模块
    def __init__(
        self,
        *,
        channels = 32,
        strides = ((1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)),
        chan_mults = (1, 2, 4, 4, 8, 8),
        input_channels = 1,
        n_fft = 1024,
        hop_length = 256,
        win_length = 1024,
        stft_normalized = False,
        stft_window_fn = torch.hann_window,
        logits_abs = True
    ):
        super().__init__()
        # 初始化卷积层
        self.init_conv = ComplexConv2d(input_channels, channels, 7, padding = 3)

        layer_channels = tuple(map(lambda mult: mult * channels, chan_mults))
        layer_channels = (channels, *layer_channels)
        layer_channels_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        curr_channels = channels

        self.layers = ModuleList([])

        for layer_stride, (chan_in, chan_out) in zip(strides, layer_channels_pairs):
            # 添加复杂短时傅里叶变换残差单元到层列表中
            self.layers.append(ComplexSTFTResidualUnit(chan_in, chan_out, layer_stride))

        # 添加最终的卷积层
        self.final_conv = ComplexConv2d(layer_channels[-1], 1, (16, 1)) # todo: remove hardcoded 16

        # stft 设置

        self.stft_normalized = stft_normalized
        self.stft_window_fn = stft_window_fn

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # 如何将对数输出转换为实数空间

        self.logits_abs = logits_abs
    # 定义一个前向传播函数，接受输入 x 和是否返回中间结果的标志
    def forward(self, x, return_intermediates = False):
        # 重新排列输入张量 x 的维度，将 'b 1 n' 转换为 'b n'
        x = rearrange(x, 'b 1 n -> b n')

        '''
        reference: The content of the paper( https://arxiv.org/pdf/2107.03312.pdf)is as follows:
        The STFT-based discriminator is illustrated in Figure 4
        and operates on a single scale, computing the STFT with a
        window length of W = 1024 samples and a hop length of
        H = 256 samples
        '''
        
        # 使用 self.stft_window_fn 函数生成 STFT 窗口
        stft_window = self.stft_window_fn(self.win_length, device = x.device)

        # 计算输入 x 的短时傅里叶变换（STFT）
        x = torch.stft(
            x,
            self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window = stft_window,
            normalized = self.stft_normalized,
            return_complex = True
        )

        # 重新排列 STFT 结果的维度，将 'b ...' 转换为 'b 1 ...'
        x = rearrange(x, 'b ... -> b 1 ...')

        intermediates = []

        # 对输入 x 进行初始卷积操作
        x = self.init_conv(x)

        intermediates.append(x)

        # 遍历所有层进行处理
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        # 对最终卷积结果进行处理，得到复数形式的 logits
        complex_logits = self.final_conv(x)

        # 如果 logits_abs 为 True，则取复数 logits 的绝对值
        if self.logits_abs:
            complex_logits = complex_logits.abs()
        else:
            complex_logits = torch.view_as_real(complex_logits)

        # 如果不需要返回中间结果，则直接返回复数 logits
        if not return_intermediates:
            return complex_logits

        # 如果需要返回中间结果，则同时返回复数 logits 和中间结果列表
        return complex_logits, intermediates
# 定义一个名为 Residual 的类，继承自 Module 类
class Residual(Module):
    # 初始化函数，接受一个名为 fn 的 Module 对象作为参数
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    # 前向传播函数，接受输入 x 和关键字参数 kwargs
    def forward(self, x, **kwargs):
        # 返回输入 x 经过 fn 处理后的结果与 x 相加的结果
        return self.fn(x, **kwargs) + x

# 定义一个名为 ChannelTranspose 的类，继承自 Module 类
class ChannelTranspose(Module):
    # 初始化函数，接受一个名为 fn 的 Module 对象作为参数
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    # 前向传播函数，接受输入 x 和关键字参数 kwargs
    def forward(self, x, **kwargs):
        # 将输入 x 的维度重新排列为 'b c n'
        x = rearrange(x, 'b c n -> b n c')
        # 将重新排列后的输入 x 经过 fn 处理后的结果与 x 相加的结果
        out = self.fn(x, **kwargs) + x
        # 将输出 out 的维度重新排列为 'b n c'
        return rearrange(out, 'b n c -> b c n')

# 定义一个名为 CausalConv1d 的类，继承自 Module 类
class CausalConv1d(Module):
    # 初始化函数，接受通道数 chan_in、输出通道数 chan_out、卷积核大小 kernel_size 和填充模式 pad_mode 等参数
    def __init__(self, chan_in, chan_out, kernel_size, pad_mode = 'reflect', **kwargs):
        super().__init__()
        # 设置卷积核大小
        kernel_size = kernel_size
        # 获取关键字参数中的膨胀值和步长
        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)
        self.pad_mode = pad_mode
        # 计算因果填充值
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)

        # 创建一个 1D 卷积层
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 对输入 x 进行填充，使用填充模式 pad_mode
        x = F.pad(x, (self.causal_padding, 0), mode = self.pad_mode)
        # 将填充后的输入 x 经过卷积层处理后返回
        return self.conv(x)

# 定义一个名为 CausalConvTranspose1d 的类，继承自 Module 类
class CausalConvTranspose1d(Module):
    # 初始化函数，接受通道数 chan_in、输出通道数 chan_out、卷积核大小 kernel_size 和步长 stride 等参数
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        # 创建一个 1D 转置卷积层
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    # 前向传播函数，接受输入 x
    def forward(self, x):
        n = x.shape[-1]

        # 将输入 x 经过转置卷积层处理后返回，并截取指定长度的输出
        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        return out

# 定义一个名为 ResidualUnit 的函数，接受输入通道数 chan_in、输出通道数 chan_out、膨胀值 dilation 等参数
def ResidualUnit(chan_in, chan_out, dilation, kernel_size = 7, squeeze_excite = False, pad_mode = 'reflect'):
    # 返回一个 Residual 类的实例，包含一系列操作
    return Residual(Sequential(
        CausalConv1d(chan_in, chan_out, kernel_size, dilation = dilation, pad_mode = pad_mode),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1, pad_mode = pad_mode),
        nn.ELU(),
        SqueezeExcite(chan_out) if squeeze_excite else None
    ))

# 定义一个名为 EncoderBlock 的函数，接受输入通道数 chan_in、输出通道数 chan_out、步长 stride 等参数
def EncoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9), squeeze_excite = False, pad_mode = 'reflect'):
    # 创建一个循环迭代器
    it = cycle(cycle_dilations)
    # 使用偏函数创建一个 ResidualUnit 函数的部分应用
    residual_unit = partial(ResidualUnit, squeeze_excite = squeeze_excite, pad_mode = pad_mode)

    return nn.Sequential(
        # 一系列残差单元和卷积操作组成的编码器块
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        CausalConv1d(chan_in, chan_out, 2 * stride, stride = stride)
    )

# 定义一个名为 DecoderBlock 的函数，接受输入通道数 chan_in、输出通道数 chan_out、步长 stride 等参数
def DecoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9), squeeze_excite = False, pad_mode = 'reflect'):
    even_stride = (stride % 2 == 0)
    padding = (stride + (0 if even_stride else 1)) // 2
    output_padding = 0 if even_stride else 1

    residual_unit = partial(ResidualUnit, squeeze_excite = squeeze_excite, pad_mode = pad_mode)

    it = cycle(cycle_dilations)
    return nn.Sequential(
        # 一系列残差单元和卷积操作组成的解码器块
        CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride = stride),
        residual_unit(chan_out, chan_out, next(it)),
        residual_unit(chan_out, chan_out, next(it)),
        residual_unit(chan_out, chan_out, next(it)),
    )

# 定义一个名为 LocalTransformer 的类，继承自 Module 类
class LocalTransformer(Module):
    # 初始化函数，接受关键字参数 dim、depth、heads、window_size、dynamic_pos_bias 等
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        window_size,
        dynamic_pos_bias = False,
        **kwargs
        ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化窗口大小
        self.window_size = window_size
        # 初始化层列表
        self.layers = ModuleList([])

        # 初始化位置偏置
        self.pos_bias = None
        # 如果需要动态位置偏置
        if dynamic_pos_bias:
            # 创建动态位置偏置对象
            self.pos_bias = DynamicPositionBias(dim = dim // 2, heads = heads)

        # 根据深度循环创建多个层
        for _ in range(depth):
            # 每个层包含局部多头注意力和前馈网络
            self.layers.append(ModuleList([
                LocalMHA(
                    dim = dim,
                    heads = heads,
                    qk_rmsnorm = True,
                    window_size = window_size,
                    use_rotary_pos_emb = not dynamic_pos_bias,
                    gate_values_per_head = True,
                    use_xpos = True,
                    **kwargs
                ),
                FeedForward(dim = dim)
            ]))

    # 前向传播函数
    def forward(self, x):
        # 获取窗口大小
        w = self.window_size

        # 如果存在位置偏置，则计算注意力偏置
        attn_bias = self.pos_bias(w, w * 2) if exists(self.pos_bias) else None

        # 遍历每个层，依次进行局部多头注意力和前馈网络操作
        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        # 返回处理后的数据
        return x
class FiLM(Module):
    # 定义 FiLM 类，继承自 Module 类
    def __init__(self, dim, dim_cond):
        # 初始化函数，接受两个参数 dim 和 dim_cond
        super().__init__()
        # 调用父类的初始化函数
        self.to_cond = nn.Linear(dim_cond, dim * 2)
        # 创建一个线性层，输入维度为 dim_cond，输出维度为 dim * 2

    def forward(self, x, cond):
        # 前向传播函数，接受输入 x 和条件 cond
        gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
        # 将条件 cond 输入到线性层中，得到 gamma 和 beta
        return x * gamma + beta
        # 返回经过 FiLM 操作后的结果

class SoundStream(Module):
    # 定义 SoundStream 类，继承自 Module 类
    def __init__(
        self,
        *,
        channels = 32,
        strides = (2, 4, 5, 8),
        channel_mults = (2, 4, 8, 16),
        codebook_dim = 512,
        codebook_size: Optional[int] = None,
        finite_scalar_quantizer_levels: Optional[List[int]] = None,
        rq_num_quantizers = 8,
        rq_commitment_weight = 1.,
        rq_ema_decay = 0.95,
        rq_quantize_dropout_multiple_of = 1,
        rq_groups = 1,
        rq_stochastic_sample_codes = False,
        rq_kwargs: dict = {},
        use_lookup_free_quantizer = False,              
        use_finite_scalar_quantizer = False,            
        input_channels = 1,
        discr_multi_scales = (1, 0.5, 0.25),
        stft_normalized = False,
        enc_cycle_dilations = (1, 3, 9),
        dec_cycle_dilations = (1, 3, 9),
        multi_spectral_window_powers_of_two = tuple(range(6, 12)),
        multi_spectral_n_ffts = 512,
        multi_spectral_n_mels = 64,
        recon_loss_weight = 1.,
        multi_spectral_recon_loss_weight = 1e-5,
        adversarial_loss_weight = 1.,
        feature_loss_weight = 100,
        quantize_dropout_cutoff_index = 1,
        target_sample_hz = 16000,
        use_local_attn = True,
        attn_window_size = 128,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_depth = 1,
        attn_xpos_scale_base = None,
        attn_dynamic_pos_bias = False,
        use_gate_loop_layers = False,
        squeeze_excite = False,
        complex_stft_discr_logits_abs = True,
        pad_mode = 'reflect',
        stft_discriminator: Optional[Module] = None,  
        complex_stft_discr_kwargs: dict = dict()
    @property
    def device(self):
        # 返回模型参数所在的设备
        return next(self.parameters()).device

    @property
    def configs(self):
        # 返回模型的配置信息
        return pickle.loads(self._configs)

    def decode_from_codebook_indices(self, quantized_indices):
        # 从量化索引解码得到输出
        assert quantized_indices.dtype in (torch.long, torch.int32)

        if quantized_indices.ndim == 3:
            quantized_indices = rearrange(quantized_indices, 'b n (g q) -> g b n q', g = self.rq_groups)

        x = self.rq.get_output_from_indices(quantized_indices)

        return self.decode(x)

    def decode(self, x, quantize = False):
        # 解码函数，接受输入 x 和是否进行量化的标志
        if quantize:
            x, *_ = self.rq(x)

        if exists(self.decoder_attn):
            x = self.decoder_attn(x)

        x = rearrange(x, 'b n c -> b c n')
        return self.decoder(x)

    def save(self, path):
        # 保存模型参数到指定路径
        path = Path(path)
        pkg = dict(
            model = self.state_dict(),
            config = self._configs,
            version = __version__
        )

        torch.save(pkg, str(path))

    @classmethod
    def init_and_load_from(cls, path, strict = True):
        # 初始化���从指定路径加载模型
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        soundstream = cls(**config)
        soundstream.load(path, strict = strict)
        soundstream.eval()
        return soundstream
    # 加载模型参数
    def load(self, path, strict = True):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 断言路径存在
        assert path.exists()
        # 加载模型参数
        pkg = torch.load(str(path), map_location = 'cpu')

        # 检查版本

        # 如果包中包含版本信息且版本小于指定版本，则打印警告信息
        if 'version' in pkg and version.parse(pkg['version']) < parsed_version:
            print(f'soundstream model being loaded was trained on an older version of audiolm-pytorch ({pkg["version"]})')

        # 检查是否有 EMA 模型
        has_ema = 'ema_model' in pkg
        # 选择要加载的模型参数
        model_pkg = pkg['ema_model'] if has_ema else pkg['model']

        # 如果有 EMA 模型，则对模型参数进行处理
        if has_ema:
            # 过滤出以 'ema_model.' 开头的键
            model_pkg = filter_by_keys(lambda k: k.startswith('ema_model.'), model_pkg)
            # 将键名中的 'ema_model.' 替换为空
            model_pkg = map_keys(lambda k: k[len('ema_model.'):], model_pkg)

        # 加载模型参数
        self.load_state_dict(model_pkg, strict = strict)

    # 从训练器保存的对象中加载模型参数
    def load_from_trainer_saved_obj(self, path):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 断言路径存在
        assert path.exists()
        # 加载模型参数
        obj = torch.load(str(path))
        self.load_state_dict(obj['model'])

    # 返回非判别器参数
    def non_discr_parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *(self.encoder_attn.parameters() if exists(self.encoder_attn) else []),
            *(self.decoder_attn.parameters() if exists(self.decoder_attn) else []),
            *self.encoder_film.parameters(),
            *self.decoder_film.parameters(),
            *self.rq.parameters()
        ]

    # 返回序列长度的倍数
    @property
    def seq_len_multiple_of(self):
        return functools.reduce(lambda x, y: x * y, self.strides)

    # 返回下采样因子
    @property
    def downsample_factor(self):
        return self.seq_len_multiple_of

    # 处理输入数据
    def process_input(
        self,
        x,
        input_sample_hz = None,
        curtail_from_left = False
    ):
        # 打包输入数据
        x, ps = pack([x], '* n')

        # 如果输入采样率存在，则重新采样输入数据
        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        # 对输入数据进行截断
        x = curtail_to_multiple(x, self.seq_len_multiple_of, from_left = curtail_from_left)

        # 如果输入数据维度为 2，则重新排列维度
        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')

        return x, ps

    # 对音频数据进行编码
    @torch.no_grad()
    def tokenize(self, audio):
        self.eval()
        return self.forward(audio, return_codes_only = True)

    # 前向传播函数
    def forward(
        self,
        x,
        target = None,
        is_denoising = None, # 如果要学习教 SoundStream 进行去噪的 film conditioner - 需要在上面传入目标
        return_encoded = False,
        return_codes_only = False,
        return_discr_loss = False,
        return_discr_losses_separately = False,
        return_loss_breakdown = False,
        return_recons_only = False,
        input_sample_hz = None,
        apply_grad_penalty = False,
        curtail_from_left = False
# 定义一个默认的音频语音流函数，参数包括步长、目标采样率和 RQ 量化器数量
def AudioLMSoundStream(
    strides = (2, 4, 5, 8),
    target_sample_hz = 16000,
    rq_num_quantizers = 12,
    **kwargs
):
    # 返回一个音频流对象，参数包括步长、目标采样率和 RQ 量化器数量
    return SoundStream(
        strides = strides,
        target_sample_hz = target_sample_hz,
        rq_num_quantizers = rq_num_quantizers,
        **kwargs
    )

# 定义一个默认的音乐语音流函数，参数包括步长、目标采样率和 RQ 量化器数量
def MusicLMSoundStream(
    strides = (3, 4, 5, 8),
    target_sample_hz = 24000,
    rq_num_quantizers = 12,
    **kwargs
):
    # 返回一个音频流对象，参数包括步长、目标采样率和 RQ 量化器数量
    return SoundStream(
        strides = strides,
        target_sample_hz = target_sample_hz,
        rq_num_quantizers = rq_num_quantizers,
        **kwargs
    )
```