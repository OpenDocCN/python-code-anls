# `.\lucidrains\naturalspeech2-pytorch\naturalspeech2_pytorch\naturalspeech2_pytorch.py`

```py
# 导入所需的库
import math
import copy
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from audiolm_pytorch import SoundStream, EncodecWrapper
from audiolm_pytorch.data import SoundDataset, get_dataloader

from beartype import beartype
from beartype.typing import Tuple, Union, Optional, List
from beartype.door import is_bearable

from naturalspeech2_pytorch.attend import Attend
from naturalspeech2_pytorch.aligner import Aligner, ForwardSumLoss, BinLoss
from naturalspeech2_pytorch.utils.tokenizer import Tokenizer, ESpeak
from naturalspeech2_pytorch.utils.utils import average_over_durations, create_mask
from naturalspeech2_pytorch.version import __version__

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm
import pyworld as pw

# 定义常量

mlist = nn.ModuleList

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# 辅助函数

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(num, den):
    return (num % den) == 0

def identity(t, *args, **kwargs):
    return t

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

# 张量辅助函数

def pad_or_curtail_to_length(t, length):
    if t.shape[-1] == length:
        return t

    if t.shape[-1] > length:
        return t[..., :length]

    return F.pad(t, (0, length - t.shape[-1]))

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

def generate_mask_from_repeats(repeats):
    repeats = repeats.int()
    device = repeats.device

    lengths = repeats.sum(dim=-1)
    max_length = lengths.amax().item()
    cumsum = repeats.cumsum(dim=-1)
    cumsum_exclusive = F.pad(cumsum, (1, -1), value=0.)

    seq = torch.arange(max_length, device=device)
    seq = repeat(seq, '... j -> ... i j', i=repeats.shape[-1])

    cumsum = rearrange(cumsum, '... i -> ... i 1')
    cumsum_exclusive = rearrange(cumsum_exclusive, '... i -> ... i 1')

    lengths = rearrange(lengths, 'b -> b 1 1')
    mask = (seq < cumsum) & (seq >= cumsum_exclusive) & (seq < lengths)
    return mask

# 正弦位置嵌入

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# 计算音高

def compute_pitch_pytorch(wav, sample_rate):
    # 使用 torchaudio 库中的 compute_kaldi_pitch 函数计算音高特征
    pitch_feature = torchaudio.functional.compute_kaldi_pitch(wav, sample_rate)
    pitch, nfcc = pitch_feature.unbind(dim=-1)
    return pitch

# 根据论文使用 pyworld 计算音高

def compute_pitch_pyworld(wav, sample_rate, hop_length, pitch_fmax=640.0):
    is_tensor_input = torch.is_tensor(wav)

    if is_tensor_input:
        device = wav.device
        wav = wav.contiguous().cpu().numpy()
    # 如果音频长度可以被 hop_length 整除，则在末尾填充一半的 hop_length 长度，使用反射模式填充
    if divisible_by(len(wav), hop_length):
        wav = np.pad(wav, (0, hop_length // 2), mode="reflect")

    # 将音频数据类型转换为双精度浮点型
    wav = wav.astype(np.double)

    # 初始化一个空列表用于存储音频样本的基频值
    outs = []

    # 遍历音频样本，提取基频值
    for sample in wav:
        # 使用 dio 函数提取音频样本的基频值和时间信息
        f0, t = pw.dio(
            sample,
            fs = sample_rate,
            f0_ceil = pitch_fmax,
            frame_period = 1000 * hop_length / sample_rate,
        )

        # 使用 stonemask 函数对基频值进行修正
        f0 = pw.stonemask(sample, f0, t, sample_rate)
        # 将修正后的基频值添加到 outs 列表中
        outs.append(f0)

    # 将 outs 列表转换为 numpy 数组
    outs = np.stack(outs)

    # 如果输入是张量形式，则将 outs 转换为张量并移动到指定设备上
    if is_tensor_input:
        outs = torch.from_numpy(outs).to(device)

    # 返回提取的基频值
    return outs
def f0_to_coarse(f0, f0_bin = 256, f0_max = 1100.0, f0_min = 50.0):
    # 计算最大和最小频率对应的梅尔频率
    f0_mel_max = 1127 * torch.log(1 + torch.tensor(f0_max) / 700)
    f0_mel_min = 1127 * torch.log(1 + torch.tensor(f0_min) / 700)

    # 计算输入频率对应的梅尔频率
    f0_mel = 1127 * (1 + f0 / 700).log()
    # 对梅尔频率进行线性变换，映射到[1, f0_bin-1]的范围
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    # 将小于等于1的值设置为1
    f0_mel[f0_mel <= 1] = 1
    # 将大于f0_bin-1的值设置为f0_bin-1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    # 对梅尔频率四舍五入取整
    f0_coarse = (f0_mel + 0.5).int()
    # 断言确保f0_coarse的取值范围在[1, 255]之间
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse

# peripheral models

# audio to mel

class AudioToMel(nn.Module):
    def __init__(
        self,
        *,
        n_mels = 100,
        sampling_rate = 24000,
        f_max = 8000,
        n_fft = 1024,
        win_length = 640,
        hop_length = 160,
        log = True
    ):
        super().__init__()
        self.log = log
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_max = f_max
        self.win_length = win_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate

    def forward(self, audio):
        # 创建STFT变换对象
        stft_transform = T.Spectrogram(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            window_fn = torch.hann_window
        )

        # 对音频进行STFT变换得到频谱图
        spectrogram = stft_transform(audio)

        # 创建梅尔频率变换对象
        mel_transform = T.MelScale(
            n_mels = self.n_mels,
            sample_rate = self.sampling_rate,
            n_stft = self.n_fft // 2 + 1,
            f_max = self.f_max
        )

        # 对频谱图进行梅尔频率变换得到梅尔频谱图
        mel = mel_transform(spectrogram)

        # 如果log为True，则将梅尔频谱图转换为对数幅度
        if self.log:
            mel = T.AmplitudeToDB()(mel)

        return mel

# phoneme - pitch - speech prompt - duration predictors

class PhonemeEncoder(nn.Module):
    def __init__(
        self,
        *,
        tokenizer: Optional[Tokenizer] = None,
        num_tokens = None,
        dim = 512,
        dim_hidden = 512,
        kernel_size = 9,
        depth = 6,
        dim_head = 64,
        heads = 8,
        conv_dropout = 0.2,
        attn_dropout = 0.,
        use_flash = False
    ):
        super().__init__()

        # 初始化模型参数
        self.tokenizer = tokenizer
        num_tokens = default(num_tokens, tokenizer.vocab_size if exists(tokenizer) else None)

        self.token_emb = nn.Embedding(num_tokens + 1, dim) if exists(num_tokens) else nn.Identity()
        self.pad_id = num_tokens

        same_padding = (kernel_size - 1) // 2

        # 定义卷积层和变换层
        self.conv = nn.Sequential(
            Rearrange('b n c -> b c n'),
            CausalConv1d(dim, dim_hidden, kernel_size),
            nn.SiLU(),
            nn.Dropout(conv_dropout),
            Rearrange('b c n -> b n c'),
        )

        self.transformer = Transformer(
            dim = dim_hidden,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            dropout = attn_dropout,
            use_flash = use_flash
        )

    @beartype
    def forward(
        self,
        x: Union[Tensor, List[str]],
        mask = None
    ):
        # 如果输入为字符串列表，则将其转换为张量
        if is_bearable(x, List[str]):
            assert exists(self.tokenizer)
            x = self.tokenizer.texts_to_tensor_ids(x)

        # 将小于0的值设置为pad_id
        is_padding = x < 0
        x = x.masked_fill(is_padding, self.pad_id)

        x = self.token_emb(x)
        x = self.conv(x)
        x = self.transformer(x, mask = mask)
        return x

class SpeechPromptEncoder(nn.Module):

    @beartype
    def __init__(
        self,
        dim_codebook,
        dims: Tuple[int] = (256, 2048, 2048, 2048, 2048, 512, 512, 512),
        *,
        depth = 6,
        heads = 8,
        dim_head = 64,
        dropout = 0.2,
        kernel_size = 9,
        padding = 4,
        use_flash_attn = True
    # 定义一个继承自 nn.Module 的类，用于实现一个包含卷积和Transformer的模型
    ):
        # 调用父类的构造函数
        super().__init__()

        # 将dim_codebook添加到dims列表的开头
        dims = [dim_codebook, *dims]

        # 设置self.dim为dims列表的第一个元素，设置self.dim_out为dims列表的最后一个元素
        self.dim, self.dim_out = dims[0], dims[-1]

        # 将dims列表中相邻的两个元素组成一对，形成一个维度对的列表
        dim_pairs = zip(dims[:-1], dims[1:])

        # 初始化一个空的模块列表
        modules = []
        # 遍历维度对列表，为每一对维度创建一个卷积层和SiLU激活函数，并添加到模块列表中
        for dim_in, dim_out in dim_pairs:
            modules.extend([
                nn.Conv1d(dim_in, dim_out, kernel_size, padding = padding),
                nn.SiLU()
            ])

        # 构建一个包含卷积层和SiLU激活函数的序列模块
        self.conv = nn.Sequential(
            Rearrange('b n c -> b c n'),
            *modules,
            Rearrange('b c n -> b n c')
        )

        # 初始化一个Transformer模块
        self.transformer = Transformer(
            dim = dims[-1],
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            dropout = dropout,
            use_flash = use_flash_attn
        )

    # 定义前向传播函数
    def forward(self, x):
        # 断言输入张量x的最后一个维度与self.dim相等
        assert x.shape[-1] == self.dim

        # 将输入张量通过卷积层和Transformer模块进行前向传播
        x = self.conv(x)
        x = self.transformer(x)
        return x
# 定义一个名为 Block 的类，继承自 nn.Module
class Block(nn.Module):
    # 初始化函数，接受输入维度 dim、输出维度 dim_out、卷积核大小 kernel、分组数 groups 和 dropout 概率
    def __init__(
        self,
        dim,
        dim_out,
        kernel = 3,
        groups = 8,
        dropout = 0.
    ):
        super().__init__()
        # 创建一个卷积层，将输入维度映射到输出维度
        self.proj = nn.Conv1d(dim, dim_out, kernel, padding = kernel // 2)
        # 对输出进行分组归一化
        self.norm = nn.GroupNorm(groups, dim_out)
        # 使用 SiLU 激活函数
        self.act = nn.SiLU()
        # 使用 dropout 进行正则化
        self.dropout = nn.Dropout(dropout)

    # 前向传播函数
    def forward(self, x):
        # 对输入进行卷积操作
        x = self.proj(x)
        # 对卷积结果进行分组归一化
        x = self.norm(x)
        # 使用激活函数
        x = self.act(x)
        # 使用 dropout
        x = self.dropout(x)
        return x

# 定义一个名为 ResnetBlock 的类，继承自 nn.Module
class ResnetBlock(nn.Module):
    # 初始化函数，接受输入维度 dim、输出维度 dim_out、卷积核大小 kernel、dropout 概率、分组数 groups 和卷积层数 num_convs
    def __init__(
        self,
        dim,
        dim_out,
        kernel,
        *,
        dropout = 0.,
        groups = 8,
        num_convs = 2
    ):
        super().__init__()

        blocks = []
        # 循环创建 num_convs 个 Block 实例
        for ind in range(num_convs):
            is_first = ind == 0
            dim_in = dim if is_first else dim_out
            # 创建一个 Block 实例
            block = Block(
                dim_in,
                dim_out,
                kernel,
                groups = groups,
                dropout = dropout
            )
            blocks.append(block)

        # 将所有 Block 实例组合成一个序列
        self.blocks = nn.Sequential(*blocks)

        # 如果输入维度和输出维度不相等，使用 1x1 卷积进行维度匹配，否则使用恒等映射
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # 前向传播函数
    def forward(self, x):
        # 将输入维度重新排列
        x = rearrange(x, 'b n c -> b c n')
        # 对输入进行 Block 序列操作
        h = self.blocks(x)
        # 将 Block 序列的输出与输入进行残差连接
        out = h + self.res_conv(x)
        # 将输出维度重新排列
        return rearrange(out, 'b c n -> b n c')

# 定义一个函数 ConvBlock，接受输入维度 dim、输出维度 dim_out、卷积核大小 kernel 和 dropout 概率
def ConvBlock(dim, dim_out, kernel, dropout = 0.):
    # 返回一个包含卷积、激活函数、dropout 的序列
    return nn.Sequential(
        Rearrange('b n c -> b c n'),
        nn.Conv1d(dim, dim_out, kernel, padding = kernel // 2),
        nn.SiLU(),
        nn.Dropout(dropout),
        Rearrange('b c n -> b n c'),
    )

# 定义一个名为 DurationPitchPredictorTrunk 的类，继承自 nn.Module
class DurationPitchPredictorTrunk(nn.Module):
    # 初始化函数，接受输入维度 dim、深度 depth、卷积核大小 kernel_size、上下文维度 dim_context、头数 heads、头维度 dim_head、dropout 概率、是否使用 ResNet 块 use_resnet_block、每个 ResNet 块的卷积层数 num_convs_per_resnet_block、每个块的卷积层数 num_convolutions_per_block、是否使用 Flash 注意力 use_flash_attn
    def __init__(
        self,
        dim = 512,
        depth = 10,
        kernel_size = 3,
        dim_context = None,
        heads = 8,
        dim_head = 64,
        dropout = 0.2,
        use_resnet_block = True,
        num_convs_per_resnet_block = 2,
        num_convolutions_per_block = 3,
        use_flash_attn = False,
    ):
        super().__init__()
        # 初始化一个空的模块列表
        self.layers = nn.ModuleList([])

        # 根据是否使用 ResNet 块选择卷积类
        conv_klass = ConvBlock if not use_resnet_block else partial(ResnetBlock, num_convs = num_convs_per_resnet_block)

        # 循环创建 depth 个层
        for _ in range(depth):
            # 每个层包含一个卷积序列、RMSNorm 归一化和注意力机制
            layer = nn.ModuleList([
                nn.Sequential(*[
                    conv_klass(dim, dim, kernel_size) for _ in range(num_convolutions_per_block)
                ]),
                RMSNorm(dim),
                Attention(
                    dim,
                    dim_context = dim_context,
                    heads = heads,
                    dim_head = dim_head,
                    dropout = dropout,
                    use_flash = use_flash_attn,
                    cross_attn_include_queries = True
                )
            ])

            self.layers.append(layer)

        # 最后的预测层，包含线性层、维度重排和 ReLU 激活函数
        self.to_pred = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...'),
            nn.ReLU()
        )
    
    # 前向传播函数，接受输入 x、编码的提示信息 encoded_prompts 和提示信息的掩码 prompt_mask
    def forward(
        self,
        x,
        encoded_prompts,
        prompt_mask = None,
    ):
        # 对每个层进行操作
        for conv, norm, attn in self.layers:
            x = conv(x)
            x = attn(norm(x), encoded_prompts, mask = prompt_mask) + x

        return self.to_pred(x)

# 定义一个名为 DurationPitchPredictor 的类，继承自 nn.Module
class DurationPitchPredictor(nn.Module):
    # 初始化函数，接受维度 dim、音素标记数 num_phoneme_tokens、分词器 tokenizer、编码提示信息的维度 dim_encoded_prompts、每个块的卷积层数 num_convolutions_per_block、是否使用 ResNet 块 use_resnet_block、每个 ResNet 块的卷积层数 num_convs_per_resnet_block、深度 depth、卷积核大小 kernel_size、头数 heads、头维度 dim_head、隐藏层维度 dim_hidden、dropout 概率、是否使用 Flash 注意力 use_flash_attn
    def __init__(
        self,
        *,
        dim,
        num_phoneme_tokens = None,
        tokenizer: Optional[Tokenizer] = None,
        dim_encoded_prompts = None,
        num_convolutions_per_block = 3,
        use_resnet_block = True,
        num_convs_per_resnet_block = 2,
        depth = 10,
        kernel_size = 3,
        heads = 8,
        dim_head = 64,
        dim_hidden = 512,
        dropout = 0.2,
        use_flash_attn = False
    ):
        super().__init__()
        # 略
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化 tokenizer 属性
        self.tokenizer = tokenizer
        # 如果存在 tokenizer，则将 num_phoneme_tokens 设置为 tokenizer 的词汇表大小，否则为 None
        num_phoneme_tokens = default(num_phoneme_tokens, tokenizer.vocab_size if exists(tokenizer) else None)

        # 如果未提供 dim_encoded_prompts，则将其设置为 dim
        dim_encoded_prompts = default(dim_encoded_prompts, dim)

        # 如果存在 num_phoneme_tokens，则创建一个 num_phoneme_tokens x dim 的嵌入层，否则创建一个恒等映射
        self.phoneme_token_emb = nn.Embedding(num_phoneme_tokens, dim) if exists(num_phoneme_tokens) else nn.Identity()

        # 初始化 to_pitch_pred 属性为 DurationPitchPredictorTrunk 类的实例
        self.to_pitch_pred = DurationPitchPredictorTrunk(
            dim = dim_hidden,
            depth = depth,
            kernel_size = kernel_size,
            dim_context = dim_encoded_prompts,
            heads = heads,
            dim_head = dim_head,
            dropout = dropout,
            use_resnet_block = use_resnet_block,
            num_convs_per_resnet_block = num_convs_per_resnet_block,
            num_convolutions_per_block = num_convolutions_per_block,
            use_flash_attn = use_flash_attn,
        )

        # 使用深拷贝创建 to_duration_pred 属性
        self.to_duration_pred = copy.deepcopy(self.to_pitch_pred)

    # 定义 forward 方法
    @beartype
    def forward(
        self,
        x: Union[Tensor, List[str]],
        encoded_prompts,
        prompt_mask = None
    ):
        # 如果 x 是 List[str] 类型，则将其转换为张量
        if is_bearable(x, List[str]):
            assert exists(self.tokenizer)
            x = self.tokenizer.texts_to_tensor_ids(x)

        # 对输入 x 进行嵌入
        x = self.phoneme_token_emb(x)

        # 使用 map 函数对 to_duration_pred 和 to_pitch_pred 进行计算
        duration_pred, pitch_pred = map(lambda fn: fn(x, encoded_prompts = encoded_prompts, prompt_mask = prompt_mask), (self.to_duration_pred, self.to_pitch_pred))

        # 返回持续时间预测和音高预测结果
        return duration_pred, pitch_pred
# 使用来自 flamingo 论文的 Perceiver Resampler，替代 "q-k-v" 注意力机制，其中 m 个查询成为网络条件的关键/值

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_context = None,
        num_latents = 64, # 论文中的 m
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        use_flash_attn = False
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std = 0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    use_flash = use_flash_attn,
                    cross_attn_include_queries = True
                ),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x, mask = None):
        batch = x.shape[0]

        x = self.proj_context(x)

        latents = repeat(self.latents, 'n d -> b n d', b = batch)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask = mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)

# 模型，即 Wavenet + Transformer

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel_size, = self.kernel_size
        dilation, = self.dilation
        stride, = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value = 0.)
        return super().forward(causal_padded_x)

class WavenetResBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dilation,
        kernel_size = 3,
        skip_conv = False,
        dim_cond_mult = None
    ):
        super().__init__()

        self.cond = exists(dim_cond_mult)
        self.to_time_cond = None

        if self.cond:
            self.to_time_cond = nn.Linear(dim * dim_cond_mult, dim * 2)

        self.conv = CausalConv1d(dim, dim, kernel_size, dilation = dilation)
        self.res_conv = CausalConv1d(dim, dim, 1)
        self.skip_conv = CausalConv1d(dim, dim, 1) if skip_conv else None

    def forward(self, x, t = None):

        if self.cond:
            assert exists(t)
            t = self.to_time_cond(t)
            t = rearrange(t, 'b c -> b c 1')
            t_gamma, t_beta = t.chunk(2, dim = -2)

        res = self.res_conv(x)

        x = self.conv(x)

        if self.cond:
            x = x * t_gamma + t_beta

        x = x.tanh() * x.sigmoid()

        x = x + res

        skip = None
        if exists(self.skip_conv):
            skip = self.skip_conv(x)

        return x, skip


class WavenetStack(nn.Module):
    def __init__(
        self,
        dim,
        *,
        layers,
        kernel_size = 3,
        has_skip = False,
        dim_cond_mult = None
    ):
        super().__init__()
        dilations = 2 ** torch.arange(layers)

        self.has_skip = has_skip
        self.blocks = mlist([])

        for dilation in dilations.tolist():
            block = WavenetResBlock(
                dim = dim,
                kernel_size = kernel_size,
                dilation = dilation,
                skip_conv = has_skip,
                dim_cond_mult = dim_cond_mult
            )

            self.blocks.append(block)
    # 定义前向传播函数，接受输入 x 和时间 t
    def forward(self, x, t):
        # 初始化残差和跳跃连接列表
        residuals = []
        skips = []

        # 如果输入 x 是张量类型，则将其重复多次，以匹配网络块的数量
        if isinstance(x, Tensor):
            x = (x,) * len(self.blocks)

        # 遍历输入 x 和网络块，计算残差和跳跃连接
        for block_input, block in zip(x, self.blocks):
            residual, skip = block(block_input, t)

            # 将计算得到的残差和跳跃连接添加到对应的列表中
            residuals.append(residual)
            skips.append(skip)

        # 如果存在跳跃连接，则返回所有跳跃连接的张量堆叠
        if self.has_skip:
            return torch.stack(skips)

        # 否则返回所有残差的列表
        return residuals
class Wavenet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        stacks,
        layers,
        init_conv_kernel = 3,
        dim_cond_mult = None
    ):
        # 初始化 Wavenet 类
        super().__init__()
        # 创建初始卷积层对象
        self.init_conv = CausalConv1d(dim, dim, init_conv_kernel)
        # 初始化堆栈列表
        self.stacks = mlist([])

        # 循环创建堆栈
        for ind in range(stacks):
            is_last = ind == (stacks - 1)

            # 创建 WavenetStack 对象
            stack = WavenetStack(
                dim,
                layers = layers,
                dim_cond_mult = dim_cond_mult,
                has_skip = is_last
            )

            # 将堆栈对象添加到堆栈列表中
            self.stacks.append(stack)

        # 创建最终卷积层对象
        self.final_conv = CausalConv1d(dim, dim, 1)

    def forward(self, x, t = None):
        # 对输入数据进行初始卷积
        x = self.init_conv(x)

        # 遍历堆栈列表，对数据进行处理
        for stack in self.stacks:
            x = stack(x, t)

        # 对处理后的数据进行最终卷积并返回结果
        return self.final_conv(x.sum(dim = 0))

class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, dim_cond = None):
        # 初始化 RMSNorm 类
        super().__init__()
        # 检查是否有条件输入
        self.cond = exists(dim_cond)
        # 根据条件初始化线性层
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        # 初始化缩放参数和 gamma 参数
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond = None):
        # 获取 gamma 参数
        gamma = default(self.gamma, 1)
        # 对输入数据进行归一化处理
        out = F.normalize(x, dim = -1) * self.scale * gamma

        # 如果没有条件输入，则直接返回处理后的数据
        if not self.cond:
            return out

        # 如果有条件输入，则根据条件计算 gamma 和 beta，并进行处理
        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))
        return out * gamma + beta

class ConditionableTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        ff_causal_conv = False,
        dim_cond_mult = None,
        cross_attn = False,
        use_flash = False
    ):
        # 初始化 ConditionableTransformer 类
        super().__init__()
        # 设置维度和层列表
        self.dim = dim
        self.layers = mlist([])

        # 检查是否有条件输入
        cond = exists(dim_cond_mult)

        # 根据条件初始化 RMSNorm 层
        maybe_adaptive_norm_kwargs = dict(scale = not cond, dim_cond = dim * dim_cond_mult) if cond else dict()
        rmsnorm = partial(RMSNorm, **maybe_adaptive_norm_kwargs)

        # 循环创建层
        for _ in range(depth):
            self.layers.append(mlist([
                rmsnorm(dim),
                Attention(dim = dim, dim_head = dim_head, heads = heads, use_flash = use_flash),
                rmsnorm(dim) if cross_attn else None,
                Attention(dim = dim, dim_head = dim_head, heads = heads, use_flash = use_flash) if cross_attn else None,
                rmsnorm(dim),
                FeedForward(dim = dim, mult = ff_mult, causal_conv = ff_causal_conv)
            ]))

        # 创建预测层
        self.to_pred = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim, bias = False)
        )

    def forward(
        self,
        x,
        times = None,
        context = None
    ):
        t = times

        # 遍历层列表，对输入数据进行处理
        for attn_norm, attn, cross_attn_norm, cross_attn, ff_norm, ff in self.layers:
            res = x
            x = attn_norm(x, cond = t)
            x = attn(x) + res

            # 如果有交叉注意力，则进行处理
            if exists(cross_attn):
                assert exists(context)
                res = x
                x = cross_attn_norm(x, cond = t)
                x = cross_attn(x, context = context) + res

            res = x
            x = ff_norm(x, cond = t)
            x = ff(x) + res

        # 返回预测结果
        return self.to_pred(x)

class Model(nn.Module):

    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        wavenet_layers = 8,
        wavenet_stacks = 4,
        dim_cond_mult = 4,
        use_flash_attn = True,
        dim_prompt = None,
        num_latents_m = 32,   # number of latents to be perceiver resampled ('q-k-v' with 'm' queries in the paper)
        resampler_depth = 2,
        cond_drop_prob = 0.,
        condition_on_prompt= False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化模型的维度
        self.dim = dim

        # 时间条件

        # 根据维度计算时间条件的维度
        dim_time = dim * dim_cond_mult

        # 创建时间条件的网络层
        self.to_time_cond = Sequential(
            LearnedSinusoidalPosEmb(dim),  # 学习的正弦位置编码
            nn.Linear(dim + 1, dim_time),   # 线性层，将输入维度转换为时间条件的维度
            nn.SiLU()                       # SiLU激活函数
        )

        # 提示条件

        self.cond_drop_prob = cond_drop_prob  # 用于分类器无指导的概率
        self.condition_on_prompt = condition_on_prompt
        self.to_prompt_cond = None

        if self.condition_on_prompt:
            self.null_prompt_cond = nn.Parameter(torch.randn(dim_time))  # 随机初始化空提示条件
            self.null_prompt_tokens = nn.Parameter(torch.randn(num_latents_m, dim))  # 随机初始化空提示标记

            nn.init.normal_(self.null_prompt_cond, std = 0.02)  # 使用正态分布初始化空提示条件
            nn.init.normal_(self.null_prompt_tokens, std = 0.02)  # 使用正态分布初始化空提示标记

            # 创建提示条件的网络层
            self.to_prompt_cond = Sequential(
                Reduce('b n d -> b d', 'mean'),  # 减少维度
                nn.Linear(dim_prompt, dim_time),  # 线性层，将输入维度转换为提示条件的维度
                nn.SiLU()  # SiLU激活函数
            )

            # 创建PerceiverResampler对象
            self.perceiver_resampler = PerceiverResampler(
                dim = dim,
                dim_context = dim_prompt,
                num_latents = num_latents_m,
                depth = resampler_depth,
                dim_head = dim_head,
                heads = heads,
                use_flash_attn = use_flash_attn
            )

        # 从对齐器和持续时间模块获取对齐的条件

        self.null_cond = None
        self.cond_to_model_dim = None

        if self.condition_on_prompt:
            self.cond_to_model_dim = nn.Conv1d(dim_prompt, dim, 1)  # 一维卷积层，将提示条件转换为模型维度
            self.null_cond = nn.Parameter(torch.zeros(dim, 1))  # 初始化空条件

        # 条件包括时间和可选的提示

        dim_cond_mult = dim_cond_mult * (2 if condition_on_prompt else 1)  # 更新条件的维度乘数

        # WaveNet

        # 创建WaveNet模型
        self.wavenet = Wavenet(
            dim = dim,
            stacks = wavenet_stacks,
            layers = wavenet_layers,
            dim_cond_mult = dim_cond_mult
        )

        # Transformer

        # 创建ConditionableTransformer模型
        self.transformer = ConditionableTransformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            ff_causal_conv = True,
            dim_cond_mult = dim_cond_mult,
            use_flash = use_flash_attn,
            cross_attn = condition_on_prompt
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        # 前向传播函数，带有条件缩放
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1.:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        times,
        prompt = None,
        prompt_mask = None,
        cond = None,
        cond_drop_prob = None
        ):
        # 获取输入张量 x 的 batch 大小
        b = x.shape[0]
        # 如果未指定条件丢弃概率，则使用默认值
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # 准备时间条件
        # 概率应该在向前移除

        # 将时间转换为条件
        t = self.to_time_cond(times)
        c = None

        # 如果存在 prompt 条件
        if exists(self.to_prompt_cond):
            assert exists(prompt)

            # 创建与 prompt 条件大小相同的概率掩码
            prompt_cond_drop_mask = prob_mask_like((b,), cond_drop_prob, self.device)

            # 将 prompt 转换为条件
            prompt_cond = self.to_prompt_cond(prompt)

            # 根据概率掩码更新 prompt 条件
            prompt_cond = torch.where(
                rearrange(prompt_cond_drop_mask, 'b -> b 1'),
                self.null_prompt_cond,
                prompt_cond,
            )

            # 将时间条件和 prompt 条件连接起来
            t = torch.cat((t, prompt_cond), dim = -1)

            # 对 prompt 进行重采样
            resampled_prompt_tokens = self.perceiver_resampler(prompt, mask = prompt_mask)

            # 根据概率掩码更新 prompt tokens
            c = torch.where(
                rearrange(prompt_cond_drop_mask, 'b -> b 1 1'),
                self.null_prompt_tokens,
                resampled_prompt_tokens
            )

        # 重新排列为通道优先格式
        x = rearrange(x, 'b n d -> b d n')

        # 将对齐的条件加到输入序列中
        if exists(self.cond_to_model_dim):
            assert exists(cond)
            # 将条件转换为模型维度
            cond = self.cond_to_model_dim(cond)

            # 创建与条件大小相同的概率掩码
            cond_drop_mask = prob_mask_like((b,), cond_drop_prob, self.device)

            # 根据概率掩码更新条件
            cond = torch.where(
                rearrange(cond_drop_mask, 'b -> b 1 1'),
                self.null_cond,
                cond
            )

            # 目前，将条件调整为潜在特征的长度
            cond = pad_or_curtail_to_length(cond, x.shape[-1])

            # 将条件加到输入张量中
            x = x + cond

        # 主要的 WaveNet 模块
        x = self.wavenet(x, t)
        x = rearrange(x, 'b d n -> b n d')

        # 使用 Transformer 模块
        x = self.transformer(x, t, context = c)
        return x
# feedforward

# GEGLU 激活函数类，用于前向传播
class GEGLU(nn.Module):
    # 前向传播函数
    def forward(self, x):
        # 将输入张量 x 按照最后一个维度分成两部分
        x, gate = x.chunk(2, dim = -1)
        # 返回 GEGLU 激活函数的结果
        return F.gelu(gate) * x

# 创建前馈神经网络层
def FeedForward(dim, mult = 4, causal_conv = False):
    # 计算内部维度
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    # 如果是因果卷积
    if causal_conv:
        # 创建因果卷积层
        conv = nn.Sequential(
            Rearrange('b n d -> b d n'),
            CausalConv1d(dim_inner, dim_inner, 3),
            Rearrange('b d n -> b n d'),
        )

    return Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        conv,
        nn.Linear(dim_inner, dim)
    )

# attention

# 注意力机制类
class Attention(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        *,
        dim_context = None,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        use_flash = False,
        cross_attn_include_queries = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal = causal, dropout = dropout, use_flash = use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    # 前向传播函数
    def forward(self, x, context = None, mask = None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim = -2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder

# Transformer 编码器类
class Transformer(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        *,
        depth,
        causal = False,
        dim_head = 64,
        heads = 8,
        use_flash = False,
        dropout = 0.,
        ff_mult = 4,
        final_norm = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        # 创建多层 Transformer 编码器
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RMSNorm(dim),
                Attention(
                    dim,
                    causal = causal,
                    dim_head = dim_head,
                    heads = heads,
                    dropout = dropout,
                    use_flash = use_flash
                ),
                RMSNorm(dim),
                FeedForward(
                    dim,
                    mult = ff_mult
                )
            ]))

        self.norm = RMSNorm(dim) if final_norm else nn.Identity()

    # 前向传播函数
    def forward(self, x, mask = None):
        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x), mask = mask) + x
            x = ff(ff_norm(x)) + x

        return self.norm(x)

# tensor helper functions

# 对数函数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# ���全除法函数
def safe_div(numer, denom):
    return numer / denom.clamp(min = 1e-10)

# 将 x 张量的维度右侧填充到与 t 张量相同维度
def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# noise schedules

# 简单线性调度函数
def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

# 余弦调度函数
def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

# sigmoid 调度函数
def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    # 根据起始时间和结束时间计算对应的 sigmoid 值
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    # 计算 gamma 值，用于调整时间范围
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    # 对 gamma 进行范围限制，确保在指定范围内
    return gamma.clamp_(min=clamp_min, max=1.)
# 将 gamma 转换为 alpha、sigma 或 logsnr

def gamma_to_alpha_sigma(gamma, scale = 1):
    # 计算 alpha 和 sigma，并乘以指定的比例
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    # 计算 logsnr，根据给定的 gamma、比例和 eps
    return log(gamma * (scale ** 2) / (1 - gamma), eps = eps)

# 高斯扩散

class NaturalSpeech2(nn.Module):

    @beartype
    def __init__(
        self,
        model: Model,
        codec: Optional[Union[SoundStream, EncodecWrapper]] = None,
        *,
        
        tokenizer: Optional[Tokenizer] = None,
        target_sample_hz = None,
        timesteps = 1000,
        use_ddim = True,
        noise_schedule = 'sigmoid',
        objective = 'v',
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        train_prob_self_cond = 0.9,
        rvq_cross_entropy_loss_weight = 0., # 默认关闭，直到确定其是否有效。不确定这是否至关重要
        dim_codebook: int = 128,
        duration_pitch_dim: int = 512,
        aligner_dim_in: int = 80,
        aligner_dim_hidden: int = 512,
        aligner_attn_channels: int = 80,
        num_phoneme_tokens: int = 150,
        pitch_emb_dim: int = 256,
        pitch_emb_pp_hidden_dim: int= 512,
        calc_pitch_with_pyworld = True,     # 使用 pyworld 或 kaldi 从 torchaudio 计算音高
        mel_hop_length = 160,
        audio_to_mel_kwargs: dict = dict(),
        scale = 1., # 在训练高分辨率图像时，将此设置为 < 1 以获得更好的收敛性
        duration_loss_weight = 1.,
        pitch_loss_weight = 1.,
        aligner_loss_weight = 1.,
        aligner_bin_loss_weight = 0.
    # 初始化函数，继承父类的初始化方法
    def __init__(
        self
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 设置条件变量
        self.conditional = model.condition_on_prompt

        # 设置模型和编解码器
        self.model = model
        self.codec = codec

        # 确保编解码器存在或目标采样率存在
        assert exists(codec) or exists(target_sample_hz)

        # 设置目标采样率和序列长度的倍数
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = None

        # 如果编解码器存在，则设置目标采样率和序列长度的倍数
        if exists(codec):
            self.target_sample_hz = codec.target_sample_hz
            self.seq_len_multiple_of = codec.seq_len_multiple_of

        # 准备条件
        if self.conditional:
            # 如果目标采样率存在，则更新音频到梅尔频谱的参数
            if exists(self.target_sample_hz):
                audio_to_mel_kwargs.update(sampling_rate = self.target_sample_hz)

            # 设置梅尔频谱的跳跃长度
            self.mel_hop_length = mel_hop_length

            # 创建音频到梅尔频谱的转换器
            self.audio_to_mel = AudioToMel(
                n_mels = aligner_dim_in,
                hop_length = mel_hop_length,
                **audio_to_mel_kwargs
            )

            # 设置是否使用 PyWorld 计算音高
            self.calc_pitch_with_pyworld = calc_pitch_with_pyworld

            # 初始化音素编码器、语音提示编码器、持续时间和音高预测器、对齐器、音高嵌入层等
            self.phoneme_enc = PhonemeEncoder(tokenizer=tokenizer, num_tokens=num_phoneme_tokens)
            self.prompt_enc = SpeechPromptEncoder(dim_codebook=dim_codebook)
            self.duration_pitch = DurationPitchPredictor(dim=duration_pitch_dim)
            self.aligner = Aligner(dim_in=aligner_dim_in, dim_hidden=aligner_dim_hidden, attn_channels=aligner_attn_channels)
            self.pitch_emb = nn.Embedding(pitch_emb_dim, pitch_emb_pp_hidden_dim)

            # 初始化对齐器损失和二值损失
            self.aligner_loss = ForwardSumLoss()
            self.bin_loss = BinLoss()
            self.aligner_bin_loss_weight = aligner_bin_loss_weight

        # 其余的 DDPM

        # 确保编解码器维度与模型维度相等
        assert not exists(codec) or model.dim == codec.codebook_dim, f'transformer model dimension {model.dim} must be equal to codec dimension {codec.codebook_dim}'

        # 设置维度
        self.dim = codec.codebook_dim if exists(codec) else model.dim

        # 确保目标是 'x0', 'eps', 'v' 中的一个
        assert objective in {'x0', 'eps', 'v'}, 'objective must be either predict x0 or noise'
        self.objective = objective

        # 根据噪声调度设置 gamma 调度
        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        # 设置缩放比例
        assert scale <= 1, 'scale must be less than or equal to 1'
        self.scale = scale

        # 设置 gamma 调度的参数
        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        # 设置时间步长和是否使用 DDIM
        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # 提出的方法，将时间差加到下一个时间步长，以修复自我条件不足和在采样时间步长小于 400 时降低 FID
        self.time_difference = time_difference

        # 训练时自我条件的概率
        self.train_prob_self_cond = train_prob_self_cond

        # 最小 SNR 损失权重
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        # 持续时间和音高的损失权重
        self.duration_loss_weight = duration_loss_weight
        self.pitch_loss_weight = pitch_loss_weight
        self.aligner_loss_weight = aligner_loss_weight

    # 设备属性
    @property
    def device(self):
        return next(self.model.parameters()).device

    # 打印方法
    def print(self, s):
        return self.accelerator.print(s)
    # 获取采样时间步长
    def get_sampling_timesteps(self, batch, *, device):
        # 在设备上创建一个从1到0的时间序列
        times = torch.linspace(1., 0., self.timesteps + 1, device=device)
        # 将时间序列重复batch次
        times = repeat(times, 't -> b t', b=batch)
        # 将时间序列拆分成相邻时间步长的对
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    # 生成DDPM采样
    @torch.no_grad()
    def ddpm_sample(self, shape, prompt=None, time_difference=None, cond_scale=1., cond=None):
        batch, device = shape[0], self.device

        # 设置时间差
        time_difference = default(time_difference, self.time_difference)

        # 获取采样时间对
        time_pairs = self.get_sampling_timesteps(batch, device=device)

        # 生成随机音频
        audio = torch.randn(shape, device=device)

        x_start = None
        last_latents = None

        # 遍历时间对
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step', total=self.timesteps):

            # 添加时间延迟
            time_next = (time_next - self.time_difference).clamp(min=0.)

            noise_cond = time

            # 获取预测的x0
            model_output = self.model.forward_with_cond_scale(audio, noise_cond, prompt=prompt, cond_scale=cond_scale, cond=cond)

            # 获取log(snr)
            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, audio), (gamma, gamma_next))

            # 获取alpha和sigma
            alpha, sigma = gamma_to_alpha_sigma(gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next, self.scale)

            # 计算x0和噪声
            if self.objective == 'x0':
                x_start = model_output
            elif self.objective == 'eps':
                x_start = safe_div(audio - sigma * model_output, alpha)
            elif self.objective == 'v':
                x_start = alpha * audio - sigma * model_output

            # 推导后验均值和方差
            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))
            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (audio * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # 获取噪声
            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(audio),
                torch.zeros_like(audio)
            )

            # 更新音频
            audio = mean + (0.5 * log_variance).exp() * noise

        return audio

    @torch.no_grad()
    # 生成一个指定形状的样本，可以设置时间差异、条件比例和条件
    def ddim_sample(self, shape, prompt = None, time_difference = None, cond_scale = 1., cond = None):
        # 获取批次大小和设备
        batch, device = shape[0], self.device

        # 设置时间差异
        time_difference = default(time_difference, self.time_difference)

        # 获取采样时间步
        time_pairs = self.get_sampling_timesteps(batch, device = device)

        # 生成随机噪声
        audio = torch.randn(shape, device = device)

        x_start = None
        last_latents = None

        # 遍历时间步
        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # 获取时间和噪声水平
            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            # 填充时间和噪声水平
            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, audio), (gamma, gamma_next))

            # 将噪声水平转换为 alpha 和 sigma
            alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next, self.scale)

            # 添加时间延迟
            times_next = (times_next - time_difference).clamp(min = 0.)

            # 预测 x0
            model_output = self.model.forward_with_cond_scale(audio, times, prompt = prompt, cond_scale = cond_scale, cond = cond)

            # 计算 x0 和噪声
            if self.objective == 'x0':
                x_start = model_output
            elif self.objective == 'eps':
                x_start = safe_div(audio - sigma * model_output, alpha)
            elif self.objective == 'v':
                x_start = alpha * audio - sigma * model_output

            # 获取预测噪声
            pred_noise = safe_div(audio - alpha * x_start, sigma)

            # 计算下一个 x
            audio = x_start * alpha_next + pred_noise * sigma_next

        return audio

    # 处理提示信息
    def process_prompt(self, prompt = None):
        if not exists(prompt):
            return None

        assert self.model.condition_on_prompt

        is_raw_prompt = prompt.ndim == 2
        assert not (is_raw_prompt and not exists(self.codec)), 'codec must be passed in if one were to train on raw prompt'

        if is_raw_prompt:
            with torch.no_grad():
                self.codec.eval()
                prompt, _, _ = self.codec(prompt, curtail_from_left = True, return_encoded = True)

        return prompt

    # 扩展编码
    def expand_encodings(self, phoneme_enc, attn, pitch):
        expanded_dur = einsum('k l m n, k j m -> k j n', attn, phoneme_enc)
        pitch_emb = self.pitch_emb(rearrange(f0_to_coarse(pitch), 'b 1 t -> b t'))
        pitch_emb = rearrange(pitch_emb, 'b t d -> b d t')
        expanded_pitch = einsum('k l m n, k j m -> k j n', attn, pitch_emb)
        expanded_encodings = expanded_dur + expanded_pitch
        return expanded_encodings

    # 生成样本
    @torch.no_grad()
    def sample(
        self,
        *,
        length,
        prompt = None,
        batch_size = 1,
        cond_scale = 1.,
        text = None,
        text_lens = None,
    ):
        # 如果不使用 DDIM，则使用 DDPM 进行采样
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample

        prompt_enc = cond = None

        # 如果是有条件的生成
        if self.conditional:
            # 确保 prompt 和 text 存在
            assert exists(prompt) and exists(text)
            # 处理 prompt
            prompt = self.process_prompt(prompt)
            # 对 prompt 进行编码
            prompt_enc = self.prompt_enc(prompt)
            # 对 text 进行音素编码
            phoneme_enc = self.phoneme_enc(text)

            # 计算音频的持续时间和音高
            duration, pitch = self.duration_pitch(phoneme_enc, prompt_enc)
            # 重新排列 pitch 的维度
            pitch = rearrange(pitch, 'b n -> b 1 n')

            # 生成基于重复的掩码
            aln_mask = generate_mask_from_repeats(duration).float()

            # 对编码进行扩展
            cond = self.expand_encodings(rearrange(phoneme_enc, 'b n d -> b d n'), rearrange(aln_mask, 'b n c -> b 1 n c'), pitch)

        # 如果 prompt 存在
        if exists(prompt):
            # 获取批量大小
            batch_size = prompt.shape[0]

        # 生成音频
        audio = sample_fn(
            (batch_size, length, self.dim),
            prompt = prompt_enc,
            cond = cond,
            cond_scale = cond_scale
        )

        # 如果存在编解码器
        if exists(self.codec):
            # 解码音频
            audio = self.codec.decode(audio)

            # 如果音频维度为 3
            if audio.ndim == 3:
                # 重新排列音频的维度
                audio = rearrange(audio, 'b 1 n -> b n')

        # 返回音频
        return audio

    def forward(
        self,
        audio,
        text = None,
        text_lens = None,
        mel = None,
        mel_lens = None,
        codes = None,
        prompt = None,
        pitch = None,
        *args,
        **kwargs
# trainer

# 定义一个循环生成器函数，用于循环遍历数据集
def cycle(dl):
    while True:
        for data in dl:
            yield data

# Trainer 类，用于训练模型
class Trainer(object):
    def __init__(
        self,
        diffusion_model: NaturalSpeech2,
        *,
        dataset: Optional[Dataset] = None,
        folder = None,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 1,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        use_ema = True,
        split_batches = True,
        dataloader = None,
        data_max_length = None,
        data_max_length_seconds = 2,
        sample_length = None
    ):
        super().__init__()

        # accelerator

        # 初始化加速器，用于加速训练过程
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        # 设置模型为扩散模型
        self.model = diffusion_model
        assert exists(diffusion_model.codec)

        self.dim = diffusion_model.dim

        # training hyperparameters

        # 设置训练超参数
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = dataloader

        if not exists(dl):
            assert exists(dataset) or exists(folder)

            if exists(dataset):
                self.ds = dataset
            elif exists(folder):
                # create dataset

                if exists(data_max_length_seconds):
                    assert not exists(data_max_length)
                    data_max_length = int(data_max_length_seconds * diffusion_model.target_sample_hz)
                else:
                    assert exists(data_max_length)

                # 创建数据集
                self.ds = SoundDataset(
                    folder,
                    max_length = data_max_length,
                    target_sample_hz = diffusion_model.target_sample_hz,
                    seq_len_multiple_of = diffusion_model.seq_len_multiple_of
                )

                dl = DataLoader(
                    self.ds,
                    batch_size = train_batch_size,
                    shuffle = True,
                    pin_memory = True,
                    num_workers = cpu_count()
                )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        # 初始化优化器
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        self.use_ema = use_ema
        self.ema = None

        if self.accelerator.is_main_process and use_ema:
            # make sure codec is not part of the EMA
            # encodec seems to be not deepcopyable, so this is a necessary hack

            codec = diffusion_model.codec
            diffusion_model.codec = None

            # 初始化指数移动平均模型
            self.ema = EMA(
                diffusion_model,
                beta = ema_decay,
                update_every = ema_update_every,
                ignore_startswith_names = set(['codec.'])
            ).to(self.device)

            diffusion_model.codec = codec
            self.ema.ema_model.codec = codec

        # sampling hyperparameters

        # 设置采样超参数
        self.sample_length = default(sample_length, data_max_length)
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        # results folder

        # 设置结果保存文件夹
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        # 设置步数计数器
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        # 使用加速器准备模型、数据加载器和优化器
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    # 打印函数
    def print(self, msg):
        return self.accelerator.print(msg)

    @property
    # 返回未包装的模型
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)
    
    # 返回设备加速器的设备
    @property
    def device(self):
        return self.accelerator.device

    # 保存训练里程碑的模型状态
    def save(self, milestone):
        # 如果不是本地主进程，则返回
        if not self.accelerator.is_local_main_process:
            return

        # 构建保存的数据字典
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        # 保存数据到文件
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    # 加载训练里程碑的模型状态
    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        # 从文件加载数据
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        # 解包模型并加载状态
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        # 打印加载的版本信息
        if 'version' in data:
            print(f"loading from version {data['version']}")

        # 如果存在加速器的缩放器和数据中的缩放器，则加载缩放器状态
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # 训练模型
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        # 使用 tqdm 显示训练进度
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                # 累积梯度并更新模型
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                # 如果是主进程，更新指数移动平均模型并保存模型
                if accelerator.is_main_process:
                    self.ema.update()

                    if divisible_by(self.step, self.save_and_sample_every):
                        milestone = self.step // self.save_and_sample_every

                        models = [(self.unwrapped_model, str(self.step))]

                        if self.use_ema:
                            models.append((self.ema.ema_model, f'{self.step}.ema'))

                        for model, label in models:
                            model.eval()

                            with torch.no_grad():
                                generated = model.sample(
                                    batch_size=self.num_samples,
                                    length=self.sample_length
                                )

                            for ind, t in enumerate(generated):
                                filename = str(self.results_folder / f'sample_{label}.flac')
                                t = rearrange(t, 'n -> 1 n')
                                torchaudio.save(filename, t.cpu().detach(), self.unwrapped_model.target_sample_hz)

                        self.print(f'{self.step}: saving to {str(self.results_folder)}')

                        self.save(milestone)

                pbar.update(1)

        self.print('training complete')
```