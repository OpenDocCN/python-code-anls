# `.\lucidrains\voicebox-pytorch\voicebox_pytorch\voicebox_pytorch.py`

```
import math
import logging
from random import random
from functools import partial
from pathlib import Path

import torch
from torch import nn, Tensor, einsum, IntTensor, FloatTensor, BoolTensor
from torch.nn import Module
import torch.nn.functional as F
from torch.cuda.amp import autocast

import torchode as to
from torchdiffeq import odeint

from beartype import beartype
from beartype.typing import Tuple, Optional, List, Union

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

from voicebox_pytorch.attend import Attend

from naturalspeech2_pytorch.aligner import Aligner, ForwardSumLoss, BinLoss, maximum_path
from naturalspeech2_pytorch.utils.tokenizer import Tokenizer
from naturalspeech2_pytorch.naturalspeech2_pytorch import generate_mask_from_repeats

from audiolm_pytorch import EncodecWrapper
from spear_tts_pytorch import TextToSemantic

from gateloop_transformer import SimpleGateLoopLayer as GateLoop

import torchaudio.transforms as T
from torchaudio.functional import DB_to_amplitude, resample

from vocos import Vocos

LOGGER = logging.getLogger(__file__)

# helper functions

# 检查值是否存在
def exists(val):
    return val is not None

# 返回输入值
def identity(t):
    return t

# 返回输入值或默认值
def default(val, d):
    return val if exists(val) else d

# 检查是否可以被整除
def divisible_by(num, den):
    return (num % den) == 0

# 检查是否为奇数
def is_odd(n):
    return not divisible_by(n, 2)

# 随机返回 True 或 False
def coin_flip():
    return random() < 0.5

# 将张量打包成指定模式
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包的张量解包成指定模式
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

# 根据概率生成掩码张量
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

# 将多个掩码张量按位与操作
def reduce_masks_with_and(*masks):
    masks = [*filter(exists, masks)]

    if len(masks) == 0:
        return None

    mask, *rest_masks = masks

    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask

# 对一维张量进行插值
def interpolate_1d(t, length, mode='bilinear'):
    " pytorch does not offer interpolation 1d, so hack by converting to 2d "

    dtype = t.dtype
    t = t.float()

    implicit_one_channel = t.ndim == 2
    if implicit_one_channel:
        t = rearrange(t, 'b n -> b 1 n')

    t = rearrange(t, 'b d n -> b d n 1')
    t = F.interpolate(t, (length, 1), mode=mode)
    t = rearrange(t, 'b d n 1 -> b d n')

    if implicit_one_channel:
        t = rearrange(t, 'b 1 n -> b n')

    t = t.to(dtype)
    return t

# 裁剪或填充张量至目标长度
def curtail_or_pad(t, target_length):
    length = t.shape[-2]

    if length > target_length:
        t = t[..., :target_length, :]
    elif length < target_length:
        t = F.pad(t, (0, 0, 0, target_length - length), value=0.)

    return t

# mask construction helpers

# 根据起始和结束索引生成掩码张量
def mask_from_start_end_indices(seq_len: int, start: Tensor, end: Tensor):
    assert start.shape == end.shape
    device = start.device

    seq = torch.arange(seq_len, device=device, dtype=torch.long)
    seq = seq.reshape(*((-1,) * start.ndim), seq_len)
    seq = seq.expand(*start.shape, seq_len)

    mask = seq >= start[..., None].long()
    mask &= seq < end[..., None].long()
    return mask

# 根据分数长度生成掩码张量
def mask_from_frac_lengths(seq_len: int, frac_lengths: Tensor):
    device = frac_lengths.device

    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.zeros_like(frac_lengths, device=device).float().uniform_(0, 1)
    start = (max_start * rand).clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)

# sinusoidal positions

# 用于 @crowsonkb 的学习正弦位置编码类
class LearnedSinusoidalPosEmb(Module):
    """ used by @crowsonkb """
    # 初始化函数，接受维度参数
    def __init__(self, dim):
        # 调用父类的初始化函数
        super().__init__()
        # 断言维度是2的倍数
        assert divisible_by(dim, 2)
        # 计算维度的一半
        half_dim = dim // 2
        # 初始化权重参数为服从标准正态分布的张量
        self.weights = nn.Parameter(torch.randn(half_dim))

    # 前向传播函数，接受输入张量 x
    def forward(self, x):
        # 重新排列输入张量 x 的维度，增加一个维度
        x = rearrange(x, 'b -> b 1')
        # 计算频率，乘以权重参数和 2π
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # 将正弦和余弦值拼接在一起，沿着最后一个维度
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # 返回傅立叶变换后的张量
        return fouriered
# 旋转位置嵌入
# https://arxiv.org/abs/2104.09864

class RotaryEmbedding(Module):
    def __init__(self, dim, theta = 50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @autocast(enabled = False)
    @beartype
    def forward(self, t: Union[int, Tensor]):
        if not torch.is_tensor(t):
            t = torch.arange(t, device = self.device)

        t = t.type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# 卷积位置生成模块

class ConvPositionEmbed(Module):
    def __init__(
        self,
        dim,
        *,
        kernel_size,
        groups = None
    ):
        super().__init__()
        assert is_odd(kernel_size)
        groups = default(groups, dim) # 默认情况下进行全深度卷积

        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups = groups, padding = kernel_size // 2),
            nn.GELU()
        )

    def forward(self, x, mask = None):

        if exists(mask):
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.)

        x = rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        out = rearrange(x, 'b c n -> b n c')

        if exists(mask):
            out = out.masked_fill(~mask, 0.)

        return out

# 规范化

class RMSNorm(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        cond_dim = None
    ):
        super().__init__()
        cond_dim = default(cond_dim, dim)
        self.scale = dim ** 0.5

        self.to_gamma = nn.Linear(cond_dim, dim)
        self.to_beta = nn.Linear(cond_dim, dim)

        # 初始化为单位矩阵

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, *, cond):
        normed = F.normalize(x, dim = -1) * self.scale

        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))

        return normed * gamma + beta

# 注意力

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0,
        flash = False,
        qk_norm = False,
        qk_norm_scale = 10
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads

        scale = qk_norm_scale if qk_norm else None

        self.attend = Attend(dropout, flash = flash, scale = scale)

        self.qk_norm = qk_norm

        if qk_norm:
            self.q_norm = MultiheadRMSNorm(dim_head, heads = heads)
            self.k_norm = MultiheadRMSNorm(dim_head, heads = heads)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)
    # 定义一个前向传播函数，接受输入张量 x，掩码 mask 和旋转嵌入 rotary_emb
    def forward(self, x, mask = None, rotary_emb = None):
        # 获取头数
        h = self.heads

        # 将输入张量 x 分别映射为查询 q，键 k，值 v
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # 将查询 q，键 k，值 v 重排维度，以适应多头注意力机制
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 如果启用了查询和键的归一化
        if self.qk_norm:
            # 对查询 q 进行归一化
            q = self.q_norm(q)
            # 对键 k 进行归一化
            k = self.k_norm(k)

        # 如果存在旋转嵌入
        if exists(rotary_emb):
            # 对查询 q 和键 k 应用旋转位置嵌入
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        # 进行注意力计算，得到输出 out
        out = self.attend(q, k, v, mask = mask)

        # 重排输出 out 的维度，以适应后续全连接层
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 将输出 out 传递给输出层，返回结果
        return self.to_out(out)
# 定义 GEGLU 类，用于实现 Gated GLU 激活函数
class GEGLU(Module):
    # 前向传播函数
    def forward(self, x):
        # 将输入张量 x 按照最后一个维度分成两部分，x 和 gate
        x, gate = x.chunk(2, dim = -1)
        # 对 gate 部分应用 GELU 激活函数，然后与 x 相乘
        return F.gelu(gate) * x

# 定义 FeedForward 函数，用于创建前馈神经网络层
def FeedForward(dim, mult = 4, dropout = 0.):
    # 计算内部维度
    dim_inner = int(dim * mult * 2 / 3)
    # 返回一个包含线性层、GEGLU 激活函数、Dropout 层和线性层的序列模块
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )

# 定义 Transformer 类，用于实现 Transformer 模型
class Transformer(Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        num_register_tokens = 0.,
        attn_flash = False,
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None,
        use_unet_skip_connection = False,
        skip_connect_scale = None,
        attn_qk_norm = False,
        use_gateloop_layers = False,
        gateloop_use_jax = False,
    ):
        super().__init__()
        # 断言深度是偶数
        assert divisible_by(depth, 2)
        # 初始化层列表
        self.layers = nn.ModuleList([])

        # 创建旋转嵌入层
        self.rotary_emb = RotaryEmbedding(dim = dim_head)

        # 设置注册令牌数量
        self.num_register_tokens = num_register_tokens
        self.has_register_tokens = num_register_tokens > 0

        # 如果存在注册令牌，则创建注册令牌参数
        if self.has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        # 根据是否自适应 RMSNorm 选择不同的 RMSNorm 类
        if adaptive_rmsnorm:
            rmsnorm_klass = partial(AdaptiveRMSNorm, cond_dim = adaptive_rmsnorm_cond_dim_in)
        else:
            rmsnorm_klass = RMSNorm

        # 设置跳跃连接的缩放因子
        self.skip_connect_scale = default(skip_connect_scale, 2 ** -0.5)

        # 循环创建 Transformer 层
        for ind in range(depth):
            layer = ind + 1
            has_skip = use_unet_skip_connection and layer > (depth // 2)

            self.layers.append(nn.ModuleList([
                nn.Linear(dim * 2, dim) if has_skip else None,
                GateLoop(dim = dim, use_jax_associative_scan = gateloop_use_jax, post_ln = True) if use_gateloop_layers else None,
                rmsnorm_klass(dim = dim),
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash, qk_norm = attn_qk_norm),
                rmsnorm_klass(dim = dim),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        # 创建最终的 RMSNorm 层
        self.final_norm = RMSNorm(dim)

    # 获取设备信息
    @property
    def device(self):
        return next(self.parameters()).device

    # 前向传播函数
    def forward(
        self,
        x,
        mask = None,
        adaptive_rmsnorm_cond = None
        ):
            # 获取输入张量的批量大小、序列长度等信息
            batch, seq_len, *_ = x.shape

            # 在左侧添加注册令牌

            if self.has_register_tokens:
                # 重复注册令牌以匹配批量大小
                register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)

                # 将注册令牌和输入张量打包
                x, ps = pack([register_tokens, x], 'b * d')

                # 如果存在掩码，则在左侧填充
                if exists(mask):
                    mask = F.pad(mask, (self.num_register_tokens, 0), value = True)

            # 跟踪跳跃连接

            skip_connects = []

            # 旋转嵌入

            positions = seq_len

            if self.has_register_tokens:
                # 创建主要位置和注册位置
                main_positions = torch.arange(seq_len, device = self.device, dtype = torch.long)
                register_positions = torch.full((self.num_register_tokens,), -10000, device = self.device, dtype = torch.long)
                positions = torch.cat((register_positions, main_positions))

            # 计算旋转嵌入
            rotary_emb = self.rotary_emb(positions)

            # 自适应 RMSNorm

            rmsnorm_kwargs = dict()
            if exists(adaptive_rmsnorm_cond):
                rmsnorm_kwargs = dict(cond = adaptive_rmsnorm_cond)

            # 通过注意力层

            for skip_combiner, maybe_gateloop, attn_prenorm, attn, ff_prenorm, ff in self.layers:

                # 在论文中，他们使用类似 U-Net 的跳跃连接
                # 不清楚这有多大帮助，因为除了简短的一两句提到外，没有给出任何消融或进一步的数字

                if not exists(skip_combiner):
                    skip_connects.append(x)
                else:
                    skip_connect = skip_connects.pop() * self.skip_connect_scale
                    x = torch.cat((x, skip_connect), dim = -1)
                    x = skip_combiner(x)

                if exists(maybe_gateloop):
                    x = maybe_gateloop(x) + x

                # 计算注意力输入
                attn_input = attn_prenorm(x, **rmsnorm_kwargs)
                x = attn(attn_input, mask = mask, rotary_emb = rotary_emb) + x

                # 计算前馈神经网络输入
                ff_input = ff_prenorm(x, **rmsnorm_kwargs) 
                x = ff(ff_input) + x

            # 移除注册令牌

            if self.has_register_tokens:
                _, x = unpack(x, ps, 'b * d')

            # 返回最终规范化结果
            return self.final_norm(x)
# 定义音频编码器解码器的基类
class AudioEncoderDecoder(nn.Module):
    pass

# 定义 MelVoco 类，继承自 AudioEncoderDecoder
class MelVoco(AudioEncoderDecoder):
    def __init__(
        self,
        *,
        log = True,
        n_mels = 100,
        sampling_rate = 24000,
        f_max = 8000,
        n_fft = 1024,
        win_length = 640,
        hop_length = 160,
        pretrained_vocos_path = 'charactr/vocos-mel-24khz'
    ):
        super().__init__()
        self.log = log
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_max = f_max
        self.win_length = win_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate

        # 加载预训练的 Vocos 模型
        self.vocos = Vocos.from_pretrained(pretrained_vocos_path)

    @property
    def downsample_factor(self):
        raise NotImplementedError

    @property
    def latent_dim(self):
        return self.num_mels

    # 对音频进行编码
    def encode(self, audio):
        # 对音频进行短时傅里叶变换
        stft_transform = T.Spectrogram(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            window_fn = torch.hann_window
        )

        spectrogram = stft_transform(audio)

        # 对频谱图进行梅尔频谱变换
        mel_transform = T.MelScale(
            n_mels = self.n_mels,
            sample_rate = self.sampling_rate,
            n_stft = self.n_fft // 2 + 1,
            f_max = self.f_max
        )

        mel = mel_transform(spectrogram)

        # 如果需要对梅尔频谱进行对数变换
        if self.log:
            mel = T.AmplitudeToDB()(mel)

        mel = rearrange(mel, 'b d n -> b n d')
        return mel

    # 对梅尔频谱进行解码
    def decode(self, mel):
        mel = rearrange(mel, 'b n d -> b d n')

        # 如果需要对梅尔频谱进行反对数变换
        if self.log:
            mel = DB_to_amplitude(mel, ref = 1., power = 0.5)

        return self.vocos.decode(mel)

# 定义 EncodecVoco 类，继承自 AudioEncoderDecoder
class EncodecVoco(AudioEncoderDecoder):
    def __init__(
        self,
        *,
        sampling_rate = 24000,
        pretrained_vocos_path = 'charactr/vocos-encodec-24khz',
        bandwidth_id = 2
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.encodec = EncodecWrapper()
        # 加载预训练的 Vocos 模型
        self.vocos = Vocos.from_pretrained(pretrained_vocos_path)

        # 注册缓冲区，存储带宽 ID
        self.register_buffer('bandwidth_id', torch.tensor([bandwidth_id]))

    @property
    def downsample_factor(self):
        return self.encodec.downsample_factor

    @property
    def latent_dim(self):
        return self.encodec.codebook_dim

    # 对音频进行编码
    def encode(self, audio):
        encoded_audio, _, _ = self.encodec(audio, return_encoded = True)
        return encoded_audio

    # 解码为编码
    def decode_to_codes(self, latents):
        _, codes, _ = self.encodec.rq(latents)
        codes = rearrange(codes, 'b n q -> b q n')
        return codes

    # 解码编码为音频
    def decode(self, latents):
        codes = self.decode_to_codes(latents)

        all_audios = []
        for code in codes:
            features = self.vocos.codes_to_features(code)
            audio = self.vocos.decode(features, bandwidth_id = self.bandwidth_id)
            all_audios.append(audio)

        return torch.stack(all_audios)

# 定义 DurationPredictor 类，继承自 Module
class DurationPredictor(Module):
    @beartype
    def __init__(
        self,
        *,
        audio_enc_dec: Optional[AudioEncoderDecoder] = None,
        tokenizer: Optional[Tokenizer] = None,
        num_phoneme_tokens: Optional[int] = None,
        dim_phoneme_emb = 512,
        dim = 512,
        depth = 10,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        ff_dropout = 0.,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
        attn_dropout = 0,
        attn_flash = False,
        attn_qk_norm = True,
        use_gateloop_layers = False,
        p_drop_prob = 0.2, # p_drop in paper
        frac_lengths_mask: Tuple[float, float] = (0.1, 1.),
        aligner_kwargs: dict = dict(dim_in = 80, attn_channels = 80)
    ):
        # 调用父类的构造函数
        super().__init__()

        # 音频编码器/解码器
        self.audio_enc_dec = audio_enc_dec

        # 如果音频编码器/解码器存在且维度不等于音频编码器/解码器的潜在维度，则创建输入投影层
        if exists(audio_enc_dec) and dim != audio_enc_dec.latent_dim:
            self.proj_in = nn.Linear(audio_enc_dec.latent_dim, dim)
        else:
            self.proj_in = nn.Identity()

        # 与音素相关

        # 如果传入了音素标记器和音素标记数，则抛出断言错误
        assert not (exists(tokenizer) and exists(num_phoneme_tokens)), 'if a phoneme tokenizer was passed into duration module, number of phoneme tokens does not need to be specified'

        # 如果音素标记器和音素标记数都不存在，则默认使用英语音素和 espeak 创建标记器
        if not exists(tokenizer) and not exists(num_phoneme_tokens):
            tokenizer = Tokenizer()

        # 如果存在音素标记器，则设置音素标记数为标记器的词汇量大小
        if exists(tokenizer):
            num_phoneme_tokens = tokenizer.vocab_size

        self.tokenizer = tokenizer

        # 创建音素嵌入层
        self.to_phoneme_emb = nn.Embedding(num_phoneme_tokens, dim_phoneme_emb)

        self.p_drop_prob = p_drop_prob
        self.frac_lengths_mask = frac_lengths_mask

        # 创建线性层，用于将音频编码器/解码器输出和音素嵌入层输出连接起来
        self.to_embed = nn.Linear(dim + dim_phoneme_emb, dim)

        # 创建空条件参数
        self.null_cond = nn.Parameter(torch.zeros(dim), requires_grad = False)

        # 创建卷积位置嵌入层
        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size,
            groups = conv_pos_embed_groups
        )

        # 创建 Transformer 模型
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout,
            attn_dropout=attn_dropout,
            attn_flash = attn_flash,
            attn_qk_norm = attn_qk_norm,
            use_gateloop_layers = use_gateloop_layers
        )

        # 创建预测层
        self.to_pred = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...')
        )

        # 对齐器相关

        # 如果使用具有 80 个通道的 mel 频谱，则将 attn_channels 设置为 80
        # 假设输入维度为具有 80 个通道的 spec
        self.aligner = Aligner(dim_hidden = dim_phoneme_emb, **aligner_kwargs)
        self.align_loss = ForwardSumLoss()

    @property
    def device(self):
        # 返回模型参数所在的设备
        return next(self.parameters()).device

    def align_phoneme_ids_with_durations(self, phoneme_ids, durations):
        # 生成重复掩码
        repeat_mask = generate_mask_from_repeats(durations.clamp(min = 1))
        # 将音素标记与持续时间对齐
        aligned_phoneme_ids = einsum('b i, b i j -> b j', phoneme_ids.float(), repeat_mask.float()).long()
        return aligned_phoneme_ids

    @torch.inference_mode()
    @beartype
    def forward_with_cond_scale(
        self,
        *args,
        texts: Optional[List[str]] = None,
        phoneme_ids = None,
        cond_scale = 1.,
        return_aligned_phoneme_ids = False,
        **kwargs
    ):
        if exists(texts):
            phoneme_ids = self.tokenizer.texts_to_tensor_ids(texts)

        forward_kwargs = dict(
            return_aligned_phoneme_ids = False,
            phoneme_ids = phoneme_ids
        )

        durations = self.forward(*args, cond_drop_prob = 0., **forward_kwargs, **kwargs)

        if cond_scale == 1.:
            if not return_aligned_phoneme_ids:
                return durations

            return durations, self.align_phoneme_ids_with_durations(phoneme_ids, durations)

        null_durations = self.forward(*args, cond_drop_prob = 1., **forward_kwargs, **kwargs)
        scaled_durations = null_durations + (durations - null_durations) * cond_scale

        if not return_aligned_phoneme_ids:
            return scaled_durations

        return scaled_durations, self.align_phoneme_ids_with_durations(phoneme_ids, scaled_durations)

    @beartype
    def forward_aligner(
        self,
        x: FloatTensor,     # (b, t, c)
        x_mask: IntTensor,  # (b, 1, t)
        y: FloatTensor,     # (b, t, c)
        y_mask: IntTensor   # (b, 1, t)
    # 定义函数的返回类型为元组，包含四个张量
    ) -> Tuple[
        FloatTensor,        # alignment_hard: (b, t)
        FloatTensor,        # alignment_soft: (b, tx, ty)
        FloatTensor,        # alignment_logprob: (b, 1, ty, tx)
        BoolTensor          # alignment_mas: (b, tx, ty)
    ]:
        # 创建注意力掩码，用于限制注意力的计算范围
        attn_mask = rearrange(x_mask, 'b 1 t -> b 1 t 1') * rearrange(y_mask, 'b 1 t -> b 1 1 t')
        # 调用aligner模型计算软对齐和对数概率
        alignment_soft, alignment_logprob = self.aligner(rearrange(y, 'b t c -> b c t'), x, x_mask)

        # 断言软对齐张量中不包含NaN值
        assert not torch.isnan(alignment_soft).any()

        # 使用最大路径算法计算最佳对齐路径
        alignment_mas = maximum_path(
            rearrange(alignment_soft, 'b 1 t1 t2 -> b t2 t1').contiguous(),
            rearrange(attn_mask, 'b 1 t1 t2 -> b t1 t2').contiguous()
        )

        # 计算硬对齐张量
        alignment_hard = torch.sum(alignment_mas, -1).float()
        # 重新排列软对齐张量的维度
        alignment_soft = rearrange(alignment_soft, 'b 1 t1 t2 -> b t2 t1')
        # 返回硬对齐、软对齐、对数概率和对齐掩码
        return alignment_hard, alignment_soft, alignment_logprob, alignment_mas

    # 定义前向传播函数，接受多个参数
    @beartype
    def forward(
        self,
        *,
        cond,
        texts: Optional[List[str]] = None,
        phoneme_ids = None,
        cond_drop_prob = 0.,
        target = None,
        cond_mask = None,
        mel = None,
        phoneme_len = None,
        mel_len = None,
        phoneme_mask = None,
        mel_mask = None,
        self_attn_mask = None,
        return_aligned_phoneme_ids = False
    ):
        # 获取输入的 batch 大小、序列长度和条件维度
        batch, seq_len, cond_dim = cond.shape

        # 对条件进行投影
        cond = self.proj_in(cond)

        # 如果未提供音素 id，则使用分词器将文本转换为音素 id
        if not exists(phoneme_ids):
            assert exists(self.tokenizer)
            phoneme_ids = self.tokenizer.texts_to_tensor_ids(texts)

        # 如果未提供条件掩码，则根据条件生成掩码
        if not exists(cond_mask):
            if coin_flip():
                frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
                cond_mask = mask_from_frac_lengths(seq_len, frac_lengths)
            else:
                cond_mask = prob_mask_like((batch, seq_len), self.p_drop_prob, self.device)

        # 根据条件掩码对条件进行掩码处理
        cond = cond * rearrange(~cond_mask, '... -> ... 1')

        # 如果条件丢弃概率大于 0，则对条件进行丢弃处理
        if cond_drop_prob > 0.:
            cond_drop_mask = prob_mask_like(cond.shape[:1], cond_drop_prob, cond.device)

            cond = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1 1'),
                self.null_cond,
                cond
            )

        # 音素 id 为 -1 表示填充
        if not exists(self_attn_mask):
            self_attn_mask = phoneme_ids != -1

        # 将音素 id 限制在大于等于 0 的范围内
        phoneme_ids = phoneme_ids.clamp(min=0)

        # 获取音素嵌入
        phoneme_emb = self.to_phoneme_emb(phoneme_ids)

        # 强制条件与输入音素具有相同的长度
        cond = curtail_or_pad(cond, phoneme_ids.shape[-1])

        # 合并音素嵌入、条件
        embed = torch.cat((phoneme_emb, cond), dim=-1)
        x = self.to_embed(embed)

        # 进行卷积嵌入
        x = self.conv_embed(x, mask=self_attn_mask) + x

        # 进行 transformer 操作
        x = self.transformer(
            x,
            mask=self_attn_mask
        )

        # 预测持续时间
        durations = self.to_pred(x)

        # 如果不是训练阶段，则返回持续时间
        if not self.training:
            if not return_aligned_phoneme_ids:
                return durations

            return durations, self.align_phoneme_ids_with_durations(phoneme_ids, durations)

        # 对齐器
        # 使用 alignment_hard 过采样音素
        # Duration Predictor 应该预测未掩码音素的持续时间，其中目标是掩码对齐硬
        assert all([exists(el) for el in (phoneme_len, mel_len, phoneme_mask, mel_mask)], '需要传递 phoneme_len���mel_len、phoneme_mask、mel_mask 给训练持续时间预测模块')

        alignment_hard, _, alignment_logprob, _ = self.forward_aligner(phoneme_emb, phoneme_mask, mel, mel_mask)
        target = alignment_hard

        if exists(self_attn_mask):
            loss_mask = cond_mask & self_attn_mask
        else:
            loss_mask = self_attn_mask

        if not exists(loss_mask):
            return F.l1_loss(x, target)

        loss = F.l1_loss(x, target, reduction='none')
        loss = loss.masked_fill(~loss_mask, 0.)

        # 掩码平均值
        num = reduce(loss, 'b n -> b', 'sum')
        den = loss_mask.sum(dim=-1).clamp(min=1e-5)
        loss = num / den
        loss = loss.mean()
        
        if not return_aligned_phoneme_ids:
            return loss

        # 对齐器损失
        align_loss = self.align_loss(alignment_logprob, phoneme_len, mel_len)
        loss = loss + align_loss

        return loss
# VoiceBox 类，继承自 Module 类
class VoiceBox(Module):
    # 初始化方法
    def __init__(
        self,
        *,
        num_cond_tokens = None, # 条件标记数量，默认为 None
        audio_enc_dec: Optional[AudioEncoderDecoder] = None, # 音频编码器解码器，默认为 None
        dim_in = None, # 输入维度，默认为 None
        dim_cond_emb = 1024, # 条件嵌入维度，默认为 1024
        dim = 1024, # 维度，默认为 1024
        depth = 24, # 深度，默认为 24
        dim_head = 64, # 头维度，默认为 64
        heads = 16, # 头数，默认为 16
        ff_mult = 4, # FeedForward 层倍数，默认为 4
        ff_dropout = 0., # FeedForward 层的 dropout，默认为 0
        time_hidden_dim = None, # 时间隐藏维度，默认为 None
        conv_pos_embed_kernel_size = 31, # 卷积位置嵌入的卷积核大小，默认为 31
        conv_pos_embed_groups = None, # 卷积位置嵌入的分组，默认为 None
        attn_dropout = 0, # 注意力 dropout，默认为 0
        attn_flash = False, # 是否使用 Flash 注意力，默认为 False
        attn_qk_norm = True, # 注意力的 QK 归一化，默认为 True
        use_gateloop_layers = False, # 是否使用 Gateloop 层，默认为 False
        num_register_tokens = 16, # 寄存器标记数量，默认为 16
        p_drop_prob = 0.3, # p_drop 在论文中的概率，默认为 0.3
        frac_lengths_mask: Tuple[float, float] = (0.7, 1.), # 长度掩码的分数，默认为 (0.7, 1)
        condition_on_text = True # 是否基于文本条件，默认为 True
    ):
        super().__init__() # 调用父类的初始化方法
        dim_in = default(dim_in, dim) # 如果输入维度为 None，则使用默认维度

        time_hidden_dim = default(time_hidden_dim, dim * 4) # 如果时间隐藏维度为 None，则使用默认维度

        self.audio_enc_dec = audio_enc_dec # 设置音频编码器解码器

        if exists(audio_enc_dec) and dim != audio_enc_dec.latent_dim: # 如果音频编码器解码器存在且维度不等于潜在维度
            self.proj_in = nn.Linear(audio_enc_dec.latent_dim, dim) # 使用线性层进行投影
        else:
            self.proj_in = nn.Identity() # 否则使用恒等映射

        # 正弦位置嵌入
        self.sinu_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim), # 学习的正弦位置嵌入
            nn.Linear(dim, time_hidden_dim), # 线性层
            nn.SiLU() # SiLU 激活函数
        )

        assert not (condition_on_text and not exists(num_cond_tokens)), 'number of conditioning tokens must be specified (whether phonemes or semantic token ids) if training conditional voicebox'

        if not condition_on_text: # 如果不基于文本条件
            dim_cond_emb = 0 # 条件嵌入维度为 0

        self.dim_cond_emb = dim_cond_emb # 设置条件嵌入维度
        self.condition_on_text = condition_on_text # 设置是否基于文本条件
        self.num_cond_tokens = num_cond_tokens # 设置条件标记数量

        if condition_on_text: # 如果基于文本条件
            self.null_cond_id = num_cond_tokens # 使用最后一个音素标记作为 CFG 的空标记
            self.to_cond_emb = nn.Embedding(num_cond_tokens + 1, dim_cond_emb) # 条件嵌入层

        self.p_drop_prob = p_drop_prob # 设置 p_drop 概率
        self.frac_lengths_mask = frac_lengths_mask # 设置长度掩码

        self.to_embed = nn.Linear(dim_in * 2 + dim_cond_emb, dim) # 输入到嵌入的线性层

        self.null_cond = nn.Parameter(torch.zeros(dim_in), requires_grad = False) # 空条件参数

        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size,
            groups = conv_pos_embed_groups
        ) # 卷积位置嵌入层

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout,
            attn_dropout= attn_dropout,
            attn_flash = attn_flash,
            attn_qk_norm = attn_qk_norm,
            num_register_tokens = num_register_tokens,
            adaptive_rmsnorm = True,
            adaptive_rmsnorm_cond_dim_in = time_hidden_dim,
            use_gateloop_layers = use_gateloop_layers
        ) # Transformer 模型

        dim_out = audio_enc_dec.latent_dim if exists(audio_enc_dec) else dim_in # 输出维度

        self.to_pred = nn.Linear(dim, dim_out, bias = False) # 预测线性层

    @property
    def device(self):
        return next(self.parameters()).device # 返回参数的设备

    @torch.inference_mode()
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs) # 前向传播计算 logits

        if cond_scale == 1.: # 如果条件缩放为 1
            return logits # 返回 logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs) # 使用条件概率为 1 计算 logits
        return null_logits + (logits - null_logits) * cond_scale # 返回缩放后的结果

    def forward(
        self,
        x,
        *,
        times,
        cond_token_ids,
        self_attn_mask = None,
        cond_drop_prob = 0.1,
        target = None,
        cond = None,
        cond_mask = None
        ):
            # 项目输入，以防代码簿维度不等于模型维度

            x = self.proj_in(x)

            cond = default(cond, target)

            if exists(cond):
                cond = self.proj_in(cond)

            # 获取形状信息

            batch, seq_len, cond_dim = cond.shape
            assert cond_dim == x.shape[-1]

            # 自动管理时间维度的形状，用于odeint times

            if times.ndim == 0:
                times = repeat(times, '-> b', b = cond.shape[0])

            if times.ndim == 1 and times.shape[0] == 1:
                times = repeat(times, '1 -> b', b = cond.shape[0])

            # 如果未提供条件掩码，则构建条件掩码

            if self.training:
                if not exists(cond_mask):
                    frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(*self.frac_lengths_mask)
                    cond_mask = mask_from_frac_lengths(seq_len, frac_lengths)
            else:
                if not exists(cond_mask):
                    cond_mask = torch.ones((batch, seq_len), device = cond.device, dtype = torch.bool)

            cond_mask_with_pad_dim = rearrange(cond_mask, '... -> ... 1')

            # 如第3.2节所述

            cond = cond * ~cond_mask_with_pad_dim

            # 无分类器指导

            cond_ids = cond_token_ids

            if cond_drop_prob > 0.:
                cond_drop_mask = prob_mask_like(cond.shape[:1], cond_drop_prob, self.device)

                cond = torch.where(
                    rearrange(cond_drop_mask, '... -> ... 1 1'),
                    self.null_cond,
                    cond
                )

                cond_ids = torch.where(
                    rearrange(cond_drop_mask, '... -> ... 1'),
                    self.null_cond_id,
                    cond_token_ids
                )

            # 音素或语义条件嵌入

            cond_emb = None

            if self.condition_on_text:
                cond_emb = self.to_cond_emb(cond_ids)

                cond_emb_length = cond_emb.shape[-2]
                if cond_emb_length != seq_len:
                    cond_emb = rearrange(cond_emb, 'b n d -> b d n')
                    cond_emb = interpolate_1d(cond_emb, seq_len)
                    cond_emb = rearrange(cond_emb, 'b d n -> b n d')

                    if exists(self_attn_mask):
                        self_attn_mask = interpolate_1d(self_attn_mask, seq_len)

            # 连接源信号、语义/音素条件嵌入和条件，并进行投影

            to_concat = [*filter(exists, (x, cond_emb, cond))]
            embed = torch.cat(to_concat, dim = -1)

            x = self.to_embed(embed)

            x = self.conv_embed(x, mask = self_attn_mask) + x

            time_emb = self.sinu_pos_emb(times)

            # 注意力

            x = self.transformer(
                x,
                mask = self_attn_mask,
                adaptive_rmsnorm_cond = time_emb
            )

            x = self.to_pred(x)

            # 如果未传入目标，则只返回对数

            if not exists(target):
                return x

            loss_mask = reduce_masks_with_and(cond_mask, self_attn_mask)

            if not exists(loss_mask):
                return F.mse_loss(x, target)

            loss = F.mse_loss(x, target, reduction = 'none')

            loss = reduce(loss, 'b n d -> b n', 'mean')
            loss = loss.masked_fill(~loss_mask, 0.)

            # 掩码均值

            num = reduce(loss, 'b n -> b', 'sum')
            den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
            loss = num / den

            return loss.mean()
# 对 CNF 的包装器

# 判断输入是否可能是音频数据，根据其形状来判断
def is_probably_audio_from_shape(t):
    return exists(t) and (t.ndim == 2 or (t.ndim == 3 and t.shape[1] == 1))

# 条件流匹配器的包装器类
class ConditionalFlowMatcherWrapper(Module):
    # 初始化方法
    @beartype
    def __init__(
        self,
        voicebox: VoiceBox,
        text_to_semantic: Optional[TextToSemantic] = None,
        duration_predictor: Optional[DurationPredictor] = None,
        sigma = 0.,
        ode_atol = 1e-5,
        ode_rtol = 1e-5,
        use_torchode = False,
        torchdiffeq_ode_method = 'midpoint',   # 使用中点法作为 torchdiffeq 的方法，与论文中一致
        torchode_method_klass = to.Tsit5,      # 使用 tsit5 作为 torchode 的方法，因为 torchode 没有中点法（由 Bryan @b-chiang 推荐）
        cond_drop_prob = 0.
    ):
        super().__init__()
        self.sigma = sigma

        self.voicebox = voicebox
        self.condition_on_text = voicebox.condition_on_text

        # 断言条件，确保不在不条件下使用 TextToSemantic
        assert not (not self.condition_on_text and exists(text_to_semantic)), 'TextToSemantic should not be passed in if not conditioning on text'
        # 断言条件，确保在使用 TextToSemantic 时存在 wav2vec 模块
        assert not (exists(text_to_semantic) and not exists(text_to_semantic.wav2vec)), 'the wav2vec module must exist on the TextToSemantic, if being used to condition on text'

        self.text_to_semantic = text_to_semantic
        self.duration_predictor = duration_predictor

        # 断言条件，确保在条件下使用 TextToSemantic 或 DurationPredictor
        if self.condition_on_text and (exists(text_to_semantic) or exists(duration_predictor)):
            assert exists(text_to_semantic) ^ exists(duration_predictor), 'you should use either TextToSemantic from Spear-TTS, or DurationPredictor for the text / phoneme to audio alignment, but not both'

        self.cond_drop_prob = cond_drop_prob

        self.use_torchode = use_torchode
        self.torchode_method_klass = torchode_method_klass

        self.odeint_kwargs = dict(
            atol = ode_atol,
            rtol = ode_rtol,
            method = torchdiffeq_ode_method
        )

    # 获取设备信息
    @property
    def device(self):
        return next(self.parameters()).device

    # 加载模型
    def load(self, path, strict = True):
        # 返回 pkg 以便训练器可以访问
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(pkg['model'], strict = strict)
        return pkg

    # 采样方法
    @torch.inference_mode()
    def sample(
        self,
        *,
        cond = None,
        texts: Optional[List[str]] = None,
        text_token_ids: Optional[Tensor] = None,
        semantic_token_ids: Optional[Tensor] = None,
        phoneme_ids: Optional[Tensor] = None,
        cond_mask = None,
        steps = 3,
        cond_scale = 1.,
        decode_to_audio = True,
        decode_to_codes = False,
        max_semantic_token_ids = 2048,
        spec_decode = False,
        spec_decode_gamma = 5 # 可能需要更高，因为语音可能比文本更容易，需要测试
    # 前向传播方法
    def forward(
        self,
        x1,
        *,
        mask = None,
        semantic_token_ids = None,
        phoneme_ids = None,
        cond = None,
        cond_mask = None,
        input_sampling_rate = None # 如果未给出，则假定与音频编码器解码器采样率相同，如果给出，则重新采样
        ):
        """
        following eq (5) (6) in https://arxiv.org/pdf/2306.15687.pdf
        """

        # 获取输入张量 x1 的批量大小、序列长度、数据类型和标准差
        batch, seq_len, dtype, σ = *x1.shape[:2], x1.dtype, self.sigma

        # 如果输入是原始音频，则转换为音频编码器/解码器传入的格式
        input_is_raw_audio, cond_is_raw_audio = map(is_probably_audio_from_shape, (x1, cond))

        if input_is_raw_audio:
            raw_audio = x1

        if any([input_is_raw_audio, cond_is_raw_audio]):
            assert exists(self.voicebox.audio_enc_dec), 'audio_enc_dec must be set on VoiceBox to train directly on raw audio'

            audio_enc_dec_sampling_rate = self.voicebox.audio_enc_dec.sampling_rate
            input_sampling_rate = default(input_sampling_rate, audio_enc_dec_sampling_rate)

            with torch.no_grad():
                self.voicebox.audio_enc_dec.eval()

                if input_is_raw_audio:
                    x1 = resample(x1, input_sampling_rate, audio_enc_dec_sampling_rate)
                    x1 = self.voicebox.audio_enc_dec.encode(x1)

                if exists(cond) and cond_is_raw_audio:
                    cond = resample(cond, input_sampling_rate, audio_enc_dec_sampling_rate)
                    cond = self.voicebox.audio_enc_dec.encode(cond)

        # 设置文本条件，可以来自持续时间模型（作为音素 id）或来自文本到语义模块，使用 wav2vec 编码的语义 id（通常是 hubert）

        assert self.condition_on_text or not (exists(semantic_token_ids) or exists(phoneme_ids)), 'semantic or phoneme ids should not be passed in if not conditioning on text'

        cond_token_ids = None

        if self.condition_on_text:
            if exists(self.text_to_semantic) or exists(semantic_token_ids):
                assert not exists(phoneme_ids), 'phoneme ids are not needed for conditioning with spear-tts text-to-semantic'

                if not exists(semantic_token_ids):
                    assert input_is_raw_audio
                    wav2vec = self.text_to_semantic.wav2vec
                    wav2vec_input = resample(raw_audio, input_sampling_rate, wav2vec.target_sample_hz)
                    semantic_token_ids = wav2vec(wav2vec_input).clone()

                cond_token_ids = semantic_token_ids
            else:
                assert exists(phoneme_ids)
                cond_token_ids = phoneme_ids

        # 主要的条件流程逻辑在下面

        # x0 是高斯噪声

        x0 = torch.randn_like(x1)

        # 随机时间

        times = torch.rand((batch,), dtype=dtype, device=self.device)
        t = rearrange(times, 'b -> b 1 1')

        # 采样 xt（论文中的 w）

        w = (1 - (1 - σ) * t) * x0 + t * x1

        flow = x1 - (1 - σ) * x0

        # 预测

        self.voicebox.train()

        loss = self.voicebox(
            w,
            cond=cond,
            cond_mask=cond_mask,
            times=times,
            target=flow,
            self_attn_mask=mask,
            cond_token_ids=cond_token_ids,
            cond_drop_prob=self.cond_drop_prob
        )

        return loss
```