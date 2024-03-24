# `.\lucidrains\nuwa-pytorch\nuwa_pytorch\nuwa_pytorch.py`

```py
# 导入 functools 模块
import functools
# 从 functools 模块中导入 partial 函数
from functools import partial

# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn、einsum 模块
from torch import nn, einsum
# 从 torch 模块中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F

# 导入 einops 模块中的 rearrange、reduce、repeat 函数
from einops import rearrange, reduce, repeat
# 从 einops.layers.torch 模块中导入 Rearrange、Reduce 类
from einops.layers.torch import Rearrange, Reduce

# 导入 nuwa_pytorch 模块中的 ReversibleSequence、DualModalityReversibleSequence 类
from nuwa_pytorch.reversible import ReversibleSequence
from nuwa_pytorch.reversible_video_audio import DualModalityReversibleSequence

# 导入 unfoldNd 模块
from unfoldNd import unfoldNd

# 导入 tqdm 模块
from tqdm import tqdm

# 常量定义

# 定义 MList 为 nn.ModuleList 类
MList = nn.ModuleList

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 将变量转换为元组的函数
def cast_tuple(val, size = 1):
    return val if isinstance(val, tuple) else (val,) * size

# 计算相同填充的函数
def calc_same_padding(kernel_size, dilation = 1):
    return dilation * (kernel_size - 1) // 2

# 将填充值调整为倍数的函数
def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# 装饰器

# 评估装饰器函数
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 张量辅助函数

# 对数函数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# sigmoid 函数
def sigmoid(t):
    return torch.where(t >= 0, 1 / (1 + torch.exp(-t)), t.exp() / (1 + t.exp()))

# 生成 Gumbel 噪声的函数
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 生成 Gumbel 采样的函数
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

# 安全除法函数
def safe_div(numer, denom, eps = 1e-6):
    return numer / (denom + eps)

# 生成概率掩码的函数
def prob_mask_like(shape, prob, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# 批处理函数
def batch_process(t, fn, chunks = 10, dim = 0):
    chunks = [fn(t_chunk) for t_chunk in t.chunk(chunks, dim = dim)]
    return torch.cat(chunks, dim = dim)

# 多重归约函数
def mult_reduce(arr):
    return functools.reduce(lambda x, y: x * y, arr, 1)

# 梯度控制

# 分数梯度函数
def frac_gradient(t, frac):
    return t * frac + t.detach() * (1 - frac)

# 标准化

# 稳定的 LayerNorm 类
class StableLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x / x.amax(dim = -1, keepdim = True).detach()
        return self.norm(x)

# 预标准化类
class PreNorm(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fn
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 三明治标准化类
class SandwichNorm(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fn
    ):
        super().__init__()
        self.prenorm = nn.LayerNorm(dim)
        self.postnorm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.prenorm(x)
        x = self.fn(x, **kwargs)
        x = self.postnorm(x)
        return x

# 相对位置嵌入（旋转）

# 旋转嵌入类
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device = device).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

# 旋转半个维度的函数
def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

# 应用旋转位置嵌入的函数
def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

# 辅助类

# 移位音频令牌类
class ShiftAudioTokens(nn.Module):
    # 初始化函数，设置音频每个时间步的音频标记数
    def __init__(
        self,
        fn,
        audio_tokens_per_timestep = 1
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 保存文件名和音频每个时间步的音频标记数
        self.fn = fn
        self.audio_tokens_per_timestep = audio_tokens_per_timestep

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 获取输入张量的第二个维度大小
        n = x.shape[1]

        # 填充到最近的时间步

        # 计算需要填充的数量
        padding = self.audio_tokens_per_timestep - (n % self.audio_tokens_per_timestep)
        # 在第二维度上进行填充
        x = F.pad(x, (0, 0, 0, padding), value = 0.)

        # 沿着时间轴进行移动

        # 将输入张量分成两部分
        x_shift, x = x.chunk(2, dim = -1)
        # 在第二维度上进行填充
        x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
        # 拼接两部分张量
        x = torch.cat((x_shift, x), dim = -1)

        # 如果需要，移除填充

        # 返回处理后的结果
        return self.fn(x[:, :n], **kwargs)
class ShiftVideoTokens(nn.Module):
    # 定义 ShiftVideoTokens 类，用于处理视频序列的移位操作
    def __init__(
        self,
        fn,
        image_size,
        shift_space = True,
        shift_time = False
    ):
        # 初始化函数，接收函数 fn、图像大小 image_size、是否移位空间 shift_space、是否移位时间 shift_time 作为参数
        super().__init__()
        self.fn = fn
        self.image_size = image_size

        self.shift_time = shift_time
        self.shift_space = shift_space

    def forward(self, x, **kwargs):
        # 前向传播函数，接收输入 x 和其他关键字参数 kwargs
        if not self.shift_time and not self.shift_space:
            return self.fn(x, **kwargs)

        image_size = self.image_size
        img_seq_len = image_size ** 2

        x_bos, x_video = x[:, :1], x[:, 1:]
        n = x_video.shape[1]

        # pad to nearest frame
        # 填充到最近的帧

        padding = img_seq_len - (n % img_seq_len)
        x_video = F.pad(x_video, (0, 0, 0, padding), value = 0.)

        # reshape to video
        # 重塑为视频

        x_video = rearrange(x_video, 'b (f h w) d -> b f h w d', h = image_size, w = image_size)

        x_image_h = x_image_w = x_frame = None

        # chunk depending on whether shifting time, space, or both
        # 根据是否移位时间、空间或两者来分块

        if self.shift_space and self.shift_time:
            x_frame, x_image_h, x_image_w, *x_rest = x_video.chunk(5, dim = -1)
        elif self.shift_space:
            x_image_h, x_image_w, *x_rest = x_video.chunk(4, dim = -1)
        elif self.shift_time:
            x_frame, *x_rest = x_video.chunk(3, dim = -1)

        # shifts
        # 移位操作

        if self.shift_space:
            x_image_h = F.pad(x_image_h, (0, 0, 0, 0, 1, -1))
            x_image_w = F.pad(x_image_w, (0, 0, 1, -1))

        if self.shift_time:
            x_frame = F.pad(x_frame, (0, 0, 0, 0, 0, 0, 1, -1))

        # concat
        # 连接操作

        x_shifted = [x_frame, x_image_h, x_image_w, *x_rest]
        x_shifted = list(filter(exists, x_shifted))

        x_video = torch.cat(x_shifted, dim = -1)

        # merge text and image sequence back together
        # 将文本和图像序列合并在一起

        x_video = rearrange(x_video, 'b f h w d -> b (f h w) d')
        x_video = x_video[:, :n]

        x = torch.cat((x_bos, x_video), dim = 1)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    # 定义 GEGLU 类，用于实现 Gated Linear Unit 激活函数
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    # 定义 FeedForward 类，用于实现前馈神经网络
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        dropout = 0.,
        chunk_size = None,  # chunk size to process feedforward, along sequence length, from Reformer paper. None means do not chunk
    ):
        super().__init__()
        inner_dim = (dim * mult * 2) // 3
        self.chunk_size = chunk_size

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias = False),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, x):
        if not exists(self.chunk_size):
            return self.net(x)

        x_chunks = x.split(self.chunk_size, dim = -2)
        out_chunks = [self.net(c) for c in x_chunks]
        return torch.cat(out_chunks, dim = -2)

# attention classes

class Attention(nn.Module):
    # 定义 Attention 类，用于实现注意力机制
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        causal = False,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5

        self.null_k = nn.Parameter(torch.randn(heads, 1, dim_head))
        self.null_v = nn.Parameter(torch.randn(heads, 1, dim_head))

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        context_mask = None,
        rotary_pos_emb = None
        ):
        # 获取输入张量 x 的 batch 大小、头数、设备信息
        b, h, device = x.shape[0], self.heads, x.device

        # 检查是否存在上下文信息
        has_context = exists(context)
        # 如果存在上下文信息，则将上下文信息作为键值对输入
        kv_input = context if has_context else x

        # 将输入张量 x 转换为查询向量 q，键值对转换为 k 和 v
        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))
        # 将查询 q、键 k、值 v 重排为 batch、头数、序列长度、维度的形式
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # 如果不存在上下文信息且存在旋转位置嵌入，则应用旋转位置嵌入
        if not has_context and exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        # 添加空键/值，用于条件丢弃
        null_k = repeat(self.null_k, 'h 1 d -> b h 1 d', b = b)
        null_v = repeat(self.null_v, 'h 1 d -> b h 1 d', b = b)

        # 将空键值与原始键值连接起来
        k = torch.cat((null_k, k), dim = -2)
        v = torch.cat((null_v, v), dim = -2)

        # 缩放
        q = q * self.scale

        # 相似度计算
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 掩码值
        mask_value = -torch.finfo(x.dtype).max

        # 如果存在键掩码，则对相似度矩阵进行掩码处理
        key_mask = mask if not has_context else context_mask
        if exists(key_mask):
            key_mask = F.pad(key_mask, (1, 0), value = True) # 始终注意空键/值
            key_mask = rearrange(key_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~key_mask, mask_value)

        # 如果是因果注意力，则对相似度矩阵进行掩码处理
        if self.causal:
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device = device, dtype = torch.bool).triu_(j - i + 1)
            sim = sim.masked_fill(mask, mask_value)

        # 注意力权重计算
        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = self.talking_heads(attn)
        attn = self.dropout(attn)

        # 聚合、合并和组合头
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# 定义一个名为 Sparse3DNA 的神经网络模块
class Sparse3DNA(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        video_shape,
        kernel_size = 3,
        dilation = 1,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = False,
        query_num_frames_chunk = None,
        rel_pos_bias = False
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 计算内部维度
        inner_dim = dim_head * heads
        # 设置头数和缩放因子
        self.heads = heads
        self.scale = dim_head ** -0.5

        # 初始化 dropout 层和线性变换层
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        # 初始化 talking heads 和输出层
        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # 转换为元组并确保卷积核大小为奇数
        self.dilation = cast_tuple(dilation, size = 3)
        self.kernel_size = cast_tuple(kernel_size, size = 3)
        assert all(map(lambda n: n % 2 == 1, self.kernel_size)), 'kernel size must be odd'

        # 计算卷积核元素数量
        self.kernel_numel = mult_reduce(self.kernel_size)

        # 如果需要，为每个头计算相对位置偏置
        self.rel_pos_bias = AxialPositionalEmbedding(heads, shape = self.kernel_size) if rel_pos_bias else None

        # 计算填充
        self.padding_frame = calc_same_padding(self.kernel_size[0], self.dilation[0])
        self.padding_height = calc_same_padding(self.kernel_size[1], self.dilation[1])
        self.padding_width = calc_same_padding(self.kernel_size[2], self.dilation[2])

        # 根据是否是因果卷积使用不同的填充
        if causal:
            self.video_padding = (self.padding_width * 2, 0, self.padding_height * 2, 0, self.padding_frame * 2, 0)
        else:
            self.video_padding = (self.padding_width, self.padding_width, self.padding_height, self.padding_height, self.padding_frame, self.padding_frame)

        # 保存视频形状并计算最大令牌数量
        self.video_shape = video_shape
        max_frames, fmap_size, _ = video_shape
        max_num_tokens = torch.empty(video_shape).numel()
        self.max_num_tokens = max_num_tokens

        # 限制内存使用，一次处理多少查询令牌
        self.query_num_frames_chunk = default(query_num_frames_chunk, max_frames)

        # 预先计算因果掩码
        ones = torch.ones((max_num_tokens,))
        ones = rearrange(ones, '(f h w) -> 1 1 f h w', f = max_frames, h = fmap_size, w = fmap_size)
        ones = F.pad(ones, self.video_padding, value = 0.)
        ones = unfoldNd(ones, kernel_size = self.kernel_size, dilation = self.dilation)
        ones = rearrange(ones, '1 k n -> n k')

        # 掩盖填充
        padding_mask = ones == 0.

        # bos 令牌永远不会被掩盖
        mask = F.pad(padding_mask, (1, 0), value = False)
        self.register_buffer('mask', mask)

# 定义一个名为 SparseCausal2DNA 的神经网络模块
class SparseCausal2DNA(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        *,
        dim,
        height = 1,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        kernel_size = 5,
        dilation = 1,
        rel_pos_bias = False
    # 定义 Transformer 层
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 计算每个头的维度
        inner_dim = heads * dim_head
        self.heads = heads
        # 缩放因子
        self.scale = dim_head ** -0.5

        # 定义用于交互的卷积层
        self.talking_heads = nn.Conv3d(heads, heads, 1, bias = False)
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        # 线性变换，将输入维度转换为内部维度的三倍
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # 线性变换，将内部维度转换为输出维度
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # 处理用于展开的变量

        # 高度信息，宽度为序列长度，时间轴 - (batch, seq) -> (batch, time, height)
        self.height = height
        # 卷积核大小
        self.kernel_size = (kernel_size, height)
        # 膨胀率
        self.dilation = (dilation, 1)
        # 因果填充
        self.causal_padding = (0, 0, calc_same_padding(kernel_size, dilation) * 2, 0)
        # 相对位置偏置
        self.rel_pos_bias = AxialPositionalEmbedding(heads, shape = self.kernel_size) if exists(rel_pos_bias) else None

        # 因果掩码

        # 注册缓冲区变量 mask
        self.register_buffer('mask', None, persistent = False)

    # 获取掩码
    def get_mask(self, t):
        # 如果 mask 存在且 mask 的倒数第三维与 t 的倒数第三维相同，则返回 mask
        if exists(self.mask) and self.mask.shape[-3] == t.shape[-3]:
            return self.mask

        device, seq_len = t.device, t.shape[-3] * self.height

        # 创建全为 1 的张量
        ones = torch.ones((seq_len,), device = device)
        # 重排张量维度
        ones = rearrange(ones, '(n m) -> 1 1 n m', m = self.height)

        # 对全为 1 的张量进行填充
        ones = F.pad(ones, self.causal_padding, value = 0.)
        # 展开张量
        ones = unfoldNd(ones, kernel_size = self.kernel_size, dilation = self.dilation)
        ones = rearrange(ones, '1 d n -> n d')

        # 创建填充掩码
        padding_mask = rearrange(ones, 'n j -> n 1 j') == 0.
        mask = F.pad(padding_mask, (1, 0), value = False)

        # 注册缓冲区变量 mask
        self.register_buffer('mask', mask, persistent = False)
        return mask

    # 前向传播方法
    def forward(
        self,
        x,
        **kwargs
        ):
            # 获取输入张量的维度信息
            b, n, h, device = x.shape[0], x.shape[1], self.heads, x.device

            # 计算每个时间步的标记数和卷积核元素数
            tokens_per_timestep = self.height
            kernel_numel = self.kernel_size[0] * self.kernel_size[1]

            # 填充到正确的长度

            bos_only = n == 1
            seq_pad = padding_to_multiple_of(n - 1, tokens_per_timestep)

            # 为视频中的最后一个标记进行填充

            padded_x = F.pad(x, (0, 0, 0, seq_pad), value = 0.) if seq_pad > 0 else x

            # 推导查询、键、值

            q, k, v = self.to_qkv(padded_x).chunk(3, dim = -1)

            # 处理仅有 bos 的情况

            if bos_only:
                return self.to_out(v)

            out_bos = v[:, :1]

            # 分割头部

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

            # 缩放

            q = q * self.scale

            # 处理 bos

            (q_bos, q), (k_bos, k), (v_bos, v) = map(lambda t: (t[:, :, 0], t[:, :, 1:]), (q, k, v))

            # 重塑键/值以进行展开

            k, v = map(lambda t: rearrange(t, 'b h (x y) d -> (b h) d x y ', y = tokens_per_timestep), (k, v))
            k, v = map(lambda t: F.pad(t, self.causal_padding), (k, v))
            k, v = map(lambda t: F.unfold(t, kernel_size = self.kernel_size, dilation = self.dilation), (k, v))
            k, v = map(lambda t: rearrange(t, '(b h f) (d j) i -> b h i (f j) d', b = b, h = h, j = kernel_numel), (k, v))

            # 添加 bos

            k_bos_repeated, v_bos_repeated = map(lambda t: repeat(t, 'b h d -> b h i 1 d', i = k.shape[-3]), (k_bos, v_bos))
            k = torch.cat((k_bos_repeated, k), dim = -2)
            v = torch.cat((v_bos_repeated, v), dim = -2)

            q = rearrange(q, 'b h (x y) d -> b h x y d', y = tokens_per_timestep)

            sim = einsum('b h n i d, b h n j d -> b h n i j', q, k)

            # 相对位置偏置

            if exists(self.rel_pos_bias):
                rel_pos_bias = self.rel_pos_bias()
                rel_pos_bias = rearrange(rel_pos_bias, 'j h -> h 1 1 j')
                rel_pos_bias = F.pad(rel_pos_bias, (1, 0), value = 0.)
                sim = sim + rel_pos_bias

            # 因果 + 填充掩码

            mask_value = -torch.finfo(x.dtype).max
            mask = self.get_mask(sim)
            sim = sim.masked_fill(mask, mask_value)

            # 注意力

            attn = sim.softmax(dim = -1, dtype = torch.float32)
            attn = self.talking_heads(attn)
            attn = self.dropout(attn)

            # 聚合、合并和组合头部

            out = einsum('b h n i j, b h n j d -> b h n i d', attn, v)
            out = rearrange(out, 'b h x y d -> b (x y) (h d)')

            # 将 bos 的输出添加回去

            out = torch.cat((out_bos, out), dim = -2)

            return self.to_out(out[:, :n])
# 定义一个名为 SparseCross2DNA 的神经网络模块
class SparseCross2DNA(nn.Module):
    # 初始化函数，接受一些参数
    def __init__(
        self,
        *,
        dim,  # 输入维度
        image_size,  # 图像大小
        heads = 8,  # 多头注意力机制中的头数，默认为8
        dim_head = 64,  # 每个头的维度，默认为64
        dropout = 0.,  # Dropout 概率，默认为0
        kernel_size = 3,  # 卷积核大小，默认为3
        dilation = 1,  # 膨胀率，默认为1
    ):
        super().__init__()  # 调用父类的初始化函数
        inner_dim = heads * dim_head  # 内部维度为头数乘以每个头的维度
        self.heads = heads  # 多头注意力机制中的头数
        self.scale = dim_head ** -0.5  # 缩放因子

        # 初始化可学习参数 null_k 和 null_v
        self.null_k = nn.Parameter(torch.randn(heads, 1, dim_head))
        self.null_v = nn.Parameter(torch.randn(heads, 1, dim_head))

        # 初始化可学习参数 talking_heads，用于多头注意力机制
        self.talking_heads = nn.Conv3d(heads, heads, 1, bias = False)
        self.dropout = nn.Dropout(dropout)  # 初始化 Dropout 层
        self.to_q = nn.Linear(dim, inner_dim, bias = False)  # 输入到查询向量的线性变换
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)  # 输入到键值对的线性变换
        self.to_out = nn.Linear(inner_dim, dim, bias = False)  # 输出线性变换

        # 处理 2D 展开的变量

        self.image_size = image_size  # 图像大小
        self.kernel_size = kernel_size  # 卷积核大小
        self.dilation = dilation  # 膨胀率
        self.padding = calc_same_padding(kernel_size, dilation)  # 计算填充大小

    # 前向传播函数，接受输入 x 和一些关键字参数
    def forward(
        self,
        x,
        *,
        context,  # 上下文信息
        context_mask = None,  # 上下文掩码，默认为 None
        **kwargs  # 其他关键字参数
        ):
            # 获取输入张量的维度信息
            b, n, h, device = x.shape[0], x.shape[1], self.heads, x.device

            # 获取模型参数的相关信息
            fmap_size, kernel_size, dilation, padding = self.image_size, self.kernel_size, self.dilation, self.padding

            # 计算上下文长度、每帧的标记数、卷积核元素数
            context_len = context.shape[-2]
            tokens_per_frame = fmap_size * fmap_size
            kernel_numel = kernel_size * kernel_size

            # 如果上下文掩码不存在，则创建一个全为 True 的掩码
            if not exists(context_mask):
                context_mask = torch.ones((b, context_len), dtype=torch.bool, device=device)

            # 初始化掩码值
            mask_value = -torch.finfo(x.dtype).max

            # 生成查询、键、值
            qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

            # 缩放查询
            q = q * self.scale

            # 处理 bos
            q_bos, q = q[:, :, 0], q[:, :, 1:]

            null_k_for_bos = repeat(self.null_k, 'h 1 d -> b h 1 d', b=b)
            null_v_for_bos = repeat(self.null_v, 'h 1 d -> b h 1 d', b=b)

            k_for_bos = torch.cat((null_k_for_bos, k), dim=-2)
            v_for_bos = torch.cat((null_v_for_bos, v), dim=-2)

            sim_bos = einsum('b h d, b h j d -> b h j', q_bos, k_for_bos)

            bos_context_mask = rearrange(context_mask, 'b j -> b 1 j')
            bos_context_mask = F.pad(bos_context_mask, (1, 0), value=True)
            sim_bos = sim_bos.masked_fill(~bos_context_mask, mask_value)

            attn_bos = sim_bos.softmax(dim=-1, dtype=torch.float32)
            out_bos = einsum('b h j, b h j d -> b h d', attn_bos, v_for_bos)
            out_bos = rearrange(out_bos, 'b h d -> b 1 (h d)')

            # 如果只有一个标记，则直接返回结果
            if n == 1:
                return self.to_out(out_bos)

            # 重塑键/值以进行展开
            k, v = map(lambda t: rearrange(t, 'b h (f x y) d -> (b h f) d x y', x=fmap_size, y=fmap_size), (k, v))
            k, v = map(lambda t: F.unfold(t, kernel_size=kernel_size, dilation=dilation, padding=padding), (k, v))
            k, v = map(lambda t: rearrange(t, '(b h f) (d j) i -> b h i (f j) d', b=b, h=h, j=kernel_numel), (k, v))

            # 添加空键/值，用于条件丢弃
            null_k = repeat(self.null_k, 'h 1 d -> b h i 1 d', b=b, i=tokens_per_frame)
            null_v = repeat(self.null_v, 'h 1 d -> b h i 1 d', b=b, i=tokens_per_frame)

            k = torch.cat((null_k, k), dim=-2)
            v = torch.cat((null_v, v), dim=-2)

            # 将查询填充到最近的帧
            q_padding = padding_to_multiple_of(q.shape[-2], tokens_per_frame)
            q = F.pad(q, (0, 0, 0, q_padding), value=0.)

            q = rearrange(q, 'b h (f i) d -> b h f i d', i=tokens_per_frame)

            # 计算相似度
            sim = einsum('b h f i d, b h i j d -> b h f i j', q, k)

            # 掩码
            context_mask = rearrange(context_mask, 'b (f x y) -> (b f) 1 x y', x=fmap_size, y=fmap_size)
            context_mask = F.unfold(context_mask.float(), kernel_size=kernel_size, dilation=dilation, padding=padding)
            context_mask = context_mask == 1.
            context_mask = rearrange(context_mask, '(b f) j i -> b 1 1 i (f j)', b=b, j=kernel_numel)
            context_mask = F.pad(context_mask, (1, 0), value=True)  # 总是关注空键/值

            sim = sim.masked_fill(~context_mask, mask_value)

            # 注意力
            attn = sim.softmax(dim=-1, dtype=torch.float32)
            attn = self.talking_heads(attn)
            attn = self.dropout(attn)

            # 聚合、合并和组合头
            out = einsum('b h f i j, b h i j d -> b h f i d', attn, v)
            out = rearrange(out, 'b h f n d -> b (f n) (h d)')

            # 将 bos 的输出添加回去
            out = torch.cat((out_bos, out), dim=1)

            return self.to_out(out[:, :n])
"""
用于实现高效的音频 <-> 视频注意力机制
主要灵感来源于 https://arxiv.org/abs/2112.04426 中的块交叉注意力机制
"""

class CrossModalityCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 输入维度
        chunk_size,  # 块大小
        context_chunk_size,  # 上下文块大小
        heads = 8,  # 头数
        dim_head = 64,  # 每个头的维度
        context_dim = None,  # 上下文维度，默认为None
        has_start_token = True,  # 是否有起始标记
        context_has_start_token = True,  # 上下文是否有起始标记
        norm = False,  # 是否进行归一化
        norm_context = False,  # 上下文是否进行归一化
        dropout = 0.  # 丢弃概率
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim  = dim_head * heads

        self.norm = nn.LayerNorm(dim) if norm else nn.Identity()  # 归一化层
        self.context_norm = nn.LayerNorm(context_dim) if norm_context else nn.Identity()  # 上下文归一化层

        self.to_q = nn.Linear(dim, inner_dim, bias = False)  # 查询线性层
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)  # 键值线性层
        self.to_out = nn.Linear(inner_dim, dim, bias = False)  # 输出线性层

        self.null_k = nn.Parameter(torch.randn(heads, dim_head))  # 空键参数
        self.null_v = nn.Parameter(torch.randn(heads, dim_head))  # 空值参数

        self.talking_heads = nn.Conv3d(heads, heads, 1)  # 三维卷积层
        self.dropout = nn.Dropout(dropout)  # 丢弃层

        self.has_start_token = has_start_token  # 是否有起始标记
        self.context_has_start_token = context_has_start_token  # 上下文是否有起始标记

        self.chunk_size = chunk_size  # 块大小
        self.context_chunk_size = context_chunk_size  # 上下文块大小

    def forward(
        self,
        seq,  # 序列输入
        context,  # 上下文输入
        mask = None,  # 掩码
        context_mask = None  # 上下文掩码

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 输入维度
        depth,  # 深度
        causal = False,  # 是否因果
        heads = 8,  # 头数
        dim_head = 64,  # 每个头的维度
        ff_mult = 4,  # FeedForward 层的倍增因子
        cross_attend = False,  # 是否跨模态注意力
        attn_dropout = 0.,  # 注意力丢弃概率
        ff_dropout = 0.,  # FeedForward 层的丢弃概率
        ff_chunk_size = None,  # FeedForward 层的块大小
        cross_2dna_attn = False,  # 是否跨 2DNA 注意力
        cross_2dna_image_size = None,  # 跨 2DNA 图像大小
        cross_2dna_kernel_size = 3,  # 跨 2DNA 卷积核大小
        cross_2dna_dilations = (1,),  # 跨 2DNA 膨胀率
        sparse_3dna_attn = False,  # 是否稀疏 3DNA 注意力
        sparse_3dna_kernel_size = 3,  # 稀疏 3DNA 卷积核大小
        sparse_3dna_video_shape = None,  # 稀疏 3DNA 视频形状
        sparse_3dna_query_num_frames_chunk = None,  # 稀疏 3DNA 查询帧块数
        sparse_3dna_dilations = (1,),  # 稀疏 3DNA 膨胀率
        sparse_3dna_rel_pos_bias = False,  # 稀疏 3DNA 相对位置偏置
        shift_video_tokens = False,  # 是否移动视频标记
        rotary_pos_emb = False  # 是否使用旋转位置嵌入
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 断言条件，如果不满足则抛出异常
        assert not (sparse_3dna_attn and not exists(sparse_3dna_video_shape)), 'sparse_3dna_video_shape must be defined if turned on'
        assert not (cross_2dna_attn and not exists(cross_2dna_image_size)), 'cross_2dna_image_size must be defined'

        # 初始化层列表
        self.layers = MList([])

        # 循环创建多个层
        for ind in range(depth):
            if sparse_3dna_attn:
                # 如果启用了稀疏3DNA注意力机制
                dilation = sparse_3dna_dilations[ind % len(sparse_3dna_dilations)]

                self_attn = Sparse3DNA(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    causal = causal,
                    kernel_size = sparse_3dna_kernel_size,
                    dilation = dilation,
                    video_shape = sparse_3dna_video_shape,
                    query_num_frames_chunk = sparse_3dna_query_num_frames_chunk,
                    rel_pos_bias = sparse_3dna_rel_pos_bias,
                )
            else:
                # 否则使用普通的注意力机制
                self_attn = Attention(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    causal = causal,
                    dropout = attn_dropout
                )

            cross_attn = None

            if cross_attend:
                if cross_2dna_attn:
                    # 如果启用了交叉2DNA注意力机制
                    dilation = cross_2dna_dilations[ind % len(cross_2dna_dilations)]

                    cross_attn = SparseCross2DNA(
                        dim = dim,
                        heads = heads,
                        dim_head = dim_head,
                        dropout = attn_dropout,
                        image_size = cross_2dna_image_size,
                        kernel_size = cross_2dna_kernel_size,
                        dilation = dilation
                    )

                else:
                    # 否则使用普通的注意力机制
                    cross_attn = Attention(
                        dim = dim,
                        heads = heads,
                        dim_head = dim_head,
                        dropout = attn_dropout
                    )

            # 创建前馈神经网络层
            ff = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)

            if sparse_3dna_attn and shift_video_tokens:
                # 如果启用了稀疏3DNA注意���机制并且需要移动视频标记
                fmap_size = sparse_3dna_video_shape[-1]
                self_attn = ShiftVideoTokens(self_attn, image_size = fmap_size)
                ff        = ShiftVideoTokens(ff, image_size = fmap_size)

            # 将当前层的各个组件添加到层列表中
            self.layers.append(MList([
                SandwichNorm(dim = dim, fn = self_attn),
                SandwichNorm(dim = dim, fn = cross_attn) if cross_attend else None,
                SandwichNorm(dim = dim, fn = ff)
            ]))

        # 初始化稳定层归一化
        self.norm = StableLayerNorm(dim)

    # 前向传播方法
    def forward(
        self,
        x,
        mask = None,
        context = None,
        context_mask = None
    ):
        # 遍历所有层
        for attn, cross_attn, ff in self.layers:
            # 使用自注意力机制更新输入
            x = attn(x, mask = mask) + x

            # 如果存在交叉注意力机制
            if exists(cross_attn):
                # 使用交叉注意力机制更新输入
                x = cross_attn(x, context = context, mask = mask, context_mask = context_mask) + x

            # 使用前馈神经网络更新输入
            x = ff(x) + x

        # 对输出进行稳定层归一化
        return self.norm(x)
# 定义一个可逆的 Transformer 模型类
class ReversibleTransformer(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        *,
        dim,  # 模型维度
        depth,  # 模型深度
        causal = False,  # 是否使用因果注意力
        heads = 8,  # 多头注意力的头数
        dim_head = 64,  # 每个头的维度
        ff_mult = 4,  # FeedForward 层的倍数
        cross_attend = False,  # 是否使用跨层注意力
        attn_dropout = 0.,  # 注意力层的 dropout 概率
        ff_dropout = 0.,  # FeedForward 层的 dropout 概率
        ff_chunk_size = None,  # FeedForward 层的分块大小
        cross_2dna_attn = False,  # 是否使用跨 2D 和 1D 注意力
        cross_2dna_image_size = None,  # 跨 2D 和 1D 注意力的图像大小
        cross_2dna_kernel_size = 3,  # 跨 2D 和 1D 注意力的卷积核大小
        cross_2dna_dilations = (1,),  # 跨 2D 和 1D 注意力的膨胀系数
        sparse_3dna_attn = False,  # 是否使用稀疏 3D 和 1D 注意力
        sparse_3dna_kernel_size = 3,  # 稀疏 3D 和 1D 注意力的卷积核大小
        sparse_3dna_video_shape = None,  # 稀疏 3D 和 1D 注意力的视频形状
        sparse_3dna_query_num_frames_chunk = None,  # 稀疏 3D 和 1D 注意力的查询帧数块大小
        sparse_3dna_dilations = (1,),  # 稀疏 3D 和 1D 注意力的膨胀系数
        sparse_3dna_rel_pos_bias = False,  # 稀疏 3D 和 1D 注意力是否使用相对位置偏置
        shift_video_tokens = False,  # 是否对视频 token 进行位移
        rotary_pos_emb = False  # 是否使用旋转位置编码
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言条件，如果不满足则抛出异常
        assert not (sparse_3dna_attn and not exists(sparse_3dna_video_shape)), 'sparse_3dna_video_shape must be defined if turned on'
        assert not (cross_2dna_attn and not exists(cross_2dna_image_size)), 'cross_2dna_image_size must be defined'

        # 初始化层列表
        self.layers = MList([])

        # 循环创建网络层
        for ind in range(depth):
            if sparse_3dna_attn:
                # 获取稀疏3DNA注意力机制的参数
                dilation = sparse_3dna_dilations[ind % len(sparse_3dna_dilations)]
                image_size = sparse_3dna_video_shape[-1]

                # 创建稀疏3DNA自注意力层
                self_attn = Sparse3DNA(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    causal = causal,
                    kernel_size = sparse_3dna_kernel_size,
                    dilation = dilation,
                    video_shape = sparse_3dna_video_shape,
                    query_num_frames_chunk = sparse_3dna_query_num_frames_chunk,
                    rel_pos_bias = sparse_3dna_rel_pos_bias,
                )
            else:
                image_size = None

                # 创建普通自注意力层
                self_attn = Attention(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    causal = causal,
                    dropout = attn_dropout
                )

            # 创建包装函数
            wrapper_fn = partial(ShiftVideoTokens, image_size = image_size, shift_space = sparse_3dna_attn and shift_video_tokens)

            # 添加自注意力层和前馈网络层到层列表
            self.layers.append(MList([
                SandwichNorm(dim = dim, fn = wrapper_fn(self_attn)),
                SandwichNorm(dim = dim, fn = wrapper_fn(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)))
            ]))

            # 如果不需要交叉注意力，则继续下一轮循环
            if not cross_attend:
                continue

            if cross_2dna_attn:
                # 获取交叉2DNA注意力机制的参数
                dilation = cross_2dna_dilations[ind % len(cross_2dna_dilations)]

                # 创建交叉2DNA注意力层
                cross_attn = SparseCross2DNA(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    dropout = attn_dropout,
                    image_size = cross_2dna_image_size,
                    kernel_size = cross_2dna_kernel_size,
                    dilation = dilation
                )
            else:
                # 创建普通交叉注意力层
                cross_attn = Attention(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    dropout = attn_dropout
                )

            # 添加交叉注意力层和前馈网络层到层列表
            self.layers.append(MList([
                SandwichNorm(dim = dim, fn = cross_attn),
                SandwichNorm(dim = dim, fn = wrapper_fn(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)))
            ]))

        # 设置注意力上下文层和路由
        attn_context_layer = ((True, False),) if cross_attend else tuple()
        route_attn = ((True, False), *attn_context_layer) * depth
        route_context = ((False, False), *attn_context_layer) * depth

        # 设置上下文路由映射和注意力路由映射
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        attn_route_map = {'mask': route_attn, 'rotary_pos_emb': route_attn}

        # 创建可逆序列网络
        self.net = ReversibleSequence(self.layers, args_route = {**context_route_map, **attn_route_map})
        # 创建稳定层归一化
        self.norm = StableLayerNorm(dim)

    # 前向传播函数
    def forward(
        self,
        x,
        **kwargs
    ):
        # 使用网络进行前向传播
        x = self.net(x, **kwargs)
        # 对结果进行归一化处理
        return self.norm(x)
# 双模态解码器（用于视频和音频合成）

class DualModalityDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_audio_tokens_per_video_frame,
        num_video_tokens_per_frame,
        sparse_3dna_video_shape,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_chunk_size = None,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilations = (1,),
        sparse_3dna_rel_pos_bias = False,
        sparse_2dna_kernel_size = 7,
        sparse_2dna_dilation = (1,),
        sparse_2dna_rel_pos_bias = False,
        shift_video_tokens = False,
        shift_audio_tokens = False,
        audio_tokens_per_timestep = 1,
        cross_modality_attn_every = 3
    # 定义前向传播函数
    def forward(
        self,
        video,
        audio,
        *,
        context,
        audio_mask = None,
        video_mask = None,
        context_mask = None,
        **kwargs
    ):
        # 遍历每个块和层类型
        for blocks, layer_type in zip(self.layers, self.layer_types):
            # 如果层类型为'intra_modality'
            if layer_type == 'intra_modality':
                # 解压块
                (video_self_attn, video_cross_attn, video_ff), (audio_self_attn, audio_cross_attn, audio_ff) = blocks

                # 视频自注意力机制
                video_ = video_self_attn(video, mask = video_mask) + video
                video_ = video_cross_attn(video_, context = context, mask = video_mask, context_mask = context_mask) + video_
                video_ = video_ff(video_) + video_

                # 音频自注意力机制
                audio_ = audio_self_attn(audio, mask = audio_mask) + audio
                audio_ = audio_cross_attn(audio_, context = context, mask = audio_mask, context_mask = context_mask) + audio_
                audio_ = audio_ff(audio_) + audio_

            # 如果层类型为'inter_modality'
            elif layer_type == 'inter_modality':
                # 解压块
                (video_to_audio_attn, video_ff), (audio_to_video_attn, audio_ff) = blocks

                # 视频到音频的注意力机制
                video_ = video_to_audio_attn(
                    video,
                    context = audio,
                    mask = video_mask,
                    context_mask = audio_mask
                ) + video

                # 音频到视频的注意力机制
                audio_ = audio_to_video_attn(
                    audio,
                    context = video,
                    mask = audio_mask,
                    context_mask = video_mask
                ) + audio

                video_ = video_ff(video_) + video_
                audio_ = audio_ff(audio_) + audio_
            else:
                raise ValueError(f'unknown layer type {layer_type}')

            video, audio = video_, audio_

        # 返回视频和音频的归一化结果
        return self.video_norm(video), self.audio_norm(audio)

# 可逆双模态解码器

class ReversibleDualModalityDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_audio_tokens_per_video_frame,
        num_video_tokens_per_frame,
        sparse_3dna_video_shape,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_chunk_size = None,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilations = (1,),
        sparse_3dna_rel_pos_bias = False,
        sparse_2dna_kernel_size = 7,
        sparse_2dna_dilation = (1,),
        sparse_2dna_rel_pos_bias = False,
        shift_video_tokens = False,
        shift_audio_tokens = False,
        audio_tokens_per_timestep = 1,
        cross_modality_attn_every = 3
    # 定义前向传播函数
    def forward(
        self,
        video,
        audio,
        *,
        context,
        audio_mask = None,
        video_mask = None,
        context_mask = None,
        **kwargs
    ):
        # 调用网络进行前向传播
        video, audio = self.net(
            video,
            audio,
            context = context,
            audio_mask = audio_mask,
            video_mask = video_mask,
            context_mask = context_mask
        )

        # 返回视频和音频的归一化结果
        return self.video_norm(video), self.audio_norm(audio)

# 嵌入
# 定义一个名为 Embedding 的神经网络模块类
class Embedding(nn.Module):
    # 初始化函数，接受形状参数和梯度分数参数
    def __init__(self, *shape, frac_gradient = 1.):
        super().__init__()
        # 设置梯度分数参数
        self.frac_gradient = frac_gradient
        # 创建 Embedding 层
        self.embed = nn.Embedding(*shape)

    # 前向传播函数
    def forward(self, x):
        # 将输入 x 传入 Embedding 层
        x = self.embed(x)

        # 如果处于训练状态且梯度分数小于1，则对 x 进行梯度分数处理
        if self.training and self.frac_gradient < 1:
            x = frac_gradient(x, self.frac_gradient)

        return x

# positional embedding

# 定义一个名为 AxialPositionalEmbedding 的神经网络模块类
class AxialPositionalEmbedding(nn.Module):
    # 初始化函数，接受维度参数和形状参数
    def __init__(
        self,
        dim,
        *,
        shape
    ):
        super().__init__()
        # 过滤形状参数中大于1的值，形成新的形状参数
        shape = tuple(filter(lambda t: t > 1, shape))

        # 设置维度、形状和轴数
        self.dim = dim
        self.shape = shape
        self.num_axials = len(shape)

        # 为每个轴创建随机参数
        for axial_ind, axial_len in enumerate(shape):
            axial_pos = nn.Parameter(torch.randn(axial_len, dim))
            setattr(self, f'axial{axial_ind + 1}', axial_pos)

    # 前向传播函数，接受 flatten 参数
    def forward(self, *, flatten = True):
        positions = None

        # 遍历每个轴
        for axial_ind in range(self.num_axials):
            axial_pos = getattr(self, f'axial{axial_ind + 1}')

            # 如果 positions 为空，则将当前轴位置赋给 positions
            if not exists(positions):
                positions = axial_pos
                continue

            # 对 positions 进行重排列，并加上当前轴位置
            positions = rearrange(positions, '... d -> ... 1 d')
            positions = positions + axial_pos

        # 如果 flatten 为 True，则对 positions 进行重排列
        if flatten:
            positions = rearrange(positions, '... d -> (...) d')

        return positions

# sampling helpers

# 定义一个名为 top_k 的函数，接受 logits 和阈值参数
def top_k(logits, thres = 0.5):
    # 获取 logits 的最后一个维度大小
    num_logits = logits.shape[-1]
    # 计算 k 值
    k = max(int((1 - thres) * num_logits), 1)
    # 获取前 k 个最大值的索引和值
    val, ind = torch.topk(logits, k)
    # 创建与 logits 相同大小的全为负无穷的张量
    probs = torch.full_like(logits, float('-inf'))
    # 根据索引将值填充到 probs 中
    probs.scatter_(1, ind, val)
    return probs

# main class

# 定义一个名为 NUWA 的神经网络模块类
class NUWA(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        *,
        dim,
        vae = None,
        image_size = None,
        max_video_frames = 5,
        text_num_tokens = 49408,
        text_max_seq_len = 256,
        text_enc_depth = 6,
        text_enc_dim_head = 64,
        text_enc_heads = 8,
        text_rotary_pos_emb = True,
        enc_reversible = False,
        dec_depth = 6,
        dec_dim_head = 64,
        dec_heads = 8,
        dec_reversible = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_chunk_size = None,
        embed_gradient_frac = 0.2,
        shift_video_tokens = True,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilation = 1,
        sparse_3dna_rel_pos_bias = False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言 VAE 或图像大小必须被指定
        assert exists(vae) ^ exists(image_size), 'either VAE or image size must be specified'

        self.vae = None
        # 如果存在 VAE，则复制一个用于评估的 VAE，并设置图像大小为 VAE 的图像大小
        if exists(vae):
            self.vae = vae.copy_for_eval()
            image_size = vae.image_size

        # 获取 VAE 的层数和图像 token 数量
        vae_num_layers = vae.num_layers
        num_image_tokens = vae.codebook_size

        self.text_max_seq_len = text_max_seq_len
        # 创建文本嵌入层
        self.text_embedding = Embedding(text_num_tokens, dim, frac_gradient = embed_gradient_frac)

        # 为文本创建位置嵌入
        self.text_abs_pos_emb = Embedding(text_max_seq_len, dim)  if not text_rotary_pos_emb else None
        self.text_rotary_pos_emb = RotaryEmbedding(dim = min(32, text_enc_dim_head)) if text_rotary_pos_emb else None

        # 根据是否可逆选择编码器的类型
        enc_transformer_klass = Transformer if not enc_reversible else ReversibleTransformer

        # 创建文本变换器
        self.text_transformer = enc_transformer_klass(
            dim = dim,
            depth = text_enc_depth,
            heads = text_enc_heads,
            dim_head = text_enc_dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            rotary_pos_emb = text_rotary_pos_emb
        )

        # 创建视频的开始 token
        self.video_bos = nn.Parameter(torch.randn(dim))
        # 创建图像嵌入层
        self.image_embedding = Embedding(num_image_tokens, dim, frac_gradient = embed_gradient_frac)

        # 计算特征图大小
        fmap_size = image_size // (2 ** vae_num_layers)

        self.video_fmap_size = fmap_size
        self.max_video_frames = max_video_frames
        video_shape = (max_video_frames, fmap_size, fmap_size)

        # 为视频创建位置嵌入
        self.video_pos_emb = AxialPositionalEmbedding(dim, shape = video_shape)

        # 设置稀疏 3D 邻近注意力的循环扩张
        sparse_3dna_dilations = tuple(range(1, sparse_3dna_dilation + 1)) if not isinstance(sparse_3dna_dilation, (list, tuple)) else sparse_3dna_dilation

        # 根据是否可逆选择解码器的类型
        dec_transformer_klass = Transformer if not dec_reversible else ReversibleTransformer

        # 创建视频变换器
        self.video_transformer = dec_transformer_klass(
            dim = dim,
            depth = dec_depth,
            heads = dec_heads,
            dim_head = dec_dim_head,
            causal = True,
            cross_attend = True,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_chunk_size = ff_chunk_size,
            shift_video_tokens = shift_video_tokens,
            sparse_3dna_video_shape = video_shape,
            sparse_3dna_attn = True,
            sparse_3dna_kernel_size = sparse_3dna_kernel_size,
            sparse_3dna_dilations = sparse_3dna_dilations,
            sparse_3dna_query_num_frames_chunk = sparse_3dna_query_num_frames_chunk,
            sparse_3dna_rel_pos_bias = sparse_3dna_rel_pos_bias
        )

        # 创建输出层
        self.to_logits = nn.Linear(dim, num_image_tokens, bias = False)

    def embed_text(self, text, mask = None):
        # 获取文本的批量大小、序列长度和设备
        batch, seq_len, device = *text.shape, text.device
        # 断言序列长度不超过文本最大序列长度
        assert seq_len <= self.text_max_seq_len, 'your input text has a greater length than what was designated on initialization'

        # 对文本进行嵌入
        tokens = self.text_embedding(text)

        if exists(self.text_abs_pos_emb):
            # 添加绝对位置嵌入
            pos_emb = self.text_abs_pos_emb(torch.arange(seq_len, device = device))
            tokens = tokens + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.text_rotary_pos_emb):
            # 如果存在旋转位置嵌入，则获取旋转位置嵌入
            rotary_pos_emb = self.text_rotary_pos_emb(seq_len, device = device)

        # 返回文本变换器的结果
        return self.text_transformer(
            tokens,
            mask = mask,
            rotary_pos_emb = rotary_pos_emb
        )

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        *,
        text,
        filter_thres = 0.9,
        temperature = 1.,
        decode_max_batchsize = 10,
        cond_scale = 2.,
        num_frames = None
    ):
        # 解包文本张量的形状和设备信息
        batch, seq_len, device = *text.shape, text.device

        # 创建文本掩码，将文本张量中非零元素标记为True
        text_mask = text != 0
        # 使用文本嵌入层对文本进行嵌入处理，同时传入文本掩码
        text_embeds = self.embed_text(text, mask = text_mask)

        # 重复视频起始符号，形状为(batch, 1, d)，其中d为视频特征维度
        bos = repeat(self.video_bos, 'd -> b 1 d', b = batch)

        # 创建空的视频索引张量，形状为(batch, 0)，设备为指定设备
        video_indices = torch.empty((batch, 0), device = device, dtype = torch.long)

        # 计算每帧视频的标记数量
        num_tokens_per_frame = self.video_fmap_size ** 2

        # 设置视频帧数，默认为最大视频帧数
        num_frames = default(num_frames, self.max_video_frames)
        total_video_tokens =  num_tokens_per_frame * num_frames
        max_video_tokens = num_tokens_per_frame * self.max_video_frames

        # 获取视频位置编码
        pos_emb = self.video_pos_emb()

        # 遍历视频标记总数
        for ind in tqdm(range(total_video_tokens)):
            # 备份视频索引输入
            video_indices_input = video_indices

            # 获取当前视频标记数量
            num_video_tokens = video_indices.shape[1]
            # 如果视频标记数量超过最大视频标记数量
            if num_video_tokens > max_video_tokens:
                # 计算当前帧标记数量
                curr_frame_tokens = num_video_tokens % num_tokens_per_frame
                # 计算回溯标记数量
                lookback_tokens = (self.max_video_frames - (0 if curr_frame_tokens == 0 else 1)) * num_tokens_per_frame + curr_frame_tokens
                # 更新视频索引输入为最近的标记
                video_indices_input = video_indices[:, -lookback_tokens:]

            # 获取帧嵌入
            frame_embeddings = self.image_embedding(video_indices_input)
            # 添加位置编码到帧嵌入
            frame_embeddings = pos_emb[:frame_embeddings.shape[1]] + frame_embeddings
            # 拼接起始符号和帧嵌入
            frame_embeddings = torch.cat((bos, frame_embeddings), dim = 1)

            # 使用视频Transformer处理帧嵌入和文本嵌入
            frame_embeddings = self.video_transformer(
                frame_embeddings,
                context = text_embeds,
                context_mask = text_mask
            )

            # 获取输出logits
            logits = self.to_logits(frame_embeddings)

            # 如果条件缩放不为1
            if cond_scale != 1:
                # 使用视频Transformer处理帧嵌入和文本嵌入，但文本掩码为全零
                uncond_frame_embeddings = self.video_transformer(
                    frame_embeddings,
                    context = text_embeds,
                    context_mask = torch.zeros_like(text_mask).bool()
                )

                # 获取无条件logits
                uncond_logits = self.to_logits(uncond_frame_embeddings)
                # 更新logits为无条件logits加上条件缩放后的值
                logits = uncond_logits + (logits - uncond_logits) * cond_scale

            # 选择最后一个标记的logits
            logits = logits[:, -1, :]

            # 对logits进行筛选，保留前k个值
            filtered_logits = top_k(logits, thres = filter_thres)
            # 使用Gumbel采样获取样本
            sample = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            # 重新排列样本的形状
            sample = rearrange(sample, 'b -> b 1')
            # 拼接样本到视频索引
            video_indices = torch.cat((video_indices, sample), dim = 1)

        # 根据视频索引获取VAE的代码簿
        codes = self.vae.codebook[video_indices]
        # 重新排列代码的形状
        codes = rearrange(codes, 'b (f h w) d -> (b f) d h w', h = self.video_fmap_size, w = self.video_fmap_size)

        # 批处理代码，通过VAE解码获取图像重构
        image_reconstructions = batch_process(codes, self.vae.decode, chunks = decode_max_batchsize)
        # 重新排列图像重构的形状
        video = rearrange(image_reconstructions, '(b f) d h w -> b f d h w', b = batch)
        # 返回视频
        return video

    # 前向传播函数
    def forward(
        self,
        *,
        text,
        video = None,
        return_loss = False,
        cond_dropout_prob = 0.2
        # 从输入的张量形状中获取批次大小、序列长度、帧数和设备信息
        batch, seq_len, frames, device = *text.shape, video.shape[1], text.device

        # 创建文本掩码，将文本中非零元素标记为True
        text_mask = text != 0
        # 使用文本嵌入模型对文本进行嵌入处理，同时应用文本掩码
        text_embeds = self.embed_text(text, mask = text_mask)

        # 如果视频数据类型为torch.long，则直接使用视频帧索引
        if video.dtype == torch.long:
            frame_indices = video
        else:
            # 否则，确保视频帧数与最大视频帧数相同，并且需要传入VAE模型以自动将视频编码为ids
            assert frames == self.max_video_frames, f'you must give the full video frames ({self.max_video_frames}) during training'
            assert exists(self.vae), 'VAE must be passed in if you wish for video to be encoded to ids automatically'
            frame_indices = self.vae.get_video_indices(video)

        # 重新排列视频帧索引的形状
        frame_indices = rearrange(frame_indices, 'b ... -> b (...)')
        # 如果不需要返回损失，则将视频帧索引的最后一帧排除在外
        frame_indices_input = frame_indices[:, :-1] if return_loss else frame_indices

        # 使用图像嵌入模型对视频帧索引进行嵌入处理
        frame_embeddings = self.image_embedding(frame_indices_input)
        # 添加视频位置编码到帧嵌入中
        frame_embeddings = self.video_pos_emb()[:-1] + frame_embeddings

        # 在帧嵌入的开头添加起始符号
        bos = repeat(self.video_bos, 'd -> b 1 d', b = batch)
        frame_embeddings = torch.cat((bos, frame_embeddings), dim = 1)

        # 如果处于训练状态且条件丢弃概率大于0，则随机丢弃条件
        if self.training and cond_dropout_prob > 0:
            # 随机生成与文本掩码相同形状的无条件掩码
            uncond_mask = prob_mask_like((batch,), cond_dropout_prob, device = device)
            # 将无条件掩码应用到文本掩码上
            text_mask *= rearrange(~uncond_mask, 'b -> b 1')

        # 使用视频变换器模型处理帧嵌入和文本嵌入
        frame_embeddings = self.video_transformer(
            frame_embeddings,
            context = text_embeds,
            context_mask = text_mask
        )

        # 将帧嵌入转换为logits
        logits = self.to_logits(frame_embeddings)

        # 如果不需要返回损失，则直接返回logits
        if not return_loss:
            return logits

        # 计算交叉熵损失
        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), frame_indices)
        return loss
# 定义一个名为NUWAVideoAudio的类，继承自nn.Module
class NUWAVideoAudio(nn.Module):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        *,
        vae,  # 视频和音频编码器
        dim,  # 模型维度
        image_size,  # 图像尺寸
        num_audio_tokens,  # 音频标记数量
        num_audio_tokens_per_video_frame,  # 每个视频帧的音频标记数量
        audio_tokens_per_timestep = 1,  # 每个时间步的音频标记数量
        max_video_frames = 5,  # 最大视频帧数
        text_num_tokens = 49408,  # 文本标记数量
        text_max_seq_len = 256,  # 文本最大序列长度
        text_enc_depth = 6,  # 文本编码器深度
        text_enc_dim_head = 64,  # 文本编码器头维度
        text_enc_heads = 8,  # 文本编码器头数
        text_rotary_pos_emb = False,  # 是否使用旋转位置嵌入
        enc_reversible = False,  # 编码器是否可逆
        dec_reversible = True,  # 解码器是否可逆
        dec_depth = 6,  # 解码器深度
        dec_dim_head = 64,  # 解码器头维度
        dec_heads = 8,  # 解码器头数
        attn_dropout = 0.,  # 注意力机制的dropout
        ff_dropout = 0.,  # 前馈网络的dropout
        ff_chunk_size = None,  # 前馈网络的分块大小
        embed_gradient_frac = 0.2,  # 嵌入梯度比例
        shift_video_tokens = True,  # 是否移动视频标记
        shift_audio_tokens = True,  # 是否移动音频标记
        sparse_3dna_kernel_size = 3,  # 稀疏3D卷积核大小
        sparse_3dna_query_num_frames_chunk = None,  # 稀疏3D卷积查询帧块数
        sparse_3dna_dilation = 1,  # 稀疏3D卷积膨胀率
        sparse_3dna_rel_pos_bias = True,  # 稀疏3D卷积相对位置偏置
        sparse_2dna_kernel_size = 7,  # 稀疏2D卷积核大小
        sparse_2dna_dilation = 1,  # 稀疏2D卷积膨胀率
        sparse_2dna_rel_pos_bias = True,  # 稀疏2D卷积相对位置偏置
        audio_loss_weight = 1.,  # 音频损失权重
        cross_modality_attn_every = 3  # 跨模态注意力的频率
        ):
        # 调用父类的构造函数
        super().__init__()
        # 复制 VAE 模型用于评估
        self.vae = vae.copy_for_eval()
        # 获取 VAE 模型的层数和图像编码数量
        vae_num_layers = vae.num_layers
        num_image_tokens = vae.codebook_size

        # 设置文本相关参数
        self.text_max_seq_len = text_max_seq_len
        self.text_embedding = Embedding(text_num_tokens, dim, frac_gradient = embed_gradient_frac)

        # 根据是否使用旋转位置编码来选择文本绝对位置编码或旋转位置编码
        self.text_abs_pos_emb = Embedding(text_max_seq_len, dim) if not text_rotary_pos_emb else None
        self.text_rotary_pos_emb = RotaryEmbedding(dim = min(32, text_enc_dim_head)) if text_rotary_pos_emb else None

        # 根据是否使用可逆编码器来选择编码器类型
        enc_transformer_klass = Transformer if not enc_reversible else ReversibleTransformer

        # 创建文本变换器
        self.text_transformer = enc_transformer_klass(
            dim = dim,
            depth = text_enc_depth,
            heads = text_enc_heads,
            dim_head = text_enc_dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # 视频相关参数

        # 初始化视频的开始符号
        self.video_bos = nn.Parameter(torch.randn(dim))
        self.image_embedding = Embedding(num_image_tokens, dim, frac_gradient = embed_gradient_frac)

        # 计算特征图大小
        fmap_size = image_size // (2 ** vae_num_layers)

        self.video_fmap_size = fmap_size
        self.max_video_frames = max_video_frames
        video_shape = (max_video_frames, fmap_size, fmap_size)

        # 创建视频位置编码
        self.video_pos_emb = AxialPositionalEmbedding(dim, shape = video_shape)

        # 音频相关参数

        # 初始化音频的开始符号
        self.audio_bos = nn.Parameter(torch.randn(dim))
        self.audio_embedding = Embedding(num_audio_tokens, dim, frac_gradient = embed_gradient_frac)

        # 计算每帧音频序列的最大长度
        max_audio_seq_len = num_audio_tokens_per_video_frame * max_video_frames
        self.audio_pos_emb = AxialPositionalEmbedding(dim, shape = (num_audio_tokens // audio_tokens_per_timestep, audio_tokens_per_timestep))

        self.audio_loss_weight = audio_loss_weight

        # 每帧视频的标记数量

        self.num_video_tokens_per_frame = fmap_size ** 2
        self.num_audio_tokens_per_video_frame = num_audio_tokens_per_video_frame

        # 稀疏3D邻近注意力的循环扩张

        sparse_3dna_dilations = tuple(range(1, sparse_3dna_dilation + 1)) if not isinstance(sparse_3dna_dilation, (list, tuple)) else sparse_3dna_dilation

        sparse_2dna_dilation = tuple(range(1, sparse_2dna_dilation + 1)) if not isinstance(sparse_2dna_dilation, (list, tuple)) else sparse_2dna_dilation

        # 根据是否使用可逆解码器来选择解码器类型
        decoder_klass = ReversibleDualModalityDecoder if dec_reversible else DualModalityDecoder

        # 创建视频音频变换器
        self.video_audio_transformer = decoder_klass(
            dim = dim,
            depth = dec_depth,
            heads = dec_heads,
            dim_head = dec_dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_chunk_size = ff_chunk_size,
            audio_tokens_per_timestep = audio_tokens_per_timestep,
            shift_audio_tokens = shift_audio_tokens,
            shift_video_tokens = shift_video_tokens,
            sparse_3dna_video_shape = video_shape,
            sparse_3dna_kernel_size = sparse_3dna_kernel_size,
            sparse_3dna_dilations = sparse_3dna_dilations,
            sparse_3dna_query_num_frames_chunk = sparse_3dna_query_num_frames_chunk,
            sparse_3dna_rel_pos_bias = sparse_3dna_rel_pos_bias,
            num_audio_tokens_per_video_frame = num_audio_tokens_per_video_frame,
            num_video_tokens_per_frame = fmap_size * fmap_size,
            cross_modality_attn_every = cross_modality_attn_every,
            sparse_2dna_kernel_size = sparse_2dna_kernel_size,
            sparse_2dna_dilation = sparse_2dna_dilation,
            sparse_2dna_rel_pos_bias = sparse_2dna_rel_pos_bias
        )

        # 线性层将维度映射到图像标记数量
        self.to_video_logits = nn.Linear(dim, num_image_tokens, bias = False)
        # 线性层将维度映射到音频标记数量
        self.to_audio_logits = nn.Linear(dim, num_audio_tokens, bias = False)
    # 将文本嵌入到模型中
    def embed_text(self, text, mask = None):
        # 获取文本的批次、序列长度和设备信息
        batch, seq_len, device = *text.shape, text.device
        # 断言文本序列长度不超过预设的最大长度
        assert seq_len <= self.text_max_seq_len, 'your input text has a greater length than what was designated on initialization'

        # 对文本进行嵌入
        tokens = self.text_embedding(text)

        # 如果存在绝对位置嵌入，则添加到嵌入的文本中
        if exists(self.text_abs_pos_emb):
            pos_emb = self.text_abs_pos_emb(torch.arange(seq_len, device = device))
            tokens = tokens + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        # 如果存在旋转位置嵌入，则获取旋转位置嵌入
        if exists(self.text_rotary_pos_emb):
            rotary_pos_emb = self.text_rotary_pos_emb(seq_len, device = device)

        # 返回经过文本变换器处理后的结果
        return self.text_transformer(
            tokens,
            mask = mask,
            rotary_pos_emb = rotary_pos_emb
        )

    # 生成文本
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        *,
        text,
        filter_thres = 0.9,
        temperature = 1.,
        decode_max_batchsize = 10,
        cond_scale = 2.,
        num_frames = None
    # 前向传播
    def forward(
        self,
        *,
        text,
        video,
        audio,
        return_loss = False,
        cond_dropout_prob = 0.2
    ):
        # 获取文本、视频、音频的批次、序列长度、帧数和设备信息
        batch, seq_len, frames, device = *text.shape, video.shape[1], text.device

        # 创建文本的掩码
        text_mask = text != 0
        # 对文本进行嵌入
        text_embeds = self.embed_text(text, mask = text_mask)

        # 准备视频表示

        # 如果视频的数据类型为整数，则直接使用视频帧索引
        if video.dtype == torch.long:
            frame_indices = video
        else:
            # 断言视频帧数与最大视频帧数相同
            assert frames == self.max_video_frames, f'you must give the full video frames ({self.max_video_frames}) during training'
            # 断言存在 VAE 模型
            assert exists(self.vae), 'VAE must be passed in if you wish for video to be encoded to ids automatically'
            # 获取视频帧索引
            frame_indices = self.vae.get_video_indices(video)

        # 重排视频帧索引的维度
        frame_indices = rearrange(frame_indices, 'b ... -> b (...)')
        frame_indices_input = frame_indices[:, :-1] if return_loss else frame_indices

        # 对视频帧进行嵌入
        frame_embeddings = self.image_embedding(frame_indices_input)
        frame_embeddings = self.video_pos_emb()[:-1] + frame_embeddings

        # 在视频帧前添加特殊标记
        video_bos = repeat(self.video_bos, 'd -> b 1 d', b = batch)
        frame_embeddings = torch.cat((video_bos, frame_embeddings), dim = 1)

        # 准备音频表示

        audio_indices_input = audio[:, :-1] if return_loss else audio

        # 对音频进行嵌入
        audio_embeddings = self.audio_embedding(audio_indices_input)
        audio_pos_emb = self.audio_pos_emb()[:audio_embeddings.shape[1]]
        audio_embeddings = audio_embeddings + rearrange(audio_pos_emb, 'n d -> 1 n d')

        # 在音频前添加特殊标记
        audio_bos = repeat(self.audio_bos, 'd -> b 1 d', b = batch)
        audio_embeddings = torch.cat((audio_bos, audio_embeddings), dim = 1)

        # 空条件，用于超级条件

        if self.training and cond_dropout_prob > 0:
            # 随机丢弃条件
            # 参考：https://openreview.net/forum?id=qw8AKxfYbI
            uncond_mask = prob_mask_like((batch,), cond_dropout_prob, device = device)
            text_mask *= rearrange(~uncond_mask, 'b -> b 1')

        # 视频和音频的双重注意力塔，具有高效的分块跨模态注意力

        frame_embeddings, audio_embeddings = self.video_audio_transformer(
            frame_embeddings,
            audio_embeddings,
            context = text_embeds,
            context_mask = text_mask
        )

        # 获取视频和音频的逻辑回归结果
        video_logits = self.to_video_logits(frame_embeddings)
        audio_logits = self.to_audio_logits(audio_embeddings)

        # 如果不需要计算损失，则直接返回逻辑回归结果
        if not return_loss:
            return video_logits, audio_logits

        # 计算视频和音频的损失
        video_loss = F.cross_entropy(rearrange(video_logits, 'b n c -> b c n'), frame_indices)
        audio_loss = F.cross_entropy(rearrange(audio_logits, 'b n c -> b c n'), audio)

        # 返回视频和音频的损失之和
        return video_loss + audio_loss * self.audio_loss_weight
# 主要用于学习素描的主类

class NUWASketch(nn.Module):
    def __init__(
        self,
        *,
        vae,  # VAE 模型
        sketch_vae,  # 素描 VAE 模型
        dim,  # 维度
        image_size,  # 图像大小
        max_video_frames = 5,  # 最大视频帧数
        sketch_max_video_frames = 2,  # 素描最大视频帧数
        sketch_enc_depth = 6,  # 素描编码器深度
        sketch_enc_dim_head = 64,  # 素描编码器头维度
        sketch_enc_heads = 8,  # 素描编码器头数
        sketch_enc_use_sparse_3dna = False,  # 是否使用稀疏 3DNA
        enc_reversible = False,  # 编码器是否可逆
        dec_depth = 6,  # 解码器深度
        dec_dim_head = 64,  # 解码器头维度
        dec_heads = 8,  # 解码器头数
        dec_reversible = False,  # 解码器是否可逆
        attn_dropout = 0.,  # 注意力机制的 dropout
        ff_dropout = 0.,  # FeedForward 层的 dropout
        ff_chunk_size = None,  # FeedForward 层的块大小
        embed_gradient_frac = 0.2,  # 嵌入梯度比例
        shift_video_tokens = True,  # 是否移动视频 token
        cross_2dna_kernel_size = 3,  # 交叉 2DNA 的卷积核大小
        cross_2dna_dilation = 1,  # 交叉 2DNA 的膨胀率
        sparse_3dna_kernel_size = 3,  # 稀疏 3DNA 的卷积核大小
        sparse_3dna_dilation = 1,  # 稀疏 3DNA 的膨胀率
        sparse_3dna_query_num_frames_chunk = None,  # 稀疏 3DNA 查询的帧块数
        ):
        # 调用父类的构造函数
        super().__init__()
        # 设置图像大小
        self.image_size = image_size

        # 设置sketch_vae属性
        self.sketch_vae = sketch_vae
        # 获取sketch_vae的层数
        sketch_vae_num_layers = sketch_vae.num_layers
        # 获取sketch_vae的编码本大小
        sketch_num_image_tokens = sketch_vae.codebook_size
        # 计算sketch的特征图大小
        sketch_fmap_size = image_size // (2 ** sketch_vae_num_layers)

        # 定义sketch的形状
        sketch_shape = (sketch_max_video_frames, sketch_fmap_size, sketch_fmap_size)

        # 设置sketch_max_video_frames属性
        self.sketch_max_video_frames = sketch_max_video_frames
        # 创建sketch的嵌入层
        self.sketch_embedding = Embedding(sketch_num_image_tokens, dim, frac_gradient = embed_gradient_frac)
        # 创建sketch的位置嵌入
        self.sketch_pos_emb = AxialPositionalEmbedding(dim, shape = sketch_shape)

        # sparse 3dna kwargs

        # 设置稀疏3dna的膨胀
        sparse_3dna_dilations = tuple(range(1, sparse_3dna_dilation + 1)) if not isinstance(sparse_3dna_dilation, (list, tuple)) else sparse_3dna_dilation

        # encoder

        # 根据enc_reversible选择不同的Transformer类
        enc_transformer_klass = Transformer if not enc_reversible else ReversibleTransformer

        # 创建sketch_transformer
        self.sketch_transformer = enc_transformer_klass(
            dim = dim,
            depth = sketch_enc_depth,
            heads = sketch_enc_heads,
            dim_head = sketch_enc_dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            shift_video_tokens = shift_video_tokens,
            sparse_3dna_video_shape = sketch_shape,
            sparse_3dna_kernel_size = sparse_3dna_kernel_size,
            sparse_3dna_dilations = sparse_3dna_dilations,
            sparse_3dna_query_num_frames_chunk = sparse_3dna_query_num_frames_chunk,
            sparse_3dna_attn = sketch_enc_use_sparse_3dna
        )

        # decoder parameters

        # 复制vae用于评估
        self.vae = vae.copy_for_eval()

        # 获取vae的层数和编码本大小
        vae_num_layers = vae.num_layers
        num_image_tokens = vae.codebook_size

        # 创建video_bos参数
        self.video_bos = nn.Parameter(torch.randn(dim))
        # 创建图像嵌入层
        self.image_embedding = Embedding(num_image_tokens, dim, frac_gradient = embed_gradient_frac)

        # 计算特征图大小
        fmap_size = image_size // (2 ** vae_num_layers)

        # 断言特征图大小相等
        assert fmap_size == sketch_fmap_size, 'feature map size of video must be equal to the feature map size of sketches (VAEs must have same number of layers)'

        # 设置video_fmap_size属性
        self.video_fmap_size = fmap_size
        # 设置最大视频帧数
        self.max_video_frames = max_video_frames
        # 定义video的形状
        video_shape = (max_video_frames, fmap_size, fmap_size)

        # 创建video的位置嵌入
        self.video_pos_emb = AxialPositionalEmbedding(dim, shape = video_shape)

        # cycle dilation for sparse 3d-nearby attention

        # 设置cross_2dna_dilations
        cross_2dna_dilations = tuple(range(1, cross_2dna_dilation + 1)) if not isinstance(cross_2dna_dilation, (list, tuple)) else cross_2dna_dilation
        # 根据dec_reversible选择不同的Transformer类
        dec_transformer_klass = Transformer if not dec_reversible else ReversibleTransformer

        # 创建video_transformer
        self.video_transformer = dec_transformer_klass(
            dim = dim,
            depth = dec_depth,
            heads = dec_heads,
            dim_head = dec_dim_head,
            causal = True,
            cross_attend = True,
            cross_2dna_attn = True,
            cross_2dna_image_size = fmap_size,
            cross_2dna_kernel_size = cross_2dna_kernel_size,
            cross_2dna_dilations = cross_2dna_dilations,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_chunk_size = ff_chunk_size,
            shift_video_tokens = shift_video_tokens,
            sparse_3dna_video_shape = video_shape,
            sparse_3dna_kernel_size = sparse_3dna_kernel_size,
            sparse_3dna_dilations = sparse_3dna_dilations,
            sparse_3dna_query_num_frames_chunk = sparse_3dna_query_num_frames_chunk,
            sparse_3dna_attn = True
        )

        # 创建输出层
        self.to_logits = nn.Linear(dim, num_image_tokens, bias = False)
    # 将 sketch 的形状解构为 batch, frames, channels, image_size, _，并获取设备信息
    batch, frames, channels, image_size, _, device = *sketch.shape, sketch.device

    # 如果存在 mask，则确保 mask 的形状为 (batch, frames)
    if exists(mask):
        assert mask.shape[:2] == (batch, frames), 'sketch mask must be in shape of (batch x frame)'

    # 获取 sketch 的索引
    sketch_indices = self.sketch_vae.get_video_indices(sketch)
    # 重新排列 sketch_indices 的形状
    sketch_indices = rearrange(sketch_indices, 'b ... -> b (...)')

    # 使用 sketch_indices 获取 sketch_tokens
    sketch_tokens = self.sketch_embedding(sketch_indices)

    # 获取 sketch_tokens 的数量
    num_tokens = sketch_tokens.shape[1]

    # 获取 sketch 的位置编码
    sketch_pos_emb = self.sketch_pos_emb()
    sketch_pos_emb = sketch_pos_emb[:num_tokens]

    # 将 sketch_tokens 与 sketch_pos_emb 相加
    sketch_tokens = sketch_tokens + sketch_pos_emb

    # 如果存在 mask，则重复 mask，使其形状为 (batch, num_tokens)
    if exists(mask):
        mask = repeat(mask, 'b f -> b (f n)', n = (num_tokens // frames)
    else:
        # 如果不存在 mask，则创建全为 True 的 mask
        mask = torch.ones((batch, num_tokens), dtype = torch.bool, device = device)

    # 使用 sketch_transformer 对 sketch_tokens 进行嵌入
    embed = self.sketch_transformer(sketch_tokens, mask = mask)
    return embed, mask

@torch.no_grad()
@eval_decorator
def generate(
    self,
    *,
    sketch,
    sketch_mask = None,
    filter_thres = 0.9,
    temperature = 1.,
    decode_max_batchsize = 10,
    cond_scale = 2.,
    num_frames = None
    # 获取批次大小和设备信息
    batch, device = sketch.shape[0], sketch.device

    # 对草图进行嵌入处理，并生成解码器上下文掩码
    sketch_embeds, decoder_context_mask = self.embed_sketch(sketch, mask = sketch_mask)

    # 创建起始符号
    bos = repeat(self.video_bos, 'd -> b 1 d', b = batch)

    # 创建空的视频索引张量
    video_indices = torch.empty((batch, 0), device = device, dtype = torch.long)

    # 计算每帧的标记数量
    num_tokens_per_frame = self.video_fmap_size ** 2

    # 设置视频帧数和总标记数量
    num_frames = default(num_frames, self.max_video_frames)
    total_video_tokens =  num_tokens_per_frame * num_frames
    max_video_tokens = num_tokens_per_frame * self.max_video_frames

    # 获取位置编码
    pos_emb = self.video_pos_emb()

    # 遍历视频标记
    for ind in tqdm(range(total_video_tokens)):
        # 复制视频索引输入
        video_indices_input = video_indices

        # 获取当前视频标记数量
        num_video_tokens = video_indices.shape[1]
        if num_video_tokens > max_video_tokens:
            # 计算回溯标记数量
            curr_frame_tokens = num_video_tokens % num_tokens_per_frame
            lookback_tokens = (self.max_video_frames - (0 if curr_frame_tokens == 0 else 1)) * num_tokens_per_frame + curr_frame_tokens
            video_indices_input = video_indices[:, -lookback_tokens:]

        # 获取帧嵌入
        frame_embeddings = self.image_embedding(video_indices_input)
        frame_embeddings = pos_emb[:frame_embeddings.shape[1]] + frame_embeddings
        frame_embeddings = torch.cat((bos, frame_embeddings), dim = 1)

        # 使用视频变换器处理帧嵌入
        frame_embeddings = self.video_transformer(
            frame_embeddings,
            context = sketch_embeds,
            context_mask = decoder_context_mask
        )

        # 获取逻辑回归结果
        logits = self.to_logits(frame_embeddings)

        if cond_scale != 1:
            # 根据条件比例对逻辑回归结果进行调整
            uncond_frame_embeddings = self.video_transformer(
                frame_embeddings,
                context = sketch_embeds,
                context_mask = torch.zeros_like(decoder_context_mask).bool()
            )

            uncond_logits = self.to_logits(uncond_frame_embeddings)
            logits = uncond_logits + (logits - uncond_logits) * cond_scale

        logits = logits[:, -1, :]

        # 过滤逻辑回归结果并进行采样
        filtered_logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
        sample = rearrange(sample, 'b -> b 1')
        video_indices = torch.cat((video_indices, sample), dim = 1)

    # 获取代码本和重构图像
    codes = self.vae.codebook[video_indices]
    codes = rearrange(codes, 'b (f h w) d -> (b f) d h w', h = self.video_fmap_size, w = self.video_fmap_size)

    image_reconstructions = batch_process(codes, self.vae.decode, chunks = decode_max_batchsize)
    video = rearrange(image_reconstructions, '(b f) d h w -> b f d h w', b = batch)
    return video

# 定义前向传播函数
def forward(
    self,
    *,
    sketch,
    sketch_mask = None,
    video = None,
    return_loss = False,
    cond_dropout_prob = 0.2
        # 处理一个草图的过程

        # 如果草图的维度为4，则重新排列成 'b c h w -> b 1 c h w'
        if sketch.ndim == 4:
            sketch = rearrange(sketch, 'b c h w -> b 1 c h w')

        # 获取一系列变量

        # 解包sketch的形状，得到batch, sketch_frames, sketch_channels, sketch_image_size, _, frames, device
        batch, sketch_frames, sketch_channels, sketch_image_size, _, frames, device = *sketch.shape, video.shape[1], sketch.device

        # 断言

        # 断言sketch_image_size必须等于self.image_size
        assert sketch_image_size == self.image_size, 'sketch image size must be equal'
        # 断言sketch_frames必须小于等于self.sketch_max_video_frames
        assert sketch_frames <= self.sketch_max_video_frames, 'sketch frames must be less than max sketch video frames'

        # 获取草图嵌入和计算掩码（暂时假设没有填充）

        # 获取草图嵌入和解码器上下文掩码
        sketch_embeds, decoder_context_mask = self.embed_sketch(sketch, mask=sketch_mask)

        # 断言

        # 断言frames必须等于self.max_video_frames
        assert frames == self.max_video_frames, f'you must give the full video frames ({self.max_video_frames}) during training'

        # 获取视频帧索引
        frame_indices = self.vae.get_video_indices(video)
        # 重新排列帧索引的形状，变为 'b ... -> b (...)'
        frame_indices = rearrange(frame_indices, 'b ... -> b (...)')
        # 如果不需要返回损失，则将帧索引输入设为frame_indices的前n-1个元素
        frame_indices_input = frame_indices[:, :-1] if not return_loss else frame_indices

        # 获取帧嵌入
        frame_embeddings = self.image_embedding(frame_indices_input)
        # 添加视频位置编码
        frame_embeddings = self.video_pos_emb()[:-1] + frame_embeddings

        # 在帧嵌入前添加开始标记
        bos = repeat(self.video_bos, 'd -> b 1 d', b=batch)
        frame_embeddings = torch.cat((bos, frame_embeddings), dim=1)

        # 如果处于训练状态且cond_dropout_prob大于0
        if self.training and cond_dropout_prob > 0:
            # 随机丢弃条件
            # 参考：https://openreview.net/forum?id=qw8AKxfYbI
            uncond_mask = prob_mask_like((batch,), cond_dropout_prob, device=device)
            sketch_mask *= rearrange(~uncond_mask, 'b -> b 1')

        # 使用视频变换器处理帧嵌入
        frame_embeddings = self.video_transformer(
            frame_embeddings,
            context=sketch_embeds,
            context_mask=decoder_context_mask
        )

        # 获取logits
        logits = self.to_logits(frame_embeddings)

        # 如果不需要返回损失，则返回logits
        if not return_loss:
            return logits

        # 计算损失
        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), frame_indices)
        return loss
```