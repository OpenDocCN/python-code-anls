# `.\lucidrains\musiclm-pytorch\musiclm_pytorch\musiclm_pytorch.py`

```
# 导入数学库
import math
# 从 functools 库中导入 wraps 和 partial 函数
from functools import wraps, partial

# 导入 torch 库
import torch
# 从 torch.nn.functional 模块中导入 F 函数
import torch.nn.functional as F
# 从 torch 模块中导入 nn 和 einsum 函数
from torch import nn, einsum

# 从 torchaudio.transforms 模块中导入 Spectrogram, TimeStretch, FrequencyMasking, TimeMasking 函数
from torchaudio.transforms import Spectrogram, TimeStretch, FrequencyMasking, TimeMasking

# 从 audiolm_pytorch 库中导入 AudioLM 类和 AudioConditionerBase 类
from audiolm_pytorch import AudioLM
from audiolm_pytorch.utils import AudioConditionerBase

# 从 torch.distributed 模块中导入 dist 函数
import torch.distributed as dist
# 从 musiclm_pytorch.distributed 模块中导入 AllGather 函数
from musiclm_pytorch.distributed import AllGather

# 从 x_clip.tokenizer 模块中导入 tokenizer 函数
from x_clip.tokenizer import tokenizer
# 从 vector_quantize_pytorch 库中导入 ResidualVQ 类
from vector_quantize_pytorch import ResidualVQ

# 从 einops 模块中导入 rearrange, repeat, reduce, pack, unpack 函数
from einops import rearrange, repeat, reduce, pack, unpack
# 从 einops.layers.torch 模块中导入 Rearrange 函数
from einops.layers.torch import Rearrange

# 从 beartype.typing 模块中导入 List, Optional, Tuple 类
# 从 beartype 模块中导入 beartype 函数
from beartype.typing import List, Optional, Tuple
from beartype import beartype

# functions

# 定义函数 exists，判断值是否存在
def exists(val):
    return val is not None

# 定义函数 first，返回列表的第一个元素
def first(it):
    return it[0]

# 定义函数 default，如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义函数 round_down_nearest_multiple，返回最接近的整数倍数
def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor

# 定义函数 Sequential，返回过滤掉空值后的 nn.Sequential 对象
def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# decorators

# 定义装饰器 once，确保函数只调用一次
def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

# 使用 once 装饰 print 函数，确保只打印一次
print_once = once(print)

# tensor functions

# 定义函数 log，计算张量的对数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 定义函数 l2norm，计算张量的 L2 范数
def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

# 定义函数 matrix_diag，返回张量的对角线元素
def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# 2d sinusoidal positional embedding
# simple vit paper shows it is good enough compared to learned

# 定义函数 posemb_sincos_2d，生成二维正弦余弦位置嵌入
def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    pe = pe.type(dtype)

    return rearrange(pe, '(h w) d -> h w d', h = h, w = w)

# biasless layernorm

# 定义类 LayerNorm，实现无偏差的 LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, dim, scale = True):
        super().__init__()
        self.learned_gamma = nn.Parameter(torch.ones(dim)) if scale else None

        self.register_buffer('gamma', torch.ones(dim), persistent = False)
        self.register_buffer('beta', torch.zeros(dim), persistent = False)

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], default(self.learned_gamma, self.gamma), self.beta)

# feedforward

# 定义类 GEGLU，实现 GEGLU 激活函数
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

# 定义函数 FeedForward，实现前馈神经网络
def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim, bias = False)
    )

# attention

# 定义类 Attention，实现注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        scale = 8
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化头数和缩放比例
        self.heads = heads
        self.scale = scale
        self.causal = causal
        # 计算每个头的内部维度
        inner_dim = dim_head * heads

        # 初始化 LayerNorm 层
        self.norm = LayerNorm(dim)

        # 初始化注意力机制的 dropout 层
        self.attn_dropout = nn.Dropout(dropout)

        # 初始化查询、键、值的线性变换层
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        # 初始化查询和键的缩放参数
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        # 初始化输出层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        # 获取输入张量 x 的形状和设备信息
        b, n, _, device = *x.shape, x.device

        # 对输入进行 LayerNorm 处理
        x = self.norm(x)

        # 对输入进行查询、键、值的线性变换
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        # 将查询、键、值分割为多头注意力
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # 对查询和键进行 L2 归一化
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # 计算相似度
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # 如果存在相对位置偏置，则加上
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # 如果存在掩码，则进行掩码处理
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # ���果启用因果注意力，则进行因果掩码处理
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 注意力权重计算
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # 聚合
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 合并多头
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# transformer

# 定义 Transformer 类，用于实现 Transformer 模型
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        # 循环创建 Transformer 层，并添加到 layers 中
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout),
            ]))

    # 前向传播函数
    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None,
        return_all_layers = False
    ):
        layers = []

        # 遍历 Transformer 层，依次进行注意力计算和前馈计算
        for attn, ff in self.layers:
            x = attn(x, rel_pos_bias = rel_pos_bias, mask = mask) + x
            x = ff(x) + x
            layers.append(x)

        # 如果不需要返回所有层的结果，则返回最后一层的结果
        if not return_all_layers:
            return x

        # 返回所有层的结果
        return x, torch.stack(layers[:-1])

# contrastive losses

# 定义 SoftmaxContrastiveLearning 类，用于实现 Softmax 对比学习
class SoftmaxContrastiveLearning(nn.Module):
    def __init__(
        self,
        *,
        layers = 1,
        decoupled_contrastive_learning = False,
        init_temp = 10
    ):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        self.all_gather = AllGather(dim = 2)

    # 获取设备信息
    @property
    def device(self):
        return next(self.parameters()).device

    # 前向传播函数
    def forward(self, audio_latents, text_latents):
        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')

        batch = audio_latents.shape[1]

        # 分布式环境下，进行数据分发
        if self.all_gather.is_distributed:
            latents = torch.stack((audio_latents, text_latents))
            latents, _ = self.all_gather(latents)
            audio_latents, text_latents = latents

        # 计算相似度矩阵
        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)

        sims = sims * self.temperatures.exp()

        cosine_sims_exp = sims.exp()

        numerator = matrix_diag(cosine_sims_exp)

        # 如果使用分离式对比学习，进行额外处理
        if self.decoupled_contrastive_learning:
            eye = torch.eye(batch, device = self.device, dtype = torch.bool)
            cosine_sims_exp = cosine_sims_exp.masked_fill(eye, 0.)

        denominator_i = reduce(cosine_sims_exp, 'l i j -> l i', 'sum')
        denominator_j = reduce(cosine_sims_exp, 'l i j -> l j', 'sum')

        contrastive_loss = -log(numerator) + 0.5 * (log(denominator_i) + log(denominator_j))

        contrastive_loss = reduce(contrastive_loss, 'l n -> l', 'mean')
        return contrastive_loss.sum()

# 定义 SigmoidContrastiveLearning 类，用于实现 Sigmoid 对比学习
class SigmoidContrastiveLearning(nn.Module):
    """ https://arxiv.org/abs/2303.15343 """

    def __init__(
        self,
        *,
        layers = 1,
        init_temp = 10,
        init_bias = -10
    ):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.bias = nn.Parameter(torch.ones(layers, 1, 1) * init_bias)

        self.all_gather = AllGather(dim = 1, all_reduce_grads = True)

    # 获取设备信息
    @property
    def device(self):
        return next(self.parameters()).device
    # 定义一个前向传播函数，接受音频和文本的潜在表示作为输入
    def forward(self, audio_latents, text_latents):
        # 获取当前设备
        device = self.device

        # 如果音频潜在表示的维度为2，则重新排列为 '... -> 1 ...'
        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        # 如果文本潜在表示的维度为2，则重新排列为 '... -> 1 ...'
        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')

        # 使用all_gather函数将文本潜在表示广播到所有设备上，并返回广播后的结果和每个设备上的大小
        text_latents, rank_sizes = self.all_gather(text_latents)

        # 获取文本潜在表示的第二维大小
        n = text_latents.shape[1]

        # 计算音频潜在表示和文本潜在表示之间的相似度
        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)

        # 对相似度进行温度调节和偏置处理
        sims = sims * self.temperatures.exp() + self.bias

        # 创建一个对角线为1的标签矩阵
        labels = torch.eye(n, device=device)

        # 如果rank_sizes存在，则根据rank_sizes将标签拆分为不同的部分
        if exists(rank_sizes):
            labels_by_ranks = labels.split(rank_sizes.tolist(), dim=0)
            labels = labels_by_ranks[dist.get_rank()]

        # 将标签矩阵重新排列为 'i j -> 1 i j'，并进行处理得到最终的标签
        labels = 2 * rearrange(labels, 'i j -> 1 i j') - torch.ones_like(sims)

        # 计算损失函数，返回负对数sigmoid损失的总和除以n
        return -F.logsigmoid(labels * sims).sum() / n
# Audio Spectrogram Transformer - https://arxiv.org/abs/2104.01778

# 定义一个函数，用于将输入转换为元组
def pair(t):
    return (t, t) if not isinstance(t, tuple) else t

# 定义一个音频频谱变换器类
class AudioSpectrogramTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        patch_size = 16,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        accept_spec = False,
        accept_spec_time_first = True,
        spec_n_fft = 128,
        spec_power = 2,
        spec_win_length = 24,
        spec_hop_length = None,
        spec_pad = 0,
        spec_center = True,
        spec_pad_mode = 'reflect',
        spec_aug_stretch_factor = 0.8,
        spec_aug_freq_mask = 80,
        spec_aug_time_mask = 80,
        patch_dropout_prob = 0.25
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.patch_size = pair(patch_size)
        patch_input_dim = self.patch_size[0] * self.patch_size[1]

        # 将输入转换为补丁令牌
        self.to_patch_tokens = Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1 = self.patch_size[0], p2 = self.patch_size[1]),
            nn.LayerNorm(patch_input_dim),
            nn.Linear(patch_input_dim, dim),
            nn.LayerNorm(dim)
        )

        self.accept_spec = accept_spec
        self.accept_spec_time_first = accept_spec_time_first

        # 创建频谱对象
        self.spec = Spectrogram(
            n_fft = spec_n_fft,
            power = spec_power,
            win_length = spec_win_length,
            hop_length = spec_hop_length,
            pad = spec_pad,
            center = spec_center,
            pad_mode = spec_pad_mode
        )

        # SpecAugment - 在音频领域中被广泛使用
        self.aug = torch.nn.Sequential(
            TimeStretch(spec_aug_stretch_factor, fixed_rate = True),
            FrequencyMasking(freq_mask_param = spec_aug_freq_mask),
            TimeMasking(time_mask_param = spec_aug_time_mask),
        )

        # 创建变换器
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout
        )

        self.norm = LayerNorm(dim)

        # 补丁丢弃概率
        self.patch_dropout_prob = patch_dropout_prob

        # 2D动态位置偏差
        mlp_hidden_dim = dim // 4

        self.dynamic_pos_bias_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    def forward(
        self,
        x,
        force_no_patch_dropout = False,
        return_all_layers = False
        ):
        # 获取输入张量的批次大小和设备信息
        batch, device = x.shape[0], x.device
        # 断言输入张量的维度是否符合要求
        assert (self.accept_spec and x.ndim == 3) or (not self.accept_spec and x.ndim == 2)

        if self.accept_spec and self.accept_spec_time_first:
            # 如果接受频谱数据且要求时间维度在前，则重新排列输入张量的维度
            x = rearrange(x, 'b t f -> b f t')

        if not self.accept_spec:
            # 如果不接受频谱数据，则对输入进行频谱转换
            x = self.spec(x)

        if self.training:
            # 如果处于训练模式，则对输入进行数据增强
            x = self.aug(x)

        # 如果音频生成的二维频谱图不是patch大小的整数倍，则自动裁剪
        height, width = x.shape[-2:]
        patch_height, patch_width = self.patch_size

        rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args), ((height, patch_height), (width, patch_width)))

        if (height, width) != (rounded_height, rounded_width): # 只是持续打印直到修复
            print_once(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        x = x[..., :rounded_height, :rounded_width]

        # 转换为patches
        x = self.to_patch_tokens(x)

        # 获取沿高度和宽度的patch数量
        _, num_patch_height, num_patch_width, _ = x.shape

        # 获取2D相对位置
        grid = torch.stack(torch.meshgrid(
            torch.arange(num_patch_height, device = device),
            torch.arange(num_patch_width, device = device)
        , indexing = 'ij'), dim = -1)

        grid = rearrange(grid, '... c -> (...) c')

        # 2D正弦余弦位置嵌入
        x = x + posemb_sincos_2d(x)

        x = rearrange(x, 'b ... c -> b (...) c')

        # patch丢弃
        if self.training and self.patch_dropout_prob > 0. and not force_no_patch_dropout:
            n, device = x.shape[1], x.device

            batch_indices = torch.arange(batch, device = device)
            batch_indices = rearrange(batch_indices, '... -> ... 1')
            num_patches_keep = max(1, int(n * (1 - self.patch_dropout_prob)))
            patch_indices_keep = torch.randn(batch, n, device = device).topk(num_patches_keep, dim = -1).indices

            x = x[batch_indices, patch_indices_keep]

            grid = repeat(grid, '... -> b ...', b = batch)
            grid = grid[batch_indices, patch_indices_keep]

        # 2D相对位置偏差
        rel_dist = rearrange(grid, '... i c -> ... i 1 c') - rearrange(grid, '... j c -> ... 1 j c')
        rel_pos_bias = self.dynamic_pos_bias_mlp(rel_dist.float())

        # 注意力机制
        x, all_layers = self.transformer(x, rel_pos_bias = rel_pos_bias, return_all_layers = True)

        # 最终全局平均和规范化（最近的论文表明这比CLS token更优越）
        x = reduce(x, 'b n d -> b d', 'mean')

        out = self.norm(x)

        if not return_all_layers:
            return out

        return out, all_layers
# 文本转换器类
class TextTransformer(nn.Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        dim,  # 维度
        depth,  # 深度
        num_tokens = tokenizer.vocab_size,  # 标记数量，默认为tokenizer的词汇量
        max_seq_len = 256,  # 最大序列长度，默认为256
        dim_head = 64,  # 头部维度，默认为64
        heads = 8,  # 头部数量，默认为8
        attn_dropout = 0.,  # 注意力丢弃率，默认为0
        ff_dropout = 0.,  # 前馈丢弃率，默认为0
        ff_mult = 4,  # 前馈倍数，默认为4
        pad_id = 0  # 填充标记ID，默认为0
    ):
        super().__init__()
        self.dim = dim  # 维度

        self.token_emb = nn.Embedding(num_tokens, dim)  # 标记嵌入层
        self.pos_emb = nn.Embedding(max_seq_len, dim)  # 位置嵌入层

        self.depth = depth  # 深度
        self.max_seq_len = max_seq_len  # 最大序列长度

        self.cls_token = nn.Parameter(torch.randn(dim))  # 类别标记

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )  # 转换器模型

        self.pad_id = pad_id  # 填充标记ID
        self.norm = LayerNorm(dim)  # 归一化层

    # 设备属性
    @property
    def device(self):
        return next(self.parameters()).device

    # 前向传播函数
    @beartype
    def forward(
        self,
        x = None,  # 输入张量，默认为None
        raw_texts: Optional[List[str]] = None,  # 原始文本列表，默认为None
        mask = None,  # 掩码，默认为None
        return_all_layers = False  # 是否返回所有层，默认为False
    ):
        assert exists(x) ^ exists(raw_texts)  # 断言，x和raw_texts必须有且只有一个存在

        if exists(raw_texts):
            x = tokenizer.tokenize(raw_texts).to(self.device)  # 使用tokenizer对原始文本进行标记化，并转移到指定设备

        if not exists(mask):
            mask = x != self.pad_id  # 生成掩码，排除填充标记

        b, n, device = *x.shape, x.device  # 获取张量形状和设备信息

        # 标记嵌入 + 位置嵌入
        x = self.token_emb(x)  # 标记嵌入
        assert n <= self.max_seq_len, f'text sequence length {n} must be less than {self.max_seq_len}'  # 断言，文本序列长度必须小于等于最大序列长度
        x = x + self.pos_emb(torch.arange(n, device = device))  # 加上位置嵌入

        # 类别标记，类似于bert
        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)  # 重复类别标记
        x, ps = pack([cls_tokens, x], 'b * d')  # 打包张量

        # 考虑使用自注意力掩码对类别标记进行注意力
        mask = F.pad(mask, (1, 0), value = True)  # 对掩码进行填充

        # 注意力
        x, all_layers = self.transformer(x, mask = mask, return_all_layers = True)  # 使用transformer进行注意力计算

        # 解包类别标记
        cls_tokens, _ = unpack(x, ps, 'b * d')  # 解包张量

        out = self.norm(cls_tokens)  # 归一化类别标记

        if not return_all_layers:
            return out  # 返回输出

        return out, all_layers  # 返回输出和所有层

# 分层对比损失
def interspersed_indices(layers, total_layers):
    assert total_layers >= layers  # 断言，总层数必须大于等于层数
    step = total_layers / layers  # 计算步长
    return (torch.arange(0, layers) * step).floor().long()  # 返回分散的索引

# 多层对比损失类
class MultiLayerContrastiveLoss(nn.Module):
    def __init__(
        self,
        *,
        audio_dim,  # 音频维度
        text_dim,  # 文本维度
        dim_latent,  # 潜在维度
        layers,  # 层数
        decoupled_contrastive_learning = False,  # 是否解耦对比学习，默认为False
        sigmoid_contrastive_loss = False  # 是否使用sigmoid对比损失，默认为False
    ):
        super().__init__()
        self.layers = layers  # 层数

        self.audio_norm = LayerNorm(audio_dim, scale = False)  # 音频归一化层
        self.audio_gamma = nn.Parameter(torch.ones(layers, 1, audio_dim))  # 音频gamma参数
        self.audio_latent_weight = nn.Parameter(torch.randn(layers, audio_dim, dim_latent))  # 音频潜在权重
        self.audio_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))  # 音频潜在偏置

        self.text_norm = LayerNorm(text_dim, scale = False)  # 文本归一化层
        self.text_gamma = nn.Parameter(torch.ones(layers, 1, text_dim))  # 文本gamma参数
        self.text_latent_weight = nn.Parameter(torch.randn(layers, text_dim, dim_latent))  # 文本潜在权重
        self.text_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))  # 文本潜在偏置

        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning = decoupled_contrastive_learning)  # 根据sigmoid_contrastive_loss选择对比学习类
        self.contrast = klass(layers = layers)  # 对比学习实例化
    # 定义一个前向传播函数，接收音频和文本的特征层作为参数
    def forward(self, *, audio_layers, text_layers):
        # 获取设备和批次大小
        device, batch = audio_layers.device, audio_layers.shape[1]

        # 对音频特征层进行降维处理，计算平均值
        audio_gap = reduce(audio_layers, 'l b n d -> l b d', 'mean')
        # 对音频特征进行归一化处理，并乘以音频的缩放参数
        audio_embeds = self.audio_norm(audio_gap) * self.audio_gamma
        # 使用音频的权重和偏置计算音频的潜在特征
        audio_latents = einsum('l b d, l d e -> l b e', audio_embeds, self.audio_latent_weight) + self.audio_latent_bias
        # 对音频的潜在特征进行L2范数归一化处理
        audio_latents = l2norm(audio_latents)

        # 获取文本特征层中的分类标记
        text_cls_tokens = text_layers[:, :, 0]
        # 对文本特征进行归一化处理，并乘以文本的缩放参数
        text_embeds = self.text_norm(text_cls_tokens) * self.text_gamma
        # 使用文本的权重和偏置计算文本的潜在特征
        text_latents = einsum('l b d, l d e -> l b e', text_embeds, self.text_latent_weight) + self.text_latent_bias
        # 对文本的潜在特征进行L2范数归一化处理
        text_latents = l2norm(text_latents)

        # 返回音频和文本潜在特征的对比结果
        return self.contrast(audio_latents, text_latents)
# 主要类

class MuLaN(nn.Module):
    # 初始化 MuLaN 类
    @beartype
    def __init__(
        self,
        audio_transformer: AudioSpectrogramTransformer,
        text_transformer: TextTransformer,
        dim_latent = 128,                       # 设置默认 latent 维度为 128
        decoupled_contrastive_learning = True,  # 是否使用 decoupled 对比学习，默认为 True
        hierarchical_contrastive_loss = False,  # 是否使用 hierarchical 对比损失，默认为 False
        hierarchical_contrastive_loss_layers = None,  # hierarchical 对比损失的层数，默认为 None
        sigmoid_contrastive_loss = False  # 是否使用 sigmoid 对比损失，默认为 False
    ):
        super().__init__()
        self.dim_latent = dim_latent

        self.audio = audio_transformer
        self.text = text_transformer

        # 将文本转换为 latent 向量
        self.text_to_latents = nn.Linear(self.text.dim, dim_latent)
        # 将音频转换为 latent 向量
        self.audio_to_latents = nn.Linear(self.audio.dim, dim_latent)

        # 根据 sigmoid_contrastive_loss 的值选择对比学习方法
        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning = decoupled_contrastive_learning)
        self.contrast = klass()

        self.multi_layer_contrastive_learning = None

        # 如果启用 hierarchical 对比损失
        if hierarchical_contrastive_loss:
            # 计算层数
            num_layers = default(hierarchical_contrastive_loss_layers, min(audio_transformer.depth, text_transformer.depth) - 1)
            assert num_layers > 0

            # 注册文本层索引和音频层索引
            self.register_buffer('text_layers_indices', interspersed_indices(num_layers, text_transformer.depth))
            self.register_buffer('audio_layers_indices', interspersed_indices(num_layers, audio_transformer.depth))

            # 初始化多层对比损失
            self.multi_layer_contrastive_learning = MultiLayerContrastiveLoss(
                audio_dim = self.audio.dim,
                text_dim = self.text.dim,
                dim_latent = dim_latent,
                layers = num_layers,
                decoupled_contrastive_learning = decoupled_contrastive_learning,
                sigmoid_contrastive_loss = sigmoid_contrastive_loss
            )

    # 获取音频 latent 向量
    def get_audio_latents(
        self,
        wavs,
        return_all_layers = False
    ):
        # 获取音频嵌入和层信息
        audio_embeds, audio_layers = self.audio(wavs, return_all_layers = True)
        audio_latents = self.audio_to_latents(audio_embeds)
        out = l2norm(audio_latents)

        if not return_all_layers:
            return out

        return out, audio_layers

    # 获取文本 latent 向量
    @beartype
    def get_text_latents(
        self,
        texts = None,
        raw_texts: Optional[List[str]] = None,
        return_all_layers = False
    ):
        # 获取文本嵌入和层信息
        text_embeds, text_layers = self.text(texts, raw_texts = raw_texts, return_all_layers = True)
        text_latents = self.text_to_latents(text_embeds)
        out = l2norm(text_latents)

        if not return_all_layers:
            return out

        return out, text_layers

    # MuLaN 类的前向传播函数
    @beartype
    def forward(
        self,
        wavs,
        texts = None,
        raw_texts: Optional[List[str]] = None,
        return_latents = False,
        return_similarities = False,
        return_pairwise_similarities = False
        # 获取输入张量的批次大小和设备信息
        batch, device = wavs.shape[0], wavs.device

        # 获取音频的潜在空间表示和层表示
        audio_latents, audio_layers = self.get_audio_latents(wavs, return_all_layers=True)
        
        # 获取文本的潜在空间表示和层表示
        text_latents, text_layers = self.get_text_latents(texts, raw_texts=raw_texts, return_all_layers=True)

        # 如果需要返回潜在空间表示，则直接返回音频和文本的潜在空间表示
        if return_latents:
            return audio_latents, text_latents

        # 如果需要返回相似度，则计算音频和文本潜在空间表示之间的相似度
        if return_similarities:
            return einsum('i d, i d -> i', audio_latents, text_latents)

        # 如果需要返回成对相似度，则计算音频和文本潜在空间表示之间的余弦相似度矩阵
        if return_pairwise_similarities:
            cosine_sim = einsum('i d, j d -> i j', audio_latents, text_latents)
            return cosine_sim

        # 计算对比损失
        cl_loss = self.contrast(audio_latents, text_latents)

        # 如果没有多层对比学习模块，则直接返回对比损失
        if not exists(self.multi_layer_contrastive_learning):
            return cl_loss

        # 从音频和文本层表示中选择指定索引的层
        audio_layers = audio_layers[self.audio_layers_indices]
        text_layers = text_layers[self.text_layers_indices]

        # 根据 ViCHA 论文中的建议，是否在所有层之间进行对比损失
        hierarchical_cl_loss = self.multi_layer_contrastive_learning(
            audio_layers=audio_layers,
            text_layers=text_layers
        )

        # 返回对比损失和多层对比学习损失的总和
        return cl_loss + hierarchical_cl_loss
# 定义 MuLaNEmbedQuantizer 类，继承自 AudioConditionerBase 类
class MuLaNEmbedQuantizer(AudioConditionerBase):
    # 初始化函数
    @beartype
    def __init__(
        self,
        mulan: MuLaN,  # MuLaN 对象
        conditioning_dims: Tuple[int, ...],  # 条件维度元组
        rq_num_quantizers = 8,  # RQ 量化器数量，默认为 8
        rq_ema_decay = 0.9,  # RQ 指数移动平均衰减率，默认为 0.9
        codebook_size = 1024,  # 代码簿大小，默认为 1024
        namespaces: Tuple[str, ...] = ('semantic', 'coarse', 'fine'),  # 命名空间元组，默认包含 'semantic', 'coarse', 'fine'
    ):
        super().__init__()  # 调用父类的初始化函数
        self.mulan = mulan  # 初始化 MuLaN 对象

        assert len(namespaces) > 0  # 断言命名空间数量大于 0
        self.namespaces = namespaces  # 初始化命名空间
        self.conditioning_dims = conditioning_dims  # 初始化条件维度

        assert len(conditioning_dims) == len(namespaces), 'number of conditioning dimensions must be equal to number of namespaces'  # 断言条件维度数量等于命名空间数量

        dim = mulan.dim_latent  # 获取 MuLaN 对象的潜在维度

        # 初始化 RQ 对象
        self.rq = ResidualVQ(
            dim = dim,
            num_quantizers = rq_num_quantizers,
            codebook_size = codebook_size,
            decay = rq_ema_decay,
            commitment_weight = 0,    # 只使用 EMA 更新代码簿
            kmeans_init = True,
            threshold_ema_dead_code = 2,
            quantize_dropout = False  # 不使用量化丢弃
        )

        self.dim = dim  # 初始化维度
        self.num_codebooks = rq_num_quantizers  # 初始化代码簿数量

        self.cond_embeddings = nn.ParameterDict({})  # 初始化条件嵌入字典

        # 遍历命名空间和条件维度，初始化条件嵌入
        for namespace, conditioning_dim in zip(namespaces, conditioning_dims):
            cond_embeddings = nn.Parameter(torch.randn(rq_num_quantizers, codebook_size, conditioning_dim))
            nn.init.normal_(cond_embeddings, std = 0.02)

            self.cond_embeddings[namespace] = cond_embeddings

        self.set_default_namespace(namespaces[0])  # 设置默认命名空间为第一个命名空间

    # 返回参数
    def parameters(self):
        return self.cond_embeddings.parameters()

    # 设置默认命名空间
    def set_default_namespace(self, namespace):
        self._default_namespace = namespace

    # 前向传播函数
    def forward(
        self,
        wavs = None,  # 音频数据，默认为 None
        texts = None,  # 文本数据，默认为 None
        namespace = None  # 命名空间，默认为 None
    ):
        assert exists(wavs) ^ exists(texts)  # 断言音频数据或文本数据必须存在其中一个

        namespace = default(namespace, self._default_namespace)  # 获取命名空间，默认为默认命名空间
        assert namespace in self.namespaces, f'namespace {namespace} not found'  # 断言命名空间必须在命名空间列表中

        cond_embeddings = self.cond_embeddings[namespace]  # 获取对应命名空间的条件嵌入

        with torch.no_grad():  # 禁用梯度计算
            self.mulan.eval()  # 设置 MuLaN 为评估模式

            # 音频和语言存在于联合嵌入空间中，因为对比学习

            if exists(wavs):  # 如果音频数据存在
                latents = self.mulan.get_audio_latents(wavs)  # 获取音频潜在表示
            elif exists(texts):  # 如果文本数据存在
                latents = self.mulan.get_text_latents(texts)  # 获取文本潜在表示

        _, indices, _ = self.rq(latents)  # ���用 RQ 对象进行量化

        batch, num_codebooks, dim = indices.shape[0], self.num_codebooks, cond_embeddings.shape[-1]  # 获取批次大小、代码簿数量和维度

        cond_embeddings = repeat(cond_embeddings, 'q c d -> b q c d', b = batch)  # 重复条件嵌入
        indices = repeat(indices, 'b q -> b q 1 d', q = num_codebooks, d = dim)  # 重复索引

        cond_embeddings = cond_embeddings.gather(2, indices)  # 根据索引获取条件嵌入
        return rearrange(cond_embeddings, 'b q 1 d -> b q d')  # 重新排列条件嵌入维度

# 定义 MusicLM 类，继承自 nn.Module
class MusicLM(nn.Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        audio_lm: AudioLM,  # AudioLM 对象
        mulan_embed_quantizer: MuLaNEmbedQuantizer  # MuLaNEmbedQuantizer 对象
    ):
        super().__init__()  # 调用父类的初始化函数
        assert not exists(audio_lm.audio_conditioner), 'mulan must not have been passed into AudioLM. it will be managed externally now, embedding the text into the joint embedding space for text-to-audio synthesis'

        self.mulan_embed_quantizer = mulan_embed_quantizer  # 初始化 MuLaNEmbedQuantizer 对象
        self.audio_lm = audio_lm  # 初始化 AudioLM 对象

    # 设备属性
    @property
    def device(self):
        return next(self.parameters()).device  # 返回参数的设备

    # 前向传播函数
    @torch.no_grad()
    def forward(
        self,
        text: str,  # 文本数据
        num_samples = 1,  # 样本数量，默认为 1
        **audio_lm_kwargs  # 音频 LM 参数
        ):
        # 调用 eval 方法
        self.eval()

        # 使用分词器对文本进行分词，并将结果转移到指定设备上
        texts = tokenizer.tokenize([text]).to(self.device)

        # 使用 mulan_embed_quantizer 对文本进行嵌入量化
        text_embeds = self.mulan_embed_quantizer(texts=texts)

        # 无法处理变长音频

        # 初始化一个空列表用于存储生成的音乐样本
        samples = []

        # 生成指定数量的音乐样本
        for _ in range(num_samples):
            # 使用 audio_lm 生成音乐，传入文本嵌入和其他参数
            music = self.audio_lm(text_embeds=text_embeds, **audio_lm_kwargs)
            samples.append(music)

        # 如果只生成一个样本，则直接返回该样本
        if num_samples == 1:
            return first(samples)

        # 获取 mulan_embed_quantizer 中的 mulan 模型
        mulan = self.mulan_embed_quantizer.mulan

        # 计算所有样本与文本的相似度，找到相似度最高的样本
        sims = torch.cat([mulan(texts=texts, wavs=music, return_similarities=True) for music in samples], dim=0)
        top_matching_index = sims.topk(1, dim=0).indices.item()

        # 返回相似度最高的样本
        return samples[top_matching_index]
```