# `.\lucidrains\video-diffusion-pytorch\video_diffusion_pytorch\video_diffusion_pytorch.py`

```
# 导入数学库
import math
# 导入拷贝库
import copy
# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum 模块
from torch import nn, einsum
# 从 torch 库中导入 F 模块
import torch.nn.functional as F
# 从 functools 库中导入 partial 函数
from functools import partial
# 从 torch.utils 库中导入 data 模块
from torch.utils import data
# 从 pathlib 库中导入 Path 类
from pathlib import Path
# 从 torch.optim 库中导入 Adam 优化器
from torch.optim import Adam
# 从 torchvision 库中导入 transforms, utils 模块
from torchvision import transforms as T, utils
# 从 torch.cuda.amp 库中导入 autocast, GradScaler 模块
from torch.cuda.amp import autocast, GradScaler
# 从 PIL 库中导入 Image 类
from PIL import Image
# 从 tqdm 库中导入 tqdm 函数
from tqdm import tqdm
# 从 einops 库中导入 rearrange 函数
from einops import rearrange
# 从 einops_exts 库中导入 check_shape, rearrange_many 函数
from einops_exts import check_shape, rearrange_many
# 从 rotary_embedding_torch 库中导入 RotaryEmbedding 类
from rotary_embedding_torch import RotaryEmbedding
# 从 video_diffusion_pytorch.text 模块中导入 tokenize, bert_embed, BERT_MODEL_DIM 常量

# 辅助函数

# 判断变量是否存在
def exists(x):
    return x is not None

# 空操作函数
def noop(*args, **kwargs):
    pass

# 判断一个数是否为奇数
def is_odd(n):
    return (n % 2) == 1

# 返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 无限循环生成器
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 将一个数分成若干组
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# 生成概率掩码
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

# 判断列表中是否全为字符串
def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# 相对位置偏置

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# 小助手模块

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    # 定义一个前向传播函数，接受输入张量 x
    def forward(self, x):
        # 获取输入张量 x 的设备信息
        device = x.device
        # 计算嵌入维度的一半
        half_dim = self.dim // 2
        # 计算嵌入的指数
        emb = math.log(10000) / (half_dim - 1)
        # 计算嵌入的指数值
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 计算嵌入矩阵
        emb = x[:, None] * emb[None, :]
        # 将正弦和余弦值拼接在一起，形成最终的嵌入矩阵
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # 返回嵌入矩阵
        return emb
# 定义一个上采样函数，使用 ConvTranspose3d 实现
def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

# 定义一个下采样函数，使用 Conv3d 实现
def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

# 定义 LayerNorm 类，用于实现层归一化
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        # 计算输入张量 x 的方差
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        # 计算输入张量 x 的均值
        mean = torch.mean(x, dim = 1, keepdim = True)
        # 返回经过层归一化处理后的结果
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

# 定义 PreNorm 类，结合层归一化和函数 fn 的处理
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        # 对输入张量 x 进行层归一化处理
        x = self.norm(x)
        # 返回经过函数 fn 处理后的结果
        return self.fn(x, **kwargs)

# 构建基础模块

# 定义 Block 类，包含投影、归一化和激活函数
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        # 投影操作
        x = self.proj(x)
        # 归一化操作
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            # 应用缩放和平移
            x = x * (scale + 1) + shift

        return self.act(x)

# 定义 ResnetBlock 类，包含 MLP、Block 和残差连接
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

# 定义 SpatialLinearAttention 类，包含注意力计算和输出转换
class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# 定义 EinopsToAndFrom 类，用于实现输入输出形状的转换
class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

# 定义 Attention 类
class Attention(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 计算缩放因子
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        # 初始化旋转嵌入和线性变换层
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    # 前向传播函数
    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        # 获取输入张量的维度和设备信息
        n, device = x.shape[-2], x.device

        # 将输入张量通过线性变换层得到查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        # 如果存在焦点存在掩码并且所有焦点都存在，则直接输出值
        if exists(focus_present_mask) and focus_present_mask.all():
            values = qkv[-1]
            return self.to_out(values)

        # 将查询、键、值按头数拆分
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)

        # 缩放查询
        q = q * self.scale

        # 将位置旋转到查询和键中以进行时间注意力
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # 计算相似度
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # 添加相对位置偏置
        if exists(pos_bias):
            sim = sim + pos_bias

        # 如果存在焦点存在掩码并且不是所有焦点都存在，则进行掩码处理
        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # 数值稳定性处理
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # 聚合值
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)
# 定义一个名为Unet3D的类，继承自nn.Module
class Unet3D(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,  # 输入数据的维度
        cond_dim = None,  # 条件数据的维度，默认为None
        out_dim = None,  # 输出数据的维度，默认为None
        dim_mults=(1, 2, 4, 8),  # 每个层级的维度倍增系数
        channels = 3,  # 输入数据的通道数，默认为3
        attn_heads = 8,  # 注意力头的数量，默认为8
        attn_dim_head = 32,  # 每个注意力头的维度，默认为32
        use_bert_text_cond = False,  # 是否使用BERT文本条件，默认为False
        init_dim = None,  # 初始化维度，默认为None
        init_kernel_size = 7,  # 初始化卷积核大小，默认为7
        use_sparse_linear_attn = True,  # 是否使用稀疏线性注意力，默认为True
        block_type = 'resnet',  # 块类型，默认为'resnet'
        resnet_groups = 8  # ResNet块的数量，默认为8
    ):
        # 调用父类的构造函数
        super().__init__()
        # 设置通道数
        self.channels = channels

        # 时间注意力和其相对位置编码

        # 创建旋转嵌入对象
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        # 定义时间注意力函数
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb))

        # 创建相对位置偏置对象
        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # 现实中不太可能生成那么多帧的视频...但是

        # 初始卷积

        # 初始化维度
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        # 创建初始卷积层
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))

        # 创建初始时间注意力层
        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # 维度

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 时间条件

        time_dim = dim * 4
        # 创建时间 MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 文本条件

        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # 层

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # 块类型

        block_klass = partial(ResnetBlock, groups = resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim = cond_dim)

        # 所有层的模块

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        # 创建中间块1
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads = attn_heads))

        # 创建中间空间注意力层
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        # 创建中间时间注意力层
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        # 创建中间块2
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        # 创建最终卷积层
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        **kwargs
    # 根据给定参数计算模型的输出 logits
    def forward(
        self,
        x,
        time,
        cond = None,
        null_cond_prob = 0.,
        focus_present_mask = None,
        prob_focus_present = 0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        # 检查是否存在条件 cond，如果 cond_dim 被指定，则必须传入 cond
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        # 获取输入 x 的 batch 大小和设备信息
        batch, device = x.shape[0], x.device

        # 如果未提供 focus_present_mask，则根据概率 prob_focus_present 创建一个概率掩码
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))

        # 根据输入 x 的时间维度创建时间相对位置偏置
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)

        # 对输入 x 进行初始卷积操作
        x = self.init_conv(x)

        # 对输入 x 进行初始时间注意力操作，使用时间相对位置偏置
        x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias)

        # 克隆输入 x，用于后续操作
        r = x.clone()

        # 如果存在时间多层感知机，则计算时间多层感知机的输出
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # 如果模型具有条件 cond，则进行条件处理
        if self.has_cond:
            # 重新获取 batch 和设备信息
            batch, device = x.shape[0], x.device
            # 根据 null_cond_prob 创建一个概率掩码
            mask = prob_mask_like((batch,), null_cond_prob, device = device)
            # 如果掩码为真，则使用 null_cond_emb 替换 cond
            cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, cond)
            # 将 cond 与 t 连接在一起
            t = torch.cat((t, cond), dim = -1)

        # 初始化一个空列表 h，用于存储中间结果
        h = []

        # 遍历下采样模块，依次进行操作
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            h.append(x)
            x = downsample(x)

        # 中间块1操作
        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
        x = self.mid_block2(x, t)

        # 遍历上采样模块，依次进行操作
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            x = upsample(x)

        # 将最终输出与 r 进行连接，并返回最终卷积结果
        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)
# gaussian diffusion trainer class

# 从输入张量 a 中根据索引张量 t 提取对应元素，然后重塑形状
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 根据给定的时间步数生成余弦调度表
def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

# 定义 GaussianDiffusion 类
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls = False,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        use_dynamic_thres = False, # from the Imagen paper
        dynamic_thres_percentile = 0.9
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        # 生成余弦调度表
        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # 注册缓冲区辅助函数，将 float64 类型转换为 float32 类型
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 计算扩散 q(x_t | x_{t-1}) 和其他参数

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 计算后验 q(x_{t-1} | x_t, x_0) 参数

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # 上述：等于 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # 下面：对后验方差进行对数计算，因为扩散链的开始处后验方差为 0

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 文本条件参数

        self.text_use_bert_cls = text_use_bert_cls

        # 在采样时使用动态阈值

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    # 计算 q(x_t | x_{t-1}) 的均值和方差
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # 从噪声中预测起始点
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    # 计算后验分布的均值、方差和截断后的对数方差
    def q_posterior(self, x_start, x_t, t):
        # 计算后验分布的均值
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # 计算后验分布的方差
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        # 获取截断后的对数方差
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 计算模型的均值、方差和截断后的对数方差
    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_scale = 1.):
        # 从噪声中预测起始值
        x_recon = self.predict_start_from_noise(x, t=t, noise = self.denoise_fn.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                # 计算动态阈值
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # 根据阈值截断，取决于是静态还是动态
            x_recon = x_recon.clamp(-s, s) / s

        # 获取模型的均值、后验方差和后验对数方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # 生成样本
    @torch.inference_mode()
    def p_sample(self, x, t, cond = None, cond_scale = 1., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        # 获取模型的均值、方差和对数方差
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised, cond = cond, cond_scale = cond_scale)
        noise = torch.randn_like(x)
        # 当 t == 0 时不添加噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # 循环生成样本
    @torch.inference_mode()
    def p_sample_loop(self, shape, cond = None, cond_scale = 1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond = cond, cond_scale = cond_scale)

        return unnormalize_img(img)

    # 生成样本
    @torch.inference_mode()
    def sample(self, cond = None, cond_scale = 1., batch_size = 16):
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size), cond = cond, cond_scale = cond_scale)

    # 插值
    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    # 从起始值生成样本
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    # 计算像素损失函数
    def p_losses(self, x_start, t, cond = None, noise = None, **kwargs):
        # 获取输入张量的形状和设备信息
        b, c, f, h, w, device = *x_start.shape, x_start.device
        # 如果没有提供噪声数据，则生成一个与输入张量相同形状的随机噪声张量
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 生成带有噪声的输入张量
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 如果条件是字符串列表，则将其转换为BERT嵌入表示，并根据需要返回CLS表示
        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr = self.text_use_bert_cls)
            cond = cond.to(device)

        # 使用去噪函数对带有噪声的输入张量进行去噪处理
        x_recon = self.denoise_fn(x_noisy, t, cond = cond, **kwargs)

        # 根据损失类型计算损失值
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        # 返回计算得到的损失值
        return loss

    # 前向传播函数
    def forward(self, x, *args, **kwargs):
        # 获取输入张量的形状信息、设备信息和图像大小
        b, device, img_size, = x.shape[0], x.device, self.image_size
        # 检查输入张量的形状是否符合要求
        check_shape(x, 'b c f h w', c = self.channels, f = self.num_frames, h = img_size, w = img_size)
        # 生成随机时间步长
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # 对输入图像进行归一化处理
        x = normalize_img(x)
        # 调用像素损失函数计算损失值并返回
        return self.p_losses(x, t, *args, **kwargs)
# trainer class

# 定义通道数与模式的映射关系
CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

# 遍历图像的所有帧并转换为指定通道数的图像
def seek_all_images(img, channels = 3):
    # 检查通道数是否有效
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    # 获取对应通道数的图像模式
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            # 尝试定位到第i帧图像并转换为指定通道数的图像
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# 将张量转换为 GIF 图像并保存
def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    # 将张量解绑定为图像列表
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    # 保存 GIF 图像
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# 将 GIF 图像转换为张量
def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    # 打开 GIF 图像
    img = Image.open(path)
    # 对 GIF 图像的每一帧进行转换为张量
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

# 定义恒等函数
def identity(t, *args, **kwargs):
    return t

# 将图像张量归一化到[-1, 1]范围
def normalize_img(t):
    return t * 2 - 1

# 将归一化后的图像张量反归一化
def unnormalize_img(t):
    return (t + 1) * 0.5

# 调整张量的帧数
def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

# 数据集类
class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        num_frames = 16,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        # 获取指定文件夹下所有指定扩展名的文件路径
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # 根据是否强制指定帧数选择相应的函数
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        # 图像转换操作
        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # 获取指定索引的文件路径并将其转换为张量
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform = self.transform)
        return self.cast_num_frames_fn(tensor)

# trainer class

# 训练器类
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        num_frames = 16,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        num_sample_rows = 4,
        max_grad_norm = None
    # 初始化 Diffusion Trainer 类
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置模型
        self.model = diffusion_model
        # 创建指数移动平均对象
        self.ema = EMA(ema_decay)
        # 复制模型用于指数移动平均
        self.ema_model = copy.deepcopy(self.model)
        # 每隔一定步数更新指数移动平均
        self.update_ema_every = update_ema_every

        # 开始使用指数移动平均的步数
        self.step_start_ema = step_start_ema
        # 每隔一定步数保存模型和生成样本
        self.save_and_sample_every = save_and_sample_every

        # 训练批次大小
        self.batch_size = train_batch_size
        # 图像大小
        self.image_size = diffusion_model.image_size
        # 梯度累积步数
        self.gradient_accumulate_every = gradient_accumulate_every
        # 训练步数
        self.train_num_steps = train_num_steps

        # 获取图像大小、通道数和帧数
        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        # 创建数据集对象
        self.ds = Dataset(folder, image_size, channels = channels, num_frames = num_frames)

        # 打印数据集信息
        print(f'found {len(self.ds)} videos as gif files at {folder}')
        # 断言数据集长度大于0
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        # 创建数据加载器
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        # 创建优化器
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        # 初始化步数
        self.step = 0

        # 是否使用混合精度训练
        self.amp = amp
        # 创建梯度缩放器
        self.scaler = GradScaler(enabled = amp)
        # 最大梯度范数
        self.max_grad_norm = max_grad_norm

        # 生成样本的行数
        self.num_sample_rows = num_sample_rows
        # 结果保存文件夹
        self.results_folder = Path(results_folder)
        # 创建结果保存文件夹
        self.results_folder.mkdir(exist_ok = True, parents = True)

        # 重置参数
        self.reset_parameters()

    # 重置参数
    def reset_parameters(self):
        # 加载模型参数到指数移动平均模型
        self.ema_model.load_state_dict(self.model.state_dict())

    # 更新指数移动平均模型
    def step_ema(self):
        # 若步数小于开始使用指数移动平均的步数，则重置参数
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        # 更新指数移动平均模型
        self.ema.update_model_average(self.ema_model, self.model)

    # 保存模型
    def save(self, milestone):
        # 保存训练状态
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        # 将数据保存到文件
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    # 加载模型
    def load(self, milestone, **kwargs):
        # 若加载最新的检查点
        if milestone == -1:
            # 获取所有里程碑
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            # 断言至少有一个里程碑
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            # 获取最大的里程碑
            milestone = max(all_milestones)

        # 加载模型数据
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        # 更新步数和模型参数
        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    # 训练方法
    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop
        ):
        # 断言日志函数是可调用的
        assert callable(log_fn)

        # 当步数小于训练步数时，执行训练循环
        while self.step < self.train_num_steps:
            # 对于每个梯度累积周期
            for i in range(self.gradient_accumulate_every):
                # 从数据加载器中获取下一个数据批次并移至 GPU
                data = next(self.dl).cuda()

                # 使用自动混合精度计算
                with autocast(enabled = self.amp):
                    # 计算模型损失
                    loss = self.model(
                        data,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask
                    )

                    # 反向传播并缩放损失
                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                # 打印当前步数和损失值
                print(f'{self.step}: {loss.item()}')

            # 记录损失值
            log = {'loss': loss.item()}

            # 如果存在最大梯度范数，则对梯度进行裁剪
            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # 更新模型参数
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            # 每隔一定步数更新指数移动平均
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # 每隔一定步数保存模型并生成样本
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size)

                # 生成所有样本视频
                all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_videos_list = torch.cat(all_videos_list, dim = 0)

                # 对视频进行填充
                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                # 重新排列视频帧以生成 GIF
                one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                video_tensor_to_gif(one_gif, video_path)
                log = {**log, 'sample': video_path}
                self.save(milestone)

            # 记录日志
            log_fn(log)
            self.step += 1

        # 训练完成后打印消息
        print('training completed')
```