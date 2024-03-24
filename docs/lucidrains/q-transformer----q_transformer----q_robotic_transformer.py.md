# `.\lucidrains\q-transformer\q_transformer\q_robotic_transformer.py`

```
# 从 random 模块导入 random 函数
from random import random
# 从 functools 模块导入 partial, cache 函数
from functools import partial, cache

# 导入 torch 模块
import torch
# 从 torch 模块中导入 F, nn, einsum, Tensor 等
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

# 从 beartype 模块中导入 beartype 函数
from beartype import beartype
# 从 beartype.typing 模块中导入 Union, List, Optional, Callable, Tuple, Dict, Any 等
from beartype.typing import Union, List, Optional, Callable, Tuple, Dict, Any

# 从 einops 模块中导入 pack, unpack, repeat, reduce, rearrange 函数
from einops import pack, unpack, repeat, reduce, rearrange
# 从 einops.layers.torch 模块中导入 Rearrange, Reduce 类
from einops.layers.torch import Rearrange, Reduce

# 从 q_transformer.attend 模块中导入 Attend 类
from q_transformer.attend import Attend

# 从 classifier_free_guidance_pytorch 模块中导入 TextConditioner, AttentionTextConditioner, NullConditioner, classifier_free_guidance 函数
from classifier_free_guidance_pytorch import (
    TextConditioner,
    AttentionTextConditioner,
    NullConditioner,
    classifier_free_guidance
)

# helpers

# 定义函数 exists，判断值是否存在
def exists(val):
    return val is not None

# 定义函数 xnor，实现逻辑运算
def xnor(x, y):
    """ (True, True) or (False, False) -> True """
    return not (x ^ y)

# 定义函数 divisible_by，判断 num 是否能被 den 整除
def divisible_by(num, den):
    return (num % den) == 0

# 定义函数 default，返回 val 或默认值 d
def default(val, d):
    return val if exists(val) else d

# 定义函数 cast_tuple，将 val 转换为元组，长度为 length
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# tensor helpers

# 定义函数 l2norm，对张量进行 L2 归一化
def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim)

# 定义函数 pack_one，将 x 按照指定模式 pattern 进行打包
def pack_one(x, pattern):
    return pack([x], pattern)

# 定义函数 unpack_one，将 x 按照指定模式 pattern 进行解包
def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

# 2d rotary positional embedding
# https://arxiv.org/abs/2104.09864

# 定义类 RotaryEmbedding，实现 2D 旋转位置嵌入
class RotaryEmbedding(Module):
    def __init__(self, dim, omega = 10000):
        super().__init__()
        inv_freq = 1.0 / (omega ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    @autocast(enabled = False)
    def forward(self, height_width):
        device, dtype = self.inv_freq.device, self.inv_freq.dtype

        axial_pos = torch.arange(height_width, device = device).type(dtype)

        freqs = torch.einsum('i, j -> i j', axial_pos, self.inv_freq)
        freqs = repeat(freqs, '... f -> ... (f c)', c = 2)

        freqs = torch.broadcast_tensors(freqs[None, :, :], freqs[:, None, :])
        freqs = torch.cat(freqs, dim = -1)
        return rearrange(freqs, '... f -> (...) f')

# 定义函数 rotate_half，对张量进行旋转
def rotate_half(x):
    x1, x2 = rearrange(x, '... (d c) -> ... d c', c = 2).unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d c -> ... (d c)')

@autocast(enabled = False)
# 定义函数 apply_rotary_pos_emb，应用旋转位置嵌入
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# sync batchnorm

# 使用缓存装饰器缓存结果
@cache
def get_is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

# 定义函数 MaybeSyncBatchnorm2d，根据是否分布式返回 SyncBatchNorm 或 BatchNorm2d
def MaybeSyncBatchnorm2d(is_distributed = None):
    is_distributed = default(is_distributed, get_is_distributed())
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm2d

# channel rmsnorm

# 定义类 RMSNorm，实现 RMS 归一化
class RMSNorm(Module):
    def __init__(self, dim, affine = True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale

# 定义类 ChanRMSNorm，实现通道 RMS 归一化
class ChanRMSNorm(Module):
    def __init__(self, dim, affine = True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1)) if affine else 1.

    def forward(self, x):
        return l2norm(x, dim = 1) * self.gamma * self.scale

# sinusoidal positions

# 定义函数 posemb_sincos_1d，生成正弦余弦位置嵌入
def posemb_sincos_1d(seq, dim, temperature = 10000, device = None, dtype = torch.float32):
    n = torch.arange(seq, device = device)
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim = 1)
    return pos_emb.type(dtype)

# helper classes

# 定义类 Residual，实现残差连接
class Residual(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 定义类 FeedForward，实现前馈网络
class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.,
        adaptive_ln = False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化自适应层归一化标志
        self.adaptive_ln = adaptive_ln

        # 计算内部维度
        inner_dim = int(dim * mult)
        # 初始化 RMS 归一化层
        self.norm = RMSNorm(dim, affine = not adaptive_ln)

        # 构建神经网络模型
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),  # 线性层
            nn.GELU(),  # GELU 激活函数
            nn.Dropout(dropout),  # Dropout 层
            nn.Linear(inner_dim, dim),  # 线性层
            nn.Dropout(dropout)  # Dropout 层
        )

    def forward(
        self,
        x,
        cond_fn: Optional[Callable] = None
    ):
        # 对输入数据进行归一化
        x = self.norm(x)

        # 断言自适应层归一化和条件函数的存在
        assert xnor(self.adaptive_ln, exists(cond_fn))

        if exists(cond_fn):
            # 如果条件函数存在，则应用条件函数
            # 自适应层归一化
            x = cond_fn(x)

        return self.net(x)
# 定义 SqueezeExcitation 类，用于实现通道注意力机制
class SqueezeExcitation(Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        # 定义通道注意力机制的结构
        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),  # 对输入进行平均池化
            nn.Linear(dim, hidden_dim, bias = False),  # 线性变换
            nn.SiLU(),  # SiLU 激活函数
            nn.Linear(hidden_dim, dim, bias = False),  # 线性变换
            nn.Sigmoid(),  # Sigmoid 激活函数
            Rearrange('b c -> b c 1 1')  # 重排维度
        )

    def forward(self, x):
        return x * self.gate(x)  # 返回加权后的输出

# 定义 MBConvResidual 类，用于实现残差连接
class MBConvResidual(Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)  # 添加随机丢弃采样

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x  # 返回残差连接后的结果

# 定义 Dropsample 类，用于实现随机丢弃采样
class Dropsample(Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        batch, device = x.shape[0], x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((batch, 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)  # 返回随机丢弃采样后的结果

# 定义 MBConv 函数，用于构建 MBConv 模块
def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.,
    is_distributed = None,
    use_layernorm = True
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    if use_layernorm:
        norm_klass = ChanRMSNorm
    else:
        norm_klass = MaybeSyncBatchnorm2d(is_distributed)

    # 构建 MBConv 模块的网络结构
    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        norm_klass(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        norm_klass(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        norm_klass(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)  # 添加残差连接

    return net  # 返回构建好的 MBConv 模块

# 定义 Attention 类，用于实现注意力机制
class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 32,
        dropout = 0.,
        window_size = 7,
        num_mem_kv = 4,
        flash = True
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)
        self.heads = heads

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)  # 线性变换得到查询、键、值

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, self.heads),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1')
        )

        self.attend = Attend(
            causal = False,
            dropout = dropout,
            flash = flash
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        rotary_emb = None
        # 解包输入张量的形状和设备信息
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # 对输入张量进行归一化处理
        x = self.norm(x)

        # 展平输入张量
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # 为查询、键、值进行投影
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # 获取值的门控信息
        g = self.to_v_gates(x)

        # 将查询、键、值按头数进行分割
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 如果存在旋转位置编码，则应用到查询和键上
        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        # 注意力机制
        out = self.attend(q, k, v)

        # 每个头部的值乘以门控信息，允许不关注某些值
        out = out * g

        # 合并头部
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # 合并头部输出
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)
# 定义一个名为 MaxViT 的类，继承自 Module 类
class MaxViT(Module):
    # 初始化方法，接收多个参数
    @beartype
    def __init__(
        self,
        *,
        num_classes,  # 类别数量
        dim,  # 维度
        depth: Tuple[int, ...],  # 深度
        heads = 8,  # 头数
        dim_head = 64,  # 头的维度
        dim_conv_stem = None,  # 卷积层的维度
        window_size = 7,  # 窗口大小
        mbconv_expansion_rate = 4,  # 扩张率
        mbconv_shrinkage_rate = 0.25,  # 收缩率
        use_layernorm = True,  # 是否使用层归一化
        dropout = 0.1,  # 丢弃率
        channels = 3,  # 通道数
        flash_attn = True  # 是否使用闪存注意力
    ):
        # 调用父类的初始化方法
        super().__init__()
        
        # 卷积层
        dim_conv_stem = default(dim_conv_stem, dim)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )
        
        # 变量
        num_stages = len(depth)
        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        self.layers = ModuleList([])
        
        # 为了高效的块-网格式注意力，设置窗口大小
        self.window_size = window_size
        w = window_size
        
        # 旋转嵌入
        assert divisible_by(dim_head, 4), f'{dim_head} must be divisible by 4 for axial rotary embedding for maxvit'
        self.axial_rotary_emb = RotaryEmbedding(dim_head)
        self.register_buffer('cached_rotary_emb', self.axial_rotary_emb(window_size), persistent = False)
        
        # 遍历各个阶段
        cond_hidden_dims = []
        
        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim
                cond_hidden_dims.append(stage_dim_in)
                
                # 定义模块列表
                block = nn.ModuleList([
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate,
                        use_layernorm = use_layernorm
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # 块状注意力
                    Residual(Attention(dim = layer_dim, heads = heads, dim_head = dim_head, dropout = dropout, window_size = w, flash = flash_attn)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
                    
                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # 网格状注意力
                    Residual(Attention(dim = layer_dim, heads = heads, dim_head = dim_head, dropout = dropout, window_size = w, flash = flash_attn)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                ])
                
                self.layers.append(block)
        
        embed_dim = dims[-1]
        self.embed_dim = dims[-1]
        self.cond_hidden_dims = cond_hidden_dims
        
        # MLP 头部输出
        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    
    # 前向传播方法
    @beartype
    def forward(
        self,
        img,  # 图像
        texts: Optional[List[str]] = None,  # 文本
        cond_fns: Optional[Tuple[Callable, ...]] = None,  # 条件函数
        cond_drop_prob = 0.,  # 条件丢弃概率
        return_embeddings = False  # 是否返回嵌入
        # 断言图像的最后两个维度是否都可以被窗口大小整除
        assert all([divisible_by(d, self.window_size) for d in img.shape[-2:])

        # 使用卷积层对输入图像进行处理
        x = self.conv_stem(img)

        # 获取缓存的旋转嵌入
        rotary_emb = self.cached_rotary_emb

        # 初始化条件函数迭代器
        cond_fns = iter(default(cond_fns, []))

        # 遍历模型的每一层
        for (
            mb_conv,
            rearr_windowed_in,
            windowed_attn,
            windowed_ff,
            rearr_windowed_out,
            rearr_grid_in,
            grid_attn,
            grid_ff,
            rearr_grid_out
        ) in self.layers:
            # 获取下一个条件函数
            cond_fn = next(cond_fns, None)

            # 如果存在条件函数，则对输入进行处理
            if exists(cond_fn):
                x = cond_fn(x)

            # 依次经过多个操作：多头卷积、重排窗口输入、窗口注意力、窗口前馈、重排窗口输出、重排网格输入、网格注意力、网格前馈、重排网格输出
            x = mb_conv(x)
            x = rearr_windowed_in(x)
            x = windowed_attn(x, rotary_emb = rotary_emb)
            x = windowed_ff(x)
            x = rearr_windowed_out(x)

            x = rearr_grid_in(x)
            x = grid_attn(x, rotary_emb = rotary_emb)
            x = grid_ff(x)
            x = rearr_grid_out(x)

        # 如果需要返回嵌入向量，则返回最终结果
        if return_embeddings:
            return x

        # 否则返回经过 MLP 头部处理后的结果
        return self.mlp_head(x)
# 定义 TransformerAttention 类，继承自 Module 类
class TransformerAttention(Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        num_mem_kv = 4,
        norm_context = False,
        adaptive_ln = False,
        dropout = 0.1,
        flash = True,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.adaptive_ln = adaptive_ln
        self.norm = RMSNorm(dim, affine = not adaptive_ln)

        self.context_norm = RMSNorm(dim_context) if norm_context else None

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = None
        if num_mem_kv > 0:
            self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))

        self.attend = Attend(
            dropout = dropout,
            flash = flash,
            causal = causal
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    # 前向传播函数
    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None,
        cond_fn: Optional[Callable] = None,
        cache: Optional[Tensor] = None,
        return_cache = False
    ):
        b = x.shape[0]

        assert xnor(exists(context), exists(self.context_norm))

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        assert xnor(exists(cond_fn), self.adaptive_ln)

        if exists(cond_fn):
            x = cond_fn(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        if exists(cache):
            ck, cv = cache
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        new_kv_cache = torch.stack((k, v))

        if exists(self.mem_kv):
            mk, mv = map(lambda t: repeat(t, '... -> b ...', b = b), self.mem_kv)

            k = torch.cat((mk, k), dim = -2)
            v = torch.cat((mv, v), dim = -2)

            if exists(mask):
                mask = F.pad(mask, (self.num_mem_kv, 0), value = True)

            if exists(attn_mask):
                attn_mask = F.pad(attn_mask, (self.num_mem_kv, 0), value = True)

        out = self.attend(q, k, v, mask = mask, attn_mask = attn_mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if not return_cache:
            return out

        return out, new_kv_cache

# 定义 Transformer 类，继承自 Module 类
class Transformer(Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.,
        ff_dropout = 0.,
        adaptive_ln = False,
        flash_attn = True,
        cross_attend = False,
        causal = False,
        final_norm = True
    ):
        super().__init__()
        self.layers = ModuleList([])

        attn_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            dropout = attn_dropout,
            flash = flash_attn
        )

        for _ in range(depth):
            self.layers.append(ModuleList([
                TransformerAttention(**attn_kwargs, causal = causal, adaptive_ln = adaptive_ln, norm_context = False),
                TransformerAttention(**attn_kwargs, norm_context = True) if cross_attend else None,
                FeedForward(dim = dim, dropout = ff_dropout, adaptive_ln = adaptive_ln)
            ]))

        self.norm = RMSNorm(dim) if final_norm else nn.Identity()

    @beartype
    # 定义一个前向传播函数，接受输入 x，条件函数列表 cond_fns，注意力掩码 attn_mask，上下文 context，缓存 cache，是否返回缓存 return_cache
    def forward(
        self,
        x,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask = None,
        context: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        return_cache = False
    ):
        # 检查是否存在缓存
        has_cache = exists(cache)

        # 如果存在缓存，将输入 x 分为前一部分 x_prev 和最后一部分 x
        if has_cache:
            x_prev, x = x[..., :-1, :], x[..., -1:, :]

        # 将条件函数列表和缓存转换为迭代器
        cond_fns = iter(default(cond_fns, []))
        cache = iter(default(cache, []))

        # 存储新的缓存
        new_caches = []

        # 遍历每个层中的注意力、可能的交叉注意力和前馈网络
        for attn, maybe_cross_attn, ff in self.layers:
            # 使用注意力模型计算输出和新的缓存
            attn_out, new_cache = attn(
                x,
                attn_mask = attn_mask,
                cond_fn = next(cond_fns, None),
                return_cache = True,
                cache = next(cache, None)
            )

            # 将新的缓存添加到列表中
            new_caches.append(new_cache)

            # 更新输入 x
            x = x + attn_out

            # 如果存在交叉注意力，确保上下文不为空，然后更新输入 x
            if exists(maybe_cross_attn):
                assert exists(context)
                x = maybe_cross_attn(x, context = context) + x

            # 使用前馈网络更新输入 x
            x = ff(x, cond_fn = next(cond_fns, None)) + x

        # 将新的缓存堆叠起来
        new_caches = torch.stack(new_caches)

        # 如果存在缓存，将 x_prev 和 x 拼接在一起
        if has_cache:
            x = torch.cat((x_prev, x), dim = -2)

        # 对输出进行归一化
        out = self.norm(x)

        # 如果不需要返回缓存，直接返回输出
        if not return_cache:
            return out

        # 如果需要返回缓存，同时返回输出和新的缓存
        return out, new_caches
# token learner module

class TokenLearner(Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        num_output_tokens = 8,
        num_layers = 2
    ):
        # 初始化 TokenLearner 类
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        # 定义神经网络结构
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups = num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups = num_output_tokens),
        )

    def forward(self, x):
        # 对输入数据进行打包
        x, ps = pack_one(x, '* c h w')
        x = repeat(x, 'b c h w -> b (g c) h w', g = self.num_output_tokens)
        # 使用神经网络进行前向传播
        attn = self.net(x)

        attn = rearrange(attn, 'b g h w -> b 1 g h w')
        x = rearrange(x, 'b (g c) h w -> b c g h w', g = self.num_output_tokens)

        # 计算均值
        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        # 对数据进行解包
        x = unpack_one(x, ps, '* c n')
        return x

# Dueling heads for Q value

class DuelingHead(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 2,
        action_bins = 256
    ):
        # 初始化 DuelingHead 类
        super().__init__()
        dim_hidden = dim * expansion_factor

        self.stem = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.SiLU()
        )

        self.to_values = nn.Sequential(
            nn.Linear(dim_hidden, 1)
        )

        self.to_advantages = nn.Sequential(
            nn.Linear(dim_hidden, action_bins)
        )

    def forward(self, x):
        x = self.stem(x)

        advantages = self.to_advantages(x)
        advantages = advantages - reduce(advantages, '... a -> ... 1', 'mean')

        values = self.to_values(x)

        q_values = values + advantages
        return q_values.sigmoid()

# Q head modules, for either single or multiple actions

class QHeadSingleAction(Module):
    def __init__(
        self,
        dim,
        *,
        num_learned_tokens = 8,
        action_bins = 256,
        dueling = False
    ):
        # 初始化 QHeadSingleAction 类
        super().__init__()
        self.action_bins = action_bins

        if dueling:
            self.to_q_values = nn.Sequential(
                Reduce('b (f n) d -> b d', 'mean', n = num_learned_tokens),
                DuelingHead(
                    dim,
                    action_bins = action_bins
                )
            )
        else:
            self.to_q_values = nn.Sequential(
                Reduce('b (f n) d -> b d', 'mean', n = num_learned_tokens),
                RMSNorm(dim),
                nn.Linear(dim, action_bins),
                nn.Sigmoid()
            )

    def get_random_actions(self, batch_size):
        return torch.randint(0, self.action_bins, (batch_size,), device = self.device)

    def get_optimal_actions(
        self,
        encoded_state,
        return_q_values = False,
        actions = None,
        **kwargs
    ):
        assert not exists(actions), 'single actions will never receive previous actions'

        q_values = self.forward(encoded_state)

        max_q, action_indices = q_values.max(dim = -1)

        if not return_q_values:
            return action_indices

        return action_indices, max_q

    def forward(self, encoded_state):
        return self.to_q_values(encoded_state)

class QHeadMultipleActions(Module):
    def __init__(
        self,
        dim,
        *,
        num_actions = 8,
        action_bins = 256,
        attn_depth = 2,
        attn_dim_head = 32,
        attn_heads = 8,
        dueling = False,
        weight_tie_action_bin_embed = False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化动作数量和动作分箱数
        self.num_actions = num_actions
        self.action_bins = action_bins

        # 初始化动作分箱的嵌入参数
        self.action_bin_embeddings = nn.Parameter(torch.zeros(num_actions, action_bins, dim))
        # 使用正态分布初始化动作分箱的嵌入参数
        nn.init.normal_(self.action_bin_embeddings, std = 0.02)

        # 初始化线性层用于将维度转换为动作分箱数
        self.to_q_values = None
        if not weight_tie_action_bin_embed:
            self.to_q_values = nn.Linear(dim, action_bins)

        # 初始化Transformer模型
        self.transformer = Transformer(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            cross_attend = True,
            adaptive_ln = False,
            causal = True,
            final_norm = True
        )

        # 初始化最终的归一化层
        self.final_norm = RMSNorm(dim)

        # 初始化是否使用dueling网络
        self.dueling = dueling
        if dueling:
            self.to_values = nn.Parameter(torch.zeros(num_actions, dim))

    @property
    def device(self):
        # 返回动作分箱嵌入参数所在的设备
        return self.action_bin_embeddings.device

    def maybe_append_actions(self, sos_tokens, actions: Optional[Tensor] = None):
        if not exists(actions):
            return sos_tokens

        batch, num_actions = actions.shape
        # 获取动作的嵌入参数
        action_embeddings = self.action_bin_embeddings[:num_actions]

        action_embeddings = repeat(action_embeddings, 'n a d -> b n a d', b = batch)
        past_action_bins = repeat(actions, 'b n -> b n 1 d', d = action_embeddings.shape[-1])

        bin_embeddings = action_embeddings.gather(-2, past_action_bins)
        bin_embeddings = rearrange(bin_embeddings, 'b n 1 d -> b n d')

        tokens, _ = pack((sos_tokens, bin_embeddings), 'b * d')
        tokens = tokens[:, :self.num_actions] # 最后一个动作分箱不需要用于提议的q-learning
        return tokens

    def get_q_values(self, embed):
        num_actions = embed.shape[-2]

        if exists(self.to_q_values):
            logits = self.to_q_values(embed)
        else:
            # 每个token预测下一个动作分箱
            action_bin_embeddings = self.action_bin_embeddings[:num_actions]
            action_bin_embeddings = torch.roll(action_bin_embeddings, shifts = -1, dims = 1)
            logits = einsum('b n d, n a d -> b n a', embed, action_bin_embeddings)

        if self.dueling:
            advantages = logits
            values = einsum('b n d, n d -> b n', embed, self.to_values[:num_actions])
            values = rearrange(values, 'b n -> b n 1')

            q_values = values + (advantages - reduce(advantages, '... a -> ... 1', 'mean'))
        else:
            q_values = logits

        return q_values.sigmoid()

    def get_random_actions(self, batch_size, num_actions = None):
        num_actions = default(num_actions, self.num_actions)
        return torch.randint(0, self.action_bins, (batch_size, num_actions), device = self.device)

    @torch.no_grad()
    def get_optimal_actions(
        self,
        encoded_state,
        return_q_values = False,
        actions: Optional[Tensor] = None,
        prob_random_action: float = 0.5,
        **kwargs
    ):
        # 断言随机动作概率在 [0, 1] 之间
        assert 0. <= prob_random_action <= 1.
        # 获取批次大小
        batch = encoded_state.shape[0]

        # 如果随机动作概率为1，则返回随机动作
        if prob_random_action == 1:
            return self.get_random_actions(batch)

        # 计算编码状态的均值作为起始符号
        sos_token = reduce(encoded_state, 'b ... d -> b 1 d', 'mean')
        # 可能附加动作到 tokens
        tokens = self.maybe_append_actions(sos_token, actions = actions)

        # 初始化动作 bins 和缓存
        action_bins = []
        cache = None

        # 遍历动作数量
        for action_idx in range(self.num_actions):

            # 使用 transformer 进行转换
            embed, cache = self.transformer(
                tokens,
                context = encoded_state,
                cache = cache,
                return_cache = True
            )

            # 获取最后一个嵌入向量
            last_embed = embed[:, action_idx]
            # 获取动作 bins 的嵌入向量
            bin_embeddings = self.action_bin_embeddings[action_idx]

            # 计算 Q 值
            q_values = einsum('b d, a d -> b a', last_embed, bin_embeddings)

            # 如果随机动作概率大于0
            if prob_random_action > 0.:
                # 创建随机掩码
                random_mask = torch.zeros_like(selected_action_bins).float().uniform_(0., 1.) < prob_random_action
                # 获取随机动作
                random_actions = self.get_random_actions(batch, 1)
                random_actions = rearrange(random_actions, '... 1 -> ...')

                # 根据随机掩码替换选定的动作 bins
                selected_action_bins = torch.where(
                    random_mask,
                    random_actions,
                    selected_action_bins
                )

            # 获取下一个动作的嵌入向量
            next_action_embed = bin_embeddings[selected_action_bins]

            # 更新 tokens
            tokens, _ = pack((tokens, next_action_embed), 'b * d')

            # 添加选定的动作 bins
            action_bins.append(selected_action_bins)

        # 将动作 bins 堆叠在一起
        action_bins = torch.stack(action_bins, dim = -1)

        # 如果不需要返回 Q 值，则返回动作 bins
        if not return_q_values:
            return action_bins

        # 获取所有 Q 值
        all_q_values = self.get_q_values(embed)
        return action_bins, all_q_values

    def forward(
        self,
        encoded_state: Tensor,
        actions: Optional[Tensor] = None
    ):
        """
        einops
        b - batch
        n - number of actions
        a - action bins
        d - dimension
        """

        # 计算编码状态的均值作为起始符号
        sos_token = reduce(encoded_state, 'b ... d -> b 1 d', 'mean')

        # 可能附加动作到 tokens
        tokens = self.maybe_append_actions(sos_token, actions = actions)

        # 使用 transformer 进行转换
        embed = self.transformer(tokens, context = encoded_state)

        # 返回 Q 值
        return self.get_q_values(embed)
# 定义一个名为 QRoboticTransformer 的类，继承自 Module 类
class QRoboticTransformer(Module):

    # 初始化方法，接受多个参数
    @beartype
    def __init__(
        self,
        *,
        vit: Union[Dict[str, Any], MaxViT],  # 接受一个字典或 MaxViT 类型的参数 vit
        num_actions = 8,                     # 默认参数 num_actions 为 8
        action_bins = 256,                   # 默认参数 action_bins 为 256
        depth = 6,                           # 默认参数 depth 为 6
        heads = 8,                           # 默认参数 heads 为 8
        dim_head = 64,                       # 默认参数 dim_head 为 64
        token_learner_ff_mult = 2,           # 默认参数 token_learner_ff_mult 为 2
        token_learner_num_layers = 2,       # 默认参数 token_learner_num_layers 为 2
        token_learner_num_output_tokens = 8, # 默认参数 token_learner_num_output_tokens 为 8
        cond_drop_prob = 0.2,                # 默认参数 cond_drop_prob 为 0.2
        use_attn_conditioner = False,        # 默认参数 use_attn_conditioner 为 False
        conditioner_kwargs: dict = dict(),   # 默认参数 conditioner_kwargs 为一个空字典
        dueling = False,                     # 默认参数 dueling 为 False
        flash_attn = True,                   # 默认参数 flash_attn 为 True
        condition_on_text = True,            # 默认参数 condition_on_text 为 True
        q_head_attn_kwargs: dict = dict(     # 默认参数 q_head_attn_kwargs 为一个字典
            attn_heads = 8,                  # 字典中的键值对
            attn_dim_head = 64,              # 字典中的键值对
            attn_depth = 2                   # 字典中的键值对
        ),
        weight_tie_action_bin_embed = True   # 默认参数 weight_tie_action_bin_embed 为 True
    ):
        super().__init__()  # 调用父类的初始化方法

        # 根据传入的 vit 参数类型进行处理
        if isinstance(vit, dict):
            vit = MaxViT(**vit)

        self.vit = vit  # 将处理后的 vit 赋值给实例变量

        self.num_vit_stages = len(vit.cond_hidden_dims)  # 计算 vit.cond_hidden_dims 的长度并赋值给实例变量

        attend_dim = vit.embed_dim  # 将 vit.embed_dim 赋值给 attend_dim

        # q-transformer 相关的动作嵌入

        assert num_actions >= 1  # 断言 num_actions 大于等于 1

        self.num_actions = num_actions  # 将 num_actions 赋值给实例变量
        self.is_single_action = num_actions == 1  # 判断 num_actions 是否等于 1，并将结果赋值给实例变量
        self.action_bins = action_bins  # 将 action_bins 赋值给实例变量

        # 条件

        self.condition_on_text = condition_on_text  # 将 condition_on_text 赋值给实例变量

        # 根据 condition_on_text 的值选择不同的条件器类
        if condition_on_text:
            conditioner_klass = AttentionTextConditioner if use_attn_conditioner else TextConditioner

            self.conditioner = conditioner_klass(
                hidden_dims = (*tuple(vit.cond_hidden_dims), *((attend_dim,) * depth * 2)),
                hiddens_channel_first = (*((True,) * self.num_vit_stages), *((False,) * depth * 2)),
                cond_drop_prob = cond_drop_prob,
                **conditioner_kwargs
            )
        else:
            self.conditioner = NullConditioner(hidden_dims = tuple())

        self.token_learner = TokenLearner(
            dim = vit.embed_dim,
            ff_mult = token_learner_ff_mult,
            num_output_tokens = token_learner_num_output_tokens,
            num_layers = token_learner_num_layers
        )

        self.num_learned_tokens = token_learner_num_output_tokens  # 将 token_learner_num_output_tokens 赋值给实例变量

        self.transformer_depth = depth  # 将 depth 赋值给实例变量

        self.transformer = Transformer(
            dim = attend_dim,
            dim_head = dim_head,
            heads = heads,
            depth = depth,
            flash_attn = flash_attn,
            adaptive_ln = condition_on_text,
            final_norm = True
        )

        self.cond_drop_prob = cond_drop_prob  # 将 cond_drop_prob 赋值给实例变量

        # Q 头

        # 根据 is_single_action 的值选择不同的 QHead 类
        if self.is_single_action:
            self.q_head = QHeadSingleAction(
                attend_dim,
                num_learned_tokens = self.num_learned_tokens,
                action_bins = action_bins,
                dueling = dueling
            )
        else:
            self.q_head = QHeadMultipleActions(
                attend_dim,
                action_bins = action_bins,
                dueling = dueling,
                weight_tie_action_bin_embed = weight_tie_action_bin_embed,
                **q_head_attn_kwargs
            )

    # 定义一个 device 属性，返回参数的设备信息
    @property
    def device(self):
        return next(self.parameters()).device

    # 获取随机动作的方法
    def get_random_actions(self, batch_size = 1):
        return self.q_head.get_random_actions(batch_size)

    # 嵌入文本的方法
    @beartype
    def embed_texts(self, texts: List[str]):
        return self.conditioner.embed_texts(texts)

    # 获取最优动作的方法
    @torch.no_grad()
    def get_optimal_actions(
        self,
        *args,
        return_q_values = False,
        actions: Optional[Tensor] = None,
        **kwargs
    ):
        encoded_state = self.encode_state(*args, **kwargs)
        return self.q_head.get_optimal_actions(encoded_state, return_q_values = return_q_values, actions = actions)
    # 获取动作函数，根据给定的视频数据和参数返回动作
    def get_actions(
        self,
        video,
        *args,
        prob_random_action = 0.,  # 否则在强化学习中称为 epsilon
        **kwargs,
    ):
        # 获取视频数据的批处理大小
        batch_size = video.shape[0]
        # 确保随机动作概率在 [0, 1] 之间
        assert 0. <= prob_random_action <= 1.

        # 如果随机数小于随机动作概率，则返回随机动作
        if random() < prob_random_action:
            return self.get_random_actions(batch_size = batch_size)

        # 否则返回最优动作
        return self.get_optimal_actions(video, *args, **kwargs)

    # 编码状态函数，根据视频数据、文本、动作等参数编码状态
    def encode_state(
        self,
        video: Tensor,
        texts: Optional[Union[List[str], Tuple[str]]] = None,
        text_embeds: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
        cond_drop_prob = 0.,
    ):
        """
        einops
        b - batch
        c - channels
        f - frames
        h - height
        w - width
        n - number of learned tokens
        """

        # 如果不是基于文本条件，则不应传入文本或文本嵌入
        if not self.condition_on_text:
            assert (not exists(texts) and not exists(text_embeds)), 'neither texts nor text embeds should be passed in'
        else:
            # 如果基于文本条件，则必须传入文本或文本嵌入
            assert exists(texts) ^ exists(text_embeds), 'either texts or text embeds must be passed in if conditioning on instructions'

        # 如果传入的文本是元组，则转换为列表
        if exists(texts) and isinstance(texts, tuple):
            texts = list(texts)

        # 构建文本条件参数字典
        text_cond_kwargs = dict(texts = texts, text_embeds = text_embeds)

        # 获取变换器深度和条件丢弃概率
        depth = self.transformer_depth
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # 获取视频帧数和设备信息
        frames, device = video.shape[2], video.device

        # 获取条件函数列表
        cond_fns, _ = self.conditioner(
            **text_cond_kwargs,
            cond_drop_prob = cond_drop_prob,
            repeat_batch = (*((frames,) * self.num_vit_stages), *((1,) * self.transformer_depth * 2))
        )

        # 分离视觉 Transformer 和 Transformer 条件函数
        vit_cond_fns, transformer_cond_fns = cond_fns[:-(depth * 2)], cond_fns[-(depth * 2):]

        # 重排视频数据维度
        video = rearrange(video, 'b c f h w -> b f c h w')
        images, packed_shape = pack_one(video, '* c h w')

        # 使用 ViT 模型获取 tokens
        tokens = self.vit(
            images,
            texts = texts,
            cond_fns = vit_cond_fns,
            cond_drop_prob = cond_drop_prob,
            return_embeddings = True
        )

        tokens = unpack_one(tokens, packed_shape, '* c h w')
        learned_tokens = self.token_learner(tokens)

        tokens_per_frame = learned_tokens.shape[-1]
        learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')

        # 因果注意力掩码

        attn_mask = ~torch.ones((frames, frames), dtype = torch.bool, device = device).triu(1)
        attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = self.num_learned_tokens, r2 = self.num_learned_tokens)

        # 正弦位置嵌入

        pos_emb = posemb_sincos_1d(frames, learned_tokens.shape[-1], dtype = learned_tokens.dtype, device = learned_tokens.device)

        learned_tokens = learned_tokens + repeat(pos_emb, 'n d -> (n r) d', r = self.num_learned_tokens)

        # 注意力

        attended_tokens = self.transformer(learned_tokens, cond_fns = transformer_cond_fns, attn_mask = attn_mask)

        return attended_tokens

    # 前向传播函数，根据视频数据、文本、动作等参数执行前向传播
    @classifier_free_guidance
    def forward(
        self,
        video: Tensor,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
        cond_drop_prob = 0.,
        # 将输入数据移动到与机器人变换器相同的设备上
        video = video.to(self.device)

        # 如果存在动作数据，则将其移动到与机器人变换器相同的设备上
        if exists(actions):
            actions = actions.to(self.device)

        # 对状态进行编码
        encoded_state = self.encode_state(
            video = video,
            texts = texts,
            text_embeds = text_embeds,
            actions = actions,
            cond_drop_prob = cond_drop_prob
        )

        # 返回 Q 值的头部
        # 支持单个和多个动作
        if self.is_single_action:
            # 对于单个动作的机器人变换器，不应传入动作数据
            assert not exists(actions), 'actions should not be passed in for single action robotic transformer'
            q_values = self.q_head(encoded_state)
        else:
            q_values = self.q_head(encoded_state, actions = actions)

        # 返回 Q 值
        return q_values
```