# `.\lucidrains\imagen-pytorch\imagen_pytorch\imagen_video.py`

```
# 导入数学、操作符、函数工具等模块
import math
import operator
import functools
from tqdm.auto import tqdm
from functools import partial, wraps
from pathlib import Path

# 导入 PyTorch 相关模块
import torch
import torch.nn.functional as F
from torch import nn, einsum

# 导入 einops 相关模块
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# 导入自定义模块
from imagen_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 返回输入值
def identity(t, *args, **kwargs):
    return t

# 返回数组的第一个元素，如果数组为空则返回默认值
def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]

# 检查一个数是否能被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 可能执行函数，如果输入值不存在则直接返回
def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)
    return inner

# 仅执行一次函数，用于打印信息
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

# 仅打印一次信息
print_once = once(print)

# 返回默认值或默认函数的值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 将输入值转换为元组
def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

# 将 uint8 类型的图像转换为 float 类型
def cast_uint8_images_to_float(images):
    if not images.dtype == torch.uint8:
        return images
    return images / 255

# 获取模块的设备信息
def module_device(module):
    return next(module.parameters()).device

# 初始化权重为零
def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

# 模型评估装饰器
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 将元组填充到指定长度
def pad_tuple_to_length(t, length, fillvalue = None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))

# 辅助类

# 简单的返回输入值的模块
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

# 创建序列模块
def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# 张量辅助函数

# 对数函数
def log(t, eps: float = 1e-12):
    return torch.log(t.clamp(min = eps))

# L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# 将右侧维度填充到相同维度
def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# 带掩码的均值计算
def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)

# 调整视频大小
def resize_video_to(
    video,
    target_image_size,
    target_frames = None,
    clamp_range = None,
    mode = 'nearest'
):
    orig_video_size = video.shape[-1]

    frames = video.shape[2]
    target_frames = default(target_frames, frames)

    target_shape = (target_frames, target_image_size, target_image_size)

    if tuple(video.shape[-3:]) == target_shape:
        return video

    out = F.interpolate(video, target_shape, mode = mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)
        
    return out

# 缩放视频时间
def scale_video_time(
    video,
    downsample_scale = 1,
    mode = 'nearest'
):
    if downsample_scale == 1:
        return video

    image_size, frames = video.shape[-1], video.shape[-3]
    assert divisible_by(frames, downsample_scale), f'trying to temporally downsample a conditioning video frames of length {frames} by {downsample_scale}, however it is not neatly divisible'

    target_frames = frames // downsample_scale
    # 调用 resize_video_to 函数，将视频调整大小为指定尺寸
    resized_video = resize_video_to(
        video,  # 原始视频
        image_size,  # 目标图像尺寸
        target_frames = target_frames,  # 目标帧数
        mode = mode  # 调整模式
    )

    # 返回调整大小后的视频
    return resized_video
# classifier free guidance functions

# 根据给定形状、概率和设备创建一个布尔类型的掩码
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

# norms and residuals

# Layer normalization模块
class LayerNorm(nn.Module):
    def __init__(self, dim, stable=False):
        super().__init__()
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

# 通道层规范化模块
class ChanLayerNorm(nn.Module):
    def __init__(self, dim, stable=False):
        super().__init__()
        self.stable = stable
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim=1, keepdim=True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

# 始终返回相同值的类
class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val

# 残差连接模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 并行执行多个函数模块
class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)

# rearranging

# 时间为中心的重排模块
class RearrangeTimeCentric(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = rearrange(x, 'b c f ... -> b ... f c')
        x, ps = pack([x], '* f c')

        x = self.fn(x)

        x, = unpack(x, ps, '* f c')
        x = rearrange(x, 'b ... f c -> b c f ...')
        return x

# attention pooling

# PerceiverAttention模块
class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        scale=8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.LayerNorm(dim)
        )
    # 前向传播函数，接收输入 x、潜在变量 latents 和可选的 mask
    def forward(self, x, latents, mask = None):
        # 对输入 x 进行归一化处理
        x = self.norm(x)
        # 对潜在变量 latents 进行归一化处理
        latents = self.norm_latents(latents)

        # 获取输入 x 的 batch 大小和头数
        b, h = x.shape[0], self.heads

        # 生成查询向量 q
        q = self.to_q(latents)

        # 将输入 x 和潜在变量 latents 连接起来，作为键值对的输入
        kv_input = torch.cat((x, latents), dim = -2)
        # 将连接后的输入转换为键和值
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        # 对查询、键、值进行维度重排
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 对查询和键进行 L2 归一化
        q, k = map(l2norm, (q, k))
        # 对查询和键进行缩放
        q = q * self.q_scale
        k = k * self.k_scale

        # 计算相似度矩阵
        sim = einsum('... i d, ... j d  -> ... i j', q, k) * self.scale

        # 如果存在 mask，则进行填充和掩码处理
        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # 计算注意力权重
        attn = sim.softmax(dim = -1)

        # 计算输出
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        # 返回输出结果
        return self.to_out(out)
# 定义 PerceiverResampler 类，继承自 nn.Module
class PerceiverResampler(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_latents_mean_pooled = 4, # 从序列的均值池化表示派生的潜在变量数量
        max_seq_len = 512,
        ff_mult = 4
    ):
        super().__init__()
        # 创建位置嵌入层
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # 初始化潜在变量
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        # 如果均值池化的潜在变量数量大于0，则创建相应的层
        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange('b (n d) -> b n d', n = num_latents_mean_pooled)
            )

        # 创建多层感知器
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    # 前向传播函数
    def forward(self, x, mask = None):
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device = device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        # 如果存在均值池化的潜在变量，则将其与原始潜在变量拼接
        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(x, dim = 1, mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim = -2)

        # 遍历每一层的注意力机制和前馈网络
        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask = mask) + latents
            latents = ff(latents) + latents

        return latents

# 定义 Conv3d 类，继承自 nn.Module
class Conv3d(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        *,
        temporal_kernel_size = None,
        **kwargs
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)

        # 创建空���卷积层
        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size = kernel_size, padding = kernel_size // 2)
        # 创建时间卷积层（如果 kernel_size 大于1）
        self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size = temporal_kernel_size) if kernel_size > 1 else None
        self.kernel_size = kernel_size

        # 初始化时间卷积层的权重为单位矩阵
        if exists(self.temporal_conv):
            nn.init.dirac_(self.temporal_conv.weight.data) # initialized to be identity
            nn.init.zeros_(self.temporal_conv.bias.data)

    # 前向传播函数
    def forward(
        self,
        x,
        ignore_time = False
    ):
        b, c, *_, h, w = x.shape

        is_video = x.ndim == 5
        ignore_time &= is_video

        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) c h w')

        x = self.spatial_conv(x)

        if is_video:
            x = rearrange(x, '(b f) c h w -> b c f h w', b = b)

        if ignore_time or not exists(self.temporal_conv):
            return x

        x = rearrange(x, 'b c f h w -> (b h w) c f')

        # 因果时间卷积 - 时间在 imagen-video 中是因果的

        if self.kernel_size > 1:
            x = F.pad(x, (self.kernel_size - 1, 0))

        x = self.temporal_conv(x)

        x = rearrange(x, '(b h w) c f -> b c f h w', h = h, w = w)

        return x

# 定义 Attention 类，继承自 nn.Module
class Attention(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False,
        context_dim = None,
        rel_pos_bias = False,
        rel_pos_bias_mlp_depth = 2,
        init_zero = False,
        scale = 8
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置缩放因子和是否因果的标志
        self.scale = scale
        self.causal = causal

        # 如果启用相对位置偏置，则创建动态位置偏置对象
        self.rel_pos_bias = DynamicPositionBias(dim = dim, heads = heads, depth = rel_pos_bias_mlp_depth) if rel_pos_bias else None

        # 初始化头数和内部维度
        self.heads = heads
        inner_dim = dim_head * heads

        # 初始化 LayerNorm
        self.norm = LayerNorm(dim)

        # 初始化空注意力偏置和空键值对
        self.null_attn_bias = nn.Parameter(torch.randn(heads))
        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        # 初始化缩放参数
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        # 如果存在上下文维度，则初始化上下文处理层
        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(context_dim) else None

        # 初始化输出层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

        # 如果初始化为零，则将输出层的偏置初始化为零
        if init_zero:
            nn.init.zeros_(self.to_out[-1].g)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_bias = None
    ):
        # 获取输入张量的形状和设备信息
        b, n, device = *x.shape[:2], x.device

        # 对输入张量进行 LayerNorm 处理
        x = self.norm(x)
        # 分别计算查询、键、值
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        # 将查询张量重排为多头形式
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # 添加空键/值以用于分类器的先验网络引导
        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # 如果存在上下文，则添加文本条件
        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # 对查询、键进行 L2 归一化
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # 计算查询/键的相似性
        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        # 相对位置编码（T5 风格）
        if not exists(attn_bias) and exists(self.rel_pos_bias):
            attn_bias = self.rel_pos_bias(n, device = device, dtype = q.dtype)

        if exists(attn_bias):
            null_attn_bias = repeat(self.null_attn_bias, 'h -> h n 1', n = n)
            attn_bias = torch.cat((null_attn_bias, attn_bias), dim = -1)
            sim = sim + attn_bias

        # 掩码
        max_neg_value = -torch.finfo(sim.dtype).max

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # 注意力
        attn = sim.softmax(dim = -1)

        # 聚合值
        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# 定义一个伪 Conv2d 函数，使用 Conv3d 但在帧维度上使用大小为1的卷积核
def Conv2d(dim_in, dim_out, kernel, stride = 1, padding = 0, **kwargs):
    # 将 kernel 转换为元组
    kernel = cast_tuple(kernel, 2)
    # 将 stride 转换为元组
    stride = cast_tuple(stride, 2)
    # 将 padding 转换为元组
    padding = cast_tuple(padding, 2)

    # 如果 kernel 的长度为2，则在前面添加1
    if len(kernel) == 2:
        kernel = (1, *kernel)

    # 如果 stride 的长度为2，则在前面添加1
    if len(stride) == 2:
        stride = (1, *stride)

    # 如果 padding 的长度为2，则在前面添加0
    if len(padding) == 2:
        padding = (0, *padding)

    # 返回一个 Conv3d 对象
    return nn.Conv3d(dim_in, dim_out, kernel, stride = stride, padding = padding, **kwargs)

# 定义一个 Pad 类
class Pad(nn.Module):
    def __init__(self, padding, value = 0.):
        super().__init__()
        self.padding = padding
        self.value = value

    # 前向传播函数
    def forward(self, x):
        return F.pad(x, self.padding, value = self.value)

# 定义一个 Upsample 函数
def Upsample(dim, dim_out = None):
    dim_out = default(dim_out, dim)

    # 返回一个包含 Upsample 和 Conv2d 的序列
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        Conv2d(dim, dim_out, 3, padding = 1)
    )

# 定义一个 PixelShuffleUpsample 类
class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU()
        )

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.init_conv_(conv)

    # 初始化卷积层的权重
    def init_conv_(self, conv):
        o, i, f, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, f, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    # 前向传播函数
    def forward(self, x):
        out = self.net(x)
        frames = x.shape[2]
        out = rearrange(out, 'b c f h w -> (b f) c h w')
        out = self.pixel_shuffle(out)
        return rearrange(out, '(b f) c h w -> b c f h w', f = frames)

# 定义一个 Downsample 函数
def Downsample(dim, dim_out = None):
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c f (h p1) (w p2) -> b (c p1 p2) f h w', p1 = 2, p2 = 2),
        Conv2d(dim * 4, dim_out, 1)
    )

# 定义一个 TemporalPixelShuffleUpsample 类
class TemporalPixelShuffleUpsample(nn.Module):
    def __init__(self, dim, dim_out = None, stride = 2):
        super().__init__()
        self.stride = stride
        dim_out = default(dim_out, dim)
        conv = nn.Conv1d(dim, dim_out * stride, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU()
        )

        self.pixel_shuffle = Rearrange('b (c r) n -> b c (n r)', r = stride)

        self.init_conv_(conv)

    # 初始化卷积层的权重
    def init_conv_(self, conv):
        o, i, f = conv.weight.shape
        conv_weight = torch.empty(o // self.stride, i, f)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.stride)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    # 前向传播函数
    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b h w) c f')
        out = self.net(x)
        out = self.pixel_shuffle(out)
        return rearrange(out, '(b h w) c f -> b c f h w', h = h, w = w)

# 定义一个 TemporalDownsample 函数
def TemporalDownsample(dim, dim_out = None, stride = 2):
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c (f p) h w -> b (c p) f h w', p = stride),
        Conv2d(dim * stride, dim_out, 1)
    )

# 定义一个 SinusoidalPosEmb 类
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    # 前向传播函数
    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1)

# 定义一个 LearnedSinusoidalPosEmb 类
class LearnedSinusoidalPosEmb(nn.Module):
    # 初始化函数，接受维度参数
    def __init__(self, dim):
        # 调用父类的初始化函数
        super().__init__()
        # 断言维度为偶数
        assert (dim % 2) == 0
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
        # 将正弦和余弦值拼接在一起
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # 将输入张量 x 和频率值拼接在一起
        fouriered = torch.cat((x, fouriered), dim = -1)
        # 返回拼接后的张量
        return fouriered
class Block(nn.Module):
    # 定义一个块模块，包含归一化、激活函数和卷积操作
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        norm = True
    ):
        super().__init__()
        # 初始化 GroupNorm 归一化层，如果不需要归一化则使用 Identity 函数
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        # 初始化激活函数为 SiLU
        self.activation = nn.SiLU()
        # 初始化卷积操作，输出维度为 dim_out，卷积核大小为 3，填充为 1
        self.project = Conv3d(dim, dim_out, 3, padding = 1)

    # 前向传播函数，对输入进行归一化、缩放平移、激活和卷积操作
    def forward(
        self,
        x,
        scale_shift = None,
        ignore_time = False
    ):
        # 对输入进行归一化
        x = self.groupnorm(x)

        # 如果有缩放平移参数，则对输入进行缩放平移操作
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        # 对归一化后的输入进行激活函数操作
        x = self.activation(x)
        # 返回卷积操作后的结果
        return self.project(x, ignore_time = ignore_time)

class ResnetBlock(nn.Module):
    # 定义一个 ResNet 块模块，包含时间 MLP、交叉注意力、块模块和全局上下文注意力
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        groups = 8,
        linear_attn = False,
        use_gca = False,
        squeeze_excite = False,
        **attn_kwargs
    ):
        super().__init__()

        self.time_mlp = None

        # 如果存在时间条件维度，则初始化时间 MLP
        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        # 如果存在条件维度，则初始化交叉注意力模块
        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = attn_klass(
                dim = dim_out,
                context_dim = cond_dim,
                **attn_kwargs
            )

        # 初始化两个块模块
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)

        # 如果使用全局上下文注意力，则初始化全局上下文模块
        self.gca = GlobalContext(dim_in = dim_out, dim_out = dim_out) if use_gca else Always(1)

        # 如果输入维度不等于输出维度，则初始化卷积操作
        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()


    # 前向传播函数，包括时间 MLP、交叉注意力、块模块和全局上下文注意力的操作
    def forward(
        self,
        x,
        time_emb = None,
        cond = None,
        ignore_time = False
    ):

        scale_shift = None
        # 如果存在时间 MLP 和时间嵌入，则进行时间 MLP 操作
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        # 第一个块模块操作
        h = self.block1(x, ignore_time = ignore_time)

        # 如果存在交叉注意力模块，则进行交叉注意力操作
        if exists(self.cross_attn):
            assert exists(cond)
            h = rearrange(h, 'b c ... -> b ... c')
            h, ps = pack([h], 'b * c')

            h = self.cross_attn(h, context = cond) + h

            h, = unpack(h, ps, 'b * c')
            h = rearrange(h, 'b ... c -> b c ...')

        # 第二个块模块操作
        h = self.block2(h, scale_shift = scale_shift, ignore_time = ignore_time)

        # 全局上下文注意力操作
        h = h * self.gca(h)

        # 返回结果加上残差连接
        return h + self.res_conv(x)

class CrossAttention(nn.Module):
    # 定义交叉注意力模块，包含查询、键值映射和输出映射
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        norm_context = False,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        # 初始化 LayerNorm 归一化层
        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        # 初始化查询映射和键值映射
        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        # 初始化输出映射
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )
    # 定义前向传播函数，接受输入 x、上下文 context 和可选的掩码 mask
    def forward(self, x, context, mask = None):
        # 获取输入 x 的形状信息，包括 batch 大小 b、序列长度 n、设备信息 device
        b, n, device = *x.shape[:2], x.device

        # 对输入 x 和上下文 context 进行归一化处理
        x = self.norm(x)
        context = self.norm_context(context)

        # 将输入 x 转换为查询 q，上下文 context 转换为键 k 和值 v
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        # 将查询 q、键 k 和值 v 重排为多头注意力的形式
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # 为先验网络添加空键/值，用于无分类器干预的指导
        nk, nv = map(lambda t: repeat(t, 'd -> b h 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # 对查询 q 和键 k 进行 L2 归一化处理
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # 计算相似度矩阵
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # 掩码处理
        max_neg_value = -torch.finfo(sim.dtype).max
        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # 对相似度矩阵进行 softmax 操作，得到注意力权重
        attn = sim.softmax(dim = -1, dtype = torch.float32)

        # 根据注意力权重计算输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 返回输出结果
        return self.to_out(out)
class LinearCrossAttention(CrossAttention):
    # 线性交叉注意力类，继承自CrossAttention类
    def forward(self, x, context, mask = None):
        # 前向传播函数，接受输入x、上下文context和掩码mask，默认为None
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        # 对输入x进行规范化
        context = self.norm_context(context)
        # 对上下文context进行规范化

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        # 将输入x和上下文context转换为查询q、键k和值v

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (q, k, v))
        # 重排查询q、键k和值v的维度

        # add null key / value for classifier free guidance in prior net
        # 为先前网络中的无分类器自由指导添加空键/值

        nk, nv = map(lambda t: repeat(t, 'd -> (b h) 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # masking
        # 掩码处理

        max_neg_value = -torch.finfo(x.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b n -> b n 1')
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.)

        # linear attention
        # 线性注意力

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = self.heads)
        return self.to_out(out)

class LinearAttention(nn.Module):
    # 线性注意力类，继承自nn.Module类
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8,
        dropout = 0.05,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            Conv2d(dim, inner_dim, 1, bias = False),
            Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            Conv2d(dim, inner_dim, 1, bias = False),
            Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            Conv2d(dim, inner_dim, 1, bias = False),
            Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, inner_dim * 2, bias = False)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            Conv2d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap, context = None):
        # 前向传播函数，接受特征图fmap和上下文context，默认为None
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            ck, cv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (ck, cv))
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

class GlobalContext(nn.Module):
    # 全局上下文类，继承自nn.Module类
    """ basically a superior form of squeeze-excitation that is attention-esque """
    # 基本上是一种类似于注意力的优越形式的挤压激励

    def __init__(
        self,
        *,
        dim_in,
        dim_out
        # 初始化函数，接受输入维度dim_in和输出维度dim_out
    # 定义一个继承自 nn.Module 的类，用于实现一个自定义的注意力机制模块
    ):
        # 调用父类的构造函数
        super().__init__()
        # 定义一个将输入特征维度转换为 K 维度的卷积层
        self.to_k = Conv2d(dim_in, 1, 1)
        # 计算隐藏层维度，取最大值为 3 或者输出维度的一半
        hidden_dim = max(3, dim_out // 2)

        # 定义一个神经网络序列，包含卷积层、激活函数和输出层
        self.net = nn.Sequential(
            Conv2d(dim_in, hidden_dim, 1),
            nn.SiLU(),  # 使用 SiLU 激活函数
            Conv2d(hidden_dim, dim_out, 1),
            nn.Sigmoid()  # 使用 Sigmoid 激活函数
        )

    # 定义前向传播函数
    def forward(self, x):
        # 将输入 x 经过 to_k 卷积层得到 context
        context = self.to_k(x)
        # 对输入 x 和 context 进行维度重排
        x, context = map(lambda t: rearrange(t, 'b n ... -> b n (...)'), (x, context))
        # 使用 einsum 计算注意力权重并与输入 x 相乘
        out = einsum('b i n, b c n -> b c i', context.softmax(dim = -1), x)
        # 对输出 out 进行维度重排
        out = rearrange(out, '... -> ... 1 1')
        # 将处理后的 out 输入到神经网络序列中得到最终输出
        return self.net(out)
# 定义一个前馈神经网络模块，包含层归一化、线性层、GELU激活函数和线性层
def FeedForward(dim, mult = 2):
    # 计算隐藏层维度
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),  # 层归一化
        nn.Linear(dim, hidden_dim, bias = False),  # 线性层
        nn.GELU(),  # GELU激活函数
        LayerNorm(hidden_dim),  # 层归一化
        nn.Linear(hidden_dim, dim, bias = False)  # 线性层
    )

# 定义一个时间标记位移模块
class TimeTokenShift(nn.Module):
    def forward(self, x):
        if x.ndim != 5:
            return x

        x, x_shift = x.chunk(2, dim = 1)  # 将输入张量按维度1分块
        x_shift = F.pad(x_shift, (0, 0, 0, 0, 1, -1), value = 0.)  # 对x_shift进行填充
        return torch.cat((x, x_shift), dim = 1)  # 在维度1上连接张量x和x_shift

# 定义一个通道前馈神经网络模块
def ChanFeedForward(dim, mult = 2, time_token_shift = True):
    # 计算隐藏层维度
    hidden_dim = int(dim * mult)
    return Sequential(
        ChanLayerNorm(dim),  # 通道层归一化
        Conv2d(dim, hidden_dim, 1, bias = False),  # 二维卷积层
        nn.GELU(),  # GELU激活函数
        TimeTokenShift() if time_token_shift else None,  # 时间标记位移模块
        ChanLayerNorm(hidden_dim),  # 通道层归一化
        Conv2d(hidden_dim, dim, 1, bias = False)  # 二维卷积层
    )

# 定义一个Transformer块模块
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        ff_time_token_shift = True,
        context_dim = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim),  # 注意力机制
                ChanFeedForward(dim = dim, mult = ff_mult, time_token_shift = ff_time_token_shift)  # 通道前馈神经网络
            ]))

    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = rearrange(x, 'b c ... -> b ... c')  # 重新排列张量维度
            x, ps = pack([x], 'b * c')  # 打包张量

            x = attn(x, context = context) + x  # 注意力机制处理后与原始张量相加

            x, = unpack(x, ps, 'b * c')  # 解包张量
            x = rearrange(x, 'b ... c -> b c ...')  # 重新排列张量维度

            x = ff(x) + x  # 通道前馈神经网络处理后与原始张量相加
        return x

# 定义一个线性注意力Transformer块模块
class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        ff_time_token_shift = True,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearAttention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim),  # 线性注意力机制
                ChanFeedForward(dim = dim, mult = ff_mult, time_token_shift = ff_time_token_shift)  # 通道前馈神经网络
            ]))

    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x  # 线性注意力机制处理后与原始张量相加
            x = ff(x) + x  # 通道前馈神经网络处理后与原始张量相加
        return x

# 定义一个交叉嵌入层模块
class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # 计算每个尺度的维度
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))  # 对输入张量进行卷积操作
        return torch.cat(fmaps, dim = 1)  # 在维度1上连接卷积结果

# 定义一个上采样合并器模块
class UpsampleCombiner(nn.Module):
    def __init__(
        self,
        dim,
        *,
        enabled = False,
        dim_ins = tuple(),
        dim_outs = tuple()
    # 初始化函数，设置输出维度和是否启用
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 将输出维度转换为元组，长度与输入维度相同
        dim_outs = cast_tuple(dim_outs, len(dim_ins))
        # 断言输入维度和输出维度长度相同
        assert len(dim_ins) == len(dim_outs)

        # 设置是否启用标志
        self.enabled = enabled

        # 如果未启用，则直接设置输出维度并返回
        if not self.enabled:
            self.dim_out = dim
            return

        # 根据输入维度和输出维度创建模块列表
        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        # 计算最终输出维度
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    # 前向传播函数，处理输入数据和特征图
    def forward(self, x, fmaps = None):
        # 获取输入数据的目标尺寸
        target_size = x.shape[-1]

        # 设置特征图为默认值空元组
        fmaps = default(fmaps, tuple())

        # 如果未启用或特征图为空或卷积模块为空，则直接返回输入数据
        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        # 将特征图调整为目标尺寸
        fmaps = [resize_video_to(fmap, target_size) for fmap in fmaps]
        # 对每个特征图应用对应的卷积模块
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        # 拼接输入数据和卷积结果，沿指定维度拼接
        return torch.cat((x, *outs), dim = 1)
# 定义一个动态位置偏置的神经网络模块
class DynamicPositionBias(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads,
        depth
    ):
        super().__init__()
        self.mlp = nn.ModuleList([])

        # 添加一个线性层、LayerNorm 和 SiLU 激活函数到 MLP 中
        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            LayerNorm(dim),
            nn.SiLU()
        ))

        # 根据深度添加多个线性层、LayerNorm 和 SiLU 激活函数到 MLP 中
        for _ in range(max(depth - 1, 0)):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                LayerNorm(dim),
                nn.SiLU()
            ))

        # 添加一个线性层到 MLP 中
        self.mlp.append(nn.Linear(dim, heads)

    # 前向传播函数
    def forward(self, n, device, dtype):
        # 创建张量 i 和 j
        i = torch.arange(n, device = device)
        j = torch.arange(n, device = device)

        # 计算位置索引
        indices = rearrange(i, 'i -> i 1') - rearrange(j, 'j -> 1 j')
        indices += (n - 1)

        # 创建位置张量
        pos = torch.arange(-n + 1, n, device = device, dtype = dtype)
        pos = rearrange(pos, '... -> ... 1')

        # 遍历 MLP 中的每一层
        for layer in self.mlp:
            pos = layer(pos)

        # 计算位置偏置
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

# 定义一个 3D UNet 神经网络模块
class Unet3D(nn.Module):
    def __init__(
        self,
        *,
        dim,
        text_embed_dim = get_encoded_dim(DEFAULT_T5_NAME),
        num_resnet_blocks = 1,
        cond_dim = None,
        num_image_tokens = 4,
        num_time_tokens = 2,
        learned_sinu_pos_emb_dim = 16,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        temporal_strides = 1,
        cond_images_channels = 0,
        channels = 3,
        channels_out = None,
        attn_dim_head = 64,
        attn_heads = 8,
        ff_mult = 2.,
        ff_time_token_shift = True,         # 在 feedforwards 的隐藏层中沿时间轴进行令牌移位
        lowres_cond = False,                # 用于级联扩散
        layer_attns = False,
        layer_attns_depth = 1,
        layer_attns_add_text_cond = True,   # 是否在自注意力块中加入文本嵌入
        attend_at_middle = True,            # 是否在瓶颈处进行一层注意力
        time_rel_pos_bias_depth = 2,
        time_causal_attn = True,
        layer_cross_attns = True,
        use_linear_attn = False,
        use_linear_cross_attn = False,
        cond_on_text = True,
        max_text_len = 256,
        init_dim = None,
        resnet_groups = 8,
        init_conv_kernel_size = 7,          # 初始卷积的内核大小
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        attn_pool_text = True,
        attn_pool_num_latents = 32,
        dropout = 0.,
        memory_efficient = False,
        init_conv_to_final_conv_residual = False,
        use_global_context_attn = True,
        scale_skip_connection = True,
        final_resnet_block = True,
        final_conv_kernel_size = 3,
        self_cond = False,
        combine_upsample_fmaps = False,      # 在所有上采样块中合并特征图
        pixel_shuffle_upsample = True,       # 可能解决棋盘伪影
        resize_mode = 'nearest'
    # 如果当前 UNet 的设置不正确，则重新初始化 UNet
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        text_embed_dim,
        channels,
        channels_out,
        cond_on_text
    # 如果当前对象的属性与传入参数相同，则直接返回当前对象
    ):
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_text == self.cond_on_text and \
            text_embed_dim == self._locals['text_embed_dim'] and \
            channels_out == self.channels_out:
            return self

        # 更新参数字典
        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            text_embed_dim = text_embed_dim,
            channels = channels,
            channels_out = channels_out,
            cond_on_text = cond_on_text
        )

        # 返回一个新的类实例，使用当前对象的属性和更新后的参数
        return self.__class__(**{**self._locals, **updated_kwargs})

    # 返回完整的unet配置及其参数状态字典的方法

    def to_config_and_state_dict(self):
        return self._locals, self.state_dict()

    # 从配置和状态字典中重新创建unet的类方法

    @classmethod
    def from_config_and_state_dict(klass, config, state_dict):
        unet = klass(**config)
        unet.load_state_dict(state_dict)
        return unet

    # 将unet持久化到磁盘的方法

    def persist_to_file(self, path):
        path = Path(path)
        path.parents[0].mkdir(exist_ok = True, parents = True)

        config, state_dict = self.to_config_and_state_dict()
        pkg = dict(config = config, state_dict = state_dict)
        torch.save(pkg, str(path))

    # 从使用`persist_to_file`保存的文件中重新创建unet的类方法

    @classmethod
    def hydrate_from_file(klass, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        assert 'config' in pkg and 'state_dict' in pkg
        config, state_dict = pkg['config'], pkg['state_dict']

        return Unet.from_config_and_state_dict(config, state_dict)

    # 带有分类器自由引导的前向传播

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        *,
        lowres_cond_img = None,
        lowres_noise_times = None,
        text_embeds = None,
        text_mask = None,
        cond_images = None,
        cond_video_frames = None,
        post_cond_video_frames = None,
        self_cond = None,
        cond_drop_prob = 0.,
        ignore_time = False
```