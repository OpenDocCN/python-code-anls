# `.\lucidrains\gigagan-pytorch\gigagan_pytorch\unet_upsampler.py`

```
# 从 math 模块中导入 log2 函数
from math import log2
# 从 functools 模块中导入 partial 函数
from functools import partial

# 导入 torch 库
import torch
# 从 torch 模块中导入 nn 模块
from torch import nn
# 从 torch.nn 模块中导入 functional 模块
import torch.nn.functional as F

# 从 einops 库中导入 rearrange 和 repeat 函数，以及 Rearrange 类
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 从 gigagan_pytorch 模块中导入各个自定义类和函数
from gigagan_pytorch.attend import Attend
from gigagan_pytorch.gigagan_pytorch import (
    BaseGenerator,
    StyleNetwork,
    AdaptiveConv2DMod,
    TextEncoder,
    CrossAttentionBlock,
    Upsample
)

# 从 beartype 库中导入 beartype 函数和相关类型注解
from beartype import beartype
from beartype.typing import Optional, List, Union, Dict, Iterable

# 辅助函数

# 判断变量是否存在
def exists(x):
    return x is not None

# 返回默认值函数
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 将输入转换为元组
def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

# 返回输入本身的函数
def identity(t, *args, **kwargs):
    return t

# 判断一个数是否为2的幂
def is_power_of_two(n):
    return log2(n).is_integer()

# 生成无限循环的迭代器
def null_iterator():
    while True:
        yield None

# 小型辅助模块

# 像素混洗上采样类
class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)

        # 创建卷积层对象
        conv = nn.Conv2d(dim, dim_out * 4, 1)
        self.init_conv_(conv)

        # 定义网络结构
        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

    # 初始化卷积层权重
    def init_conv_(self, conv):
        o, *rest_shape = conv.weight.shape
        conv_weight = torch.empty(o // 4, *rest_shape)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    # 前向传播函数
    def forward(self, x):
        return self.net(x)

# 下采样函数
def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

# RMS 归一化类
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    # 前向传播函数
    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# 构建块模块

# 基础块类
class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        num_conv_kernels = 0
    ):
        super().__init__()
        self.proj = AdaptiveConv2DMod(dim, dim_out, kernel = 3, num_conv_kernels = num_conv_kernels)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    # 前向传播函数
    def forward(
        self,
        x,
        conv_mods_iter: Optional[Iterable] = None
    ):
        conv_mods_iter = default(conv_mods_iter, null_iterator())

        x = self.proj(
            x,
            mod = next(conv_mods_iter),
            kernel_mod = next(conv_mods_iter)
        )

        x = self.norm(x)
        x = self.act(x)
        return x

# ResNet 块类
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        groups = 8,
        num_conv_kernels = 0,
        style_dims: List = []
    ):
        super().__init__()
        style_dims.extend([
            dim,
            num_conv_kernels,
            dim_out,
            num_conv_kernels
        ])

        self.block1 = Block(dim, dim_out, groups = groups, num_conv_kernels = num_conv_kernels)
        self.block2 = Block(dim_out, dim_out, groups = groups, num_conv_kernels = num_conv_kernels)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # 前向传播函数
    def forward(
        self,
        x,
        conv_mods_iter: Optional[Iterable] = None
    ):
        h = self.block1(x, conv_mods_iter = conv_mods_iter)
        h = self.block2(h, conv_mods_iter = conv_mods_iter)

        return h + self.res_conv(x)

# 线性注意力类
class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    # 初始化函数，设置缩放因子和头数
    def __init__(
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        # 初始化 RMSNorm 层
        self.norm = RMSNorm(dim)
        # 创建卷积层，用于计算查询、键、值
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        # 创建输出层，包含卷积层和 RMSNorm 层
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    # 前向传播函数
    def forward(self, x):
        b, c, h, w = x.shape

        # 对输入进行归一化处理
        x = self.norm(x)

        # 将输入通过卷积层得到查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # 对查询和键进行 softmax 处理
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        # 对查询进行缩放
        q = q * self.scale

        # 计算上下文信息
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # 计算输出
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        # 初始化注意力机制模块
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        # 归一化层
        self.norm = RMSNorm(dim)
        # 注意力计算
        self.attend = Attend(flash = flash)

        # 将输入转换为查询、键、值
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        # 输出转换
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        # 归一化输入
        x = self.norm(x)

        # 将输入转换为查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        # 注意力计算
        out = self.attend(q, k, v)

        # 重排输出形状
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# feedforward

def FeedForward(dim, mult = 4):
    # 前馈神经网络
    return nn.Sequential(
        RMSNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Conv2d(dim * mult, dim, 1)
    )

# transformers

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 1,
        flash_attn = True,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        # 构建多层Transformer
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, flash = flash_attn),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class LinearTransformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 1,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        # 构建多层LinearTransformer
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

# model

class UnetUpsampler(BaseGenerator):

    @beartype
    def __init__(
        self,
        dim,
        *,
        image_size,
        input_image_size,
        init_dim = None,
        out_dim = None,
        text_encoder: Optional[Union[TextEncoder, Dict]] = None,
        style_network: Optional[Union[StyleNetwork, Dict]] = None,
        style_network_dim = None,
        dim_mults = (1, 2, 4, 8, 16),
        channels = 3,
        resnet_block_groups = 8,
        full_attn = (False, False, False, True, True),
        cross_attn = (False, False, False, True, True),
        flash_attn = True,
        self_attn_dim_head = 64,
        self_attn_heads = 8,
        self_attn_dot_product = True,
        self_attn_ff_mult = 4,
        attn_depths = (1, 1, 1, 1, 1),
        cross_attn_dim_head = 64,
        cross_attn_heads = 8,
        cross_ff_mult = 4,
        mid_attn_depth = 1,
        num_conv_kernels = 2,
        resize_mode = 'bilinear',
        unconditional = True,
        skip_connect_scale = None
    ):
        # 初始化UnetUpsampler模型
        super().__init__()

    @property
    def allowable_rgb_resolutions(self):
        # 计算允许的RGB分辨率
        input_res_base = int(log2(self.input_image_size))
        output_res_base = int(log2(self.image_size))
        allowed_rgb_res_base = list(range(input_res_base, output_res_base))
        return [*map(lambda p: 2 ** p, allowed_rgb_res_base)]

    @property
    def device(self):
        # 获取模型所在设备
        return next(self.parameters()).device

    @property
    def total_params(self):
        # 计算模型总参数数量
        return sum([p.numel() for p in self.parameters()])

    def resize_image_to(self, x, size):
        # 调整输入图像大小
        return F.interpolate(x, (size, size), mode = self.resize_mode)
    # 定义一个前向传播函数，接受低分辨率图像、风格、噪声、文本等参数，并返回RGB图像
    def forward(
        self,
        lowres_image,
        styles = None,
        noise = None,
        texts: Optional[List[str]] = None,
        global_text_tokens = None,
        fine_text_tokens = None,
        text_mask = None,
        return_all_rgbs = False,
        replace_rgb_with_input_lowres_image = True   # discriminator should also receive the low resolution image the upsampler sees
    ):
        # 将输入的低分辨率图像赋值给x
        x = lowres_image
        # 获取x的形状
        shape = x.shape
        # 获取批处理大小
        batch_size = shape[0]

        # 断言x的最后两个维度与输入图像大小相同
        assert shape[-2:] == ((self.input_image_size,) * 2)

        # 处理文本编码
        # 需要全局文本标记自适应选择主要贡献中的内核
        # 需要细节文本标记进行交叉注意力
        if not self.unconditional:
            if exists(texts):
                assert exists(self.text_encoder)
                global_text_tokens, fine_text_tokens, text_mask = self.text_encoder(texts)
            else:
                assert all([*map(exists, (global_text_tokens, fine_text_tokens, text_mask))])
        else:
            assert not any([*map(exists, (texts, global_text_tokens, fine_text_tokens))])

        # 风格
        if not exists(styles):
            assert exists(self.style_network)

            noise = default(noise, torch.randn((batch_size, self.style_network.dim), device = self.device))
            styles = self.style_network(noise, global_text_tokens)

        # 将风格投影到卷积调制
        conv_mods = self.style_to_conv_modulations(styles)
        conv_mods = conv_mods.split(self.style_embed_split_dims, dim = -1)
        conv_mods = iter(conv_mods)

        # 初始卷积
        x = self.init_conv(x)

        h = []

        # 下采样阶段
        for block1, block2, cross_attn, attn, downsample in self.downs:
            x = block1(x, conv_mods_iter = conv_mods)
            h.append(x)

            x = block2(x, conv_mods_iter = conv_mods)

            x = attn(x)

            if exists(cross_attn):
                x = cross_attn(x, context = fine_text_tokens, mask = text_mask)

            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, conv_mods_iter = conv_mods)
        x = self.mid_attn(x)
        x = self.mid_block2(x, conv_mods_iter = conv_mods)

        # rgbs
        rgbs = []

        init_rgb_shape = list(x.shape)
        init_rgb_shape[1] = self.channels

        rgb = self.mid_to_rgb(x)
        rgbs.append(rgb)

        # 上采样阶段
        for upsample, upsample_rgb, to_rgb, block1, block2, cross_attn, attn in self.ups:

            x = upsample(x)
            rgb = upsample_rgb(rgb)

            res1 = h.pop() * self.skip_connect_scale
            res2 = h.pop() * self.skip_connect_scale

            fmap_size = x.shape[-1]
            residual_fmap_size = res1.shape[-1]

            if residual_fmap_size != fmap_size:
                res1 = self.resize_image_to(res1, fmap_size)
                res2 = self.resize_image_to(res2, fmap_size)

            x = torch.cat((x, res1), dim = 1)
            x = block1(x, conv_mods_iter = conv_mods)

            x = torch.cat((x, res2), dim = 1)
            x = block2(x, conv_mods_iter = conv_mods)

            if exists(cross_attn):
                x = cross_attn(x, context = fine_text_tokens, mask = text_mask)

            x = attn(x)

            rgb = rgb + to_rgb(x)
            rgbs.append(rgb)

        x = self.final_res_block(x, conv_mods_iter = conv_mods)

        assert len([*conv_mods]) == 0

        rgb = rgb + self.final_to_rgb(x)

        if not return_all_rgbs:
            return rgb

        # 仅保留那些特征图大于要上采样的输入图像的rgbs
        rgbs = list(filter(lambda t: t.shape[-1] > shape[-1], rgbs))

        # 并将原始输入图像作为最小的rgb返回
        rgbs = [lowres_image, *rgbs]

        return rgb, rgbs
```