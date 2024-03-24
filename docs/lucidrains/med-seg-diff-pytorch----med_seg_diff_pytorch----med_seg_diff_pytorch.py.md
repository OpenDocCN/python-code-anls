# `.\lucidrains\med-seg-diff-pytorch\med_seg_diff_pytorch\med_seg_diff_pytorch.py`

```py
# 导入所需的库
import math
import copy
from random import random
from functools import partial
from collections import namedtuple

# 导入第三方库
from beartype import beartype

# 导入 PyTorch 库
import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.fft import fft2, ifft2

# 导入 einops 库
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# 导入 tqdm 库
from tqdm.auto import tqdm

# 定义常量
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# 辅助函数

# 判断变量是否存在
def exists(x):
    return x is not None

# 返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 返回输入本身
def identity(t, *args, **kwargs):
    return t

# 标准化函数

# 将图像标准化到 -1 到 1 之间
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

# 将标准化后的图像反标准化到 0 到 1 之间
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# 小型辅助模块

# 残差模块
class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# 上采样模块
def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

# 下采样模块
def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

# 层归一化模块
class LayerNorm(Module):
    def __init__(self, dim, bias = False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)

# 正弦位置编码模块
class SinusoidalPosEmb(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# 构建块模块

# 基础块模块
class Block(Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

# ResNet 块模块
class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

# 前馈网络模块
def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    # 返回一个包含多个层的神经网络模型
    return nn.Sequential(
        # 对输入数据进行层归一化
        LayerNorm(dim),
        # 1x1卷积层，将输入维度转换为inner_dim
        nn.Conv2d(dim, inner_dim, 1),
        # GELU激活函数
        nn.GELU(),
        # 1x1卷积层，将inner_dim维度转换为dim
        nn.Conv2d(inner_dim, dim, 1),
    )
class LinearAttention(Module):
    # 定义线性注意力机制模块
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    # 定义注意力机制模块
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class Transformer(Module):
    # 定义变压器模块
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 4,
        depth = 1
    ):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                Residual(Attention(dim, dim_head = dim_head, heads = heads)),
                Residual(FeedForward(dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

# vision transformer for dynamic ff-parser

class ViT(Module):
    # 定义视觉变压器模块
    def __init__(
        self,
        dim,
        *,
        image_size,
        patch_size,
        channels = 3,
        channels_out = None,
        dim_head = 32,
        heads = 4,
        depth = 4,
    ):
        super().__init__()
        assert exists(image_size)
        assert (image_size % patch_size) == 0

        num_patches_height_width = image_size // patch_size

        self.pos_emb = nn.Parameter(torch.zeros(dim, num_patches_height_width, num_patches_height_width))

        channels_out = default(channels_out, channels)

        patch_dim = channels * (patch_size ** 2)
        output_patch_dim = channels_out * (patch_size ** 2)

        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = patch_size, p2 = patch_size),
            nn.Conv2d(patch_dim, dim, 1),
            LayerNorm(dim)
        )

        self.transformer = Transformer(
            dim = dim,
            dim_head = dim_head,
            depth = depth
        )

        self.to_patches = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, output_patch_dim, 1),
            Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size),
        )

        nn.init.zeros_(self.to_patches[-2].weight)
        nn.init.zeros_(self.to_patches[-2].bias)
    # 定义前向传播函数，接收输入 x
    def forward(self, x):
        # 将输入 x 转换为 tokens
        x = self.to_tokens(x)
        # 将输入 x 与位置编码相加
        x = x + self.pos_emb

        # 使用 Transformer 处理输入 x
        x = self.transformer(x)
        # 将处理后的结果转换为 patches
        return self.to_patches(x)
# 定义一个名为 Conditioning 的类，继承自 Module 类
class Conditioning(Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        fmap_size,
        dim,
        dynamic = True,
        image_size = None,
        dim_head = 32,
        heads = 4,
        depth = 4,
        patch_size = 16
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个可学习的参数 ff_parser_attn_map，维度为 (dim, fmap_size, fmap_size)
        self.ff_parser_attn_map = nn.Parameter(torch.ones(dim, fmap_size, fmap_size))

        # 设置是否为动态模式
        self.dynamic = dynamic

        # 如果是动态模式
        if dynamic:
            # 创建一个 ViT 模型，用于动态调整 ff_parser_attn_map
            self.to_dynamic_ff_parser_attn_map = ViT(
                dim = dim,
                channels = dim * 2 * 2,  # 输入和条件的通道数，考虑到复数（实部和虚部）
                channels_out = dim,
                image_size = image_size,
                patch_size = patch_size,
                heads = heads,
                dim_head = dim_head
            )

        # 创建 LayerNorm 层，用于输入和条件的归一化
        self.norm_input = LayerNorm(dim, bias = True)
        self.norm_condition = LayerNorm(dim, bias = True)

        # 创建一个 ResnetBlock 模块
        self.block = ResnetBlock(dim, dim)

    # 前向传播函数，接受输入 x 和条件 c
    def forward(self, x, c):
        # 获取 ff_parser_attn_map 参数
        ff_parser_attn_map = self.ff_parser_attn_map

        # 对输入 x 进行二维傅立叶变换
        dtype = x.dtype
        x = fft2(x)

        # 如果是动态模式
        if self.dynamic:
            # 对条件 c 进行二维傅立叶变换
            c_complex = fft2(c)
            x_as_real, c_as_real = map(torch.view_as_real, (x, c_complex))
            x_as_real, c_as_real = map(lambda t: rearrange(t, 'b d h w ri -> b (d ri) h w'), (x_as_real, c_as_real))

            # 将 x 和 c 连接起来
            to_dynamic_input = torch.cat((x_as_real, c_as_real), dim = 1)

            # 使用 ViT 模型调整 ff_parser_attn_map
            dynamic_ff_parser_attn_map = self.to_dynamic_ff_parser_attn_map(to_dynamic_input)

            # 更新 ff_parser_attn_map
            ff_parser_attn_map = ff_parser_attn_map + dynamic_ff_parser_attn_map

        # 使用 ff_parser_attn_map 对 x 进行调制
        x = x * ff_parser_attn_map

        # 对 x 进行逆二维傅立叶变换，并取实部
        x = ifft2(x).real
        x = x.type(dtype)

        # 在论文中的公式 3
        # 对 x 和 c 进���归一化，然后相乘再乘以 c
        normed_x = self.norm_input(x)
        normed_c = self.norm_condition(c)
        c = (normed_x * normed_c) * c

        # 添加一个额外的块以允许更多信息的整合
        # 在 Condition 块之后有一个下采样（但也许有一个更好的地方可以进行条件化，而不是就在下采样之前）
        
        # 返回经过块处理后的 c
        return self.block(c)

# 定义一个名为 Unet 的类，继承自 Module 类
@beartype
class Unet(Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        image_size,
        mask_channels = 1,
        input_img_channels = 3,
        init_dim = None,
        out_dim = None,
        dim_mults: tuple = (1, 2, 4, 8),
        full_self_attn: tuple = (False, False, False, True),
        attn_dim_head = 32,
        attn_heads = 4,
        mid_transformer_depth = 1,
        self_condition = False,
        resnet_block_groups = 8,
        conditioning_klass = Conditioning,
        skip_connect_condition_fmaps = False,    # 是否在后续解码器上采样部分连接条件 fmaps
        dynamic_ff_parser_attn_map = False,      # 允许 ff-parser 根据输入动态调整。暂时排除条件
        conditioning_kwargs: dict = dict(
            dim_head = 32,
            heads = 4,
            depth = 4,
            patch_size = 16
        )
    ):
        # 调用父类的构造函数
        super().__init__()

        # 设置图像大小
        self.image_size = image_size

        # 确定维度

        # 输入图像通道数
        self.input_img_channels = input_img_channels
        # mask 通道数
        self.mask_channels = mask_channels
        # 是否自身条件
        self.self_condition = self_condition

        # 输出通道数为 mask 通道数
        output_channels = mask_channels
        # 如果有自身条件，mask 通道数变为原来的两倍，否则不变
        mask_channels = mask_channels * (2 if self_condition else 1)

        # 初始化维度为默认维度或者给定的维度
        init_dim = default(init_dim, dim)
        # 初始化卷积层，输入为 mask 通道数，输出为 init_dim，卷积核大小为 7x7，填充为 3
        self.init_conv = nn.Conv2d(mask_channels, init_dim, 7, padding = 3)
        # 条件初始化卷积层，输入为输入图像通道数，输出为 init_dim，卷积核大小为 7x7，填充为 3
        self.cond_init_conv = nn.Conv2d(input_img_channels, init_dim, 7, padding = 3)

        # 维度列表，包括初始化维度和后续维度的倍数
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # 输入输出维度对
        in_out = list(zip(dims[:-1], dims[1:]))

        # 部分 ResnetBlock 类的初始化
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # 时间嵌入维度
        time_dim = dim * 4

        # 时间 MLP 模型
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 注意力相关参数
        attn_kwargs = dict(
            dim_head = attn_dim_head,
            heads = attn_heads
        )

        # conditioner 设置

        if conditioning_klass == Conditioning:
            conditioning_klass = partial(
                Conditioning,
                dynamic = dynamic_ff_parser_attn_map,
                **conditioning_kwargs
            )

        # 层

        num_resolutions = len(in_out)
        assert len(full_self_attn) == num_resolutions

        # 条件器列表
        self.conditioners = ModuleList([])

        # 是否跳过连接条件特征图
        self.skip_connect_condition_fmaps = skip_connect_condition_fmaps

        # 下采样编码块
        self.downs = ModuleList([])

        curr_fmap_size = image_size

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out, full_self_attn)):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = Attention if full_attn else LinearAttention

            self.conditioners.append(conditioning_klass(curr_fmap_size, dim_in, image_size = curr_fmap_size))

            self.downs.append(ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(attn_klass(dim_in, **attn_kwargs)),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

            if not is_last:
                curr_fmap_size //= 2

        # 中间块

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_transformer = Transformer(mid_dim, depth = mid_transformer_depth, **attn_kwargs)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        # 条件编码路径与主编码路径相同

        self.cond_downs = copy.deepcopy(self.downs)
        self.cond_mid_block1 = copy.deepcopy(self.mid_block1)

        # 上采样解码块

        self.ups = ModuleList([])

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(in_out), reversed(full_self_attn))):
            is_last = ind == (len(in_out) - 1)
            attn_klass = Attention if full_attn else LinearAttention

            skip_connect_dim = dim_in * (2 if self.skip_connect_condition_fmaps else 1)

            self.ups.append(ModuleList([
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim = time_dim),
                Residual(attn_klass(dim_out, **attn_kwargs)),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        # 投影到预测

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, output_channels, 1)
    # 定义前向传播函数，接受输入 x、时间 time、条件 cond、自身条件 x_self_cond
    def forward(
        self,
        x,
        time,
        cond,
        x_self_cond = None
    ):
        # 获取输入 x 的数据类型和是否跳过连接的条件特征图
        dtype, skip_connect_c = x.dtype, self.skip_connect_condition_fmaps

        # 如果存在自身条件，将其与输入 x 进行拼接
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        # 对输入 x 进行初始卷积
        x = self.init_conv(x)
        # 复制输入 x 作为中间结果
        r = x.clone()

        # 对条件 cond 进行初始卷积
        c = self.cond_init_conv(cond)

        # 对时间 time 进行多层感知机处理
        t = self.time_mlp(time)

        # 初始化中间结果列表
        h = []

        # 遍历下采样模块、条件下采样模块和条件器
        for (block1, block2, attn, downsample), (cond_block1, cond_block2, cond_attn, cond_downsample), conditioner in zip(self.downs, self.cond_downs, self.conditioners):
            # 对输入 x 进行第一个块的处理
            x = block1(x, t)
            # 对条件 c 进行第一个块的处理
            c = cond_block1(c, t)

            # 将当前处理结果加入中间结果列表
            h.append([x, c] if skip_connect_c else [x])

            # 对输入 x 进行第二个块的处理
            x = block2(x, t)
            # 对条件 c 进行第二个块的处理
            c = cond_block2(c, t)

            # 对输入 x 进行注意力机制处理
            x = attn(x)
            # 对条件 c 进行注意力机制处理
            c = cond_attn(c)

            # 使用条件器对条件 c 进行处理
            c = conditioner(x, c)

            # 将当前处理结果加入中间结果列表
            h.append([x, c] if skip_connect_c else [x])

            # 对输入 x 进行下采样
            x = downsample(x)
            # 对条件 c 进行下采样
            c = cond_downsample(c)

        # 对输入 x 进行中间块1的处理
        x = self.mid_block1(x, t)
        # 对条件 c 进行中间块1的处理
        c = self.cond_mid_block1(c, t)

        # 将条件 c 加到输入 x 上
        x = x + c

        # 对输入 x 进行中间变换器处理
        x = self.mid_transformer(x)
        # 对输入 x 进行中间块2的处理
        x = self.mid_block2(x, t)

        # 遍历上采样模块
        for block1, block2, attn, upsample in self.ups:
            # 将中间结果与 h 中的结果拼接
            x = torch.cat((x, *h.pop()), dim = 1)
            # 对输入 x 进行第一个块的处理
            x = block1(x, t)

            # 将中间结果与 h 中的结果拼接
            x = torch.cat((x, *h.pop()), dim = 1)
            # 对输入 x 进行第二个块的处理
            x = block2(x, t)
            # 对输入 x 进行注意力机制处理
            x = attn(x)

            # 对输入 x 进行上采样
            x = upsample(x)

        # 将输入 x 与初始输入 r 拼接
        x = torch.cat((x, r), dim = 1)

        # 对拼接后的结果进行最终残差块处理
        x = self.final_res_block(x, t)
        # 返回最终卷积结果
        return self.final_conv(x)
# 高斯扩散训练器类

# 从输入张量 a 中提取指定索引 t 对应的值，并根据 x_shape 的形状重新组织输出
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 线性的 beta 调度函数，根据总步数 timesteps 计算出 beta 的线性变化范围
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

# 余弦形式的 beta 调度函数，根据总步数 timesteps 和参数 s 计算出 beta 的余弦变化范围
def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦调度函数
    参考 https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# 医学分割扩散模块类，继承自 Module 类
class MedSegDiff(Module):
    def __init__(
        self,
        model,
        *,
        timesteps=1000,
        sampling_timesteps=None,
        objective='pred_noise',
        beta_schedule='cosine',
        ddim_sampling_eta=1.
        ):
        # 调用父类的构造函数
        super().__init__()

        # 如果传入的模型不是 Unet 类型，则取其 module 属性
        self.model = model if isinstance(model, Unet) else model.module

        # 获取模型的输入图像通道数、掩模通道数、自身条件、图像大小等属性
        self.input_img_channels = self.model.input_img_channels
        self.mask_channels = self.model.mask_channels
        self.self_condition = self.model.self_condition
        self.image_size = self.model.image_size

        # 设置目标类型
        self.objective = objective

        # 检查目标类型是否合法
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # 根据 beta_schedule 选择不同的 beta 调度
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # 计算 alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        # 获取时间步数
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # 设置采样相关参数
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # 默认采样时间步数为训练时间步数
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # 注册缓冲区函数
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # 注册缓冲区
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
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    @property
    def device(self):
        # 返回参数的设备信息
        return next(self.parameters()).device

    def predict_start_from_noise(self, x_t, t, noise):
        # 预测起始值从噪声
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        # 从起始值预测噪声
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        # 预测 v
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )
    # 根据给定的输入 x_t, t 和 v 预测起始值
    def predict_start_from_v(self, x_t, t, v):
        return (
            # 使用累积平方根系数乘积提取 t 时刻的值，与输入 x_t 相乘
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            # 使用累积平方根系数乘积提取 t 时刻的值，与输入 v 相乘
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    # 计算后验分布的均值和方差
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            # 提取 t 时刻的系数1，与输入 x_start 相乘
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            # 提取 t 时刻的系数2，与输入 x_t 相乘
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # 提取 t 时刻的后验方差
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        # 提取 t 时刻的修剪后的后验对数方差
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 模型预测函数，根据不同的目标类型进行预测
    def model_predictions(self, x, t, c, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, c, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    # 计算均值和方差，可选择是否对去噪后的值进行裁剪
    def p_mean_variance(self, x, t, c, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, c, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # 生成样本，根��输入 x, t, c 生成预测图像
    @torch.no_grad()
    def p_sample(self, x, t, c, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, c = c, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # 若 t == 0 则无噪声
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    # 循环生成样本，根据给定的形状和条件
    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device = device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, cond, self_cond)

        img = unnormalize_to_zero_to_one(img)
        return img

    # 禁用梯度计算
    @torch.no_grad()
    # 从给定形状和条件图像中生成 DDIM 采样结果
    def ddim_sample(self, shape, cond_img, clip_denoised = True):
        # 获取形状参数和设备信息
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # 生成时间步长序列
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # 生成随机初始图像
        img = torch.randn(shape, device = device)

        x_start = None

        # 遍历时间步长对
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_img, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        # 将图像还原到 [0, 1] 范围内
        img = unnormalize_to_zero_to_one(img)
        return img

    # 生成采样结果
    @torch.no_grad()
    def sample(self, cond_img):
        batch_size, device = cond_img.shape[0], self.device
        cond_img = cond_img.to(self.device)

        image_size, mask_channels = self.image_size, self.mask_channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, mask_channels, image_size, image_size), cond_img)

    # 生成 Q 采样结果
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # 计算 P 损失
    def p_losses(self, x_start, t, cond, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 生成噪声样本

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # 如果进行自条件生成，50% 的时间，从当前时间预测 x_start，并使用 unet 进行条件生成
        # 这种技术会使训练速度减慢 25%，但似乎显著降低 FID

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():

                # 预测 x_0

                x_self_cond = self.model_predictions(x, t, cond).pred_x_start
                x_self_cond.detach_()

        # 预测并进行梯度下降

        model_out = self.model(x, t, cond, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        return F.mse_loss(model_out, target)
    # 定义一个前向传播函数，接受输入图像、条件图像以及其他参数
    def forward(self, img, cond_img, *args, **kwargs):
        # 如果输入图像维度为3，则将其重排为'b h w -> b 1 h w'
        if img.ndim == 3:
            img = rearrange(img, 'b h w -> b 1 h w')

        # 如果条件图像维度为3，则将其重排为'b h w -> b 1 h w'
        if cond_img.ndim == 3:
            cond_img = rearrange(cond_img, 'b h w -> b 1 h w')

        # 获取设备信息并将输入图像和条件图像移动到该设备上
        device = self.device
        img, cond_img = img.to(device), cond_img.to(device)

        # 获取输入图像的形状信息
        b, c, h, w, device, img_size, img_channels, mask_channels = *img.shape, img.device, self.image_size, self.input_img_channels, self.mask_channels

        # 断言输入图像的高度和宽度必须为img_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # 断言条件图像的通道数必须为img_channels
        assert cond_img.shape[1] == img_channels, f'your input medical must have {img_channels} channels'
        # 断言输入图像的通道数必须为mask_channels
        assert img.shape[1] == mask_channels, f'the segmented image must have {mask_channels} channels'

        # 生成一个随机整数张量，范围为[0, num_timesteps)，形状为(b,)
        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # 对输入图像进行归一化到[-1, 1]范围内
        img = normalize_to_neg_one_to_one(img)
        # 调用p_losses函数计算损失并返回结果
        return self.p_losses(img, times, cond_img, *args, **kwargs)
```