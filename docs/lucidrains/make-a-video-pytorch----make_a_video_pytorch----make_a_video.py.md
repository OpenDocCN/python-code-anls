# `.\lucidrains\make-a-video-pytorch\make_a_video_pytorch\make_a_video.py`

```py
# 导入数学库
import math
# 导入 functools 库
import functools
# 从 operator 库中导入 mul 函数
from operator import mul

# 导入 torch 库
import torch
# 从 torch.nn 中导入 functional 模块
import torch.nn.functional as F
# 从 torch 中导入 nn、einsum 模块
from torch import nn, einsum

# 从 einops 中导入 rearrange、repeat、pack、unpack 函数，以及 Rearrange 类
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# 从 make_a_video_pytorch.attend 模块中导入 Attend 类

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在，则返回变量值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 对元组中的元素进行乘法运算
def mul_reduce(tup):
    return functools.reduce(mul, tup)

# 判断一个数是否可以被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 创建 nn.ModuleList 对象
mlist = nn.ModuleList

# 用于时间条件

# 正弦位置编码
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.theta = theta
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        assert dtype == torch.float, 'input to sinusoidal pos emb must be a float type'

        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device, dtype = dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1).type(dtype)

# 3D 归一化

# RMS 归一化
class RMSNorm(nn.Module):
    def __init__(self, chan, dim = 1):
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(chan))

    def forward(self, x):
        dim = self.dim
        right_ones = (dim + 1) if dim < 0 else (x.ndim - 1 - dim)
        gamma = self.gamma.reshape(-1, *((1,) * right_ones))
        return F.normalize(x, dim = dim) * (x.shape[dim] ** 0.5) * gamma

# 前馈网络

# 移位令牌
def shift_token(t):
    t, t_shift = t.chunk(2, dim = 1)
    t_shift = F.pad(t_shift, (0, 0, 0, 0, 1, -1), value = 0.)
    return torch.cat((t, t_shift), dim = 1)

# GEGLU 激活函数
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = 1)
        return x * F.gelu(gate)

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()

        inner_dim = int(dim * mult * 2 / 3)
        self.proj_in = nn.Sequential(
            nn.Conv3d(dim, inner_dim * 2, 1, bias = False),
            GEGLU()
        )

        self.proj_out = nn.Sequential(
            RMSNorm(inner_dim),
            nn.Conv3d(inner_dim, dim, 1, bias = False)
        )

    def forward(self, x, enable_time = True):

        is_video = x.ndim == 5
        enable_time &= is_video

        if not is_video:
            x = rearrange(x, 'b c h w -> b c 1 h w')

        x = self.proj_in(x)

        if enable_time:
            x = shift_token(x)

        out = self.proj_out(x)

        if not is_video:
            out = rearrange(out, 'b c 1 h w -> b c h w')

        return out

# 最佳相对位置编码

# 连续位置偏置
class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims = 1,
        layers = 2
    ):
        super().__init__()
        self.num_dims = num_dims

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads)

    @property
    def device(self):
        return next(self.parameters()).device
    # 定义一个前向传播函数，接受多个维度参数
    def forward(self, *dimensions):
        # 获取当前设备
        device = self.device

        # 将维度转换为张量
        shape = torch.tensor(dimensions, device=device)
        # 计算相对位置的形状
        rel_pos_shape = 2 * shape - 1

        # 计算步长

        # 将相对位置形状进行翻转，并计算累积乘积
        strides = torch.flip(rel_pos_shape, (0,)).cumprod(dim=-1)
        # 在步长张量两端填充1，并再次翻转
        strides = torch.flip(F.pad(strides, (1, -1), value=1), (0,))

        # 获取所有位置并计算所有相对距离

        # 生成每个维度的位置张量
        positions = [torch.arange(d, device=device) for d in dimensions]
        # 创建网格坐标
        grid = torch.stack(torch.meshgrid(*positions, indexing='ij'), dim=-1)
        # 重新排列网格坐标
        grid = rearrange(grid, '... c -> (...) c')
        # 计算相对距离
        rel_dist = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

        # 获取所有维度上的相对位置

        # 生成每个维度上的相对位置张量
        rel_positions = [torch.arange(-d + 1, d, device=device) for d in dimensions]
        # 创建相对位置网格
        rel_pos_grid = torch.stack(torch.meshgrid(*rel_positions, indexing='ij'), dim=-1)
        # 重新排列相对位置网格
        rel_pos_grid = rearrange(rel_pos_grid, '... c -> (...) c')

        # MLP 输入

        # 将相对位置网格转换为浮点数
        bias = rel_pos_grid.float()

        # 遍历网络的每一层
        for layer in self.net:
            # 将相对位置网格传入每一层
            bias = layer(bias)

        # 将相对距离转换为偏置的索引

        # 将相对距离加上形状减一确保为正数
        rel_dist += (shape - 1)
        # 乘以步长
        rel_dist *= strides
        # 沿着最后一个维度求和，得到索引
        rel_dist_indices = rel_dist.sum(dim=-1)

        # 选择每个唯一相对位置组合的偏置

        # 根据索引选择偏置
        bias = bias[rel_dist_indices]
        # 重新排列偏置
        return rearrange(bias, 'i j h -> h i j')
# 定义注意力机制类
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        flash = False,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        # 创建 Attend 对象
        self.attend = Attend(flash = flash, causal = causal)

        # 创建 RMSNorm 对象
        self.norm = RMSNorm(dim, dim = -1)

        # 创建线性变换层
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # 初始化权重为零，实现跳跃连接
        nn.init.zeros_(self.to_out.weight.data)

    def forward(
        self,
        x,
        rel_pos_bias = None
    ):
        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        out = self.attend(q, k, v, bias = rel_pos_bias)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义主要贡献 - 伪 3D 卷积类
class PseudoConv3d(nn.Module):
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

        # 创建空间卷积层和时间卷积层
        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size = kernel_size, padding = kernel_size // 2)
        self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size = temporal_kernel_size, padding = temporal_kernel_size // 2) if kernel_size > 1 else None

        # 初始化时间卷积层的权重为单位矩阵，偏置为零
        if exists(self.temporal_conv):
            nn.init.dirac_(self.temporal_conv.weight.data)
            nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(
        self,
        x,
        enable_time = True
    ):
        b, c, *_, h, w = x.shape

        is_video = x.ndim == 5
        enable_time &= is_video

        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) c h w')

        x = self.spatial_conv(x)

        if is_video:
            x = rearrange(x, '(b f) c h w -> b c f h w', b = b)

        if not enable_time or not exists(self.temporal_conv):
            return x

        x = rearrange(x, 'b c f h w -> (b h w) c f')

        x = self.temporal_conv(x)

        x = rearrange(x, '(b h w) c f -> b c f h w', h = h, w = w)

        return x

# 定义分解的时空注意力类
class SpatioTemporalAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        add_feed_forward = True,
        ff_mult = 4,
        pos_bias = True,
        flash = False,
        causal_time_attn = False
    ):
        super().__init__()
        assert not (flash and pos_bias), 'learned positional attention bias is not compatible with flash attention'

        # 创建空间注意力和时间注意力对象
        self.spatial_attn = Attention(dim = dim, dim_head = dim_head, heads = heads, flash = flash)
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim // 2, heads = heads, num_dims = 2) if pos_bias else None

        self.temporal_attn = Attention(dim = dim, dim_head = dim_head, heads = heads, flash = flash, causal = causal_time_attn)
        self.temporal_rel_pos_bias = ContinuousPositionBias(dim = dim // 2, heads = heads, num_dims = 1) if pos_bias else None

        self.has_feed_forward = add_feed_forward
        if not add_feed_forward:
            return

        # 创建前馈网络对象
        self.ff = FeedForward(dim = dim, mult = ff_mult)

    def forward(
        self,
        x,
        enable_time = True
        ):
        # 从输入张量 x 的形状中提取出 b, c, h, w，*_, h, w 表示忽略中间的维度，只取最后两个维度
        b, c, *_, h, w = x.shape
        # 判断输入张量是否为视频，即维度是否为 5
        is_video = x.ndim == 5
        # 更新 enable_time 变量，如果是视频则为 True
        enable_time &= is_video

        # 根据输入张量的维度不同进行不同的重排操作
        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) (h w) c')
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')

        # 如果存在空间相对位置偏置函数，则计算空间相对位置偏置
        space_rel_pos_bias = self.spatial_rel_pos_bias(h, w) if exists(self.spatial_rel_pos_bias) else None

        # 对输入张量进行空间注意力操作，并加上原始输入张量
        x = self.spatial_attn(x, rel_pos_bias = space_rel_pos_bias) + x

        # 根据输入张量的维度不同进行不同的重排操作，恢复原始形状
        if is_video:
            x = rearrange(x, '(b f) (h w) c -> b c f h w', b = b, h = h, w = w)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)

        # 如果 enable_time 为 True，则进行时间维度的处理
        if enable_time:

            # 对输入张量进行时间维度的重排操作
            x = rearrange(x, 'b c f h w -> (b h w) f c')

            # 如果存在时间相对位置偏置函数，则计算时间相对位置偏置
            time_rel_pos_bias = self.temporal_rel_pos_bias(x.shape[1]) if exists(self.temporal_rel_pos_bias) else None

            # 对输入张量进行时间注意力操作，并加上原始输入张量
            x = self.temporal_attn(x, rel_pos_bias = time_rel_pos_bias) + x

            # 恢复原始形状
            x = rearrange(x, '(b h w) f c -> b c f h w', w = w, h = h)

        # 如果存在前馈网络，则对输入张量进行前馈操作，并加上原始输入张量
        if self.has_feed_forward:
            x = self.ff(x, enable_time = enable_time) + x

        # 返回处理后的张量
        return x
# 定义 ResNet 块
class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size = 3,
        temporal_kernel_size = None,
        groups = 8
    ):
        super().__init__()
        # 创建伪 3D 卷积层
        self.project = PseudoConv3d(dim, dim_out, 3)
        # 添加 Group Normalization
        self.norm = nn.GroupNorm(groups, dim_out)
        # 添加 SiLU 激活函数
        self.act = nn.SiLU()

    def forward(
        self,
        x,
        scale_shift = None,
        enable_time = False
    ):
        # 对输入进行卷积操作
        x = self.project(x, enable_time = enable_time)
        # 对卷积结果进行归一化
        x = self.norm(x)

        # 如果存在 scale_shift 参数，则进行缩放和平移操作
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

# 定义 ResNet 块
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        timestep_cond_dim = None,
        groups = 8
    ):
        super().__init__()

        self.timestep_mlp = None

        # 如果存在时间步条件维度，则创建 MLP 网络
        if exists(timestep_cond_dim):
            self.timestep_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timestep_cond_dim, dim_out * 2)
            )

        # 创建两个 Block 实例
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        # 如果输入维度和输出维度不同，创建伪 3D 卷积层，否则创建恒等映射
        self.res_conv = PseudoConv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x,
        timestep_emb = None,
        enable_time = True
    ):
        # 断言时间步条件嵌入和时间步 MLP 是否同时存在
        assert not (exists(timestep_emb) ^ exists(self.timestep_mlp))

        scale_shift = None

        # 如果存在时间步 MLP 和时间步嵌入，则进行处理
        if exists(self.timestep_mlp) and exists(timestep_emb):
            time_emb = self.timestep_mlp(timestep_emb)
            to_einsum_eq = 'b c 1 1 1' if x.ndim == 5 else 'b c 1 1'
            time_emb = rearrange(time_emb, f'b c -> {to_einsum_eq}')
            scale_shift = time_emb.chunk(2, dim = 1)

        # 对输入进行第一个 Block 处理
        h = self.block1(x, scale_shift = scale_shift, enable_time = enable_time)

        # 对第一�� Block 处理结果进行第二个 Block 处理
        h = self.block2(h, enable_time = enable_time)

        return h + self.res_conv(x)

# 像素混洗上采样和下采样，其中时间维度可以配置

# 定义下采样模块
class Downsample(nn.Module):
    def __init__(
        self,
        dim,
        downsample_space = True,
        downsample_time = False,
        nonlin = False
    ):
        super().__init__()
        assert downsample_space or downsample_time

        # 如果需要空间下采样，则创建相应的模块
        self.down_space = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
            nn.Conv2d(dim * 4, dim, 1, bias = False),
            nn.SiLU() if nonlin else nn.Identity()
        ) if downsample_space else None

        # 如果需要时间下采样，则创建相应的模块
        self.down_time = nn.Sequential(
            Rearrange('b c (f p) h w -> b (c p) f h w', p = 2),
            nn.Conv3d(dim * 2, dim, 1, bias = False),
            nn.SiLU() if nonlin else nn.Identity()
        ) if downsample_time else None

    def forward(
        self,
        x,
        enable_time = True
    ):
        is_video = x.ndim == 5

        if is_video:
            x = rearrange(x, 'b c f h w -> b f c h w')
            x, ps = pack([x], '* c h w')

        # 如果存在空间下采样模块，则进行处理
        if exists(self.down_space):
            x = self.down_space(x)

        if is_video:
            x, = unpack(x, ps, '* c h w')
            x = rearrange(x, 'b f c h w -> b c f h w')

        # 如果不是视频或者不存在时间下采样模块或者不启用时间，则直接返回结果
        if not is_video or not exists(self.down_time) or not enable_time:
            return x

        # 如果需要时间下采样，则进行处理
        x = self.down_time(x)

        return x

# 定义上采样模块
class Upsample(nn.Module):
    def __init__(
        self,
        dim,
        upsample_space = True,
        upsample_time = False,
        nonlin = False
    # 定义一个类，继承自 nn.Module
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 断言是否需要上采样空间或时间
        assert upsample_space or upsample_time

        # 如果需要上采样空间，则定义空间上采样的操作
        self.up_space = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),  # 使用 1x1 卷积进行通道扩展
            nn.SiLU() if nonlin else nn.Identity(),  # 使用 SiLU 激活函数或者恒等映射
            Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)', p1 = 2, p2 = 2)  # 重新排列张量维度
        ) if upsample_space else None

        # 如果需要上采样时间，则定义时间上采样的操作
        self.up_time = nn.Sequential(
            nn.Conv3d(dim, dim * 2, 1),  # 使用 1x1x1 卷积进行通道扩展
            nn.SiLU() if nonlin else nn.Identity(),  # 使用 SiLU 激活函数或者恒等映射
            Rearrange('b (c p) f h w -> b c (f p) h w', p = 2)  # 重新排列张量维度
        ) if upsample_time else None

        # 初始化函数
        self.init_()

    # 初始化函数
    def init_(self):
        # 如果存在空间上采样操作，则初始化空间上采样的卷积层
        if exists(self.up_space):
            self.init_conv_(self.up_space[0], 4)

        # 如果存在时间上采样操作，则初始化时间上采样的卷积层
        if exists(self.up_time):
            self.init_conv_(self.up_time[0], 2)

    # 初始化卷积层的权重
    def init_conv_(self, conv, factor):
        o, *remain_dims = conv.weight.shape
        conv_weight = torch.empty(o // factor, *remain_dims)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = factor)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    # 前向传播函数
    def forward(
        self,
        x,
        enable_time = True
    ):
        # 判断输入是否为视频
        is_video = x.ndim == 5

        # 如果是视频，则重新排列张量维度
        if is_video:
            x = rearrange(x, 'b c f h w -> b f c h w')
            x, ps = pack([x], '* c h w')

        # 如果存在空间上采样操作，则进行空间上采样
        if exists(self.up_space):
            x = self.up_space(x)

        # 如果是视频，则恢复原始张量维度
        if is_video:
            x, = unpack(x, ps, '* c h w')
            x = rearrange(x, 'b f c h w -> b c f h w')

        # 如果不是视频或者不存在时间上采样���作或者不启用时间上采样，则直接返回结果
        if not is_video or not exists(self.up_time) or not enable_time:
            return x

        # 进行时间上采样
        x = self.up_time(x)

        return x
# space time factorized 3d unet

class SpaceTimeUnet(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 维度
        channels = 3,  # 通道数，默认为3
        dim_mult = (1, 2, 4, 8),  # 维度倍增因子
        self_attns = (False, False, False, True),  # 是否使用自注意力机制
        temporal_compression = (False, True, True, True),  # 是否进行时间压缩
        resnet_block_depths = (2, 2, 2, 2),  # ResNet块的深度
        attn_dim_head = 64,  # 注意力机制的头数
        attn_heads = 8,  # 注意力头数
        condition_on_timestep = True,  # 是否在时间步上进行条件化
        attn_pos_bias = True,  # 是否使用位置偏置
        flash_attn = False,  # 是否使用快闪注意力
        causal_time_attn = False  # 是否使用因果时间注意力
    ):
        super().__init__()
        assert len(dim_mult) == len(self_attns) == len(temporal_compression) == len(resnet_block_depths)
        num_layers = len(dim_mult)

        dims = [dim, *map(lambda mult: mult * dim, dim_mult)]  # 计算每一层的维度
        dim_in_out = zip(dims[:-1], dims[1:])

        # determine the valid multiples of the image size and frames of the video

        self.frame_multiple = 2 ** sum(tuple(map(int, temporal_compression)))  # 计算视频帧数的倍数
        self.image_size_multiple = 2 ** num_layers  # 计算图像大小的倍数

        # timestep conditioning for DDPM, not to be confused with the time dimension of the video

        self.to_timestep_cond = None
        timestep_cond_dim = (dim * 4) if condition_on_timestep else None

        if condition_on_timestep:
            self.to_timestep_cond = nn.Sequential(
                SinusoidalPosEmb(dim),  # 添加正弦位置编码
                nn.Linear(dim, timestep_cond_dim),  # 线性变换
                nn.SiLU()  # 激活函数
            )

        # layers

        self.downs = mlist([])  # 下采样层
        self.ups = mlist([])  # 上采样层

        attn_kwargs = dict(
            dim_head = attn_dim_head,
            heads = attn_heads,
            pos_bias = attn_pos_bias,
            flash = flash_attn,
            causal_time_attn = causal_time_attn
        )

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, timestep_cond_dim = timestep_cond_dim)  # 中间块1
        self.mid_attn = SpatioTemporalAttention(dim = mid_dim, **attn_kwargs)  # 中间注意力机制
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, timestep_cond_dim = timestep_cond_dim)  # 中间块2

        for _, self_attend, (dim_in, dim_out), compress_time, resnet_block_depth in zip(range(num_layers), self_attns, dim_in_out, temporal_compression, resnet_block_depths):
            assert resnet_block_depth >= 1

            self.downs.append(mlist([
                ResnetBlock(dim_in, dim_out, timestep_cond_dim = timestep_cond_dim),  # 下采样块
                mlist([ResnetBlock(dim_out, dim_out) for _ in range(resnet_block_depth)]),  # ResNet块
                SpatioTemporalAttention(dim = dim_out, **attn_kwargs) if self_attend else None,  # 注意力机制
                Downsample(dim_out, downsample_time = compress_time)  # 下采样
            ]))

            self.ups.append(mlist([
                ResnetBlock(dim_out * 2, dim_in, timestep_cond_dim = timestep_cond_dim),  # 上采样块
                mlist([ResnetBlock(dim_in + (dim_out if ind == 0 else 0), dim_in) for ind in range(resnet_block_depth)]),  # ResNet块
                SpatioTemporalAttention(dim = dim_in, **attn_kwargs) if self_attend else None,  # 注意力机制
                Upsample(dim_out, upsample_time = compress_time)  # 上采样
            ]))

        self.skip_scale = 2 ** -0.5  # 论文显示更快的收敛速度

        self.conv_in = PseudoConv3d(dim = channels, dim_out = dim, kernel_size = 7, temporal_kernel_size = 3)  # 输入卷积层
        self.conv_out = PseudoConv3d(dim = dim, dim_out = channels, kernel_size = 3, temporal_kernel_size = 3)  # 输出卷积层

    def forward(
        self,
        x,
        timestep = None,
        enable_time = True
        ):

        # some asserts

        # 断言条件：self.to_timestep_cond 和 timestep 存在性相同
        assert not (exists(self.to_timestep_cond) ^ exists(timestep))
        # 判断 x 是否为视频，维度是否为5
        is_video = x.ndim == 5

        # 如果启用时间和 x 是视频
        if enable_time and is_video:
            # 获取视频帧数
            frames = x.shape[2]
            # 断言条件：视频帧数必须能被 self.frame_multiple 整除
            assert divisible_by(frames, self.frame_multiple), f'number of frames on the video ({frames}) must be divisible by the frame multiple ({self.frame_multiple})'

        # 获取图片或视频的高度和宽度
        height, width = x.shape[-2:]
        # 断言条件：图片或视频的高度和宽度必须是 self.image_size_multiple 的倍数
        assert divisible_by(height, self.image_size_multiple) and divisible_by(width, self.image_size_multiple), f'height and width of the image or video must be a multiple of {self.image_size_multiple}'

        # main logic

        # 如果 timestep 存在，则根据条件转换为 t
        t = self.to_timestep_cond(rearrange(timestep, '... -> (...)')) if exists(timestep) else None

        # 对输入 x 进行卷积操作
        x = self.conv_in(x, enable_time = enable_time)

        # 初始化 hiddens 列表
        hiddens = []

        # 遍历 downs 列表中的元素
        for init_block, blocks, maybe_attention, downsample in self.downs:
            # 对 x 进行初始化块操作
            x = init_block(x, t, enable_time = enable_time)

            # 将当前 x 添加到 hiddens 列表中
            hiddens.append(x.clone())

            # 遍历 blocks 列表中的元素
            for block in blocks:
                # 对 x 进行块操作
                x = block(x, enable_time = enable_time)

            # 如果 maybe_attention 存在，则对 x 进行注意力操作
            if exists(maybe_attention):
                x = maybe_attention(x, enable_time = enable_time)

            # 将当前 x 添加到 hiddens 列表中
            hiddens.append(x.clone())

            # 对 x 进行下采样操作
            x = downsample(x, enable_time = enable_time)

        # 对 x 进行中间块1操作
        x = self.mid_block1(x, t, enable_time = enable_time)
        # 对 x 进行中间注意力操作
        x = self.mid_attn(x, enable_time = enable_time)
        # 对 x 进行中间块2操作
        x = self.mid_block2(x, t, enable_time = enable_time)

        # 遍历反转后的 ups 列表中的��素
        for init_block, blocks, maybe_attention, upsample in reversed(self.ups):
            # 对 x 进行上采样操作
            x = upsample(x, enable_time = enable_time)

            # 将 hiddens 列表中的元素与 x 进行拼接
            x = torch.cat((hiddens.pop() * self.skip_scale, x), dim = 1)

            # 对 x 进行初始化块操作
            x = init_block(x, t, enable_time = enable_time)

            # 将 hiddens 列表中的元素与 x 进行拼接
            x = torch.cat((hiddens.pop() * self.skip_scale, x), dim = 1)

            # 遍历 blocks 列表中的元素
            for block in blocks:
                # 对 x 进行块操作
                x = block(x, enable_time = enable_time)

            # 如果 maybe_attention 存在，则对 x 进行注意力操作
            if exists(maybe_attention):
                x = maybe_attention(x, enable_time = enable_time)

        # 对 x 进行输出卷积操作
        x = self.conv_out(x, enable_time = enable_time)
        # 返回结果 x
        return x
```