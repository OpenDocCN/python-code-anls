# `.\lucidrains\metnet3-pytorch\metnet3_pytorch\metnet3_pytorch.py`

```py
# 导入必要的库
from pathlib import Path
from functools import partial
from collections import namedtuple
from contextlib import contextmanager

import torch
from torch import nn, Tensor, einsum
import torch.distributed as dist
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential

# 导入 einops 库中的函数和层
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

# 导入 beartype 库中的类型注解
from beartype import beartype
from beartype.typing import Tuple, Union, List, Optional, Dict, Literal

import pickle

# 定义一些辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 将单个元素打包成指定模式的元组
def pack_one(x, pattern):
    return pack([x], pattern)

# 从元组中解包单个元素
def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

# 将值转换为元组
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# 安全除法，避免分母为零
def safe_div(num, den, eps = 1e-10):
    return num / den.clamp(min = eps)

# 张量归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# 准备在分布式训练中使用的批量归一化

# 根据是否处于分布式环境选择使用 SyncBatchNorm 还是 BatchNorm2d
def MaybeSyncBatchnorm2d(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm2d

# 冻结批量归一化层
@contextmanager
def freeze_batchnorm(bn):
    assert not exists(next(bn.parameters(), None))

    was_training = bn.training
    was_tracking_stats = bn.track_running_stats
    bn.eval()
    bn.track_running_stats = False

    yield bn

    bn.train(was_training)
    bn.track_running_stats = was_tracking_stats

# 损失缩放

# 自定义损失缩放函数
class LossScaleFunction(Function):
    @staticmethod
    def forward(ctx, x, eps):
        ctx.eps = eps
        assert x.ndim == 4
        return x

    @staticmethod
    def backward(ctx, grads):
        num_channels = grads.shape[1]

        safe_div_ = partial(safe_div, eps = ctx.eps)

        weight = safe_div_(1., grads.norm(p = 2, keepdim = True, dim = (-1, -2)))
        l1_normed_weight = safe_div_(weight, weight.sum(keepdim = True, dim = 1))

        scaled_grads = num_channels * l1_normed_weight * grads

        return scaled_grads, None

# 损失缩放器
class LossScaler(Module):
    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return LossScaleFunction.apply(x, self.eps)

# 中心裁剪

# 中心填充模块
class CenterPad(Module):
    def __init__(self, target_dim):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, x):
        target_dim = self.target_dim
        *_, height, width = x.shape
        assert target_dim >= height and target_dim >= width

        height_pad = target_dim - height
        width_pad = target_dim - width
        left_height_pad = height_pad // 2
        left_width_pad = width_pad // 2

        return F.pad(x, (left_height_pad, height_pad - left_height_pad, left_width_pad, width_pad - left_width_pad), value = 0.)

# 中心裁剪模块
class CenterCrop(Module):
    def __init__(self, crop_dim):
        super().__init__()
        self.crop_dim = crop_dim

    def forward(self, x):
        crop_dim = self.crop_dim
        *_, height, width = x.shape
        assert (height >= crop_dim) and (width >= crop_dim)

        cropped_height_start_idx = (height - crop_dim) // 2
        cropped_width_start_idx = (width - crop_dim) // 2

        height_slice = slice(cropped_height_start_idx, cropped_height_start_idx + crop_dim)
        width_slice = slice(cropped_width_start_idx, cropped_width_start_idx + crop_dim)
        return x[..., height_slice, width_slice]

# 下采样和上采样

# 下采样使用最大池化，上采样使用转置卷积
# todo: 弄清楚从 4km 到 1km 的 4 倍上采样

# 2 倍下采样
Downsample2x = partial(nn.MaxPool2d, kernel_size = 2, stride = 2)

# 2 倍上采样
def Upsample2x(dim, dim_out = None):
    # 如果未提供输出维度，则使用输入维度作为输出维度
    dim_out = default(dim_out, dim)
    # 返回一个转置卷积层，输入维度为dim，输出维度为dim_out，卷积核大小为2，步长为2
    return nn.ConvTranspose2d(dim, dim_out, kernel_size = 2, stride = 2)
# 定义一个条件可选的 ResNet 块
class Block(Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        # 使用卷积层进行投影
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        # 使用通道层归一化
        self.norm = ChanLayerNorm(dim_out)
        # 使用 ReLU 激活函数
        self.act = nn.ReLU()

    def forward(self, x, scale_shift = None):
        # 对输入进行投影
        x = self.proj(x)
        # 对投影结果进行归一化
        x = self.norm(x)

        # 如果存在 scale_shift 参数，则进行缩放和平移
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        # 对结果进行激活
        x = self.act(x)
        return x

# 定义一个 ResNet 块
class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        cond_dim = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.mlp = None

        # 如果存在条件维度，则创建一个 MLP
        if exists(cond_dim):
            self.mlp = Sequential(
                nn.ReLU(),
                nn.Linear(cond_dim, dim_out * 2)
            )

        # 创建两个 Block 实例
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        # 如果输入维度和输出维度不同，则使用卷积层进行投影
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond = None):

        scale_shift = None

        # 断言条件：MLP 和条件参数 cond 必须同时存在或同时不存在
        assert not (exists(self.mlp) ^ exists(cond))

        # 如果存在 MLP 和条件参数 cond，则进行处理
        if exists(self.mlp) and exists(cond):
            cond = self.mlp(cond)
            cond = rearrange(cond, 'b c -> b c 1 1')
            scale_shift = cond.chunk(2, dim = 1)

        # 对输入进行第一个 Block 处理
        h = self.block1(x, scale_shift = scale_shift)

        # 对第一个 Block 处理结果进行第二个 Block 处理
        h = self.block2(h)

        # 返回结果加上残差连接
        return h + self.res_conv(x)

# 定义一个包含多个 ResNet 块的模块
class ResnetBlocks(Module):
    def __init__(
        self,
        dim,
        *,
        dim_in = None,
        depth = 1,
        cond_dim = None
    ):
        super().__init__()
        curr_dim = default(dim_in, dim)

        blocks = []
        # 根据深度循环创建多个 ResNet 块
        for _ in range(depth):
            blocks.append(ResnetBlock(dim = curr_dim, dim_out = dim, cond_dim = cond_dim))
            curr_dim = dim

        self.blocks = ModuleList(blocks)

    def forward(self, x, cond = None):

        for block in self.blocks:
            x = block(x, cond = cond)

        return x

# 多头 RMS 归一化，用于查询/键归一化注意力
class RMSNorm(Module):
    def __init__(
        self,
        dim,
        *,
        heads
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# 在 ResNet 块中使用层归一化的原因
class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * var.clamp(min = self.eps).rsqrt() * self.g + self.b

# MBConv

# 定义一个 Squeeze-and-Excitation 模块
class SqueezeExcitation(Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        # 构建门控网络
        self.gate = Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

# 定义一个 MBConv 残差模块
class MBConvResidual(Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

# 定义一个 Dropout 模块
class Dropsample(Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
    # 定义一个前向传播函数，接受输入张量 x
    def forward(self, x):
        # 获取输入张量 x 的设备信息
        device = x.device

        # 如果概率为 0 或者不处于训练状态，则直接返回输入张量 x
        if self.prob == 0. or (not self.training):
            return x

        # 生成一个与输入张量 x 形状相同的随机掩码，用于随机丢弃部分数据
        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        # 对输入张量 x 进行随机丢弃操作，并进行归一化处理
        return x * keep_mask / (1 - self.prob)
# 定义一个 MBConv 模块，用于 MobileNetV3 的基本块
def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    # 计算隐藏层维度
    hidden_dim = int(expansion_rate * dim_out)
    # 如果 downsample 为真，则步长为 2，否则为 1
    stride = 2 if downsample else 1

    # 创建一个 MaybeSyncBatchnorm2d 类的实例
    batchnorm_klass = MaybeSyncBatchnorm2d()

    # 构建网络结构
    net = Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        batchnorm_klass(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        batchnorm_klass(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        batchnorm_klass(dim_out)
    )

    # 如果输入维度等于输出维度且不下采样，则添加 MBConvResidual 模块
    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

# 定义一个 XCAttention 类，实现特定的线性注意力机制
class XCAttention(Module):
    """
    this specific linear attention was proposed in https://arxiv.org/abs/2106.09681 (El-Nouby et al.)
    """

    @beartype
    def __init__(
        self,
        *,
        dim,
        cond_dim: Optional[int] = None,
        dim_head = 32,
        heads = 8,
        scale = 8,
        flash = False,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.has_cond = exists(cond_dim)

        self.film = None

        # 如果有条件输入，则构建 FILM 网络
        if self.has_cond:
            self.film = Sequential(
                nn.Linear(cond_dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim * 2),
                Rearrange('b (r d) -> r b 1 d', r = 2)
            )

        # LayerNorm 层
        self.norm = nn.LayerNorm(dim, elementwise_affine = not self.has_cond)

        # QKV 线性映射
        self.to_qkv = Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv = 3, h = heads)
        )

        self.scale = scale

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.attn_dropout = nn.Dropout(dropout)

        # 输出映射
        self.to_out = Sequential(
            Rearrange('b h d n -> b n (h d)'),
            nn.Linear(dim_inner, dim)
        )

    # 前向传播函数
    def forward(
        self,
        x,
        cond: Optional[Tensor] = None
    ):
        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack_one(x, 'b * c')

        x = self.norm(x)

        # 条件输入
        if exists(self.film):
            assert exists(cond)

            gamma, beta = self.film(cond)
            x = x * gamma + beta

        # 余弦相似度线性注意力机制
        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        sim = einsum('b h i n, b h j n -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j n -> b h i n', attn, v)

        out = self.to_out(out)

        out = unpack_one(out, ps, 'b * c')
        return rearrange(out, 'b h w c -> b c h w')

# 定义一个 Attention 类，实现注意力机制
class Attention(Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        heads = 32,
        dim_head = 32,
        dropout = 0.,
        window_size = 8,
        num_registers = 1
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言寄存器数量大于0
        assert num_registers > 0
        # 断言维度应该可以被每个头的维度整除
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        # 计算内部维度
        dim_inner = dim_head * heads
        self.heads = heads
        # 缩放因子
        self.scale = dim_head ** -0.5

        # 检查是否有条件
        self.has_cond = exists(cond_dim)

        self.film = None

        # 如果有条件
        if self.has_cond:
            # 创建 FILM 模块
            self.film = Sequential(
                nn.Linear(cond_dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim * 2),
                Rearrange('b (r d) -> r b 1 d', r = 2)
            )

        # 归一化层
        self.norm = nn.LayerNorm(dim, elementwise_affine = not self.has_cond)

        # 线性变换到查询、键、值
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        # 查询和键的 RMS 归一化
        self.q_norm = RMSNorm(dim_head, heads = heads)
        self.k_norm = RMSNorm(dim_head, heads = heads)

        # 注意力机制
        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        # 输出层
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

        # 相对位置偏差

        num_rel_pos_bias = (2 * window_size - 1) ** 2

        # 创建相对位置偏差的 Embedding
        self.rel_pos_bias = nn.Embedding(num_rel_pos_bias + 1, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        rel_pos_indices = F.pad(rel_pos_indices, (num_registers, 0, num_registers, 0), value = num_rel_pos_bias)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(
        self,
        x: Tensor,
        cond: Optional[Tensor] = None
    ):
        # 获取设备、头数、偏差索引
        device, h, bias_indices = x.device, self.heads, self.rel_pos_indices

        # 归一化输入
        x = self.norm(x)

        # 条件
        if exists(self.film):
            assert exists(cond)

            gamma, beta = self.film(cond)
            x = x * gamma + beta

        # 为查询、键、值进行投影
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # 分割头
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 缩放
        q, k = self.q_norm(q), self.k_norm(k)

        # 相似度
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 添加位置偏差
        bias = self.rel_pos_bias(bias_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # 注意力
        attn = self.attend(sim)

        # 聚合
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 合并头部输出
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# 定义一个名为 MaxViT 的类，继承自 Module 类
class MaxViT(Module):
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        *,
        dim,  # 特征维度
        depth,  # 模型深度
        cond_dim = 32,   # 用于条件化的前导时间嵌入
        heads = 32,  # 多头注意力机制中的头数
        dim_head = 32,  # 每个头的维度
        window_size = 8,  # 窗口大小
        mbconv_expansion_rate = 4,  # MBConv 层的扩张率
        mbconv_shrinkage_rate = 0.25,  # MBConv 层的收缩率
        dropout = 0.1,  # 丢弃率
        num_register_tokens = 4  # 寄存器令牌数量
    ):
        super().__init__()
        # 如果 depth 是整数，则转换为元组
        depth = (depth,) if isinstance(depth, int) else depth
        # 断言寄存器令牌数量大于0
        assert num_register_tokens > 0

        self.cond_dim = cond_dim

        # 变量

        num_stages = len(depth)

        # 计算每个阶段的维度
        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # 窗口大小

        self.window_size = window_size

        self.register_tokens = nn.ParameterList([])

        # 遍历各个阶段

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                # 创建 MBConv 层
                conv = MBConv(
                    stage_dim_in,
                    layer_dim,
                    downsample = is_first,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                )

                # 创建块级别的注意力机制
                block_attn = Attention(dim = layer_dim, cond_dim = cond_dim, heads = heads, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)

                # 创建网格级别的注意力机制
                grid_attn = Attention(dim = layer_dim, cond_dim = cond_dim, heads = heads, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)

                # 创建寄存器令牌
                register_tokens = nn.Parameter(torch.randn(num_register_tokens, layer_dim))

                # 将 MBConv 层、块级别注意力机制、网格级别注意力机制组合成一个模块列表
                self.layers.append(ModuleList([
                    conv,
                    block_attn,
                    grid_attn
                ]))

                # 将寄存器令牌添加到参数列表中
                self.register_tokens.append(register_tokens)

    # 前向传播函数，接受输入张量 x 和条件张量 cond
    def forward(
        self,
        x: Tensor,
        cond: Tensor
    ):
        # 断言条件的形状与输入 x 的形状一致
        assert cond.shape == (x.shape[0], self.cond_dim)

        # 获取输入 x 的批量大小和窗口大小
        b, w = x.shape[0], self.window_size

        # 遍历每个层和对应的注册令牌
        for (conv, block_attn, grid_attn), register_tokens in zip(self.layers, self.register_tokens):
            # 对输入 x 进行卷积操作
            x = conv(x)

            # block-like attention

            # 重新排列输入 x 的维度
            x = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w)

            # 准备注册令牌
            r = repeat(register_tokens, 'n d -> b x y n d', b = b, x = x.shape[1],y = x.shape[2])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b x y * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            # 对输入 x 进行块状注意力操作，并与原始输入相加
            x = block_attn(x, cond = cond) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b x y * d')
            x = rearrange(x, 'b x y w1 w2 d -> b d (x w1) (y w2)')

            r = unpack_one(r, register_batch_ps, '* n d')

            # grid-like attention

            # 重新排列输入 x 的维度
            x = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w)

            # 准备注册令牌
            r = reduce(r, 'b x y n d -> b n d', 'mean')
            r = repeat(r, 'b n d -> b x y n d', x = x.shape[1], y = x.shape[2])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b x y * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            # 对输入 x 进行网格状注意力操作，并与原始输入相加
            x = grid_attn(x, cond = cond) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b x y * d')
            x = rearrange(x, 'b x y w1 w2 d -> b d (w1 x) (w2 y)')

        # 返回处理后的输入 x
        return x
# 定义一个命名元组 Predictions，包含 surface、hrrr、precipitation 三个字段
Predictions = namedtuple('Predictions', [
    'surface',
    'hrrr',
    'precipitation'
])

# 定义一个命名元组 LossBreakdown，包含 surface、hrrr、precipitation 三个字段
LossBreakdown = namedtuple('LossBreakdown', [
    'surface',
    'hrrr',
    'precipitation'
])

# 定义一个类 MetNet3，继承自 Module
class MetNet3(Module):
    # 初始化方法
    @beartype
    def __init__(
        self,
        *,
        dim = 512,
        num_lead_times = 722,
        lead_time_embed_dim = 32,
        input_spatial_size = 624,
        attn_depth = 12,
        attn_dim_head = 64,
        attn_heads = 32,
        attn_dropout = 0.1,
        vit_window_size = 8,
        vit_mbconv_expansion_rate = 4,
        vit_mbconv_shrinkage_rate = 0.25,
        input_2496_channels = 2 + 14 + 1 + 2 + 20,
        input_4996_channels = 16 + 1,
        surface_and_hrrr_target_spatial_size = 128,
        precipitation_target_bins: Dict[str, int] = dict(
            mrms_rate = 512,
            mrms_accumulation = 512
        ),
        surface_target_bins: Dict[str, int] = dict(
            omo_temperature = 256,
            omo_dew_point = 256,
            omo_wind_speed = 256,
            omo_wind_component_x = 256,
            omo_wind_component_y = 256,
            omo_wind_direction = 180
        ),
        hrrr_norm_strategy: Union[
            Literal['none'],
            Literal['precalculated'],
            Literal['sync_batchnorm']
        ] = 'none',
        hrrr_channels = 617,
        hrrr_norm_statistics: Optional[Tensor] = None,
        hrrr_loss_weight = 10,
        crop_size_post_16km = 48,
        resnet_block_depth = 2,
    
    # 类方法，从路径加载模型
    @classmethod
    def init_and_load_from(cls, path, strict = True):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 断言路径存在
        assert path.exists()
        # 加载模型
        pkg = torch.load(str(path), map_location = 'cpu')

        # 断言模型配置信息在加载的包中
        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        # 从包中加载配置信息
        config = pickle.loads(pkg['config'])
        # 创建模型实例
        tokenizer = cls(**config)
        # 加载模型
        tokenizer.load(path, strict = strict)
        return tokenizer

    # 保存模型
    def save(self, path, overwrite = True):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 断言路径不存在或允许覆盖
        assert overwrite or not path.exists(), f'{str(path)} already exists'

        # 构建保存的包
        pkg = dict(
            model_state_dict = self.state_dict(),
            config = self._configs
        )

        # 保存模型
        torch.save(pkg, str(path))

    # 加载模型
    def load(self, path, strict = True):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 断言路径存在
        assert path.exists()

        # 加载模型
        pkg = torch.load(str(path))
        state_dict = pkg.get('model_state_dict')

        # 断言状态字典存在
        assert exists(state_dict)

        # 加载模型状态字典
        self.load_state_dict(state_dict, strict = strict)

    # 前向传播方法
    @beartype
    def forward(
        self,
        *,
        lead_times,
        hrrr_input_2496,
        hrrr_stale_state,
        input_2496,
        input_4996,
        surface_targets: Optional[Dict[str, Tensor]] = None,
        precipitation_targets: Optional[Dict[str, Tensor]] = None,
        hrrr_target: Optional[Tensor] = None,
```