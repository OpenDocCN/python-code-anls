# `.\lucidrains\x-unet\x_unet\x_unet.py`

```
# 导入必要的库
from functools import partial
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
# 导入 einops 库中的函数和类
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
# 导入 beartype 库中的函数和类型
from beartype import beartype
from beartype.typing import Tuple, Union, Optional

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 返回值或默认值
def default(val, d):
    return val if exists(val) else d

# 检查一个数是否为2的幂
def is_power_two(n):
    return math.log2(n).is_integer()

# 检查一个数是否可以被另一个数整除
def divisible_by(num, denom):
    return (num % denom) == 0

# 将值转换为元组
def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

# 辅助类

# 上采样函数
def Upsample(dim, dim_out):
    return nn.ConvTranspose3d(dim, dim_out, (1, 4, 4), (1, 2, 2), (0, 1, 1))

# 下采样函数
def Downsample(dim, dim_out):
    return nn.Sequential(
        Rearrange('b c f (h s1) (w s2) -> b (c s1 s2) f h w', s1 = 2, s2 = 2),
        nn.Conv3d(dim * 4, dim_out, 1)
    )

# 标准化

# 残差连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + eps).sqrt() * self.gamma

# 权重标准化卷积
class WeightStandardizedConv3d(nn.Conv3d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight

        mean = reduce(weight, 'o ... -> o 1 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1 1', partial(torch.var, unbiased = False))
        weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# ResNet 块

# 块类
class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        weight_standardize = False,
        frame_kernel_size = 1
    ):
        super().__init__()
        kernel_conv_kwargs = partial(kernel_and_same_pad, frame_kernel_size)
        conv = nn.Conv3d if not weight_standardize else WeightStandardizedConv3d

        self.proj = conv(dim, dim_out, **kernel_conv_kwargs(3, 3))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)

# ResNet 块类
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        frame_kernel_size = 1,
        nested_unet_depth = 0,
        nested_unet_dim = 32,
        weight_standardize = False
    ):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups = groups, weight_standardize = weight_standardize, frame_kernel_size = frame_kernel_size)

        if nested_unet_depth > 0:
            self.block2 = NestedResidualUnet(dim_out, depth = nested_unet_depth, M = nested_unet_dim, frame_kernel_size = frame_kernel_size, weight_standardize = weight_standardize, add_residual = True)
        else:
            self.block2 = Block(dim_out, dim_out, groups = groups, weight_standardize = weight_standardize, frame_kernel_size = frame_kernel_size)

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

# ConvNeXT 2

# 全局响应归一化
class GRN(nn.Module):
    """ global response normalization, proposed in updated convnext paper """
    # 初始化函数，设置参数维度和容差值
    def __init__(self, dim, eps = 1e-5):
        # 调用父类的初始化函数
        super().__init__()
        # 设置容差值
        self.eps = eps
        # 初始化 gamma 参数为全零张量
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1, 1))
        # 初始化 bias 参数为全零张量
        self.bias = nn.Parameter(torch.zeros(dim, 1, 1, 1))

    # 前向传播函数
    def forward(self, x):
        # 计算 x 在指定维度上的 L2 范数
        spatial_l2_norm = x.norm(p = 2, dim = (2, 3, 4), keepdim = True)
        # 计算特征的归一化值
        feat_norm = spatial_l2_norm / spatial_l2_norm.mean(dim = -1, keepdim = True).clamp(min = self.eps)
        # 返回经过归一化和缩放后的特征值
        return x * feat_norm * self.gamma + self.bias + x
# 定义一个卷积块类，用于构建下一个卷积块
class ConvNextBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        mult = 2,
        frame_kernel_size = 1,
        nested_unet_depth = 0,
        nested_unet_dim = 32
    ):
        super().__init__()
        kernel_conv_kwargs = partial(kernel_and_same_pad, frame_kernel_size)

        # 深度卷积
        self.ds_conv = nn.Conv3d(dim, dim, **kernel_conv_kwargs(7, 7), groups = dim)

        inner_dim = dim_out * mult

        # 构建一个包含多个层的神经网络
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv3d(dim, inner_dim, **kernel_conv_kwargs(3, 3), groups = dim_out),
            nn.GELU(),
            GRN(inner_dim),
            nn.Conv3d(inner_dim, dim_out, **kernel_conv_kwargs(3, 3), groups = dim_out)
        )

        # 嵌套的残差 UNet
        self.nested_unet = NestedResidualUnet(dim_out, depth = nested_unet_depth, M = nested_unet_dim, add_residual = True) if nested_unet_depth > 0 else nn.Identity()

        # 残差卷积
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)
        h = self.net(h)
        h = self.nested_unet(h)
        return h + self.res_conv(x)

# 前馈神经网络
def FeedForward(dim, mult = 4.):
    inner_dim = int(dim * mult)
    return Residual(nn.Sequential(
        LayerNorm(dim),
        nn.Conv3d(dim, inner_dim, 1, bias = False),
        nn.GELU(),
        LayerNorm(inner_dim),   # properly credit assign normformer
        nn.Conv3d(inner_dim, dim, 1, bias = False)
    ))

# 注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm = LayerNorm(dim)

        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(inner_dim, dim, 1, bias = False)

    def forward(self, x):
        f, h, w = x.shape[-3:]

        residual = x.clone()

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) ... -> b h (...) c', h = self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h (f x y) d -> b (h d) f x y', f = f, x = h, y = w)
        return self.to_out(out) + residual

# Transformer 块
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        **kwargs
    ):
        super().__init__()
        self.attn = Attention(dim, **kwargs)
        self.ff = FeedForward(dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x

# 特征图整合器
class FeatureMapConsolidator(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_ins = tuple(),
        dim_outs = tuple(),
        resize_fmap_before = True,
        conv_block_fn = None
    ):
        super().__init__()
        assert len(dim_ins) == len(dim_outs)
        self.needs_consolidating = len(dim_ins) > 0

        block_fn = default(conv_block_fn, Block)

        # 特征图卷积层列表
        self.fmap_convs = nn.ModuleList([block_fn(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.resize_fmap_before = resize_fmap_before

        self.final_dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    # 调整特征图大小
    def resize_fmaps(self, fmaps, height, width):
        return [F.interpolate(fmap, (fmap.shape[-3], height, width)) for fmap in fmaps]
    # 定义一个前向传播函数，接受输入 x 和特征图 fmaps，默认为 None
    def forward(self, x, fmaps = None):
        # 获取输入 x 的高度和宽度
        target_height, target_width = x.shape[-2:]

        # 如果未提供特征图 fmaps，则设置为空元组
        fmaps = default(fmaps, tuple())

        # 如果不需要合并特征图，则直接返回输入 x
        if not self.needs_consolidating:
            return x

        # 如果需要在卷积之前调整特征图大小
        if self.resize_fmap_before:
            # 调整特征图大小
            fmaps = self.resize_fmaps(fmaps, target_height, target_width)

        # 初始化一个空列表用于存储输出
        outs = []
        # 遍历特征图和卷积层，将卷积后的结果添加到输出列表中
        for fmap, conv in zip(fmaps, self.fmap_convs):
            outs.append(conv(fmap))

        # 如果需要在卷积之前调整特征图大小
        if self.resize_fmap_before:
            # 调整输出列表中的特征图大小
            outs = self.resize_fmaps(outs, target_height, target_width)

        # 将输入 x 和所有输出特征图连接在一起，沿着通道维度
        return torch.cat((x, *outs), dim = 1)
# 定义一个函数，返回一个类型为 type 或者包含 type 的元组
def MaybeTuple(type):
    return Union[type, Tuple[type, ...]]

# 根据卷积核大小计算 padding 大小
def kernel_and_same_pad(*kernel_size):
    paddings = tuple(map(lambda k: k // 2, kernel_size))
    return dict(kernel_size = kernel_size, padding = paddings)

# 定义 XUnet 类
class XUnet(nn.Module):

    # 初始化函数
    @beartype
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        frame_kernel_size = 1,
        dim_mults: MaybeTuple(int) = (1, 2, 4, 8),
        num_blocks_per_stage: MaybeTuple(int) = (2, 2, 2, 2),
        num_self_attn_per_stage: MaybeTuple(int) = (0, 0, 0, 1),
        nested_unet_depths: MaybeTuple(int) = (0, 0, 0, 0),
        nested_unet_dim = 32,
        channels = 3,
        use_convnext = False,
        resnet_groups = 8,
        consolidate_upsample_fmaps = True,
        skip_scale = 2 ** -0.5,
        weight_standardize = False,
        attn_heads: MaybeTuple(int) = 8,
        attn_dim_head: MaybeTuple(int) = 32
    def forward(self, x):
        is_image = x.ndim == 4

        # 验证

        assert not (is_image and not self.train_as_images), 'you specified a frame kernel size for the convolutions in this unet, but you are passing in images'
        assert not (not is_image and self.train_as_images), 'you specified no frame kernel size dimension, yet you are passing in a video. fold the frame dimension into the batch'

        # 将图像转换为帧数为 1 的视频

        if is_image:
            x = rearrange(x, 'b c h w -> b c 1 h w')

        # 初始卷积

        x = self.init_conv(x)

        # 残差

        r = x.clone()

        # 下采样和上采样

        down_hiddens = []
        up_hiddens = []

        for init_block, blocks, attn_blocks, downsample in self.downs:
            x = init_block(x)

            for block in blocks:
                x = block(x)

            for attn_block in attn_blocks:
                x = attn_block(x)

            down_hiddens.append(x)
            x = downsample(x)

        x = self.mid(x)
        x = self.mid_attn(x) + x
        x = self.mid_after(x)

        up_hiddens.append(x)
        x = self.mid_upsample(x)


        for init_block, blocks, attn_blocks, upsample in self.ups:
            x = torch.cat((x, down_hiddens.pop() * self.skip_scale), dim=1)

            x = init_block(x)

            for block in blocks:
                x = block(x)

            for attn_block in attn_blocks:
                x = attn_block(x)

            up_hiddens.insert(0, x)
            x = upsample(x)

        # 合并特征图

        x = self.consolidator(x, up_hiddens)

        # 最终残差

        x = torch.cat((x, r), dim = 1)

        # 最终卷积

        out = self.final_conv(x)

        if is_image:
            out = rearrange(out, 'b c 1 h w -> b c h w')

        return out

# 定义 PixelShuffleUpsample 类
class PixelShuffleUpsample(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        scale_factor = 2
    ):
        super().__init__()
        self.scale_squared = scale_factor ** 2
        dim_out = default(dim_out, dim)
        conv = nn.Conv3d(dim, dim_out * self.scale_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c r s) f h w -> b c f (h r) (w s)', r = scale_factor, s = scale_factor)
        )

        self.init_conv_(conv)

    # 初始化卷积层
    def init_conv_(self, conv):
        o, i, *rest_dims = conv.weight.shape
        conv_weight = torch.empty(o // self.scale_squared, i, *rest_dims)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.scale_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        x = self.net(x)
        return x

# 定义 NestedResidualUnet 类
class NestedResidualUnet(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        dim,
        *,
        depth,
        M = 32,
        frame_kernel_size = 1,
        add_residual = False,
        groups = 4,
        skip_scale = 2 ** -0.5,
        weight_standardize = False
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 设置模型深度和下采样、上采样模块
        self.depth = depth
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # 根据是否需要标准化权重选择卷积层类型
        conv = WeightStandardizedConv3d if weight_standardize else nn.Conv3d

        # 循环构建下采样模块
        for ind in range(depth):
            is_first = ind == 0
            dim_in = dim if is_first else M

            down = nn.Sequential(
                conv(dim_in, M, (1, 4, 4), stride = (1, 2, 2), padding = (0, 1, 1)),
                nn.GroupNorm(groups, M),
                nn.SiLU()
            )

            # 添加到下采样模块列表
            self.downs.append(down)

            # 构建上采样模块
            up = nn.Sequential(
                PixelShuffleUpsample(2 * M, dim_in),
                nn.GroupNorm(groups, dim_in),
                nn.SiLU()
            )

            # 添加到上采样模块列表
            self.ups.append(up)

        # 中间层模块
        self.mid = nn.Sequential(
            conv(M, M, **kernel_and_same_pad(frame_kernel_size, 3, 3)),
            nn.GroupNorm(groups, M),
            nn.SiLU()
        )

        # 设置跳跃连接的缩放因子和是否添加残差连接
        self.skip_scale = skip_scale
        self.add_residual = add_residual

    # 前向传播函数
    def forward(self, x, residual = None):
        # 判断输入是否为视频
        is_video = x.ndim == 5

        # 如果需要添加残差连接，则复制输入作为残差
        if self.add_residual:
            residual = default(residual, x.clone())

        # 获取输入张量的高度和宽度
        *_, h, w = x.shape

        # 计算模型层数
        layers = len(self.ups)

        # 检查输入张量的高度和宽度是否符合要求
        for dim_name, size in (('height', h), ('width', w)):
            assert divisible_by(size, 2 ** layers), f'{dim_name} dimension {size} must be divisible by {2 ** layers} ({layers} layers in nested unet)'
            assert (size % (2 ** self.depth)) == 0, f'the unet has too much depth for the image {dim_name} ({size}) being passed in'

        # hiddens

        # 存储中间特征
        hiddens = []

        # unet

        # 下采样过程
        for down in self.downs:
            x = down(x)
            hiddens.append(x.clone().contiguous())

        # 中间层处理
        x = self.mid(x)

        # 上采样过程
        for up in reversed(self.ups):
            x = torch.cat((x, hiddens.pop() * self.skip_scale), dim = 1)
            x = up(x)

        # 添加残差连接
        if self.add_residual:
            x = x + residual
            x = F.silu(x)

        # 返回处理后的张量
        return x
```