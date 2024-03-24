# `.\lucidrains\lumiere-pytorch\lumiere_pytorch\lumiere.py`

```py
"""
einstein notation
b - batch
t - time
c - channels
h - height
w - width
"""

from copy import deepcopy
from functools import wraps

import torch
from torch import nn, einsum, Tensor, is_tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype import beartype
from beartype.typing import List, Tuple, Optional, Type

from einops import rearrange, pack, unpack, repeat

from optree import tree_flatten, tree_unflatten

from x_transformers.x_transformers import (
    Attention,
    RMSNorm
)

# helpers

# 检查变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 将单个张量按照指定模式打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 将单个张量按照指定模式解包
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 判断一个数是否可以被另一个数整除
def divisible_by(num, den):
    return (num % den) == 0

# 判断一个数是否为奇数
def is_odd(n):
    return not divisible_by(n, 2)

# 压缩字典中存在值的键值对
def compact_values(d: dict):
    return {k: v for k, v in d.items() if exists(v)}

# extract dimensions using hooks

# 使用钩子函数提取模块的输出形状
@beartype
def extract_output_shapes(
    modules: List[Module],
    model: Module,
    model_input,
    model_kwargs: dict = dict()
):
    shapes = []
    hooks = []

    def hook_fn(_, input, output):
        return shapes.append(output.shape)

    for module in modules:
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)

    with torch.no_grad():
        model(model_input, **model_kwargs)

    for hook in hooks:
        hook.remove()

    return shapes

# freezing text-to-image, and only learning temporal parameters

# 冻结所有层，只学习时间参数
@beartype
def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

# function that takes in the entire text-to-video network, and sets the time dimension

# 设置时间维度
def set_time_dim_(
    klasses: Tuple[Type[Module]],
    model: Module,
    time_dim: int
):
    for model in model.modules():
        if isinstance(model, klasses):
            model.time_dim = time_dim

# decorator for residual

# 用于添加残差的装饰器
def residualize(fn):
    @wraps(fn)
    def inner(
        self,
        x,
        *args,
        **kwargs
    ):
        residual = x
        out = fn(self, x, *args, **kwargs)
        return out + residual

    return inner

# decorator for converting an input tensor from either image or video format to 1d time

# 将输入张量从图像或视频格式转换为1维时间的装饰器
def image_or_video_to_time(fn):

    @wraps(fn)
    def inner(
        self,
        x,
        batch_size = None,
        **kwargs
    ):

        is_video = x.ndim == 5

        if is_video:
            batch_size = x.shape[0]
            x = rearrange(x, 'b c t h w -> b h w c t')
        else:
            assert exists(batch_size) or exists(self.time_dim)
            rearrange_kwargs = dict(b = batch_size, t = self.time_dim)
            x = rearrange(x, '(b t) c h w -> b h w c t', **compact_values(rearrange_kwargs))

        x, ps = pack_one(x, '* c t')

        x = fn(self, x, **kwargs)

        x = unpack_one(x, ps, '* c t')

        if is_video:
            x = rearrange(x, 'b h w c t -> b c t h w')
        else:
            x = rearrange(x, 'b h w c t -> (b t) c h w')

        return x

    return inner

# handle channel last

# 处理通道在最后的情况
def handle_maybe_channel_last(fn):

    @wraps(fn)
    def inner(
        self,
        x,
        *args,
        **kwargs
    ):

        if self.channel_last:
            x = rearrange(x, 'b c ... -> b ... c')

        out = fn(self, x, *args, **kwargs)

        if self.channel_last:
            out = rearrange(out, 'b c ... -> b ... c')

        return out

    return inner

# helpers

# 创建一个序列模块，过滤掉不存在的模块
def Sequential(*modules):
    modules = list(filter(exists, modules))
    return nn.Sequential(*modules)

# 定义一个带有残差连接的模块
class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, t, *args, **kwargs):
        return self.fn(t, *args, **kwargs) + t

# temporal down and upsample
# 初始化一维双线性插值卷积核
def init_bilinear_kernel_1d_(conv: Module):
    # 初始化卷积核权重为零
    nn.init.zeros_(conv.weight)
    # 如果存在偏置项，初始化为零
    if exists(conv.bias):
        nn.init.zeros_(conv.bias)

    # 获取卷积核的通道数
    channels = conv.weight.shape[0]
    # 创建双线性插值核
    bilinear_kernel = Tensor([0.5, 1., 0.5])
    # 创建对角线掩码
    diag_mask = torch.eye(channels).bool()
    # 将双线性插值核应用到卷积核的对角线位置
    conv.weight.data[diag_mask] = bilinear_kernel

# 时间下采样模块
class TemporalDownsample(Module):
    def __init__(
        self,
        dim,
        channel_last = False,
        time_dim = None
    ):
        super().__init__()
        self.time_dim = time_dim
        self.channel_last = channel_last

        # 创建一维卷积层，用于时间下采样
        self.conv = nn.Conv1d(dim, dim, kernel_size = 3, stride = 2, padding = 1)
        # 初始化卷积核为双线性插值核
        init_bilinear_kernel_1d_(self.conv)

    # 前向传播函数
    @handle_maybe_channel_last
    @image_or_video_to_time
    def forward(
        self,
        x
    ):
        # 断言时间维度大于1，以便进行压缩
        assert x.shape[-1] > 1, 'time dimension must be greater than 1 to be compressed'

        return self.conv(x)

# 时间上采样模块
class TemporalUpsample(Module):
    def __init__(
        self,
        dim,
        channel_last = False,
        time_dim = None
    ):
        super().__init__()
        self.time_dim = time_dim
        self.channel_last = channel_last

        # 创建一维转置卷积层，用于时间上采样
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        # 初始化卷积核为双线性插值核
        init_bilinear_kernel_1d_(self.conv)

    # 前向传播函数
    @handle_maybe_channel_last
    @image_or_video_to_time
    def forward(
        self,
        x
    ):
        return self.conv(x)

# 卷积膨胀块
class ConvolutionInflationBlock(Module):
    def __init__(
        self,
        *,
        dim,
        conv2d_kernel_size = 3,
        conv1d_kernel_size = 3,
        groups = 8,
        channel_last = False,
        time_dim = None
    ):
        super().__init__()
        assert is_odd(conv2d_kernel_size)
        assert is_odd(conv1d_kernel_size)

        self.time_dim = time_dim
        self.channel_last = channel_last

        # 空间卷积层
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, conv2d_kernel_size, padding = conv2d_kernel_size // 2),
            nn.GroupNorm(groups, num_channels = dim),
            nn.SiLU()
        )

        # 时间卷积层
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(dim, dim, conv1d_kernel_size, padding = conv1d_kernel_size // 2),
            nn.GroupNorm(groups, num_channels = dim),
            nn.SiLU()
        )

        # 投影输出层
        self.proj_out = nn.Conv1d(dim, dim, 1)

        # 初始化投影输出层的权重和偏置为零
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    # 前向传播函数
    @residualize
    @handle_maybe_channel_last
    def forward(
        self,
        x,
        batch_size = None
    ):
        is_video = x.ndim == 5

        if is_video:
            batch_size = x.shape[0]
            x = rearrange(x, 'b c t h w -> (b t) c h w')

        x = self.spatial_conv(x)

        rearrange_kwargs = compact_values(dict(b = batch_size, t = self.time_dim))

        assert len(rearrange_kwargs) > 0, 'either batch_size is passed in on forward, or time_dim is set on init'
        x = rearrange(x, '(b t) c h w -> b h w c t', **rearrange_kwargs)

        x, ps = pack_one(x, '* c t')

        x = self.temporal_conv(x)
        x = self.proj_out(x)

        x = unpack_one(x, ps, '* c t')

        if is_video:
            x = rearrange(x, 'b h w c t -> b c t h w')
        else:
            x = rearrange(x, 'b h w c t -> (b t) c h w')

        return x

# 注意力膨胀块
class AttentionInflationBlock(Module):
    def __init__(
        self,
        *,
        dim,
        depth = 1,
        prenorm = True,
        residual_attn = True,
        time_dim = None,
        channel_last = False,
        **attn_kwargs
    # 初始化函数，继承父类的初始化方法
    def __init__(
        self,
        time_dim,
        channel_last,
        depth,
        dim,
        attn_kwargs = {},
        prenorm = False,
        residual_attn = False
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 设置时间维度和是否通道在最后的标志
        self.time_dim = time_dim
        self.channel_last = channel_last

        # 初始化时间注意力模块列表
        self.temporal_attns = ModuleList([])

        # 根据深度循环创建注意力模块
        for _ in range(depth):
            # 创建注意力模块序列
            attn = Sequential(
                RMSNorm(dim) if prenorm else None,
                Attention(
                    dim = dim,
                    **attn_kwargs
                )
            )

            # 如果开启残差连接，则将注意力模块包装成残差模块
            if residual_attn:
                attn = Residual(attn)

            # 将创建的注意力模块添加到时间注意力模块列表中
            self.temporal_attns.append(attn)

        # 创建输出投影层
        self.proj_out = nn.Linear(dim, dim)

        # 初始化输出投影层的权重和偏置为零
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    # 前向传播函数，添加了装饰器
    @residualize
    @handle_maybe_channel_last
    def forward(
        self,
        x,
        batch_size = None
    ):
        # 判断输入是否为视频数据
        is_video = x.ndim == 5
        # 断言判断输入数据维度是否符合要求
        assert is_video ^ (exists(batch_size) or exists(self.time_dim)), 'either a tensor of shape (batch, channels, time, height, width) is passed in, or (batch * time, channels, height, width) along with `batch_size`'

        # 如果通道在最后，则重新排列输入数据
        if self.channel_last:
            x = rearrange(x, 'b ... c -> b c ...')

        # 如果是视频数据，则重新排列输入数据
        if is_video:
            batch_size = x.shape[0]
            x = rearrange(x, 'b c t h w -> b h w t c')
        else:
            assert exists(batch_size) or exists(self.time_dim)

            rearrange_kwargs = dict(b = batch_size, t = self.time_dim)
            x = rearrange(x, '(b t) c h w -> b h w t c', **compact_values(rearrange_kwargs))

        # 打包输入数据
        x, ps = pack_one(x, '* t c')

        # 遍历时间注意力模块列表，对输入数据进行注意力���作
        for attn in self.temporal_attns:
            x = attn(x)

        # 输出投影层处理数据
        x = self.proj_out(x)

        # 解包数据
        x = unpack_one(x, ps, '* t c')

        # 根据是否为视频数据重新排列输出数据
        if is_video:
            x = rearrange(x, 'b h w t c -> b c t h w')
        else:
            x = rearrange(x, 'b h w t c -> (b t) c h w')

        # 如果通道在最后，则重新排列输出数据
        if self.channel_last:
            x = rearrange(x, 'b c ... -> b ... c')

        # 返回处理后的输出数据
        return x
# 定义一个包装器类，用于在模块后添加钩子
class PostModuleHookWrapper(Module):
    def __init__(self, temporal_module: Module):
        super().__init__()
        self.temporal_module = temporal_module

    # 在前向传播过程中，对输出进行处理并返回
    def forward(self, _, input, output):
        output = self.temporal_module(output)
        return output

# 将临时模块插入到模块列表中
def insert_temporal_modules_(modules: List[Module], temporal_modules: ModuleList):
    assert len(modules) == len(temporal_modules)

    # 遍历模块列表和临时模块列表，为每个模块注册一个后向钩子
    for module, temporal_module in zip(modules, temporal_modules):
        module.register_forward_hook(PostModuleHookWrapper(temporal_module))

# 主要的文本到图像模型包装器
class Lumiere(Module):

    # 初始化函数
    @beartype
    def __init__(
        self,
        model: Module,
        *,
        image_size: int,
        unet_time_kwarg: str,
        conv_module_names: List[str],
        attn_module_names: List[str] = [],
        downsample_module_names: List[str] = [],
        upsample_module_names: List[str] = [],
        channels: int = 3,
        conv_inflation_kwargs: dict = dict(),
        attn_inflation_kwargs: dict = dict(),
        downsample_kwargs: dict = dict(),
        upsample_kwargs: dict = dict(),
        conv_klass = ConvolutionInflationBlock,
        attn_klass = AttentionInflationBlock,
        downsample_klass = TemporalDownsample,
        upsample_klass = TemporalUpsample
    @property
    def downsample_factor(self):
        return 2 ** len(self.downsamples)

    # 返回模型的参数
    def parameters(self):
        return [
            *self.convs.parameters(),
            *self.attns.parameters(),
            *self.downsamples.parameters(),
            *self.upsamples.parameters(),
        ]

    # 前向传播函数
    @beartype
    def forward(
        self,
        video: Tensor,
        *args,
        **kwargs
    ) -> Tensor:

        assert video.ndim == 5
        batch, channels, time, height, width = video.shape

        assert channels == self.channels
        assert (height, width) == (self.image_size, self.image_size)

        assert divisible_by(time, self.downsample_factor)

        # 将视频转换为一堆图像
        images = rearrange(video, 'b c t h w -> (b t) c h w')

        # 为所有时间层设置正确的时间维度
        set_time_dim_(self.temporal_klasses, self, time)

        # 将所有图像传入文本到图像模型
        images = self.model(images, *args, **kwargs)

        # 将结果重塑回去成去噪视频
        return rearrange(images, '(b t) c h w -> b c t h w', b = batch)
```