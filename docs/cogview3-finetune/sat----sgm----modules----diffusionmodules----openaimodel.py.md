# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\openaimodel.py`

```
# 导入操作系统模块，用于处理文件和目录操作
import os
# 导入数学模块，提供数学函数和常量
import math
# 从 abc 模块导入抽象方法装饰器，用于定义抽象基类
from abc import abstractmethod
# 从 functools 模块导入 partial 函数，用于偏函数应用
from functools import partial
# 从 typing 模块导入类型注解，用于类型提示
from typing import Iterable, List, Optional, Tuple, Union

# 导入 numpy 库，通常用于数值计算
import numpy as np
# 导入 torch 库，通常用于深度学习
import torch as th
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的功能模块，提供激活函数等
import torch.nn.functional as F
# 从 einops 导入 rearrange 函数，用于重排张量
from einops import rearrange

# 导入自定义模块中的 SpatialTransformer 类
from ...modules.attention import SpatialTransformer
# 导入自定义模块中的实用函数
from ...modules.diffusionmodules.util import (
    avg_pool_nd,  # 平均池化函数
    checkpoint,   # 检查点函数
    conv_nd,      # 卷积函数
    linear,       # 线性变换函数
    normalization, # 归一化函数
    timestep_embedding, # 时间步嵌入函数
    zero_module,  # 零模块函数
)

# 导入自定义模块中的实用函数
from ...util import default, exists

# 定义一个空的占位函数，用于将模块转换为半精度浮点数
# dummy replace
def convert_module_to_f16(x):
    pass

# 定义一个空的占位函数，用于将模块转换为单精度浮点数
def convert_module_to_f32(x):
    pass


# 定义一个用于注意力池化的类，继承自 nn.Module
## go
class AttentionPool2d(nn.Module):
    """
    从 CLIP 中改编: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    # 初始化方法，设置各类参数
    def __init__(
        self,
        spacial_dim: int,  # 空间维度
        embed_dim: int,    # 嵌入维度
        num_heads_channels: int,  # 头通道数量
        output_dim: int = None,  # 输出维度（可选）
    ):
        # 调用父类初始化方法
        super().__init__()
        # 定义位置嵌入参数，初始化为正态分布
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        # 定义查询、键、值的卷积投影
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        # 定义输出的卷积投影
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        # 计算头的数量
        self.num_heads = embed_dim // num_heads_channels
        # 初始化注意力机制
        self.attention = QKVAttention(self.num_heads)

    # 前向传播方法
    def forward(self, x):
        # 获取输入的批次大小和通道数
        b, c, *_spatial = x.shape
        # 将输入重塑为 (批次, 通道, 高*宽) 的形状
        x = x.reshape(b, c, -1)  # NC(HW)
        # 在最后一维上添加均值作为额外的特征
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        # 将位置嵌入加到输入上
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        # 对输入进行查询、键、值投影
        x = self.qkv_proj(x)
        # 应用注意力机制
        x = self.attention(x)
        # 对结果进行输出投影
        x = self.c_proj(x)
        # 返回第一个通道的结果
        return x[:, :, 0]


# 定义一个时间步模块的基类，继承自 nn.Module
class TimestepBlock(nn.Module):
    """
    任何模块的 forward() 方法接受时间步嵌入作为第二个参数。
    """

    # 定义抽象的前向传播方法
    @abstractmethod
    def forward(self, x, emb):
        """
        将模块应用于 `x`，并给定 `emb` 时间步嵌入。
        """


# 定义一个时间步嵌入的顺序模块，继承自 nn.Sequential 和 TimestepBlock
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    一个顺序模块，将时间步嵌入作为额外输入传递给支持的子模块。
    """

    # 重写前向传播方法
    def forward(
        self,
        x: th.Tensor,  # 输入张量
        emb: th.Tensor,  # 时间步嵌入张量
        context: Optional[th.Tensor] = None,  # 上下文张量（可选）
    ):
        # 遍历所有子模块
        for layer in self:
            module = layer

            # 如果子模块是 TimestepBlock，则使用时间步嵌入进行计算
            if isinstance(module, TimestepBlock):
                x = layer(x, emb)
            # 如果子模块是 SpatialTransformer，则使用上下文进行计算
            elif isinstance(module, SpatialTransformer):
                x = layer(x, context)
            # 否则，仅使用输入进行计算
            else:
                x = layer(x)
        # 返回最终的输出
        return x


# 定义一个上采样模块，继承自 nn.Module
class Upsample(nn.Module):
    """
    一个可选卷积的上采样层。
    :param channels: 输入和输出的通道数。
    :param use_conv: 布尔值，确定是否应用卷积。
    :param dims: 确定信号是 1D、2D 还是 3D。如果是 3D，则在内两个维度上进行上采样。
    """
    # 初始化方法，设置类的基本属性
        def __init__(
            self, channels, use_conv, dims=2, out_channels=None, padding=1, third_up=False
        ):
            # 调用父类初始化方法
            super().__init__()
            # 保存输入的通道数
            self.channels = channels
            # 如果没有指定输出通道数，则默认与输入通道数相同
            self.out_channels = out_channels or channels
            # 保存是否使用卷积的标志
            self.use_conv = use_conv
            # 保存维度信息
            self.dims = dims
            # 保存是否进行第三层上采样的标志
            self.third_up = third_up
            # 如果使用卷积，初始化卷积层
            if use_conv:
                self.conv = conv_nd(
                    dims, self.channels, self.out_channels, 3, padding=padding
                )
    
    # 前向传播方法，定义输入如何通过网络进行处理
        def forward(self, x):
            # 确保输入的通道数与初始化时指定的通道数一致
            assert x.shape[1] == self.channels
            # 如果输入为三维数据
            if self.dims == 3:
                # 根据是否需要第三层上采样确定时间因子
                t_factor = 1 if not self.third_up else 2
                # 对输入进行上采样
                x = F.interpolate(
                    x,
                    (t_factor * x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                    mode="nearest",
                )
            else:
                # 对输入进行上采样，比例因子为2
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            # 如果使用卷积，则将输入通过卷积层处理
            if self.use_conv:
                x = self.conv(x)
            # 返回处理后的输出
            return x
# 定义一个转置上采样的类，继承自 nn.Module
class TransposedUpsample(nn.Module):
    "Learned 2x upsampling without padding"  # 文档字符串，描述该类的功能

    # 初始化方法，设置输入通道、输出通道和卷积核大小
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()  # 调用父类的初始化方法
        self.channels = channels  # 保存输入通道数量
        self.out_channels = out_channels or channels  # 如果没有指定输出通道，则与输入通道相同

        # 定义一个转置卷积层，用于上采样
        self.up = nn.ConvTranspose2d(
            self.channels, self.out_channels, kernel_size=ks, stride=2
        )

    # 前向传播方法，执行上采样操作
    def forward(self, x):
        return self.up(x)  # 返回上采样后的结果


# 定义一个下采样层的类，继承自 nn.Module
class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    # 初始化方法，设置输入通道、是否使用卷积、维度等参数
    def __init__(
        self, channels, use_conv, dims=2, out_channels=None, padding=1, third_down=False
    ):
        super().__init__()  # 调用父类的初始化方法
        self.channels = channels  # 保存输入通道数量
        self.out_channels = out_channels or channels  # 如果没有指定输出通道，则与输入通道相同
        self.use_conv = use_conv  # 保存是否使用卷积的标志
        self.dims = dims  # 保存信号的维度
        stride = 2 if dims != 3 else ((1, 2, 2) if not third_down else (2, 2, 2))  # 确定步幅
        if use_conv:  # 如果使用卷积
            # print(f"Building a Downsample layer with {dims} dims.")  # 打印信息，表示正在构建下采样层
            # print(
            #     f"  --> settings are: \n in-chn: {self.channels}, out-chn: {self.out_channels}, "
            #     f"kernel-size: 3, stride: {stride}, padding: {padding}"
            # )  # 打印卷积层的设置参数
            # if dims == 3:
            #     print(f"  --> Downsampling third axis (time): {third_down}")  # 打印是否在第三维进行下采样
            # 定义卷积操作
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:  # 如果不使用卷积
            assert self.channels == self.out_channels  # 确保输入通道与输出通道相同
            # 定义平均池化操作
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    # 前向传播方法，执行下采样操作
    def forward(self, x):
        assert x.shape[1] == self.channels  # 确保输入的通道数匹配
        return self.op(x)  # 返回下采样后的结果


# 定义一个残差块的类，继承自 TimestepBlock
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    # 初始化方法，用于创建类的实例
    def __init__(
        self,
        channels,  # 输入通道数
        emb_channels,  # 嵌入通道数
        dropout,  # 丢弃率
        out_channels=None,  # 输出通道数，默认为 None
        use_conv=False,  # 是否使用卷积
        use_scale_shift_norm=False,  # 是否使用缩放位移归一化
        dims=2,  # 数据维度，默认为 2
        use_checkpoint=False,  # 是否使用检查点
        up=False,  # 是否进行上采样
        down=False,  # 是否进行下采样
        kernel_size=3,  # 卷积核大小，默认为 3
        exchange_temb_dims=False,  # 是否交换时间嵌入维度
        skip_t_emb=False,  # 是否跳过时间嵌入
    ):
        # 调用父类初始化方法
        super().__init__()
        # 设置输入通道数
        self.channels = channels
        # 设置嵌入通道数
        self.emb_channels = emb_channels
        # 设置丢弃率
        self.dropout = dropout
        # 设置输出通道数，如果未提供则默认与输入通道数相同
        self.out_channels = out_channels or channels
        # 设置是否使用卷积
        self.use_conv = use_conv
        # 设置是否使用检查点
        self.use_checkpoint = use_checkpoint
        # 设置是否使用缩放位移归一化
        self.use_scale_shift_norm = use_scale_shift_norm
        # 设置是否交换时间嵌入维度
        self.exchange_temb_dims = exchange_temb_dims

        # 如果卷积核大小是可迭代的，计算每个维度的填充大小
        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            # 否则直接计算单个卷积核的填充大小
            padding = kernel_size // 2

        # 创建输入层的序列，包括归一化、激活函数和卷积操作
        self.in_layers = nn.Sequential(
            normalization(channels),  # 归一化
            nn.SiLU(),  # SiLU 激活函数
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),  # 卷积层
        )

        # 判断是否进行上采样或下采样
        self.updown = up or down

        # 如果进行上采样，初始化上采样层
        if up:
            self.h_upd = Upsample(channels, False, dims)  # 上采样层
            self.x_upd = Upsample(channels, False, dims)  # 上采样层
        # 如果进行下采样，初始化下采样层
        elif down:
            self.h_upd = Downsample(channels, False, dims)  # 下采样层
            self.x_upd = Downsample(channels, False, dims)  # 下采样层
        # 否则使用身份映射
        else:
            self.h_upd = self.x_upd = nn.Identity()  # 身份映射层

        # 设置是否跳过时间嵌入
        self.skip_t_emb = skip_t_emb
        # 根据是否使用缩放位移归一化计算嵌入输出通道数
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        # 如果跳过时间嵌入，输出警告并设置嵌入层为 None
        if self.skip_t_emb:
            print(f"Skipping timestep embedding in {self.__class__.__name__}")  # 警告信息
            assert not self.use_scale_shift_norm  # 确保不使用缩放位移归一化
            self.emb_layers = None  # 嵌入层设置为 None
            self.exchange_temb_dims = False  # 不交换时间嵌入维度
        # 否则创建嵌入层的序列
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),  # SiLU 激活函数
                linear(
                    emb_channels,  # 嵌入通道数
                    self.emb_out_channels,  # 嵌入输出通道数
                ),
            )

        # 创建输出层的序列，包括归一化、激活函数、丢弃层和卷积层
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),  # 归一化
            nn.SiLU(),  # SiLU 激活函数
            nn.Dropout(p=dropout),  # 丢弃层
            zero_module(
                conv_nd(
                    dims,  # 数据维度
                    self.out_channels,  # 输出通道数
                    self.out_channels,  # 输出通道数
                    kernel_size,  # 卷积核大小
                    padding=padding,  # 填充
                )
            ),  # 卷积层
        )

        # 根据输入和输出通道数设置跳过连接
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()  # 身份映射层
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding  # 卷积层
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)  # 卷积层，卷积核大小为 1
    # 定义前向传播函数，接受输入张量和时间步嵌入
    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # 调用检查点函数以保存中间计算结果，减少内存使用
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    # 定义实际的前向传播逻辑
    def _forward(self, x, emb):
        # 如果设置了 updown，则进行上采样和下采样
        if self.updown:
            # 分离输入层的最后一层和其他层
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            # 通过其他输入层处理输入 x
            h = in_rest(x)
            # 更新隐藏状态
            h = self.h_upd(h)
            # 更新输入 x
            x = self.x_upd(x)
            # 通过卷积层处理隐藏状态
            h = in_conv(h)
        else:
            # 直接通过输入层处理输入 x
            h = self.in_layers(x)

        # 如果跳过时间嵌入，则初始化嵌入输出为零张量
        if self.skip_t_emb:
            emb_out = th.zeros_like(h)
        else:
            # 通过嵌入层处理时间嵌入，确保数据类型与 h 一致
            emb_out = self.emb_layers(emb).type(h.dtype)
        # 扩展 emb_out 的形状以匹配 h 的形状
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        # 如果使用缩放和偏移规范化
        if self.use_scale_shift_norm:
            # 分离输出层中的规范化层和其他层
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            # 将嵌入输出分割为缩放和偏移
            scale, shift = th.chunk(emb_out, 2, dim=1)
            # 对隐藏状态进行规范化并应用缩放和偏移
            h = out_norm(h) * (1 + scale) + shift
            # 通过剩余的输出层处理隐藏状态
            h = out_rest(h)
        else:
            # 如果交换时间嵌入的维度
            if self.exchange_temb_dims:
                # 重新排列嵌入输出的维度
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            # 将嵌入输出与隐藏状态相加
            h = h + emb_out
            # 通过输出层处理隐藏状态
            h = self.out_layers(h)
        # 返回输入 x 与处理后的隐藏状态的跳跃连接
        return self.skip_connection(x) + h
# 定义一个注意力模块，允许空间位置相互关注
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    # 初始化方法，定义模块的基本参数
    def __init__(
        self,
        channels,  # 输入通道数
        num_heads=1,  # 注意力头的数量，默认为1
        num_head_channels=-1,  # 每个头的通道数，默认为-1
        use_checkpoint=False,  # 是否使用检查点
        use_new_attention_order=False,  # 是否使用新的注意力顺序
    ):
        # 调用父类初始化方法
        super().__init__()
        self.channels = channels  # 保存输入通道数
        # 判断 num_head_channels 是否为 -1
        if num_head_channels == -1:
            self.num_heads = num_heads  # 如果为 -1，直接使用 num_heads
        else:
            # 断言通道数可以被 num_head_channels 整除
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels  # 计算头的数量
        self.use_checkpoint = use_checkpoint  # 保存检查点标志
        self.norm = normalization(channels)  # 初始化归一化层
        self.qkv = conv_nd(1, channels, channels * 3, 1)  # 创建卷积层用于计算 q, k, v
        # 根据是否使用新注意力顺序选择相应的注意力类
        if use_new_attention_order:
            # 在分割头之前分割 qkv
            self.attention = QKVAttention(self.num_heads)
        else:
            # 在分割 qkv 之前分割头
            self.attention = QKVAttentionLegacy(self.num_heads)

        # 初始化输出投影层
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    # 前向传播方法
    def forward(self, x, **kwargs):
        # TODO 添加跨帧注意力并使用混合检查点
        # 使用检查点机制来调用内部前向传播函数
        return checkpoint(
            self._forward, (x,), self.parameters(), True
        )  # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # pytorch

    # 内部前向传播方法
    def _forward(self, x):
        b, c, *spatial = x.shape  # 解包输入张量的形状
        x = x.reshape(b, c, -1)  # 将输入张量重塑为 (batch_size, channels, spatial_dim)
        qkv = self.qkv(self.norm(x))  # 计算 q, k, v
        h = self.attention(qkv)  # 应用注意力机制
        h = self.proj_out(h)  # 对注意力结果进行投影
        return (x + h).reshape(b, c, *spatial)  # 返回重塑后的结果

# 计算注意力操作的 FLOPS
def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape  # 解包输入张量的形状
    num_spatial = int(np.prod(spatial))  # 计算空间维度的总数
    # 进行两个矩阵乘法，具有相同数量的操作。
    # 第一个计算权重矩阵，第二个计算值向量的组合。
    matmul_ops = 2 * b * (num_spatial**2) * c  # 计算矩阵乘法的操作数
    model.total_ops += th.DoubleTensor([matmul_ops])  # 将操作数累加到模型的总操作数中

# 旧版 QKV 注意力模块
class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    # 初始化方法，设置注意力头的数量
    def __init__(self, n_heads):
        super().__init__()  # 调用父类初始化方法
        self.n_heads = n_heads  # 保存注意力头的数量
    # 定义前向传播方法，接收 QKV 张量
    def forward(self, qkv):
        """
        应用 QKV 注意力机制。
        :param qkv: 一个形状为 [N x (H * 3 * C) x T] 的张量，包含 Q、K 和 V。
        :return: 一个形状为 [N x (H * C) x T] 的张量，经过注意力处理后输出。
        """
        # 获取输入张量的批量大小、宽度和长度
        bs, width, length = qkv.shape
        # 确保宽度可以被 (3 * n_heads) 整除，以分割 Q、K 和 V
        assert width % (3 * self.n_heads) == 0
        # 计算每个头的通道数
        ch = width // (3 * self.n_heads)
        # 将 qkv 张量重塑并分割成 Q、K 和 V 三个部分
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        # 计算缩放因子，用于稳定性
        scale = 1 / math.sqrt(math.sqrt(ch))
        # 使用爱因斯坦求和约定计算注意力权重，乘以缩放因子
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # 使用 f16 比后续除法更稳定
        # 对权重进行 softmax 归一化，并保持原始数据类型
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        # 根据权重和 V 计算输出张量
        a = th.einsum("bts,bcs->bct", weight, v)
        # 将输出张量重塑为原始批量大小和通道数
        return a.reshape(bs, -1, length)

    # 定义静态方法以计算模型的浮点运算数
    @staticmethod
    def count_flops(model, _x, y):
        # 调用辅助函数计算注意力层的浮点运算数
        return count_flops_attn(model, _x, y)
# 定义一个名为 QKVAttention 的类，继承自 nn.Module
class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    # 初始化方法，接收注意力头的数量
    def __init__(self, n_heads):
        super().__init__()  # 调用父类的初始化方法
        self.n_heads = n_heads  # 保存注意力头的数量

    # 前向传播方法，接收 qkv 张量并执行注意力计算
    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape  # 解包 qkv 张量的维度
        assert width % (3 * self.n_heads) == 0  # 确保宽度能够被注意力头数量整除
        ch = width // (3 * self.n_heads)  # 计算每个头的通道数
        q, k, v = qkv.chunk(3, dim=1)  # 将 qkv 张量分成 Q, K, V 三部分
        scale = 1 / math.sqrt(math.sqrt(ch))  # 计算缩放因子
        weight = th.einsum(
            "bct,bcs->bts",  # 定义爱因斯坦求和约定，计算权重
            (q * scale).view(bs * self.n_heads, ch, length),  # 缩放后的 Q 重塑形状
            (k * scale).view(bs * self.n_heads, ch, length),  # 缩放后的 K 重塑形状
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)  # 计算权重的 softmax，确保其和为 1
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))  # 计算最终的注意力输出
        return a.reshape(bs, -1, length)  # 将输出重塑回原始批量形状

    @staticmethod
    # 计算 FLOPs 的静态方法
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)  # 调用函数计算注意力层的 FLOPs


# 定义一个名为 Timestep 的类，继承自 nn.Module
class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()  # 调用父类的初始化方法
        self.dim = dim  # 保存时间步的维度

    # 前向传播方法，接收时间步张量
    def forward(self, t):
        return timestep_embedding(t, self.dim)  # 调用时间步嵌入函数


# 定义一个字典，将字符串类型映射到对应的 PyTorch 数据类型
str_to_dtype = {
    "fp32": th.float32,  # fp32 对应 float32
    "fp16": th.float16,  # fp16 对应 float16
    "bf16": th.bfloat16   # bf16 对应 bfloat16
}

# 定义一个名为 UNetModel 的类，继承自 nn.Module
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    """
    # 参数 resblock_updown：是否在上采样/下采样过程中使用残差块
    # 参数 use_new_attention_order：是否使用不同的注意力模式以提高效率
    """

    # 初始化方法
    def __init__(
        # 输入通道数
        self,
        in_channels,
        # 模型通道数
        model_channels,
        # 输出通道数
        out_channels,
        # 残差块的数量
        num_res_blocks,
        # 注意力分辨率
        attention_resolutions,
        # dropout 比例，默认为 0
        dropout=0,
        # 通道的倍增因子，默认值为 (1, 2, 4, 8)
        channel_mult=(1, 2, 4, 8),
        # 是否使用卷积重采样，默认为 True
        conv_resample=True,
        # 数据维度，默认为 2
        dims=2,
        # 类别数，默认为 None
        num_classes=None,
        # 是否使用检查点，默认为 False
        use_checkpoint=False,
        # 是否使用 fp16 精度，默认为 False
        use_fp16=False,
        # 注意力头数，默认为 -1
        num_heads=-1,
        # 每个头的通道数，默认为 -1
        num_head_channels=-1,
        # 上采样时的头数，默认为 -1
        num_heads_upsample=-1,
        # 是否使用尺度偏移归一化，默认为 False
        use_scale_shift_norm=False,
        # 是否使用残差块进行上采样/下采样，默认为 False
        resblock_updown=False,
        # 是否使用新的注意力顺序，默认为 False
        use_new_attention_order=False,
        # 是否使用空间变换器，支持自定义变换器
        use_spatial_transformer=False,  # custom transformer support
        # 变换器的深度，默认为 1
        transformer_depth=1,  # custom transformer support
        # 上下文维度，默认为 None
        context_dim=None,  # custom transformer support
        # 嵌入数，默认为 None
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        # 是否使用传统模式，默认为 True
        legacy=True,
        # 是否禁用自注意力，默认为 None
        disable_self_attentions=None,
        # 注意力块的数量，默认为 None
        num_attention_blocks=None,
        # 是否禁用中间自注意力，默认为 False
        disable_middle_self_attn=False,
        # 是否在变换器中使用线性输入，默认为 False
        use_linear_in_transformer=False,
        # 空间变换器的注意力类型，默认为 "softmax"
        spatial_transformer_attn_type="softmax",
        # 输入通道数，默认为 None
        adm_in_channels=None,
        # 是否使用 Fairscale 检查点，默认为 False
        use_fairscale_checkpoint=False,
        # 是否将计算卸载到 CPU，默认为 False
        offload_to_cpu=False,
        # 中间变换器的深度，默认为 None
        transformer_depth_middle=None,
        # 配置条件嵌入维度，默认为 None
        cfg_cond_embed_dim=None,
        # 数据类型，默认为 "fp32"
        dtype="fp32",
    # 将模型的主体转换为 float16
    def convert_to_fp16(self):
        """
        将模型的主体转换为 float16。
        """
        # 对输入块应用转换模块，将其转换为 float16
        self.input_blocks.apply(convert_module_to_f16)
        # 对中间块应用转换模块，将其转换为 float16
        self.middle_block.apply(convert_module_to_f16)
        # 对输出块应用转换模块，将其转换为 float16
        self.output_blocks.apply(convert_module_to_f16)

    # 将模型的主体转换为 float32
    def convert_to_fp32(self):
        """
        将模型的主体转换为 float32。
        """
        # 对输入块应用转换模块，将其转换为 float32
        self.input_blocks.apply(convert_module_to_f32)
        # 对中间块应用转换模块，将其转换为 float32
        self.middle_block.apply(convert_module_to_f32)
        # 对输出块应用转换模块，将其转换为 float32
        self.output_blocks.apply(convert_module_to_f32)
    # 定义前向传播函数，接收输入数据和其他参数
    def forward(self, x, timesteps=None, context=None, y=None, scale_emb=None, **kwargs):
        """
        应用模型于输入批次。
        :param x: 输入张量，形状为 [N x C x ...]。
        :param timesteps: 一维时间步批次。
        :param context: 通过 crossattn 插入的条件信息。
        :param y: 标签张量，形状为 [N]，如果是类条件。
        :return: 输出张量，形状为 [N x C x ...]。
        """
        # 如果输入数据类型不匹配，则转换为模型所需的数据类型
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
    
        # 确保 y 的存在性与类数设置一致
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        # 初始化存储中间结果的列表
        hs = []
    
        # 生成时间步嵌入
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        # 如果提供了缩放嵌入，则进行相应处理
        if scale_emb is not None:
            assert hasattr(self, "w_proj"), "w_proj not found in the model"
            t_emb = t_emb + self.w_proj(scale_emb.to(self.dtype))
        # 通过时间嵌入生成最终嵌入
        emb = self.time_embed(t_emb)
    
        # 如果模型是类条件，则将标签嵌入加入到最终嵌入中
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
    
        # 将输入数据赋值给 h
        # h = x.type(self.dtype)
        h = x
        # 通过输入模块处理 h，并保存中间结果
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        # 通过中间模块进一步处理 h
        h = self.middle_block(h, emb, context)
        # 通过输出模块处理 h，并逐层合并中间结果
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        # 将 h 转换回原输入数据类型
        h = h.type(x.dtype)
        # 检查是否支持预测码本 ID
        if self.predict_codebook_ids:
            assert False, "not supported anymore. what the f*** are you doing?"
        else:
            # 返回最终输出结果
            return self.out(h)
```